"""
End-to-end inference + trajectory plotting for TSformer-VO.

Works with PNG images and runs on CPU or CUDA.
Saves .npy predictions, trajectory plots, and timing.json to results/.

Usage:
    python run_inference.py
    python run_inference.py --model Model2 --sequences 04 07 10
    python run_inference.py --model Model3 --sequences 10
"""

import argparse
import glob
import json
import os
import pickle
import queue
import time
from functools import partial

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from datasets.utils import euler_to_rotation
from timesformer.models.vit import VisionTransformer


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

MEAN_ANGLES = np.array([1.7061e-5,  9.5582e-4, -5.5258e-5])
STD_ANGLES  = np.array([2.8256e-3,  1.7771e-2,  3.2326e-3])
MEAN_T      = np.array([-8.6736e-5, -1.6038e-2,  9.0033e-1])
STD_T       = np.array([ 2.5584e-2,  1.8545e-2,  3.0352e-1])


def load_model(checkpoint_path, checkpoint_name, model_params, device):
    model = VisionTransformer(
        img_size=model_params["image_size"],
        num_classes=model_params["num_classes"],
        patch_size=model_params["patch_size"],
        embed_dim=model_params["dim"],
        depth=model_params["depth"],
        num_heads=model_params["heads"],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        num_frames=model_params["num_frames"],
        attention_type=model_params["attention_type"],
    )
    ckpt = torch.load(
        os.path.join(checkpoint_path, f"{checkpoint_name}.pth"),
        map_location=device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_windows(frame_paths, window_size, overlap):
    stride = window_size - overlap
    windows = []
    i = 0
    while i + window_size <= len(frame_paths):
        windows.append(frame_paths[i: i + window_size])
        i += stride
    return windows


def preprocess_window(paths, transform):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(transform(img).unsqueeze(0))
    imgs = torch.cat(imgs, dim=0)                  # T x C x H x W
    return imgs.permute(1, 0, 2, 3).unsqueeze(0)  # 1 x C x T x H x W


def post_processing(pred_poses, window_size):
    if window_size == 2:
        return pred_poses.squeeze(1)

    n = pred_poses.shape[0]
    q = queue.Queue(window_size - 1)
    idx = 0
    poses = []

    while not q.full():
        q.put(pred_poses[idx])
        idx += 1

    while idx < n:
        if idx == window_size - 1:
            poses.append(q.queue[0][0])
            avg = (q.queue[0][1] + q.queue[1][0]) / 2
            poses.append(avg)
            if window_size == 4:
                avg = (q.queue[0][2] + q.queue[1][1] + q.queue[2][0]) / 3
                poses.append(avg)

        elif idx < n - 1:
            if window_size == 3:
                avg = (q.queue[0][1] + q.queue[1][0]) / 2
                poses.append(avg)
            elif window_size == 4:
                avg = (q.queue[0][2] + q.queue[1][1] + q.queue[2][0]) / 3
                poses.append(avg)

        else:
            if window_size == 3:
                poses.append(q.queue[1][1])
            elif window_size == 4:
                avg = (q.queue[1][2] + q.queue[2][1]) / 2
                poses.append(avg)
                poses.append(q.queue[2][2])
            idx += 1

        if idx < n - 1:
            idx += 1
            q.get()
            q.put(pred_poses[idx])

    return np.asarray(poses)


def recover_trajectory(poses):
    trajectory = []
    T = np.eye(4)
    for p in poses:
        angles = p[:3]
        t = p[3:]
        z, y, x = np.multiply(angles, STD_ANGLES) + MEAN_ANGLES
        t = np.multiply(t, STD_T) + MEAN_T
        R = np.asarray(euler_to_rotation(z, y, x, seq="zyx"))
        T_r = np.vstack([np.hstack([R, t.reshape(3, 1)]), [0, 0, 0, 1]])
        T = T @ T_r
        trajectory.append(T[:3, 3].copy())
    return np.asarray(trajectory)


def load_gt_trajectory(poses_path):
    positions = []
    with open(poses_path) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            positions.append([vals[3], vals[7], vals[11]])
    return np.asarray(positions)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(model_name, sequences, data_root, camera_id, poses_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = os.path.join("checkpoints", model_name)
    checkpoint_name = [f[:-4] for f in os.listdir(checkpoint_path) if f.endswith(".pth")][0]

    with open(os.path.join(checkpoint_path, "args.pkl"), "rb") as f:
        args = pickle.load(f)

    model_params = args["model_params"]
    window_size  = args["window_size"]
    overlap      = args["overlap"]

    print(f"Model: {model_name} | checkpoint: {checkpoint_name}")
    print(f"window_size={window_size}, overlap={overlap}, num_frames={model_params['num_frames']}")

    model = load_model(checkpoint_path, checkpoint_name, model_params, device)

    transform = transforms.Compose([
        transforms.Resize(model_params["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183],
        ),
    ])

    results_dir = os.path.join("results", model_name)
    os.makedirs(results_dir, exist_ok=True)

    # Load existing timing file so runs accumulate rather than overwrite
    timing_path = os.path.join(results_dir, "timing.json")
    timing = json.load(open(timing_path)) if os.path.exists(timing_path) else {}

    for seq in sequences:
        img_dir = os.path.join(data_root, seq, f"image_{camera_id}")
        frame_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not frame_paths:
            print(f"  [skip] No PNG frames found for sequence {seq} in {img_dir}")
            continue

        n_frames  = len(frame_paths)
        windows   = build_windows(frame_paths, window_size, overlap)
        print(f"\nSequence {seq}: {n_frames} frames, {len(windows)} windows")

        raw_preds = []
        t_start = time.perf_counter()
        with torch.no_grad():
            for win in tqdm(windows, desc=f"Seq {seq}", unit="win"):
                x = preprocess_window(win, transform).to(device)
                pred = model(x.float())
                pred = pred.reshape(window_size - 1, 6)
                raw_preds.append(pred.cpu().numpy())
        elapsed = time.perf_counter() - t_start

        raw_preds = np.stack(raw_preds, axis=0)

        # Save raw predictions
        npy_path = os.path.join(results_dir, f"pred_poses_{seq}.npy")
        np.save(npy_path, raw_preds)

        # Record timing
        timing[seq] = {
            "elapsed_s":       round(elapsed, 3),
            "frames":          n_frames,
            "windows":         len(windows),
            "ms_per_frame":    round(elapsed / n_frames * 1000, 3),
            "ms_per_window":   round(elapsed / len(windows) * 1000, 3),
        }
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)

        print(f"  elapsed: {elapsed:.1f}s  ({timing[seq]['ms_per_frame']} ms/frame)")
        print(f"  Saved predictions → {npy_path}")
        print(f"  Saved timing      → {timing_path}")

        # Post-process and recover trajectory
        poses      = post_processing(raw_preds, window_size)
        trajectory = recover_trajectory(poses)

        # Load ground truth if available
        gt_path = os.path.join(poses_dir, f"{seq}.txt")
        gt = load_gt_trajectory(gt_path) if os.path.exists(gt_path) else None

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(trajectory[:, 0], trajectory[:, 2], "b-", linewidth=1.5, label="Predicted")
        ax.scatter(trajectory[0, 0],  trajectory[0, 2],  c="green", s=80, zorder=5, label="Start")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 2], c="red",   s=80, zorder=5, label="End")
        if gt is not None:
            ax.plot(gt[:, 0], gt[:, 2], "r--", linewidth=1.5, label="Ground Truth")
        elapsed_str = f"{elapsed:.1f}s  ({timing[seq]['ms_per_frame']} ms/frame)"
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"TSformer-VO | {model_name} | Seq {seq} | {elapsed_str}")
        ax.legend()
        ax.grid(True)
        ax.set_aspect("equal")

        plot_path = os.path.join(results_dir, f"trajectory_{seq}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot        → {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="Model1",
                        choices=["Model1", "Model2", "Model3"])
    parser.add_argument("--sequences", nargs="+", default=["04", "10"])
    parser.add_argument("--data_root", default="data/sequences_png")
    parser.add_argument("--camera_id", default="0")
    parser.add_argument("--poses_dir", default="data/sequences_png/poses")
    args = parser.parse_args()

    run(args.model, args.sequences, args.data_root, args.camera_id, args.poses_dir)
