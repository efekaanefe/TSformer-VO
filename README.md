# TSformer-VO Inference Tutorial — `run_inference.py`

A step-by-step guide to running **monocular visual odometry inference** with the
TSformer-VO Transformer models on KITTI-style sequences. Given a folder of PNG frames,
the script slides a short clip window over them, predicts the per-clip relative motion
with a trained model, recovers the full trajectory, and saves predictions, a trajectory
plot, and timing.

```bash
python run_inference.py --model Model2 --sequences 04 07 10
```

> This is the **argparse-driven** inference entry point added to this fork. The original
> upstream workflow (editing variables in `predict_poses.py`) is still described in the
> main [`README.md`](README.md); this file documents the easier `run_inference.py` CLI.

---

## 1. What the script does

For the chosen `--model` and **each** `--sequences` entry:

1. Loads the model checkpoint and its saved hyper-parameters (`checkpoints/<model>/`).
2. Builds overlapping clip **windows** of `Nf` frames (`Nf = 2/3/4` for Model1/2/3).
3. Runs each window through the network → `(Nf-1) × 6` relative-pose predictions
   (3 ZYX Euler angles + 3 translation, normalised).
4. Merges the overlapping windows, de-normalises, and **chains** the relative motions into
   an absolute trajectory.
5. Saves the raw predictions (`.npy`), a trajectory plot vs. ground truth, and `timing.json`.

> **Note:** this script plots the trajectory in the model's own (already roughly metric)
> frame — it does **not** apply Umeyama alignment or compute ATE. For aligned trajectories
> and ATE, see §8.

---

## 2. Requirements

- **Python 3.8**, **PyTorch 1.10.1** (CPU or CUDA), plus `torchvision`, `numpy`,
  `matplotlib`, `Pillow`, `einops`, `tqdm`, `pyyaml` (full pinned list in
  [`requirements.txt`](requirements.txt)).

Create the environment exactly as upstream recommends:

```bash
conda create -n tsformer-vo python==3.8.0
conda activate tsformer-vo
pip install -r requirements.txt
```

The script auto-selects the device:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

so it runs on CPU with no changes (just slower — see the timing numbers in §7).

> **This machine:** the prepared interpreter is
> `C:/Users/EFO/miniconda3/envs/tsformer-vo/python.exe` (PyTorch 1.10.1, CPU).
> **Run from the repo root** so the local `datasets/` and `timesformer/` packages import:
> ```powershell
> cd D:/Python_Related/Robotics/Robotics-Masters-Related/EE584-Machine_Vision/term_project_repo/TSformer-VO
> C:/Users/EFO/miniconda3/envs/tsformer-vo/python.exe run_inference.py --model Model2 --sequences 10
> ```

---

## 3. Pre-trained checkpoints

The three variants ship under `checkpoints/`, each with a `.pth` weight file and an
`args.pkl` holding `window_size`, `overlap`, and the model dimensions:

| `--model` | Frames/clip `Nf` | Poses/clip | Checkpoint folder |
|-----------|------------------|------------|-------------------|
| `Model1`  | 2 | 1 | `checkpoints/Model1/` |
| `Model2`  | 3 | 2 | `checkpoints/Model2/` |
| `Model3`  | 4 | 3 | `checkpoints/Model3/` |

The script picks the single `.pth` it finds in the folder automatically — you only pass
`--model`. (To re-download, see the Google Drive links in the main `README.md`.)

---

## 4. Expected dataset layout

```
<data_root>/
  <seq>/
    image_<camera_id>/*.png      # the frames (image_0 = left grayscale)
    calib.txt                    # not used by this script, but kept with the sequence
poses/                           # default poses_dir = <data_root>/poses
  <seq>.txt                      # 12-value 3x4 ground-truth rows (optional)
```

- `--data_root` defaults to `data/sequences_png` (relative to the repo).
- Ground truth is **optional**: without `<seq>.txt` the trajectory is still produced and
  plotted, just with no red GT overlay.
- Sequence `22` (Isaac Sim) keeps its frames in `image_2`, so run it with `--camera_id 2`.

---

## 5. Quick start

Run the default model on a short, smooth sequence:

```bash
python run_inference.py --model Model1 --sequences 10
```

Expected console output:

```
Device: cpu
Model: Model1 | checkpoint: checkpoint_model1_exp12
window_size=2, overlap=1, num_frames=2

Sequence 10: 1201 frames, 1200 windows
Seq 10: 100%|██████████| 1200/1200 [..]
  elapsed: ...s  (... ms/frame)
  Saved predictions → results/Model1/pred_poses_10.npy
  Saved timing      → results/Model1/timing.json
  Saved plot        → results/Model1/trajectory_10.png
```

---

## 6. Command-line reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `Model1` | Which variant to run: `Model1`, `Model2`, or `Model3`. |
| `--sequences` | `04 10` | One or more sequence IDs to process. |
| `--data_root` | `data/sequences_png` | Root folder holding `<seq>/image_*/`. |
| `--camera_id` | `0` | Which camera's frames to read (`image_<id>`). |
| `--poses_dir` | `data/sequences_png/poses` | Folder with ground-truth `<seq>.txt` files. |

There is no `--output_dir`: results always go to **`results/<model>/`**. There are no
`--stride`, `--max_frames`, or bundle-adjustment flags (those belong to the classical
pipeline) — window size and overlap come from each checkpoint's `args.pkl`.

```bash
python run_inference.py --help
```

---

## 7. Outputs explained

All written under `results/<model>/`:

| File | Contents |
|------|----------|
| `pred_poses_<seq>.npy` | Raw per-window predictions, shape `(num_windows, Nf-1, 6)`. The 6 = 3 normalised ZYX Euler angles + 3 normalised translation. |
| `trajectory_<seq>.png` | Predicted trajectory (solid blue) vs. ground truth (dashed red), with start/end markers. |
| `timing.json` | Per-sequence timing, accumulated across runs into one JSON object. |

A `timing.json` entry:

```json
"10": {
  "elapsed_s": 1410.76,
  "frames": 1201,
  "windows": 1199,
  "ms_per_frame": 1174.7,
  "ms_per_window": 1176.6
}
```

> The `.npy` holds **relative, normalised** predictions, *not* absolute poses. To turn
> them into a trajectory, the script de-normalises with the dataset mean/std, converts the
> Euler angles to rotations, builds each `4×4` relative transform, and chains them
> (`post_processing` + `recover_trajectory` in this file). The classical-repo tool in §8
> reuses the exact same recovery.

---

## 8. Aligned trajectories and ATE (cross-repo)

`run_inference.py` plots the **raw** trajectory only. To get the Umeyama-aligned plots and
the per-sequence `scale_to_gt` / `ate_m` used in the report, run the visualiser from the
classical-VO repo, which reads these same `.npy` files:

```bash
# from the Visual-Inertial-Odometry repo
python tools/visualize_tsformer.py --models Model1 Model2 Model3 --sequences 04 07 10 --mode all
```

It writes `trajectory_<seq>_<mode>.png` (raw / aligned / anchored) into
`results/<model>/views/` and appends `scale_to_gt` / `ate_m` to each model's `timing.json`.

---

## 9. Common recipes

```bash
# All three models on the report's sequences
python run_inference.py --model Model1 --sequences 00 01 02 03 04 05 06 07 08 09 10
python run_inference.py --model Model2 --sequences 00 01 02 03 04 05 06 07 08 09 10
python run_inference.py --model Model3 --sequences 00 01 02 03 04 05 06 07 08 09 10

# Sequence 22 (Isaac Sim) — frames live in image_2
python run_inference.py --model Model2 --sequences 22 --camera_id 2

# A dataset in a custom location
python run_inference.py --model Model3 --sequences 04 --data_root D:/datasets/kitti/sequences_png --poses_dir D:/datasets/kitti/poses
```

---
