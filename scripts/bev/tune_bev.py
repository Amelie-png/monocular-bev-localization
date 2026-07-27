import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from src.bev import BevEstimator
from src.depth import compute_depth_normalization_stats
from src.evaluation.calibrate_transform import calibrate_convention, fit_scale_for_video, make_transform
from src.utils import load_split_file


def build_calibration_set_with_params(
  video_name, depth_window, low_pct, high_pct, fov_scale, midas,
  track_label_dir="data/processed/track_labels",
  tracking_dir="data/processed/trackings",
  depth_dir="data/processed/depths",
  sync_dir="data/processed/sync",
  gt_dir="data/processed/player_positions"):
  label_path = Path(track_label_dir) / f"{video_name}_track_labels.parquet"
  if not label_path.exists():
    return pd.DataFrame()
  labels = pd.read_parquet(label_path)

  valid = labels[labels["status"] == "valid"][["track_id", "player_name"]]
  if valid.empty:
    return pd.DataFrame()

  tracking_file = Path(tracking_dir) / f"{video_name}_trackings.parquet"
  if not tracking_file.exists():
    return pd.DataFrame()
  tracking_df = pd.read_parquet(tracking_file).merge(valid, on="track_id", how="inner")
  if tracking_df.empty:
    return pd.DataFrame()

  estimator = BevEstimator(depth_window=depth_window, fov_scale=fov_scale)

  dmin = dmax = None
  video_depth_dir = Path(depth_dir) / video_name
  if midas:
    dmin, dmax = compute_depth_normalization_stats(
      video_name, depth_dir=depth_dir, low_pct=low_pct, high_pct=high_pct
    )

  rows = []
  for frame_id, g in tracking_df.groupby("frame_id"):
    depth_map = None
    if midas:
      depth_path = video_depth_dir / f"frame_{frame_id:06d}.npy"
      if not depth_path.exists():
        continue
      depth_map = np.load(depth_path)

    for _, row in g.iterrows():
      bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]
      bev = (estimator.estimate(bbox, depth_map=depth_map, depth_min=dmin, depth_max=dmax) if midas else estimator.estimate(bbox))
      rows.append({
        "frame_id": frame_id, 
        "track_id": row["track_id"], 
        "player_name": row["player_name"],
        "bev_x": bev["bev_x"], 
        "bev_y": bev["bev_y"],
      })

  bev_df = pd.DataFrame(rows)
  if bev_df.empty:
    return bev_df

  sync_df = pd.read_parquet(Path(sync_dir) / f"{video_name}_sync.parquet")
  bev_df = bev_df.merge(sync_df[["frame_id", "tick", "cam_x", "cam_y", "yaw_deg"]], on="frame_id", how="inner")

  match_name = "_".join(video_name.split("_")[:2])
  gt_df = pd.read_parquet(Path(gt_dir) / f"{match_name}_positions.parquet")
  gt_df = gt_df[gt_df["is_alive"]]

  merged = bev_df.merge(gt_df[["tick", "player_name", "x", "y"]], on=["tick", "player_name"], how="inner")
  return merged.rename(columns={"x": "gt_x", "y": "gt_y"})


def evaluate_bev_config(video_names, depth_window, low_pct, high_pct, fov_scale, midas):
  errors = []
  for v in video_names:
    calib = build_calibration_set_with_params(v, depth_window, low_pct, high_pct, fov_scale, midas)
    if calib.empty:
      continue
    convention = calibrate_convention(calib)
    scale = fit_scale_for_video(calib, convention)
    transform_fn = make_transform(convention, scale)
    dx, dy = transform_fn(calib["bev_x"].values, calib["bev_y"].values, calib["cam_x"].values, calib["cam_y"].values, calib["yaw_deg"].values)
    err = np.hypot(dx - calib["gt_x"].values, dy - calib["gt_y"].values)
    errors.append(err.mean())
  return np.mean(errors) if errors else float("inf")


def grid_search(video_names, midas):
  depth_windows = [3, 5, 7, 9] if midas else [5]
  low_pcts = [1, 2, 5] if midas else [2]
  high_pcts = [95, 98, 99] if midas else [98]
  fov_scales = [0.7, 0.85, 1.0, 1.15, 1.3]

  best = None
  for dw, lp, hp, fs in itertools.product(depth_windows, low_pcts, high_pcts, fov_scales):
    if lp >= hp:
      continue
    err = evaluate_bev_config(video_names, dw, lp, hp, fs, midas)
    print(f"depth_window={dw} low={lp} high={hp} fov_scale={fs} -> mean_error={err:.2f}")
    if best is None or err < best[1]:
      best = ((dw, lp, hp, fs), err)

  print(f"\nBest: {best}")
  return best


if __name__ == "__main__":
  videos = load_split_file("data/splits/train.txt")
  print("=== Tuning heuristic (fov_scale only) ===")
  grid_search(videos, midas=False)
  print("\n=== Tuning midas ===")
  grid_search(videos, midas=True)