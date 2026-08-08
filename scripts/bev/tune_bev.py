import itertools
import numpy as np
import pandas as pd
from pathlib import Path

from src.bev import BevEstimator
from src.depth import compute_depth_normalization_stats
from src.evaluation.calibrate_transform import calibrate_convention, fit_scale_for_video, make_transform
from src.utils import load_split_file


def load_video_base_data(
    video_name,
    track_label_dir="data/processed/track_labels",
    tracking_dir="data/processed/trackings",
    sync_dir="data/processed/sync",
    gt_dir="data/processed/player_positions"):
  """
  Loads everything that does not depend on tuning parameters
  (labels, tracking, sync, ground truth) once per video and reuse across every config.
  """
  label_path = Path(track_label_dir) / f"{video_name}_track_labels.parquet"
  tracking_file = Path(tracking_dir) / f"{video_name}_trackings.parquet"
  sync_file = Path(sync_dir) / f"{video_name}_sync.parquet"
  if not (label_path.exists() and tracking_file.exists() and sync_file.exists()):
    return None

  labels = pd.read_parquet(label_path)
  valid = labels[labels["status"] == "valid"][["track_id", "player_name"]]
  if valid.empty:
    return None

  tracking_df = pd.read_parquet(tracking_file).merge(valid, on="track_id", how="inner")
  if tracking_df.empty:
    return None

  sync_df = pd.read_parquet(sync_file)
  tracking_df = tracking_df.merge(
    sync_df[["frame_id", "tick", "cam_x", "cam_y", "yaw_deg"]], on="frame_id", how="inner"
  )

  match_name = "_".join(video_name.split("_")[:2])
  gt_df = pd.read_parquet(Path(gt_dir) / f"{match_name}_positions.parquet")
  gt_df = gt_df[gt_df["is_alive"]]

  return {"tracking_df": tracking_df, "gt_df": gt_df}

def build_calibration_set_from_base(base, depth_window, fov_scale, midas, dmin, dmax, depth_dir, video_name):
  """
  Runs BEV estimation with the given params on already-loaded base data.
  """
  tracking_df, gt_df = base["tracking_df"], base["gt_df"]
  estimator = BevEstimator(depth_window=depth_window, fov_scale=fov_scale)
  video_depth_dir = Path(depth_dir) / video_name if midas else None

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
      bev = (estimator.estimate(bbox, depth_map=depth_map, depth_min=dmin, depth_max=dmax)
             if midas else estimator.estimate(bbox))
      rows.append({
        "frame_id": frame_id, "track_id": row["track_id"], "player_name": row["player_name"],
        "bev_x": bev["bev_x"], "bev_y": bev["bev_y"],
        "tick": row["tick"], "cam_x": row["cam_x"], "cam_y": row["cam_y"], "yaw_deg": row["yaw_deg"],
      })

  bev_df = pd.DataFrame(rows)
  if bev_df.empty:
    return bev_df

  merged = bev_df.merge(gt_df[["tick", "player_name", "x", "y"]], on=["tick", "player_name"], how="inner")
  return merged.rename(columns={"x": "gt_x", "y": "gt_y"})

def evaluate_bev_config(video_bases, depth_window, low_pct, high_pct, fov_scale, midas, depth_dir="data/processed/depths", norm_cache=None):
  errors = []
  for video_name, base in video_bases.items():
    if base is None:
      continue

    dmin = dmax = None
    if midas:
      key = (video_name, low_pct, high_pct)
      if norm_cache is not None and key in norm_cache:
        dmin, dmax = norm_cache[key]
      else:
        dmin, dmax = compute_depth_normalization_stats(
          video_name, depth_dir=depth_dir, low_pct=low_pct, high_pct=high_pct
        )
        if norm_cache is not None:
          norm_cache[key] = (dmin, dmax)

    calib = build_calibration_set_from_base(base, depth_window, fov_scale, midas, dmin, dmax, depth_dir, video_name)
    if calib.empty:
      continue

    convention = calibrate_convention(calib)
    scale = fit_scale_for_video(calib, convention)
    transform_fn = make_transform(convention, scale)
    dx, dy = transform_fn(calib["bev_x"].values, calib["bev_y"].values, calib["cam_x"].values, calib["cam_y"].values, calib["yaw_deg"].values)
    err = np.hypot(dx - calib["gt_x"].values, dy - calib["gt_y"].values)
    errors.append(err.mean())

  return np.mean(errors) if errors else float("inf")


def grid_search(video_names, midas, out_dir="outputs/tuning/bev"):
  video_bases = {v: load_video_base_data(v) for v in video_names}
  norm_cache = {}

  if midas:
    depth_windows, low_pcts, high_pcts = [3, 5, 7], [1, 2], [95, 98, 99]
    fov_scales = [0.85, 1.0, 1.15]
    variant_label = "midas"
  else:
    depth_windows, low_pcts, high_pcts = [5], [2], [98]
    fov_scales = [0.7, 0.85, 1.0, 1.15, 1.3]
    variant_label = "heuristic"

  results = []
  best = None
  for dw, lp, hp, fs in itertools.product(depth_windows, low_pcts, high_pcts, fov_scales):
    if lp >= hp:
      continue
    err = evaluate_bev_config(video_bases, dw, lp, hp, fs, midas, norm_cache=norm_cache)
    print(f"depth_window={dw} low={lp} high={hp} fov_scale={fs} -> mean_error={err:.2f}")
    results.append({"depth_window": dw, "low_pct": lp, "high_pct": hp, "fov_scale": fs, "mean_error": err})
    if best is None or err < best[1]:
      best = ((dw, lp, hp, fs), err)

  print(f"\nBest ({variant_label}): {best}")

  variant_root = Path(out_dir) / variant_label
  variant_root.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(results).to_csv(variant_root / "tuning_summary.csv", index=False)
  print(f"Saved to: {variant_root / 'tuning_summary.csv'}")

  return best


if __name__ == "__main__":
  videos = load_split_file("data/splits/train.txt")
  print("=== Tuning heuristic (fov_scale only) ===")
  grid_search(videos, midas=False)
  print("\n=== Tuning midas (reduced grid) ===")
  grid_search(videos, midas=True)