import pandas as pd
from pathlib import Path

from src.evaluation import (relative_spatial_accuracy, match_frame, calibrate_convention, 
                            fit_scale_for_video, make_transform, build_calibration_set)
from src.utils import load_split_file


def video_round_number(video_name):
  return int(video_name.split("_")[-1])


def build_eval_dataset_notrack(
    video_name, transform_fn, round_number, max_dist=None,
    bev_dir="data/processed/bev/heuristic_notrack",
    sync_dir="data/processed/sync",
    gt_dir="data/processed/player_positions"):
  """
  Build evaluation dataset for pseudo tracked videos, using Hungarian assignment.
  """
  bev_path = Path(bev_dir) / f"{video_name}_estimations.parquet"
  if not bev_path.exists():
    print(f"[{video_name}] no notrack bev file at {bev_path}, skipping")
    return pd.DataFrame()

  bev_df = pd.read_parquet(bev_path)

  sync_df = pd.read_parquet(Path(sync_dir) / f"{video_name}_sync.parquet")
  bev_df = bev_df.merge(sync_df[["frame_id", "tick", "cam_x", "cam_y", "yaw_deg"]], on="frame_id", how="inner")

  match_name = "_".join(video_name.split("_")[:2])
  gt_df = pd.read_parquet(Path(gt_dir) / f"{match_name}_positions.parquet")
  gt_df = gt_df[gt_df["is_alive"] & (gt_df["round_number"] == round_number)]

  wx, wy = transform_fn(bev_df["bev_x"].values, bev_df["bev_y"].values, bev_df["cam_x"].values, bev_df["cam_y"].values, bev_df["yaw_deg"].values)
  bev_df = bev_df.assign(world_x=wx, world_y=wy)

  rows = []
  for tick, bev_g in bev_df.groupby("tick"):
    gt_g = gt_df[gt_df["tick"] == tick]
    if gt_g.empty:
      continue
    pred_points = list(zip(bev_g["track_id"], bev_g["world_x"], bev_g["world_y"]))
    gt_points = list(zip(gt_g["player_name"], gt_g["x"], gt_g["y"]))
    for pred_id, player_name, dist in match_frame(pred_points, gt_points, max_dist=max_dist):
      pred_row = bev_g[bev_g["track_id"] == pred_id].iloc[0]
      gt_row = gt_g[gt_g["player_name"] == player_name].iloc[0]
      rows.append({
        "frame_id": pred_row["frame_id"], "tick": tick,
        "player_name": player_name, "world_x": pred_row["world_x"], "world_y": pred_row["world_y"],
        "x": gt_row["x"], "y": gt_row["y"], "euclidean_error": dist,
      })

  return pd.DataFrame(rows)


def run_evaluation_notrack(video_names, midas=False, out_dir="outputs/eval"):
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  # Reuse the same calibration convention/scale as tracked
  variant = "midas" if midas else "heuristic"
  calib_frames = [build_calibration_set(v, variant=variant) for v in video_names]
  calib_all = pd.concat([c for c in calib_frames if not c.empty], ignore_index=True)
  convention = calibrate_convention(calib_all)

  all_euc, all_rel = [], []
  for v, calib_v in zip(video_names, calib_frames):
    if calib_v.empty:
      print(f"[{v}] no calibration rows available, skipping")
      continue
    scale = fit_scale_for_video(calib_v, convention)
    transform_fn = make_transform(convention, scale)

    bev_dir = f"data/processed/bev/{variant}_notrack"
    merged = build_eval_dataset_notrack(v, transform_fn, bev_dir=bev_dir, round_number=video_round_number(v))
    if merged.empty:
      print(f"[{v}] no matched notrack eval rows, skipping")
      continue

    merged["video_name"] = v
    rel = relative_spatial_accuracy(merged)
    rel["video_name"] = v

    all_euc.append(merged)
    all_rel.append(rel)
    print(f"[{v}] scale={scale:.4f}, mean euclidean_error={merged['euclidean_error'].mean():.2f}")

  if not all_euc:
    print(f"[{variant}_notrack] No videos produced usable eval rows.")
    return pd.DataFrame(), pd.DataFrame()

  euc_df = pd.concat(all_euc, ignore_index=True)
  rel_df = pd.concat(all_rel, ignore_index=True)

  euc_df.to_parquet(out_dir / f"{variant}_notrack_euclidean_error.parquet")
  rel_df.to_parquet(out_dir / f"{variant}_notrack_relative_spatial_accuracy.parquet")

  print(f"\n[{variant}_notrack] Summary")
  print(f"  Mean relative spatial error: {rel_df['abs_error'].mean():.3f}")
  print(f"  Mean euclidean error: {euc_df['euclidean_error'].mean():.3f}")
  return euc_df, rel_df


if __name__ == "__main__":
  videos = load_split_file("data/splits/train.txt")
  run_evaluation_notrack(videos, midas=False)