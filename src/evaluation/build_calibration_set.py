import pandas as pd
from pathlib import Path

def build_calibration_set(
    video_name, 
    variant="heuristic",
    track_label_dir="data/processed/track_labels",
    bev_dir=None, 
    sync_dir="data/processed/sync",
    gt_dir="data/processed/player_positions"
  ):
  bev_dir = Path(bev_dir or f"data/processed/bev/{variant}")
  bev_df = pd.read_parquet(bev_dir / f"{video_name}_estimations.parquet")

  labels = pd.read_parquet(Path(track_label_dir) / f"{video_name}_track_labels.parquet")
  valid = labels[labels["status"] == "valid"][["track_id", "player_name"]]
  bev_df = bev_df.merge(valid, on="track_id", how="inner")

  sync_df = pd.read_parquet(Path(sync_dir) / f"{video_name}_sync.parquet")
  bev_df = bev_df.merge(sync_df[["frame_id", "tick", "cam_x", "cam_y", "yaw_deg"]], on="frame_id", how="inner")

  match_name = "_".join(video_name.split("_")[:2])
  gt_df = pd.read_parquet(Path(gt_dir) / f"{match_name}_positions.parquet")
  gt_df = gt_df[gt_df["is_alive"]]

  merged = bev_df.merge(gt_df[["tick", "player_name", "x", "y"]], on=["tick", "player_name"], how="inner")
  return merged.rename(columns={"x": "gt_x", "y": "gt_y"})