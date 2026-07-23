import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

def match_frame(pred_points, gt_points, max_dist=None):
  """
  pred_points: list of (track_id, x, y) in WORLD coords
  gt_points: list of (player_name, x, y)
  """
  if not pred_points or not gt_points:
    return []
  pred_xy = np.array([[p[1], p[2]] for p in pred_points])
  gt_xy = np.array([[g[1], g[2]] for g in gt_points])
  cost = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=2)
  pred_idx, gt_idx = linear_sum_assignment(cost)
  matches = []
  for pi, gi in zip(pred_idx, gt_idx):
    dist = cost[pi, gi]
    if max_dist is None or dist <= max_dist:
      matches.append((pred_points[pi][0], gt_points[gi][0], dist))
  return matches


def build_eval_dataset(
  video_name,
  transform_fn,
  round_number,
  variant="heuristic",
  max_dist=None,
  track_label_dir="data/processed/track_labels",
  gt_dir="data/processed/player_positions",
  sync_dir="data/processed/sync",
  bev_dir=None):
  bev_dir = Path(bev_dir or f"data/processed/bev/{variant}")

  bev_df = pd.read_parquet(bev_dir / f"{video_name}_estimations.parquet")
  bev_df = bev_df[bev_df["track_id"] != -1]

  # exclude POV-self / dead-body tracks using manual labels
  label_path = Path(track_label_dir) / f"{video_name}_track_labels.parquet"
  if label_path.exists():
    labels = pd.read_parquet(label_path)
    exclude_ids = set(labels[labels["status"].isin(["invalid_pov", "dead_body"])]["track_id"])
    bev_df = bev_df[~bev_df["track_id"].isin(exclude_ids)]

  sync_df = pd.read_parquet(Path(sync_dir) / f"{video_name}_sync.parquet")

  match_name = "_".join(video_name.split("_")[:2])
  gt_df = pd.read_parquet(Path(gt_dir) / f"{match_name}_positions.parquet")
  gt_df = gt_df[gt_df["is_alive"] & (gt_df["round_number"] == round_number)]

  bev_df = bev_df.merge(sync_df[["frame_id", "tick", "cam_x", "cam_y", "yaw_deg"]], on="frame_id", how="inner")

  wx, wy = transform_fn(bev_df["bev_x"].values, bev_df["bev_y"].values, bev_df["cam_x"].values, bev_df["cam_y"].values, bev_df["yaw_deg"].values)
  bev_df = bev_df.assign(world_x=wx, world_y=wy)

  rows = []
  for tick, bev_g in bev_df.groupby("tick"):
    gt_g = gt_df[gt_df["tick"] == tick]
    if gt_g.empty:
      continue
    pred_points = list(zip(bev_g["track_id"], bev_g["world_x"], bev_g["world_y"]))
    gt_points = list(zip(gt_g["player_name"], gt_g["x"], gt_g["y"]))
    matches = match_frame(pred_points, gt_points, max_dist=max_dist)

    for track_id, player_name, dist in matches:
      pred_row = bev_g[bev_g["track_id"] == track_id].iloc[0]
      gt_row = gt_g[gt_g["player_name"] == player_name].iloc[0]
      rows.append({
        "frame_id": pred_row["frame_id"], "tick": tick,
        "track_id": track_id, "player_name": player_name,
        "world_x": pred_row["world_x"], "world_y": pred_row["world_y"],
        "x": gt_row["x"], "y": gt_row["y"], "euclidean_error": dist,
      })

  return pd.DataFrame(rows)