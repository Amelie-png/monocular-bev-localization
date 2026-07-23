import numpy as np
import pandas as pd
from itertools import combinations

def trajectory_consistency(bev_df, max_speed=None):
  """
  Frame-to-frame displacement per track. No GT needed.
  Flags implausible jumps if max_speed (units/frame) is given.
  """
  results = []
  for track_id, g in bev_df.sort_values("frame_id").groupby("track_id"):
    dx = g["bev_x"].diff()
    dy = g["bev_y"].diff()
    dist = np.sqrt(dx**2 + dy**2)
    results.append({
      "track_id": track_id,
      "mean_step_dist": dist.mean(),
      "max_step_dist": dist.max(),
      "n_implausible_jumps": (dist > max_speed).sum() if max_speed else None,
    })
  return pd.DataFrame(results)


def relative_spatial_accuracy(merged_df):
  """
  Compares predicted vs GT pairwise distances between matched players,
  per frame. Rotation/translation-invariant by construction.

  Args:
    merged_df: output of build_eval_dataset, must have frame_id, player_name, world_x, world_y, x, y
  """
  rows = []
  for frame_id, g in merged_df.groupby("frame_id"):
    if len(g) < 2:
      continue
    for (_, a), (_, b) in combinations(g.iterrows(), 2):
      pred_dist = np.hypot(a["world_x"] - b["world_x"], a["world_y"] - b["world_y"])
      gt_dist = np.hypot(a["x"] - b["x"], a["y"] - b["y"])
      rows.append({
        "frame_id": frame_id,
        "pair": (a["player_name"], b["player_name"]),
        "pred_dist": pred_dist,
        "gt_dist": gt_dist,
        "abs_error": abs(pred_dist - gt_dist),
      })
  return pd.DataFrame(rows)