import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.evaluation.calibrate_transform import make_transform
from src.utils import track_color

def compute_world_coords(video_name, variant, eval_dir="outputs/eval", bev_dir=None, sync_dir="data/processed/sync"):
  """Projects all tracks (not just GT-matched ones) using the per-video
  scale/convention already saved during run_evaluation.py.
  """
  bev_dir = Path(bev_dir or f"data/processed/bev/{variant}")
  bev_df = pd.read_parquet(bev_dir / f"{video_name}_estimations.parquet")
  bev_df = bev_df[bev_df["track_id"] != -1]

  euc_path = Path(eval_dir) / f"{variant}_euclidean_error.parquet"
  euc_df = pd.read_parquet(euc_path)
  video_calib = euc_df[euc_df["video_name"] == video_name]
  if video_calib.empty:
    raise ValueError(f"No calibration found for {video_name} in {euc_path}")
  scale = float(video_calib["scale"].iloc[0])
  convention = video_calib["convention"].iloc[0]
  transform_fn = make_transform(convention, scale)

  sync_df = pd.read_parquet(Path(sync_dir) / f"{video_name}_sync.parquet")
  bev_df = bev_df.merge(sync_df[["frame_id", "cam_x", "cam_y", "yaw_deg"]], on="frame_id", how="inner")

  wx, wy = transform_fn(bev_df["bev_x"].values, bev_df["bev_y"].values, bev_df["cam_x"].values, bev_df["cam_y"].values, bev_df["yaw_deg"].values)
  return bev_df.assign(world_x=wx, world_y=wy)


def render_trajectory_video(video_name, variant, out_dir="outputs/trajectory", canvas_size=800, decay=0.90, fps=30, padding_ratio=0.1):
  df = compute_world_coords(video_name, variant)
  if df.empty:
    print(f"[{video_name}] no world-coordinate rows, skipping")
    return None

  x_min, x_max = df["world_x"].min(), df["world_x"].max()
  y_min, y_max = df["world_y"].min(), df["world_y"].max()
  x_pad = (x_max - x_min) * padding_ratio + 1e-6
  y_pad = (y_max - y_min) * padding_ratio + 1e-6
  x_min, x_max = x_min - x_pad, x_max + x_pad
  y_min, y_max = y_min - y_pad, y_max + y_pad

  def to_px(x, y):
    px = int((x - x_min) / (x_max - x_min) * canvas_size)
    py = int(canvas_size - (y - y_min) / (y_max - y_min) * canvas_size)
    return px, py

  out_path = Path(out_dir) / f"{video_name}_{variant}_trajectory.mp4"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_size, canvas_size))

  # Persistent canvas that fades toward white each frame to get comet trail effect
  canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.float32)

  frame_ids = sorted(df["frame_id"].unique())
  for frame_id in tqdm(frame_ids, desc=f"{video_name} ({variant}) trajectory", unit="frame", leave=False):
    canvas = canvas * decay + 255 * (1 - decay)  # fade existing trail toward white
    frame_rows = df[df["frame_id"] == frame_id]

    for _, row in frame_rows.iterrows():
      px, py = to_px(row["world_x"], row["world_y"])
      if 0 <= px < canvas_size and 0 <= py < canvas_size:
        cv2.circle(canvas, (px, py), 3, track_color(row["track_id"]), -1)

    frame_img = canvas.copy()
    for _, row in frame_rows.iterrows():
      px, py = to_px(row["world_x"], row["world_y"])
      if 0 <= px < canvas_size and 0 <= py < canvas_size:
        color = track_color(row["track_id"])
        cv2.circle(frame_img, (px, py), 7, color, -1)
        cv2.circle(frame_img, (px, py), 7, (0, 0, 0), 1)  # outline for contrast

    writer.write(np.clip(frame_img, 0, 255).astype(np.uint8))

  writer.release()
  print(f"Trajectory video saved to: {out_path}")
  return out_path


if __name__ == "__main__":
  from src.utils import load_split_file
  videos = load_split_file("data/splits/train.txt")
  for v in videos:
    render_trajectory_video(v, "heuristic")
    render_trajectory_video(v, "midas")