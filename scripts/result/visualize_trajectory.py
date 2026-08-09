from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import filter_missing, report_skip, track_color, load_split_file, compute_fit_scale


def trajectory_video_output_path(video_name, variant, out_dir=None):
  out_dir = out_dir or f"outputs/videos/trajectories_camera_relative/{variant}"
  return Path(out_dir) / f"{video_name}.mp4"


def render_camera_relative_trajectory(
    video_name, variant, bev_dir=None, out_dir=None,
    canvas_size=800, padding_ratio=0.15, decay=0.90,
    fps=30, scale=None):
  """
  Fading-trail trajectory, plotted directly from bev_x/bev_y against a fixed
  camera-pinned canvas -- no calibration/ground truth required.

  scale: if given, use this fixed scale (e.g. for cross-variant consistency).
  If None, auto-fit independently to this video+variant's own range.
  """
  bev_dir = Path(bev_dir or f"data/processed/bev/{variant}")
  bev_file = bev_dir / f"{video_name}_estimations.parquet"
  if not bev_file.exists():
    print(f"BEV estimations not found: {bev_file}")
    return None

  df = pd.read_parquet(bev_file)
  fit_scale = scale if scale is not None else compute_fit_scale(df, canvas_size, padding_ratio)

  out_path = trajectory_video_output_path(video_name, variant, out_dir)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_size, canvas_size))

  camera_x, camera_y = canvas_size // 2, canvas_size - 20
  canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.float32)

  for frame_id in tqdm(sorted(df["frame_id"].unique()), desc=f"{video_name} ({variant}) trail", unit="frame", leave=False):
    canvas = canvas * decay + 255 * (1 - decay)
    g = df[df["frame_id"] == frame_id]

    for _, row in g.iterrows():
      px = int(camera_x + row["bev_x"] * fit_scale)
      py = int(camera_y - row["bev_y"] * fit_scale)
      if 0 <= px < canvas_size and 0 <= py < canvas_size:
        cv2.circle(canvas, (px, py), 3, track_color(row["track_id"]), -1)

    frame_img = canvas.copy()
    for _, row in g.iterrows():
      px = int(camera_x + row["bev_x"] * fit_scale)
      py = int(camera_y - row["bev_y"] * fit_scale)
      if 0 <= px < canvas_size and 0 <= py < canvas_size:
        color = track_color(row["track_id"])
        cv2.circle(frame_img, (px, py), 7, color, -1)
        cv2.circle(frame_img, (px, py), 7, (0, 0, 0), 1)

    writer.write(np.clip(frame_img, 0, 255).astype(np.uint8))

  writer.release()
  print(f"Camera-relative trajectory video saved to: {out_path}")
  return out_path


def run_trajectory_visualization(video_names=None, variants=("heuristic", "midas"), fps=30, force=False):
  all_names = video_names if video_names else load_split_file("data/splits/train.txt")

  for variant in variants:
    def out_path_fn(v, variant=variant):
      return trajectory_video_output_path(v, variant)

    todo = filter_missing(all_names, out_path_fn, force=force)
    report_skip(f"trajectory visualization ({variant})", all_names, todo)

    for v in tqdm(todo, desc=f"Rendering trajectory videos ({variant})", unit="video"):
      render_camera_relative_trajectory(v, variant, fps=fps)

  print("\nTrajectory visualization complete!")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--video", type=str, nargs="+", default=None, help="Specific videos to visualize")
  parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
  parser.add_argument("--force", action="store_true", default=None, help="Force re-render")

  args = parser.parse_args()
  run_trajectory_visualization(video_names=args.video, fps=args.fps, force=args.force)