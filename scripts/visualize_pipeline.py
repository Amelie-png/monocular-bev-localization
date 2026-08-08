from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import subprocess
from tqdm import tqdm

from src.detection import DetectionVisualizer
from src.detection.frame_builder import build_frame_detections
from src.bev import BevVisualizer
from src.utils import load_split_file, track_color


def render_detection_video(video_name, det_dir="data/processed/detections", out_dir="outputs/detection_videos/final", fps=30):
  df = pd.read_parquet(Path(det_dir) / f"{video_name}_detections.parquet")
  frame_paths, detections_list = build_frame_detections(df)
  out_path = Path(out_dir) / f"{video_name}.mp4"
  DetectionVisualizer().create_detection_video(frame_paths, detections_list, out_path, fps=fps)
  return out_path


def render_tracking_video(video_name, track_dir="data/processed/trackings",
                           out_dir="outputs/tracking_videos/final", fps=30):
  df = pd.read_parquet(Path(track_dir) / f"{video_name}_trackings.parquet")
  frame_paths, detections_list = build_frame_detections(df)
  out_path = Path(out_dir) / f"{video_name}.mp4"
  DetectionVisualizer().create_detection_video(frame_paths, detections_list, out_path, fps=fps)
  return out_path


def render_bev_video(video_name, variant, bev_dir=None, out_dir=None, fps=30):
  bev_dir = Path(bev_dir or f"data/processed/bev/{variant}")
  out_dir = Path(out_dir or f"outputs/bev/{variant}")
  df = pd.read_parquet(bev_dir / f"{video_name}_estimations.parquet")

  frame_estimations = []
  for frame_id in sorted(df["frame_id"].unique()):
    g = df[df["frame_id"] == frame_id]
    frame_estimations.append([
      {"bev_x": row["bev_x"], "bev_y": row["bev_y"], "track_id": row["track_id"]}
      for _, row in g.iterrows()
    ])

  out_path = out_dir / f"{video_name}.mp4"
  BevVisualizer(scale=3).create_bev_video(frame_estimations, out_path, fps=fps)
  return out_path


def render_camera_relative_trajectory(video_name, variant, bev_dir=None,
                                       out_dir="outputs/trajectories_camera_relative",
                                       canvas_size=800, scale=3, decay=0.90, fps=30):
  """
  Fading-trail trajectory, plotted directly from bev_x/bev_y against a fixed
  camera-pinned canvas -- no calibration/ground truth required, so this can
  ship as part of the user-facing pipeline.
  """
  bev_dir = Path(bev_dir or f"data/processed/bev/{variant}")
  df = pd.read_parquet(bev_dir / f"{video_name}_estimations.parquet")

  out_path = Path(out_dir) / variant / f"{video_name}.mp4"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_size, canvas_size))

  camera_x, camera_y = canvas_size // 2, canvas_size - 20
  canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.float32)

  for frame_id in tqdm(sorted(df["frame_id"].unique()), desc=f"{video_name} ({variant}) trail",
                        unit="frame", leave=False):
    canvas = canvas * decay + 255 * (1 - decay)
    g = df[df["frame_id"] == frame_id]

    for _, row in g.iterrows():
      px = int(camera_x + row["bev_x"] * scale)
      py = int(camera_y - row["bev_y"] * scale)
      if 0 <= px < canvas_size and 0 <= py < canvas_size:
        cv2.circle(canvas, (px, py), 3, track_color(row["track_id"]), -1)

    frame_img = canvas.copy()
    for _, row in g.iterrows():
      px = int(camera_x + row["bev_x"] * scale)
      py = int(camera_y - row["bev_y"] * scale)
      if 0 <= px < canvas_size and 0 <= py < canvas_size:
        color = track_color(row["track_id"])
        cv2.circle(frame_img, (px, py), 7, color, -1)
        cv2.circle(frame_img, (px, py), 7, (0, 0, 0), 1)

    writer.write(np.clip(frame_img, 0, 255).astype(np.uint8))

  writer.release()
  print(f"Camera-relative trajectory video saved to: {out_path}")
  return out_path


def combine_videos_grid(video_paths, out_path, layout="hstack"):
  video_paths = [p for p in video_paths if p and Path(p).exists()]
  if len(video_paths) < 2:
    print(f"Not enough videos to combine for {out_path}, skipping")
    return None
  inputs = []
  for p in video_paths:
    inputs += ["-i", str(p)]
  n = len(video_paths)
  filter_complex = f"{''.join(f'[{i}:v]' for i in range(n))}{layout}=inputs={n}[v]"
  Path(out_path).parent.mkdir(parents=True, exist_ok=True)
  cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
         "-map", "[v]", "-c:v", "libx264", "-crf", "18", str(out_path)]
  subprocess.run(cmd, check=True)
  return out_path


def run_visualization(video_names, fps=30):
  for v in tqdm(video_names, desc="Visualizing", unit="video"):
    tracked = render_tracking_video(v, fps=fps)
    render_detection_video(v, fps=fps)
    heur = render_bev_video(v, "heuristic", fps=fps)
    midas = render_bev_video(v, "midas", fps=fps)
    render_camera_relative_trajectory(v, "heuristic", fps=fps)
    render_camera_relative_trajectory(v, "midas", fps=fps)
    combine_videos_grid([tracked, heur, midas], Path(f"outputs/combined/{v}.mp4"))

if __name__ == "__main__":
  videos = load_split_file("data/splits/train.txt")
  run_visualization(videos)