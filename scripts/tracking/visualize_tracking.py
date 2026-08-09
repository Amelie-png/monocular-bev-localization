from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.detection import DetectionVisualizer, build_frame_detections
from src.utils import filter_missing, report_skip, load_split_file


def tracking_video_output_path(video_name, output_dir="outputs/videos/tracking"):
  return Path(output_dir) / f"{video_name}.mp4"


def render_tracking_video(video_name, track_dir="data/processed/trackings", output_dir="outputs/videos/tracking", fps=30):
  track_file = Path(track_dir) / f"{video_name}_trackings.parquet"
  if not track_file.exists():
    print(f"Tracking file not found: {track_file}")
    return None

  df = pd.read_parquet(track_file)
  frame_paths, detections_list = build_frame_detections(df)

  out_path = tracking_video_output_path(video_name, output_dir)
  DetectionVisualizer().create_detection_video(frame_paths, detections_list, out_path, fps=fps)
  return out_path


def run_tracking_visualization(video_names=None, track_dir="data/processed/trackings", output_dir="outputs/videos/tracking", fps=30, force=False):
  track_dir = Path(track_dir)
  tracking_files = list(track_dir.glob("*_trackings.parquet"))
  all_names = video_names if video_names else [f.stem.replace("_trackings", "") for f in tracking_files]

  def out_path_fn(v):
    return tracking_video_output_path(v, output_dir)

  todo = filter_missing(all_names, out_path_fn, force=force)
  report_skip("tracking visualization", all_names, todo)

  for v in tqdm(todo, desc="Rendering tracking videos", unit="video"):
    render_tracking_video(v, track_dir=track_dir, output_dir=output_dir, fps=fps)

  print("\nTracking visualization complete!")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--video", type=str, nargs="+", default=None, help="Specific videos to visualize")
  parser.add_argument("--output", type=str, default="outputs/videos/tracking", help="Path to output directory")
  parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
  parser.add_argument("--force", action="store_true", default=None, help="Force re-render")

  args = parser.parse_args()

  video_names = args.video if args.video else load_split_file("data/splits/train.txt")
  run_tracking_visualization(video_names=video_names, output_dir=args.output, fps=args.fps, force=args.force)