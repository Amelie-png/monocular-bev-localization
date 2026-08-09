from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.detection import DetectionVisualizer
from src.detection.frame_builder import build_frame_detections
from src.utils import filter_missing, report_skip, load_split_file


def detection_video_output_path(video_name, output_dir="outputs/videos/detection"):
  return Path(output_dir) / f"{video_name}.mp4"


def render_detection_video(video_name, det_dir="data/processed/detections", output_dir="outputs/videos/detection", fps=30):
  det_file = Path(det_dir) / f"{video_name}_detections.parquet"
  if not det_file.exists():
    print(f"Detection file not found: {det_file}")
    return None

  df = pd.read_parquet(det_file)
  frame_paths, detections_list = build_frame_detections(df)

  out_path = detection_video_output_path(video_name, output_dir)
  DetectionVisualizer().create_detection_video(frame_paths, detections_list, out_path, fps=fps)
  return out_path


def run_detection_visualization(video_names=None, det_dir="data/processed/detections", output_dir="outputs/videos/detection", fps=30, force=False):
  det_dir = Path(det_dir)
  detection_files = list(det_dir.glob("*_detections.parquet"))
  all_names = video_names if video_names else [f.stem.replace("_detections", "") for f in detection_files]

  def out_path_fn(v):
    return detection_video_output_path(v, output_dir)

  todo = filter_missing(all_names, out_path_fn, force=force)
  report_skip("detection visualization", all_names, todo)

  for v in tqdm(todo, desc="Rendering detection videos", unit="video"):
    render_detection_video(v, det_dir=det_dir, output_dir=output_dir, fps=fps)

  print("\nDetection visualization complete!")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--video", type=str, nargs="+", default=None, help="Specific videos to visualize")
  parser.add_argument("--output", type=str, default="outputs/videos/detection", help="Path to output directory")
  parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
  parser.add_argument("--force", action="store_true", default=None, help="Force re-render")

  args = parser.parse_args()

  video_names = args.video if args.video else load_split_file("data/splits/train.txt")
  run_detection_visualization(video_names=video_names, output_dir=args.output, fps=args.fps, force=args.force)