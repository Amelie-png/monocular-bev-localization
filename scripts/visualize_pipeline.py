from pathlib import Path
import subprocess
from tqdm import tqdm

from scripts.detection.visualize_detection import render_detection_video
from scripts.tracking.visualize_tracking import render_tracking_video
from scripts.bev.visualize_bev import render_bev_video, render_combined_bev_pair
from scripts.result.visualize_trajectory import render_camera_relative_trajectory
from src.utils import load_split_file


def combine_videos_grid(video_paths, out_path, layout="hstack", target_height=800):
  video_paths = [p for p in video_paths if p and Path(p).exists()]
  if len(video_paths) < 2:
    print(f"Not enough videos to combine for {out_path}, skipping")
    return None

  inputs = []
  for p in video_paths:
    inputs += ["-i", str(p)]

  n = len(video_paths)
  scale_filters = "".join(f"[{i}:v]scale=-2:{target_height}[s{i}];" for i in range(n))
  stack_inputs = "".join(f"[s{i}]" for i in range(n))
  filter_complex = f"{scale_filters}{stack_inputs}{layout}=inputs={n}[v]"

  Path(out_path).parent.mkdir(parents=True, exist_ok=True)
  cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
         "-map", "[v]", "-c:v", "libx264", "-crf", "18", str(out_path)]
  subprocess.run(cmd, check=True)
  return out_path


def run_visualization(video_names, fps=30):
  for v in tqdm(video_names, desc="Visualizing", unit="video"):
    tracked = render_tracking_video(v, fps=fps)
    render_detection_video(v, fps=fps)

    render_bev_video(v, "heuristic", fps=fps)
    render_bev_video(v, "midas", fps=fps)

    heur_shared, midas_shared = render_combined_bev_pair(v, fps=fps)

    render_camera_relative_trajectory(v, "heuristic", fps=fps)
    render_camera_relative_trajectory(v, "midas", fps=fps)

    combine_videos_grid([tracked, heur_shared, midas_shared], Path(f"outputs/videos/combined/{v}.mp4"))

  print("\nPipeline visualization complete!")


if __name__ == "__main__":
  videos = load_split_file("data/splits/train.txt")
  run_visualization(videos)