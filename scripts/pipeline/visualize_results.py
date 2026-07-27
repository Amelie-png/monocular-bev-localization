from pathlib import Path
import pandas as pd
import subprocess
from tqdm import tqdm

from src.bev import BevVisualizer
from src.evaluation.calibrate_transform import make_transform

def render_bev_video(video_name, variant, transform_fn=None, minimap=False, minimap_bounds=None):
  """
  If transform_fn/minimap given, project bev_x/bev_y using the same world
  transform from evaluation and draw on a real minimap image instead of
  a camera-relative canvas. Otherwise falls back to camera-relative view.
  """
  bev_dir = Path(f"data/processed/bev/{variant}")
  df = pd.read_parquet(bev_dir / f"{video_name}_estimations.parquet")
  match_number = video_name.split('_')[1]

  if minimap and transform_fn is not None:
    sync_df = pd.read_parquet(f"data/processed/sync/{video_name}_sync.parquet")
    df = df.merge(sync_df[["frame_id", "cam_x", "cam_y", "yaw_deg"]], on="frame_id", how="inner")
    wx, wy = transform_fn(df["bev_x"].values, df["bev_y"].values, df["cam_x"].values, df["cam_y"].values, df["yaw_deg"].values)
    df = df.assign(plot_x=wx, plot_y=wy)
    visualizer = BevVisualizer(background=f"data/minimap/{match_number}.png", bounds=minimap_bounds)
  else:
    df = df.assign(plot_x=df["bev_x"], plot_y=df["bev_y"])
    visualizer = BevVisualizer(scale=3)

  frame_estimations = []
  for frame_id in sorted(df["frame_id"].unique()):
    g = df[df["frame_id"] == frame_id]
    frame_estimations.append([
      {"bev_x": row["plot_x"], "bev_y": row["plot_y"], "track_id": row["track_id"]}
      for _, row in g.iterrows()
    ])

  out_path = Path(f"outputs/viz/{video_name}_{variant}_bev.mp4")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  visualizer.create_bev_video(frame_estimations, out_path, fps=30)
  return out_path


def combine_videos_grid(video_paths, out_path, layout="hstack"):
  inputs = []
  for p in video_paths:
    inputs += ["-i", str(p)]
  n = len(video_paths)
  filter_complex = f"{''.join(f'[{i}:v]' for i in range(n))}{layout}=inputs={n}[v]"
  cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex,
         "-map", "[v]", "-c:v", "libx264", "-crf", "18", str(out_path)]
  subprocess.run(cmd, check=True)


if __name__ == "__main__":
  from src.utils import load_split_file
  videos = load_split_file("data/splits/train.txt")

  # TODO: hardcode calibrated convention from evaluation
  transform_fn = make_transform("fwd_cos_sin_negx", scale=1.0)  # scale unused

  for v in tqdm(videos, desc="Rendering"):
    heur = render_bev_video(v, "heuristic")
    midas = render_bev_video(v, "midas")
    tracked = Path(f"outputs/viz/{v}_tracked.mp4")

    combine_videos_grid(
      [tracked, heur, midas],
      Path(f"outputs/viz/{v}_combined.mp4"),
      layout="hstack",
    )