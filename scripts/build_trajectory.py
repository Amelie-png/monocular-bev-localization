from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.utils import filter_missing, report_skip, load_split_file

def trajectory_output_path(video_name, variant):
  return Path(f"data/processed/trajectories/{variant}/{video_name}_trajectories.parquet")

def build_trajectory(video_names, variant="heuristic", force=False):
  bev_dir = Path(f"data/processed/bev/{variant}")
  out_dir = Path(f"data/processed/trajectories/{variant}")
  out_dir.mkdir(parents=True, exist_ok=True)

  def out_path(v):
    return trajectory_output_path(v, variant)

  todo = filter_missing(video_names, out_path, force=force)
  report_skip(f"trajectories-{variant}", video_names, todo)

  for video_name in tqdm(todo, desc=f"Trajectories ({variant})", unit="video"):
    bev_file = bev_dir / f"{video_name}_estimations.parquet"
    if not bev_file.exists():
      print(f"Missing bev estimations for {video_name}, skipping")
      continue

    df = pd.read_parquet(bev_file)
    traj_df = df.sort_values(["track_id", "frame_id"]).reset_index(drop=True)
    traj_df.to_parquet(out_path(video_name), index=False)

if __name__ == "__main__":
  videos = load_split_file("data/splits/train.txt")
  build_trajectory(videos, variant="heuristic")
  build_trajectory(videos, variant="midas")