from pathlib import Path

from scripts.no_track.make_pseudo_tracks import make_pseudo_tracks
from scripts.bev.run_bev import run_bev
from scripts.utils.config_loader import load_pipeline_config
from src.utils import load_split_file

def run_notrack_pipeline(config, videos, force=False):
  make_pseudo_tracks(videos)
  run_bev(
    config=config.heuristic_bev,
    video_names=videos,
    midas=False,
    force=force,
    tracking_dir=Path("data/processed/pseudo_tracks"),
    output_dir=Path("data/processed/bev/heuristic_notrack"),
  )

if __name__ == "__main__":
  config = load_pipeline_config("configs")
  videos = load_split_file("data/splits/train.txt")
  run_notrack_pipeline(config, videos)