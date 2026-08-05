from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

from src.depth import MiDaSEstimator
from src.utils import filter_missing, report_skip, mark_done

def depth_output_path(video_name):
  return Path(f"data/processed/depths/{video_name}/.done")

def run_depth(output_dir=Path("data/processed/depths"), video_names=None, force=False):
  # Paths
  tracking_dir = Path("data/processed/trackings")
  output_dir.mkdir(parents=True, exist_ok=True)

  vis_dir = Path("outputs/depths/visualizations")
  vis_dir.mkdir(parents=True, exist_ok=True)

  print("Loading MiDaS Estimator")
  estimator = MiDaSEstimator()
  print("Estimator loaded")

  tracking_files = list(tracking_dir.glob("*_trackings.parquet"))
  all_names = video_names if video_names else [f.stem.replace("_trackings", "") for f in tracking_files]
  todo = filter_missing(all_names, depth_output_path, force=force)
  report_skip("depth", all_names, todo)
  tracking_files = [f for f in tracking_files if f.stem.replace("_trackings", "") in todo]

  for tracking_file in tqdm(tracking_files, desc="Videos", unit="video"):
    video_name = tracking_file.stem.replace('_trackings', '')
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")

    video_depth_dir = output_dir / video_name
    video_depth_dir.mkdir(exist_ok=True)
    
    # Load tracking data
    tracking_df = pd.read_parquet(tracking_file)

    # Get sample ids
    frame_ids = sorted(tracking_df["frame_id"].unique())
    sample_ids = set(
      np.linspace(
        0,
        len(frame_ids) - 1,
        min(5, len(frame_ids)),
        dtype=int
      )
    )

    # Process each frame
    depth_file = None
    for idx, frame_id in enumerate(tqdm(frame_ids, desc=video_name, unit="frame", leave=False)):
      depth_file = video_depth_dir / f"frame_{frame_id:06d}.npy"
      vis_file = vis_dir / f"{video_name}_frame_{frame_id:05d}.png"
      need_depth = force or not depth_file.exists()
      need_vis = (idx in sample_ids) and (force or not vis_file.exists())

      # Per-frame resume: skip frames already computed, even if the video-level .done marker isn't set yet (e.g. crashed mid-video)
      if not need_depth and not need_vis:
        continue

      frame_rows = tracking_df[tracking_df["frame_id"] == frame_id]
      frame_path = frame_rows.iloc[0]["frame_path"]

      image = cv2.imread(frame_path)
      if image is None:
        print(f"Could not load {frame_path}")
        continue

      depth = estimator.estimate(image)

      if idx in sample_ids:
        colored = estimator.visualize(depth)

        comparison = np.hstack([
          image,
          colored
        ])

        cv2.imwrite(
          str(vis_dir / f"{video_name}_frame_{frame_id:05d}.png"),
          comparison
        )

      np.save(depth_file, depth)
    
    mark_done(video_depth_dir / ".done")
    print(f"Saved depths to: {video_depth_dir}")
    print(f"Saved sample depth visualizations to {vis_dir}")

  print("\nAll depths complete!")

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--video", type=str, nargs="+", default=None, help="Specific videos to process")
  parser.add_argument("--output", type=str, default=None, help="Path to output directory")
  parser.add_argument("--force", action="store_true", default=None, help="Force depths")
  
  args = parser.parse_args()

  output_dir = Path(args.output) if args.output else Path("data/processed/depths")
  
  run_depth(output_dir=output_dir, video_names=args.video, force=args.force)

  