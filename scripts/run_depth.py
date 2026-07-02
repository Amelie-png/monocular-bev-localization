from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

from src.depth import MiDaSEstimator

def run_depth(output_dir=Path("data/processed/depths"), video_names=None):
  # Paths
  tracking_dir = Path("data/processed/trackings")
  output_dir.mkdir(parents=True, exist_ok=True)

  vis_dir = Path("outputs/depths/visualizations")
  vis_dir.mkdir(parents=True, exist_ok=True)

  estimator = MiDaSEstimator()

  tracking_files = list(tracking_dir.glob("*_trackings.parquet"))
  if video_names:
    tracking_files = [f for f in tracking_files if f.stem.replace("_trackings", "") in video_names]

  for tracking_file in tracking_files:
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
    for idx, frame_id in enumerate(tqdm(frame_ids, desc=f"{video_name}", unit="frame", leave=False)):
      frame_rows = tracking_df[tracking_df["frame_id"] == frame_id]

      depth_file = video_depth_dir / f"frame_{frame_id:06d}.npy"

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
    
    print(f"Saved depths to: {depth_file}")
    print(f"Saved sample depth visualizations to {vis_dir}")

  print("\nAll depths complete!")

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
  parser.add_argument("--max-age", type=str, default=None, help="Max age")
  parser.add_argument("--n-init", type=float, default=None, help="N init")
  parser.add_argument("--nn-budget", type=float, default=None, help="NN budget")
  parser.add_argument("--video", type=str, nargs="+", default=None, help="Specific videos to process")
  parser.add_argument("--output", type=str, default=None, help="Path to output directory")
  
  args = parser.parse_args()

  """
  if args.config:
    config = load_config(args.config)
  else:
    config = TrackConfig()

  # Override with command line args if provided
  if args.max_age is not None:
    config.max_age = args.max_age
  if args.n_init is not None:
    config.n_init = args.n_init
  if args.nn_budget is not None:
    config.nn_budget = args.nn_budget
  """

  output_dir = Path(args.output) if args.output else Path("data/processed/depths")
  
  run_depth(output_dir=output_dir, video_names=args.video)

  