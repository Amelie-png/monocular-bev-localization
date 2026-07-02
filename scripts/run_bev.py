from pathlib import Path
import pandas as pd
import numpy as np

from src.bev import BEVEstimator, BevVisualizer, BevConfig

def run_bev(output_dir=Path("data/processed/bev/heuristic"), video_names=None, midas=False):
  # Paths
  tracking_dir = Path("data/processed/trackings")

  output_dir.mkdir(parents=True, exist_ok=True)

  estimator = BEVEstimator() # pass in image width when using map as background

  tracking_files = list(tracking_dir.glob("*_trackings.parquet"))
  if video_names:
    tracking_files = [f for f in tracking_files if f.stem.replace("_trackings", "") in video_names]

  # Process each tracking file
  for tracking_file in tracking_files:
    video_name = tracking_file.stem.replace('_trackings', '')
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")

    depth_dir = Path("outputs/depths") / video_name # TODO Modify later
    
    # Load tracking data
    tracking_df = pd.read_parquet(tracking_file)

    all_estimations = []

    # Process each frame
    for frame_id in sorted(tracking_df["frame_id"].unique()):
      frame_rows = tracking_df[tracking_df["frame_id"] == frame_id]

      if midas:
        depth_map = np.load(depth_dir / f"frame_{frame_id:06d}.npy")

      for _, row in frame_rows.iterrows():
        bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]
        if midas:
          bev = estimator.estimate(bbox, depth_map=depth_map)
        else:
          bev = estimator.estimate(bbox)
        all_estimations.append({
          "frame_id": frame_id,
          "frame_path": row["frame_path"],
          "track_id": row["track_id"],
          "bev_x": bev["bev_x"],
          "bev_y": bev["bev_y"],
          "depth": bev["depth"]
        })
    
    # Save tracking results
    estimation_df = pd.DataFrame(all_estimations)
    if estimation_df.empty:
      print("No estimation found.")
      continue
    output_file = output_dir / f"{video_name}_estimations.parquet"
    estimation_df.to_parquet(output_file, index=False)
    
    print(f"Saved tracks to: {output_file}")

    # Print statistics
    total_frames = tracking_df["frame_id"].nunique()
    unique_estimation_ids = estimation_df["track_id"].nunique()
    total_estimation_instances = len(estimation_df)
    frames_with_estimation = estimation_df["frame_id"].nunique()
    
    print(f"\nEstimation Statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with estimations: {frames_with_estimation} ({frames_with_estimation/total_frames*100:.1f}%)")
    print(f"  Total estimations: {total_estimation_instances}")
    print(f"  Unique estimation IDs: {unique_estimation_ids}")
    print(f"  Avg estimations per frame: {total_estimation_instances/total_frames:.2f}")

    print(f"\n{'='*60}")
    print(f"Visualizing: {video_name}")
    print(f"{'='*60}")

    visualizer = BevVisualizer(scale=3)

    if midas:
      visualizer = BevVisualizer()

    frame_estimations = []

    for frame_id in sorted(estimation_df["frame_id"].unique()):
      frame_rows = estimation_df[estimation_df["frame_id"] == frame_id]

      estimations = []

      for _, row in frame_rows.iterrows():
        estimations.append({
          "bev_x": row["bev_x"],
          "bev_y": row["bev_y"],
          "track_id": row["track_id"]
        })

      frame_estimations.append(estimations)

    video_file = (
      output_dir /
      f"{video_name}_bev.mp4"
    )

    visualizer.create_bev_video(
      frame_estimations,
      video_file,
      fps=30
    )

  print("\nAll estimations complete!")

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
  parser.add_argument("--midas", action="store_true", default=None, help="Use MiDaS depth")
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

  if args.output:
    output_dir = Path(args.output)
  elif args.midas:
    output_dir = Path("data/processed/bev/midas")
  else:
    output_dir = Path("data/processed/bev/heuristic")
  
  run_bev(output_dir=output_dir, video_names=args.video, midas=args.midas)
