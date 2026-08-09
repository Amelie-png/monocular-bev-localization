from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

from src.bev import BevEstimator, BevConfig
from src.depth import compute_depth_normalization_stats
from src.utils import filter_missing, report_skip

def heuristic_output_path(video_name, output_dir="data/processed/bev/heuristic"):
  return Path(output_dir) / f"{video_name}_estimations.parquet"

def midas_output_path(video_name, output_dir="data/processed/bev/midas"):
  return Path(output_dir) / f"{video_name}_estimations.parquet"

def load_config(config_path=None):
  """
  Use configs from YAML files or use defaults.
  """
  if config_path and Path(config_path).exists():
    with open(config_path) as f:
      config_dict = yaml.safe_load(f)
    return BevConfig(**config_dict)
  return BevConfig()

def run_bev(
    config, 
    output_dir=None, 
    tracking_dir=Path("data/processed/trackings"),
    video_names=None, midas=False, image_width=None, force=False):

  if output_dir is None:
    output_dir = Path("data/processed/bev/midas") if midas else Path("data/processed/bev/heuristic")
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  print("Loading BEV Estimator")
  print(f"Config: {config.to_dict()}")
  estimator = BevEstimator(
    image_width=image_width if image_width else 1920,
    depth_window=config.depth_window,
    fov_scale=config.fov_scale,
  )
  print("Estimator loaded")

  step_label = "bev-midas" if midas else "bev-heuristic"

  tracking_files = list(tracking_dir.glob("*_trackings.parquet"))
  all_names = video_names if video_names else [f.stem.replace("_trackings", "") for f in tracking_files]
  def output_path_fcn(v):
    return (midas_output_path(v, output_dir) if midas else heuristic_output_path(v, output_dir))
  todo = filter_missing(all_names, output_path_fcn, force=force)
  report_skip(f"bev-{step_label}", all_names, todo)
  tracking_files = [f for f in tracking_files if f.stem.replace("_trackings", "") in todo]

  # Process each tracking file
  for tracking_file in tqdm(tracking_files, desc="Videos", unit="video"):
    video_name = tracking_file.stem.replace('_trackings', '')
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")

    depth_dir = Path("data/processed/depths") / video_name
    
    # Load tracking data
    tracking_df = pd.read_parquet(tracking_file)

    all_estimations = []

    depth_min = depth_max = None
    if midas:
      depth_min, depth_max = compute_depth_normalization_stats(
        video_name,
        depth_dir="data/processed/depths",
        low_pct=config.low_pct,
        high_pct=config.high_pct,
      )

    # Process each frame
    frame_ids = sorted(tracking_df["frame_id"].unique())
    for frame_id in tqdm(frame_ids, desc=video_name, unit="frame", leave=False):
      frame_rows = tracking_df[tracking_df["frame_id"] == frame_id]

      if midas:
        depth_path = depth_dir / f"frame_{frame_id:06d}.npy"
        if not depth_path.exists():
          print(f"Missing depth for frame {frame_id}, skipping")
          continue
        depth_map = np.load(depth_path)

      for _, row in frame_rows.iterrows():
        bbox = [row["x1"], row["y1"], row["x2"], row["y2"]]
        if midas:
          bev = estimator.estimate(bbox, depth_map=depth_map, depth_min=depth_min, depth_max=depth_max)
        else:
          bev = estimator.estimate(bbox)
        all_estimations.append({
          "video_name": video_name,
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
    print(f"Saved estimations to: {output_file}")

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

  print("\nAll estimations complete!")

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
  parser.add_argument("--midas", action="store_true", default=None, help="Use MiDaS depth")
  parser.add_argument("--depth_window", type=int, default=None, help="Depth window")
  parser.add_argument("--image_width", type=int, default=None, help="Image width for BEV estimator")
  parser.add_argument("--video", type=str, nargs="+", default=None, help="Specific videos to process")
  parser.add_argument("--output", type=str, default=None, help="Path to output directory")
  parser.add_argument("--tracking", type=str, default="data/processed/trackings", help="Path to tracking directory")
  parser.add_argument("--force", action="store_true", default=None, help="Force BEV estimation")
  
  args = parser.parse_args()

  config = load_config(args.config) if args.config else BevConfig()

  # Override with command line args if provided
  if args.depth_window is not None:
    config.depth_window = args.depth_window
  
  run_bev(
    config=config, 
    output_dir=args.output, 
    tracking_dir=Path(args.tracking),
    video_names=args.video, 
    midas=args.midas, 
    image_width=args.image_width, 
    force=args.force)
