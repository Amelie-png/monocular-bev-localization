from pathlib import Path
import pandas as pd
import cv2
import yaml
from tqdm import tqdm

from src.tracking import PlayerTracker, TrackConfig

def load_config(config_path=None):
  """
  Use configs from YAML files or use defaults.
  """
  if config_path and Path(config_path).exists():
    with open(config_path) as f:
      config_dict = yaml.safe_load(f)
    return TrackConfig(**config_dict)
  return TrackConfig()

def run_tracking(config, output_dir=Path("data/processed/trackings"), video_names=None):
  # Paths
  detections_dir = Path("data/processed/detections")

  output_dir.mkdir(parents=True, exist_ok=True)

  # Initialize model
  print("Initializing DeepSORT tracker...")
  print(f"Config: {config.to_dict()}")
  tracker = PlayerTracker(config)
  print("Tracker loaded")

  detection_files = list(detections_dir.glob("*_detections.parquet"))
  if video_names:
    detection_files = [f for f in detection_files if f.stem.replace("_detections", "") in video_names]

  # Process each detection file
  for detection_file in detection_files:
    video_name = detection_file.stem.replace('_detections', '')
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Load detection data
    detection_df = pd.read_parquet(detection_file)

    all_tracks = []

    # Process each frame
    for frame_id in sorted(detection_df["frame_id"].unique()):
      frame_rows = detection_df[detection_df["frame_id"] == frame_id]

      frame_path = frame_rows.iloc[0]["frame_path"]

      # Load image
      image = cv2.imread(frame_path)
      if image is None:
        print(f"Could not load {frame_path}")
        continue

      detections = []

      for _, row in frame_rows.iterrows():
        detections.append({
          "bbox": [
            row["x1"],
            row["y1"],
            row["x2"],
            row["y2"]
          ],
          "confidence": row["confidence"]
        })

      # Track with DeepSORT
      tracks = tracker.update(frame=image, detections=detections)

      # Store results
      for track in tracks:
        all_tracks.append({
          "frame_id": frame_id,
          "frame_path": frame_path,

          "track_id": track["track_id"],

          "x1": track["bbox"][0],
          "y1": track["bbox"][1],
          "x2": track["bbox"][2],
          "y2": track["bbox"][3],

          "confidence": track["confidence"]
        })
    
    # Reset tracker for new video
    tracker.reset()
    
    # Save tracking results
    tracks_df = pd.DataFrame(all_tracks)
    if tracks_df.empty:
      print("No tracks found.")
      continue
    output_file = output_dir / f"{video_name}_trackings.parquet"
    tracks_df.to_parquet(output_file, index=False)
    
    print(f"Saved tracks to: {output_file}")

    # Print statistics
    total_frames = detection_df["frame_id"].nunique()
    tracks_df = pd.DataFrame(all_tracks)
    unique_track_ids = tracks_df["track_id"].nunique()
    total_track_instances = len(tracks_df)
    frames_with_tracks = tracks_df["frame_id"].nunique()
    
    print(f"\nTracking Statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with tracks: {frames_with_tracks} ({frames_with_tracks/total_frames*100:.1f}%)")
    print(f"  Total tracks: {total_track_instances}")
    print(f"  Unique track IDs: {unique_track_ids}")
    print(f"  Avg tracks per frame: {total_track_instances/total_frames:.2f}")

  print("\nAll tracking complete!")

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

  output_dir = Path(args.output) if args.output else Path("data/processed/trackings")
  
  run_tracking(config, output_dir=output_dir, video_names=args.video)