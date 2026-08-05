from pathlib import Path
import pandas as pd
import cv2
import yaml
from tqdm import tqdm

from src.tracking import PlayerTracker, ByteTrackConfig
from src.utils import filter_missing, report_skip

def tracking_output_path(video_name):
  return Path(f"data/processed/trackings/{video_name}_trackings.parquet")

def load_config(config_path=None):
  """
  Use configs from YAML files or use defaults.
  """
  if config_path and Path(config_path).exists():
    with open(config_path) as f:
      config_dict = yaml.safe_load(f)
    return ByteTrackConfig(**config_dict)
  return ByteTrackConfig()

def run_tracking(config, output_dir=Path("data/processed/trackings"), video_names=None, force=False):
  # Paths
  detections_dir = Path("data/processed/detections")

  output_dir.mkdir(parents=True, exist_ok=True)

  # Initialize model
  print("Initializing ByteTracker...")
  print(f"Config: {config.to_dict()}")
  tracker = PlayerTracker(config)
  print("Tracker loaded")

  detection_files = list(detections_dir.glob("*_detections.parquet"))
  all_names = video_names if video_names else [f.stem.replace("_detections", "") for f in detection_files]
  todo = filter_missing(all_names, tracking_output_path, force=force)
  report_skip("tracking", all_names, todo)
  detection_files = [f for f in detection_files if f.stem.replace("_detections", "") in todo]

  # Process each detection file
  for detection_file in tqdm(detection_files, desc="Videos", unit="video"):
    video_name = detection_file.stem.replace('_detections', '')
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Load detection data
    detection_df = pd.read_parquet(detection_file)

    all_tracks = []

    # Process each frame
    frame_ids = sorted(detection_df["frame_id"].unique())
    for frame_id in tqdm(frame_ids, desc=video_name, unit="frame", leave=False):
      frame_rows = detection_df[detection_df["frame_id"] == frame_id]
      frame_path = frame_rows.iloc[0]["frame_path"]

      # Check existence of image
      if not Path(frame_path).exists():
        print(f"Missing frame file: {frame_path}")
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

      # Track with ByteTrack
      tracks = tracker.update(detections=detections)

      # Store results
      for track in tracks:
        all_tracks.append({
          "video_name": video_name,
          "frame_id": frame_id,
          "frame_path": frame_path,

          "track_id": track["track_id"],

          "x1": track["bbox"][0],
          "y1": track["bbox"][1],
          "x2": track["bbox"][2],
          "y2": track["bbox"][3],

          "confidence": track["confidence"],
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
  parser.add_argument("--activation", type=float, default=None, help="track activation threshold")
  parser.add_argument("--lost_buffer", type=int, default=None, help="lost track buffer")
  parser.add_argument("--consecutive", type=int, default=None, help="minimum consecutive frames")
  parser.add_argument("--video", type=str, nargs="+", default=None, help="Specific videos to process")
  parser.add_argument("--output", type=str, default=None, help="Path to output directory")
  parser.add_argument("--force", action="store_true", default=None, help="Force tracking")

  args = parser.parse_args()

  config = load_config(args.config) if args.config else ByteTrackConfig()

  # Override with command line args if provided
  if args.activation is not None:
    config.track_activation_threshold = args.activation
  if args.lost_buffer is not None:
    config.lost_track_buffer = args.lost_buffer
  if args.consecutive is not None:
    config.minimum_consecutive_frames = args.consecutive

  output_dir = Path(args.output) if args.output else Path("data/processed/trackings")
  
  run_tracking(config, output_dir=output_dir, video_names=args.video, force=args.force)