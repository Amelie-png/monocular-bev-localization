from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm

from src.detection.detector import PlayerDetector
from src.tracking.tracker import PlayerTracker
from src.tracking.visualizer import TrackingVisualizer

# Paths
frame_metadata_dir = Path("data/processed/frame_metadata")
output_dir = Path("data/processed/tracking")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize models
print("Loading YOLOv8 detector...")
detector = PlayerDetector(model_name='yolov8m.pt', confidence_threshold=0.5)

print("Initializing DeepSORT tracker...")
tracker = PlayerTracker(max_age=30, n_init=3, nn_budget=100)

visualizer = TrackingVisualizer()

# Process each video
for metadata_file in frame_metadata_dir.glob("*_metadata.parquet"):
  video_name = metadata_file.stem.replace('_metadata', '')
  print(f"\n{'='*60}")
  print(f"Processing: {video_name}")
  print(f"{'='*60}")
  
  # Load frame metadata
  frames_df = pd.read_parquet(metadata_file)
  
  # Reset tracker for new video
  tracker.reset()
  
  all_tracks = []
  frame_list = []
  
  # Process each frame
  for idx, row in tqdm(frames_df.iterrows(), total=len(frames_df), desc="Tracking"):
    frame_path = row['frame_path']
    frame_id = row['frame_id']
    
    # Load image
    image = cv2.imread(frame_path)
    if image is None:
      print(f"Could not load {frame_path}")
      continue
    
    # Detect with YOLOv8
    detections = detector.detect(image)
    
    # Track with DeepSORT
    tracked = tracker.update(frame=image, detections=detections)
    
    # Store results
    all_tracks.append({
      'frame_id': frame_id,
      'frame_path': frame_path,
      'timestamp': row['timestamp'],
      'num_tracks': len(tracked),
      'tracks': tracked
    })
    
    frame_list.append(frame_path)
  
  # Save tracking results
  tracks_df = pd.DataFrame(all_tracks)
  if tracks_df.empty:
    print("No tracks found.")
    continue
  output_file = output_dir / f"{video_name}_tracking.parquet"
  tracks_df.to_parquet(output_file, index=False)
  
  print(f"Saved tracks to: {output_file}")

  # Print statistics
  total_frames = len(all_tracks)
  frames_with_tracks = sum(1 for t in all_tracks if t['num_tracks'] > 0)
  total_track_instances = sum(t['num_tracks'] for t in all_tracks)
  unique_track_ids = set()
  
  for t in all_tracks:
    for track in t['tracks']:
      unique_track_ids.add(track['track_id'])
  
  print(f"\nTracking Statistics:")
  print(f"  Total frames: {total_frames}")
  print(f"  Frames with tracks: {frames_with_tracks} ({frames_with_tracks/total_frames*100:.1f}%)")
  print(f"  Total track detections: {total_track_instances}")
  print(f"  Unique track IDs: {len(unique_track_ids)}")
  print(f"  Avg tracks per frame: {total_track_instances/total_frames:.2f}")

print("\nAll tracking complete!")