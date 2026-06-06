from pathlib import Path
import cv2
import pandas as pd

from src.tracking.visualizer import TrackingVisualizer

# Paths
tracking_dir = Path("data/processed/tracking")
frames_dir = Path("data/processed/frames")
output_dir = Path("outputs/tracking_videos")
output_dir.mkdir(parents=True, exist_ok=True)

sample_dir = Path("outputs/tracking_samples")
sample_dir.mkdir(parents=True, exist_ok=True)

# Initialize visualizer
visualizer = TrackingVisualizer()

# Process each tracking file
for tracking_file in tracking_dir.glob("*_tracking.parquet"): 
  video_name = tracking_file.stem.replace("_tracking", "")
  print(f"\n{'='*60}")
  print(f"Visualizing tracking: {video_name}")
  print(f"{'='*60}")
  
  # Load tracking results
  tracking_df = pd.read_parquet(tracking_file)
  
  # Create annotated video
  output_video = output_dir / f"{video_name}_tracking.mp4"
  visualizer.create_tracking_video(
    trackings_list=tracking_df,
    output_path=output_video,
    fps=30
  )
  
  # Save sample frames
  print("Saving sample frames...")
  sample_indices = [0, len(tracking_df)//4, len(tracking_df)//2, 3*len(tracking_df)//4]
  
  for idx in sample_indices:
      if idx >= len(tracking_df):
          continue
      
      frame_data = tracking_df.iloc[idx]
      frame = cv2.imread(frame_data["frame_path"])
      annotated = visualizer.draw_trackings(frame, frame_data["tracks"])
      
      sample_path = sample_dir / f"{video_name}_frame_{idx:04d}.jpg"
      cv2.imwrite(str(sample_path), annotated)
  
  print(f"Saved {len(sample_indices)} sample frames")

print("\nAll tracking visualizations complete!")
print(f"Videos: {output_dir}")
print(f"Samples: {sample_dir}")