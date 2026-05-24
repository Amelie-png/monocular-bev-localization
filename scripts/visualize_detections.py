from pathlib import Path
import pandas as pd
import cv2

from src.detection.visualizer import DetectionVisualizer

# Paths
detections_dir = Path("data/processed/detections")
output_dir = Path("outputs/detection_videos")
output_dir.mkdir(parents=True, exist_ok=True)

# Also save sample images
sample_dir = Path("outputs/detection_samples")
sample_dir.mkdir(parents=True, exist_ok=True)

# Initialize visualizer
visualizer = DetectionVisualizer(color=(0, 255, 0), thickness=2)

# Process each detection file
for detection_file in detections_dir.glob("*_detections.parquet"): 
  video_name = detection_file.stem.replace('_detections', '')
  print(f"\n{'='*60}")
  print(f"Visualizing: {video_name}")
  print(f"{'='*60}")
  
  # Load detections
  detections_df = pd.read_parquet(detection_file)
  
  # Prepare data for video creation
  frame_paths = []
  detections_list = []
  
  for frame_id, frame_group in (detections_df.groupby("frame_id")):
    frame_path = (frame_group.iloc[0]["frame_path"])

    detections=[]

    for _, row in (frame_group.iterrows()):
      detections.append({
        "bbox":{
          "x1":row["x1"],
          "y1":row["y1"],
          "x2":row["x2"],
          "y2":row["y2"]
        },

        "confidence": row["confidence"],

        "class_id": row["class_id"],

        "class_name": row["class_name"]
      })

    frame_paths.append(frame_path)

    detections_list.append(detections)
  
  # Create annotated video
  output_video = output_dir / f"{video_name}_detections.mp4"
  visualizer.create_detection_video(
    frame_paths=frame_paths,
    detections_list=detections_list,
    output_path=output_video,
    fps=30
  )
  
  # Also save some sample frames
  print("Saving sample frames...")
  sample_indices = [0, len(frame_paths)//4, len(frame_paths)//2, 3*len(frame_paths)//4]
  
  for idx in sample_indices:
    if idx >= len(frame_paths):
      continue
        
    frame = cv2.imread(frame_paths[idx])
    annotated = visualizer.draw_detections(frame, detections_list[idx])
    
    sample_path = sample_dir / f"{video_name}_frame_{idx:04d}.jpg"
    cv2.imwrite(str(sample_path), annotated)
  
  print(f"Saved {len(sample_indices)} sample frames to {sample_dir}")

print("\nAll visualizations complete!")
print(f"Check {output_dir} for videos")
print(f"Check {sample_dir} for sample images")