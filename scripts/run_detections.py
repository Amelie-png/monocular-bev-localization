from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm

from src.detection.detector import PlayerDetector

# Paths
frame_metadata_dir = Path("data/processed/frame_metadata")
output_dir = Path("data/processed/detections")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize detector
print("Loading YOLOv8 model...")
detector = PlayerDetector(
  model_name="yolov8m.pt",
  confidence_threshold=0.5
)
print("Model loaded")

# Process each video's frames
for metadata_file in frame_metadata_dir.glob("*_metadata.parquet"):
  video_name = metadata_file.stem.replace("_metadata", "")
  print(f"\n{'='*60}")
  print(f"Processing: {video_name}")
  print(f"{'='*60}")
  
  # Load frame metadata
  frames_df = pd.read_parquet(metadata_file)
  
  all_detections = []
  
  # Run detection on each frame
  for idx, row in tqdm(frames_df.iterrows(), total=len(frames_df), desc="Detecting"):
    frame_path = row['frame_path']
    
    # Load image
    image = cv2.imread(frame_path)
    if image is None:
      print(f"Could not load {frame_path}")
      continue
    
    # Detect
    detections = detector.detect(image)

    # Flatten detections
    for detection_idx, det in enumerate(detections):
      bbox = det["bbox"]

      all_detections.append({
        "frame_id": int(row["frame_id"]),
        "frame_path": str(frame_path),

        "detection_id": detection_idx,

        "x1": float(bbox["x1"]),
        "y1": float(bbox["y1"]),
        "x2": float(bbox["x2"]),
        "y2": float(bbox["y2"]),

        "confidence": float(det["confidence"]),

        "class_id": int(det["class_id"]),
        "class_name": str(det["class_name"])
      })
  
  # Save detection results
  detections_df = pd.DataFrame(all_detections)
  if detections_df.empty:
    print("No detections found.")
    continue
  output_file = output_dir / f"{video_name}_detections.parquet"
  detections_df.to_parquet(output_file, index=False)
  
  print(f"Saved detections to: {output_file}")
  
  # Print statistics
  total_frames = len(frames_df)
  frames_with_detections = (detections_df["frame_id"].nunique())
  total_detections = len(detections_df)
  
  print(f"\nStatistics:")
  print(f"  Total frames: {total_frames}")
  print(f"  Frames with detections: {frames_with_detections} ({frames_with_detections/total_frames*100:.1f}%)")
  print(f"  Total detections: {total_detections}")
  print(f"  Avg detections per frame: {total_detections/total_frames:.2f}")

print("\nAll detections complete!")