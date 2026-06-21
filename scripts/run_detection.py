from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm
import yaml

from src.detection import PlayerDetector, DetectionConfig

def load_config(config_path=None):
  """
  Use configs from YAML files or use defaults.
  """
  if config_path and Path(config_path).exists():
    with open(config_path) as f:
      config_dict = yaml.safe_load(f)
    return DetectionConfig(**config_dict)
  return DetectionConfig()

def run_detection(config, output_dir=Path("data/processed/detections"), video_names=None):
  """
  Run detection on video frames.

  Args:
    config: DetectionConfig object
    video_names: List of videos to process, optional (None = all)
  """
  # Paths
  frame_metadata_dir = Path("data/processed/frame_metadata")

  output_dir.mkdir(parents=True, exist_ok=True)

  # Initialize detector
  print(f"Loading YOLOv8 model: {config.model_name}")
  print(f"Config: {config.to_dict()}")
  detector = PlayerDetector(
    model_name=config.model_name,
    confidence_threshold=config.confidence_threshold,
    crop_bottom_ratio=config.crop_bottom_ratio,
  )
  print("Model loaded")

  metadata_files = list(frame_metadata_dir.glob("*_metadata.parquet"))
  if video_names:
    metadata_files = [f for f in metadata_files if f.stem.replace("_metadata", "") in video_names]

  # Process each video's frames
  for metadata_file in metadata_files:
    video_name = metadata_file.stem.replace("_metadata", "")
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")
    
    # Load frame metadata
    frames_df = pd.read_parquet(metadata_file)
    
    # Process in batch
    all_detections = []
    batch_size = config.batch_size

    for batch_start in tqdm(range(0, len(frames_df), batch_size), desc="Processing batches"):
      batch_end = min(batch_start + batch_size, len(frames_df))
      batch_rows = frames_df.iloc[batch_start:batch_end]

      # Load image batch
      images = []
      frame_ids = []
      frame_paths = []

      for _, row in batch_rows.iterrows():
        image = cv2.imread(row["frame_path"])
        if image is None:
          print(f"Could not load {row['frame_path']}")
          continue

        images.append(image)
        frame_ids.append(int(row["frame_id"]))
        frame_paths.append(str(row["frame_path"]))

      if not images:
        continue

      batch_detection = detector.detect_batch(images)

      for frame_id, frame_path, detections in zip(frame_ids, frame_paths, batch_detection):
        for detection_idx, det in enumerate(detections):
          x1, y1, x2, y2 = det["bbox"]
          all_detections.append({
            "frame_id": frame_id,
            "frame_path": frame_path,
            "detection_id": detection_idx,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": det["confidence"],
            "class_id": det["class_id"],
            "class_name": det["class_name"],
          })

    if not all_detections:
      print("No detections found")
      continue
    
    # Save detection results
    detections_df = pd.DataFrame(all_detections)
    output_file = output_dir / f"{video_name}_detections.parquet"
    detections_df.to_parquet(output_file, index=False)
    
    # Print statistics
    total_frames = len(frames_df)
    frames_with_detections = (detections_df["frame_id"].nunique())
    total_detections = len(detections_df)
    
    print(f"\nStatistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames with detections: {frames_with_detections} ({frames_with_detections/total_frames*100:.1f}%)")
    print(f"  Total detections: {total_detections}")
    print(f"  Avg detections per frame: {total_detections/total_frames:.2f}")
    print(f"  Saved to: {output_file}\n")

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
  parser.add_argument("--model", type=str, default=None, help="Model name")
  parser.add_argument("--confidence", type=float, default=None, help="Confidence threshold")
  parser.add_argument("--crop", type=float, default=None, help="Crop bottom ratio")
  parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
  parser.add_argument("--video", type=str, nargs="+", default=None, help="Specific videos to process")
  parser.add_argument("--output", type=str, default=None, help="Path to output directory")
  
  args = parser.parse_args()

  if args.config:
    config = load_config(args.config)
  else:
    config = DetectionConfig()
  
  # Override with command line args if provided
  if args.model is not None:
    config.model_name = args.model
  if args.confidence is not None:
    config.confidence_threshold = args.confidence
  if args.crop is not None:
    config.crop_bottom_ratio = args.crop
  if args.batch_size is not None:
    config.batch_size = args.batch_size

  output_dir = Path(args.output) if args.output else Path("data/processed/detections")
  
  run_detection(config, output_dir=output_dir, video_names=args.video)