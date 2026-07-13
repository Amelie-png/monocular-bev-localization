from pathlib import Path
import pandas as pd

from src.detection import DetectionVisualizer
from src.evaluation import compute_tracking_metrics
from src.utils import load_split_file

def evaluate_trackings(
    video_names, 
    output_video=True, 
    trackings_dir=Path("data/processed/trackings"),
    output_dir=Path("outputs/tracking_videos"),
    metrics_dir=Path("outputs/tracking_metrics"),):
  """
  Evaluate tracking: compute metrics and create visualizations.
    
  Args:
    video_names: List of video names to evaluate
    output_video: Whether to create MP4 videos
  """
  # Paths
  frame_metadata_dir = Path("data/processed/frame_metadata")
  output_dir.mkdir(parents=True, exist_ok=True)
  metrics_dir.mkdir(parents=True, exist_ok=True)

  visualizer = DetectionVisualizer(color=(0, 255, 0), thickness=2)

  all_metrics = []

  for video_name in video_names:
    tracking_file = trackings_dir / f"{video_name}_trackings.parquet"
    metadata_file = frame_metadata_dir / f"{video_name}_metadata.parquet"

    if not tracking_file.exists():
      print(f"Tracking file not found: {tracking_file}")
      continue

    print(f"\n{'='*60}")
    print(f"Evaluating: {video_name}")
    print(f"{'='*60}")
  
    # Load data
    tracking_df = pd.read_parquet(tracking_file)
    metadata_df = pd.read_parquet(metadata_file)

    total_frames = len(metadata_df)
    metrics = compute_tracking_metrics(tracking_df, frame_count=total_frames)
    metrics['video_name'] = video_name
    all_metrics.append(metrics)

    metrics_file = metrics_dir / f"{video_name}_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

    if not output_video:
      continue

    # Prepare frame data
    frame_data = []

    for frame_id in sorted(metadata_df['frame_id'].unique()):
      frame_info = metadata_df[metadata_df['frame_id'] == frame_id].iloc[0]
      frame_trackings = tracking_df[tracking_df['frame_id'] == frame_id]
      
      trackings=[]
      for _, row in frame_trackings.iterrows():
        trackings.append({
          "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
          "confidence": row["confidence"],
          "class_name": "person",
          "track_id": row["track_id"]
        })

      frame_data.append({
        'frame_path': frame_info['frame_path'],
        'frame_id': frame_id,
        'trackings': trackings
      })

    # Create video
    if output_video:
      print("Creating video...")
      output_video_path = output_dir / f"{video_name}_trackings.mp4"
      
      frame_paths = [f['frame_path'] for f in frame_data]
      trackings_list = [f['trackings'] for f in frame_data]
      
      visualizer.create_detection_video(
        frame_paths=frame_paths,
        detections_list=trackings_list,
        output_path=output_video_path,
        fps=30
      )

  # Save metrics
  if all_metrics:
    metrics_df = pd.DataFrame(all_metrics)
    all_metrics_file = metrics_dir / "all_metrics.csv"
    metrics_df.to_csv(all_metrics_file, index=False)
    print(f"\nAll metrics saved to: {all_metrics_file}")
    return metrics_df
  
  return pd.DataFrame()

if __name__ == "__main__":
  import argparse
    
  parser = argparse.ArgumentParser()
  parser.add_argument("--split", type=str, default="data/splits/train.txt", help="Split file")
  parser.add_argument("--videos", type=str, nargs="+", default=None, help="Specific videos to evaluate")
  parser.add_argument("--no-video", action="store_true", help="Skip video creation")
  parser.add_argument("--trackings-dir", type=str, default="data/processed/trackings")
  parser.add_argument("--output-dir", type=str, default="outputs/tracking_videos")
  parser.add_argument("--metrics-dir", type=str, default="outputs/tracking_metrics")
  
  args = parser.parse_args()

  if args.videos:
    video_names = args.videos
  else:
    video_names = load_split_file(args.split)
  
  evaluate_trackings(
    video_names=video_names,
    output_video=not args.no_video,
    trackings_dir=Path(args.trackings_dir),
    output_dir=Path(args.output_dir),
    metrics_dir=Path(args.metrics_dir),
  )
  