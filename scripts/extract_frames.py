from pathlib import Path
from src.data.video_parser import extract_frames

video_dir = Path("data/raw/videos")
output_dir = Path("data/processed/frames")
output_dir.mkdir(parents=True, exist_ok=True)
metadata_dir = Path("data/processed/frame_metadata")
metadata_dir.mkdir(parents=True, exist_ok=True)

for video_path in sorted(video_dir.glob("*.mp4")):
  frame_metadata = extract_frames(
    video_path=video_path,
    output_dir=output_dir,
    fps_override=30  # Set to 30 for 30fps, None for native
  )
  
  video_name = video_path.stem
  metadata_file = metadata_dir / f"{video_name}_metadata.parquet"
  frame_metadata.to_parquet(metadata_file, index=False)
  print(f"Saved metadata: {metadata_file}\n")

print(f"All frames extracted!")