from pathlib import Path
from src.data.video_parser import extract_frames

video_dir = Path("data/raw/videos")

for video_path in video_dir.glob("*.mp4"):
  extract_frames(
    video_path=video_path
  )