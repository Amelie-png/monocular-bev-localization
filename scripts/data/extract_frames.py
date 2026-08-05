from pathlib import Path
from tqdm import tqdm

from src.data.video_parser import extract_frames
from src.utils import filter_missing, report_skip

def frames_output_path(video_name):
  return [
    Path(f"data/processed/frames/{video_name}/.done"),
    Path(f"data/processed/frame_metadata/{video_name}_metadata.parquet"),
  ]

def run_frame_extraction(fps, video_names, force=False):
  video_dir = Path("data/raw/videos")
  output_dir = Path("data/processed/frames")
  output_dir.mkdir(parents=True, exist_ok=True)
  metadata_dir = Path("data/processed/frame_metadata")
  metadata_dir.mkdir(parents=True, exist_ok=True)

  # Filter videos to process for resumability
  video_files = list(video_dir.glob("*.mp4"))
  all_names = video_names if video_names else [f.stem for f in video_files]

  missing_files = set(all_names) - {f.stem for f in video_files}
  if missing_files:
    print(f"WARNING: requested video(s) not found in {video_dir}: {sorted(missing_files)}")

  todo = filter_missing(all_names, frames_output_path, force=force)
  report_skip("frame extraction", all_names, todo)
  video_files = [f for f in video_files if f.stem in todo]

  for video_path in tqdm(video_files, desc="Videos", unit="video"):
    frame_metadata = extract_frames(
      video_path=video_path,
      output_dir=output_dir,
      fps_override=fps  # Set to 30 for 30fps, None for native
    )
    
    video_name = video_path.stem
    metadata_file = metadata_dir / f"{video_name}_metadata.parquet"
    frame_metadata.to_parquet(metadata_file, index=False)
    print(f"Saved metadata: {metadata_file}\n")

  print(f"All frames extracted!")

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--fps", type=int, default=None, help="Video parsing fps")
  parser.add_argument("--video", type=str, nargs="+", default=None, help="Specific videos to process")
  parser.add_argument("--force", action="store_true", default=None, help="Force extraction")
  
  args = parser.parse_args()
  
  run_frame_extraction(fps=args.fps, video_names=args.video, force=args.force)