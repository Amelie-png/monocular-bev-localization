import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils import mark_done

def extract_frames(video_path, output_dir, fps_override=None):
  """
  Extract frames from video file.
  
  Args:
    video_path: Path to video file
    output_dir: Directory to save frames
    fps_override: If set, sample at this fps instead of native video fps
  
  Return:
    DataFrame with frame_id, timestamp, frame_path
  """
  video_path = Path(video_path)
  output_dir = Path(output_dir)

  video_name = video_path.stem
  frame_dir = output_dir / video_name
  frame_dir.mkdir(parents=True, exist_ok=True)

  cap = cv2.VideoCapture(str(video_path))

  if not cap.isOpened():
    raise ValueError(f"Could not open video: {video_path}")

  metadata = []
  frame_id = 0

  native_fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  print(f"Processing {video_path.name}")
  print(f"  Native FPS: {native_fps:.2f}")
  print(f"  Total frames: {total_frames}")

  target_fps = fps_override if fps_override is not None else native_fps

  sample_interval = 1.0 / target_fps
  next_sample_time = 0.0

  actual_frame_count = 0

  with tqdm(total=total_frames, desc=video_name, unit="frame", leave=False) as pbar:
    while True:
      success, frame=cap.read()

      if not success:
        break

      timestamp = frame_id / native_fps
      if timestamp >= next_sample_time:
        frame_filename = f"frame_{actual_frame_count:06d}.png"
        frame_path = frame_dir / frame_filename
        
        # Save frame
        cv2.imwrite(str(frame_path), frame)
        
        metadata.append({
          "frame_id": actual_frame_count,
          "original_frame_id": frame_id,
          "timestamp": timestamp,
          "frame_path": str(frame_path),
          "video_name": video_name
        })
        
        actual_frame_count += 1
        next_sample_time += sample_interval
      
      frame_id += 1
      pbar.update(1)

  cap.release()

  mark_done(frame_dir / ".done")

  print(f"  Extracted {actual_frame_count} frames to {frame_dir}")
    
  return pd.DataFrame(metadata)