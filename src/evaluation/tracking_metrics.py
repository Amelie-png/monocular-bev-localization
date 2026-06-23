import numpy as np

def compute_tracking_metrics(df, frame_count=None):
  """
  Compute tracking metrics given DataFrame file.

  Args:
    df: dataframe file of one single round
    frame_count: Total frames in video (if None, use unique frame_ids)
  
  Return:
    Dict of metrics
  """
  if df.empty:
    return {
      "total_frames": frame_count or 0,
      "total_tracks": 0,
      "avg_tracks_per_frame": 0.0,
      "frames_with_tracks": 0,
      "mean_confidence": 0.0,
    }

  total_frames = frame_count if frame_count else df['frame_id'].nunique()

  frames_with_tracks = df["frame_id"].nunique()

  track_counts = df.groupby("frame_id").size()
  track_lengths = df.groupby("track_id").size()
  unique_track_ids = df["track_id"].nunique()

  metrics = {
    # Frame statistics
    "total_frames": int(total_frames),
    "frames_with_tracks": int(frames_with_tracks),
    "pct_frames_with_tracks": float(frames_with_tracks / total_frames * 100) if total_frames > 0 else 0,
    
    # Track count statistics
    "total_tracks": len(track_lengths),
    "unique_track_ids": int(unique_track_ids),
    "avg_tracks_per_frame": float(np.mean(track_counts)) if len(track_counts) > 0 else 0,
    
    # Track length statistics
    "mean_track_lengths": float(np.mean(track_lengths)) if len(track_lengths) > 0 else 0,
    "median_track_lengths": float(np.median(track_lengths)) if len(track_lengths) > 0 else 0,
    "min_track_lengths": float(np.min(track_lengths)) if len(track_lengths) > 0 else 0,
    "max_track_lengths": float(np.max(track_lengths)) if len(track_lengths) > 0 else 0,
  }
  
  return metrics