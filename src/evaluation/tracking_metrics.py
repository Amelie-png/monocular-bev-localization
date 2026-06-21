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
  
  frame_groups = df.groupby('frame_id').agg({
    'track_length': ['count', 'mean', 'min', 'max']
  }).reset_index()

  frame_groups.columns = ['frame_id', 'count', 'length_mean', 'length_min', 'length_max']

  total_frames = frame_count if frame_count else df['frame_id'].nunique()

  frames_with_tracks = len(frame_groups)
  frames_without_tracks = max(0, total_frames - frames_with_tracks)

  track_counts = frame_groups['count'].values
  track_lengths = df['track_length'].values

  metrics = {
    # Frame statistics
    "total_frames": int(total_frames),
    "frames_with_tracks": int(frames_with_tracks),
    "frames_without_tracks": int(frames_without_tracks),
    "pct_frames_with_tracks": float(frames_with_tracks / total_frames * 100) if total_frames > 0 else 0,
    
    # Track count statistics
    "total_tracks": int(len(df)),
    "avg_tracks_per_frame": float(np.mean(track_counts)) if len(track_counts) > 0 else 0,
    "median_tracks_per_frame": float(np.median(track_counts)) if len(track_counts) > 0 else 0,
    "max_tracks_per_frame": int(np.max(track_counts)) if len(track_counts) > 0 else 0,
    "min_tracks_per_frame": int(np.min(track_counts)) if len(track_counts) > 0 else 0,
    
    # Track length statistics
    "mean_track_lengths": float(np.mean(track_lengths)) if len(track_lengths) > 0 else 0,
    "median_track_lengths": float(np.median(track_lengths)) if len(track_lengths) > 0 else 0,
    "std_track_lengths": float(np.std(track_lengths)) if len(track_lengths) > 0 else 0,
    "min_track_lengths": float(np.min(track_lengths)) if len(track_lengths) > 0 else 0,
    "max_track_lengths": float(np.max(track_lengths)) if len(track_lengths) > 0 else 0,
  }
  
  return metrics