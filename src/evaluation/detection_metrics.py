import numpy as np

def compute_detection_metrics(df, frame_count=None):
  """
  Compute detection metrics given DataFrame file.

  Args:
    df: dataframe file of one single round
    frame_count: Total frames in video (if None, use unique frame_ids)
  
  Return:
    Dict of metrics
  """
  if df.empty:
    return {
      "total_frames": frame_count or 0,
      "total_detections": 0,
      "avg_detections_per_frame": 0.0,
      "frames_with_detections": 0,
      "mean_confidence": 0.0,
    }
  
  frame_groups = df.groupby('frame_id').agg({
    'confidence': ['count', 'mean', 'min', 'max']
  }).reset_index()

  frame_groups.columns = ['frame_id', 'count', 'conf_mean', 'conf_min', 'conf_max']

  total_frames = frame_count if frame_count else df['frame_id'].nunique()

  frames_with_dets = len(frame_groups)
  frames_without_dets = max(0, total_frames - frames_with_dets)

  detection_counts = frame_groups['count'].values
  confidences = df['confidence'].values

  metrics = {
    # Frame statistics
    "total_frames": int(total_frames),
    "frames_with_detections": int(frames_with_dets),
    "frames_without_detections": int(frames_without_dets),
    "pct_frames_with_detections": float(frames_with_dets / total_frames * 100) if total_frames > 0 else 0,
    
    # Detection count statistics
    "total_detections": int(len(df)),
    "avg_detections_per_frame": float(np.mean(detection_counts)) if len(detection_counts) > 0 else 0,
    "median_detections_per_frame": float(np.median(detection_counts)) if len(detection_counts) > 0 else 0,
    "max_detections_per_frame": int(np.max(detection_counts)) if len(detection_counts) > 0 else 0,
    "min_detections_per_frame": int(np.min(detection_counts)) if len(detection_counts) > 0 else 0,
    
    # Confidence statistics
    "mean_confidence": float(np.mean(confidences)) if len(confidences) > 0 else 0,
    "median_confidence": float(np.median(confidences)) if len(confidences) > 0 else 0,
    "std_confidence": float(np.std(confidences)) if len(confidences) > 0 else 0,
    "min_confidence": float(np.min(confidences)) if len(confidences) > 0 else 0,
    "max_confidence": float(np.max(confidences)) if len(confidences) > 0 else 0,
  }
  
  return metrics