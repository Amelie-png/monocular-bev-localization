import cv2
import numpy as np

class TrackingVisualizer:
  """
  Visualize tracking results with persistent IDs.
  """
  def __init__(self):
    self.colors = self._generate_colors(num_colors=200) # generate colours for different track id

  def _generate_colors(self, num_colors=200):
    """
    Generate distinct colours for track id
    """
    colors = []
    
    for i in range(num_colors):
      hue = (i * 180 // num_colors) % 180
      color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
      colors.append(tuple(map(int, color)))
    
    return colors
  
  def draw_trackings(self, image, tracked_detections):
    """
    Draw bounding boxes with track id
    
    Args:
      image: numpy array (BGR)
      tracked_detections: List of tracked detection dicts from PlayerTracker
    
    Returns:
      Annotated image (new copy)
    """
    annotated = image.copy()
    
    for track in tracked_detections:
      track_id = int(track["track_id"])
      bbox = track["bbox"]
      x1, y1, x2, y2 = map(int, bbox)
      color = self.colors[track_id % len(self.colors)]
      confidence = track["confidence"]
      
      # Draw rectangle
      cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

      # Draw label with confidence
      if confidence is not None:
        label = f"ID: {track_id} {confidence:.2f}"
      else:
        label = f"ID: {track_id}"
      label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      
      # Background for text
      cv2.rectangle(
        annotated,
        (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0], y1),
        color,
        -1  # Filled
      )
      
      # Text
      cv2.putText(
        annotated,
        label,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),  # Black text
        1
      )

    cv2.putText(
      annotated,
      f"Frame: {getattr(self, 'frame_count', 0)}",
      (20, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 255, 0),
      2
    )
    
    return annotated
  
  def create_tracking_video(self, trackings_list, output_path, fps=30):
    """
    Create video with tracking visualizations.
    
    Args:
      frame_paths: List of paths to frame images
      trackings_list: List of tracked detection lists (one per frame)
      output_path: Where to save video
      fps: Frame rate
    """
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(trackings_list.iloc[0]["frame_path"]))
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for _, frame_data in trackings_list.iterrows():
        frame_path = frame_data["frame_path"]
        tracked = frame_data["tracks"]
        
        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
          print(f"Could not read {frame_path}")
          continue
        
        # Draw tracks
        self.frame_count = frame_data["frame_id"]
        annotated = self.draw_trackings(frame, tracked)
        
        # Write
        out.write(annotated)
    
    out.release()

    print(f"Tracking video saved to: {output_path}")