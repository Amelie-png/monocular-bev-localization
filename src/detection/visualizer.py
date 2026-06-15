import cv2

class DetectionVisualizer:
  """
  Visualize detection results on images.
  """
  def __init__(self, color=(0, 255, 0), thickness=2):
    """
    Args:
      color: BGR color for bounding boxes
      thickness: Line thickness
    """
    self.color = color
    self.thickness = thickness
  
  def draw_detections(self, image, detections):
    """
    Draw bounding boxes on image.
    
    Args:
      image: numpy array (BGR)
      detections: List of detection dicts from PlayerDetector
    
    Returns:
      Annotated image (new copy)
    """
    annotated = image.copy()
    
    for det in detections:
      x1, y1, x2, y2 = det["bbox"]
      confidence = det["confidence"]
      
      # Draw rectangle
      cv2.rectangle(annotated, (x1, y1), (x2, y2), self.color, self.thickness)

      label = f"{det['class_name']} {confidence:.2f}"
      if "track_id" in det:
        label = (
          f"{det['class_name']} "
          f"ID:{det['track_id']} "
          f"{confidence:.2f}"
        )
      label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
      
      # Background for text
      cv2.rectangle(
        annotated,
        (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0], y1),
        self.color,
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
    
    return annotated
  
  def create_detection_video(self, frame_paths, detections_list, output_path, fps=30):
    """
    Create video with detection visualizations.
    
    Args:
      frame_paths: List of paths to frame images
      detections_list: List of detection lists (one per frame)
      output_path: Where to save video
      fps: Frame rate
    """
    # Read first frame to get dimensions
    if len(frame_paths) == 0:
      return
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
      raise ValueError(f"Could not load first frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame_path, detections in zip(frame_paths, detections_list):
      # Load frame
      frame = cv2.imread(str(frame_path))
      if frame is None:
        print(f"Could not load {frame_path}")
        continue
      
      # Draw detections
      annotated = self.draw_detections(frame, detections)
      
      # Write to video
      out.write(annotated)
    
    out.release()
    print(f"Detection video saved to: {output_path}")