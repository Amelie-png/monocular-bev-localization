from ultralytics import YOLO

class PlayerDetector:
  """
  Abstraction for YOLOv8 for detecting players in CS2 frames.
  """
  def __init__(self, model_name="yolov8m.pt", confidence_threshold=0.5, crop_bottom_ratio=0):
    """
    Args:
      model_name: YOLOv8 model variant (n/s/m/l/x)
      confidence_threshold: Minimum confidence for detections
      crop_bottom_ratio: Ratio of bottom to crop (0 = no crop)
    """
    self.model = YOLO(model_name)
    self.conf_threshold = confidence_threshold
    self.crop_bottom_ratio = crop_bottom_ratio
    self.person_class_id = 0

  def _crop_image(self, image):
    """
    Helper function to crop image based on crop_bottom_ratio.
    """
    if self.crop_bottom_ratio == 0:
      return image
    
    height = image.shape[0]
    crop_height = int(height * (1 - self.crop_bottom_ratio))
    return image[:crop_height, :]

  def _extract_boxes(self, result):
    """
    Helper function to extract boxes given result from image detection.
    """
    detections = []

    for box in result.boxes:
      class_id = int(box.cls[0])

      if class_id == self.person_class_id:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
        confidence = float(box.conf[0])

        detections.append({
          "bbox": [x1, y1, x2, y2],
          "confidence": confidence,
          "class_id": class_id,
          "class_name": "person"
        })

    return detections

  def detect(self, image):
    """
    Run detection on a single image.
    
    Args:
      image: numpy array (BGR format from cv2.imread)
    
    Return:
      List of dicts with keys: bbox, confidence, class_id
    """
    cropped_image = self._crop_image(image)
    results = self.model(cropped_image, conf=self.conf_threshold, verbose=False)

    return self._extract_boxes(results[0])
  
  def detect_batch(self, images):
    """
    Run detection on a batch of images.
    
    Args:
      images: List of numpy arrays (BGR format from cv2.imread)
    
    Return:
      List of detection lists (one per image)
    """
    cropped_images = [self._crop_image(img) for img in images]
    results = self.model(cropped_images, conf=self.conf_threshold, verbose=False)

    return [self._extract_boxes(result) for result in results]