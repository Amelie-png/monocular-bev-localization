from dataclasses import dataclass
from src.utils.device import detect_available_device

@dataclass
class DetectionConfig:
  """
  Config for detection pipeline.
  """
  model_name: str = "yolo26m.pt"
  confidence_threshold: float = 0.55
  crop_bottom_ratio: float = 0.0
  batch_size: int = 8
  device: str = None

  def to_dict(self):
    """
    Convert to dict for saving/logging
    """
    return {
      'model_name': self.model_name,
      'confidence_threshold': self.confidence_threshold,
      'crop_bottom_ratio': self.crop_bottom_ratio,
      'batch_size': self.batch_size,
    }
  
  def __post_init__(self):
    """
    Auto-detect device if not specified (None).
    """
    if self.device is None:
      self.device, self.device_name = detect_available_device()