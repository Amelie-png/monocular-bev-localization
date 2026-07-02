from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import cv2
import numpy as np

from src.utils.device import detect_available_device

class MiDaSEstimator:
  """
  Abstraction for MiDaSEstimator.
  """
  def __init__(self, model_name="Intel/dpt-hybrid-midas"):
    self.device, device_name = detect_available_device()
    print(f"Loading MiDaS on {device_name}")

    self.processor = DPTImageProcessor.from_pretrained(model_name)
    self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
    self.model.eval()
  
  @torch.no_grad()
  @torch.inference_mode()
  def estimate(self, image):
    """
    Estimate depth map.

    Args:
      image: BGR image

    Return:
      depth_map: HxW numpy array
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    inputs = self.processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(self.device) for k, v in inputs.items()}

    prediction = self.model(**inputs).predicted_depth

    prediction = torch.nn.functional.interpolate(
      prediction.unsqueeze(1),
      size=rgb.shape[:2],
      mode="bicubic",
      align_corners=False,
    ).squeeze()

    depth = prediction.cpu().numpy()

    return depth
  
  def visualize(self, depth):
    low = np.percentile(depth, 2)
    high = np.percentile(depth, 98)

    depth = np.clip(depth, low, high)

    depth_uint8 = cv2.normalize(
      depth,
      None,
      0,
      255,
      norm_type=cv2.NORM_MINMAX,
      dtype=cv2.CV_8U
    )

    colored = cv2.applyColorMap(
      depth_uint8,
      cv2.COLORMAP_TURBO
    )

    return colored
