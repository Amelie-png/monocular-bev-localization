from transformers import DPTImageProcessor, DPTForDepthEstimation
from pathlib import Path
import torch
import cv2
import numpy as np

from src.utils import detect_available_device

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

def compute_depth_normalization_stats(video_name, depth_dir="data/processed/depths", n_samples=20, low_pct=2, high_pct=98):
  """
  Sample depth maps across the video and compute stable global percentile
  bounds, so the same physical distance maps to the same normalized value
  in every frame.
  """
  video_depth_dir = Path(depth_dir) / video_name
  files = sorted(video_depth_dir.glob("frame_*.npy"))
  if not files:
    raise ValueError(f"No depth files found for {video_name}")

  step = max(1, len(files) // n_samples)
  sample_files = files[::step][:n_samples]

  values = [np.load(f).flatten() for f in sample_files]
  all_vals = np.concatenate(values)

  depth_min = float(np.percentile(all_vals, low_pct))
  depth_max = float(np.percentile(all_vals, high_pct))
  return depth_min, depth_max