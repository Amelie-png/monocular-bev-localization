import cv2
import pandas as pd
from tqdm import tqdm

def flag_occlusion(det_df, iou_threshold=0.3):
  def iou(a, b):
    xa1, ya1, xa2, ya2 = a; xb1, yb1, xb2, yb2 = b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1); area_b = (xb2 - xb1) * (yb2 - yb1)
    return inter / (area_a + area_b - inter + 1e-6)

  flags = {}
  groups = list(det_df.groupby("frame_id"))
  for frame_id, g in tqdm(groups, desc="Flagging occlusion", unit="frame", leave=False):
    boxes = g[["x1", "y1", "x2", "y2"]].values
    max_iou = 0.0
    for i in range(len(boxes)):
      for j in range(i + 1, len(boxes)):
        max_iou = max(max_iou, iou(boxes[i], boxes[j]))
    flags[frame_id] = max_iou > iou_threshold
  return pd.Series(flags, name="occluded")


def flag_rapid_camera_movement(sync_df, threshold_deg=5.0):
  s = sync_df.sort_values("frame_id").copy()
  raw_delta = s["yaw_deg"].diff()
  wrapped_delta = (raw_delta + 180) % 360 - 180
  s["yaw_delta"] = wrapped_delta.abs()
  s["rapid_camera"] = s["yaw_delta"] > threshold_deg
  return s.set_index("frame_id")["rapid_camera"]


def flag_motion_blur(frame_paths_by_id, blur_threshold=100.0):
  flags = {}
  items = list(frame_paths_by_id.items())
  for frame_id, path in tqdm(items, desc="Flagging motion blur", unit="frame", leave=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
      continue
    flags[frame_id] = cv2.Laplacian(img, cv2.CV_64F).var() < blur_threshold
  return pd.Series(flags, name="blurry")