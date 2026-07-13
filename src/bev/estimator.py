import numpy as np

class BevEstimator:
  """
  Estimate bird's-eye-view coordinates from player detections.
  """
  def __init__(
    self,
    image_width=1920,
    depth_scale=10000,
    depth_window=5,
  ):
    self.image_width = image_width
    self.depth_scale = depth_scale
    self.depth_window = depth_window

  def estimate(self, bbox, depth=None, depth_map=None):
    """
    Estimate BEV position.

    Args:
      bbox: [x1, y1, x2, y2]
      depth: Optional depth value.
      depth_map: Optional MiDaS depth map.

    Return:
      Dict of BEV x,y positions and depth
    """
    # Depth
    if depth is not None:
      player_depth = depth
    elif depth_map is not None:
      player_depth = self.extract_depth(depth_map, bbox)
    else:
      player_depth = self.heuristic_depth(bbox)

    return self.project(bbox, player_depth)
  
  def heuristic_depth(self, bbox):
    """
    Estimate depth from bounding box height (heuristically).
    """
    _, y1, _, y2 = bbox

    bbox_height = max(1.0, y2 - y1)

    return self.depth_scale / bbox_height
  
  def extract_depth(self, depth_map, bbox):
    """
    Extract player depth from MiDaS depth map.

    Uses the median depth around the player's feet.
    """
    x1, _, x2, y2 = bbox

    cx = int((x1 + x2) / 2)
    cy = int(y2 - 5)

    h, w = depth_map.shape

    patch_half = self.depth_window // 2

    x0 = max(0, cx - patch_half)
    x1 = min(w, cx + patch_half + 1)

    y0 = max(0, cy - patch_half)
    y1 = min(h, cy + patch_half + 1)

    patch = depth_map[y0:y1, x0:x1]

    if patch.size == 0:
      return 0.0
    
    depth = float(np.median(patch))

    depth_min = float(depth_map.min())
    depth_max = float(depth_map.max())

    depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)

    distance = 1.0 - depth
    distance *= 300

    return distance
  
  def project(self, bbox, player_depth):
    """
    Project player into BEV coordinates.

    Args:
      bbox: bounding box of detections
      player_depth: depth of player
    
    Return:
      Dict of BEV x,y positions and depth
    """
    x1, _, x2, _ = bbox
    center_x = (x1 + x2) / 2

    # Left-right position
    image_center = self.image_width / 2
    offset = (center_x - image_center) / image_center

    # BEV coord
    bev_x = offset * player_depth
    bev_y = player_depth

    return {
      "bev_x": float(bev_x),
      "bev_y": float(bev_y),
      "depth": float(player_depth)
    }