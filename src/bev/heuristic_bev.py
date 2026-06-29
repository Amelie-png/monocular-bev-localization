class HeuristicBEVEstimator:
  """
  Abstraction for heuristic bev estimation.
  """
  def __init__(
    self,
    image_width=1920,
    depth_scale=10000
  ):
    self.image_width = image_width
    self.depth_scale = depth_scale

  def estimate(self, bbox):
    """
    Estimate BEV position based on detection bbox.

    Args:
      bbox: bounding box of detections
    
    Return:
      Dict of BEV x,y positions and depth
    """
    x1, y1, x2, y2 = bbox
    bbox_height = max(1, y2 - y1)
    center_x = (x1 + x2) / 2

    # Distance
    depth = self.depth_scale / bbox_height

    # Left-right position
    cx = self.image_width / 2 # Screen center
    offset = (center_x - cx) / cx

    # BEV coord
    bev_x = offset * depth
    bev_y = depth

    return {
      "bev_x": float(bev_x),
      "bev_y": float(bev_y),
      "depth": float(depth)
    }