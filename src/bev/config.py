from dataclasses import dataclass

@dataclass
class BevConfig:
  """
  Config for estimation pipeline.
  """
  depth_window: int = 5
  low_pct: float = 2.0
  high_pct: float = 98.0
  fov_scale: float = 1.0

  def to_dict(self):
    """
    Convert to dict for saving/logging
    """
    return {
      'depth_window': self.depth_window,
      'low_pct': self.low_pct,
      'high_pct': self.high_pct,
      'fov_scale': self.fov_scale,
    }