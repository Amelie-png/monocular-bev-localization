from dataclasses import dataclass

@dataclass
class BevConfig:
  """
  Config for estimation pipeline.
  """
  depth_window: int = 5

  def to_dict(self):
    """
    Convert to dict for saving/logging
    """
    return {
      'depth_window': self.depth_window,
    }