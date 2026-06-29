from dataclasses import dataclass

@dataclass
class BevConfig:
  """
  Config for estimation pipeline.
  """

  def to_dict(self):
    """
    Convert to dict for saving/logging
    """
    return {}