from dataclasses import dataclass

@dataclass
class TrackConfig:
  """
  Config for tracking pipeline.
  """
  max_age: int = 30
  n_init: int = 3
  nn_budget: int = 100

  def to_dict(self):
    """
    Convert to dict for saving/logging
    """
    return {
      'max_age': self.max_age,
      'n_init': self.n_init,
      'nn_budget': self.nn_budget,
    }