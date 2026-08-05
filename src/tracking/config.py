from dataclasses import dataclass

# ONly used for DeepSort Tracker, not used in current pipeline
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
  
@dataclass
class ByteTrackConfig:
  """
  Config for ByteTrack tracking pipeline.
  """
  track_activation_threshold: float = 0.25
  lost_track_buffer: int = 30
  minimum_consecutive_frames: int = 3

  def to_dict(self):
    """
    Convert to dict for saving/logging
    """
    return {
      'track_activation_threshold': self.track_activation_threshold,
      'lost_track_buffer': self.lost_track_buffer,
      'minimum_consecutive_frames': self.minimum_consecutive_frames,
    }