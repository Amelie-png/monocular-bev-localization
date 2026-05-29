import pandas as pd
from pathlib import Path

class RoundSynchronizer:
  """
  Synchronizes video frames to demo ticks using linear interpolation.
  """
  def __init__(self, video_start_time, video_end_time, tick_start, tick_end, tick_rate=64):
    self.video_start_time = video_start_time
    self.video_end_time = video_end_time
    self.tick_start = tick_start
    self.tick_end = tick_end
    self.tick_rate = tick_rate

    self.video_duration = video_end_time - video_start_time
    self.tick_duration = tick_end - tick_start

  def video_time_to_tick(self, video_time):
    """
    Convert video timestamp to demo tick using linear interpolation.
    """
    normalized_time = (video_time - self.video_start_time) / self.video_duration

    tick = self.tick_start + normalized_time * self.tick_duration

    return int(round(tick))
  
  def tick_to_video_time(self, tick):
    """
    Convert demo tick to video timestamp.
    """
    normalized_tick = (tick - self.tick_start) / self.tick_duration

    time = self.video_start_time + normalized_tick * self.video_duration

    return time
  
def synchronize_round(frame_metadata_path, tick_data_path, round_config):
  """
  Create frame-to-tick mapping for one round.
  
  Args:
    frame_metadata_path: Path to frame metadata parquet
    tick_data_path: Path to demo tick data parquet
    round_config: Dict with 'video_start', 'video_end', 'tick_start', 'tick_end'
  
  Returns:
    DataFrame with frame_id, timestamp, tick, and nearest player positions
  """
  frames_df = pd.read_parquet(frame_metadata_path)
  ticks_df = pd.read_parquet(tick_data_path)
  
  sync = RoundSynchronizer(
    video_start_time=round_config["video_start"],
    video_end_time=round_config["video_end"],
    tick_start=round_config["tick_start"],
    tick_end=round_config["tick_end"]
  )
  
  frames_df["tick"] = frames_df["timestamp"].apply(sync.video_time_to_tick)
  
  # For each frame, find nearest tick in demo data
  def get_nearest_tick_data(target_tick):
    closest_idx = (ticks_df["tick"] - target_tick).abs().idxmin()
    return ticks_df.loc[closest_idx, "tick"]
  
  frames_df["matched_tick"] = frames_df["tick"].apply(get_nearest_tick_data)
  
  return frames_df
