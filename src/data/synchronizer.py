import pandas as pd
import numpy as np

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
  
def synchronize_round(frame_metadata_path, pov_positions_df, round_config):
  """
  Create frame-to-tick mapping for one round.
  
  Args:
    frame_metadata_path: Path to frame metadata parquet
    pov_positions_df: Demo tick DataFrame for camera pov only
    round_config: Dict with 'video_start', 'video_end', 'tick_start', 'tick_end'
  
  Returns:
    DataFrame with frame_id, timestamp, tick, and nearest player positions
  """
  frames_df = pd.read_parquet(frame_metadata_path)
  
  sync = RoundSynchronizer(
    video_start_time=round_config["video_start"],
    video_end_time=round_config["video_end"],
    tick_start=round_config["tick_start"],
    tick_end=round_config["tick_end"]
  )
  
  frames_df["tick"] = frames_df["timestamp"].apply(sync.video_time_to_tick)

  pov_positions_df = pov_positions_df.sort_values("tick").reset_index(drop=True)
  pov_ticks = pov_positions_df["tick"].values

  def nearest_idx(target_tick):
    idx = np.searchsorted(pov_ticks, target_tick)
    idx = min(idx, len(pov_ticks) - 1)
    if idx > 0 and abs(pov_ticks[idx - 1] - target_tick) < abs(pov_ticks[idx] - target_tick):
      idx -= 1
    return idx
  
  idxs = frames_df["tick"].apply(nearest_idx).values
  frames_df["matched_tick"] = pov_positions_df.loc[idxs, "tick"].values
  frames_df["cam_x"] = pov_positions_df.loc[idxs, "x"].values
  frames_df["cam_y"] = pov_positions_df.loc[idxs, "y"].values
  frames_df["yaw_deg"] = pov_positions_df.loc[idxs, "yaw"].values
  frames_df["pitch_deg"] = pov_positions_df.loc[idxs, "pitch"].values
  
  return frames_df
