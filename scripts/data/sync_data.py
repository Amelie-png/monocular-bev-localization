from pathlib import Path
import pandas as pd
from src.data.synchronizer import synchronize_round

metadata_dir = Path("data/processed/frame_metadata")
positions_dir = Path("data/processed/player_positions")
round_info_dir = Path("data/processed/recording_plans")
sync_dir = Path("data/processed/sync")
sync_dir.mkdir(parents=True, exist_ok=True)

# MANUAL CONFIG
VIDEO_START_TIME = 3.0

for metadata_file in metadata_dir.glob("*_metadata.parquet"):
  video_name = metadata_file.stem.replace("_metadata", "")

  print(f"\n{'='*60}")
  print(f"Synchronizing {video_name}")
  print(f"{'='*60}")

  # Parse video_name
  parts = video_name.split("_")
  match_name = "_".join(parts[:2])
  round_number = int(parts[-1])

  # Load metadata
  frames_df = pd.read_parquet(metadata_file)
  positions_file = positions_dir / f"{match_name}_positions.parquet"
  round_info_file = round_info_dir / f"{match_name}_recording_plan.parquet"

  round_info = pd.read_parquet(round_info_file)
  round_data = round_info[round_info["round_number"] == round_number].iloc[0]
  player_pov = round_data["player_pov"]

  positions_df = pd.read_parquet(positions_file)
  pov_positions = positions_df[
    (positions_df["round_number"] == round_number) &
    (positions_df["player_name"] == player_pov)
  ]

  if pov_positions.empty:
    print(f"WARNING: no POV rows for {player_pov} in round {round_number}")
    continue

  # Infer duration
  video_end = frames_df["timestamp"].max()

  config = {
    'round_number': round_number,
    'video_start': VIDEO_START_TIME,
    'video_end': video_end,
    'tick_start': int(round_data["start_tick"]),
    'tick_end': int(round_data["end_tick"])
  }

  print(f"POV: {player_pov} | Ticks: {config['tick_start']}-{config['tick_end']} | Video: {config['video_start']:.1f}s-{config['video_end']:.1f}s")
  
  # Synchronize
  sync_df = synchronize_round(
    frame_metadata_path=metadata_file,
    pov_positions_df=pov_positions,
    round_config=config
  )
  
  # Save sync mapping
  sync_file = sync_dir / f"{video_name}_sync.parquet"
  sync_df.to_parquet(sync_file, index=False)
  
  print(f"Synchronized {len(sync_df)} frames")
  print(f"Saved to: {sync_file}")

print("\nAll rounds synchronized!")