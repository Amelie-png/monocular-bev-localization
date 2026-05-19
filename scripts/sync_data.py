from pathlib import Path
import pandas as pd
from src.data.synchronizer import synchronize_round

metadata_dir = Path("data/processed/frame_metadata")
positions_dir = Path("data/processed/player_positions")
sync_dir = Path("data/processed/sync")
sync_dir.mkdir(parents=True, exist_ok=True)

# MANUAL CONFIGURATION
round_configs = {
  'match_1_round_1': {
    'frame_metadata': 'match_1_round_1_metadata.parquet',
    'position_data': 'match_1_positions.parquet',
    'round_number': 0,  # 0-indexed
    'video_start': 0.0,  # MANUALLY DETERMINED: when freeze ends in video
    'video_end': 62.0,   # MANUALLY DETERMINED: when round ends in video
    'tick_start': None,  # Will auto-fill from round_info
    'tick_end': None     # Will auto-fill from round_info
  },
  'match_1_round_2': {
    'frame_metadata': 'match_1_round_2_metadata.parquet',
    'position_data': 'match_1_positions.parquet',
    'round_number': 0,  # 0-indexed
    'video_start': 0.0,  # MANUALLY DETERMINED: when freeze ends in video
    'video_end': 62.0,   # MANUALLY DETERMINED: when round ends in video
    'tick_start': None,  # Will auto-fill from round_info
    'tick_end': None     # Will auto-fill from round_info
  },
  # Add more rounds here
}

# Process each round
for video_name, config in round_configs.items():
  print(f"\n{'='*60}")
  print(f"Synchronizing {video_name}")
  print(f"{'='*60}")
  
  frame_meta_path = metadata_dir / config['frame_metadata']
  position_path = positions_dir / config['position_data']
  
  # Get tick range from round info if not manually specified
  if config['tick_start'] is None:
    round_info_path = positions_dir / config['position_data'].replace('_positions.parquet', '_round_info.parquet')
    round_info = pd.read_parquet(round_info_path)
    round_data = round_info[round_info['round_number'] == config['round_number']].iloc[0]
    
    config['tick_start'] = round_data['start_tick']
    config['tick_end'] = round_data['end_tick']
    
    print(f"Auto-detected ticks: {config['tick_start']} - {config['tick_end']}")
  
  # Synchronize
  sync_df = synchronize_round(
    frame_metadata_path=frame_meta_path,
    tick_data_path=position_path,
    round_config=config
  )
  
  # Save sync mapping
  sync_file = sync_dir / f"{video_name}_sync.parquet"
  sync_df.to_parquet(sync_file, index=False)
  
  print(f"Synchronized {len(sync_df)} frames")
  print(f"Saved to: {sync_file}")

print("\nAll rounds synchronized!")