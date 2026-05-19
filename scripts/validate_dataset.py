import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Load one round's sync data
sync_file = Path("data/processed/sync/match_1_round_1_sync.parquet")
position_file = Path("data/processed/player_positions/match_1_positions.parquet")

sync_df = pd.read_parquet(sync_file)
positions_df = pd.read_parquet(position_file)

# Pick a random frame
sample_idx = len(sync_df) // 2  # Middle frame
frame_info = sync_df.iloc[sample_idx]

# Load image
image = cv2.imread(frame_info['frame_path'])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get positions at this tick
tick = frame_info['matched_tick']
player_positions = positions_df[positions_df['tick'] == tick]

print(f"Frame {frame_info['frame_id']} at tick {tick}")
print(f"Players visible: {len(player_positions)}")
print(player_positions[['player_name', 'X', 'Y', 'Z']])

# Visualize (simple overlay, no perfect alignment yet)
plt.figure(figsize=(12, 8))
plt.imshow(image)
plt.title(f"Frame {frame_info['frame_id']} | Tick {tick} | {len(player_positions)} players")
plt.axis('off')
plt.tight_layout()
plt.savefig("data/sample/validation_frame.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nDataset validation complete!")
print("Check data/sample/validation_frame.png")