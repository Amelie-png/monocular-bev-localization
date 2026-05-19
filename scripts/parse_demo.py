from pathlib import Path
import pandas as pd
from src.data.demo_parser import parse_demo, get_round_info

demo_dir = Path("data/raw/demos")
save_dir = Path("data/processed/player_positions")
save_dir.mkdir(parents=True, exist_ok=True)

for demo_path in demo_dir.glob("*.dem"):

  print(f"\n{'='*60}")
  print(f"Parsing {demo_path.name}")
  print(f"{'='*60}")

  df = parse_demo(demo_path)

  print(f"Total ticks parsed: {len(df)}")
  print(f"Rounds found: {df['round_number'].nunique()}")
  print(f"Players: {df['player_name'].nunique()}")

  output_file = save_dir / f"{demo_path.stem}_positions.parquet"
  df.to_parquet(output_file, index=False)
  print(f"Saved to: {output_file}")

  rounds = sorted(df['round_number'].unique())
  round_info_list = []
  
  for round_num in rounds:
    info = get_round_info(df, round_num)
    round_info_list.append(info)
    print(f"  Round {round_num}: ticks {info['start_tick']}-{info['end_tick']} "
      f"({info['duration_seconds']:.1f}s)"
    )

  round_info_df = pd.DataFrame(round_info_list)
  info_file = save_dir / f"{demo_path.stem}_round_info.parquet"
  round_info_df.to_parquet(info_file, index=False)

print("\nDemo parsing complete!")