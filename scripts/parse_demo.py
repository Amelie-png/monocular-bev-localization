from pathlib import Path
from src.data.demo_parser import parse_demo, generate_recording_plan

demo_dir = Path("data/raw/demos")
save_dir = Path("data/processed/player_positions")
save_dir.mkdir(parents=True, exist_ok=True)
plan_save_dir = Path("data/processed/recording_plans")
plan_save_dir.mkdir(parents=True, exist_ok=True)

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

  plan_df = generate_recording_plan(df)
  output_plan = plan_save_dir / f"{demo_path.stem}_recording_plan.parquet"
  plan_df.to_parquet(output_plan, index=False)
  print(f"Saved to: {output_plan}")
  print("\nRecording plan:")
  print(plan_df[["round_number", "player_pov", "start_tick", "end_tick", "duration_seconds"]])

print("\nDemo parsing complete!")