from pathlib import Path
from src.data.demo_parser import parse_demo

demo_dir = Path("data/raw/demos")
save_dir = Path("data/processed/player_positions")

for demo_path in demo_dir.glob("*.dem"):

  print(f"Parsing {demo_path}")

  df = parse_demo(demo_path)

  output_file = (
    save_dir /
    f"{demo_path.stem}_positions.parquet"
  )

  df.to_parquet(output_file)

print("Done")