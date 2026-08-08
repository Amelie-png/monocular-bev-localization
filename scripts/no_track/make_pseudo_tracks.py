from pathlib import Path
import pandas as pd
from tqdm import tqdm

def make_pseudo_tracks(video_names, out_dir="data/processed/pseudo_tracks"):
  det_dir = Path("data/processed/detections")
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  for v in tqdm(video_names, desc="Pseudo-tracks"):
    df = pd.read_parquet(det_dir / f"{v}_detections.parquet")
    df = df.rename(columns={"detection_id": "track_id"})  # per-frame only, not persistent
    df["video_name"] = v
    df.to_parquet(out_dir / f"{v}_trackings.parquet", index=False)

if __name__ == "__main__":
  from src.utils import load_split_file
  make_pseudo_tracks(load_split_file("data/splits/train.txt"))