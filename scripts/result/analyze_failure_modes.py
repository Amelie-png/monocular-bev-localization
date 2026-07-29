import pandas as pd
from tqdm import tqdm

from src.evaluation.failure_modes import flag_occlusion, flag_rapid_camera_movement, flag_motion_blur
from src.utils import load_split_file


def compute_flags_for_video(v):
  det_df = pd.read_parquet(f"data/processed/detections/{v}_detections.parquet")
  sync_df = pd.read_parquet(f"data/processed/sync/{v}_sync.parquet")
  meta_df = pd.read_parquet(f"data/processed/frame_metadata/{v}_metadata.parquet")

  occl = flag_occlusion(det_df)
  rapid = flag_rapid_camera_movement(sync_df)
  blur = flag_motion_blur(dict(zip(meta_df["frame_id"], meta_df["frame_path"])))
  return occl, rapid, blur


def analyze(video_names, variant="heuristic", flags_cache=None):
  euc_df = pd.read_parquet(f"outputs/eval/{variant}_euclidean_error.parquet")
  results = []

  for v in tqdm(video_names, desc=f"Analyzing failure modes ({variant})", unit="video"):
    if flags_cache is not None and v in flags_cache:
      occl, rapid, blur = flags_cache[v]
    else:
      occl, rapid, blur = compute_flags_for_video(v)
      if flags_cache is not None:
        flags_cache[v] = (occl, rapid, blur)

    sub = euc_df[euc_df["video_name"] == v].copy()
    if sub.empty:
      continue
    sub["occluded"] = sub["frame_id"].map(occl).fillna(False)
    sub["rapid_camera"] = sub["frame_id"].map(rapid).fillna(False)
    sub["blurry"] = sub["frame_id"].map(blur).fillna(False)
    results.append(sub)

  if not results:
    print(f"[{variant}] no rows to analyze")
    return pd.DataFrame()

  full = pd.concat(results, ignore_index=True)
  for cond in ["occluded", "rapid_camera", "blurry"]:
    print(f"\n{cond}:")
    print(full.groupby(cond)["euclidean_error"].agg(["mean", "median", "count"]))

  full.to_parquet(f"outputs/eval/{variant}_failure_mode_breakdown.parquet")
  return full


if __name__ == "__main__":
  videos = load_split_file("data/splits/train.txt")
  flags_cache = {}
  analyze(videos, variant="heuristic", flags_cache=flags_cache)
  analyze(videos, variant="midas", flags_cache=flags_cache)