from pathlib import Path
import pandas as pd

def validate_video(video_name):
  issues = []

  meta = pd.read_parquet(f"data/processed/frame_metadata/{video_name}_metadata.parquet")
  det = pd.read_parquet(f"data/processed/detections/{video_name}_detections.parquet")
  trk = pd.read_parquet(f"data/processed/trackings/{video_name}_trackings.parquet")
  bev_h = pd.read_parquet(f"data/processed/bev/heuristic/{video_name}_estimations.parquet")
  bev_m = pd.read_parquet(f"data/processed/bev/midas/{video_name}_estimations.parquet")

  meta_ids = set(meta["frame_id"])
  det_ids = set(det["frame_id"])
  trk_ids = set(trk["frame_id"])
  bev_h_ids = set(bev_h["frame_id"])
  bev_m_ids = set(bev_m["frame_id"])

  if not det_ids.issubset(meta_ids):
    issues.append(f"detection has {len(det_ids - meta_ids)} frame_ids not in metadata")
  if not trk_ids.issubset(det_ids):
    issues.append(f"tracking has {len(trk_ids - det_ids)} frame_ids not in detection")
  if not bev_h_ids.issubset(trk_ids):
    issues.append(f"heuristic bev has {len(bev_h_ids - trk_ids)} frame_ids not in tracking")
  if not bev_m_ids.issubset(trk_ids):
    issues.append(f"midas bev has {len(bev_m_ids - trk_ids)} frame_ids not in tracking")

  # track_id sets should match between the two bev variants for the same frames
  h_tracks = set(zip(bev_h["frame_id"], bev_h["track_id"]))
  m_tracks = set(zip(bev_m["frame_id"], bev_m["track_id"]))
  common = bev_h_ids & bev_m_ids
  h_common = {(f, t) for f, t in h_tracks if f in common}
  m_common = {(f, t) for f, t in m_tracks if f in common}
  if h_common != m_common:
    issues.append(f"heuristic/midas disagree on (frame_id, track_id) pairs for {len(h_common ^ m_common)} entries")

  # depth files must exist for every frame_id midas bev claims to cover
  depth_dir = Path(f"data/processed/depths/{video_name}")
  missing_depth = [f for f in bev_m_ids if not (depth_dir / f"frame_{f:06d}.npy").exists()]
  if missing_depth:
    issues.append(f"{len(missing_depth)} midas bev frames reference missing .npy depth files")

  extra = det_ids - meta_ids
  if extra:
    print(f"  extra frame_ids: {extra}")

  if issues:
    print(f"[{video_name}] ISSUES FOUND:")
    for i in issues:
      print(f"  - {i}")
  else:
    print(f"[{video_name}] OK — {len(meta_ids)} frames, {len(trk['track_id'].unique())} tracks")

  return issues

if __name__ == "__main__":
  from src.utils import load_split_file
  for v in load_split_file("data/splits/train.txt"):
    validate_video(v)