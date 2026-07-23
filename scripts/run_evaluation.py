import pandas as pd
from pathlib import Path
from src.evaluation import (build_calibration_set, calibrate_convention, fit_scale_for_video, 
                            make_transform, build_eval_dataset, trajectory_consistency, relative_spatial_accuracy)
from src.utils import load_split_file

def video_round_number(video_name):
  return int(video_name.split("_")[-1])

def run_evaluation(video_names, variant="heuristic", out_dir="outputs/eval"):
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  # Calibrate rotation convention once, pooled across all videos
  calib_frames = [build_calibration_set(v, variant=variant) for v in video_names]
  calib_all = pd.concat([c for c in calib_frames if not c.empty], ignore_index=True)
  convention = calibrate_convention(calib_all)

  all_traj, all_rel, all_euc = [], [], []
  for v, calib_v in zip(video_names, calib_frames):
    if calib_v.empty:
      print(f"[{v}] no valid calibration rows, skipping")
      continue

    # Refit scale per video (may vary round to round)
    scale = fit_scale_for_video(calib_v, convention)
    transform_fn = make_transform(convention, scale)

    merged = build_eval_dataset(v, transform_fn, round_number=video_round_number(v), variant=variant)
    if merged.empty:
      print(f"[{v}] no matched eval rows, skipping")
      continue

    bev_df = pd.read_parquet(f"data/processed/bev/{variant}/{v}_estimations.parquet")
    bev_df = bev_df[bev_df["track_id"] != -1]

    traj = trajectory_consistency(bev_df); traj["video_name"] = v
    rel = relative_spatial_accuracy(merged); rel["video_name"] = v
    merged["video_name"] = v; merged["scale"] = scale; merged["convention"] = convention

    all_traj.append(traj); all_rel.append(rel); all_euc.append(merged)
    print(f"[{v}] scale={scale:.4f}, mean euclidean_error={merged['euclidean_error'].mean():.2f}")

  traj_df = pd.concat(all_traj, ignore_index=True)
  rel_df = pd.concat(all_rel, ignore_index=True)
  euc_df = pd.concat(all_euc, ignore_index=True)

  traj_df.to_parquet(out_dir / f"{variant}_trajectory_consistency.parquet")
  rel_df.to_parquet(out_dir / f"{variant}_relative_spatial_accuracy.parquet")
  euc_df.to_parquet(out_dir / f"{variant}_euclidean_error.parquet")

  print(f"\n[{variant}] Summary")
  print(f"  Mean trajectory step dist: {traj_df['mean_step_dist'].mean():.3f}")
  print(f"  Mean relative spatial error: {rel_df['abs_error'].mean():.3f}")
  print(f"  Mean euclidean error: {euc_df['euclidean_error'].mean():.3f}")
  return traj_df, rel_df, euc_df

if __name__ == "__main__":
  videos = load_split_file("data/splits/train.txt")
  run_evaluation(videos, variant="heuristic")
  run_evaluation(videos, variant="midas")