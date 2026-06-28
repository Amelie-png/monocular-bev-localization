from pathlib import Path
from scripts.evaluate_tracking import load_split_file, evaluate_trackings
from scripts.run_tracking import run_tracking
from src.tracking import ByteTrackConfig
import yaml
import pandas as pd

ACTIVATION = [0.25, 0.45, 0.65]

LOST_BUFFER = [30, 60, 90]

CONSECUTIVE = [1, 2, 3]

TUNE_ROOT = Path("outputs/tuning/tracking")
TUNE_ROOT.mkdir(parents=True, exist_ok=True)

train_videos = load_split_file("data/splits/train.txt")

results = []

for activation in ACTIVATION:
  for lost_buffer in LOST_BUFFER:
    for consecutive in CONSECUTIVE:
      experiment_name = (
        f"activation{activation}_"
        f"lost_buffer{lost_buffer}_"
        f"consecutive{consecutive}"
      )

      experiment_root = TUNE_ROOT / experiment_name
      tracking_output_dir = experiment_root / "trackings"
      metrics_output_dir = experiment_root / "metrics"

      tracking_output_dir.mkdir(parents=True, exist_ok=True)
      metrics_output_dir.mkdir(parents=True, exist_ok=True)

      config = ByteTrackConfig(
        track_activation_threshold=activation,
        lost_track_buffer=lost_buffer,
        minimum_consecutive_frames=consecutive
      )
      with open(experiment_root / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f)

      run_tracking(
        config=config,
        output_dir=tracking_output_dir,
        video_names=train_videos
      )

      metrics_df = evaluate_trackings(
        video_names=train_videos,
        output_video=False,
        trackings_dir=tracking_output_dir,
        metrics_dir=metrics_output_dir,
      )

      if not metrics_df.empty:
        results.append({
          "experiment": experiment_name,
          'track_activation_threshold': activation,
          'lost_track_buffer': lost_buffer,
          'minimum_consecutive_frames': consecutive,
          "total_frames": metrics_df["total_frames"].sum(),
          "total_tracks": metrics_df["total_tracks"].sum(),
          "frames_with_tracks": metrics_df["frames_with_tracks"].sum(),
          "pct_frames_with_tracks": metrics_df["pct_frames_with_tracks"].mean(),
          "avg_tracks_per_frame": metrics_df["avg_tracks_per_frame"].mean(),
          "mean_track_lengths": metrics_df["mean_track_lengths"].mean(),
          "unique_track_ids": metrics_df["unique_track_ids"],
        })
      else:
        print(f"No metrics found for {experiment_name}")

if results:
  results_df = pd.DataFrame(results)
  results_file = TUNE_ROOT / "tuning_summary.csv"
  results_df.to_csv(results_file, index=False)
  print(f"\nTuning summary saved to: {results_file}")