from pathlib import Path
import yaml
import pandas as pd
from scripts.tracking.evaluate_tracking import evaluate_trackings
from scripts.tracking.run_tracking import run_tracking
from src.tracking import ByteTrackConfig
from src.utils import load_split_file

ACTIVATION = [0.25, 0.45, 0.65]
LOST_BUFFER = [30, 60]
CONSECUTIVE = [2, 4]

TUNE_ROOT = Path("outputs/tuning/tracking")
TUNE_ROOT.mkdir(parents=True, exist_ok=True)


def run_tracking_tuning(train_videos):
  results = []

  for activation in ACTIVATION:
    for lost_buffer in LOST_BUFFER:
      for consecutive in CONSECUTIVE:
        experiment_name = f"activation{activation}_lost_buffer{lost_buffer}_consecutive{consecutive}"
        experiment_root = TUNE_ROOT / experiment_name
        data_dir = experiment_root / "data"
        metrics_dir = experiment_root / "metrics"

        data_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        config = ByteTrackConfig(
          track_activation_threshold=activation,
          lost_track_buffer=lost_buffer,
          minimum_consecutive_frames=consecutive,
        )
        with open(experiment_root / "config.yaml", "w") as f:
          yaml.dump(config.to_dict(), f)

        print(f"\n{'='*60}\nTuning experiment: {experiment_name}\n{'='*60}")

        run_tracking(
          config=config,
          output_dir=data_dir,
          video_names=train_videos,
          force=True,
        )

        metrics_df = evaluate_trackings(
          video_names=train_videos,
          output_video=False,
          trackings_dir=data_dir,
          metrics_dir=metrics_dir,
        )

        if not metrics_df.empty:
          results.append({
            "experiment": experiment_name,
            "track_activation_threshold": activation,
            "lost_track_buffer": lost_buffer,
            "minimum_consecutive_frames": consecutive,
            "total_frames": metrics_df["total_frames"].sum(),
            "total_tracks": metrics_df["total_tracks"].sum(),
            "frames_with_tracks": metrics_df["frames_with_tracks"].sum(),
            "pct_frames_with_tracks": metrics_df["pct_frames_with_tracks"].mean(),
            "avg_tracks_per_frame": metrics_df["avg_tracks_per_frame"].mean(),
            "mean_track_lengths": metrics_df["mean_track_lengths"].mean(),
            "total_track_fragments": int(metrics_df["unique_track_ids"].sum()),
          })
        else:
          print(f"No metrics found for {experiment_name}")

  if results:
    results_df = pd.DataFrame(results)
    results_file = TUNE_ROOT / "tuning_summary.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nTuning summary saved to: {results_file}")
    return results_df

  print("No tuning results produced.")
  return pd.DataFrame()


if __name__ == "__main__":
  train_videos = load_split_file("data/splits/train.txt")
  run_tracking_tuning(train_videos)