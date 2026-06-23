from pathlib import Path
from scripts.evaluate_tracking import load_split_file, evaluate_trackings
from scripts.run_tracking import run_tracking
from src.tracking import TrackConfig
import yaml
import pandas as pd

MAX_AGES = [15, 30, 60]

N_INITS = [2, 3, 5]

NN_BUDGET = [100]

TUNE_ROOT = Path("outputs/tuning/tracking")
TUNE_ROOT.mkdir(parents=True, exist_ok=True)

train_videos = load_split_file("data/splits/train.txt")

results = []

for age in MAX_AGES:
  for n_init in N_INITS:
    for budget in NN_BUDGET:
      experiment_name = (
        f"max_age{age}_"
        f"n_init{n_init}_"
        f"nn_budget{budget}"
      )

      experiment_root = TUNE_ROOT / experiment_name
      tracking_output_dir = experiment_root / "trackings"
      metrics_output_dir = experiment_root / "metrics"

      tracking_output_dir.mkdir(parents=True, exist_ok=True)
      metrics_output_dir.mkdir(parents=True, exist_ok=True)

      config = TrackConfig(
        max_age=age,
        n_init=n_init,
        nn_budget=budget
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
          "max_age": age,
          "n_init": n_init,
          "nn_budget": budget,
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