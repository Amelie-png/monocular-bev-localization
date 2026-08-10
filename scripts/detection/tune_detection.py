from pathlib import Path
import yaml
import pandas as pd
from scripts.detection.evaluate_detection import evaluate_detections
from scripts.detection.run_detection import run_detection
from src.utils import load_split_file
from src.detection import DetectionConfig

MODELS = ["yolo26n.pt", "yolo26m.pt"]

CONF_THRESHOLDS = [0.35, 0.45, 0.55]

TUNE_ROOT = Path("outputs/tuning/detection")
TUNE_ROOT.mkdir(parents=True, exist_ok=True)

def run_detection_tuning(train_videos):
  results = []

  for model in MODELS:
    for conf in CONF_THRESHOLDS:
      experiment_name = f"{model.replace('.pt', '')}_conf{int(conf * 100):02d}"
      experiment_root = TUNE_ROOT / experiment_name
      data_dir = experiment_root / "data"
      metrics_dir = experiment_root / "metrics"

      data_dir.mkdir(parents=True, exist_ok=True)
      metrics_dir.mkdir(parents=True, exist_ok=True)

      config = DetectionConfig(model_name=model, confidence_threshold=conf)
      with open(experiment_root / "config.yaml", "w") as f:
        yaml.dump(config.to_dict(), f)

      print(f"\n{'='*60}\nTuning experiment: {experiment_name}\n{'='*60}")

      run_detection(
        config=config,
        output_dir=data_dir,
        video_names=train_videos,
        force=True,
      )

      metrics_df = evaluate_detections(
        video_names=train_videos,
        output_video=False,
        detections_dir=data_dir,
        metrics_dir=metrics_dir,
      )

      if not metrics_df.empty:
        results.append({
          "experiment": experiment_name,
          "model": model,
          "confidence_threshold": conf,
          "total_frames": metrics_df["total_frames"].sum(),
          "total_detections": metrics_df["total_detections"].sum(),
          "frames_with_detections": metrics_df["frames_with_detections"].sum(),
          "pct_frames_with_detections": metrics_df["pct_frames_with_detections"].mean(),
          "avg_detections_per_frame": metrics_df["avg_detections_per_frame"].mean(),
          "median_detections_per_frame": metrics_df["median_detections_per_frame"].mean(),
          "mean_confidence": metrics_df["mean_confidence"].mean(),
          "median_confidence": metrics_df["median_confidence"].mean(),
          "std_confidence": metrics_df["std_confidence"].mean(),
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
  run_detection_tuning(train_videos)
