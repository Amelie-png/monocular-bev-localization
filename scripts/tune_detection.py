from pathlib import Path
from scripts.evaluate_detection import load_split_file, evaluate_detections
from scripts.run_detection import run_detection
from src.detection import DetectionConfig
import yaml
import pandas as pd

MODELS = [
  "yolo26n.pt",
  "yolo26m.pt"
]

CONF_THRESHOLDS = [
  0.35,
  0.45,
  0.55
]

TUNE_ROOT = Path("outputs/tuning/detection")
TUNE_ROOT.mkdir(parents=True, exist_ok=True)

train_videos = load_split_file("data/splits/train.txt")

results = []

for model in MODELS:
  for conf in CONF_THRESHOLDS:
    experiment_name = (
      f"{model.replace('.pt','')}_"
      f"conf{int(conf*100):02d}"
    )

    experiment_root = TUNE_ROOT / experiment_name
    detection_output_dir = experiment_root / "detections"
    metrics_output_dir = experiment_root / "metrics"

    detection_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    config = DetectionConfig(
      model_name=model,
      confidence_threshold=conf
    )
    with open(experiment_root / "config.yaml", "w") as f:
      yaml.dump(config.to_dict(), f)

    run_detection(
      config=config,
      output_dir=detection_output_dir,
      video_names=train_videos
    )

    metrics_df = evaluate_detections(
      video_names=train_videos,
      output_video=False,
      detections_dir=detection_output_dir,
      metrics_dir=metrics_output_dir,
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