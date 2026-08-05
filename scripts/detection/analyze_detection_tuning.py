import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def score_detection(df, out_dir="outputs/plots"):
  """
  Score detection tuning using a heuristic method.
  """
  df = df.copy()
  df["coverage"] = df["pct_frames_with_detections"] / 100
  df["quality_score"] = (
    3 * df["coverage"] * df["mean_confidence"]
    / (2 * df["coverage"] + df["mean_confidence"] + 1e-9)
  )

  df = df.sort_values("quality_score", ascending=False)

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  fig, ax = plt.subplots(figsize=(7, 4))
  for model, g in df.groupby("model"):
    g = g.sort_values("confidence_threshold")
    ax.plot(g["confidence_threshold"], g["coverage"], marker="o", label=f"{model} coverage")
  ax.set_xlabel("confidence_threshold")
  ax.set_ylabel("fraction of frames with >=1 detection")
  ax.set_title("Detection coverage vs. confidence threshold")
  ax.legend()
  fig.tight_layout()
  fig.savefig(out_dir / "detection_coverage_vs_threshold.png", dpi=150)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(7, 4))
  for model, g in df.groupby("model"):
    g = g.sort_values("confidence_threshold")
    ax.plot(g["confidence_threshold"], g["mean_confidence"], marker="o", label=model)
  ax.set_xlabel("confidence_threshold")
  ax.set_ylabel("mean detection confidence")
  ax.set_title("Detection confidence vs. threshold")
  ax.legend()
  fig.tight_layout()
  fig.savefig(out_dir / "detection_confidence_vs_threshold.png", dpi=150)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(6, 5))
  for model, g in df.groupby("model"):
    ax.scatter(g["coverage"], g["mean_confidence"], label=model, s=60)
    for _, row in g.iterrows():
      ax.annotate(f"{row['confidence_threshold']}", (row["coverage"], row["mean_confidence"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
  ax.set_xlabel("coverage (pct frames with detection)")
  ax.set_ylabel("mean confidence")
  ax.set_title("Coverage vs. confidence tradeoff (labeled by threshold)")
  ax.legend()
  fig.tight_layout()
  fig.savefig(out_dir / "detection_tradeoff_scatter.png", dpi=150)
  plt.close(fig)

  print(df[["experiment", "model", "confidence_threshold", "coverage", "mean_confidence", "quality_score"]].to_string(index=False))
  best = df.iloc[0]
  print(f"\nBest by quality_score proxy: {best['experiment']} (model={best['model']}, confidence_threshold={best['confidence_threshold']})")
  return df

if __name__ == "__main__":
  df = pd.read_csv("outputs/tuning/detection/tuning_summary.csv")
  score_detection(df)