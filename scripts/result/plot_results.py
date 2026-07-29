import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_error_comparison(labels, title, filename, eval_dir="outputs/eval", out_dir="outputs/plots"):
  out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
  fig, ax = plt.subplots(figsize=(6, 4))
  data = [pd.read_parquet(Path(eval_dir) / f"{l}_euclidean_error.parquet")["euclidean_error"].dropna() for l in labels]
  ax.boxplot(data, tick_labels=labels, showfliers=False)
  ax.set_ylabel("Euclidean positional error")
  ax.set_title(title)
  fig.tight_layout()
  fig.savefig(out_dir / filename, dpi=150)
  plt.close(fig)

if __name__ == "__main__":
  plot_error_comparison(["heuristic", "midas"], "Heuristic vs Depth-based BEV", "heuristic_vs_midas.png")
  plot_error_comparison(["heuristic", "heuristic_notrack"], "With vs Without Tracking (heuristic)", "heuristic_tracking_ablation.png")