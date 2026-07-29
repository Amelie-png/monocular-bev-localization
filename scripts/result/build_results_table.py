import pandas as pd
from pathlib import Path

def summarize(label, eval_dir="outputs/eval"):
  eval_dir = Path(eval_dir)
  traj = pd.read_parquet(eval_dir / f"{label}_trajectory_consistency.parquet") \
    if (eval_dir / f"{label}_trajectory_consistency.parquet").exists() else None
  rel = pd.read_parquet(eval_dir / f"{label}_relative_spatial_accuracy.parquet")
  euc = pd.read_parquet(eval_dir / f"{label}_euclidean_error.parquet")
  return {
    "variant": label,
    "mean_trajectory_step_dist": traj["mean_step_dist"].mean() if traj is not None else None,
    "mean_relative_spatial_error": rel["abs_error"].mean(),
    "mean_euclidean_error": euc["euclidean_error"].mean(),
    "median_euclidean_error": euc["euclidean_error"].median(),
  }

if __name__ == "__main__":
  rows = [summarize("heuristic"), summarize("midas")]
  pd.DataFrame(rows).to_csv("outputs/eval/results_table.csv", index=False)
  print(pd.DataFrame(rows).to_string(index=False))