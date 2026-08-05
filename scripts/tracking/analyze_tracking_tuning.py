import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def parse_unique_track_ids(series_str):
  """Parses the stringified pandas Series column back into a list of ints."""
  nums = []
  for line in series_str.strip().splitlines():
    line = line.strip()
    if not line or line.startswith("Name") or line.startswith("dtype"):
      continue
    parts = line.split()
    if len(parts) >= 2:
      nums.append(int(parts[-1]))
  return nums


def score_tracking(df, out_dir="outputs/plots"):
  df = df.copy()
  df["unique_track_ids_list"] = df["unique_track_ids"].apply(parse_unique_track_ids)
  df["total_fragments"] = df["unique_track_ids_list"].apply(sum)

  # Min-max normalize three signals, each oriented so higher = better,
  # then average into one transparent composite score.
  def norm(col, invert=False):
    lo, hi = df[col].min(), df[col].max()
    if hi == lo:
      return pd.Series(1.0, index=df.index)
    v = (df[col] - lo) / (hi - lo)
    return 1 - v if invert else v

  df["frag_score"] = norm("total_fragments", invert=True) # fewer fragments = better
  df["persistence_score"] = norm("mean_track_lengths") # longer tracks = better
  df["coverage_score"] = norm("pct_frames_with_tracks") # more coverage = better
  df["combined_score"] = (df["frag_score"] + df["persistence_score"] + df["coverage_score"]) / 3

  df = df.sort_values("combined_score", ascending=False)

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(1, 3, figsize=(15, 4))
  for buf, g in df.groupby("lost_track_buffer"):
    g = g.sort_values("track_activation_threshold")
    axes[0].plot(g["track_activation_threshold"], g["total_fragments"], marker="o", label=f"buffer={buf}")
    axes[1].plot(g["track_activation_threshold"], g["mean_track_lengths"], marker="o", label=f"buffer={buf}")
    axes[2].plot(g["track_activation_threshold"], g["pct_frames_with_tracks"], marker="o", label=f"buffer={buf}")

  axes[0].set_title("Total track fragments (lower better)")
  axes[1].set_title("Mean track length (higher better)")
  axes[2].set_title("Coverage % (higher better)")
  for ax in axes:
    ax.set_xlabel("track_activation_threshold")
    ax.legend()
  fig.tight_layout()
  fig.savefig(out_dir / "tracking_sweep_comparison.png", dpi=150)
  plt.close(fig)

  print(df[["experiment", "track_activation_threshold", "lost_track_buffer", "total_fragments", "mean_track_lengths", "pct_frames_with_tracks", "combined_score"]].to_string(index=False))
  best = df.iloc[0]
  print(f"\nBest by combined_score proxy: {best['experiment']} "
        f"(activation={best['track_activation_threshold']}, "
        f"lost_buffer={best['lost_track_buffer']})")
  return df

if __name__ == "__main__":
  df = pd.read_csv("outputs/tuning/tracking/tuning_summary.csv")
  score_tracking(df)