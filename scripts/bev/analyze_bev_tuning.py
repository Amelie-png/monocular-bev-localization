import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_heuristic_tuning(csv_path="outputs/tuning/bev/heuristic/tuning_summary.csv", out_dir="outputs/plots"):
  df = pd.read_csv(csv_path).sort_values("fov_scale")
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  fig, ax = plt.subplots(figsize=(6, 4))
  ax.plot(df["fov_scale"], df["mean_error"], marker="o")
  best = df.loc[df["mean_error"].idxmin()]
  ax.scatter([best["fov_scale"]], [best["mean_error"]], color="red", zorder=5,
             label=f"best: fov_scale={best['fov_scale']}")
  ax.set_xlabel("fov_scale")
  ax.set_ylabel("mean calibration error")
  ax.set_title("Heuristic BEV: fov_scale tuning")
  ax.legend()
  fig.tight_layout()
  fig.savefig(out_dir / "bev_heuristic_tuning.png", dpi=150)
  plt.close(fig)

  print(f"Best Heuristic config: {best.to_dict()}")


def plot_midas_tuning(csv_path="outputs/tuning/bev/midas/tuning_summary.csv", out_dir="outputs/plots"):
  df = pd.read_csv(csv_path)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  depth_windows = sorted(df["depth_window"].unique())
  fig, axes = plt.subplots(1, len(depth_windows), figsize=(6 * len(depth_windows), 4), squeeze=False)
  axes = axes[0]

  for ax, dw in zip(axes, depth_windows):
    sub = df[df["depth_window"] == dw]
    for (lp, hp), g in sub.groupby(["low_pct", "high_pct"]):
      g = g.sort_values("fov_scale")
      ax.plot(g["fov_scale"], g["mean_error"], marker="o", label=f"low={lp}, high={hp}")
    ax.set_title(f"depth_window={dw}")
    ax.set_xlabel("fov_scale")
    ax.set_ylabel("mean calibration error")
    ax.legend(fontsize=8)

  fig.tight_layout()
  fig.savefig(out_dir / "bev_midas_tuning.png", dpi=150)
  plt.close(fig)

  best = df.loc[df["mean_error"].idxmin()]
  print(f"Best MiDaS config: {best.to_dict()}")


if __name__ == "__main__":
  plot_heuristic_tuning()
  plot_midas_tuning()