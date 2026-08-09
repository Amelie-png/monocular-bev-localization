from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.bev import BevVisualizer
from src.utils import filter_missing, report_skip, load_split_file, compute_fit_scale


def bev_video_output_path(video_name, variant, out_dir=None):
  out_dir = out_dir or f"outputs/videos/bev/{variant}"
  return Path(out_dir) / f"{video_name}.mp4"


def render_bev_video(video_name, variant, bev_dir=None, out_dir=None, fps=30, canvas_size=800, padding_ratio=0.15, scale=None):
  """
  scale: if given, use this fixed scale (e.g. for combined-video consistency
  via render_combined_bev_pair). If None, auto-fit independently to this
  video+variant's own bev_x/bev_y range so no points clip outside the canvas.
  """
  bev_dir = Path(bev_dir or f"data/processed/bev/{variant}")
  bev_file = bev_dir / f"{video_name}_estimations.parquet"
  if not bev_file.exists():
    print(f"BEV estimations not found: {bev_file}")
    return None

  df = pd.read_parquet(bev_file)
  fit_scale = scale if scale is not None else compute_fit_scale(df, canvas_size, padding_ratio)

  frame_estimations = []
  for frame_id in sorted(df["frame_id"].unique()):
    g = df[df["frame_id"] == frame_id]
    frame_estimations.append([
      {"bev_x": row["bev_x"], "bev_y": row["bev_y"], "track_id": row["track_id"]}
      for _, row in g.iterrows()
    ])

  out_path = bev_video_output_path(video_name, variant, out_dir)
  BevVisualizer(scale=fit_scale).create_bev_video(frame_estimations, out_path, fps=fps)
  return out_path


def render_combined_bev_pair(
    video_name, canvas_size=800, padding_ratio=0.15,
    heuristic_bev_dir=None, midas_bev_dir=None, fps=30,
    combined_out_dir="outputs/videos/bev_combined_scale"):
  """
  Renders heuristic and MiDaS BEV videos at a SHARED scale -- fit to
  whichever variant has the larger extent -- so a viewer can visually
  compare spread/jitter between the two panels honestly. Used only for
  the side-by-side combined comparison video, not the standalone renders.
  """
  heuristic_dir = Path(heuristic_bev_dir or "data/processed/bev/heuristic")
  midas_dir = Path(midas_bev_dir or "data/processed/bev/midas")

  heuristic_file = heuristic_dir / f"{video_name}_estimations.parquet"
  midas_file = midas_dir / f"{video_name}_estimations.parquet"
  if not heuristic_file.exists() or not midas_file.exists():
    print(f"[{video_name}] missing heuristic or midas estimations, skipping combined pair")
    return None, None

  heuristic_df = pd.read_parquet(heuristic_file)
  midas_df = pd.read_parquet(midas_file)

  heuristic_scale = compute_fit_scale(heuristic_df, canvas_size, padding_ratio)
  midas_scale = compute_fit_scale(midas_df, canvas_size, padding_ratio)
  # Smaller scale = fits the larger extent, so neither variant clips.
  shared_scale = min(heuristic_scale, midas_scale)

  heur_path = render_bev_video(
    video_name, "heuristic", bev_dir=heuristic_dir,
    out_dir=Path(combined_out_dir) / "heuristic",
    fps=fps, scale=shared_scale)
  midas_path = render_bev_video(
    video_name, "midas", bev_dir=midas_dir,
    out_dir=Path(combined_out_dir) / "midas",
    fps=fps, scale=shared_scale)
  return heur_path, midas_path


def run_bev_visualization(video_names=None, variants=("heuristic", "midas"), fps=30, force=False):
  all_names = video_names if video_names else load_split_file("data/splits/train.txt")

  for variant in variants:
    def out_path_fn(v, variant=variant):
      return bev_video_output_path(v, variant)

    todo = filter_missing(all_names, out_path_fn, force=force)
    report_skip(f"bev visualization ({variant})", all_names, todo)

    for v in tqdm(todo, desc=f"Rendering BEV videos ({variant})", unit="video"):
      render_bev_video(v, variant, fps=fps)

  print("\nBEV visualization complete!")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--video", type=str, nargs="+", default=None, help="Specific videos to visualize")
  parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
  parser.add_argument("--force", action="store_true", default=None, help="Force re-render")

  args = parser.parse_args()
  run_bev_visualization(video_names=args.video, fps=args.fps, force=args.force)