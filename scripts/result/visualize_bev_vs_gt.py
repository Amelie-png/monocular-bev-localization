import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def render_bev_vs_gt_video(video_name, variant, eval_dir="outputs/eval", out_dir="outputs/videos/bev_vs_gt", canvas_size=800, fps=30, padding_ratio=0.1):
  df = pd.read_parquet(Path(eval_dir) / f"{variant}_euclidean_error.parquet")
  df = df[df["video_name"] == video_name]
  if df.empty:
    print(f"[{video_name}] no matched eval rows for {variant}, skipping")
    return None

  x_min = min(df["world_x"].min(), df["x"].min())
  x_max = max(df["world_x"].max(), df["x"].max())
  y_min = min(df["world_y"].min(), df["y"].min())
  y_max = max(df["world_y"].max(), df["y"].max())
  x_pad = (x_max - x_min) * padding_ratio + 1e-6
  y_pad = (y_max - y_min) * padding_ratio + 1e-6
  x_min, x_max = x_min - x_pad, x_max + x_pad
  y_min, y_max = y_min - y_pad, y_max + y_pad

  def to_px(x, y):
    px = int((x - x_min) / (x_max - x_min) * canvas_size)
    py = int(canvas_size - (y - y_min) / (y_max - y_min) * canvas_size)
    return px, py

  out_path = Path(out_dir) / f"{video_name}_{variant}_bev_vs_gt.mp4"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_size, canvas_size))

  GT_COLOR, PRED_COLOR, LINE_COLOR = (0, 200, 0), (0, 0, 220), (180, 180, 180)  # BGR

  frame_ids = sorted(df["frame_id"].unique())
  for frame_id in tqdm(frame_ids, desc=f"{video_name} ({variant}) bev-vs-gt", unit="frame", leave=False):
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    frame_rows = df[df["frame_id"] == frame_id]

    for _, row in frame_rows.iterrows():
      gt_px = to_px(row["x"], row["y"])
      pred_px = to_px(row["world_x"], row["world_y"])
      cv2.line(canvas, gt_px, pred_px, LINE_COLOR, 1)
      cv2.circle(canvas, gt_px, 6, GT_COLOR, -1)
      cv2.circle(canvas, pred_px, 6, PRED_COLOR, -1)
      cv2.putText(canvas, str(row["player_name"]), (gt_px[0] + 8, gt_px[1] - 8),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    mean_err = frame_rows["euclidean_error"].mean()
    cv2.putText(canvas, f"frame {frame_id} | mean error={mean_err:.1f}",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    writer.write(canvas)

  writer.release()
  print(f"BEV-vs-GT video saved to: {out_path}")
  return out_path


if __name__ == "__main__":
  from src.utils import load_split_file
  videos = load_split_file("data/splits/train.txt")
  for v in videos:
    render_bev_vs_gt_video(v, "heuristic")
    render_bev_vs_gt_video(v, "midas")