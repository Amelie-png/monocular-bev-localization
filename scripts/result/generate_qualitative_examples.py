import cv2
import pandas as pd
from pathlib import Path

def select_examples(breakdown_df, condition, n_worst=3, n_contrast=1):
  """
  breakdown_df: output of analyze_failure_modes.py, has frame_id, video_name,
    euclidean_error, occluded/rapid_camera/blurry columns
  Returns a list of (frame_id, video_name, euclidean_error, role) tuples.
  """
  cond_true = breakdown_df[breakdown_df[condition]]

  worst = cond_true.sort_values("euclidean_error", ascending=False).head(n_worst).assign(role=f"{condition}_worst")

  contrast = cond_true.sort_values("euclidean_error", ascending=True).head(n_contrast).assign(role=f"{condition}_low_error_despite_condition")

  selected = pd.concat([worst, contrast], ignore_index=True)
  return list(zip(selected["frame_id"], selected["video_name"], selected["euclidean_error"], selected["role"]))


def draw_annotated_frame(
    video_name, frame_id, euclidean_error, role,
    tracking_dir="data/processed/trackings",
    out_dir="outputs/qualitative"):
  tracking_df = pd.read_parquet(Path(tracking_dir) / f"{video_name}_trackings.parquet")
  frame_rows = tracking_df[tracking_df["frame_id"] == frame_id]
  if frame_rows.empty:
    print(f"No tracking rows for {video_name} frame {frame_id}, skipping")
    return None

  frame_path = frame_rows.iloc[0]["frame_path"]
  img = cv2.imread(frame_path)
  if img is None:
    print(f"Could not load {frame_path}")
    return None

  for _, row in frame_rows.iterrows():
    x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"id:{row['track_id']}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

  label = f"{role} | frame {frame_id} | error={euclidean_error:.1f}"
  cv2.putText(img, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / f"{video_name}_frame{frame_id:06d}_{role}.png"
  cv2.imwrite(str(out_path), img)
  print(f"Saved {out_path}")
  return out_path


def generate_all_examples(variant="heuristic", eval_dir="outputs/eval"):
  breakdown_path = Path(eval_dir) / f"{variant}_failure_mode_breakdown.parquet"
  breakdown_df = pd.read_parquet(breakdown_path)

  conditions = ["occluded", "rapid_camera", "blurry"]
  saved = []
  for cond in conditions:
    examples = select_examples(breakdown_df, cond)
    for frame_id, video_name, err, role in examples:
      path = draw_annotated_frame(video_name, frame_id, err, role)
      if path:
        saved.append(path)

  anchors = [
    ("match_2_round_2", "best_overall"),
    ("match_3_round_2", "midas_worst_round"),
  ]
  for video_name, role in anchors:
    sub = breakdown_df[breakdown_df["video_name"] == video_name]
    if sub.empty:
      continue
    row = sub.loc[sub["euclidean_error"].idxmax()] if "worst" in role else sub.loc[sub["euclidean_error"].idxmin()]
    path = draw_annotated_frame(video_name, row["frame_id"], row["euclidean_error"], role)
    if path:
      saved.append(path)

  return saved


if __name__ == "__main__":
  generate_all_examples(variant="heuristic")
  generate_all_examples(variant="midas")