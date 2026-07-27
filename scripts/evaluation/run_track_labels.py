from pathlib import Path
import pandas as pd

from src.evaluation import build_track_label_table

CANONICAL_ROSTER = {
  "iM", "makazze", "Aleksib", "wOnderful", "b1t",
  "ropz", "ZywOo", "apEX", "flameZ", "mezii",
}

LABEL_DIR = Path("data/track_labels")
TRACKING_DIR = Path("data/processed/trackings")


def load_raw_text(video_name):
  path = LABEL_DIR / f"{video_name}.txt"
  if not path.exists():
    raise FileNotFoundError(f"No label file for {video_name} at {path}")
  return path.read_text()


def validate_labels(out_df, raw_text, video_name):
  issues = []

  # Spot typos
  seen_names = set()
  for line in raw_text.strip().splitlines():
    line = line.strip()
    if not line:
      continue
    name = line.partition("—")[0].strip()
    seen_names.add(name)
  unknown_names = seen_names - CANONICAL_ROSTER - {"POV_INVALID", "DEAD_BODY"}
  if unknown_names:
    issues.append(f"Unrecognized name(s), check for typos: {sorted(unknown_names)}")

  # Spot missing players
  missing_players = CANONICAL_ROSTER - seen_names
  if missing_players:
    issues.append(f"Players with no line at all in this file: {sorted(missing_players)}")

  # Spot track id errors
  tracking_file = TRACKING_DIR / f"{video_name}_trackings.parquet"
  if tracking_file.exists():
    real_track_ids = set(pd.read_parquet(tracking_file)["track_id"].unique())
    labeled_ids = set(out_df["track_id"])
    phantom_ids = labeled_ids - real_track_ids
    if phantom_ids:
      issues.append(f"Labeled track_id(s) not found in tracking output: {sorted(phantom_ids)}")

    # Spot unlabeled ids
    unlabeled_ids = real_track_ids - labeled_ids - {-1}
    if unlabeled_ids:
      issues.append(f"{len(unlabeled_ids)} track_id(s) in tracking output have no label at all: "
                     f"{sorted(unlabeled_ids)[:15]}{'...' if len(unlabeled_ids) > 15 else ''}")
  else:
    issues.append(f"No tracking file found at {tracking_file} — can't cross-check track_ids")

  # Spot ambiguous / conflicting labels
  ambiguous = out_df[out_df["status"].isin(["ambiguous", "ambiguous_invalids"])]
  if not ambiguous.empty:
    issues.append(f"{len(ambiguous)} ambiguous track_id(s) excluded from eval: "
                   f"{sorted(ambiguous['track_id'].tolist())}")

  return issues


def run_all(video_names):
  all_issues = {}
  for video_name in video_names:
    print(f"\n{'='*60}\n{video_name}\n{'='*60}")
    raw_text = load_raw_text(video_name)
    out_df = build_track_label_table(raw_text, video_name)

    issues = validate_labels(out_df, raw_text, video_name)
    if issues:
      print(f"\n[{video_name}] WARNINGS:")
      for i in issues:
        print(f"  - {i}")
      all_issues[video_name] = issues
    else:
      print(f"[{video_name}] No issues found.")

  print(f"\n{'='*60}\nSummary: {len(all_issues)}/{len(video_names)} video(s) with warnings\n{'='*60}")
  return all_issues


if __name__ == "__main__":
  from src.utils import load_split_file
  videos = load_split_file("data/splits/train.txt")
  run_all(videos)