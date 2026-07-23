import pandas as pd
from pathlib import Path

LABEL_MAP = {
  "POV_INVALID": "invalid_pov",
  "DEAD_BODY": "dead_body",
}

def parse_manual_labels(raw_text):
  rows = []
  for line in raw_text.strip().splitlines():
    line = line.strip()
    if not line:
      continue
    name, _, ids_str = line.partition("—")
    name = name.strip()
    ids_str = ids_str.strip().rstrip(",")
    if not ids_str:
      continue  # player never appeared on screen this round
    ids = [int(x.strip()) for x in ids_str.split(",") if x.strip()]
    label_status = LABEL_MAP.get(name)
    for tid in ids:
      rows.append({"track_id": tid, "player_name": name, "label_status": label_status})
  return pd.DataFrame(rows)


def build_track_label_table(raw_text, video_name):
  df = parse_manual_labels(raw_text)

  rows = []
  for track_id, group in df.groupby("track_id"):
    invalids = group["label_status"].dropna().unique().tolist()
    players = sorted(group.loc[group["label_status"].isna(), "player_name"].unique().tolist())
    if invalids:
      status = invalids[0] if len(invalids) == 1 else "ambiguous_invalids"
      rows.append({"video_name": video_name, "track_id": track_id, "status": status, "player_name": None})
    elif len(players) > 1:
      rows.append({"video_name": video_name, "track_id": track_id, "status": "ambiguous", "player_name": None})
    else:
      rows.append({"video_name": video_name, "track_id": track_id, "status": "valid", "player_name": players[0]})

  out = pd.DataFrame(rows)
  out_path = Path(f"data/processed/track_labels/{video_name}_track_labels.parquet")
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out.to_parquet(out_path, index=False)
  print(f"Saved {len(out)} track labels to {out_path}")
  print(out["status"].value_counts())
  return out