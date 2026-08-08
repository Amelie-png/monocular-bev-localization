import pandas as pd

def build_frame_detections(df, class_name="person"):
  """
  Groups a detections/trackings dataframe by frame_id and returns
  (frame_paths, detections_list) matching DetectionVisualizer's input contract.
  """
  frame_paths, detections_list = [], []
  for frame_id in sorted(df["frame_id"].unique()):
    g = df[df["frame_id"] == frame_id]
    frame_paths.append(g.iloc[0]["frame_path"])
    dets = []
    for _, row in g.iterrows():
      d = {
        "bbox": [row["x1"], row["y1"], row["x2"], row["y2"]],
        "confidence": row["confidence"],
        "class_name": class_name,
      }
      if "track_id" in row.index and pd.notna(row["track_id"]):
        d["track_id"] = row["track_id"]
      dets.append(d)
    detections_list.append(dets)
  return frame_paths, detections_list