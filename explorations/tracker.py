from deep_sort_realtime.deepsort_tracker import DeepSort

class PlayerTracker:
  """
  Multi-object tracker using DeepSORT
  Assigns persistent IDs to detections across frames.
  """
  def __init__(self, config):
    """
    Args:
      config: TrackConfig object
    """
    self.max_age = config.max_age
    self.n_init = config.n_init
    self.nn_budget = config.nn_budget
    self.tracker = DeepSort(embedder='mobilenet', max_age=self.max_age, n_init=self.n_init, nn_budget=self.nn_budget)
    self.frame_count = 0

  def update(self, frame, detections):
    """
    Update tracker with new detections.

    Args:
      frame: current frame (numpy array)
      detections: detections (List of dicts with bbox) in current frame given by YOLOv8

    Return:
      List of tracked objects with track_id
    """

    self.frame_count += 1

    detection_list = []

    for detection in detections:
      x1, y1, x2, y2 = map(int, detection["bbox"])
      confidence = float(detection["confidence"])

      w = x2 - x1
      h = y2 - y1

      detection_list.append(([x1, y1, w, h], float(confidence), "player"))

    tracks = self.tracker.update_tracks(detection_list, frame=frame)

    tracked_detections = []

    for track in tracks:
      if not track.is_confirmed():
        continue

      track_id = int(track.track_id)
      lrtb = track.to_ltrb()

      tracked_detections.append({
        "track_id": track_id,
        "bbox": lrtb,
        "confidence": getattr(track, "det_conf", None),
        "is_confirmed": track.is_confirmed()
      })

    return tracked_detections
  
  def reset(self):
    """
    Reset tracker for new video/sequence.
    """
    self.tracker = DeepSort(embedder="mobilenet", max_age=self.max_age, n_init=self.n_init, nn_budget=self.nn_budget)
    self.frame_count = 0