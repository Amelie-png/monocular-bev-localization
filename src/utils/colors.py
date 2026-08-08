import numpy as np

def track_color(track_id):
  """Deterministic BGR color per track_id so same player gets the same
  color across every visualization (BEV, trajectory, tracking video)."""
  rng = np.random.RandomState(int(track_id) * 9973 % (2**31))
  color = rng.randint(60, 230, size=3)
  return tuple(int(c) for c in color)