def compute_fit_scale(df, canvas_size=800, padding_ratio=0.15):
  """Scale that fits a dataframe's bev_x/bev_y range into the canvas."""
  max_extent = max(df["bev_x"].abs().max(), df["bev_y"].abs().max(), 1e-6)
  return (canvas_size / 2) / (max_extent * (1 + padding_ratio))