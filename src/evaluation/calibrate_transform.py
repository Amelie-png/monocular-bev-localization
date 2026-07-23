import numpy as np

def rotate(bev_x, bev_y, yaw_deg, convention):
  yaw = np.radians(yaw_deg)
  c, s = np.cos(yaw), np.sin(yaw)
  if convention == "fwd_cos_sin":
    return bev_y * c - bev_x * s, bev_y * s + bev_x * c
  elif convention == "fwd_sin_cos":
    return bev_y * s + bev_x * c, bev_y * c - bev_x * s
  elif convention == "fwd_cos_sin_negx":
    return bev_y * c + bev_x * s, bev_y * s - bev_x * c
  elif convention == "fwd_sin_cos_negx":
    return bev_y * s - bev_x * c, bev_y * c + bev_x * s
  raise ValueError(convention)

def fit_scale(dx, dy, target_dx, target_dy):
  """
  1D least-squares scale minimizing ||s*(dx,dy) - (target_dx,target_dy)||.
  """
  num = (dx * target_dx + dy * target_dy).sum()
  denom = (dx * dx + dy * dy).sum()
  return num / denom if denom > 0 else 1.0

def calibrate_convention(calib_df):
  """
  Pick the best rotation convention using pooled data across videos.
  calib_df needs: bev_x, bev_y, cam_x, cam_y, yaw_deg, gt_x, gt_y
  Convention should generalize across videos even though scale doesn't.
  """
  conventions = ["fwd_cos_sin", "fwd_sin_cos", "fwd_cos_sin_negx", "fwd_sin_cos_negx"]
  results = {}
  for conv in conventions:
    dx, dy = rotate(calib_df["bev_x"].values, calib_df["bev_y"].values, calib_df["yaw_deg"].values, conv)
    target_dx = calib_df["gt_x"].values - calib_df["cam_x"].values
    target_dy = calib_df["gt_y"].values - calib_df["cam_y"].values
    s = fit_scale(dx, dy, target_dx, target_dy)
    pred_x = calib_df["cam_x"].values + s * dx
    pred_y = calib_df["cam_y"].values + s * dy
    err = np.hypot(pred_x - calib_df["gt_x"].values, pred_y - calib_df["gt_y"].values)
    results[conv] = {"scale": s, "mean_error": err.mean()}

  best = min(results, key=lambda k: results[k]["mean_error"])
  print("Convention results:", results)
  return best

def fit_scale_for_video(calib_df, convention):
  """
  Refit scale per video (scale might vary round to round).
  """
  dx, dy = rotate(calib_df["bev_x"].values, calib_df["bev_y"].values, calib_df["yaw_deg"].values, convention)
  target_dx = calib_df["gt_x"].values - calib_df["cam_x"].values
  target_dy = calib_df["gt_y"].values - calib_df["cam_y"].values
  return fit_scale(dx, dy, target_dx, target_dy)

def make_transform(convention, scale):
  def transform_fn(bev_x, bev_y, cam_x, cam_y, yaw_deg):
    dx, dy = rotate(bev_x, bev_y, yaw_deg, convention)
    return cam_x + scale * dx, cam_y + scale * dy
  return transform_fn