from omegaconf import OmegaConf
from types import SimpleNamespace
from pathlib import Path

from src.detection import DetectionConfig
from src.tracking import ByteTrackConfig
from src.bev import BevConfig

STEP_CONFIG_CLASSES = {
  "detection": DetectionConfig,
  "tracking": ByteTrackConfig,
  "heuristic_bev": BevConfig,
  "midas_bev": BevConfig,
}

def load_step_config(step_name, config_dir="configs"):
  """
  Load config for any step. Start from dataclass defaults, merge tuned YAML if present. Materialize back into a
  real dataclass instance.

  Args:
    step_name: String of step name to config
    config_dir: Directory of configs

  Return:
    Dataclass instance of merged configs
  """
  config_cls = STEP_CONFIG_CLASSES[step_name]
  schema = OmegaConf.structured(config_cls)
  
  yaml_path = Path(config_dir) / f"{step_name}.yaml"
  if yaml_path.exists():
    overrides = OmegaConf.load(yaml_path)
    merged = OmegaConf.merge(schema, overrides)
  else:
    print(f"No config found for step '{step_name}' at {yaml_path}, using defaults")
    merged = schema
  
  return OmegaConf.to_object(merged)

def load_pipeline_config(config_dir="configs"):
  """
  Build pipeline config object with one attribute per step.
  
  Args:
    config_dir: Directory of configs
  
  Return:
    Config object with dataclass instance as attribute
  """
  cfg = {name: load_step_config(name, config_dir) for name in STEP_CONFIG_CLASSES}
  return SimpleNamespace(**cfg)