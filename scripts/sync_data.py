from src.data.synchronizer import synchronize

mapping = synchronize(frame_metadata_path, tick_data_path)

mapping.to_parquet(...)