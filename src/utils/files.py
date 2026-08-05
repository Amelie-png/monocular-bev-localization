from pathlib import Path

def filter_missing(video_names, expected_path_fcn, force=False):
  """
  Filter list of video names provided to return those that are not yet processed. Return all video names if force flag is true.

  Args:
    video_names: List of video names to filter
    expected_path_fcn: callable(video_name) -> Path or list[Path]
      Returns the output path(s) that must exist for this video to be considered 'done' for this step.
    force: Boolean for force restart (ie return all videos)

  Return:
    List of video names to process
  """
  if force:
    return list(video_names)
  
  todo = []
  for video in video_names:
    paths = expected_path_fcn(video)
    if isinstance(paths, (str, Path)):
      paths = [paths]
    if not all(Path(p).exists() for p in paths):
      todo.append(video)

  return todo

def report_skip(step_name, all_videos, todo_videos):
  """
  Report filter results.

  Args:
    step_name: step that is filtered
    all_videos: all videos processed
    todo_videos: filtered videos
  """
  skipped = set(all_videos) - set(todo_videos)
  if skipped:
    print(f"[{step_name}] Skipping {len(skipped)} already-done video(s): {sorted(skipped)}")
  if todo_videos:
    print(f"[{step_name}] Running on {len(todo_videos)} video(s): {sorted(todo_videos)}")
  else:
    print(f"[{step_name}] Nothing to do.")

def mark_done(done_path):
  """
  Touch a marker file, creating parent dirs if needed.

  Args:
    done_path: path to create marker file
  """
  done_path = Path(done_path)
  done_path.parent.mkdir(parents=True, exist_ok=True)
  done_path.touch()

def is_done(done_path):
  """
  Check for done_path.

  Args:
    done_path: path to maker file
  
  Return:
    True if file is found, false otherwise
  """
  return Path(done_path).exists()

def load_split_file(split_file):
  """
  Load video names from split file.

  Args:
    split_file: plain text file containing videos names to split

  Return:
    List of videos from split_file
  """
  with open(split_file) as f:
    return [line.strip() for line in f if line.strip()]