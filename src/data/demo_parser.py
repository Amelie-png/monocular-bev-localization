from demoparser2 import DemoParser
import pandas as pd

FIELDS = [
  "X", "Y", "Z",
  "player_steamid",
  "player_name",
  "is_alive",
  "total_rounds_played", # Which round (0-indexed)
  "game_time", # Time in seconds since match start
  "is_freeze_period", # Is it freeze time
  "pitch",
  "yaw"
]

def parse_demo(demo_path):
  """
  Parse CS2 demo file and extract player positions per tick.

  Args:
    demo_path: path to .dem file

  Return:
    DataFrame with columns of FIELDS
  """
  parser = DemoParser(str(demo_path))

  df = parser.parse_ticks(FIELDS)
  df = df.rename(columns={
    "total_rounds_played": "round_number", 
    "X": "x", "Y": "y", "Z": "z",
  })

  return df.reset_index(drop=True)

def clean_demo_data(df):
  """
  Clean and filter parsed demo data.
  Keep active players and rounds.
  Rename columns for clarity.
  """

  filtered = df.loc[df["is_alive"] & ~df["is_freeze_period"]]

  df = df.rename(columns={
    "total_rounds_played": "round_number",
    "X": "x", "Y": "y", "Z": "z",
  })

  columns = [
    "tick",
    "round_number",
    "game_time",
    "player_steamid",
    "player_name",
    "x", "y", "z",
    "pitch",
    "yaw"
  ]

  return filtered[columns].reset_index(drop=True)

def generate_recording_plan(df):
  """
  Extract metadata and build recording plan given round number.

  Args:
    df: parsed demo DataFrame

  Return:
    dict with round_number, player_pov, start_tick, end_tick, duration info
  """
  recording_plan = []

  for round_num, round_df in df.groupby("round_number"):
    active_df = round_df.loc[~round_df["is_freeze_period"]]

    if active_df.empty:
      continue

    start_tick = active_df["tick"].min()
    end_tick = active_df["tick"].max()
    round_ticks = active_df["tick"].nunique()

    start_time = active_df['game_time'].min()
    end_time = active_df['game_time'].max()

    player_scores=[]

    for player, player_df in round_df.groupby("player_name"):
      alive_ticks = player_df[player_df["is_alive"]]["tick"].nunique()

      survival_ratio = (alive_ticks / round_ticks)

      player_scores.append({
        "player": player,
        "survival_ratio": survival_ratio
      })

    scores_df = pd.DataFrame(player_scores)

    best_player = scores_df.sort_values("survival_ratio", ascending=False).iloc[0]

    recording_plan.append({
      "round_number": int(round_num),
      "player_pov": best_player["player"],
      "start_tick": int(start_tick),
      "end_tick": int(end_tick),
      "tick_count": int(end_tick - start_tick),
      "start_game_time": float(start_time),
      "end_game_time": float(end_time),
      "duration_seconds": float(end_time - start_time)
    })

  return pd.DataFrame(recording_plan)