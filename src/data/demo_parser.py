from demoparser2 import DemoParser

def parse_demo(demo_path):
  """
  Parse CS2 demo file and extract player positions per tick.

  Args:
    demo_path: path to .dem file

  Return:
    DataFrame with columns: tick, player_name, player_steamid, X, Y, Z, 
    is_alive, team_name, round_number, game_time
  """
  parser = DemoParser(str(demo_path))

  fields = [
    "X", "Y", "Z",
    "player_steamid",
    "player_name",
    "is_alive",
    "team_name",
    "total_rounds_played",  # Which round (0-indexed)
    "game_time",            # Time in seconds since match start
    "round_in_progress",    # Is round active
    "is_freeze_period"      # Is it freeze time
  ]

  df = parser.parse_ticks(fields)
  df = clean_demo_data(df)

  return df

def clean_demo_data(df):
  """
  Clean and filter parsed demo data.
  Keep active players and rounds.
  Rename columns for clarity.
  """

  df = df[
    (df['is_alive'] == True) &
    (df['is_freeze_period'] == False)
  ].copy()

  df = df.rename(columns={
    'total_rounds_played': 'round_number'
  })

  columns = [
    "tick",
    "round_number",
    "game_time",
    "player_steamid",
    "player_name",
    "team_name",
    "X", "Y", "Z"
  ]

  df = df[columns].reset_index(drop=True)

  return df

def get_round_info(df, round_number):
  """
  Extract metadata given round number.

  Args:
    df: parsed demo DataFrame
    round_number: which round (0-indexed)

  Return:
    dict with start_tick, end_tick, duration info
  """

  round_df = df[df['round_number'] == round_number]

  if len(round_df) == 0:
    raise ValueError(f"Round {round_number} not found in demo data")
  
  info = {
    'round_number': round_number,
    'start_tick': round_df['tick'].min(),
    'end_tick': round_df['tick'].max(),
    'tick_count': round_df['tick'].max() - round_df['tick'].min(),
    'start_game_time': round_df['game_time'].min(),
    'end_game_time': round_df['game_time'].max(),
    'duration_seconds': round_df['game_time'].max() - round_df['game_time'].min()
  }
    
  return info