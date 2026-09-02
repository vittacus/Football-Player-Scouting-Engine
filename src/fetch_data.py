import pandas as pd
import soccerdata as sd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

#Starting with the Premier League, will expand after
LEAGUE = "ENG-Premier League"
SEASONS = ["2425", "2526"]

# Pulling plyare stats (goals, assists, minutes, etc.) from FBref, returning as DataFrame 
def fetch_season_stats(league: str, seasons: list[str]) -> pd.DataFrame:
    fbref = sd.FBref(leagues=league, seasons=seasons)
    player_stats = fbref.read_player_season_stats(stat_type="standard")
   
    player_stats.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in player_stats.columns
    ]
    return player_stats

if __name__ == "__main__":
    print(f"Fetching {LEAGUE} player stats for seasons: {SEASONS}...")
    
    stats = fetch_season_stats(LEAGUE, SEASONS)
    
    output_path = RAW_DATA_DIR / "player_season_stats.csv"
    stats.to_csv(output_path)
    
    print(f"Saved {len(stats)} rows to {output_path}")
    print(stats.head())