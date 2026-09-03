import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "player_season_stats.csv"
RAW_KEEPER_PATH = Path(__file__).parent.parent / "data" / "raw" / "player_keeper_stats.csv"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Looking at every 10 matches (10 * 90 minutes)
MIN_MINUTES = 900
MIN_KEEPER_STARTS = 10

# Load player stats and filter any players with minimum minutes
def load_and_filter_data(path: Path = RAW_DATA_PATH, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
   
    df = pd.read_csv(path)
    
    before_count = len(df)
    df = df[df["Playing Time_Min"] >= min_minutes].copy()
    after_count = len(df)
    
    print(f"Filtered {before_count} -> {after_count} players (min {min_minutes} minutes)")
    
    return df

def load_and_filter_keepers(path: Path = RAW_KEEPER_PATH, min_starts: int = MIN_KEEPER_STARTS) -> pd.DataFrame:
    df = pd.read_csv(path)

    before_count = len(df)
    df = df[df["Playing Time_Starts"] >= min_starts].copy()
    after_count = len(df)

    print(f"Filtered {before_count} -> {after_count} keepers (min {min_starts} starts)")

    return df


FEATURE_COLUMN_MAP = {
    "player": "player",
    "team": "team",
    "season": "season",
    "pos": "position",
    "age": "age",
    "Playing Time_Min": "minutes",
    "Per 90 Minutes_Gls": "goals_per90",
    "Per 90 Minutes_Ast": "assists_per90",
    "Per 90 Minutes_G+A": "goal_contributions_per90",
}

KEEPER_FEATURE_COLUMN_MAP = {
    "player": "player",
    "team": "team",
    "season": "season",
    "pos": "position",
    "age": "age",
    "Playing Time_Starts": "starts",
    "Performance_Save%": "save_pct",
    "Performance_CS%": "clean_sheet_pct",
    "Performance_GA90": "goals_against_per90",
}

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[list(FEATURE_COLUMN_MAP.keys())].rename(columns=FEATURE_COLUMN_MAP)
    return features

def select_keeper_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df[list(KEEPER_FEATURE_COLUMN_MAP.keys())].rename(columns=KEEPER_FEATURE_COLUMN_MAP)
    return features

if __name__ == "__main__":
    df = load_and_filter_data()
    features = select_features(df)
    output_path = PROCESSED_DATA_DIR / "player_features.csv"
    features.to_csv(output_path, index=False)
    print(f"Saved {len(features)} rows to {output_path}")

    keeper_df = load_and_filter_keepers()
    keeper_features = select_keeper_features(keeper_df)
    keeper_output_path = PROCESSED_DATA_DIR / "keeper_features.csv"
    keeper_features.to_csv(keeper_output_path, index=False)
    print(f"Saved {len(keeper_features)} rows to {keeper_output_path}")
    print(keeper_features.describe())