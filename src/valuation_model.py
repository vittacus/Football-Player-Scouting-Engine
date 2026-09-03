import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "player_features.csv"

VALUE_WEIGHTS = {
    "goals_per90": 2.0,
    "assists_per90": 1.5,
    "goal_contributions_per90": 1.0,
}

KEEPER_VALUE_WEIGHTS = {
    "save_pct": 2.0,
    "clean_sheet_pct": 1.5,
}
GOALS_AGAINST_PENALTY = 0.5

AGE_PENALTY_PER_YEAR = 0.02
PEAK_AGE = 24

def load_features(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

def load_keeper_features(path: Path = None) -> pd.DataFrame:
    if path is None:
        path = Path(__file__).parent.parent / "data" / "processed" / "keeper_features.csv"
    return pd.read_csv(path)

def compute_value_score(df: pd.DataFrame, weights: dict = VALUE_WEIGHTS, peak_age: int = PEAK_AGE, age_penalty: float = AGE_PENALTY_PER_YEAR) -> pd.DataFrame:
    df = df.copy()

    raw_score = sum(df[col] * weight for col, weight in weights.items())
    years_past_peak = (df["age"] - peak_age).clip(lower=0)
    penalty = years_past_peak * age_penalty

    df["value_score"] = (raw_score - penalty).round(3)

    return df.sort_values("value_score", ascending=False)

#Dividing by 100 to keep scores similar to that of the attackers score
def compute_keeper_value_score(df: pd.DataFrame, weights: dict = KEEPER_VALUE_WEIGHTS, ga_penalty: float = GOALS_AGAINST_PENALTY) -> pd.DataFrame:
    df = df.copy()

    raw_score = sum((df[col] / 100) * weight for col, weight in weights.items())
    penalty = df["goals_against_per90"] * ga_penalty

    df["value_score"] = (raw_score - penalty).round(3)

    return df.sort_values("value_score", ascending=False)

if __name__ == "__main__":
    df = load_features()
    scored = compute_value_score(df)
    print(scored[["player", "team", "age", "goals_per90", "assists_per90", "value_score"]].head(10))

    keeper_df = load_keeper_features()
    keeper_scored = compute_keeper_value_score(keeper_df)
    print(keeper_scored[["player", "team", "save_pct", "clean_sheet_pct", "goals_against_per90", "value_score"]].head(10))