import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "player_features.csv"

VALUE_WEIGHTS = {
    "goals_per90": 2.0,
    "assists_per90": 1.5,
    "goal_contributions_per90": 1.0,
}

# Taking peak age at 24 and decreasing value score each year after
AGE_PENALTY_PER_YEAR = 0.02
PEAK_AGE = 24

def load_features(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

# Computes an interpretable composite "value score" from weighted attacking
# stats, with a mild penalty for age past estimated peak.
def compute_value_score(df: pd.DataFrame, weights: dict = VALUE_WEIGHTS, peak_age: int = PEAK_AGE, age_penalty: float = AGE_PENALTY_PER_YEAR) -> pd.DataFrame:
    df = df.copy()

    raw_score = sum(df[col] * weight for col, weight in weights.items())
    years_past_peak = (df["age"] - peak_age).clip(lower=0)  # never a negative penalty for younger players
    penalty = years_past_peak * age_penalty

    df["value_score"] = (raw_score - penalty).round(3)

    return df.sort_values("value_score", ascending=False)

if __name__ == "__main__":
    df = load_features()
    scored = compute_value_score(df)
    print(scored[["player", "team", "age", "goals_per90", "assists_per90", "value_score"]].head(10))