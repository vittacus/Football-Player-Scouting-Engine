import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


PROCESSED_DATA_PATH = Path(__file__).parent.parent / "data" / "processed" / "player_features.csv"

# These are the stats that define a player's profile
SIMILARITY_FEATURES = ["age", "goals_per90", "assists_per90", "goal_contributions_per90"]

def load_features(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)

# Scaling so no stat dominates the similarity features
def normalize_features(df: pd.DataFrame, feature_cols: list[str] = SIMILARITY_FEATURES) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled_values, columns=[f"{c}_scaled" for c in feature_cols], index=df.index)
    return pd.concat([df, scaled_df], axis=1), scaler

# Finding players most statistically similar to the given player_name
def find_similar_players(df: pd.DataFrame, player_name: str, feature_cols: list[str] = SIMILARITY_FEATURES, top_n: int = 5) -> pd.DataFrame:
    scaled_cols = [f"{c}_scaled" for c in feature_cols]
    
    target_row = df[df["player"] == player_name]
    if target_row.empty:
        raise ValueError(f"Player '{player_name}' not found in dataset")
    
    target_vector = target_row[scaled_cols].values
    all_vectors = df[scaled_cols].values
    
    similarities = cosine_similarity(target_vector, all_vectors)[0]
    df = df.copy()
    df["similarity_score"] = similarities
    
    # Exclude the player themselves, sort by most similar
    results = df[df["player"] != player_name].sort_values("similarity_score", ascending=False)
    
    return results[["player", "team", "position", "age", "goals_per90", "assists_per90", "similarity_score"]].head(top_n)

if __name__ == "__main__":
    df = load_features()
    df, scaler = normalize_features(df)
    
    test_player = "Bukayo Saka"
    print(f"Players most similar to {test_player}:")
    print(find_similar_players(df, test_player))