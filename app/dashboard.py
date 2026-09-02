import streamlit as st
import sys
from pathlib import Path

# Allow importing from src/
sys.path.append(str(Path(__file__).parent.parent))

from src.similarity_model import load_features as load_similarity_features, normalize_features, find_similar_players
from src.valuation_model import load_features as load_valuation_features, compute_value_score

st.set_page_config(page_title="Player Scouting Engine", layout="wide")
st.title("Player Scouting Engine")
st.caption("Find statistically similar players and compare attacking value scores")

@st.cache_data
def get_similarity_data():
    df = load_similarity_features()
    df, _ = normalize_features(df)
    return df


@st.cache_data
def get_valuation_data():
    df = load_valuation_features()
    return compute_value_score(df)


sim_df = get_similarity_data()
val_df = get_valuation_data()

# Since we are looking at goals/assists, looking mainly at attacking positions
ATTACKING_POSITIONS = ["FW", "MF", "FW,MF", "MF,FW"]
sim_df_attacking = sim_df[sim_df["position"].isin(ATTACKING_POSITIONS)]

player_list = sorted(sim_df["player"].unique())
selected_player = st.selectbox("Search for a player", player_list)

if selected_player:
    st.subheader(f"Players similar to {selected_player}")
    similar = find_similar_players(sim_df_attacking, selected_player)
    st.dataframe(similar, width="stretch")

st.title("Player Scouting Engine")
st.caption("Find statistically similar players and compare attacking value scores")
st.caption("Currently scoped to attacking players (forwards & midfielders) — defenders and goalkeepers aren't comparable on these stats yet.")