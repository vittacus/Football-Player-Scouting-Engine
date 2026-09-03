import streamlit as st
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Allow importing from src/
sys.path.append(str(Path(__file__).parent.parent))

from src.similarity_model import load_features as load_similarity_features, normalize_features, find_similar_players
from src.valuation_model import load_features as load_valuation_features, compute_value_score

st.set_page_config(page_title="Player Scouting Engine", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #0B0F1F; }
    .block-container { background-color: #0B0F1F; padding: 40px 48px !important; max-width: 1100px; }

    h1 {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #F2F4F8;
        border-bottom: 3px solid #6B2FA0;
        padding-bottom: 12px;
        display: inline-block;
    }

    .scoreboard {
        background-color: #14224A;
        border: 1px solid #6B2FA0;
        padding: 18px 32px 18px 24px;
        margin: 20px 0 32px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .scoreboard-club {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 22px;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: #F2F4F8;
    }
    .scoreboard-sub {
        font-size: 13px;
        color: #A8B4D9;
        margin-top: 2px;
    }
    .scoreboard-value {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 30px;
        color: #6EC8FF;
        text-align: right;
        line-height: 1;
    }
    .scoreboard-value-label {
        font-size: 11px;
        color: #A8B4D9;
        text-align: right;
    }

    .stat-panel { padding: 4px 0 4px 16px; }
    .stat-panel.reference { border-left: 3px solid #6B2FA0; }
    .stat-panel.target { border-left: 3px solid #6EC8FF; }
    .stat-panel-title {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 19px;
        color: #F2F4F8;
    }
    .stat-panel.target .stat-panel-title { color: #6EC8FF; }
    .stat-panel-sub {
        font-size: 12px;
        color: #8C9BC4;
        margin-bottom: 12px;
    }
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid rgba(242,244,248,0.08);
        font-size: 14px;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-label { color: #8C9BC4; }
    .stat-value { font-family: 'Oswald', sans-serif; color: #F2F4F8; }

    .similarity-line {
        text-align: center;
        font-size: 13px;
        color: #8C9BC4;
        margin: 20px 0;
        padding-top: 16px;
        border-top: 1px solid rgba(242,244,248,0.1);
    }
    .similarity-line strong { color: #6EC8FF; font-family: 'Oswald', sans-serif; }

    [data-testid="stExpander"] {
        background-color: transparent;
        border: none !important;
        border-bottom: 1px solid rgba(242,244,248,0.1) !important;
        border-radius: 0;
    }

    [data-testid="stExpander"]:focus-within {
        outline: 1px solid #6B2FA0 !important;
    }

    [data-baseweb="select"] > div {
        background-color: #14224A;
        border: 1px solid #6B2FA0 !important;
        border-radius: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_similarity_data():
    df = load_similarity_features()
    df, _ = normalize_features(df)
    return df


@st.cache_data
def get_valuation_data():
    df = load_valuation_features()
    return compute_value_score(df)


def render_stat_panel(title, subtitle, stats, variant):
    rows_html = "".join(
        f'<div class="stat-row"><span class="stat-label">{label}</span><span class="stat-value">{value}</span></div>'
        for label, value in stats
    )
    st.markdown(
        f"""
        <div class="stat-panel {variant}">
            <div class="stat-panel-title">{title}</div>
            <div class="stat-panel-sub">{subtitle}</div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


sim_df = get_similarity_data()
val_df = get_valuation_data()

ATTACKING_POSITIONS = ["FW", "MF", "FW,MF", "MF,FW"]
sim_df_attacking = sim_df[sim_df["position"].isin(ATTACKING_POSITIONS)]
sim_df_attacking = (
    sim_df_attacking.sort_values("season")
    .drop_duplicates("player", keep="last")
)

val_df_latest = val_df.sort_values("season").drop_duplicates("player", keep="last")

ROLE_POSITIONS = {
    "Any role": None,
    "Striker": ["FW"],
    "Winger": ["FW,MF", "MF,FW"],
    "Attacking midfielder": ["MF"],
    "Box-to-box midfielder": ["MF"],
}

st.title("Player Scouting Engine")
st.caption("Find attacking signings that fit your club's needs")

with st.expander("About this project"):
    st.markdown(
        """
        This tool compares attacking players (forwards & midfielders) across the
        Premier League using two models:

        - **Similarity search**, using cosine similarity on scaled per-90 stats
          (age, goals, assists, goal contributions) to find players with a
          statistically similar attacking profile to a given reference player.
        - **Value score**, a weighted composite of attacking output with a mild
          age penalty past estimated peak. This was built after an earlier regression
          approach (predicting playing time from attacking stats) showed no
          meaningful relationship in the data.

        Defenders and goalkeepers aren't included, since this feature set only
        meaningfully differentiates attacking output. The role filter maps common
        scouting language onto broad position tags. It's a heuristic, not a
        precise tactical classification.

        **A note on the value score:** it isn't a market price or a percentage, but
        a weighted combination of goals, assists, and goal contributions per 90
        minutes (with a small penalty for age past 24). It only means something in
        *comparison* to other players in this tool. Use it to rank and compare, not
        as a standalone number.
        """
    )

col_a, col_b, col_c = st.columns(3)
with col_a:
    team_list = sorted(sim_df_attacking["team"].unique())
    selected_team = st.selectbox("Your club", team_list)

with col_b:
    team_players = sorted(sim_df_attacking[sim_df_attacking["team"] == selected_team]["player"].unique())
    if not team_players:
        st.warning(f"No attacking players found for {selected_team}")
        st.stop()
    selected_player = st.selectbox("Compare against which player?", team_players)

with col_c:
    selected_role = st.selectbox("Target role", list(ROLE_POSITIONS.keys()))

if selected_player:
    selected_value = val_df_latest[val_df_latest["player"] == selected_player]

    similar = find_similar_players(sim_df_attacking, selected_player)
    similar = similar[similar["team"] != selected_team]

    similar_with_value = similar.merge(
        val_df_latest[["player", "value_score"]], on="player", how="left"
    )

    role_positions = ROLE_POSITIONS[selected_role]
    if role_positions is not None:
        filtered = similar_with_value[similar_with_value["position"].isin(role_positions)]
        if filtered.empty:
            st.info(f"No targets match the {selected_role} role -- showing all targets instead.")
        else:
            similar_with_value = filtered

    similar_with_value = similar_with_value.sort_values("value_score", ascending=False).reset_index(drop=True)

    value_display = f"{selected_value['value_score'].values[0]:.2f}" if not selected_value.empty else "—"

    st.markdown(
        f"""
        <div class="scoreboard">
            <div>
                <div class="scoreboard-club">{selected_team}</div>
                <div class="scoreboard-sub">Scouting report — benchmarked against {selected_player}</div>
            </div>
            <div>
                <div class="scoreboard-value">{value_display}</div>
                <div class="scoreboard-value-label">VALUE SCORE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Head-to-head comparison")

    option_labels = [
        f"{i+1:02d} · {row['player']} — Value {row['value_score']:.2f}"
        for i, row in similar_with_value.iterrows()
    ]
    label_to_player = dict(zip(option_labels, similar_with_value["player"]))

    selected_label = st.selectbox("Compare " + selected_player + " against:", option_labels)
    compare_target = label_to_player[selected_label]

    reference_row = sim_df_attacking[sim_df_attacking["player"] == selected_player].iloc[0]
    reference_value = val_df_latest[val_df_latest["player"] == selected_player]
    reference_value_display = f"{reference_value['value_score'].values[0]:.2f}" if not reference_value.empty else "—"

    target_row = sim_df_attacking[sim_df_attacking["player"] == compare_target].iloc[0]
    target_value_row = similar_with_value[similar_with_value["player"] == compare_target].iloc[0]

    col_left, col_right = st.columns(2)
    with col_left:
        render_stat_panel(
            title=selected_player,
            subtitle=f"{selected_team} · {reference_row['position']}",
            stats=[
                ("Age", int(reference_row["age"])),
                ("Goals /90", f"{reference_row['goals_per90']:.2f}"),
                ("Assists /90", f"{reference_row['assists_per90']:.2f}"),
                ("Goal Contributions /90", f"{reference_row['goal_contributions_per90']:.2f}"),
                ("Value score", reference_value_display),
            ],
            variant="reference",
        )
    with col_right:
        render_stat_panel(
            title=compare_target,
            subtitle=f"{target_row['team']} · {target_row['position']}",
            stats=[
                ("Age", int(target_row["age"])),
                ("Goals /90", f"{target_row['goals_per90']:.2f}"),
                ("Assists /90", f"{target_row['assists_per90']:.2f}"),
                ("Goal Contributions /90", f"{target_row['goal_contributions_per90']:.2f}"),
                ("Value score", f"{target_value_row['value_score']:.2f}"),
            ],
            variant="target",
        )

    st.markdown(
        f"""
        <div class="similarity-line">
            Statistical similarity between {selected_player} and {compare_target}:
            <strong>{target_value_row['similarity_score']:.3f}</strong> (1.0 = identical attacking profile)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Value comparison")
    st.caption(f"How {compare_target} stacks up against the other {len(similar_with_value)} targets")

    bar_colors = [
        "#A8D8FF" if player == compare_target else "#6EC8FF"
        for player in similar_with_value["player"]
    ]
    fig = go.Figure(go.Bar(
        x=similar_with_value["value_score"],
        y=similar_with_value["player"],
        orientation="h",
        marker_color=bar_colors,
        text=similar_with_value["value_score"].round(2),
        textposition="outside",
        textfont=dict(color="#F2F4F8"),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F2F4F8",
        xaxis=dict(gridcolor="rgba(242,244,248,0.1)", tickfont=dict(color="#8C9BC4"), title="Value score"),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#F2F4F8")),
        margin=dict(l=0, r=40, t=10, b=10),
        height=max(220, 60 * len(similar_with_value)),
    )
    st.plotly_chart(fig, use_container_width=True)