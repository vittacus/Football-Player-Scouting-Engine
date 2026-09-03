import streamlit as st
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Allow importing from src/
sys.path.append(str(Path(__file__).parent.parent))

from src.similarity_model import (
    load_features as load_similarity_features,
    load_keeper_features as load_similarity_keeper_features,
    normalize_features,
    find_similar_players,
    KEEPER_SIMILARITY_FEATURES,
)
from src.valuation_model import (
    load_features as load_valuation_features,
    load_keeper_features as load_valuation_keeper_features,
    compute_value_score,
    compute_keeper_value_score,
)

st.set_page_config(page_title="Player Scouting Engine", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap');

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #E9E4E0 !important;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .block-container { background-color: #E9E4E0; padding: 40px 48px !important; max-width: 1100px; }

    h1, h2, h3, p, span, label, div { color: #172A39; }

    .title-row { display: flex; align-items: center; gap: 16px; margin-bottom: 0; }
    h1 {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 0 !important;
    }
    .pl-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.08em;
        color: #FC563C;
        border: 1px solid #FC563C;
        padding: 4px 10px;
        text-transform: uppercase;
    }
    .title-underline {
        height: 3px;
        background-color: #FC563C;
        margin: 12px 0 20px 0;
    }

    /* Uppercase, tracked, monospace micro-labels on every form field */
    [data-testid="stWidgetLabel"] p {
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 12px !important;
        color: #6E7575 !important;
    }

    .scoreboard {
        background-color: #172A39;
        border-radius: 0;
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
        color: #E9E4E0 !important;
    }
    .scoreboard-sub { font-size: 13px; color: #B8BEC2 !important; margin-top: 2px; }
    .scoreboard-value {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 30px;
        color: #FC563C !important;
        text-align: right;
        line-height: 1;
    }
    .scoreboard-value-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.06em;
        color: #B8BEC2 !important;
        text-align: right;
    }

    /* Cards: white background, top accent border (not left), matching the mockup */
    .stat-panel {
        background-color: #FFFFFF;
        border-top: 3px solid #172A39;
        padding: 20px 24px;
    }
    .stat-panel.target { border-top: 3px solid #FC563C; }
    .stat-panel-title { font-family: 'Oswald', sans-serif; font-weight: 700; font-size: 22px; color: #172A39; }
    .stat-panel.target .stat-panel-title { color: #FC563C !important; }
    .stat-panel-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.04em;
        color: #6E7575;
        margin-bottom: 14px;
    }
    .stat-row {
        display: flex; justify-content: space-between; padding: 8px 0;
        border-bottom: 1px solid rgba(23,42,57,0.1); font-size: 14px;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-label { color: #6E7575; }
    .stat-value { font-family: 'Oswald', sans-serif; color: #172A39; }

    .similarity-line {
        text-align: center; font-size: 13px; color: #6E7575; margin: 20px 0;
        padding-top: 16px; border-top: 1px solid rgba(23,42,57,0.15);
    }
    .similarity-line strong { color: #FC563C !important; font-family: 'Oswald', sans-serif; }

    /* Chart wrapper: white card behind the plot area, like the mockup */
    .chart-card { background-color: #FFFFFF; padding: 20px; margin-top: 8px; }

    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid rgba(23,42,57,0.2) !important;
        border-radius: 0 !important;
    }
    [data-testid="stExpander"] summary {
        background-color: #FFFFFF !important;
        color: #172A39 !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 13px !important;
    }
    [data-testid="stExpander"] p { color: #172A39 !important; font-family: 'IBM Plex Sans', sans-serif !important; text-transform: none; letter-spacing: normal; }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #172A39 !important;
        border-radius: 0 !important;
    }
    div[data-baseweb="select"] > div * {
        color: #172A39 !important;
    }

    [role="listbox"] { background-color: #FFFFFF !important; }
    [role="option"] {
        background-color: #FFFFFF !important;
        color: #172A39 !important;
    }
    [role="option"]:hover { background-color: #E9E4E0 !important; }
    [role="option"] * { color: #172A39 !important; }

    /* Radio group styled as a bordered segmented control */
    [data-testid="stRadio"] > div {
        border: 1px solid #172A39;
        gap: 0 !important;
        display: inline-flex;
    }
    [data-testid="stRadio"] label {
        border-radius: 0 !important;
        padding: 8px 16px !important;
        margin: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 12px !important;
    }

    button, .stButton > button { border-radius: 0 !important; }
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


@st.cache_data
def get_keeper_similarity_data():
    df = load_similarity_keeper_features()
    df, _ = normalize_features(df, feature_cols=KEEPER_SIMILARITY_FEATURES)
    return df


@st.cache_data
def get_keeper_valuation_data():
    df = load_valuation_keeper_features()
    return compute_keeper_value_score(df)


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

st.markdown(
    """
    <div class="title-row">
        <h1>Player Scouting Engine</h1>
        <div class="pl-badge">Premier League</div>
    </div>
    <div class="title-underline"></div>
    """,
    unsafe_allow_html=True,
)
st.caption("Find attacking and goalkeeping signings that fit your club's needs")

with st.expander("About this project"):
    st.markdown(
        """
        This tool compares players across the Premier League using two models:

        - **Similarity search**, using cosine similarity on scaled per-90 stats
          to find players with a statistically similar profile to a given
          reference player. Attacking players are compared on goals, assists, and
          goal contributions per 90. Goalkeepers are compared on save percentage,
          clean sheet percentage, and goals against per 90.
        - **Value score**, a weighted composite of the relevant stats for each
          profile type. The attacking version was built after an earlier
          regression approach (predicting playing time from attacking stats)
          showed no meaningful relationship in the data.

        Defenders aren't included yet, since usable defensive stats weren't
        available from the current data source. The role filter (attackers only)
        maps common scouting language onto broad position tags. It's a heuristic,
        not a precise tactical classification.

        **A note on the value score:** it isn't a market price or a percentage,
        but a weighted combination of relevant stats for that profile type. It
        only means something in *comparison* to other players in this tool. Use
        it to rank and compare, not as a standalone number.
        """
    )

col_p, col_a, col_b, col_c = st.columns(4)

with col_p:
    profile_type = st.radio("Profile type", ["Attacker", "Goalkeeper"], horizontal=True)

if profile_type == "Attacker":
    sim_df_group = sim_df_attacking
    val_df_group = val_df_latest
    group_stat_labels = [
        ("Goals /90", "goals_per90"),
        ("Assists /90", "assists_per90"),
        ("Goal Contributions /90", "goal_contributions_per90"),
    ]
else:
    sim_df_group = get_keeper_similarity_data()
    val_df_group = get_keeper_valuation_data().sort_values("season").drop_duplicates("player", keep="last")
    group_stat_labels = [
        ("Save %", "save_pct"),
        ("Clean Sheet %", "clean_sheet_pct"),
        ("Goals Against /90", "goals_against_per90"),
    ]

with col_a:
    team_list = sorted(sim_df_group["team"].unique())
    selected_team = st.selectbox("Your club", team_list)

with col_b:
    team_players = sorted(sim_df_group[sim_df_group["team"] == selected_team]["player"].unique())
    if not team_players:
        st.warning(f"No {profile_type.lower()}s found for {selected_team}")
        st.stop()
    selected_player = st.selectbox("Compare against which player?", team_players)

selected_role = None
with col_c:
    if profile_type == "Attacker":
        selected_role = st.selectbox("Target role", list(ROLE_POSITIONS.keys()))

if selected_player:
    similarity_feature_cols = KEEPER_SIMILARITY_FEATURES if profile_type == "Goalkeeper" else None

    selected_value = val_df_group[val_df_group["player"] == selected_player]

    if similarity_feature_cols:
        similar = find_similar_players(sim_df_group, selected_player, feature_cols=similarity_feature_cols)
    else:
        similar = find_similar_players(sim_df_group, selected_player)
    similar = similar[similar["team"] != selected_team]

    similar_with_value = similar.merge(
        val_df_group[["player", "value_score"]], on="player", how="left"
    )

    if profile_type == "Attacker":
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

    reference_row = sim_df_group[sim_df_group["player"] == selected_player].iloc[0]
    reference_value = val_df_group[val_df_group["player"] == selected_player]
    reference_value_display = f"{reference_value['value_score'].values[0]:.2f}" if not reference_value.empty else "—"

    target_row = sim_df_group[sim_df_group["player"] == compare_target].iloc[0]
    target_value_row = similar_with_value[similar_with_value["player"] == compare_target].iloc[0]

    reference_stats = [("Age", int(reference_row["age"]))]
    for label, col in group_stat_labels:
        reference_stats.append((label, f"{reference_row[col]:.2f}"))
    reference_stats.append(("Value score", reference_value_display))

    target_stats = [("Age", int(target_row["age"]))]
    for label, col in group_stat_labels:
        target_stats.append((label, f"{target_row[col]:.2f}"))
    target_stats.append(("Value score", f"{target_value_row['value_score']:.2f}"))

    col_left, col_right = st.columns(2)
    with col_left:
        render_stat_panel(
            title=selected_player,
            subtitle=f"{selected_team} · {reference_row['position']}",
            stats=reference_stats,
            variant="reference",
        )
    with col_right:
        render_stat_panel(
            title=compare_target,
            subtitle=f"{target_row['team']} · {target_row['position']}",
            stats=target_stats,
            variant="target",
        )

    profile_label = "attacking" if profile_type == "Attacker" else "goalkeeping"
    st.markdown(
        f"""
        <div class="similarity-line">
            Statistical similarity between {selected_player} and {compare_target}:
            <strong>{target_value_row['similarity_score']:.3f}</strong> (1.0 = identical {profile_label} profile)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Value comparison")
    st.caption(f"How {compare_target} stacks up against the other {len(similar_with_value)} targets")

    bar_colors = [
        "#172A39" if player == compare_target else "#FC563C"
        for player in similar_with_value["player"]
    ]
    fig = go.Figure(go.Bar(
        x=similar_with_value["value_score"],
        y=similar_with_value["player"],
        orientation="h",
        marker_color=bar_colors,
        text=similar_with_value["value_score"].round(2),
        textposition="outside",
        textfont=dict(color="#172A39"),
    ))
    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color="#172A39",
        xaxis=dict(gridcolor="rgba(23,42,57,0.1)", tickfont=dict(color="#6E7575"), title="Value score"),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#172A39")),
        margin=dict(l=20, r=40, t=20, b=20),
        height=max(220, 60 * len(similar_with_value)),
    )
    st.plotly_chart(fig, use_container_width=True)