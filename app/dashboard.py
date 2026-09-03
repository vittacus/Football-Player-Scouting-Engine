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
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@500&family=Space+Grotesk:wght@500;700&display=swap');

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F1ECE4 !important;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .block-container { background-color: #F1ECE4; padding: 44px 52px !important; max-width: 1120px; }

    h1, h2, h3, p, span, label, div { color: #152238; }
    p { line-height: 1.6; }

    .title-row { display: flex; align-items: center; gap: 16px; margin-bottom: 0; }
    h1 {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 40px;
        letter-spacing: 0.01em;
        margin-bottom: 0 !important;
    }
    .pl-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.05em;
        color: #E2583D;
        border: 1.5px solid #E2583D;
        padding: 5px 12px;
        border-radius: 6px;
        text-transform: uppercase;
    }
    .title-underline {
        height: 3px;
        background-color: #E2583D;
        margin: 16px 0 24px 0;
        width: 100%;
    }

    .how-it-works {
        display: flex;
        gap: 16px;
        margin: 0 0 24px 0;
    }
    .how-step {
        flex: 1;
        background-color: #FFFFFF;
        padding: 20px 22px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(21,34,56,0.08);
    }
    .how-step-num {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 22px;
        color: #E2583D;
    }
    .how-step-text {
        font-size: 13.5px;
        color: #152238;
        margin-top: 6px;
        line-height: 1.5;
    }

    [data-testid="stWidgetLabel"] p {
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 11px !important;
        color: #6B7280 !important;
        margin-bottom: 6px !important;
    }

    .scoreboard {
        background-color: #152238;
        border-radius: 10px;
        padding: 22px 36px 22px 28px;
        margin: 24px 0 36px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 16px rgba(21,34,56,0.15);
    }
    .scoreboard-club {
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 23px;
        letter-spacing: 0.01em;
        text-transform: uppercase;
        color: #F1ECE4 !important;
    }
    .scoreboard-sub { font-size: 13px; color: #9BA6BD !important; margin-top: 4px; }
    .scoreboard-value {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 52px;
        color: #E2583D !important;
        text-align: right;
        line-height: 1;
    }
    .scoreboard-value-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        letter-spacing: 0.05em;
        color: #9BA6BD !important;
        text-align: right;
    }

    .side-label {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 6px 14px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .side-label.reference { background-color: #152238; color: #F1ECE4; }
    .side-label.target { background-color: #E2583D; color: #F1ECE4; }

    .vs-badge {
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 18px;
        color: #6B7280;
        text-align: center;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
    }

    .stat-panel {
        background-color: #FFFFFF;
        border-top: 3px solid #152238;
        border-radius: 10px;
        padding: 24px 28px;
        box-shadow: 0 2px 10px rgba(21,34,56,0.08);
    }
    .stat-panel.target { border-top: 3px solid #E2583D; }
    .stat-panel-title { font-family: 'Oswald', sans-serif; font-weight: 600; font-size: 23px; color: #152238; }
    .stat-panel.target .stat-panel-title { color: #E2583D !important; }
    .stat-panel-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.03em;
        color: #6B7280;
        margin-bottom: 16px;
    }
    .stat-row {
        display: flex; justify-content: space-between; align-items: center; padding: 11px 0;
        border-bottom: 1px solid rgba(21,34,56,0.08); font-size: 14px;
    }
    .stat-row:last-child { border-bottom: none; }
    .stat-label { color: #6B7280; }
    .stat-value-wrap { display: flex; align-items: center; gap: 10px; }
    .stat-value { font-family: 'Oswald', sans-serif; font-weight: 500; color: #152238; }

    .edge-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 4px;
        white-space: nowrap;
    }
    .edge-badge.reference { background-color: rgba(21,34,56,0.1); color: #152238; }
    .edge-badge.target { background-color: rgba(226,88,61,0.12); color: #E2583D; }

    .value-callout {
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid rgba(21,34,56,0.1);
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }
    .value-callout-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #6B7280;
    }
    .value-callout-number {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 34px;
        line-height: 1;
    }
    .stat-panel.reference .value-callout-number { color: #152238; }
    .stat-panel.target .value-callout-number { color: #E2583D; }

    .stat-glossary {
        font-size: 11px;
        color: #9BA6BD;
        margin-top: 14px;
        line-height: 1.5;
        font-style: italic;
    }

    .summary-card {
        background-color: #FFFFFF;
        border-left: 5px solid #E2583D;
        border-radius: 10px;
        padding: 22px 26px;
        margin: 24px 0;
        box-shadow: 0 2px 10px rgba(21,34,56,0.08);
    }
    .summary-verdict {
        font-family: 'Oswald', sans-serif;
        font-size: 19px;
        font-weight: 500;
    }
    .summary-verdict .delta-up { color: #1E7A4C; }
    .summary-verdict .delta-number { font-family: 'Space Grotesk', sans-serif; font-weight: 700; }
    .summary-sub {
        font-size: 12.5px;
        color: #6B7280;
        margin-top: 10px;
        padding-top: 10px;
        border-top: 1px solid rgba(21,34,56,0.08);
    }
    .summary-sub strong { color: #E2583D; font-family: 'Space Grotesk', sans-serif; }

    .legend-row {
        display: flex;
        gap: 24px;
        align-items: center;
        margin: 16px 0 4px 0;
        font-size: 12px;
        color: #6B7280;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .legend-item { display: flex; align-items: center; gap: 8px; }

    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 2px 10px rgba(21,34,56,0.06);
    }
    [data-testid="stExpander"] summary {
        background-color: #FFFFFF !important;
        color: #152238 !important;
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 12px !important;
    }
    [data-testid="stExpander"] p {
        color: #152238 !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        text-transform: none;
        letter-spacing: normal;
        font-size: 14px;
    }

    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid rgba(21,34,56,0.25) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] > div * { color: #152238 !important; }

    [role="listbox"] { background-color: #FFFFFF !important; border-radius: 8px !important; }
    [role="option"] { background-color: #FFFFFF !important; color: #152238 !important; }
    [role="option"]:hover { background-color: #F1ECE4 !important; }
    [role="option"] * { color: #152238 !important; }

    button, .stButton > button { border-radius: 8px !important; }
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


ATTACKER_GLOSSARY = (
    "Goals per 90 and Assists per 90 = goals or assists per 90 minutes played "
    "(a full match), so players with different amounts of playing time can "
    "be compared fairly. Goal Contributions per 90 = goals + assists per 90."
)
KEEPER_GLOSSARY = (
    "Save % = shots saved out of shots faced. Clean Sheet % = matches with "
    "zero goals conceded. Goals Against per 90 = goals conceded per 90 minutes "
    "(lower is better)."
)

HIGHER_IS_BETTER = {
    "goals_per90": True,
    "assists_per90": True,
    "goal_contributions_per90": True,
    "save_pct": True,
    "clean_sheet_pct": True,
    "goals_against_per90": False,
}


def render_stat_panel(title, subtitle, stats, value_score, variant, glossary):
    rows_html = "".join(
        f'<div class="stat-row"><span class="stat-label">{label}</span>'
        f'<span class="stat-value-wrap">{badge}<span class="stat-value">{value}</span></span></div>'
        for label, value, badge in stats
    )
    st.markdown(
        f"""
        <div class="stat-panel {variant}">
            <div class="stat-panel-title">{title}</div>
            <div class="stat-panel-sub">{subtitle}</div>
            {rows_html}
            <div class="value-callout">
                <span class="value-callout-label">Value Score</span>
                <span class="value-callout-number">{value_score}</span>
            </div>
            <div class="stat-glossary">{glossary}</div>
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

with st.expander("About this project (methodology & limitations)"):
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

st.markdown(
    """
    <div class="how-it-works">
        <div class="how-step">
            <div class="how-step-num">1</div>
            <div class="how-step-text">Pick your club and one player currently on your roster you want to find an upgrade or replacement for.</div>
        </div>
        <div class="how-step">
            <div class="how-step-num">2</div>
            <div class="how-step-text">We find players at other clubs with a statistically similar style of play to your player.</div>
        </div>
        <div class="how-step">
            <div class="how-step-num">3</div>
            <div class="how-step-text">Each option gets a "Value Score" so you can see, in one number, whether they'd actually be an upgrade.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

col_p, col_a, col_b, col_c = st.columns(4)

with col_p:
    profile_type = st.selectbox("Profile type", ["Attacker", "Goalkeeper"])

if profile_type == "Attacker":
    sim_df_group = sim_df_attacking
    val_df_group = val_df_latest
    group_stat_labels = [
        ("Goals per 90", "goals_per90"),
        ("Assists per 90", "assists_per90"),
        ("Goal Contributions per 90", "goal_contributions_per90"),
    ]
    group_glossary = ATTACKER_GLOSSARY
else:
    sim_df_group = get_keeper_similarity_data()
    val_df_group = get_keeper_valuation_data().sort_values("season").drop_duplicates("player", keep="last")
    group_stat_labels = [
        ("Save %", "save_pct"),
        ("Clean Sheet %", "clean_sheet_pct"),
        ("Goals Against per 90", "goals_against_per90"),
    ]
    group_glossary = KEEPER_GLOSSARY

with col_a:
    team_list = sorted(sim_df_group["team"].unique())
    selected_team = st.selectbox("Your club", team_list)

with col_b:
    team_players = sorted(sim_df_group[sim_df_group["team"] == selected_team]["player"].unique())
    if not team_players:
        st.warning(f"No {profile_type.lower()}s found for {selected_team}")
        st.stop()
    selected_player = st.selectbox("Your current player", team_players)

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
                st.info(f"No targets match the {selected_role} role, showing all targets instead.")
            else:
                similar_with_value = filtered

    similar_with_value = similar_with_value.sort_values("value_score", ascending=False).reset_index(drop=True)

    value_display = f"{selected_value['value_score'].values[0]:.2f}" if not selected_value.empty else "—"

    st.markdown(
        f"""
        <div class="scoreboard">
            <div>
                <div class="scoreboard-club">{selected_team}</div>
                <div class="scoreboard-sub">Scouting report for {selected_player}</div>
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

    selected_label = st.selectbox("Pick a target to compare against " + selected_player + ":", option_labels)
    compare_target = label_to_player[selected_label]

    reference_row = sim_df_group[sim_df_group["player"] == selected_player].iloc[0]
    reference_value = val_df_group[val_df_group["player"] == selected_player]
    reference_score = reference_value["value_score"].values[0] if not reference_value.empty else None
    reference_value_display = f"{reference_score:.2f}" if reference_score is not None else "—"

    target_row = sim_df_group[sim_df_group["player"] == compare_target].iloc[0]
    target_value_row = similar_with_value[similar_with_value["player"] == compare_target].iloc[0]
    target_score = target_value_row["value_score"]

    reference_stats = [("Age", int(reference_row["age"]), "")]
    target_stats = [("Age", int(target_row["age"]), "")]

    for label, col in group_stat_labels:
        ref_val = reference_row[col]
        tgt_val = target_row[col]
        higher_wins = HIGHER_IS_BETTER.get(col, True)

        if higher_wins:
            ref_wins = ref_val > tgt_val
            tgt_wins = tgt_val > ref_val
        else:
            ref_wins = ref_val < tgt_val
            tgt_wins = tgt_val < ref_val

        ref_badge = '<span class="edge-badge reference">Edge</span>' if ref_wins else ""
        tgt_badge = '<span class="edge-badge target">Edge</span>' if tgt_wins else ""

        reference_stats.append((label, f"{ref_val:.2f}", ref_badge))
        target_stats.append((label, f"{tgt_val:.2f}", tgt_badge))

    col_left, gap, col_right = st.columns([1, 0.08, 1])
    with col_left:
        st.markdown('<div class="side-label reference">Your Player</div>', unsafe_allow_html=True)
        render_stat_panel(
            title=selected_player,
            subtitle=f"{selected_team} · {reference_row['position']}",
            stats=reference_stats,
            value_score=reference_value_display,
            variant="reference",
            glossary=group_glossary,
        )
    with gap:
        st.markdown('<div class="vs-badge">VS</div>', unsafe_allow_html=True)
    with col_right:
        st.markdown('<div class="side-label target">Scouting Target</div>', unsafe_allow_html=True)
        render_stat_panel(
            title=compare_target,
            subtitle=f"{target_row['team']} · {target_row['position']}",
            stats=target_stats,
            value_score=f"{target_score:.2f}",
            variant="target",
            glossary=group_glossary,
        )

    st.markdown(
        f"""
        <div class="legend-row">
            <div class="legend-item"><span class="edge-badge reference">Edge</span> {selected_player}</div>
            <div class="legend-item"><span class="edge-badge target">Edge</span> {compare_target}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Verdict always shows a positive magnitude, and the subject named first
    # carries the direction -- avoids ever printing a negative "higher than"
    profile_label = "attacking" if profile_type == "Attacker" else "goalkeeping"
    if reference_score is not None:
        diff = target_score - reference_score
        magnitude = abs(diff)
        if diff > 0:
            verdict_text = (
                f'<strong>{compare_target}</strong> scores '
                f'<span class="delta-up delta-number">+{magnitude:.2f}</span> higher than '
                f'<strong>{selected_player}</strong> and currently rates as better value.'
            )
        elif diff < 0:
            verdict_text = (
                f'<strong>{selected_player}</strong> scores '
                f'<span class="delta-up delta-number">+{magnitude:.2f}</span> higher than '
                f'<strong>{compare_target}</strong>. Your current player currently rates as better value.'
            )
        else:
            verdict_text = (
                f'<strong>{selected_player}</strong> and <strong>{compare_target}</strong> '
                f'rate as equal value on this measure.'
            )
    else:
        verdict_text = f'<strong>{compare_target}</strong> is being compared against <strong>{selected_player}</strong>.'

    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-verdict">{verdict_text}</div>
            <div class="summary-sub">
                Statistical similarity between {selected_player} and {compare_target}:
                <strong>{target_value_row['similarity_score']:.3f}</strong> (1.0 = identical {profile_label} profile)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Value comparison")
    st.caption(f"All {len(similar_with_value)} targets vs. {selected_player}'s value score (dashed line)")

    bar_colors = [
        "#152238" if player == compare_target else "#E2583D"
        for player in similar_with_value["player"]
    ]
    fig = go.Figure(go.Bar(
        x=similar_with_value["value_score"],
        y=similar_with_value["player"],
        orientation="h",
        marker_color=bar_colors,
        text=similar_with_value["value_score"].round(2),
        textposition="outside",
        textfont=dict(color="#152238", family="Space Grotesk", size=13),
    ))

    if reference_score is not None:
        fig.add_vline(
            x=reference_score,
            line_dash="dash",
            line_color="#152238",
            line_width=2,
            annotation_text=f"{selected_player}: {reference_score:.2f}",
            annotation_position="top",
            annotation_font=dict(color="#152238", size=12, family="JetBrains Mono"),
        )

    fig.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color="#152238",
        font_family="IBM Plex Sans",
        xaxis=dict(gridcolor="rgba(21,34,56,0.08)", tickfont=dict(color="#6B7280"), title="Value score"),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#152238")),
        margin=dict(l=20, r=50, t=50, b=20),
        height=max(260, 64 * len(similar_with_value)),
    )
    st.plotly_chart(fig, use_container_width=True)