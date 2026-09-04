# Player Scouting Engine

A tool that can find statistically similar Premier League players to a club's existing roster, scores them on a composite value metric. Theis then surfaces head-to-head comparisons to support team scouting decisions for attacking players as well as goalkeepers.

**Live Dashboard** (https://vittacus-football-player-scouting-engine-appdashboard-walnhp.streamlit.app/)

![Dashboard Preview]
(
<img width="701" height="631" alt="Screenshot 2026-09-03 at 6 00 52 PM" src="https://github.com/user-attachments/assets/12229150-19e7-433a-a5f3-5e256b24af04" />
<img width="688" height="541" alt="Screenshot 2026-09-03 at 6 01 22 PM" src="https://github.com/user-attachments/assets/e5a7614b-9bb2-4c48-9da7-4e987e102dd1" />
<img width="697" height="393" alt="Screenshot 2026-09-03 at 6 01 43 PM" src="https://github.com/user-attachments/assets/ff490008-d584-4ed1-ab3d-adbe1102fde1" />
)


## The Question

If a club wants to find an upgrade or replacement for a player already on their roster, can we find players elsewhere within the league with a similar statistical profile? If so, can we then rank those options by an interpretable measure of value rather than simply relying on name recognition or transfer fees alone?

## Data

- **Source:** FBref, accessed via the `soccerdata` Python library
- **League:** Premier League
- **Seasons:** 2024-25 and 2025-26 (Simply taking the recent 2 years of play for player form)
- **Player groups:** Attacking players (Strikers, Midfielders, Wingers) and
  goalkeepers, pulled and processed as two separate pipelines since a "good" player
  means something different for each role

## Features

**Attackers**, filtered to players with at least 900 minutes played (roughly
10 full matches), so per-90 rates aren't distorted by tiny sample sizes:

| Feature | What it captures |
|---|---|
| Age | Used for a mild value penalty past estimated peak (24) |
| Goals per 90 | Scoring output, normalized for playing time |
| Assists per 90 | Creative output, normalized for playing time |
| Goal Contributions per 90 | Goals + assists per 90, a combined output measure |

**Goalkeepers**, filtered to at least 10 starts:

| Feature | What it captures |
|---|---|
| Save % | Shots saved out of shots faced |
| Clean Sheet % | Matches with zero goals conceded |
| Goals Against per 90 | Goals conceded per 90 minutes (lower is better) |

Note: Defenders are not included because FBref's defensive stat tables have limited data available, 
only having miscellaneous stats. Because of the lack of reliable stats, it felt better to leave out
a weak defender profile and acknowledge this as a future gap!

## Models

**Similarity search:** cosine similarity on scaled per-90 stats, comparing a
selected player against every other player in the same group (attacker or
goalkeeper), excluding players already on the selected club. Features are
standardized first (`StandardScaler`) so no single stat ends up dominating the
comparison just because it happens to have larger raw numbers (ex: a player with over 900 minutes played vs a player
with <500.

**Value score:** I originally attempted to make this as a linear regression predicting
minutes played from attacking output. That approach was scrapped after
testing showed essentially no relationship (R² near zero). Instead, minutes played
is driven by factors like squad rotation and position (ex: a low scoring
starting defender plays far more than a rotated striker, but may have less stats) that attacking
stats alone can't capture. The replacement is a transparent, weighted
composite score instead, with every input and its weight stated explicitly
rather than relying on regression coefficients fit to a target that turned
out to not be meaningful. Goalkeeper scoring follows the same pattern, with
`goals_against_per90` subtracted as a penalty since it's the one stat where
lower is better.

## Dashboard

Built with Streamlit and Plotly. Given a club and a current player, the
dashboard will do the following:

- Shows that player's own value score as a baseline
- Lets the user pick from a top 5 rank of similar players both in position as well as statistic output at other
  clubs to compare head-to-head
- Marks which player have the edge on each individual stat
- States a plain language verdict ("X scores +0.13 higher than Y") rather
  than requiring the reader to interpret a chart to get the answer
- Plots all candidate targets against the current player's score as a
  reference line, so the full field is visible at a glance
- Includes a role filter (for attackers only) mapping common scouting language (striker, winger, attacking midfielder)
  onto the broader position tags the data actually provides

## Limitations
One of the main limitations is the fact that the dashboard pulls from only one league and two seasons, so findings are not usable for
other leagues/eras of play (say you wanted to compare 2000s players with current). Additionally, the value score is a designed composite, not a market price/transfer valuation.
Therefore, it's really most meaningful when comparing players to each other only. Because of this stat based approach, it doesn't capture injury history, specific tactical systems per team, or transfer market
realities (such as rival teams selling players to each other, which rarely occurs). Some other limitations include the lack of defenders covered in the dashboard, which was discussed previously.

## What I'd Build Next

- Add a defender profile once a data source with real defensive stats
  (tackles, interceptions, clearances, aerials won) is available
- Expand beyond the Premier League to compare across leagues for best players on a global scale
- Pull real transfer market data (such as estimated market value) to validate
  the composite value score against an external benchmark
- Add player photos and club crests for a more scannable interface
- Extend the season range as new data becomes available each year

## Project Structure

```
src/fetch_data.py        → pulls and caches raw player stats from FBref (standard + keeper)
src/features.py           → filters, cleans, and builds per-90 feature sets for each player group
src/similarity_model.py   → scales features and computes cosine similarity between players
src/valuation_model.py    → computes the composite value score for each player group
app/dashboard.py           → Streamlit dashboard tying similarity + valuation together
```
