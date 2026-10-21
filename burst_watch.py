"""
Pass 2: take the per-player-game burst numbers from scan_tracking.py,
join the weekly NFL injury report (nflverse) and compare every game with
the player's own healthy baseline.

usage: python burst_watch.py [output/burst_by_player_game_all.csv]
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nflreadpy as nfl
import polars as pl

SEASON = 2021
MAX_WEEK = 8
MIN_CLEAN_GAMES = int(os.environ.get("MIN_CLEAN", 2))     # own baseline needs at least two healthy games
LOWER_BODY = {"Hamstring", "Groin", "Calf", "Quadricep", "Thigh", "Ankle", "Knee", "Foot", "Hip", "Toe", "Achilles"}

OUT = "output"
FIG = "figures"

C_CLEAN, C_QUEST, C_OUT = "#4C72B0", "#DD8452", "#C44E52"


def load_injury_report():
    # rosters give us gsis_it_id, which is the nflId used in the tracking data
    ro = (nfl.load_rosters_weekly([SEASON])
          .filter(pl.col("week") <= MAX_WEEK)
          .select(["gsis_it_id", "gsis_id", "week"]).unique()
          .with_columns(pl.col("gsis_it_id").cast(pl.Int64))
          .to_pandas().rename(columns={"gsis_it_id": "nflId"}))
    inj = (nfl.load_injuries([SEASON])
           .filter(pl.col("week") <= MAX_WEEK + 1)
           .select(["gsis_id", "week", "report_primary_injury", "report_status", "practice_status"])
           .to_pandas())
    # a player can appear twice in a week when the report was updated, keep the last word
    inj = inj.drop_duplicates(["gsis_id", "week"], keep="last")
    return ro, inj


def add_injury_flags(pg):
    ro, inj = load_injury_report()
    df = pg.merge(ro, on=["nflId", "week"], how="left")
    print(f"id match: {df.gsis_id.notna().mean():.1%}")
    df = df.merge(inj, on=["gsis_id", "week"], how="left")

    # was he ruled out the week after this game?
    nxt = inj[inj.report_status == "Out"][["gsis_id", "week", "report_primary_injury"]].copy()
    nxt["week"] -= 1
    nxt = nxt.rename(columns={"report_primary_injury": "next_week_out_injury"})
    df = df.merge(nxt, on=["gsis_id", "week"], how="left")

    df["listed"] = df.report_status.notna() | df.practice_status.notna()
    df["clean"] = ~df.listed
    df["questionable"] = df.report_status == "Questionable"
    df["pre_out"] = df.next_week_out_injury.notna()
    df["lower_body"] = df.report_primary_injury.isin(LOWER_BODY)
    return df


def add_baseline(df):
    base = (df[df.clean].groupby("nflId")
            .agg(base_a=("peak_a_med", "median"), base_s=("top_s_med", "median"),
                 clean_games=("gameId", "size")))
    df = df.merge(base, on="nflId", how="left")
    df = df[df.clean_games >= MIN_CLEAN_GAMES].copy()
    df["burst_idx_a"] = 100 * (df.peak_a_med / df.base_a - 1)
    df["burst_idx_s"] = 100 * (df.top_s_med / df.base_s - 1)
    return df


def status_group(df):
    g = np.where(df.pre_out, "ruled out next week",
         np.where(df.questionable, "questionable, played",
          np.where(df.clean, "healthy", "other listing")))
    return pd.Series(g, index=df.index)


def chart_by_status(df):
    order = ["healthy", "questionable, played", "ruled out next week"]
    colors = [C_CLEAN, C_QUEST, C_OUT]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=False)
    for ax, col, title in zip(axes, ["burst_idx_a", "burst_idx_s"],
                              ["peak acceleration (first 1.5 s)", "top speed before the throw"]):
        data = [df.loc[df.group == grp, col].dropna() for grp in order]
        rng = np.random.default_rng(1)
        for i, (d, c) in enumerate(zip(data, colors)):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(d)), d, s=12, alpha=0.45, color=c, lw=0)
        bp = ax.boxplot(data, positions=range(3), widths=0.5, showfliers=False,
                        medianprops=dict(color="black", lw=1.4), patch_artist=False)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{o}\n(n={len(d)})" for o, d in zip(order, data)], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("% vs own healthy baseline")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Route burst of WR/TE/RB by injury-report status, 2021 weeks 1-8", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/burst_by_status.png", dpi=160)
    plt.close(fig)


def pick_story_players(df):
    picks = {}
    po = df[df.pre_out & df.lower_body] if (df.pre_out & df.lower_body).any() else df[df.pre_out]
    if len(po):
        picks["drop before being ruled out"] = po.sort_values("burst_idx_a").iloc[0].nflId
    q = df[df.questionable & ~df.pre_out]
    if len(q):
        picks["questionable but full burst"] = q.sort_values("burst_idx_a").iloc[-1].nflId
    un = df[df.clean & (df.week > 1)]
    if len(un):
        picks["never listed, burst gone"] = un.sort_values("burst_idx_a").iloc[0].nflId
    return picks


def chart_story_players(df, picks):
    fig, axes = plt.subplots(1, len(picks), figsize=(4.2 * len(picks), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, pid) in zip(axes, picks.items()):
        p = df[df.nflId == pid].sort_values("week")
        ax.plot(p.week, p.burst_idx_a, color="black", lw=1.2, marker="o", ms=4, zorder=1)
        ax.scatter(p.week[p.questionable], p.burst_idx_a[p.questionable], s=70, color=C_QUEST,
                   zorder=2, label="listed questionable")
        ax.scatter(p.week[p.pre_out], p.burst_idx_a[p.pre_out], s=70, color=C_OUT, marker="s",
                   zorder=3, label="ruled out next week")
        for _, r in p[p.listed].iterrows():
            if isinstance(r.report_primary_injury, str):
                ax.annotate(r.report_primary_injury, (r.week, r.burst_idx_a), (0, 8),
                            textcoords="offset points", ha="center", fontsize=8)
        ax.axhline(0, color="grey", lw=0.8, ls="--")
        ax.set_title(f"{p.displayName.iloc[0]} ({p.position.iloc[0]})\n{label}", fontsize=10)
        ax.set_xlabel("week")
        ax.set_xticks(range(1, MAX_WEEK + 1))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("peak acceleration, % vs own baseline")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/story_players.png", dpi=160)
    plt.close(fig)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/burst_by_player_game_all.csv"
    pg = pd.read_csv(src)
    df = add_injury_flags(pg)
    df = add_baseline(df)
    df["group"] = status_group(df)
    os.makedirs(FIG, exist_ok=True)

    cols = ["week", "gameId", "nflId", "displayName", "position", "routes", "peak_a_med", "top_s_med",
            "base_a", "base_s", "clean_games", "burst_idx_a", "burst_idx_s", "group",
            "report_status", "report_primary_injury", "practice_status", "next_week_out_injury"]
    df[cols].sort_values(["week", "displayName"]).to_csv(f"{OUT}/burst_by_player_game.csv", index=False)

    summary = (df.groupby("group")
               .agg(n=("gameId", "size"), players=("nflId", "nunique"),
                    med_idx_a=("burst_idx_a", "median"), med_idx_s=("burst_idx_s", "median"),
                    share_below_minus5=("burst_idx_a", lambda s: (s < -5).mean()))
               .round(2))
    summary.to_csv(f"{OUT}/summary_by_status.csv")
    print(summary)
    try:
        from scipy.stats import mannwhitneyu
        for grp in ["questionable, played", "ruled out next week"]:
            a = df.loc[df.group == "healthy", "burst_idx_a"].dropna()
            b = df.loc[df.group == grp, "burst_idx_a"].dropna()
            if len(b) > 5:
                print(f"healthy vs {grp}: p = {mannwhitneyu(a, b).pvalue:.3f}")
    except ImportError:
        pass

    # the unlisted limpers: lowest burst while not on the report at all
    lead = (df[df.clean & (df.week > 1)]
            .sort_values("burst_idx_a")
            [["week", "displayName", "position", "routes", "burst_idx_a", "burst_idx_s", "next_week_out_injury"]]
            .head(15))
    lead.to_csv(f"{OUT}/limp_leaderboard.csv", index=False)
    print("\nlowest burst while not listed:\n", lead.to_string(index=False))

    chart_by_status(df)
    picks = pick_story_players(df)
    if picks:
        chart_story_players(df, picks)
    print("\nstory players:", {k: df.loc[df.nflId == v, "displayName"].iloc[0] for k, v in picks.items()})


if __name__ == "__main__":
    main()
