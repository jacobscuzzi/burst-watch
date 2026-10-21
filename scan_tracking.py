"""
Pass 1: go through the tracking files and measure route burst for every
WR/TE/RB on every dropback. Only the pre-throw frames exist in this data,
which is fine - that's the acceleration phase we care about.

usage: python scan_tracking.py            # all 8 weeks (~10 min)
       python scan_tracking.py --weeks 1  # quick check
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

DATA = "data"
OUT = "output"

BURST_FRAMES = 15      # first 1.5 s after the snap -> peak acceleration
MIN_FRAMES_TOPSPEED = 15  # need at least 1.5 s before the throw for a top speed to mean anything
MIN_ROUTES = 8         # routes per game before we trust a player's median

SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}
THROW_EVENTS = {"pass_forward", "autoevent_passforward"}


def load_meta():
    games = pd.read_csv(f"{DATA}/games.csv")
    plays = pd.read_csv(f"{DATA}/plays.csv")
    players = pd.read_csv(f"{DATA}/players.csv")
    pff = pd.read_csv(f"{DATA}/pffScoutingData.csv",
                      usecols=["gameId", "playId", "nflId", "pff_role"])

    # scrambles have no throw and the receiver stops running his route, drop them
    plays = plays[plays.passResult.isin(["C", "I", "IN", "S"])]

    routes = pff[pff.pff_role == "Pass Route"].merge(
        players[["nflId", "officialPosition"]], on="nflId")
    routes = routes[routes.officialPosition.isin(["WR", "TE", "RB"])]
    routes = routes.merge(plays[["gameId", "playId", "passResult"]], on=["gameId", "playId"])
    return games, plays, players, routes[["gameId", "playId", "nflId", "passResult"]]


def play_windows(track):
    """snap frame and end frame (throw or sack) per play, both event spellings"""
    ev = track.loc[track.event.notna(), ["playId", "frameId", "event"]].drop_duplicates()
    snap = ev[ev.event.isin(SNAP_EVENTS)].groupby("playId").frameId.min().rename("f_snap")
    throw = ev[ev.event.isin(THROW_EVENTS)].groupby("playId").frameId.min().rename("f_throw")
    sack = ev[ev.event == "qb_sack"].groupby("playId").frameId.min().rename("f_sack")
    w = pd.concat([snap, throw, sack], axis=1).dropna(subset=["f_snap"])
    w["f_end"] = w.f_throw.fillna(w.f_sack)
    return w.dropna(subset=["f_end"]).reset_index()


def scan_game(game_id, routes):
    path = f"{DATA}/tracking/tracking_{game_id}.csv"
    track = pd.read_csv(path, usecols=["gameId", "playId", "nflId", "frameId", "s", "a", "event"])
    win = play_windows(track)

    t = track.dropna(subset=["nflId"]).copy()
    t["nflId"] = t.nflId.astype(int)
    t = t.merge(routes[["gameId", "playId", "nflId"]], on=["gameId", "playId", "nflId"])
    t = t.merge(win[["playId", "f_snap", "f_end"]], on="playId")
    t = t[(t.frameId >= t.f_snap) & (t.frameId <= t.f_end)]
    t["rel"] = t.frameId - t.f_snap

    key = ["gameId", "playId", "nflId"]
    burst = t[t.rel < BURST_FRAMES].groupby(key).a.max().rename("peak_a")
    top = t.groupby(key).s.max().rename("top_s")
    n_frames = t.groupby(key).rel.max().rename("frames_to_throw")
    out = pd.concat([burst, top, n_frames], axis=1).reset_index()
    # quick game / screens: the guy never gets to speed, don't count his top speed
    out.loc[out.frames_to_throw < MIN_FRAMES_TOPSPEED, "top_s"] = np.nan
    out["t_end"] = out.frames_to_throw / 10.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", default="all", help="e.g. 1 or 1,2,3")
    args = ap.parse_args()

    games, plays, players, routes = load_meta()
    if args.weeks != "all":
        wk = [int(x) for x in args.weeks.split(",")]
        games = games[games.week.isin(wk)]
    tag = "all" if args.weeks == "all" else "wk" + args.weeks.replace(",", "-")

    parts = []
    t0 = time.time()
    for i, gid in enumerate(games.gameId, 1):
        parts.append(scan_game(gid, routes))
        print(f"[{i}/{len(games)}] {gid}  {time.time()-t0:.0f}s", flush=True)
    per_route = pd.concat(parts, ignore_index=True)
    per_route = per_route.merge(routes, on=["gameId", "playId", "nflId"])
    os.makedirs(OUT, exist_ok=True)
    per_route.to_csv(f"{OUT}/burst_by_route_{tag}.csv", index=False)

    # one row per player per game
    g = per_route.groupby(["gameId", "nflId"])
    pg = pd.DataFrame({
        "routes": g.size(),
        "peak_a_med": g.peak_a.median(),
        "top_s_med": g.top_s.median(),
        "t_end_med": g.t_end.median(),
    }).reset_index()
    pg = pg[pg.routes >= MIN_ROUTES]
    pg = pg.merge(games[["gameId", "week"]], on="gameId")
    pg = pg.merge(players[["nflId", "displayName", "officialPosition"]], on="nflId")
    pg = pg.rename(columns={"officialPosition": "position"})
    pg = pg[["gameId", "week", "nflId", "displayName", "position", "routes",
             "peak_a_med", "top_s_med", "t_end_med"]].sort_values(["week", "gameId", "nflId"])
    pg.to_csv(f"{OUT}/burst_by_player_game_{tag}.csv", index=False)

    print(f"\n{len(per_route)} routes, {len(pg)} player-games with >= {MIN_ROUTES} routes")
    print("time to throw (s):", per_route.t_end.describe()[["25%", "50%", "75%"]].round(2).to_dict())
    print(pg.groupby("position")[["peak_a_med", "top_s_med"]].median().round(2))


if __name__ == "__main__":
    main()
