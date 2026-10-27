"""Possession as a three-state sequence model: evidence per frame, one decode per half.

The baseline labels every frame on its own and then smooths. That gets the per-frame view roughly
right and the possessions badly wrong, because a majority vote has no idea that a possession is
one object with a start and an end. Here the possessions are what is decoded:

  1. Evidence. For every frame, a score for each of H, A and D, built from the tracking
     (who is near the ball, how fast the ball moves, whether it is off the pitch, whether the
     players have stopped running) and from the Smart Tagging (set pieces mark restarts, phases
     carry a team).
  2. Decode. Viterbi over {H, A, D} with a fixed cost for every change of state. The best path
     is the one whose evidence outweighs the changes it makes, so a single touch or a deflection
     cannot flip the team -- only control sustained long enough to pay the switching cost can.
     That is CHALLENGE.md's "controlled possession", stated as an optimisation.

Scores are additive log-odds-like numbers, so "worth" is easy to read: one frame of a player
standing on the ball with no opponent near is worth about CONTROL_WEIGHT, and switching team costs
SWITCH_COST of that. Every constant below is set from football and tracking knowledge, not fitted
to any labels of this match -- see TECHNICAL.md.
"""

import difflib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402

# Params
H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])
FPS = GAMES[GAME]["fps"]

# -- ball control --------------------------------------------------------------------------------
# A frame says "team T has it" in proportion to how much closer T's nearest player is to the ball
# than the other team's nearest, capped, and discounted when nobody is really on the ball or when
# the ball is travelling at pass speed (a pass in flight belongs to whoever played it, not to
# whoever it happens to fly past).
CONTROL_WEIGHT = 2.0         # score of one frame of unambiguous control
GAP_CAP_M = 3.0              # a 3 m lead over the nearest opponent is as clear as it gets
REACH_M = 0.5                # within this of the ball counts as on it
REACH_DECAY_M = 1.5          # proximity weight falls by e every 1.5 m beyond REACH_M
FLIGHT_SPEED = 6.0           # m/s; above this the ball is passed or shot, not carried
FLIGHT_DISCOUNT = 0.25
BALL_SPEED_LAG = 2           # frames either side for the ball speed estimate

# -- dead ball -----------------------------------------------------------------------------------
PITCH_MARGIN_M = 0.3         # ball this far beyond a line is out
OUT_OF_PLAY = 4.0            # score for D while the ball is off the pitch
MOTION_DEAD = 1.6            # m/s median player speed; below this the game has usually stopped
MOTION_SCALE = 0.6
MOTION_WEIGHT = 1.5
MOTION_WINDOW_S = 2.0        # smoothing of the collective speed, centred
NO_PLAYERS = 1.0             # D score when no player is tracked at all
LIVE_BALL_SPEED = 4.0        # m/s; a moving ball on the pitch is a ball in play
LIVE_BALL = 1.5

# -- Smart Tagging -------------------------------------------------------------------------------
# Codes are matched case-insensitively (DATA.md). An attacking phase's Team is the team with the
# ball; a defending phase's Team is the team without it.
ATTACKING_CODES = {"build up", "progression", "ball in final third", "creating chances",
                   "attacking transition", "long ball", "ball in the box", "players in the box"}
DEFENDING_CODES = {"defending in defensive third", "defending in middle third",
                   "defending in attacking third", "defensive transition"}
SET_PIECE_CODE = "set pieces"
KICKOFF_START_TYPE = "kick off (start)"
PHASE_WEIGHT = 0.08          # per frame: a 10 s phase is worth less than 2 s of clear control
KICK_SPEED = 3.0             # m/s; the restart is the first frame the ball moves this fast
SET_PIECE_LEAD_S = 3.0       # dead time assumed before a set-piece tag starts
SET_PIECE_DEAD = 2.0
RESTART_LIVE_S = 2.0         # after the kick the ball is alive and with the restarting team
RESTART_LIVE = 2.0
RESTART_TEAM = 1.0
BEFORE_KICKOFF_DEAD = 6.0    # everything in a half before its kickoff is a stoppage

# -- decode --------------------------------------------------------------------------------------
SWITCH_COST = 40.0           # H <-> A: about a second of clear control by the new team
STOP_COST = 25.0             # alive -> D
START_COST = 25.0            # D -> alive


def ball_speed(ball, lag=BALL_SPEED_LAG, fps=FPS):
    """m/s from a centred difference; NaN wherever either end is untracked."""
    speed = np.full(len(ball), np.nan)
    step = ball[2 * lag:] - ball[:-2 * lag]
    speed[lag:-lag] = np.hypot(step[:, 0], step[:, 1]) * fps / (2 * lag)
    return speed


def nearest_by_team(ball, players, team):
    """(home, away) distance in metres from the ball to each team's nearest tracked player.
    inf where the ball or that team is untracked."""
    gap = players - ball[:, None, :].astype(np.float32)
    dist = np.hypot(gap[..., 0], gap[..., 1])
    dist[np.isnan(dist)] = np.inf
    return dist[:, team == 0].min(axis=1), dist[:, team == 1].min(axis=1)


def rolling_nanmean(values, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN. NaN where the window is all NaN."""
    seen = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(seen, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(seen)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    n = count[high] - count[low]
    out = np.full(len(values), np.nan)
    out[n > 0] = (total[high] - total[low])[n > 0] / n[n > 0]
    return out


def collective_speed(speed, players, fps=FPS):
    """Median player speed per frame, m/s. Uses the ATD speed channel where it has values and
    falls back to position differences where it does not."""
    with np.errstate(invalid="ignore"):
        if np.isnan(speed).all():
            step = np.full(speed.shape, np.nan, dtype=np.float32)
            delta = players[1:] - players[:-1]
            step[1:] = np.hypot(delta[..., 0], delta[..., 1]) * fps
            speed = step
        present = (~np.isnan(speed)).sum(axis=1)
        median = np.full(len(speed), np.nan)
        rows = present > 0
        median[rows] = np.nanmedian(speed[rows], axis=1)
    return median


def control_scores(ball, players, team, ball_v):
    """Per-frame log-odds that H rather than A controls the ball. 0 where it says nothing."""
    d_home, d_away = nearest_by_team(ball, players, team)
    closest = np.minimum(d_home, d_away)
    usable = np.isfinite(closest)
    llr = np.zeros(len(ball))

    # Only one team tracked: the lead is +/-inf before clipping, i.e. the full cap. Neither: NaN,
    # which `usable` already excludes.
    with np.errstate(invalid="ignore"):
        lead = np.clip(np.where(usable, d_away - d_home, 0.0), -GAP_CAP_M, GAP_CAP_M)
    lead[~np.isfinite(lead)] = 0.0
    reach = np.exp(-np.maximum(np.where(usable, closest, 0.0) - REACH_M, 0.0) / REACH_DECAY_M)
    flight = np.where(ball_v > FLIGHT_SPEED, FLIGHT_DISCOUNT, 1.0)  # NaN compares False -> 1.0
    llr[usable] = (CONTROL_WEIGHT * lead / GAP_CAP_M * reach * flight)[usable]
    return llr


def dead_scores(ball, ball_v, motion, pitch):
    """Per-frame score for D relative to the ball being alive."""
    length, width = pitch
    score = np.zeros(len(ball))

    with np.errstate(invalid="ignore"):
        out = ((ball[:, 0] < -PITCH_MARGIN_M) | (ball[:, 0] > length + PITCH_MARGIN_M)
               | (ball[:, 1] < -PITCH_MARGIN_M) | (ball[:, 1] > width + PITCH_MARGIN_M))
        score[out] += OUT_OF_PLAY

        stillness = np.clip((MOTION_DEAD - motion) / MOTION_SCALE, -1.0, 1.0)
        score += np.where(np.isnan(motion), NO_PLAYERS, MOTION_WEIGHT * stillness)

        live = (ball_v > LIVE_BALL_SPEED) & ~out
        score[live] -= LIVE_BALL
    return score


def tag_side(team_tag, home_name, away_name):
    """H, A or None for a Smart Tagging Team value. The tagging and the ATD metadata spell clubs
    differently ("Atletic Club"), so the closer of the two names wins; N/A and blanks are None."""
    tag = (team_tag or "").strip().lower()
    if not tag or tag in ("n/a", "na", "none"):
        return None
    home = difflib.SequenceMatcher(None, tag, (home_name or "").lower()).ratio()
    away = difflib.SequenceMatcher(None, tag, (away_name or "").lower()).ratio()
    if home == away:
        return None
    return H if home > away else A


def restart_frame(start, end, ball, ball_v, inside):
    """First frame in [start - lead, end) where the ball is kicked on the pitch, else the middle."""
    lo = max(0, start - int(SET_PIECE_LEAD_S * FPS))
    with np.errstate(invalid="ignore"):
        kicked = np.flatnonzero((ball_v[lo:end] > KICK_SPEED) & inside[lo:end])
    return lo + int(kicked[0]) if kicked.size else (start + end) // 2


def tagging_scores(events, n_frames, ball, ball_v, pitch, home_name, away_name):
    """(team, dead): additive per-frame scores from the Smart Tagging.

    team is (n, 2) for H and A, dead is (n,) for D. Also returns the kickoff restart frames, which
    decide where each half's leading stoppage ends."""
    length, width = pitch
    team = np.zeros((n_frames, 2))
    dead = np.zeros(n_frames)
    kickoffs = []
    with np.errstate(invalid="ignore"):
        inside = ((ball[:, 0] >= 0) & (ball[:, 0] <= length)
                  & (ball[:, 1] >= 0) & (ball[:, 1] <= width))

    for event in events:
        code = (event["code"] or "").strip().lower()
        start = max(0, event["start_frame"])
        end = min(n_frames, max(event["end_frame"], start + 1))
        if start >= n_frames:
            continue
        side = tag_side(event["tags"].get("Team"), home_name, away_name)

        if code in ATTACKING_CODES and side is not None:
            team[start:end, side] += PHASE_WEIGHT
        elif code in DEFENDING_CODES and side is not None:
            team[start:end, 1 - side] += PHASE_WEIGHT
        elif code == SET_PIECE_CODE:
            kick = restart_frame(start, end, ball, ball_v, inside)
            lead = max(0, start - int(SET_PIECE_LEAD_S * FPS))
            dead[lead:kick] += SET_PIECE_DEAD
            live_end = min(n_frames, kick + int(RESTART_LIVE_S * FPS))
            dead[kick:live_end] -= RESTART_LIVE
            if side is not None:
                team[kick:live_end, side] += RESTART_TEAM
            if (event["tags"].get("Type") or "").strip().lower() == KICKOFF_START_TYPE:
                kickoffs.append(kick)
    return team, dead, sorted(kickoffs)


def viterbi(scores, switch=SWITCH_COST, stop=STOP_COST, start=START_COST, first=D):
    """Best state path through (n, 3) scores with a cost on every change of state.

    Plain Python floats rather than numpy per step: three states make the per-step numpy call
    overhead the whole cost, and this way a half of 80k frames decodes in well under a second."""
    cost = [[0.0, switch, stop],
            [switch, 0.0, stop],
            [start, start, 0.0]]
    n = len(scores)
    back = np.zeros((n, 3), dtype=np.int8)
    rows = scores.tolist()
    prev = [-start, -start, 0.0] if first == D else [0.0, 0.0, -stop]
    prev = [prev[s] + rows[0][s] for s in range(3)]

    for t in range(1, n):
        row = rows[t]
        cur = [0.0, 0.0, 0.0]
        for s in range(3):
            best_k, best = 0, prev[0] - cost[0][s]
            for k in (1, 2):
                value = prev[k] - cost[k][s]
                if value > best:
                    best_k, best = k, value
            cur[s] = best + row[s]
            back[t, s] = best_k
        prev = cur

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(prev))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def frame_scores(atd, events, home_name, away_name, game=GAME):
    """(n, 3) evidence for H, A, D over the whole video, plus the kickoff frames found."""
    pitch = GAMES[game]["pitch"]
    ball, players, team = atd["ball"], atd["players"], atd["team"]
    n = len(ball)

    ball_v = ball_speed(ball)
    motion = rolling_nanmean(collective_speed(atd["speed"], players),
                             int(MOTION_WINDOW_S * FPS / 2))
    llr = control_scores(ball, players, team, ball_v)

    scores = np.zeros((n, 3))
    scores[:, H] = llr / 2
    scores[:, A] = -llr / 2
    scores[:, D] = dead_scores(ball, ball_v, motion, pitch)

    tag_team, tag_dead, kickoffs = tagging_scores(events, n, ball, ball_v, pitch,
                                                  home_name, away_name)
    scores[:, :2] += tag_team
    scores[:, D] += tag_dead
    return scores, kickoffs


def predict(atd, events, home_name, away_name, game=GAME):
    """Label array over the whole video: H/A/D inside the two halves, X everywhere else."""
    scores, kickoffs = frame_scores(atd, events, home_name, away_name, game)
    labels = np.full(len(scores), "X", dtype="<U1")

    for _, first, last in half_video_blocks(game):
        stop = last + 1
        half_scores = scores[first:stop].copy()
        # A half begins dead until its kickoff. The kickoff itself is found in the data (the
        # tagged restart, timed by the first kick of the ball), not read off a published frame.
        kick = next((k for k in kickoffs if first <= k < stop), None)
        if kick is not None:
            half_scores[:kick - first, D] += BEFORE_KICKOFF_DEAD
        labels[first:stop] = LABELS[viterbi(half_scores)]
    return labels
