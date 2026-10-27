"""Sanity checks on the real bundle that print summaries only -- never rows of data.

    python solution/check_data.py

Every line it prints is an aggregate or a schema name, so its output can be shared (in an issue, or
with a teammate) without sharing the match data itself. It answers the questions the model makes
assumptions about:

  - does the raw ATD have the shape the fast loader expects, and what channels does the ball have?
  - which side does each Smart Tagging Team value map to?
  - does an attacking phase's Team really have the ball, and a defending phase's Team not? Checked
    against the tracking's own nearest-player evidence -- no labels of any kind.
  - are the two kickoffs found where DATA.md says they are?
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402
import load_smart_tagging  # noqa: E402
import atd as atd_loader  # noqa: E402
import make_submission  # noqa: E402
import model  # noqa: E402

# Params
# Published in DATA.md / FAQ.md. Used here only to check that kickoff detection works; the model
# never reads them.
PUBLISHED_KICKOFFS = (11185, 113742)
KICKOFF_SEARCH_S = 60


def channel_names(meta_path):
    """Every channel-like element the metadata declares, as (tag, id) -- the schema, not data."""
    root = ET.parse(meta_path).getroot()
    out = []
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        if "channel" in tag.lower():
            out.append((tag, element.get("id") or element.get("channelId")
                        or element.get("playerChannelId")))
    return out


def main():
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    home, away = make_submission.team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))

    with open(raw_path) as handle:
        first = handle.readline()
    n_slots, per_slot, per_ball = atd_loader._line_shape(first)
    print("\nraw ATD: {} slots x {} fields, ball has {} fields".format(n_slots, per_slot, per_ball))
    channels = channel_names(meta_path)
    ball_like = sorted({cid for _, cid in channels if cid and not cid.lower().startswith("player")})
    print("non-player channels declared in metadata: {}".format(ball_like or "none found"))

    atd = atd_loader.load(GAME)
    ball = atd["ball"]
    graded = np.zeros(len(ball), dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    print("ball tracked: {:.1f}% of all frames, {:.1f}% of graded (DATA.md: 56.9% / 59.3%)".format(
        100 * np.mean(~np.isnan(ball[:, 0])), 100 * np.mean(~np.isnan(ball[graded, 0]))))
    print("ATD speed channel present: {}".format(not np.isnan(atd["speed"]).all()))

    events = load_smart_tagging.load(GAME)
    print("\nSmart Tagging Team values -> side")
    for value in sorted({e["tags"].get("Team") or "" for e in events}):
        side = model.tag_side(value, home, away)
        print("  {:<20} -> {}".format(repr(value), "-" if side is None else "HA"[side]))

    ball_v = model.ball_speed(ball)
    llr = model.control_scores(ball, atd["players"], atd["team"], ball_v)
    print("\nphase Team vs the tracking's nearest-player evidence (agreement over frames where it "
          "has a view)")
    print("  (the model assumes attacking phases well above 50% and defending phases well below)")
    for codes, name in ((model.ATTACKING_CODES, "attacking phases: Team has ball"),
                        (model.DEFENDING_CODES, "defending phases: Team has ball")):
        agree = total = 0
        for event in events:
            if (event["code"] or "").strip().lower() not in codes:
                continue
            side = model.tag_side(event["tags"].get("Team"), home, away)
            if side is None:
                continue
            part = llr[event["start_frame"]:event["end_frame"]]
            part = part[np.abs(part) > 0.5]
            says_home = part > 0
            agree += int(np.sum(says_home if side == model.H else ~says_home))
            total += len(part)
        print("  {:<32} {:.0f}% of {} frames".format(
            name, 100.0 * agree / total if total else float("nan"), total))

    _, kickoffs = model.frame_scores(atd, events, home, away)
    print("\nkickoffs found at video frames {}  (published: {})".format(
        kickoffs, list(PUBLISHED_KICKOFFS)))
    for published in PUBLISHED_KICKOFFS:
        fps = GAMES[GAME]["fps"]
        near = [k for k in kickoffs if abs(k - published) <= KICKOFF_SEARCH_S * fps]
        if near:
            nearest = min(near, key=lambda k: abs(k - published))
            print("  {} -> off by {:+.1f} s".format(published, (nearest - published) / fps))
        else:
            print("  {} -> not found within {} s".format(published, KICKOFF_SEARCH_S))


if __name__ == "__main__":
    main()
