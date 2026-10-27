"""Read the raw ATD into dense numpy arrays, one row per video frame.

The starter baseline streams the file line by line in Python, which takes minutes. Every line of
the raw file has the same shape -- "<frame_id>:<slot>;<slot>;...;<slot>:<ball>" with a fixed
number of slots and fixed fields per group, NaN where nothing was seen -- so the whole file can be
turned into one float array in a single numpy call and reshaped. If any line breaks that shape
the fast path refuses and the slow per-line path takes over; it never guesses.

Coordinates come back in metres (the ATD stores 0-1 fractions of the pitch), speeds in m/s, and
everything is indexed by video frame: row i is submission row i.
"""

import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

# Params
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","
HOME, AWAY = 0, 1


def slot_teams(meta_path):
    """Array over track slots: HOME or AWAY. First <Team> in the metadata is home (FIFA/EPTS),
    which for this match is Barcelona -- the same convention starter/baseline.py relies on."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == team_ids[0] else AWAY for team_id in slots], dtype=np.int8)


def _line_shape(line):
    """(n_slots, fields_per_slot, fields_in_ball) from one raw line."""
    _, players, ball = line.rstrip("\n").split(FIELD_SEP)
    groups = players.split(GROUP_SEP)
    return len(groups), len(groups[0].split(VALUE_SEP)), len(ball.split(VALUE_SEP))


def _parse_fast(text, width):
    """Whole file as an (n_lines, width) float array, or None if the lines are not all `width`."""
    flat = (text.replace("\r", "").replace(FIELD_SEP, VALUE_SEP).replace(GROUP_SEP, VALUE_SEP)
            .replace("\n", VALUE_SEP))
    # numpy's C text parser: reads NaN, and with warnings raised it rejects anything it cannot
    # read instead of silently stopping early.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            values = np.fromstring(flat.rstrip(VALUE_SEP), dtype=np.float64, sep=VALUE_SEP)
    except (ValueError, DeprecationWarning):
        return None
    if values.size % width:
        return None
    table = values.reshape(-1, width)
    # A shape that merely divides evenly is not proof: the frame ids must count up by one.
    if not np.all(np.diff(table[:, 0]) > 0):
        return None
    return table


def _parse_slow(lines, width):
    """Per-line fallback: lines of the wrong shape are skipped, not padded."""
    rows = []
    for line in lines:
        flat = line.strip().replace(FIELD_SEP, VALUE_SEP).replace(GROUP_SEP, VALUE_SEP)
        values = flat.split(VALUE_SEP)
        if len(values) == width:
            rows.append(np.array(values, dtype=np.float64))
    return np.vstack(rows)


def load(game=GAME, data_dir=None, n_frames=None):
    """dict of per-frame arrays over the tactical video:

        ball      (n, 2)          ball x, y in metres; NaN where the ball is not tracked
        players   (n, slots, 2)   x, y in metres per track slot; NaN where the slot is empty
        speed     (n, slots)      the ATD's own speed channel, m/s
        team      (slots,)        HOME / AWAY per slot
    """
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = n_frames or source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]

    with open(raw_path) as handle:
        text = handle.read()
    first = text[:text.index("\n")] if "\n" in text else text
    n_slots, per_slot, per_ball = _line_shape(first)
    row_width = 1 + n_slots * per_slot + per_ball

    table = _parse_fast(text, row_width)
    if table is None:
        table = _parse_slow(text.splitlines(), row_width)
    del text

    team = slot_teams(meta_path)
    if len(team) != n_slots:
        raise ValueError("metadata lists {} slots, raw data has {}".format(len(team), n_slots))

    video_frame = table[:, 0].astype(np.int64) - 1  # ATD counts from 1, submissions from 0
    keep = (video_frame >= 0) & (video_frame < n_frames)
    table, video_frame = table[keep], video_frame[keep]

    groups = table[:, 1:1 + n_slots * per_slot].reshape(-1, n_slots, per_slot)
    ball_fields = table[:, 1 + n_slots * per_slot:]

    scale = np.array([length, width])
    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball[video_frame] = ball_fields[:, :2] * scale
    players[video_frame] = groups[:, :, :2] * scale
    if per_slot >= 3:
        speed[video_frame] = groups[:, :, 2]
    return {"ball": ball, "players": players, "speed": speed, "team": team}
