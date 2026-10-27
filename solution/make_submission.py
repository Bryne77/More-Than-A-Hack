"""Data in, submission out: the one command that regenerates our possession file.

    pip install -r requirements.txt
    python solution/make_submission.py --team <team_name>
    python evaluation/validate_submission.py evaluation/results/<team_name>.csv

Reads the tactical ATD and the Smart Tagging from data/ (or MTAH_DATA_DIR), writes
evaluation/results/<team_name>.csv, and prints the possession counts it produced. Deterministic:
the same data gives the same file, byte for byte.
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, half_video_blocks, require  # noqa: E402
import load_smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
import atd as atd_loader  # noqa: E402
import model  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
FPS = 25


def team_names(meta_path):
    """(home, away) club names from the ATD metadata, in file order."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def summarise(labels, game=GAME):
    """Possession counts per half, from our own output only -- no ground truth involved."""
    lines = []
    for half, first, last in half_video_blocks(game):
        part = labels[first:last + 1]
        changes = np.flatnonzero(part[1:] != part[:-1]) + 1
        bounds = np.concatenate(([0], changes, [len(part)]))
        runs = [(part[a], b - a) for a, b in zip(bounds[:-1], bounds[1:])]
        spells = [n for label, n in runs if label in ("H", "A")]
        stops = [n for label, n in runs if label == "D"]
        alive = sum(n for label, n in runs if label in ("H", "A"))
        home = sum(n for label, n in runs if label == "H")
        lines.append(
            "half {}: {} possessions, mean {:.1f} s | {} stoppages, {:.0f}% of time dead | "
            "home share of alive time {:.0f}%".format(
                half, len(spells), np.mean(spells) / FPS if spells else 0.0, len(stops),
                100.0 * (len(part) - alive) / len(part), 100.0 * home / alive if alive else 0.0))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; the file is <team>.csv")
    parser.add_argument("--data-dir", default=None, help="defaults to data/ or MTAH_DATA_DIR")
    parser.add_argument("--out", default=RESULTS_DIR)
    args = parser.parse_args()

    started = time.time()
    meta_path = require(["atd_meta"], GAME, args.data_dir)[0]
    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))

    print("reading tracking ...")
    atd = atd_loader.load(GAME, args.data_dir)
    print("  {} frames, {} track slots, ball tracked on {:.1f}% of frames".format(
        len(atd["ball"]), len(atd["team"]), 100.0 * np.mean(~np.isnan(atd["ball"][:, 0]))))

    events = load_smart_tagging.load(GAME, args.data_dir)
    print("  {} Smart Tagging instances".format(len(events)))

    print("decoding ...")
    labels = model.predict(atd, events, home, away)

    path = write_submission(os.path.join(args.out, "{}.csv".format(args.team)), labels)
    reason = pm.validate_submission(pm.read_label_csv(path), len(labels))
    print("\nwrote {}  ({:.0f} s)".format(os.path.relpath(path, ROOT), time.time() - started))
    print("format check: {}".format(reason or "OK"))
    for line in summarise(labels):
        print("  " + line)
    if reason:
        sys.exit(1)


if __name__ == "__main__":
    main()
