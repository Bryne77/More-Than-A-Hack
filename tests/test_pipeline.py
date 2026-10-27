"""End-to-end check of solution/ on a synthetic match in the real file formats.

    python tests/test_pipeline.py

Writes a fake bundle to a temporary folder, runs the same code path as make_submission.py,
validates the file with the official checks and scores it against the simulation's own truth,
next to the starter baseline on the same data. Takes about a minute, mostly writing the fake ATD.
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, os.path.join(ROOT, "solution"), os.path.join(ROOT, "starter"),
             os.path.join(ROOT, "evaluation"), HERE):
    sys.path.insert(0, path)

import atd as atd_loader  # noqa: E402
import baseline  # noqa: E402
import load_smart_tagging  # noqa: E402
import make_submission  # noqa: E402
import model  # noqa: E402
import possession_metrics as pm  # noqa: E402
import synthetic  # noqa: E402
from games import GAME, GAMES  # noqa: E402

# Params
# A stretch of each half -- the decoder still runs over the whole of both halves, the untracked
# rest of which must come out as a stoppage rather than as invented possessions.
BLOCKS = ((11185, 12 * 60 * 25), (110808, 6 * 60 * 25))
MIN_SPELL_F1 = 0.6


def main():
    with tempfile.TemporaryDirectory() as folder:
        truth = synthetic.write_bundle(folder, BLOCKS, seed=7)

        atd = atd_loader.load(GAME, folder)
        assert atd["ball"].shape == (synthetic.N_FRAMES, 2)
        assert len(atd["team"]) == synthetic.N_SLOTS
        first = BLOCKS[0][0]
        seen = ~np.isnan(atd["ball"][first:first + 1000, 0])
        assert seen.any(), "ball never parsed"

        events = load_smart_tagging.load(GAME, folder)
        labels = model.predict(atd, events, synthetic.HOME_NAME, synthetic.AWAY_NAME)
        path = make_submission.write_submission(os.path.join(folder, "test.csv"), labels)
        reason = pm.validate_submission(pm.read_label_csv(path), synthetic.N_FRAMES)
        assert reason is None, reason

        meta, raw = (os.path.join(folder, "tactical", "BAR-ATH_tactical_atd." + ext)
                     for ext in ("xml", "txt"))
        codes = baseline.nearest_player_labels(raw, baseline.team_of_each_track(meta),
                                               synthetic.N_FRAMES, GAMES[GAME]["pitch"])

    gt = pm.label_codes(truth)
    rows = [("solution", labels)] + [("baseline_smooth{}".format(w), baseline.smooth(codes, w))
                                     for w in (25, 50)]
    print("{:<20} {:>8} {:>8} {:>6}  (synthetic truth: {} possessions)".format(
        "", "macroF1", "spellF1", "poss", len(pm.possession_spells(gt, pm.valid_runs(gt)))))
    results = {}
    for name, pred in rows:
        result = pm.evaluate(gt, pm.label_codes(pred))
        results[name] = result
        n_pred = len(pm.possession_spells(pm.label_codes(pred), pm.valid_runs(gt)))
        print("{:<20} {:>8.4f} {:>8.4f} {:>6}".format(
            name, result["macro_f1"], result["spell_f1_mean"], n_pred))

    assert results["solution"]["spell_f1_mean"] >= MIN_SPELL_F1, "solution spellF1 too low"
    assert results["solution"]["spell_f1_mean"] > results["baseline_smooth50"]["spell_f1_mean"]
    print("\nOK")


if __name__ == "__main__":
    main()
