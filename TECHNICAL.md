# Ball possession as a decoded sequence — technical document

*Draft. Sections marked **TODO** are filled in once the pipeline has run on the real match.*

## The approach

The baseline labels each frame on its own (nearest player to the ball) and then smooths with a
majority vote. A vote over a window has no notion that a possession is one object with a beginning
and an end, which is exactly what the ranking metric, `spellF1`, scores. So we decode the
possessions directly:

1. **Evidence per frame.** Each frame gets an additive score for each of the three states `H`,
   `A` and `D`, built from the tracking and the Smart Tagging (below).
2. **Viterbi decoding per half** over `{H, A, D}`, with a fixed cost on every change of state:
   40 for `H`↔`A`, 25 for going dead, 25 for coming back alive.

The switching cost is how *controlled possession* is put into practice. A change of team is only
decoded when the new team's evidence over a stretch is greater than the cost of changing. About
one second of clear control is enough; a touch, a deflection or a ball flying past a player is
not. Frames with no ball tracked carry no team evidence, so the decoder keeps the current state
through them. That matches the rule that loose and contested balls stay with the last team in
possession.

We chose this over a learned classifier because there are no labels we may train on for this
match, and the possession rules are easier to state as costs and evidence than to learn from
another game's labels.

## Assets used, and how

| asset | used for |
|---|---|
| tactical ATD, players | distance from the ball to each team's nearest player; median player speed |
| tactical ATD, ball | ball position, ball speed (centred difference over ±2 frames), ball off the pitch |
| Smart Tagging, set pieces | a dead stretch before each restart and a live stretch after it, with the restarting team when tagged; the two `Kick Off (Start)` tags mark where each half's opening stoppage ends |
| Smart Tagging, phases | a weak team prior: attacking phases favour their `Team`, defending phases favour the other team |
| homography, videos, broadcast | **not used** |

### Evidence in detail

- **Ball control** (only when the ball is tracked). The score is the lead of one team's nearest
  player over the other team's nearest player, capped at 3 m. It is weighted down when nobody is
  within reach of the ball and cut to a quarter when the ball moves faster than 6 m/s (a pass or
  shot in flight).
- **Dead ball.** Ball more than 0.3 m outside the pitch: strong `D`. Median player speed below
  about 1.6 m/s, smoothed over 2 s: `D`, because players stop running when play stops. A ball
  moving faster than 4 m/s on the pitch: alive.
- **Restarts.** For each set-piece tag, the restart is the first frame where the ball is kicked
  (above 3 m/s) on the pitch. Before it is dead, and the 2 s after it are alive and belong to the
  tagged team. Each half is dead until its own detected kickoff.

Every constant is set from football and tracking knowledge (running speeds, pass speeds, reach)
and is listed with its reasoning at the top of `solution/model.py`. **None is fitted to labels of
this match.** The public 5 minutes are used only to check that the pipeline is wired correctly.

## Training data

None. The method has no learned parameters.

## Possession rules implemented

- A team keeps possession through loose balls, aerial duels, deflections and tracking dropouts
  until the other team shows sustained control. That means about a second of clearly nearer
  players, worth the 40-point switching cost.
- A pass belongs to the passer's team while in flight, because evidence from a fast ball is
  discounted.
- Dead ball: the ball is off the pitch, or play has visibly stopped (players walking), or a
  restart is about to be taken. Possession resumes at the restart kick.
- Dead-ball ownership is not modelled; `D` carries no team, as the challenge requires.

## Limitations

- The constants are reasoned, not measured, so the balance between switching cost and evidence
  may be off for real tracking noise. **TODO** after the first real run.
- Without a tracked ball there is no team evidence. Long dropouts in crowded areas, where changes
  of possession often happen, are decoded late or missed.
- Restart detection depends on the set-piece tags, which are automatic and miss some restarts.
- Not used yet: the homography, the videos and the broadcast feed.

## Validation

- `tests/test_pipeline.py` builds a synthetic match in the real file formats and checks the whole
  pipeline end to end against the simulation's own truth. This tests mechanics only. Synthetic
  play is far cleaner than real tracking, so its scores say nothing about the real match.
- `solution/check_data.py` checks the model's assumptions on the real bundle without labels:
  team-name mapping, the sign of the phase tags against the tracking's own evidence, and kickoff
  detection against the published kickoff frames.
- **TODO**: public-window result (pipeline check only, per EVALUATION.md).

## Reproduction

Python 3.12–3.14.

```bash
pip install -r requirements-solution.txt
# unpack the bundle into data/ or set MTAH_DATA_DIR
python solution/make_submission.py --team <team_name>
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

The output is deterministic: the same data and the same pinned versions produce the same file.
