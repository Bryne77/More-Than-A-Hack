"""A fake match in the real bundle's file formats, with its own known answer.

Used to test the pipeline end to end without the gated data: the ATD metadata XML, the raw ATD
text, and the Smart Tagging XML are written exactly as DATA.md and the starter loaders describe
them, from a simulated game whose true H/A/D labels are known because they drove the simulation.

This is NOT a model of real football and nothing should be tuned against it. It is a mechanics
test: frames line up, files parse, possessions come out as whole objects, the decoder does not
flicker on deflections or lose the thread across ball dropouts.
"""

import os

import numpy as np

# Params
N_FRAMES = 199319
FPS = 25
N_SLOTS = 46
PER_TEAM = 11
LENGTH, WIDTH = 105.3, 68.0
HOME_NAME, AWAY_NAME = "FC Barcelona", "Athletic Club"
TAG_NAMES = {0: "Barcelona", 1: "Atletic Club"}   # spelt the way the real tagging spells them


def simulate(rng, first, n_frames):
    """(truth labels, ball xy, player xy, player speed, tags) for frames first..first+n_frames.

    Positions are in metres, NaN where untracked."""
    truth = np.full(n_frames, "D", dtype="<U1")
    ball = np.full((n_frames, 2), np.nan)
    pos = np.zeros((n_frames, 2 * PER_TEAM, 2))
    tags = []

    # Players wander around a formation; the formation shifts with the ball.
    home_base = np.column_stack([rng.uniform(10, 50, PER_TEAM), rng.uniform(5, 63, PER_TEAM)])
    away_base = np.column_stack([rng.uniform(55, 95, PER_TEAM), rng.uniform(5, 63, PER_TEAM)])
    base = np.vstack([home_base, away_base])
    drift = np.zeros_like(base)

    t = 0
    team = int(rng.integers(2))
    # Kick-off: a dead stretch, then play.
    segments = [("D", int(rng.integers(6, 10) * FPS), None)]
    while sum(n for _, n, _ in segments) < n_frames:
        for _ in range(int(rng.integers(1, 5))):
            segments.append(("P", int(rng.uniform(4, 25) * FPS), team))
            team = 1 - team
        segments.append(("D", int(rng.uniform(5, 20) * FPS), None))
        team = int(rng.integers(2))

    restart_team = team
    for kind, length, side in segments:
        if t >= n_frames:
            break
        stop = min(n_frames, t + length)
        if kind == "D":
            # Stoppage: everyone walks, the ball sits on a line or goes untracked.
            spot = np.array([rng.uniform(0, LENGTH), rng.choice([-1.5, WIDTH + 1.5])])
            for i in range(t, stop):
                drift += rng.normal(0, 0.02, drift.shape)
                pos[i] = base + drift
            if rng.random() < 0.6:
                ball[t:stop] = spot
            truth[t:stop] = "D"
            restart_team = int(rng.integers(2))
            restart = stop
            if restart < n_frames:
                is_kickoff = t == 0
                tags.append({
                    "code": "SET PIECES",
                    "start": first + restart - 2 * FPS, "end": first + restart + 3 * FPS,
                    "Team": "N/A" if rng.random() < 0.3 and not is_kickoff
                            else TAG_NAMES[restart_team],
                    "Type": "Kick Off (Start)" if is_kickoff else "Throw-in",
                })
        else:
            if t > 0 and truth[t - 1] == "D":
                side = restart_team
            members = np.arange(PER_TEAM) + side * PER_TEAM
            carrier = int(rng.choice(members))
            flight = None
            for i in range(t, stop):
                drift += rng.normal(0, 0.08, drift.shape)
                pos[i] = base + drift
                if flight is None and rng.random() < 0.02:
                    target = int(rng.choice(members))
                    flight = (pos[i, carrier].copy(), target, i,
                              max(2, int(np.linalg.norm(pos[i, target] - pos[i, carrier])
                                         / 15.0 * FPS)))
                if flight is not None:
                    origin, target, launched, duration = flight
                    frac = (i - launched) / duration
                    ball[i] = origin + frac * (pos[i, target] - origin)
                    if frac >= 1:
                        carrier, flight = target, None
                else:
                    ball[i] = pos[i, carrier] + rng.normal(0, 0.2, 2)
                    # Keep the nearest opponent a few metres off the carrier.
                    others = np.arange(PER_TEAM) + (1 - side) * PER_TEAM
                    near = others[np.argmin(np.linalg.norm(pos[i, others] - ball[i], axis=1))]
                    gap = pos[i, near] - ball[i]
                    dist = np.linalg.norm(gap)
                    if dist < 3.0:
                        pos[i, near] = ball[i] + gap / max(dist, 1e-6) * 3.0
            # A deflection: an opponent touches it for a few frames, possession does not change.
            if stop - t > 4 * FPS and rng.random() < 0.5:
                k = t + int(rng.integers(FPS, stop - t - FPS))
                opponent = int(rng.choice(np.arange(PER_TEAM) + (1 - side) * PER_TEAM))
                ball[k:k + 5] = pos[k:k + 5, opponent]
            truth[t:stop] = "HA"[side]
            code = "BUILD UP" if rng.random() < 0.5 else "PROGRESSION"
            tags.append({"code": code, "start": first + t, "end": first + stop,
                         "Team": TAG_NAMES[side], "Type": None})
        t = stop

    # Tracking noise: the ball drops out in chunks (~40% of frames), players occasionally.
    i = 0
    while i < n_frames:
        gap = int(rng.exponential(2.0 * FPS))
        if rng.random() < 0.4:
            ball[i:i + gap] = np.nan
        i += gap + 1
    speed = np.zeros((n_frames, 2 * PER_TEAM))
    speed[1:] = np.linalg.norm(pos[1:] - pos[:-1], axis=2) * FPS
    pos[rng.random(pos.shape[:2]) < 0.05] = np.nan
    return truth, ball, pos, speed, tags


def write_bundle(folder, blocks, seed=0):
    """Write data/tactical/* for the given (first, n_frames) blocks; return the full truth array."""
    rng = np.random.default_rng(seed)
    tactical = os.path.join(folder, "tactical")
    os.makedirs(tactical, exist_ok=True)

    truth = np.full(N_FRAMES, "X", dtype="<U1")
    ball = np.full((N_FRAMES, 2), np.nan)
    players = np.full((N_FRAMES, N_SLOTS, 2), np.nan)
    speed = np.full((N_FRAMES, N_SLOTS), np.nan)
    tags = []
    # 11 home players in slots 0-10, 11 away players in slots 23-33; the rest stay empty.
    slots = np.concatenate([np.arange(PER_TEAM), 23 + np.arange(PER_TEAM)])
    for first, n in blocks:
        t, b, p, s, g = simulate(rng, first, n)
        truth[first:first + n] = t
        ball[first:first + n] = b
        players[first:first + n][:, slots] = p
        speed[first:first + n][:, slots] = s
        tags.extend(g)

    with open(os.path.join(tactical, "BAR-ATH_tactical_atd.xml"), "w") as handle:
        handle.write('<?xml version="1.0" encoding="utf-8"?>\n<main><Metadata><Teams>\n')
        handle.write('<Team id="T1"><Name>{}</Name></Team>\n'.format(HOME_NAME))
        handle.write('<Team id="T2"><Name>{}</Name></Team>\n'.format(AWAY_NAME))
        handle.write('</Teams><Players>\n')
        for slot in range(N_SLOTS):
            handle.write('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
                         '<ShirtNumber>{0}</ShirtNumber></Player>\n'.format(
                             slot + 1, "T1" if slot < 23 else "T2"))
        handle.write('</Players></Metadata></main>\n')

    def fmt(x):
        return "NaN" if np.isnan(x) else "{:.5f}".format(x)

    with open(os.path.join(tactical, "BAR-ATH_tactical_atd.txt"), "w") as handle:
        for frame in range(N_FRAMES):
            groups = ";".join("{},{},{}".format(fmt(players[frame, k, 0] / LENGTH),
                                                fmt(players[frame, k, 1] / WIDTH),
                                                fmt(speed[frame, k]))
                              for k in range(N_SLOTS))
            handle.write("{}:{}:{},{},{}\n".format(
                frame + 1, groups, fmt(ball[frame, 0] / LENGTH), fmt(ball[frame, 1] / WIDTH),
                "NaN"))

    with open(os.path.join(tactical, "BAR-ATH_tactical_smart_tagging.xml"), "w") as handle:
        handle.write("<file><ALL_INSTANCES>\n")
        for k, tag in enumerate(tags):
            labels = "".join("<label><group>{}</group><text>{}</text></label>".format(g, tag[g])
                             for g in ("Team", "Type") if tag.get(g))
            handle.write("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end>"
                         "<code>{}</code>{}</instance>\n".format(
                             k, tag["start"] / FPS, tag["end"] / FPS, tag["code"], labels))
        handle.write("</ALL_INSTANCES></file>\n")
    return truth
