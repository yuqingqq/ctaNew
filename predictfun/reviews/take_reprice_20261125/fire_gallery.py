"""A gallery of real fires against normal conditions (USER 2026-11-27: "can you show me more charts of real fires vs normal conditions").
Four consumed days; declared before the run. Not chosen by eye: 8 of the top taker's fires drawn at random (seeded) from VISUAL_TWINS'
fires; for each, its nearest look-alike on the nine key book inputs (VISUAL_TWINS, whatever its outcome) and an ordinary moment = a random
moment of the same day and direction within +-30 min (seeded), with their Binance futures windows (SIGNAL_VS_NOISE's window()).
Output (write-once): fire_gallery_examples.json. Drawn by fire_gallery_chart.py.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/fire_gallery.py [<dir>]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import signal_vs_noise as SVN  # noqa: E402
import visual_twins as VT  # noqa: E402

C, TR, DAYS = SVN.C, SVN.TR, SVN.DAYS
N_FIRES, SEED, WIN_MS = 8, 20261127, 1800000.


def entry(P, Z, i, f=None):
    e = dict(day=str(P.day.iat[i]), t=float(P.t.iat[i]), d=float(P.d.iat[i]), y=float(P.y.iat[i]), score=float(P.s_gbm_v3.iat[i]),
             key_inputs={c: float(P[c].iat[i]) for c in VT.KEY})
    if f is not None:
        e['distance_to_fire'] = float(np.sqrt(((Z[i] - Z[f]) ** 2).sum()))
    return e


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (900, 901))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'fire_gallery_examples.json'
    C.require(not fn.exists(), 'fire_gallery_examples.json exists (write-once)')
    P, Z = VT.load()
    T = VT.twins(P, Z)
    F = np.array(sorted(T))
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(F, N_FIRES, replace=False))
    t = P.t.to_numpy(float)
    out = []
    for f in pick:
        f = int(f)
        look = int(T[f]['any'][0][0])
        cand = np.flatnonzero((P.kind.to_numpy() == 'random') & (P.day.to_numpy() == P.day.iat[f]) & (P.d.to_numpy() == P.d.iat[f]) & (np.abs(t - t[f]) <= WIN_MS))
        C.require(len(cand) > 0, 'no ordinary moment within 30 min')
        ordi = int(rng.choice(cand))
        s = dict(fire=entry(P, Z, f), lookalike=entry(P, Z, look, f), ordinary=entry(P, Z, ordi, f))
        for k, e in s.items():
            C.require(e['day'] in DAYS, 'an undeclared day')
            e.update(SVN.window(e['day'], e['t'], e['d']))
        out.append(s)
        print({k: (e['day'], round(e['t']), e['d'], e['y'], round(e['score'], 3), e['first_move_their_way_ms']) for k, e in s.items()}, flush=True)
    TR.dump(fn, dict(asof='2026-11-27', seed=SEED, fires_in_pool=int(len(F)), sets=out))


if __name__ == '__main__':
    main()
