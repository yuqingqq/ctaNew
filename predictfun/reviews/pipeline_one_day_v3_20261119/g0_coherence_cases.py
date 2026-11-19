"""G0 (plan 0c3d2523 step 0) boundary cases of the benchmark v3 coherence flag, declared before they ran: pf_pipeline_bench.predict on stub model
heads (the committed source, unchanged since its v3 run) for a valid forecast, an impossible endpoint, zero expected quantity with E[q * move]
-100 / +100 / 0 c, a certain no-fill (P = 0), a certain fill (P = 1), E[q | fill] clipped at the remaining with E[q] / P rounding above it,
nonfinite heads (NaN, inf), and E[q * move] 5e-10 / 5e-9 c at E[q] = 0 (inside / beyond the declared 1e-9 tolerance). Each row's flag must
equal the expected one AND the shared contract's verdict on the rebuilt two-atom forecast (pf_pipeline_contract: accepted, and map_value gives
back E[q], E[q * move] and the value within 1e-9, the rebuilt E[q | fill] set back to the remaining within the same tolerance). No data read.
Usage (run_light): g0_coherence_cases.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'research'))
import pf_pipeline_bench as BX                            # noqa: E402
import pf_pipeline_contract as PC                         # noqa: E402


class Stub:
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def predict(self, X, num_threads=1):
        return self.a


pb = next(x for x in np.linspace(.05, .95, 9001) if (x * 30.) / x > 30.)
CASES = [  # name, P, E[q | fill] head, E[q * move | fill] head, side, expected flag
    ('valid', .5, 10., 1., 0, True), ('implied endpoint mid below 0', .5, 10., -1000., 0, False),
    ('E[q] = 0, E[q * move] -100 c', .5, -1., -200., 0, False), ('E[q] = 0, E[q * move] +100 c', .5, -1., 200., 1, False),
    ('E[q] = 0, E[q * move] 0', .5, -1., 0., 0, True), ('certain no-fill (P = 0)', 0., 10., 5., 0, True), ('certain fill (P = 1)', 1., 10., 1., 1, True),
    ('E[q | fill] clipped at the remaining, E[q] / P above it', pb, 40., 3., 0, True), ('nonfinite E[q * move] head (NaN)', .5, 10., np.nan, 0, False),
    ('nonfinite E[q * move] head (inf)', .5, 10., np.inf, 1, False), ('E[q] = 0, E[q * move] 5e-10 c (inside the tolerance)', .5, -1., 1e-9, 0, True),
    ('E[q] = 0, E[q * move] 5e-9 c (beyond it)', .5, -1., 1e-8, 0, False)]
n = len(CASES); df = pd.DataFrame({x: [0.] * n for x in BX.SETS['J']}).assign(side=[c[4] for c in CASES], p=[49.5 if c[4] == 0 else 50.5 for c in CASES], pf_mid=[50.] * n,
                                                                            fp_remaining=[30.] * n)
pr = BX.predict(dict(p=Stub([c[1] for c in CASES]), q=Stub([c[2] for c in CASES]), qm=Stub([c[3] for c in CASES])), df, 'J')
tol = lambda x: 1e-9 * max(1., abs(x)); ok_all = True
for i, (name, P, qh, qmh, side, want) in enumerate(CASES):
    Eq, Eqm, Vv, flag = float(pr['Eq'][i]), float(pr['Eqm'][i]), float(pr['V'][i]), bool(pr['ok'][i]); sd = 'bid' if side == 0 else 'ask'; rem = 30.
    tg = PC.Target(f'case{i}', sd, 'retain', 1000., 1137., 6000., 50., float(df.p[i]), rem, 'B', 'c', 'e'); verdict = False
    try:
        qf = Eq / P if P > 0 else 0.; mv = Eqm / Eq if Eq > 0 else 0.; qf = rem if rem < qf <= rem + tol(rem) else qf
        atoms = (PC.Atom(P, qf, mv), PC.Atom(1. - P, 0., 0.)) if 0. < P < 1. else (PC.Atom(1., qf if P >= 1 else 0., mv),)
        v = PC.map_value(tg, PC.Forecast(tg, atoms, 1000., 0., 'synthetic', 'g0'))
        verdict = abs(v.quantity - Eq) <= tol(Eq) and abs(v.movement - Eqm) <= tol(Eqm) and abs(v.retained - Vv) <= tol(Vv)
    except ValueError:
        verdict = False
    good = flag == want == verdict; ok_all &= good
    print(f'{"ok " if good else "BAD"} {name:<56} E[q] {Eq:>10.6g} E[q*move] {Eqm:>11.6g} flag {flag!s:<5} expected {want!s:<5} contract {verdict!s:<5}')
print(f'G0 COHERENCE CASES (flag = expected = the contract on all {n}): {"PASS" if ok_all else "FAIL"}')
sys.exit(0 if ok_all else 1)
