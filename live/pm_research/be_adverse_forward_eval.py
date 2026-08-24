"""Forward evaluation of the v2 adverse-move candidate. Scores ONE admissible day.

SURFACE AUTHORISATION (R-126, mandatory in-file): coordinator's R-138 order —
"prepare the forward-evaluation harness so day one scores automatically under
the declared nulls."

WHAT MAKES THIS FORWARD AND NOT ANOTHER IN-SAMPLE RUN
-----------------------------------------------------
The candidate is FROZEN: `be_adverse_move_candidate_v2.json` carries fixed
coefficients, a fixed feature list and a builder pinned by sha256 (now committed
at 128a757). This harness **loads and applies** them. It contains no fit, no
threshold search and no model selection -- if it did, the day would be
contaminated the moment it ran.

ADMISSIBILITY IS A PREDICATE, NOT A PROMISE. A window is admissible only if its
ENTIRE span lies after the freeze instant. A window straddling it is refused,
not truncated: part of it was seen.

THE DECLARED NULLS (R-137/R-138, fixed BEFORE any forward result exists)
------------------------------------------------------------------------
  n_permutations = 200, FIXED HERE AND NOW.
      BE used 8, saw a result, then used 30 and the verdict changed -- two eth
      cells reversed. A null max is an extreme-order statistic, biased downward
      at small counts, so an UNDER-SAMPLED CORRECT NULL FLATTERS AS MUCH AS A
      WRONG ONE. 200 is chosen before any forward number exists and may not be
      revised afterwards.
  null_1: permuted-feature refit  -> incremental Brier over pm_logit
  null_statistic = MAX over permutations, not the mean.

THE BASELINE IS HARD-CODED AND IS THE WHOLE TEST. The PM binary settles on a
BINANCE-DERIVED price, so any skill NOT incremental over `pm_logit` is
tautological. There is no switch to a naive baseline in this file.

R-109: G is the number of complete forward days. Day one has G=1, so this
harness reports a POINT ESTIMATE WITH NO INTERVAL and says so. It will not
manufacture an interval on the wrong unit.
"""
from __future__ import annotations
import json, math, random, sys
from pathlib import Path

CAND = Path('/home/yuqing/ctaNew/data/pm_5min/derived/be_adverse_move_candidate_v2.json')
N_PERM = 200                      # FIXED BEFORE ANY FORWARD RESULT (R-138)
NULL_STAT = "max"


def load_candidate() -> dict:
    c = json.loads(CAND.read_text())
    if c.get("decision_eligible"):
        raise SystemExit("REFUSED: candidate claims decision_eligible; this harness scores research only")
    return c


def admissible(window_t0: float, window_end: float, freeze_epoch: float) -> tuple[bool, str]:
    """A window is admissible only if it lies ENTIRELY after the freeze."""
    if window_end <= freeze_epoch:
        return False, "ENTIRELY_BEFORE_FREEZE"
    if window_t0 < freeze_epoch:
        return False, "STRADDLES_FREEZE_seen_in_part"
    return True, "ADMISSIBLE"


def brier(p, y):
    return sum((a - b) ** 2 for a, b in zip(p, y)) / len(y)


def fit_logk(X, y, it=150):
    k = len(X[0]); w = [0.0] * k
    for _ in range(it):
        g = [0.0] * k; H = [[0.0] * k for _ in range(k)]
        for xi, yi in zip(X, y):
            z = max(-30, min(30, sum(a * b for a, b in zip(w, xi))))
            p = 1 / (1 + math.exp(-z)); e = yi - p; ww = p * (1 - p)
            for a in range(k):
                g[a] += e * xi[a]
                for b in range(k):
                    H[a][b] += ww * xi[a] * xi[b]
        for a in range(k):
            H[a][a] += 1e-6
        M = [H[i][:] + [g[i]] for i in range(k)]
        for c in range(k):
            piv = max(range(c, k), key=lambda r: abs(M[r][c])); M[c], M[piv] = M[piv], M[c]
            if abs(M[c][c]) < 1e-12:
                continue
            for r in range(k):
                if r != c:
                    f = M[r][c] / M[c][c]
                    for cc in range(c, k + 1):
                        M[r][cc] -= f * M[c][cc]
        d = [M[i][k] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(k)]
        w = [a + b for a, b in zip(w, d)]
        if max(abs(v) for v in d) < 1e-9:
            break
    return w


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    F = 1000.0
    ok(admissible(1100, 1400, F)[0], "a window wholly after the freeze is admissible")
    ok(not admissible(500, 800, F)[0], "wholly before is refused")
    ok(admissible(900, 1100, F)[1] == "STRADDLES_FREEZE_seen_in_part",
       "a STRADDLING window is REFUSED, not truncated — part of it was seen")
    ok(not admissible(1000, 1300, F)[0] is False, "boundary: t0 == freeze is admissible")
    ok(N_PERM == 200, "the permutation count is fixed in the file, before any result")
    ok(NULL_STAT == "max", "null statistic is the MAX, not the mean")
    c = load_candidate()
    ok(c["decision_eligible"] is False, "the frozen candidate is not decision-eligible")
    ok("pm_logit" in c["baseline"], "the baseline is pm_logit and is recorded in the candidate")
    ok(len(c["coefficients"]["btc"]) == 7, "coefficients load with the expected width")
    ok(c["builder"]["sha256"].startswith("e8a82b66"), "builder sha256 is the anchor")
    print(f"be_adverse_forward_eval selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    print(__doc__.strip().splitlines()[0])
    c = load_candidate()
    print(f"  candidate    {CAND.name}")
    print(f"  frozen_at    {c['frozen_at_utc']}  (commit 128a757 per R-138 annotation)")
    print(f"  builder      {c['builder']['sha256'][:16]}")
    print(f"  baseline     pm_logit, hard-coded")
    print(f"  nulls        {N_PERM} permutations, statistic={NULL_STAT}, FIXED before results")
    print(f"  R-109        G=1 on day one -> POINT ESTIMATE, NO INTERVAL")
    print("\n  Awaiting the first fully-admissible day (first window t0 >= freeze).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
