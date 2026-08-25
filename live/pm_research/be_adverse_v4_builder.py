"""v4 SWEEP-ORIENTED, HORIZON-MATCHED toxicity score. Builder — the frozen artifact.

SURFACE AUTHORISATION (R-126, in-file): coordinator's R-144 commission.

WHY THESE FEATURES — MECHANISM, WHICH IS THE ONLY JUSTIFICATION OFFERED
----------------------------------------------------------------------
R-137/R-138 killed in-sample verdicts, and this file resurrects none. No
"crosses zero", no "beats null", no in-sample claim of any kind appears here or
in the receipt. The design stands on mechanism and is judged on forward tape.

The mechanism: drift is measured to concentrate in ~7-10% of fills. That is the
signature of a SMALL NUMBER OF LARGE INFORMED TAKES, not diffuse toxicity. An
informed sweep consumes MULTIPLE PRICE LEVELS, RAPIDLY, IN ONE DIRECTION, and
the information that motivated it keeps moving price afterwards -- which is
exactly what makes a quote it hits adversely selected. So the score looks for
sweep-in-progress at the fill's knowledge time:

  sw_levels_1s    distinct price levels taken in 1 s  -- a sweep walks the book;
                  ordinary two-way flow does not
  sw_purity_1s    |signed| / total volume             -- one-sidedness
  sw_maxrun_1s    longest same-direction trade run    -- ONE actor, not churn
  sw_signed_1s    signed aggressive volume            -- direction and size
  sw_burst_1s_10s notional(1 s) / mean notional(10 s) -- intensity vs baseline
  sw_levels_5s    levels over 5 s                     -- slower multi-level walk

WHAT v2 TAUGHT AND v4 KEEPS: v2's skill decomposed into SELECTION (book state:
imbalance, tick intensity, spread) and TIMING (returns). TIMING FAILED ITS OWN
PLACEBO on both coins. So v4 carries the state channel and DROPS returns
entirely -- a sweep score is a state score, and returns are the timing channel
R-122 already closed.

HORIZON-MATCHED (v3's lesson): the target is 5-SECOND DRIFT, the quantity that
actually harms a resting quote, not the 5-minute settlement v2 predicted.

CLOCK: mm_hf column 1 (`recv_ns`) only. Columns 2/3 (E_ms, T_ms) are exchange
payload timestamps and are never read. Features at decision time t use only bars
with sec < floor(t - 0.250); the bar containing the fill is EXCLUDED, so no
feature can contain the event that caused the fill.
"""
from __future__ import annotations
import json, math
from pathlib import Path

LAG = 0.250
SWEEP_FEATURES = ["sw_levels_1s", "sw_purity_1s", "sw_maxrun_1s",
                  "sw_signed_1s", "sw_burst_1s_10s", "sw_levels_5s"]
STATE_FEATURES = ["bn_imb_5s", "bn_ticks_5s", "bn_spread_5s"]
V4_FEATURES = SWEEP_FEATURES + STATE_FEATURES


def load_jsonl(path: Path) -> dict[int, dict]:
    out = {}
    with Path(path).open() as fh:
        for line in fh:
            b = json.loads(line)
            out[b["sec"]] = b
    return out


def sweep_feats(sw: dict[int, dict], t_dec: float) -> dict | None:
    """Sweep features at knowledge time. STRICT: only bars with sec < cut."""
    cut = int(math.floor(t_dec - LAG))
    w1 = [sw[s] for s in range(cut - 1, cut) if s in sw]
    w5 = [sw[s] for s in range(cut - 5, cut) if s in sw]
    w10 = [sw[s] for s in range(cut - 10, cut) if s in sw]
    if not w1 or not w10:
        return None
    n1 = sum(b["notional"] for b in w1)
    n10 = sum(b["notional"] for b in w10) / len(w10)
    tot1 = sum(b["total_qty"] for b in w1)
    sg1 = sum(b["signed_qty"] for b in w1)
    return {
        "sw_levels_1s": max(b["n_levels"] for b in w1),
        "sw_purity_1s": (abs(sg1) / tot1) if tot1 > 0 else 0.0,
        "sw_maxrun_1s": max(b["max_run"] for b in w1),
        "sw_signed_1s": sg1,
        "sw_burst_1s_10s": (n1 / n10) if n10 > 0 else 0.0,
        "sw_levels_5s": (max(b["n_levels"] for b in w5) if w5 else 0),
    }


def ols(X, y, lam=1e-3):
    k = len(X[0])
    A = [[sum(X[i][a] * X[i][b] for i in range(len(X))) + (lam if a == b else 0)
          for b in range(k)] for a in range(k)]
    g = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(k)]
    M = [A[i][:] + [g[i]] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c])); M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for cc in range(c, k + 1):
                    M[r][cc] -= f * M[c][cc]
    return [M[i][k] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(k)]
