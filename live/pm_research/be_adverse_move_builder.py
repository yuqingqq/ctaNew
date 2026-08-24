"""BE adverse-move candidate builder — the ONE artifact the freeze pins.

SURFACE AUTHORISATION (R-126, in-file and mandatory): this file exists by the
coordinator's R-132 re-freeze order. It is not new capability — it is the v1
builder, which lived in a scratch directory and therefore could not be re-run,
re-hashed or verified by anyone. That is DA's defect in its strongest form: the
v1 freeze pinned a builder THAT WAS NOT IN THE REPOSITORY AT ALL.

R-133(b): the source profile is USERSPACE_KNOWLEDGE_TIME, NOT
exact_receipt_events. `recv_ns` is stamped when a userspace reader parsed the
line, not when the NIC saw it; the v1 profile overstated that.

CLOCK: mm_hf column 1 (`recv_ns`) only. Columns 2/3 (E_ms, T_ms) are exchange
payload timestamps and are never read anywhere in this file.

Sections below are the v1 scratch builder, unchanged in behaviour:
  build_bars()  bookTicker -> knowledge-time second bars
  feats()       bars -> features, strict cutoff floor(t - 0.250)
  fit_logk()    logistic; the pm_logit baseline is HARD-CODED at the call site
"""


# ---------------- from build_bn.py ----------------
import gzip, sys, json
from pathlib import Path

def build_bars(SYM, OUT):
    """bookTicker -> knowledge-time second bars. recv_ns (col 1) ONLY."""
    OUT = Path(sys.argv[2])
    root = Path(f'data/mm_hf/raw/bookTicker/{SYM}')
    files = sorted(root.glob('*.csv.gz'))

    bars = {}          # sec -> [last_mid, n_ticks, sum_imb, hi, lo, first_mid, sum_spread]
    for f in files:
        with gzip.open(f, 'rb') as fh:
            for line in fh:
                p = line.split(b',')
                if len(p) < 8:
                    continue
                try:
                    recv = int(p[0])                      # COLUMN 1 ONLY
                    bq = float(p[5]); aq = float(p[7])
                    bp = float(p[4]); ap = float(p[6])
                except ValueError:
                    continue
                sec = recv // 1_000_000_000
                mid = (bp + ap) / 2.0
                imb = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0.0
                b = bars.get(sec)
                if b is None:
                    bars[sec] = [mid, 1, imb, mid, mid, mid, ap - bp]
                else:
                    b[0] = mid; b[1] += 1; b[2] += imb
                    if mid > b[3]: b[3] = mid
                    if mid < b[4]: b[4] = mid
                    b[6] += ap - bp

    with OUT.open('w') as out:
        for sec in sorted(bars):
            last, n, simb, hi, lo, first, sspr = bars[sec]
            out.write(json.dumps({
                "sec": sec, "last": last, "first": first, "hi": hi, "lo": lo,
                "n_ticks": n, "imb_mean": simb / n, "spread_mean": sspr / n,
            }) + "\n")
    print(f"{SYM}: {len(bars):,} second-bars -> {OUT}")


# ---------------- from fit.py ----------------
import json, math, sys
from pathlib import Path
from collections import defaultdict

S = Path('/tmp/claude-1001/-home-yuqing-ctaNew/4d51dee5-e81e-484a-8b0e-fe88a41ff88a/scratchpad/adv')
LAG = 0.250
COMPLETE_DAYS = {"2026-08-21", "2026-08-22", "2026-08-23"}   # 08-20/08-24 partial

def load_bars(coin):
    f = S / f"bn_{coin}.jsonl"
    bars = {}
    with f.open() as fh:
        for line in fh:
            b = json.loads(line)
            bars[b["sec"]] = b
    return bars

def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))

def feats(bars, t_dec, t0):
    """Only bars with sec < floor(t_dec - LAG). Strict: excludes the decision's bar."""
    cut = int(math.floor(t_dec - LAG))
    last = None
    for s in range(cut - 1, cut - 6, -1):
        if s in bars: last = bars[s]; break
    if last is None: return None, "NO_BAR_AT_CUTOFF"
    def mid_at(sec):
        for s in range(sec, sec - 30, -1):
            if s in bars: return bars[s]["last"]
        return None
    m0 = mid_at(int(t0))
    if not m0: return None, "NO_BAR_AT_T0"
    win = [bars[s] for s in range(cut - 60, cut) if s in bars]
    if len(win) < 30: return None, "LOOKBACK_TOO_SPARSE"
    miss = 60 - len(win)
    runs, run = 0, 0
    for s in range(cut - 60, cut):
        if s in bars:
            runs = max(runs, run); run = 0
        else:
            run += 1
    runs = max(runs, run)
    w5 = [bars[s] for s in range(cut - 5, cut) if s in bars]
    if not w5: return None, "NO_5S_WINDOW"
    r = lambda a, b: math.log(a / b) if a and b and a > 0 and b > 0 else 0.0
    return {
        "bn_ret_since_t0": r(last["last"], m0),
        "bn_ret_5s":  r(last["last"], mid_at(cut - 5) or last["last"]),
        "bn_ret_30s": r(last["last"], mid_at(cut - 30) or last["last"]),
        "bn_ret_60s": r(last["last"], mid_at(cut - 60) or last["last"]),
        "bn_rng_60s": r(max(b["hi"] for b in win), min(b["lo"] for b in win)),
        "bn_imb_5s": sum(b["imb_mean"] for b in w5) / len(w5),
        "bn_ticks_5s": sum(b["n_ticks"] for b in w5),
        "bn_spread_5s": sum(b["spread_mean"] for b in w5) / len(w5),
        "_miss_secs": miss, "_max_blind_run": runs,
    }, None

def spearman(x, y):
    n = len(x)
    if n < 10: return None
    def rank(v):
        o = sorted(range(n), key=lambda i: v[i]); rk = [0.0]*n; i = 0
        while i < n:
            j = i
            while j+1 < n and v[o[j+1]] == v[o[i]]: j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j+1): rk[o[k]] = avg
            i = j + 1
        return rk
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx)/n, sum(ry)/n
    num = sum((a-mx)*(b-my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a-mx)**2 for a in rx)); dy = math.sqrt(sum((b-my)**2 for b in ry))
    return num/(dx*dy) if dx and dy else None

def logistic_1d(x, y, iters=300):
    """y ~ a + b*x by Newton steps. The BASELINE the features must beat."""
    a = b = 0.0
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for xi, yi in zip(x, y):
            p = 1.0/(1.0+math.exp(-(a+b*xi))); w = p*(1-p); e = yi-p
            g0 += e; g1 += e*xi; h00 += w; h01 += w*xi; h11 += w*xi*xi
        det = h00*h11 - h01*h01
        if abs(det) < 1e-12: break
        da = ( h11*g0 - h01*g1)/det; db = (-h01*g0 + h00*g1)/det
        a += da; b += db
        if abs(da) < 1e-10 and abs(db) < 1e-10: break
    return a, b


# The FIT DRIVER is deliberately not in this file. A builder that runs a fit
# on import cannot be imported to verify itself, and the freeze pins the
# BUILDER, not a particular scoring run.

