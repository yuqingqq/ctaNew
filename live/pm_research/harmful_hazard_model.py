"""Items 3-4 of HARMFUL_FILL_HAZARD_TOXICITY_PLAN: PM-native features + the
linear hazard x toxicity reference.

SURFACE AUTHORISATION (R-126, in-file): the user's plan, §10 items 3-4, adopted
by direct instruction.

FEATURES (§4.1, §4.3). Computed from the RAW window tape at each exposure row's
decision time t, using only events with recv <= t - 0.250 (the programme's
frozen PM knowledge lag, same as the engine's STATE_LAG_S). Multiscale windows
0.25/0.5/1/2.5/5 s -- sub-second first, per §4.3's ruling that 1 s bars alone
are 250-1,250 ms stale. All flow features are SIDE-SIGNED so positive means
THREATENING to the resting order: a resting BUY_UP is threatened by sell-
aggressor flow and falling mid; a SELL_UP by the mirror. No feature reads the
event that causes the labelled fill (strict cutoff before t).

MODELS (§5). Per coin, never pooled. The linear reference:
  hazard    logistic  P(any same-generation fill in [t, t+H] | x)   on OK rows
  tox_sign  logistic  P(V_cancel > 0 | fill, x)                     on filled rows
  tox_mag   ridge     E[V_cancel | fill, x]                         on filled rows
  expected_cancel_value = p_fill x tox_mag     (§2)
`P(V_cancel>0)` and the magnitude head are reported SEPARATELY so a noisy
magnitude cannot hide real tail discrimination.

SPLIT (§9): train 2026-08-20/21, development 2026-08-22. R-109: G=1 dev day ->
point estimates, NO intervals. The matched-random control (§8) uses 200 draws,
declared here before any result -- an under-sampled null flatters.

    python3 live/pm_research/harmful_hazard_model.py --selftest
    python3 live/pm_research/harmful_hazard_model.py run
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import flow_intensity as fi

ROWS = fi.PM / "derived/harmful_exposure_rows_v3.json"   # v1/v2 receipts are INVALID (user audits, 2026-08-25)
# v3.1 SEMANTICS (user audit 3): rows are chronological decision rows; the
# label is the value of cancelling THE GENERATION FROM THIS ROW ONWARD — not a
# fixed 1 s horizon. The hazard target is "any tranche ahead of this row"; the
# toxicity target is the latency-aware preventable value at TARGET_LATENCY_MS,
# declared here, an ASSUMED cancel latency for the diagnostic fit only — the
# action evaluator sweeps the full 5–250 ms grid regardless.
TARGET_LATENCY_MS = 50
EXPECTED_SCHEMA = "harmful_exposure_v3_4_fill_scoped_markout"
OUT = fi.PM / "derived/harmful_hazard_model_v1.json"
LAG_S = 0.250
# §4.1 asks for 25/50/100/250/500/1000 ms. The finest micro scales measure the
# burst state AT the knowledge cutoff (a 50 ms window ending at t-250ms), which
# the 250 ms floor of the first pass could not see. PM tape is sparse at 50 ms
# except during bursts -- which is exactly when the answer matters.
SCALES = (0.05, 0.10, 0.25, 0.5, 1.0, 2.5, 5.0)
TRAIN_DAYS = ("2026-08-20", "2026-08-21")
DEV_DAY = "2026-08-22"
BUDGETS = (0.05, 0.10, 0.15)
N_RANDOM = 200                    # matched-random draws, fixed pre-result


# ---------------------------------------------------------------- tape reading
def window_streams(path: Path, up_id: str, down_id: str) -> dict[str, list]:
    """Trades and quotes for one window, sorted by window-relative recv time."""
    slug = path.name.split(".jsonl")[0]
    ws = int(slug.rsplit("-", 1)[1])
    trades: list[tuple[float, int, float]] = []      # (t, dir +1 buyUp, shares)
    quotes: list[tuple[float, float, float]] = []    # (t, bid, ask)
    levels: list[tuple[float, str, float, float]] = []  # (t, side, px, new_total)
    seen_tx: set[str] = set()
    for line in fi._gz_lines(path):
        if not any(m in line for m in (fi.TRADE_MARK, fi.QUOTE_MARK)):
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            recv = int(parts[0]) / 1e9 - ws
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            continue
        if recv < -60.0 or recv > fi.WINDOW_S:
            continue
        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict):
                continue
            et = msg.get("event_type")
            aid = str(msg.get("asset_id"))
            if et == "price_change":
                for pc in msg.get("price_changes", []):
                    if str(pc.get("asset_id")) != up_id:
                        continue
                    try:
                        b = float(pc["best_bid"]); a = float(pc["best_ask"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if 0.0 <= b < a <= 1.0:
                        quotes.append((recv, b, a))
                    try:
                        levels.append((recv, str(pc["side"]).upper(),
                                       float(pc["price"]), float(pc["size"])))
                    except (KeyError, TypeError, ValueError):
                        pass
            elif et == "last_trade_price" and aid in (up_id, down_id):
                tx = str(msg.get("transaction_hash") or "")
                if tx and tx in seen_tx:
                    continue
                if tx:
                    seen_tx.add(tx)
                try:
                    size = float(msg["size"])
                    native = str(msg["side"]).upper()
                except (KeyError, TypeError, ValueError):
                    continue
                side = fi.fold_side(native, aid == down_id)
                trades.append((recv, 1 if side == "BUY" else -1, size))
    trades.sort(); quotes.sort(); levels.sort(key=lambda x: x[0])
    # level-size DELTAS on the UP book: first sight of a price initializes
    # state (absorbed by the 60s pre-window tail), later sights are deltas.
    state: dict[tuple[str, float], float] = {}
    deltas: list[tuple[float, str, float]] = []      # (t, side, delta_shares)
    for t, sd, px, tot in levels:
        k = (sd, px)
        prev = state.get(k)
        state[k] = tot
        if prev is not None and tot != prev:
            deltas.append((t, sd, tot - prev))
    return {"trades": trades, "quotes": quotes, "deltas": deltas,
            "tt": [x[0] for x in trades], "qt": [x[0] for x in quotes],
            "dt": [x[0] for x in deltas]}


# MICRO SCALES READ THE DENSE STREAM. Measured on the scored population: PM
# trades run ~7/sec (a 10 ms trade window is non-empty at only 14.6% of
# decisions) while price_change QUOTES run ~494/sec with p90 inter-event gap
# 6 ms -- the 10 ms information lives in the QUOTE stream. So each scale
# carries quote-side features (event intensity = book churn, and best-bid/ask
# movement toward the resting order) beside the sparse trade-flow ones, which
# at micro scales act as burst detectors rather than continuous signals.
FEATURE_NAMES: list[str] = []
for _w in SCALES:
    _tag = str(_w).replace(".", "p")
    FEATURE_NAMES += [f"threat_flow_{_tag}", f"tot_flow_{_tag}",
                      f"purity_{_tag}", f"maxrun_{_tag}", f"mid_threat_{_tag}",
                      f"quote_rate_{_tag}", f"touch_threat_{_tag}"]
FEATURE_NAMES += ["spread", "dist_to_touch", "qahead", "fill_frac", "quote_age"]


def features(st: dict[str, list], t: float, side: str, level: float,
             resting: float, qahead: float, size: float = 5.0) -> list | None:
    """Feature vector at knowledge time t - LAG_S. None if no quote yet."""
    cut = t - LAG_S
    sgn = -1.0 if side == "BUY_UP" else 1.0     # threatening taker direction
    qi = bisect.bisect_right(st["qt"], cut) - 1
    if qi < 0:
        return None
    qt, bid, ask = st["quotes"][qi]
    mid_now = (bid + ask) / 2.0
    out: list[float] = []
    for w in SCALES:
        lo = bisect.bisect_left(st["tt"], cut - w)
        hi = bisect.bisect_right(st["tt"], cut)
        seg = st["trades"][lo:hi]
        signed = sum(d * s for _t, d, s in seg)
        tot = sum(s for _t, _d, s in seg)
        run = maxrun = 0; last = 0
        for _t, d, _s in seg:
            run = run + 1 if d == last else 1
            last = d
            if run > maxrun: maxrun = run
        j = bisect.bisect_right(st["qt"], cut - w) - 1
        mid_then = ((st["quotes"][j][1] + st["quotes"][j][2]) / 2.0
                    if j >= 0 else mid_now)
        qlo = bisect.bisect_left(st["qt"], cut - w)
        qhi = bisect.bisect_right(st["qt"], cut)
        b_then, a_then = ((st["quotes"][j][1], st["quotes"][j][2])
                          if j >= 0 else (bid, ask))
        # movement of OUR side's best toward the order: for a resting BUY the
        # threat is the best BID falling (the book pulling away above us / the
        # touch collapsing toward our level); mirrored for a SELL.
        touch_threat = ((b_then - bid) if side == "BUY_UP"
                        else (ask - a_then)) * 100.0
        out += [sgn * signed,                       # threat_flow
                tot,                                # tot_flow
                (abs(signed) / tot) if tot > 0 else 0.0,
                float(maxrun),
                sgn * (mid_now - mid_then) * 100.0,  # mid moving against us
                float(qhi - qlo),                   # quote_rate: book churn
                touch_threat]
    dist = (bid - level) if side == "BUY_UP" else (level - ask)
    out += [(ask - bid) * 100.0, dist * 100.0, qahead,
            1.0 - (resting / size if size > 0 else 0.0),
            cut - qt]
    return out


# ------------------------------------------------------------------- modelling
def fit_logistic(X, y, lam=1e-3, it=120):
    k = len(X[0]); w = [0.0] * k
    for _ in range(it):
        g = [0.0] * k; H = [[0.0] * k for _ in range(k)]
        for xi, yi in zip(X, y):
            z = max(-30, min(30, sum(a * b for a, b in zip(w, xi))))
            p = 1 / (1 + math.exp(-z)); e = yi - p; ww = p * (1 - p)
            for a in range(k):
                g[a] += e * xi[a] - (lam * w[a] if a else 0.0) / len(X)
                for b in range(k):
                    H[a][b] += ww * xi[a] * xi[b]
        for a in range(k):
            H[a][a] += lam
        w2 = _solve(H, g, w)
        if w2 is None:
            break
        w, done = w2
        if done:
            break
    return w


def _solve(H, g, w):
    k = len(g)
    M = [H[i][:] + [g[i]] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for cc in range(c, k + 1):
                    M[r][cc] -= f * M[c][cc]
    d = [M[i][k] / M[i][i] for i in range(k)]
    w = [a + b for a, b in zip(w, d)]
    return w, max(abs(x) for x in d) < 1e-9


def fit_ridge(X, y, lam=1.0):
    k = len(X[0])
    H = [[sum(X[i][a] * X[i][b] for i in range(len(X))) + (lam if a == b else 0)
          for b in range(k)] for a in range(k)]
    g = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(k)]
    r = _solve(H, g, [0.0] * k)
    return r[0] if r else [0.0] * k


def predict_p(w, x):
    return 1 / (1 + math.exp(-max(-30, min(30, sum(a * b for a, b in zip(w, x))))))


def zscale(train_X, all_X):
    """Scales IN PLACE over all_X (returns the same list object): building a
    second full copy of an era-scale matrix doubled peak memory and OOM'd."""
    k = len(train_X[0])
    mu = [sum(x[i] for x in train_X) / len(train_X) for i in range(k)]
    sd = [math.sqrt(sum((x[i] - mu[i]) ** 2 for x in train_X) / len(train_X)) or 1.0
          for i in range(k)]
    for j in range(len(all_X)):
        x = all_X[j]
        all_X[j] = [1.0] + [(x[i] - mu[i]) / sd[i] for i in range(k)]
    return all_X, mu, sd


# ------------------------------------------------------------------ evaluation
def budget_eval(dev, scores, budgets=BUDGETS, n_random=N_RANDOM, seed=20260825):
    """Harm captured / value sacrificed at cancellation budgets vs matched random."""
    rng = random.Random(seed)
    harm = [max(r.get("v_cancel_cents", 0.0), 0.0) for r in dev]
    val = [r.get("v_cancel_cents", 0.0) for r in dev]
    total_harm = sum(harm)
    order = sorted(range(len(dev)), key=lambda i: -scores[i])
    out = {}
    for b in budgets:
        k = max(1, int(len(dev) * b))
        top = order[:k]
        cap = sum(harm[i] for i in top)
        net = sum(val[i] for i in top)
        rand_caps = []
        for _ in range(n_random):
            pick = rng.sample(range(len(dev)), k)
            rand_caps.append(sum(harm[i] for i in pick))
        out[f"{int(b*100)}%"] = {
            "n_cancelled": k,
            "harm_captured_share": cap / total_harm if total_harm else 0.0,
            "net_cancel_value_cents": net,
            "random_matched_mean_share": (sum(rand_caps) / n_random / total_harm
                                          if total_harm else 0.0),
            "random_matched_max_share": (max(rand_caps) / total_harm
                                         if total_harm else 0.0),
            "beats_random_max": (cap > max(rand_caps)),
        }
    return out


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    st = {"trades": [(1.0, -1, 5.0), (1.1, -1, 3.0), (1.2, 1, 1.0)],
          "tt": [1.0, 1.1, 1.2],
          "quotes": [(0.5, 0.50, 0.52), (1.15, 0.48, 0.50)],
          "qt": [0.5, 1.15]}
    f = features(st, 2.0, "BUY_UP", 0.50, 5.0, 2.0)
    ok(f is not None and len(f) == len(FEATURE_NAMES), "feature width matches names")
    ok(f[FEATURE_NAMES.index("quote_rate_5p0")] >= 1.0,
       "quote churn is counted from the DENSE stream")
    ok(f[FEATURE_NAMES.index("touch_threat_1p0")] > 0,
       "best bid falling (0.50->0.48) is positive touch threat for a resting BUY")
    i5 = FEATURE_NAMES.index("threat_flow_5p0")
    ok(abs(f[i5] - 7.0) < 1e-9,
       "BUY_UP threatened by sell flow: -( -5-3+1 ) = +7 signed threat")
    # at w=5s the toy's window predates its first quote; the conservative
    # fallback (mid_then = mid_now) correctly reports ZERO change there. The
    # signed-threat assertion belongs at the 1s scale, where the earlier quote
    # (mid 0.51) is inside the window and the mid has FALLEN to 0.49.
    ok(abs(f[FEATURE_NAMES.index("mid_threat_5p0")]) < 1e-9,
       "a window with no earlier quote reports zero mid change, not a fabrication")
    im = FEATURE_NAMES.index("mid_threat_1p0")
    ok(abs(f[im] - 2.0) < 1e-9,
       "falling mid (0.51->0.49) is +2c of threat for a resting BUY")
    ok(features(st, 0.1, "BUY_UP", 0.5, 5.0, 0.0) is None,
       "no quote before knowledge cutoff -> refuse, not fabricate")
    f2 = features(st, 1.35, "BUY_UP", 0.50, 5.0, 2.0)
    lo = bisect.bisect_left(st["tt"], 1.35 - LAG_S - 0.25)
    hi = bisect.bisect_right(st["tt"], 1.35 - LAG_S)
    ok(hi - lo == 2 and abs(f2[FEATURE_NAMES.index("threat_flow_0p25")] - 8.0) < 1e-9,
       "the 1.2s BUY trade is AFTER the knowledge cutoff and is excluded")

    X = [[x] for x in (-2, -1, -0.5, 0.5, 1, 2)] * 20
    y = [1 if x[0] > 0 else 0 for x in X]
    # zscale mutates all_X IN PLACE (memory fix for era scale) — the test must
    # not reuse X afterwards; the first in-place version silently broke the
    # ridge check below by handing it the already-scaled matrix.
    raw = [list(x) for x in X]
    Xs, _, _ = zscale(X, X)
    ok(Xs is X, "zscale scales in place and returns the same object")
    w = fit_logistic(Xs, y)
    ok(predict_p(w, Xs[3]) > 0.9 and predict_p(w, Xs[0]) < 0.1,
       "logistic separates a separable toy")
    wr = fit_ridge([[1.0, x[0]] for x in raw], [2 * x[0] + 1 for x in raw])
    ok(abs(wr[1] - 2.0) < 0.05 and abs(wr[0] - 1.0) < 0.05, "ridge recovers a line")

    dev = [{"v_cancel_cents": v} for v in (10.0, 8.0, -5.0, -5.0, 0.5, -1.0,
                                           0.2, -0.3, 0.1, -2.0)]
    good = [10.0, 8.0, -5.0, -5.0, 0.5, -1.0, 0.2, -0.3, 0.1, -2.0]
    ev = budget_eval(dev, good, budgets=(0.2,), n_random=50)
    ok(ev["20%"]["harm_captured_share"] > 0.9,
       "a perfect score captures the harm at budget")
    # on a 10-row toy a random draw can TIE the perfect pick (1-in-45 per
    # draw), so the toy asserts >=; strict > is the bar on real data only.
    ok(ev["20%"]["harm_captured_share"] >= ev["20%"]["random_matched_max_share"],
       "a perfect score is never beaten by matched random")
    bad = list(reversed(good))
    ev2 = budget_eval(dev, bad, budgets=(0.2,), n_random=50)
    ok(ev2["20%"]["harm_captured_share"] < 0.1,
       "an inverted score captures nothing — the metric is directional")
    # ext_feats sign regression, on synthetic events (no tape dependence):
    # bids pulled hard over the window => ofi < 0 => POSITIVE threat for BUY.
    import harmful_hazard_model as _hm
    _key = ("BTCUSDT", "2099010100", "full")
    _hm._BN_CACHE.clear(); _hm._BN_TCACHE.clear()
    import datetime as _dt
    base = _dt.datetime(2099, 1, 1, 0, 30, 0, tzinfo=_dt.timezone.utc).timestamp()
    ts = [base - 0.9 + 0.1 * i for i in range(9)]
    vals = [(100.0, 50.0 - 5 * i, 100.1, 10.0) for i in range(9)]   # bids pulled
    hh = _dt.datetime.fromtimestamp(base - 0.001, _dt.timezone.utc)
    _hm._BN_CACHE[("BTCUSDT", f"{hh:%Y%m%d_%H}", "full")] = (ts, vals)
    _hm._BN_TCACHE[("BTCUSDT", f"{hh:%Y%m%d_%H}")] = (
        [base - 0.4], [(-1.0, 40.0)])                               # big SELL print
    fB = _hm.ext_feats(base, "BUY_UP", "btc")
    fS = _hm.ext_feats(base, "SELL_UP", "btc")
    iofi = _hm.EXT_NAMES.index("bnf_ofi_1p0")
    ok(fB[iofi] > 0, "bids being PULLED reads as POSITIVE threat to a resting BUY")
    ok(fS[iofi] < 0, "and as negative (protective) for a resting SELL")
    ipr = _hm.EXT_NAMES.index("bnf_bigprint_0p5")
    ok(fB[ipr] > 0 and fS[ipr] == 0.0,
       "a big SELL print threatens the BUY side only")
    _hm._BN_CACHE.clear(); _hm._BN_TCACHE.clear()
    # depth20 parser: positive control + malformed refusal (rule 15)
    good = b"9999999999999999999,1,1,7,100.0@2.0|99.9@1.0|99.8@1.0|99.7@1.0|99.6@1.0|99.5@3.0,100.1@1.0|100.2@1.0|100.3@1.0|100.4@1.0|100.5@1.0|100.6@4.0\n"
    g = _hm._parse_depth_line(good, 0)
    ok(g is not None and g != "preera" and abs(g[2] - 9.0) < 1e-9
       and abs(g[1] - 6.0) < 1e-9 and abs(g[4] - 9.0) < 1e-9,
       "depth parser sums totals and top5 correctly")
    ok(_hm._parse_depth_line(b"1,2,3,4,garbage,alsogarbage\n", 0) is None,
       "depth parser REFUSES a malformed level blob")
    ok(_hm._parse_depth_line(good, 10**19 + 1) == "preera",
       "depth parser excludes pre-era rows")
    # depth_feats sign regression on synthetic snapshots: bids drain 100->60
    # over 2s, asks flat; book ends tilted to asks.
    base2 = _dt.datetime(2099, 1, 1, 0, 30, 0,
                         tzinfo=_dt.timezone.utc).timestamp()
    dts = [base2 - 60.0 + 0.1 * i for i in range(600)]
    dvals = []
    for t in dts:
        frac = min(1.0, max(0.0, (t - (base2 - 2.0)) / 2.0))
        btot = 100.0 - 40.0 * frac
        dvals.append((btot * 0.2, btot, 20.0, 100.0))
    hh2 = _dt.datetime.fromtimestamp(base2 - 0.001, _dt.timezone.utc)
    _hm._BN_DCACHE[("BTCUSDT", f"{hh2:%Y%m%d_%H}")] = (dts, dvals)
    ph = hh2 - _dt.timedelta(hours=1)
    _hm._BN_DCACHE[("BTCUSDT", f"{ph:%Y%m%d_%H}")] = ([], [])
    dB = _hm.depth_feats(base2, "BUY_UP", "btc")
    dS = _hm.depth_feats(base2, "SELL_UP", "btc")
    ip = _hm.DEPTH_NAMES.index("bnd_pull_2p0")
    ok(dB[ip] > 0.2, "bid drain reads as POSITIVE pull threat for a resting BUY")
    ok(abs(dS[ip]) < 0.05, "flat asks read as ~zero pull for a resting SELL")
    ii = _hm.DEPTH_NAMES.index("bnd_imb20_now")
    ok(dB[ii] > 0 and dS[ii] < 0,
       "ask-tilted deep book threatens the BUY side, protects the SELL side")
    _hm._BN_DCACHE.clear()
    # PM thinning family: pulls on OUR side read as threat; consumption is
    # excluded; opposite-side stacking reads on the oppstack feature.
    # NOTE: PM knowledge cutoff is t - LAG_S = t - 0.25, so all events sit
    # before 9.75; an event at 9.8 was (correctly) excluded by the first
    # version of this test.
    stt = {"trades": [], "tt": [],
           "deltas": [(9.4, "BUY", -40.0), (9.6, "BUY", -60.0),
                      (9.7, "SELL", +30.0)],
           "dt": [9.4, 9.6, 9.7], "quotes": [], "qt": []}
    tb = _hm.thin_feats(stt, 10.0, "BUY_UP")
    ts_ = _hm.thin_feats(stt, 10.0, "SELL_UP")
    i1 = _hm.THIN_NAMES.index("pmt_pullshare_1p0")
    io = _hm.THIN_NAMES.index("pmt_oppstack_5p0")
    ok(tb[i1] > 0.99, "BUY-side pulls = pullshare ~1 for a resting BUY")
    ok(ts_[i1] < 0.01, "and ~0 for a resting SELL (its side is quiet)")
    ok(tb[io] > 0.99, "SELL-side adds stack against the resting BUY")
    stt2 = dict(stt, trades=[(9.6, -1, 100.0)], tt=[9.6])
    tb2 = _hm.thin_feats(stt2, 10.0, "BUY_UP")
    ok(tb2[i1] < 0.01,
       "a size drop fully explained by executed volume is NOT a pull")
    # level-delta construction: first sight initializes, later sights delta
    import io as _io
    lv = [(1.0, "BUY", 0.48, 100.0), (2.0, "BUY", 0.48, 60.0),
          (3.0, "BUY", 0.48, 60.0), (4.0, "SELL", 0.52, 10.0)]
    state = {}; ds = []
    for t, sd, px, tot in lv:
        k = (sd, px); prev = state.get(k); state[k] = tot
        if prev is not None and tot != prev:
            ds.append((t, sd, tot - prev))
    ok(ds == [(2.0, "BUY", -40.0)],
       "first sight initializes; unchanged size emits nothing")
    # cross-lead override: coin='eth' with sym='BTCUSDT' must read the BTC
    # book, not the ETH one.
    base3 = _dt.datetime(2099, 1, 1, 0, 30, 0,
                         tzinfo=_dt.timezone.utc).timestamp()
    hh3 = _dt.datetime.fromtimestamp(base3 - 0.001, _dt.timezone.utc)
    _hm._BN_CACHE.clear()
    for symn, imb in (("BTCUSDT", (90.0, 10.0)), ("ETHUSDT", (10.0, 90.0))):
        t3 = [base3 - 1.0 + 0.1 * i for i in range(10)]
        v3 = [(100.0, imb[0], imb[1]) for _ in t3]
        _hm._BN_CACHE[(symn, f"{hh3:%Y%m%d_%H}")] = (t3, v3)
        ph3 = hh3 - _dt.timedelta(hours=1)
        _hm._BN_CACHE.setdefault((symn, f"{ph3:%Y%m%d_%H}"), ([], []))
    fl = _hm.fine_feats(base3, "SELL_UP", "eth", sym="BTCUSDT")
    fo = _hm.fine_feats(base3, "SELL_UP", "eth")
    ii2 = _hm.FINE_NAMES.index("bnf_imb_now")
    ok(fl[ii2] > 0.7 and fo[ii2] < -0.7,
       "sym override reads the BTC book (bid-heavy), not ETH (ask-heavy)")
    _hm._BN_CACHE.clear()
    print(f"harmful_hazard_model selftest: {checks} checks OK")
    return 0


ROWS_ERA = fi.PM / "derived/harmful_exposure_rows_v3_eraB.json"


def run(era: bool = False) -> dict[str, Any]:
    """AUDIT 5 BLOCKER 2: the era rebuild wrote an artifact this runner never
    read, with a split naming days it does not contain. In era mode the split
    is DERIVED FROM THE RECEIPT: last day = development, the rest = training,
    printed so the population is never implicit."""
    import policy_optimizer_queue_realistic as qr
    src = ROWS_ERA if era else ROWS
    data = json.loads(src.read_text())
    # AUDIT 4 BLOCKER 3: no schema guard meant this runner could silently train
    # on an obsolete artifact (the six-window v3.0 smoke was sitting at the v3
    # path). Wrong schema is a REFUSAL, never a warning.
    if data.get("schema") != EXPECTED_SCHEMA:
        raise SystemExit(
            f"REFUSED: artifact schema {data.get('schema')!r} != "
            f"{EXPECTED_SCHEMA!r} — regenerate the dataset; this runner will "
            f"not train on an obsolete artifact")
    rows = [r for r in data["rows"] if r["status"] == "OK"]
    if era:
        days = data["days"]
        train_days, dev_day = tuple(days[:-1]), days[-1]
    else:
        train_days, dev_day = TRAIN_DAYS, DEV_DAY
    print(f"population: {src.name}  train {train_days} -> dev {dev_day}")
    paths = fi._archive_paths(); tokens = fi.token_map()
    streams: dict[str, dict] = {}
    out: dict[str, Any] = {"protocol": "HARMFUL_HAZARD_LINEAR_V1",
                           "feature_names": FEATURE_NAMES,
                           "split": {"train": TRAIN_DAYS, "dev": DEV_DAY},
                           "n_random": N_RANDOM, "coins": {}}
    for coin in COINS:
        crows = [r for r in rows if r["coin"] == coin]
        feats = []; kept = []
        for r in crows:
            slug = r["slug"]
            if slug not in streams:
                up, dn = tokens[slug]
                streams[slug] = window_streams(paths[slug], up, dn)
                # LRU cap: an unbounded cache held every window's quote stream
                # (~150k tuples each) for the whole run and OOM'd at era scale.
                # Rows arrive grouped by window, so 4 is generous.
                if len(streams) > 4:
                    streams.pop(next(iter(streams)))
            f = features(streams[slug], r["t_start"], r["side"], r["level"],
                         r["resting"], r["qahead"])
            if f is None:
                continue
            feats.append(f); kept.append(r)
        tr = [i for i, r in enumerate(kept) if r["day"] in train_days]
        dv = [i for i, r in enumerate(kept) if r["day"] == dev_day]
        if len(tr) < 500 or len(dv) < 200:
            out["coins"][coin] = {"status": "INSUFFICIENT", "n_train": len(tr),
                                  "n_dev": len(dv)}
            continue
        Xall, mu, sd = zscale([feats[i] for i in tr], feats)
        # AUDIT 5 BLOCKER 3: `any_fill_ahead` counts fills BEFORE the assumed
        # cancellation takes effect — unpreventable by definition. The hazard
        # label is latency-specific: preventable shares at TARGET_LATENCY_MS.
        Lh = str(TARGET_LATENCY_MS)
        y_fill = [1 if (kept[i].get("latency") or {}).get(Lh, {}).get(
                      "preventable_shares", 0.0) > 0 else 0
                  for i in range(len(kept))]
        w_haz = fit_logistic([Xall[i] for i in tr], [y_fill[i] for i in tr])
        L = str(TARGET_LATENCY_MS)
        fill_tr = [i for i in tr if y_fill[i] and "latency" in kept[i]]
        tgt = lambda i: kept[i]["latency"][L]["preventable_value_cents"]
        w_sign = fit_logistic([Xall[i] for i in fill_tr],
                              [1 if tgt(i) > 0 else 0
                               for i in fill_tr]) if len(fill_tr) >= 100 else None
        w_mag = fit_ridge([Xall[i] for i in fill_tr],
                          [tgt(i) for i in fill_tr],
                          lam=10.0) if len(fill_tr) >= 100 else None
        dev_rows = [kept[i] for i in dv]
        p_fill = [predict_p(w_haz, Xall[i]) for i in dv]
        tox = ([sum(a * b for a, b in zip(w_mag, Xall[i])) for i in dv]
               if w_mag else [0.0] * len(dv))
        ecv = [p * t for p, t in zip(p_fill, tox)]
        # diagnostics that stay diagnostics (§8: AUC/Brier are not the verdict)
        fills_dev = [i for i in dv if y_fill[i]]
        auc = _auc([p_fill[j] for j in range(len(dv))],
                   [y_fill[dv[j]] for j in range(len(dv))])
        # AUDIT 4 BLOCKER 2: the local row-summing budget_eval read a removed
        # field and silently returned 0. The GENERATION-NATIVE evaluator is the
        # only one allowed to score a policy.
        import harmful_action_eval as ae
        gate = ae.evaluate_policy(dev_rows, ecv, latency_ms=TARGET_LATENCY_MS)
        gate_sign = None
        if w_sign:
            s_sign = [predict_p(w_sign, Xall[i]) * predict_p(w_haz, Xall[i])
                      for i in dv]
            gate_sign = ae.evaluate_policy(dev_rows, s_sign,
                                           latency_ms=TARGET_LATENCY_MS)
        out["coins"][coin] = {
            "n_train": len(tr), "n_dev": len(dv),
            "n_fill_train": len(fill_tr), "n_fill_dev": len(fills_dev),
            "hazard_auc_dev_DIAGNOSTIC": auc,
            "budget_gate_expected_value_score": gate,
            "budget_gate_sign_score": gate_sign,
            "models": {"hazard": w_haz, "tox_sign": w_sign, "tox_mag": w_mag,
                       "mu": mu, "sd": sd},
        }
        print(f"  {coin}: train {len(tr)} dev {len(dv)} "
              f"fills(tr/dev) {len(fill_tr)}/{len(fills_dev)} AUC(diag) {auc:.3f}")
        for b, g in gate["budgets"].items():
            print(f"    ecv @{b}: net {g['net_cents']:+8.0f}c "
                  f"harm {g['harm_avoided_cents']:+8.0f}c "
                  f"sac {g['sacrifice_cents']:8.0f}c "
                  f"rand_max {g['random_net_max']:+8.0f}c "
                  f"beats_max_on_NET={g['beats_random_max_on_NET']}")
    OUT.write_text(json.dumps(out))
    print(f"receipt {OUT}")
    return out


def _auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels); neg = len(labels) - pos
    if not pos or not neg:
        return None
    rank_sum = 0.0
    for i, (_s, l) in enumerate(pairs, 1):
        if l:
            rank_sum += i
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


# ---------------------------------------------------------------- fine arm
# The REDUCED Binance set (audit-validated direction: mechanism-chosen, not
# scoreboard-chosen): current side-signed top-of-book imbalance + side-signed
# mid movement at 10/25/50/100/250 ms. Era-pure per event via the collector
# ledger (CLAUDE.md rule 5). Committed here so the full pipeline is in-repo
# (rule 12) — the scratch-dir version is what voided an earlier freeze.
FINE_SCALES = (0.010, 0.025, 0.050, 0.100, 0.250)
FINE_NAMES = [f"bnf_midbps_{str(w).replace('.', 'p')}" for w in FINE_SCALES]     + ["bnf_imb_now"]
_BN_SYM = {"btc": "BTCUSDT", "eth": "ETHUSDT"}
_BN_CACHE: dict = {}


def _era_boundary_ns() -> int:
    runs = [json.loads(l) for l in
            open('/home/yuqing/ctaNew/data/mm_hf/collector_runs.jsonl')]
    return max(r['started_at_ns'] for r in runs)


def _bn_hour(sym: str, h) -> tuple[list, list]:
    import gzip, glob
    key = (sym, f"{h:%Y%m%d_%H}")
    if key in _BN_CACHE:
        return _BN_CACHE[key]
    lo = _era_boundary_ns()
    ts: list = []; vals: list = []
    for ext in ('.csv.gz', '.csv'):
        fs = glob.glob(f"/home/yuqing/ctaNew/data/mm_hf/raw/bookTicker/"
                       f"{sym}/{h:%Y%m%d_%H}{ext}")
        if not fs:
            continue
        op = gzip.open if fs[0].endswith('.gz') else open
        with op(fs[0], 'rb') as fh:
            for line in fh:
                q = line.split(b',')
                if len(q) < 8:
                    continue
                try:
                    r = int(q[0])
                    if r < lo:
                        continue                    # era purity, per event
                    b = float(q[4]); a = float(q[6])
                    vals.append(((b + a) / 2.0, float(q[5]), float(q[7])))
                    ts.append(r / 1e9)
                except ValueError:
                    continue
        break
    if len(_BN_CACHE) > 3:
        _BN_CACHE.pop(next(iter(_BN_CACHE)))
    _BN_CACHE[key] = (ts, vals)
    return ts, vals


def fine_feats(T: float, side: str, coin: str, sym: str | None = None) -> list | None:
    """Reduced fine set at absolute time T, cutoff T - 1 ms, side-signed."""
    import bisect, datetime as dt
    sym = sym or _BN_SYM.get(coin)
    if sym is None:
        return None
    sgn = -1.0 if side == "BUY_UP" else 1.0
    cut = T - 0.001
    h = dt.datetime.fromtimestamp(cut, dt.timezone.utc)
    ts, vals = _bn_hour(sym, h)
    if not ts or ts[0] > cut:
        prev = h - dt.timedelta(hours=1)
        t2, v2 = _bn_hour(sym, prev)
        ts = t2 + ts; vals = v2 + vals
    i = bisect.bisect_right(ts, cut) - 1
    if i < 0:
        return None
    mid = vals[i][0]; bq, aq = vals[i][1], vals[i][2]
    out = []
    for w in FINE_SCALES:
        j = bisect.bisect_right(ts, cut - w) - 1
        m0 = vals[j][0] if j >= 0 else mid
        out.append(sgn * (mid - m0) / mid * 1e4)
    out.append(sgn * ((bq - aq) / (bq + aq) if bq + aq > 0 else 0.0))
    return out


# ---------------------------------------------------------- extended fine arm
# DECLARED BEFORE ANY NUMBER EXISTS (user request, 2026-08-25): two families
# the reduced arm does not carry, each with a stated mechanism —
#   OFI (order-flow imbalance): bookTicker QUANTITY DELTAS, Cont-style. Pulled
#     bids / stacked asks are informed traders clearing the path BEFORE the
#     level moves — earlier warning than the imbalance level itself.
#   BIG PRINT: largest threat-side trade in the window over the trailing
#     median print size. A SUM of signed flow (tested neutral before) hides
#     exactly this outlier; informed actors reveal themselves in SIZE.
# PM-side book thinning and depth20 remain queued (parse cost); stated, not
# silently skipped. Tested as ONE extended arm vs the reduced arm on identical
# rows — increments only, no per-family scoreboard on consumed data.
OFI_SCALES = (0.1, 0.5, 1.0)
PRINT_SCALES = (0.5, 2.0)
EXT_NAMES = [f"bnf_ofi_{str(w).replace('.', 'p')}" for w in OFI_SCALES] +             [f"bnf_bigprint_{str(w).replace('.', 'p')}" for w in PRINT_SCALES]
_BN_TCACHE: dict = {}


def _bn_hour_full(sym: str, h):
    """bookTicker with prices AND quantities (for OFI deltas)."""
    import gzip, glob
    key = (sym, f"{h:%Y%m%d_%H}", "full")
    if key in _BN_CACHE:
        return _BN_CACHE[key]
    lo = _era_boundary_ns()
    ts = []; vals = []
    for ext in ('.csv.gz', '.csv'):
        fs = glob.glob(f"/home/yuqing/ctaNew/data/mm_hf/raw/bookTicker/"
                       f"{sym}/{h:%Y%m%d_%H}{ext}")
        if not fs:
            continue
        op = gzip.open if fs[0].endswith('.gz') else open
        with op(fs[0], 'rb') as fh:
            for line in fh:
                q = line.split(b',')
                if len(q) < 8:
                    continue
                try:
                    r = int(q[0])
                    if r < lo:
                        continue
                    vals.append((float(q[4]), float(q[5]),
                                 float(q[6]), float(q[7])))
                    ts.append(r / 1e9)
                except ValueError:
                    continue
        break
    if len(_BN_CACHE) > 3:
        _BN_CACHE.pop(next(iter(_BN_CACHE)))
    _BN_CACHE[key] = (ts, vals)
    return ts, vals


def _bn_trades(sym: str, h):
    import gzip, glob
    key = (sym, f"{h:%Y%m%d_%H}")
    if key in _BN_TCACHE:
        return _BN_TCACHE[key]
    lo = _era_boundary_ns()
    ts = []; vals = []
    for ext in ('.csv.gz', '.csv'):
        fs = glob.glob(f"/home/yuqing/ctaNew/data/mm_hf/raw/trade/"
                       f"{sym}/{h:%Y%m%d_%H}{ext}")
        if not fs:
            continue
        op = gzip.open if fs[0].endswith('.gz') else open
        with op(fs[0], 'rb') as fh:
            for line in fh:
                q = line.split(b',')
                if len(q) < 7:
                    continue
                try:
                    r = int(q[0])
                    if r < lo:
                        continue
                    vals.append((1.0 if q[6].strip() != b'1' else -1.0,
                                 float(q[5])))
                    ts.append(r / 1e9)
                except ValueError:
                    continue
        break
    if len(_BN_TCACHE) > 3:
        _BN_TCACHE.pop(next(iter(_BN_TCACHE)))
    _BN_TCACHE[key] = (ts, vals)
    return ts, vals


def ext_feats(T: float, side: str, coin: str) -> list | None:
    """OFI + big-print at cutoff T-1ms, side-signed toward THREAT."""
    import bisect, datetime as dt
    sym = _BN_SYM.get(coin)
    if sym is None:
        return None
    sgn = -1.0 if side == "BUY_UP" else 1.0
    cut = T - 0.001
    h = dt.datetime.fromtimestamp(cut, dt.timezone.utc)
    ts, vals = _bn_hour_full(sym, h)
    if not ts or ts[0] > cut - max(OFI_SCALES):
        prev = h - dt.timedelta(hours=1)
        t2, v2 = _bn_hour_full(sym, prev)
        ts = t2 + ts; vals = v2 + vals
    hi = bisect.bisect_right(ts, cut)
    if hi == 0:
        return None
    out = []
    for w in OFI_SCALES:
        lo_i = bisect.bisect_left(ts, cut - w)
        ofi = 0.0
        for i in range(max(lo_i, 1), hi):
            bp0, bq0, ap0, aq0 = vals[i - 1]
            bp1, bq1, ap1, aq1 = vals[i]
            e = ((bq1 if bp1 >= bp0 else 0.0) - (bq0 if bp1 <= bp0 else 0.0)
                 - ((aq1 if ap1 <= ap0 else 0.0) - (aq0 if ap1 >= ap0 else 0.0)))
            ofi += e
        # SIGN, reviewed: threat to a resting BUY is FALLING support (ofi<0),
        # so threat = -ofi for BUY, +ofi for SELL == sgn * ofi with sgn(BUY)=-1.
        # The first version had sgn * -ofi — inverted. Linear-invariant for the
        # fit (the coefficient absorbs it) but semantically wrong for any
        # reader of the coefficient or any nonlinear successor.
        out.append(sgn * ofi)
    tts, tvals = _bn_trades(sym, h)
    if tts and tts[0] > cut - 60.0:
        prev = h - dt.timedelta(hours=1)
        t2, v2 = _bn_trades(sym, prev)
        tts = t2 + tts; tvals = v2 + tvals
    thi = bisect.bisect_right(tts, cut)
    base_lo = bisect.bisect_left(tts, cut - 60.0)
    sizes = sorted(q for _d, q in tvals[base_lo:thi]) or [1.0]
    med = sizes[len(sizes) // 2] or 1.0
    for w in PRINT_SCALES:
        tlo = bisect.bisect_left(tts, cut - w)
        threat = [q for d, q in tvals[tlo:thi] if (d * sgn) > 0]
        out.append((max(threat) / med) if threat else 0.0)
    return out


# ---------------------------------------------------------- PM thinning family
# I3, DECLARED BEFORE SCORING. Mechanism: informed PM participants and
# competing MMs PULL resting orders just before adverse moves — the PM book
# thins on the side we rest on (and stacks opposite) ahead of a toxic fill.
# PM-native, complementary to the Binance families. Scale-free shares (no
# fitted normalizer); pulls are trade-corrected (a size drop explained by
# executed volume is consumption, not a pull) — level-blind netting, declared
# approximate. First sight of a price level initializes state; the 60s
# pre-window stream absorbs that.
THIN_SCALES = (1.0, 5.0)
THIN_NAMES = [f"pmt_pullshare_{str(w).replace('.', 'p')}" for w in THIN_SCALES] \
             + ["pmt_oppstack_5p0"]


def thin_feats(st: dict[str, list], t: float, side: str) -> list:
    cut = t - LAG_S
    ours = "BUY" if side == "BUY_UP" else "SELL"
    out = []
    for w, want_opp in ((THIN_SCALES[0], False), (THIN_SCALES[1], False),
                        (THIN_SCALES[1], True)):
        lo = bisect.bisect_left(st["dt"], cut - w)
        hi = bisect.bisect_right(st["dt"], cut)
        pulled = added = 0.0
        which = ("SELL" if ours == "BUY" else "BUY") if want_opp else ours
        for i in range(lo, hi):
            _, sd, d = st["deltas"][i]
            if sd != which:
                continue
            if d < 0:
                pulled += -d
            else:
                added += d
        tlo = bisect.bisect_left(st["tt"], cut - w)
        thi = bisect.bisect_right(st["tt"], cut)
        eat_dir = -1 if which == "BUY" else 1     # taker dir consuming `which`
        consumed = sum(tr[2] for tr in st["trades"][tlo:thi]
                       if tr[1] == eat_dir)
        pull_net = max(0.0, pulled - consumed)
        if want_opp:
            out.append(added / (added + pull_net + 1e-9))
        else:
            out.append(pull_net / (pull_net + added + 1e-9))
    return out


# ------------------------------------------------------------- depth20 family
# DECLARED BEFORE SCORING; geometry-amended after data verification (the
# 20-level book spans only ~0.4-0.8 bps, so "depth within X bps" collapses
# to the whole visible book — features use level structure instead):
#   bnd_pull_{w}:  drain of TOTAL visible threat-side depth over w, vs its
#                  trailing 60s mean. Pulled support ahead of a move.
#   bnd_near_now:  1 - top5/total on the threat side — thin near the touch.
#   bnd_imb20_now: (B20-A20)/(B20+A20), threat-signed — deep imbalance beyond
#                  the L1 the reduced arm already sees.
# Snapshots are ~100ms cadence (max gap seen 414ms) so values can be up to
# ~0.1s stale at the cutoff; inherent to the feed, disclosed.
DEPTH_SCALES = (0.5, 2.0)
DEPTH_NAMES = [f"bnd_pull_{str(w).replace('.', 'p')}" for w in DEPTH_SCALES] + \
              ["bnd_near_now", "bnd_imb20_now"]
_BN_DCACHE: dict = {}


def _parse_depth_line(line: bytes, lo: int):
    """One depth20 row -> (recv_s, b_top5, b_tot, a_top5, a_tot) | None.
    None for pre-era rows AND malformed rows; the loader counts the latter."""
    q = line.rstrip().split(b',')
    if len(q) != 6:
        return None
    try:
        r = int(q[0])
        if r < lo:
            return "preera"
        bt5 = bt = at5 = at = 0.0
        for i, lv in enumerate(q[4].split(b'|')):
            px, qty = lv.split(b'@')
            v = float(qty)
            bt += v
            if i < 5:
                bt5 += v
        for i, lv in enumerate(q[5].split(b'|')):
            px, qty = lv.split(b'@')
            v = float(qty)
            at += v
            if i < 5:
                at5 += v
        return (r / 1e9, bt5, bt, at5, at)
    except (ValueError, IndexError):
        return None


def _bn_depth(sym: str, h):
    import gzip, glob
    key = (sym, f"{h:%Y%m%d_%H}")
    if key in _BN_DCACHE:
        return _BN_DCACHE[key]
    lo = _era_boundary_ns()
    ts = []; vals = []; bad = 0; total = 0
    for ext in ('.csv.gz', '.csv'):
        fs = glob.glob(f"/home/yuqing/ctaNew/data/mm_hf/raw/depth20/"
                       f"{sym}/{h:%Y%m%d_%H}{ext}")
        if not fs:
            continue
        op = gzip.open if fs[0].endswith('.gz') else open
        with op(fs[0], 'rb') as fh:
            for line in fh:
                total += 1
                got = _parse_depth_line(line, lo)
                if got == "preera":
                    continue
                if got is None:
                    bad += 1
                    continue
                ts.append(got[0]); vals.append(got[1:])
        break
    if total and bad > 0.01 * total:
        raise SystemExit(
            f"REFUSED: depth20 {sym} {h:%Y%m%d_%H}: {bad}/{total} malformed")
    if len(_BN_DCACHE) > 2:
        _BN_DCACHE.pop(next(iter(_BN_DCACHE)))
    _BN_DCACHE[key] = (ts, vals)
    return ts, vals


def depth_feats(T: float, side: str, coin: str) -> list | None:
    """Depth20 family at cutoff T-1ms, threat-signed (higher = more danger)."""
    import bisect, datetime as dt
    sym = _BN_SYM.get(coin)
    if sym is None:
        return None
    cut = T - 0.001
    h = dt.datetime.fromtimestamp(cut, dt.timezone.utc)
    ts, vals = _bn_depth(sym, h)
    if not ts or ts[0] > cut - 60.0:
        prev = h - dt.timedelta(hours=1)
        t2, v2 = _bn_depth(sym, prev)
        ts = t2 + ts; vals = v2 + vals
    hi = bisect.bisect_right(ts, cut)
    if hi == 0:
        return None
    bt5, bt, at5, at = vals[hi - 1]
    # threat side: a resting BUY dies when BID support drains; SELL when asks
    thr_now = (bt5, bt) if side == "BUY_UP" else (at5, at)
    m_lo = bisect.bisect_left(ts, cut - 60.0)
    if m_lo >= hi:          # feed gap >60s: no trailing base — drop the row
        return None
    idx = 1 if side == "BUY_UP" else 3
    base = sum(v[idx] for v in vals[m_lo:hi]) / (hi - m_lo)
    if base <= 0:
        return None
    out = []
    for w in DEPTH_SCALES:
        j = bisect.bisect_right(ts, cut - w)
        if j == 0:
            return None
        then = vals[j - 1][idx]
        out.append((then - thr_now[1]) / base)      # positive = drained
    out.append(1.0 - (thr_now[0] / thr_now[1] if thr_now[1] > 0 else 1.0))
    imb = (bt - at) / (bt + at) if (bt + at) > 0 else 0.0
    out.append((-1.0 if side == "BUY_UP" else 1.0) * imb)
    return out


def run_fine(era: bool = True, lead: bool = False) -> dict[str, Any]:
    """PAIRED comparison on IDENTICAL rows. Arms:
      PM_ONLY          anchor
      PM_PLUS_FINE     current best (reduced fine spec)
      PM_FINE_SHIFTED  CONTROL: fine features at T-5s (causal, misaligned).
                       Declared expectation: the fine gain collapses.
      PM_FINE_EXTENDED reduced + OFI + big-print   (candidate 2)
      PM_FINE_PLUS_DEPTH reduced + depth20 family  (candidate 3)
    Multiplicity: 3 candidate specs in the development race. Increments are
    read WITHIN this run only (populations differ across runs when a family
    drops rows). Development evidence — consumed era tape."""
    import gzip
    import policy_optimizer_queue_realistic as qr
    import harmful_action_eval as ae
    src = ROWS_ERA if era else ROWS
    data = json.loads(src.read_text())
    if data.get("schema") != EXPECTED_SCHEMA:
        raise SystemExit(f"REFUSED: schema {data.get('schema')!r}")
    rows = [r for r in data["rows"] if r["status"] == "OK"]
    days = data["days"]; train_days, dev_day = tuple(days[:-1]), days[-1]
    print(f"population: {src.name}  train {train_days} -> dev {dev_day}")
    paths = fi._archive_paths(); tokens = fi.token_map()
    # I5 lead mode: eth only, btc-book features appended to the reduced spec.
    ARMS = (("PM_ONLY", "PM_PLUS_FINE", "PM_FINE_PLUS_BTCLEAD",
             "PM_FINE_LEADSHIFT") if lead
            else ("PM_ONLY", "PM_PLUS_FINE", "PM_FINE_PLUS_THIN"))
    COINS = ("eth",) if lead else ("btc", "eth")
    out: dict[str, Any] = {
        "paired_arms": {}, "schema": data["schema"],
        "as_of": data.get("as_of"),
        "multiplicity_candidate_specs": 5 if lead else 4,
        "control_arms": ["PM_FINE_LEADSHIFT"] if lead else [],
        "knowledge_cutoffs": {"pm_features_s": 0.250, "pm_thin_s": 0.250,
                              "binance_fine_s": 0.001},
        "declared": "families+mechanisms declared pre-score in code comments; "
                    "depth geometry amended after data verification, before "
                    "scoring (20 levels span <1bp)"}
    for coin in COINS:
        crows = [r for r in rows if r["coin"] == coin]
        streams: dict = {}
        FAM: dict = {"pm": [], "fn": [], "th": [], "bl": [], "ls": []}
        kept: list = []
        for r in crows:
            slug = r["slug"]
            if slug not in streams:
                up, dn = tokens[slug]
                streams[slug] = window_streams(paths[slug], up, dn)
                if len(streams) > 4:
                    streams.pop(next(iter(streams)))
            fp = features(streams[slug], r["t_start"], r["side"], r["level"],
                          r["resting"], r["qahead"])
            if fp is None:
                continue
            T = r["t0"] + r["t_start"]
            ff = fine_feats(T, r["side"], coin)
            if ff is None:
                continue
            th = thin_feats(streams[slug], r["t_start"], r["side"])
            if lead:
                bl = fine_feats(T, r["side"], coin, sym="BTCUSDT")
                lsh = fine_feats(T - 5.0, r["side"], coin, sym="BTCUSDT")
                if bl is None or lsh is None:
                    continue
                FAM["bl"].append(bl); FAM["ls"].append(lsh)
            FAM["pm"].append(fp); FAM["fn"].append(ff); FAM["th"].append(th)
            kept.append({k: r.get(k) for k in
                         ("slug", "day", "t0", "t_start", "side", "gen",
                          "latency")})
        tr = [i for i, r in enumerate(kept) if r["day"] in train_days]
        dv = [i for i, r in enumerate(kept) if r["day"] == dev_day]
        Lh = str(TARGET_LATENCY_MS)
        y = [1 if (kept[i].get("latency") or {}).get(Lh, {}).get(
                 "preventable_shares", 0.0) > 0 else 0
             for i in range(len(kept))]
        tgt = lambda i: kept[i]["latency"][Lh]["preventable_value_cents"]
        print(f"  {coin}: rows {len(kept)} train {len(tr)} dev {len(dv)}")
        dev_scores: dict = {}
        for arm in ARMS:
            add = {"PM_ONLY": (), "PM_PLUS_FINE": ("fn",),
                   "PM_FINE_PLUS_THIN": ("fn", "th"),
                   "PM_FINE_PLUS_BTCLEAD": ("fn", "bl"),
                   "PM_FINE_LEADSHIFT": ("fn", "ls")}[arm]
            XA = [FAM["pm"][i] + [v for f in add for v in FAM[f][i]]
                  for i in range(len(kept))]
            Xs, mu, sd = zscale([XA[i] for i in tr], XA)
            w = fit_logistic([Xs[i] for i in tr], [y[i] for i in tr])
            ft = [i for i in tr if y[i]]
            wm = (fit_ridge([Xs[i] for i in ft], [tgt(i) for i in ft],
                            lam=10.0) if len(ft) >= 100 else None)
            auc = _auc([predict_p(w, Xs[i]) for i in dv], [y[i] for i in dv])
            ecv = [predict_p(w, Xs[i]) *
                   (sum(a * b for a, b in zip(wm, Xs[i])) if wm else 0.0)
                   for i in dv]
            gate = ae.evaluate_policy([keptrow(kept[i]) for i in dv], ecv,
                                      latency_ms=TARGET_LATENCY_MS)
            dev_scores[arm] = ecv
            out["paired_arms"].setdefault(coin, {})[arm] = {
                "auc": auc, "gate": gate}
            print(f"    {arm:<18} AUC {auc:.3f}")
            for b, g in gate["budgets"].items():
                print(f"      @{b}: net {g['net_cents']:+8.0f}c "
                      f"harm {g['harm_avoided_cents']:+8.0f} "
                      f"sac {g['sacrifice_cents']:8.0f} "
                      f"rand_max {g['random_net_max']:+8.0f} "
                      f"beats_NET={g['beats_random_max_on_NET']}")
            del XA, Xs
        # score dump: everything the offline confirmations need, dev rows only
        DUMP = fi.PM / ("derived/harmful_scores_"
                        f"{coin}_{'v5leadctl' if lead else 'v3'}.jsonl.gz")
        with gzip.open(DUMP, "wt") as fh:
            for pos, i in enumerate(dv):
                k = kept[i]; L = k["latency"][Lh]
                fh.write(json.dumps({
                    "slug": k["slug"], "side": k["side"], "gen": k["gen"],
                    "t_abs": k["t0"] + k["t_start"],
                    "hour": int(((k["t0"] + k["t_start"]) // 3600) % 24),
                    "pv50": L["preventable_value_cents"],
                    "ps50": L["preventable_shares"],
                    "af": keptrow(k)["any_fill_ahead"],
                    "scores": {a: dev_scores[a][pos] for a in ARMS}}) + "\n")
        print(f"  score dump {DUMP}")
        del dev_scores, FAM
    OUTF = fi.PM / ("derived/harmful_fine_comparison_"
                    f"{'v5leadctl' if lead else 'v3'}.json")
    OUTF.write_text(json.dumps(out))
    print(f"receipt {OUTF}")
    return out


def keptrow(r: dict) -> dict:
    r2 = dict(r)
    lat = r2.get("latency") or {}
    L = str(TARGET_LATENCY_MS)
    r2["any_fill_ahead"] = lat.get(L, {}).get("preventable_shares", 0.0) > 0         or any(v.get("preventable_shares", 0.0) > 0 or v.get("stale_shares", 0.0) > 0
               for v in lat.values())
    return r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--era", action="store_true",
                    help="run on the era-B artifact with a receipt-derived split")
    ap.add_argument("--lead", action="store_true",
                    help="I5: eth-only, btc-book lead features appended")
    ap.add_argument("--fine", action="store_true",
                    help="paired PM_ONLY vs PM+reduced-fine comparison")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd != "run":
        ap.print_help(); return 2
    if a.fine or a.lead:
        run_fine(era=True, lead=a.lead)
    else:
        run(era=a.era)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
