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
    trades.sort(); quotes.sort()
    return {"trades": trades, "quotes": quotes,
            "tt": [x[0] for x in trades], "qt": [x[0] for x in quotes]}


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
    k = len(train_X[0])
    mu = [sum(x[i] for x in train_X) / len(train_X) for i in range(k)]
    sd = [math.sqrt(sum((x[i] - mu[i]) ** 2 for x in train_X) / len(train_X)) or 1.0
          for i in range(k)]
    return [[1.0] + [(x[i] - mu[i]) / sd[i] for i in range(k)] for x in all_X], mu, sd


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
    Xs, _, _ = zscale(X, X)
    w = fit_logistic(Xs, y)
    ok(predict_p(w, Xs[3]) > 0.9 and predict_p(w, Xs[0]) < 0.1,
       "logistic separates a separable toy")
    wr = fit_ridge([[1.0, x[0]] for x in X], [2 * x[0] + 1 for x in X])
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
    for coin in ("btc", "eth"):
        crows = [r for r in rows if r["coin"] == coin]
        feats = []; kept = []
        for r in crows:
            slug = r["slug"]
            if slug not in streams:
                up, dn = tokens[slug]
                streams[slug] = window_streams(paths[slug], up, dn)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--era", action="store_true",
                    help="run on the era-B artifact with a receipt-derived split")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd != "run":
        ap.print_help(); return 2
    run(era=a.era)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
