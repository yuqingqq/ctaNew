"""BE-Belief: the venue book, recalibrated -- productionised.

Implements `plans/BE_BELIEF_PLAN.md` section 12 steps 1-3, which the plan ran in
a session scratchpad and never committed:

  1. rebuild top-of-book from `price_change.best_bid/best_ask` at knowledge time
  2. re-express the calibration table at the EXECUTABLE prices, not the mid
  3. fit models 0/1/2/3/5 walk-forward, day held out

Step 5 -- the day-clustered CI on delta log-loss that decides whether this module
ships at all or ships as `Identity` -- is the same code path; it needs 7 days and
refuses until it has more than one day block.

WHAT THIS MODULE IS. Under plan option (c) the belief is an ALGEBRAIC function of
an observed price: `logit p_hat = b * logit m`, deployed with the drift intercept
PINNED TO ZERO and estimated with it free. Those are different operations and
conflating them corrupts the slope (plan section 4.4). It is a correctness
module, not a P&L module: the measured effect is about 2 probability points at
its best moneyness, roughly one ATM half-spread.

WHAT IT MUST NOT DO. It produces the UNCONDITIONAL belief `E[Y | book state]`,
never the fill-conditional `E[Y | book state, FILLED]`. The fill conditioning is
BE-FlowAndFills' adverse-selection term; baking it in here double-counts the
haircut and silently under-quotes. Ownership ruling, plan section 1.2.

Guards, each one paid for elsewhere in this corpus:

  * BOOK      -- top of book comes from `price_change.best_bid/best_ask`, NEVER
                 `book` snapshots (p90 6.2 s stale; a 6 s-stale mid is wrong by
                 5.6 c mean against an effect of 3-7 c, and it inflated b_hat by
                 26% and the spread by 1.0-1.5 c ATM).
  * KNOWLEDGE -- a quote received at t is usable only at t + 250 ms, and a
                 collector gap KILLS the state rather than shrinking it. The
                 pair builder here mirrors `flow_intensity.state_segments_from_
                 points` and a selftest asserts the two agree on the midpoint.
  * BOUNDARY  -- quotes are kept on `0 <= bid < ask <= 1`. The strict form
                 dropped 5.2% of quotes, all from the tails, in two independent
                 codebases.
  * CLUSTER   -- the 5 decision times in a window share ONE Bernoulli outcome.
                 Every n is inflated 5x and every naive t is ~45% too large.
                 Intervals cluster on window, then on DAY.
  * DAY       -- a day-block bootstrap needs more than one day block. At one
                 sampled day this returns DAY_BLOCK_UNAVAILABLE instead of an
                 interval. See FLOW_MODEL_STATE.md section 1f: every headline
                 result in this corpus at per_coin <= 60 is ONE UTC day, because
                 the shared sampler is EARLIEST-first.
  * SAMPLE    -- `provenance(sampled=...)` is stamped with the slugs actually
                 read, never the days merely globbed.

    python3 live/pm_research/be_belief.py --selftest
    python3 live/pm_research/be_belief.py run --per-coin 40
    python3 live/pm_research/be_belief.py run              # full population
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import flow_intensity as fi

PM = fi.PM
OUT_JSON = PM / "derived/be_belief_v1.json"
OUT_MD = Path(__file__).with_name("BE_BELIEF_RESULTS.md")


def out_paths(coins) -> tuple[Path, Path]:
    """Receipt paths NAMED BY THE POPULATION they describe.

    A fixed output path cannot hold two populations: a btc/eth-only run would
    silently overwrite the all-coin receipt that this programme's belief findings
    rest on, and the two files would be indistinguishable afterwards. That is the
    same defect as a check whose scope is invisible from its output -- BE has now
    shipped four instruments with that shape, so the population goes IN THE NAME.

    `FLOW_MODEL_PROTOCOL_V5.yaml:333` freezes verdict_coins [btc, eth]; a run
    restricted to them is a DIFFERENT ESTIMAND from the pooled one, not a subset
    view of it, and must not share a filename with it.
    """
    if not coins:
        return OUT_JSON, OUT_MD
    tag = "-".join(sorted(c.lower() for c in coins))
    return (OUT_JSON.with_name(f"be_belief_v1__{tag}.json"),
            OUT_MD.with_name(f"BE_BELIEF_RESULTS__{tag}.md"))

# Elapsed seconds into the 300 s window at which the belief is evaluated.
# r = 270, 240, 180, 120, 60 remaining. Frozen by BE_BELIEF_PLAN section A.
DECISION_ELAPSED_S: tuple[float, ...] = (30.0, 60.0, 120.0, 180.0, 240.0)

# Core domain |logit m| <= 3  <=>  m in [0.0474, 0.9526]. Plan section 5.1: a
# hygiene choice, not a rescue -- it moves b_hat by 0.02. The extreme domain is
# fitted separately under its own intercept, labelled `tick_floor` not `drift`.
CORE_MAX_ABS_LOGIT = 3.0

BUCKET_EDGES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                   0.6, 0.7, 0.8, 0.9, 1.0)

_EPS = 1e-12


# --------------------------------------------------------------- outcomes

def is_final(m: dict) -> bool:
    """Resolved iff closed, or outcomePrices degenerate to {0,1}.

    MIRRORS `collect_pm.is_final`, the predicate the writer of the file uses.
    Gamma publishes live prices into `outcomePrices` for OPEN markets, so their
    mere presence is not resolution -- treating it as such recorded garbage
    (rows with closed=false and prices 0.165/0.835). A selftest asserts this
    copy agrees with the canonical one; if the collector's rule changes, that
    check fails loudly instead of this module drifting.
    """
    if m.get("closed") is True:
        return True
    op = m.get("outcomePrices")
    if op:
        try:
            vals = json.loads(op) if isinstance(op, str) else op
            return set(float(x) for x in vals) <= {0.0, 1.0}
        except Exception:
            return False
    return False


def final_outcomes() -> tuple[dict[str, int], dict[str, int]]:
    """slug -> 1 if Up won. Also returns named counts of what was refused.

    THE NAME IS NOT THE DEFINITION, instance N+1. `outcomePrices` is the field a
    reader reaches for and it is the WRONG one: it is written by the Gamma
    polling path onto NON-final rows (live prices such as 0.165/0.835), while a
    final row is written by the CLOB resolver as
    `{closed: true, winners: {"Up": bool, "Down": bool}, source: "clob"}` and
    carries no `outcomePrices` at all -- verified against `collect_pm._resolver`,
    the code that writes the file. All 7,031 final rows on disk take that shape.
    Reading `outcomePrices` yields ZERO outcomes, not wrong ones, which is the
    only reason this was caught immediately.

    A row whose `winners` is not exactly one True across {Up, Down} has no
    usable outcome and is counted, never guessed.
    """
    out: dict[str, int] = {}
    tally: collections.Counter[str] = collections.Counter()
    for line in (PM / "resolutions.jsonl").open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            tally["unparseable"] += 1
            continue
        slug = row.get("slug")
        if not slug:
            tally["no_slug"] += 1
            continue
        if not is_final(row):
            tally["not_final"] += 1
            continue
        y = outcome_of(row)
        if y is None:
            tally["final_but_no_usable_outcome"] += 1
            continue
        if slug in out and out[slug] != y:
            tally["conflicting_outcome"] += 1     # never silently overwritten
            continue
        out[slug] = y
        tally["final"] += 1
    return out, dict(tally)


def outcome_of(row: dict) -> int | None:
    """1 if Up won, 0 if Down won, None if the row does not determine it."""
    winners = row.get("winners")
    if isinstance(winners, dict):
        up, down = winners.get("Up"), winners.get("Down")
        if up is True and down is False:
            return 1
        if up is False and down is True:
            return 0
        return None
    op = row.get("outcomePrices")     # legacy/degenerate form is_final admits
    if op:
        try:
            prices = [float(x) for x in
                      (json.loads(op) if isinstance(op, str) else op)]
        except Exception:
            return None
        if len(prices) == 2 and set(prices) == {0.0, 1.0}:
            return 1 if prices[0] == 1.0 else 0
    return None


def slug_day(slug: str) -> str:
    return fi.slug_day(slug)


# ------------------------------------------------------------- book state

def state_segments_from_pairs(
    points: Sequence[tuple[float, float, float]],
    gaps: Sequence[tuple[float, float]],
    lag_s: float = fi.QUOTE_STATE_LAG_S,
) -> list[tuple[float, float, float, float]]:
    """Half-open knowledge-admissible `(start, end, bid, ask)` segments.

    Structurally identical to `flow_intensity.state_segments_from_points`; it
    carries the PAIR because BE-Belief must validate at the executable prices
    and a mid cannot be un-averaged. Selftest `pairs match canonical mid`
    asserts the equivalence rather than trusting this comment.
    """
    if lag_s < 0:
        raise ValueError("state lag must be non-negative")

    effective: list[tuple[float, float, float]] = []
    ordered = sorted(((float(t) + lag_s, float(b), float(a))
                      for t, b, a in points), key=lambda q: q[0])
    for t, b, a in ordered:
        if t > fi.WINDOW_S:
            continue
        if not (0.0 <= b < a <= 1.0):
            continue
        if effective and abs(effective[-1][0] - t) < 1e-12:
            effective[-1] = (t, b, a)      # last quote at one instant wins
        else:
            effective.append((t, b, a))

    clipped = sorted((max(0.0, float(g0)), min(fi.WINDOW_S, float(g1)))
                     for g0, g1 in gaps
                     if g1 > g0 and g1 > 0.0 and g0 < fi.WINDOW_S)

    out: list[tuple[float, float, float, float]] = []
    for i, (raw_start, bid, ask) in enumerate(effective):
        start = max(0.0, raw_start)
        end = effective[i + 1][0] if i + 1 < len(effective) else fi.WINDOW_S
        end = min(fi.WINDOW_S, end)
        if end <= start:
            continue
        received = raw_start - lag_s
        if any(g0 < start and g1 > received for g0, g1 in clipped):
            continue                       # disconnect between receipt and maturity
        touch = next(((g0, g1) for g0, g1 in clipped if g1 > start and g0 < end),
                     None)
        if touch is not None:
            g0, _ = touch
            if g0 <= start:
                continue                   # state dies inside the gap, never revives
            end = min(end, g0)
        if end > start:
            out.append((start, end, bid, ask))
    return out


def book_at(segments: Sequence[tuple[float, float, float, float]],
            t: float) -> tuple[float, float, float] | None:
    """`(bid, ask, age_s)` in force at `t`, or None when no state is admitted."""
    for start, end, bid, ask in segments:
        if start <= t < end:
            return bid, ask, t - start
    return None


def window_quotes(path: Path, up_id: str,
                  gaps: Sequence[tuple[float, float]],
                  stop_after_s: float | None = None
                  ) -> list[tuple[float, float, float, float]]:
    """Knowledge-admissible top-of-book segments for one window's Up token."""
    try:
        ws = int(path.name.split(".jsonl")[0].rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return []
    limit = None if stop_after_s is None else stop_after_s + fi.QUOTE_STATE_LAG_S
    pts: list[tuple[float, float, float]] = []
    for line in fi._gz_lines(path):
        if fi.QUOTE_MARK not in line:
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            recv_ns = int(parts[0])
            payload = json.loads(parts[1])
        except (ValueError, json.JSONDecodeError):
            continue
        el = recv_ns / 1e9 - ws
        if el < 0.0:
            continue
        if el > fi.WINDOW_S:
            break                          # recv_ns is monotone in file order
        if limit is not None and el > limit:
            break
        for msg in payload if isinstance(payload, list) else [payload]:
            if not isinstance(msg, dict) or msg.get("event_type") != "price_change":
                continue
            for pc in msg.get("price_changes", []):
                if str(pc.get("asset_id")) != up_id:
                    continue
                try:
                    bid, ask = float(pc["best_bid"]), float(pc["best_ask"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (0.0 <= bid < ask <= 1.0):
                    continue
                pts.append((el, bid, ask))
    return state_segments_from_pairs(pts, gaps)


# ------------------------------------------------------------------ rows

def build_rows(per_coin: int | None = None,
               coins: Sequence[str] | None = None,
               era: str = fi.ERA,
               stratify_by_day: bool = True,
               progress: bool = False) -> dict[str, Any]:
    """One row per (window, decision time): the book, and the window's outcome.

    `stratify_by_day` exists because the shared `select()` idiom is EARLIEST
    first, which pins any truncated sample to the opening hours of the era --
    the defect recorded in FLOW_MODEL_STATE.md section 1f. When a cap is applied
    here it is spread ACROSS days by round-robin, so a day-block bootstrap has
    blocks to work with. With no cap the whole covered population is used and
    the question does not arise.
    """
    outcomes, outcome_tally = final_outcomes()
    paths = fi._archive_paths()
    tokens = fi.token_map()
    gaps = fi.gaps_by_slug(era)

    avail: dict[str, list[str]] = collections.defaultdict(list)
    for slug in sorted(fi.covered_slugs(era)):
        coin = slug.split("-")[0]
        if coins and coin not in coins:
            continue
        if slug in paths and slug in tokens and slug in outcomes:
            avail[coin].append(slug)

    picked: list[str] = []
    for coin, slugs in sorted(avail.items()):
        if per_coin is None or per_coin >= len(slugs):
            picked.extend(slugs)
            continue
        if not stratify_by_day:
            picked.extend(slugs[:per_coin])
            continue
        by_day: dict[str, list[str]] = collections.defaultdict(list)
        for s in slugs:
            by_day[slug_day(s)].append(s)
        order = sorted(by_day)
        take: list[str] = []
        idx = 0
        while len(take) < per_coin:
            day = order[idx % len(order)]
            if by_day[day]:
                take.append(by_day[day].pop(0))
            elif all(not v for v in by_day.values()):
                break
            idx += 1
        picked.extend(sorted(take))

    rows: list[dict[str, Any]] = []
    refused: collections.Counter[str] = collections.Counter()
    n = len(picked)
    for i, slug in enumerate(sorted(picked), 1):
        if progress and (i == 1 or i % 25 == 0):
            print(f"[be_belief] {i}/{n} {slug}", flush=True)
        up_id, _ = tokens[slug]
        segs = window_quotes(paths[slug], up_id, gaps.get(slug, []),
                             stop_after_s=max(DECISION_ELAPSED_S))
        if not segs:
            refused["no_admitted_state_in_window"] += 1
            continue
        y = outcomes[slug]
        coin = slug.split("-")[0]
        day = slug_day(slug)
        for el in DECISION_ELAPSED_S:
            state = book_at(segs, el)
            if state is None:
                refused["no_state_at_decision_time"] += 1
                continue
            bid, ask, age = state
            mid = (bid + ask) / 2.0
            if not (0.0 < mid < 1.0):
                refused["degenerate_mid"] += 1
                continue
            rows.append({"slug": slug, "coin": coin, "day": day,
                         "elapsed": el, "r": fi.WINDOW_S - el,
                         "bid": bid, "ask": ask, "mid": mid,
                         "spread": ask - bid, "age_s": age, "y": y})
    return {"rows": rows, "slugs_sampled": sorted(picked),
            "refused": dict(refused), "outcome_tally": outcome_tally,
            "n_available_per_coin": {c: len(v) for c, v in sorted(avail.items())}}


# --------------------------------------------------------------- the maps

def logit(p: float) -> float:
    p = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(p / (1.0 - p))


def expit(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logistic(design: Sequence[Sequence[float]], y: Sequence[int],
                 max_iter: int = 60, tol: float = 1e-10,
                 ridge: float = 1e-8) -> list[float] | None:
    """Newton-Raphson MLE. Returns None if it does not converge.

    `ridge` is a numerical floor on the Hessian only -- 1e-8 cannot shrink a
    coefficient that the data identifies, and without it a separated fold
    raises instead of returning a refusal.
    """
    k = len(design[0])
    beta = [0.0] * k
    for _ in range(max_iter):
        grad = [0.0] * k
        hess = [[ridge if i == j else 0.0 for j in range(k)] for i in range(k)]
        for row, yi in zip(design, y):
            z = sum(b * x for b, x in zip(beta, row))
            mu = expit(z)
            w = max(mu * (1.0 - mu), 1e-12)
            resid = yi - mu
            for i in range(k):
                grad[i] += resid * row[i]
                for j in range(k):
                    hess[i][j] += w * row[i] * row[j]
        step = _solve(hess, grad)
        if step is None:
            return None
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < tol:
            return beta
    return None


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None on a singular system."""
    n = len(b)
    m = [row[:] + [bi] for row, bi in zip(a, b)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-14:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def design_for(model: str, mids: Sequence[float]) -> list[list[float]]:
    x = [logit(m) for m in mids]
    if model == "anchored":              # logit p = b * logit m   (DEPLOY form)
        return [[xi] for xi in x]
    if model == "affine":                # logit p = a + b * logit m (ESTIMATE)
        return [[1.0, xi] for xi in x]
    if model == "two_slope":             # a + b_low*x*1{m<.5} + b_high*x*1{m>=.5}
        return [[1.0, xi if xi < 0 else 0.0, xi if xi >= 0 else 0.0] for xi in x]
    raise ValueError(f"unknown model {model!r}")


def predict(model: str, beta: Sequence[float], mids: Sequence[float]) -> list[float]:
    return [expit(sum(b * xi for b, xi in zip(beta, row)))
            for row in design_for(model, mids)]


def fit_isotonic_bins(mids: Sequence[float], y: Sequence[int],
                      n_bins: int = 10) -> list[float]:
    """10-bin isotonic on the mid. Plan section 3.1 keeps it as a REFERENCE.

    It is measured WORSE than the raw book out of sample (+0.0012 log-loss);
    it is retained so that claim stays falsifiable, not because it is a
    candidate.
    """
    tot = [0.0] * n_bins
    cnt = [0] * n_bins
    for m, yi in zip(mids, y):
        k = min(int(m * n_bins), n_bins - 1)
        tot[k] += yi
        cnt[k] += 1

    # Pool-adjacent-violators. A block is (value, weight, bins_covered) so the
    # solution can be expanded back onto the ORIGINAL bin grid exactly; an empty
    # bin carries zero weight and simply inherits its block's value.
    blocks: list[list[float]] = []
    for k in range(n_bins):
        w = float(cnt[k])
        # LAPLACE THE BIN, NEVER CLAMP THE PREDICTION. A bin with no successes
        # gave v = 0.0 exactly; PAVA does not merge a degenerate BOUNDARY bin
        # (its neighbour is larger, so the pool condition never fires), and
        # predict_isotonic then clamped to _EPS = 1e-12 -- so ONE test row cost
        # -log(1e-12) = 27.63 nats, i.e. +0.0124 on a 2,237-row day against an
        # observed isotonic delta of +0.0132. The clamp turned a missing
        # observation into a near-certain prediction of the opposite outcome.
        # (k+0.5)/(n+1) keeps the bin interior and is monotone-preserving.
        v = ((tot[k] + 0.5) / (w + 1.0)) if w else 0.0
        blocks.append([v, w, 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0] - 1e-15:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            w = w1 + w2
            merged = ((v1 * w1 + v2 * w2) / w) if w else (v1 + v2) / 2.0
            blocks.append([merged, w, n1 + n2])

    out: list[float] = []
    for v, _w, n_covered in blocks:
        out.extend([v] * int(n_covered))
    if len(out) != n_bins:                 # cannot happen; refuse rather than guess
        raise AssertionError(f"isotonic expanded to {len(out)} of {n_bins} bins")
    return out


def predict_isotonic(bins: Sequence[float], mids: Sequence[float]) -> list[float]:
    n = len(bins)
    return [min(max(bins[min(int(m * n), n - 1)], _EPS), 1.0 - _EPS) for m in mids]


# --------------------------------------------------------------- scoring

def log_loss(p: Sequence[float], y: Sequence[int]) -> float:
    return -sum(yi * math.log(max(pi, _EPS)) + (1 - yi) * math.log(max(1 - pi, _EPS))
                for pi, yi in zip(p, y)) / len(y)


def brier(p: Sequence[float], y: Sequence[int]) -> float:
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(y)


def cluster_bootstrap_delta(per_cluster: Sequence[tuple[list[float], list[float],
                                                        list[int]]],
                            n_boot: int, seed: int) -> dict[str, float] | None:
    """Paired delta (challenger - baseline) in log-loss and Brier, resampled
    over CLUSTERS. Returns None when there is fewer than one clusterable unit."""
    import random
    if len(per_cluster) < 2:
        return None
    rng = random.Random(seed)
    lo_ll: list[float] = []
    lo_br: list[float] = []
    idx = range(len(per_cluster))
    for _ in range(n_boot):
        pick = [per_cluster[rng.choice(idx)] for _ in idx]
        pb = [v for c in pick for v in c[0]]
        pc = [v for c in pick for v in c[1]]
        yy = [v for c in pick for v in c[2]]
        if not yy:
            continue
        lo_ll.append(log_loss(pc, yy) - log_loss(pb, yy))
        lo_br.append(brier(pc, yy) - brier(pb, yy))
    if not lo_ll:
        return None
    lo_ll.sort()
    lo_br.sort()
    q = lambda v, a: v[min(len(v) - 1, max(0, int(a * len(v))))]
    return {"d_logloss_lo": q(lo_ll, 0.025), "d_logloss_hi": q(lo_ll, 0.975),
            "d_brier_lo": q(lo_br, 0.025), "d_brier_hi": q(lo_br, 0.975),
            "n_clusters": len(per_cluster)}


# ------------------------------------------------------------ walk-forward

MODELS = ("raw_book", "anchored_b", "affine_ab", "two_slope", "isotonic10")


def _fit_all(train: Sequence[dict[str, Any]]) -> dict[str, Any]:
    mids = [r["mid"] for r in train]
    y = [r["y"] for r in train]
    fits: dict[str, Any] = {}
    for name, model in (("anchored_b", "anchored"), ("affine_ab", "affine"),
                        ("two_slope", "two_slope")):
        beta = fit_logistic(design_for(model, mids), y)
        fits[name] = {"model": model, "beta": beta}
    fits["isotonic10"] = {"model": "isotonic", "bins": fit_isotonic_bins(mids, y)}
    return fits


def _predict_all(fits: dict[str, Any], test: Sequence[dict[str, Any]]
                 ) -> dict[str, list[float]]:
    mids = [r["mid"] for r in test]
    out = {"raw_book": [min(max(m, _EPS), 1 - _EPS) for m in mids]}
    for name, f in fits.items():
        if f["model"] == "isotonic":
            out[name] = predict_isotonic(f["bins"], mids)
        elif f["beta"] is None:
            out[name] = None
        else:
            out[name] = predict(f["model"], f["beta"], mids)
    return out


def walk_forward(rows: Sequence[dict[str, Any]], n_boot: int = 2000,
                 seed: int = 20260823) -> dict[str, Any]:
    """Fit on days strictly before d, score day d. Never refit within a test day."""
    core = [r for r in rows if abs(logit(r["mid"])) <= CORE_MAX_ABS_LOGIT]
    days = sorted({r["day"] for r in core})
    per_day: list[dict[str, Any]] = []
    pooled: dict[str, list[float]] = {m: [] for m in MODELS}
    pooled_y: list[int] = []
    pooled_key: list[tuple[str, str]] = []

    for d in days[1:]:
        train = [r for r in core if r["day"] < d]
        test = [r for r in core if r["day"] == d]
        if len(train) < 300 or not test:
            per_day.append({"day": d, "status": "WARMUP",
                            "n_train": len(train), "n_test": len(test)})
            continue
        fits = _fit_all(train)
        preds = _predict_all(fits, test)
        y = [r["y"] for r in test]
        row: dict[str, Any] = {"day": d, "status": "SCORED",
                               "n_train": len(train), "n_test": len(test),
                               "n_train_windows": len({r["slug"] for r in train}),
                               "n_test_windows": len({r["slug"] for r in test}),
                               "fit": {k: v.get("beta") for k, v in fits.items()
                                       if v["model"] != "isotonic"},
                               "score": {}}
        base = preds["raw_book"]
        for m in MODELS:
            p = preds.get(m)
            if p is None:
                row["score"][m] = {"status": "FIT_FAILED"}
                continue
            row["score"][m] = {"logloss": log_loss(p, y), "brier": brier(p, y),
                               "d_logloss": log_loss(p, y) - log_loss(base, y),
                               "d_brier": brier(p, y) - brier(base, y)}
            pooled[m].extend(p)
        pooled_y.extend(y)
        pooled_key.extend((r["day"], r["slug"]) for r in test)
        per_day.append(row)

    scored = [r for r in per_day if r.get("status") == "SCORED"]
    out: dict[str, Any] = {"days": days, "n_days": len(days),
                           "per_day": per_day, "n_scored_days": len(scored)}

    if not scored:
        out["pooled"] = {"status": "NO_SCORED_DAY"}
        return out

    intervals: dict[str, Any] = {}
    for m in MODELS:
        if m == "raw_book" or not pooled[m]:
            continue
        by_win: dict[tuple[str, str], tuple[list[float], list[float], list[int]]] = {}
        by_day: dict[str, tuple[list[float], list[float], list[int]]] = {}
        for pb, pc, yy, key in zip(pooled["raw_book"], pooled[m], pooled_y, pooled_key):
            for store, k in ((by_win, key), (by_day, key[0])):
                cell = store.setdefault(k, ([], [], []))
                cell[0].append(pb)
                cell[1].append(pc)
                cell[2].append(yy)
        win_ci = cluster_bootstrap_delta(list(by_win.values()), n_boot, seed)
        day_ci = cluster_bootstrap_delta(list(by_day.values()), n_boot, seed + 1)
        intervals[m] = {
            "point_d_logloss": log_loss(pooled[m], pooled_y)
                               - log_loss(pooled["raw_book"], pooled_y),
            "point_d_brier": brier(pooled[m], pooled_y)
                             - brier(pooled["raw_book"], pooled_y),
            "window_clustered": win_ci,
            "day_clustered": day_ci if day_ci else "DAY_BLOCK_UNAVAILABLE",
        }
    out["pooled"] = {"n_rows": len(pooled_y), "n_scored_days": len(scored),
                     "intervals": intervals}
    return out


def in_sample_fits(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive `a`/`b` on the whole sample, core and extreme domains.

    NOT a result on its own -- BE_BELIEF_PLAN section 6.1 rule 1 is walk-forward
    only. These are the diagnostics section 4.4 and 5.1 need: the intercept is
    the DRIFT channel and is estimated everywhere and deployed nowhere.
    """
    def fit(subset: Sequence[dict[str, Any]], model: str) -> list[float] | None:
        if len(subset) < 50:
            return None
        return fit_logistic(design_for(model, [r["mid"] for r in subset]),
                            [r["y"] for r in subset])

    core = [r for r in rows if abs(logit(r["mid"])) <= CORE_MAX_ABS_LOGIT]
    extreme = [r for r in rows if abs(logit(r["mid"])) > CORE_MAX_ABS_LOGIT]
    out: dict[str, Any] = {
        "core": {"n": len(core), "affine": fit(core, "affine"),
                 "anchored": fit(core, "anchored"),
                 "two_slope": fit(core, "two_slope")},
        "extreme": {"n": len(extreme), "affine": fit(extreme, "affine")},
        "all": {"n": len(rows), "affine": fit(rows, "affine")},
    }
    out["by_day"] = {d: {"n": len(g), "affine": fit(g, "affine")}
                     for d, g in sorted(_group(core, "day").items())}
    out["by_coin"] = {c: {"n": len(g), "affine": fit(g, "affine")}
                      for c, g in sorted(_group(core, "coin").items())}
    out["by_r"] = {str(int(r_)): {"n": len(g), "affine": fit(g, "affine")}
                   for r_, g in sorted(_group(core, "r").items())}
    return out


def _group(rows: Sequence[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    out: dict[Any, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return dict(out)


def calibration_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The plan's section 2.2 table: gap at the mid AND at the executable prices.

    `realised - ask` is what an aggressor buying Up earns; `bid - realised` is
    what one selling Up earns. The trading question is never "realised vs mid".
    """
    out = []
    for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        cell = [r for r in rows if lo <= r["mid"] < hi or (hi == 1.0 and r["mid"] == 1.0)]
        if not cell:
            continue
        n = len(cell)
        mean = lambda f: sum(f(r) for r in cell) / n
        out.append({
            "bucket": f"{lo:.1f}-{hi:.1f}", "n": n,
            "n_windows": len({r["slug"] for r in cell}),
            "mid": mean(lambda r: r["mid"]), "bid": mean(lambda r: r["bid"]),
            "ask": mean(lambda r: r["ask"]), "realised": mean(lambda r: r["y"]),
            "gap_vs_mid": mean(lambda r: r["y"] - r["mid"]),
            "buy_up_at_ask": mean(lambda r: r["y"] - r["ask"]),
            "sell_up_at_bid": mean(lambda r: r["bid"] - r["y"]),
            "spread": mean(lambda r: r["spread"]),
            "age_s": mean(lambda r: r["age_s"]),
        })
    return out


# ------------------------------------------------------------------- run

def run(per_coin: int | None, coins: Sequence[str] | None, n_boot: int,
        progress: bool = True) -> dict[str, Any]:
    built = build_rows(per_coin=per_coin, coins=coins, progress=progress)
    rows = built["rows"]
    if not rows:
        raise SystemExit("no rows built -- check era coverage and resolutions")
    res: dict[str, Any] = {
        "protocol": "be_belief_v1",
        "status": "DEVELOPMENT_NOT_DECISION_ELIGIBLE",
        "implements": "BE_BELIEF_PLAN.md section 12 steps 1-3",
        "decision_elapsed_s": list(DECISION_ELAPSED_S),
        "core_max_abs_logit": CORE_MAX_ABS_LOGIT,
        "n_rows": len(rows),
        "n_windows": len({r["slug"] for r in rows}),
        "n_days": len({r["day"] for r in rows}),
        "days": sorted({r["day"] for r in rows}),
        "per_coin_cap": per_coin,
        "sampling_rule": ("DAY_STRATIFIED_ROUND_ROBIN" if per_coin
                          else "WHOLE_COVERED_POPULATION"),
        "refused": built["refused"],
        "outcome_tally": built["outcome_tally"],
        "n_available_per_coin": built["n_available_per_coin"],
        "up_rate": sum(r["y"] for r in rows) / len(rows),
        "up_rate_by_window": (
            sum({r["slug"]: r["y"] for r in rows}.values())
            / len({r["slug"] for r in rows})),
        "calibration": calibration_table(rows),
        "in_sample": in_sample_fits(rows),
        "walk_forward": walk_forward(rows, n_boot=n_boot),
    }
    res["provenance"] = fi.provenance(sampled=built["slugs_sampled"])
    return res



# --- MONITOR, NOT GATE -------------------------------------------------------
# `STEP5_MIN_DAYS = 7` and `would_ship_today: RECALIBRATION|IDENTITY` used to
# live here: BE_BELIEF_PLAN section 12 step 5's automatic promotion bar. THE PLAN
# DELETED THAT GATE AND THE CODE KEPT ENFORCING IT -- so at 7 days this file
# would have printed an automatic promotion verdict the plan says cannot exist,
# on a rule whose type-I rate is 25% at three day-clusters (the day-clustered CI
# is exactly [min, max] of three numbers, so "excludes 0" means "all three days
# share a sign"). Deleting prose does not delete a rule that is implemented.
#
# Promotion now requires a NEW FROZEN PROTOCOL with a CALENDAR trigger. This
# function reports; it never decides.
MONITOR_MIN_DAYS = 2        # below this the monitor records a status, not a b_hat


def population_of(res: dict[str, Any]) -> str:
    """DERIVE the population from the receipt. Never default it.

    The first version of this read `res.get("population", "ALL_COINS_POOLED")`.
    Nothing ever SET `population`, so the default fired unconditionally and the
    btc/eth receipt -- the one whose entire reason for existing is that it is NOT
    all coins -- declared itself `ALL_COINS_POOLED`. A hardcoded label sitting in
    the position of a computed value, shipped inside the fix for that same
    defect, and it defeated `out_paths()`'s own rule that the population goes in
    the name by making the body contradict the name.

    So the population is read off the coins the fit actually saw, and an
    unreadable receipt REFUSES rather than guessing the commonest case.
    """
    by_coin = (res.get("in_sample") or {}).get("by_coin") or {}
    if not by_coin:
        return "UNDECLARED — receipt carries no per-coin fit; population UNKNOWN"
    coins = sorted(by_coin)
    verdict = sorted(("btc", "eth"))
    if coins == verdict:
        return "VERDICT_COINS_ONLY [btc, eth] (FLOW_MODEL_PROTOCOL_V5:333)"
    return f"{len(coins)} COINS POOLED {coins}"


def data_vintage(res: dict[str, Any]) -> str:
    """Fingerprint WHAT THE RUN SAW, not when it ran.

    Two receipts printed byte-identical `days_sampled` across samples 1,071 core
    rows apart, and a whole comparison was built on the assumption they described
    the same data. A day list is not a vintage.

    The first fix read `provenance["generated_at"]` -- a key `fi.provenance()`
    does not return, so it was unconditionally "UNSTAMPED": the field that exists
    to catch a vintage split was wired to something that never arrives.

    A wall-clock stamp would also have been the weaker choice. What distinguishes
    these runs is COVERAGE (btc 680 -> 892 -> 901 windows), which is a property of
    the data, so the fingerprint is built from coverage and REFUSES when it
    cannot be.
    """
    per_coin = (res.get("coverage") or {}).get("n_available_per_coin") \
        or res.get("n_available_per_coin")
    if not isinstance(per_coin, dict) or not per_coin:
        prov = res.get("provenance") or {}
        days = prov.get("days_sampled")
        if days:
            return (f"COVERAGE_UNKNOWN — {len(days)} day(s) {days[0]}..{days[-1]}; "
                    f"A DAY LIST IS NOT A VINTAGE and this receipt carries no "
                    f"per-coin coverage, so two different samples are "
                    f"indistinguishable here")
        return "UNSTAMPED — receipt carries neither coverage nor days"
    tot = sum(per_coin.values())
    parts = ",".join(f"{c}:{n}" for c, n in sorted(per_coin.items()))
    return f"windows_total={tot} [{parts}]"


def monitor(res: dict[str, Any]) -> dict[str, Any]:
    """Report what the promotion protocol will need. Decide nothing.

    Every field here exists because a review found it MISSING from a reading
    that was made anyway:

      n_day_clusters   at k=3 a percentile cluster bootstrap returns [min, max]
                       exactly (P(all-one-cluster) = 1/27 > 0.025), so a reader
                       cannot tell a 95% CI from a sample range without k.
      interval_method  at small n_days the convention decides the verdict.
      data_cut         two receipts printed BYTE-IDENTICAL `days_sampled` across
                       samples 1,071 rows apart. A day list is not a vintage.
      no_edge_benchmark  a k-parameter map fitted on n_eff is EXPECTED to lose
                       out of sample by k/(2*n_eff) even when b == 1 exactly.
                       Testing delta against 0 conflates "b = 1" with "b != 1
                       but not estimable at this n".
      scored_population  the same delta reads 8.3x differently row-weighted vs
                       day-weighted, and the plan's own unit is days.
    """
    wf = res.get("walk_forward", {})
    pooled = wf.get("pooled", {})
    iv = pooled.get("intervals", {})
    n_days = wf.get("n_days", 0)
    out: dict[str, Any] = {
        "role": "MONITOR — reports, never promotes",
        "promotion_rule": "a NEW FROZEN PROTOCOL with a calendar trigger; no bar lives here",
        "n_days_present": n_days,
        "n_scored_days": wf.get("n_scored_days", 0),
        "status": ("REPORTING" if n_days >= MONITOR_MIN_DAYS
                   else "INSUFFICIENT_DAYS_FOR_ANY_INTERVAL"),
        "interval_method": "percentile cluster bootstrap over UTC days",
        "data_vintage": data_vintage(res),
        "population": population_of(res),
        "models": {},
    }
    # PER-DAY DELTAS COME FROM `walk_forward.per_day`, which is where they live.
    # The first version read them from a `per_day` key inside `day_clustered`
    # that does not exist, so the day-weighted column -- the statistic the plan's
    # own section 6.1 rule 4 makes PRIMARY -- rendered empty in every row of
    # every receipt while the row-weighted figure stood beside it unlabelled.
    # The two readings differed by 8.3x the last time only one was reported.
    day_delta: dict[str, list[float]] = {}
    for rec in (wf.get("per_day") or []):
        if rec.get("status") != "SCORED":
            continue
        for mm, sc in (rec.get("score") or {}).items():
            if mm == "raw_book" or not isinstance(sc, dict):
                continue
            if isinstance(sc.get("d_logloss"), (int, float)):
                day_delta.setdefault(mm, []).append(sc["d_logloss"])

    for m, v in iv.items():
        d = v.get("day_clustered")
        point = v.get("point_d_logloss")
        if not isinstance(d, dict):
            out["models"][m] = {"reading": "DAY_BLOCK_UNAVAILABLE", "point": point}
            continue
        lo, hi = d["d_logloss_lo"], d["d_logloss_hi"]
        k = d["n_clusters"]
        per_day = day_delta.get(m) or []
        degenerate = (k <= 3 and per_day
                      and abs(lo - min(per_day)) < 1e-12
                      and abs(hi - max(per_day)) < 1e-12)
        out["models"][m] = {
            "reading": ("BEATS_THE_BOOK" if hi < 0 else
                        "WORSE_THAN_THE_BOOK" if lo > 0 else
                        "INDISTINGUISHABLE_FROM_THE_BOOK"),
            "point_row_weighted": point,
            "point_day_weighted": (sum(per_day) / len(per_day)) if per_day else None,
            "ci95": [lo, hi],
            "n_day_clusters": k,
            # DETECTED, not assumed: is this "CI" just the range of k numbers?
            "ci_is_sample_range": bool(degenerate),
            "ci_caveat": ("at k<=3 this interval IS [min,max] of the per-day "
                          "deltas; 'excludes 0' means 'all days share a sign', "
                          "a 25% event under a symmetric null" if degenerate else None),
        }
    return out



def report(res: dict[str, Any]) -> list[str]:
    p = res["provenance"]
    L = ["# BE-Belief — walk-forward recalibration of the venue book", "",
         "**Status: DEVELOPMENT, not decision eligible.** Produced by",
         "`be_belief.py`, implementing `plans/BE_BELIEF_PLAN.md` §12 steps 1–3.",
         "", "## Receipt", "",
         f"- rows {res['n_rows']:,} over {res['n_windows']:,} windows, "
         f"{res['n_days']} UTC day(s): {', '.join(res['days'])}",
         f"- sampling rule `{res['sampling_rule']}`"
         + (f", cap {res['per_coin_cap']}/coin" if res['per_coin_cap'] else ""),
         f"- `days_sampled` {p.get('days_sampled')} (n={p.get('n_days_sampled')}); "
         f"`days_declared` {p.get('days_declared')}",
         f"- decision times, elapsed s: {res['decision_elapsed_s']} "
         f"(r = 270, 240, 180, 120, 60)",
         f"- up-rate {res['up_rate']:.4f} per row, "
         f"{res['up_rate_by_window']:.4f} per window",
         f"- refused: {res['refused'] or 'none'}",
         "",
         "One outcome per window is shared by every decision time in it, so "
         "every n here is inflated ~5× and intervals cluster on window, then day.",
         "", "## Calibration — at the mid AND at the executable prices", "",
         "`spread` here is POOLED ACROSS COINS and is therefore a pooling "
         "artefact in exactly the way U8 named: ATM spread is **1 tick** on "
         "btc/eth and 3–7 ticks on the thin coins, so a pooled figure reports "
         "neither. Read it as a mixture, never as a venue spread.", "",
         "| bucket | rows | windows | mid | bid | ask | realised | gap vs mid | "
         "buy Up at ask | sell Up at bid | spread | age s |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in res["calibration"]:
        L.append(f"| {c['bucket']} | {c['n']} | {c['n_windows']} | {c['mid']:.3f} | "
                 f"{c['bid']:.3f} | {c['ask']:.3f} | {c['realised']:.3f} | "
                 f"{c['gap_vs_mid']:+.3f} | {c['buy_up_at_ask']:+.3f} | "
                 f"{c['sell_up_at_bid']:+.3f} | {c['spread']:.3f} | {c['age_s']:.1f} |")

    ins = res["in_sample"]
    L += ["", "## In-sample `a` / `b` — DIAGNOSTIC, never deployed", "",
          "`a` is the drift channel: estimated always, deployed never (§4.3).", "",
          "| sample | n | a | b |", "|---|---:|---:|---:|"]
    for label, key in (("core |logit m| ≤ 3", "core"), ("extreme", "extreme"),
                       ("all rows", "all")):
        f = ins[key].get("affine")
        if f:
            L.append(f"| {label} | {ins[key]['n']:,} | {f[0]:+.3f} | {f[1]:.3f} |")
    L += ["", "| day | n | a | b |", "|---|---:|---:|---:|"]
    for d, v in ins["by_day"].items():
        if v.get("affine"):
            L.append(f"| {d} | {v['n']:,} | {v['affine'][0]:+.3f} | {v['affine'][1]:.3f} |")
    L += ["", "| r (s remaining) | n | a | b |", "|---|---:|---:|---:|"]
    for r_, v in sorted(ins["by_r"].items(), key=lambda kv: -float(kv[0])):
        if v.get("affine"):
            L.append(f"| {r_} | {v['n']:,} | {v['affine'][0]:+.3f} | {v['affine'][1]:.3f} |")

    wf = res["walk_forward"]
    L += ["", "## Walk-forward — fit on days < d, score day d", "",
          f"{wf['n_scored_days']} scored day(s) of {wf['n_days']} present.", ""]
    if wf["n_scored_days"]:
        L += ["| day | n test | model | log-loss | Δ | Brier | Δ |",
              "|---|---:|---|---:|---:|---:|---:|"]
        for row in wf["per_day"]:
            if row.get("status") != "SCORED":
                L.append(f"| {row['day']} | — | {row.get('status')} | | | | |")
                continue
            for m in MODELS:
                s = row["score"].get(m, {})
                if "logloss" not in s:
                    continue
                L.append(f"| {row['day']} | {row['n_test']:,} | `{m}` | "
                         f"{s['logloss']:.4f} | {s['d_logloss']:+.4f} | "
                         f"{s['brier']:.4f} | {s['d_brier']:+.4f} |")
        pooled = wf.get("pooled", {})
        iv = pooled.get("intervals", {})
        if iv:
            L += ["", "### Pooled deltas vs the raw book, with intervals", "",
                  "| model | Δ log-loss | window-clustered 95% | day-clustered 95% |",
                  "|---|---:|---|---|"]
            for m, v in iv.items():
                w = v["window_clustered"]
                d = v["day_clustered"]
                ws = (f"[{w['d_logloss_lo']:+.4f}, {w['d_logloss_hi']:+.4f}] "
                      f"({w['n_clusters']} win)") if w else "unavailable"
                ds = (f"[{d['d_logloss_lo']:+.4f}, {d['d_logloss_hi']:+.4f}] "
                      f"({d['n_clusters']} days)") if isinstance(d, dict) else f"**{d}**"
                L.append(f"| `{m}` | {v['point_d_logloss']:+.4f} | {ws} | {ds} |")
    v = monitor(res)
    L += ["", "## Monitor — reports, never promotes", "",
          f"**Role:** {v['role']}. **Promotion:** {v['promotion_rule']}.",
          "",
          f"Population **{v['population']}** · vintage **{v['data_vintage']}** · "
          f"{v['n_days_present']} day(s) present, {v['n_scored_days']} scored · "
          f"status **{v['status']}**.",
          "",
          "> The §12 step-5 gate that used to be rendered here is **DELETED**. It "
          "read *day-clustered CI excludes 0, else Identity* at 7 days and "
          "printed `would_ship_today`. The plan deleted it and this file kept "
          "enforcing it — so at 7 days a machine-generated receipt would have "
          "announced an automatic promotion the plan says cannot exist. "
          "**Deleting prose does not delete a rule that is implemented.**",
          "",
          "Sign convention: Δ is challenger **minus** baseline on log-loss, so "
          "**negative beats the raw book**.", "",
          "| model | Δ row-wtd | Δ **day-wtd** | day-clustered 95% | k | reading |",
          "|---|---:|---:|---|---:|---|"]
    for m, mv in v["models"].items():
        if "ci95" not in mv:
            L.append(f"| `{m}` | {mv['point']:+.5f} | — | — | — | "
                     f"**{mv['reading']}** |")
            continue
        dw = mv.get("point_day_weighted")
        # A receipt written before the monitor existed carries no per-day
        # deltas, so the day-weighted column is genuinely UNKNOWN, not zero.
        # Render the refusal; never fabricate the number the plan calls primary.
        dws = f"{dw:+.5f}" if isinstance(dw, (int, float)) else "—"
        L.append(f"| `{m}` | {mv['point_row_weighted']:+.5f} | "
                 f"{dws} | "
                 f"[{mv['ci95'][0]:+.5f}, {mv['ci95'][1]:+.5f}] | "
                 f"{mv['n_day_clusters']} | **{mv['reading']}** |")
    caveats = {m: mv.get("ci_caveat") for m, mv in v["models"].items()
               if mv.get("ci_is_sample_range")}
    if caveats:
        L += ["",
              "> **⚠ DETECTED, not assumed — the intervals above are NOT 95% "
              "intervals.** For " + ", ".join(f"`{m}`" for m in caveats) + ": "
              + next(iter(caveats.values())) + " Both columns are shown because "
              "the point estimate is a ROW average while the interval under it "
              "resamples DAYS, and the plan's primary unit is days — the two "
              "readings differed by 8.3× the last time only one was reported."]
    L += ["", "## What this does and does not license", "",
          "- **Nothing here promotes anything.** Promotion requires a new frozen "
          "protocol with a calendar trigger. Where the table says "
          "`DAY_BLOCK_UNAVAILABLE`, the sample has one day block and no "
          "day-clustered interval exists — that is a refusal, not a small "
          "number.",
          "- The deployed map pins `a = 0`. Any gain attributable to `a` is a "
          "bet that the observed drift continues, which is the directional "
          "claim this programme does not make.",
          "- Nothing here is P&L. The unconditional gap is not harvestable: the "
          "measured selection haircut is 60–97 % and it is BE-FlowAndFills' "
          "term, not this module's."]
    return L


# -------------------------------------------------------------- selftest

def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # 1-3: the link and its inverse.
    ok(abs(expit(logit(0.37)) - 0.37) < 1e-12, "expit inverts logit")
    ok(abs(logit(0.5)) < 1e-12, "logit is 0 at the anchor")
    ok(expit(-800) >= 0.0 and expit(800) <= 1.0, "expit does not overflow")

    # 4-6: the resolution predicate must agree with the code that WRITES the
    # file. If the collector's rule changes, this fails rather than drifting.
    try:
        import collect_pm
        cases = [{"closed": True}, {"outcomePrices": '["1", "0"]'},
                 {"outcomePrices": '["0.4", "0.6"]'}, {"closed": False}]
        ok(all(is_final(c) == collect_pm.is_final(c) for c in cases),
           "is_final agrees with collect_pm.is_final")
    except ImportError:
        ok(True, "collect_pm unavailable; local is_final unchecked")
    ok(is_final({"outcomePrices": '["0.165", "0.835"]'}) is False,
       "live gamma prices are NOT a resolution")
    ok(is_final({"outcomePrices": '["1", "0"]'}) is True, "degenerate is final")

    # 7-10: the OUTCOME field. A final row carries `winners`, not prices.
    ok(outcome_of({"winners": {"Up": True, "Down": False}}) == 1, "Up wins reads 1")
    ok(outcome_of({"winners": {"Up": False, "Down": True}}) == 0, "Down wins reads 0")
    ok(outcome_of({"winners": {"Up": False, "Down": False}}) is None,
       "no winner is refused, not guessed")
    ok(outcome_of({"outcomePrices": '["0.165", "0.835"]'}) is None,
       "live gamma prices are not an outcome")

    # 7-9: the pair state builder must agree with the canonical mid builder.
    pts = [(1.0, 0.40, 0.44), (2.0, 0.50, 0.52), (200.0, 0.10, 0.90)]
    gaps = [(5.0, 9.0)]
    pair = state_segments_from_pairs(pts, gaps)
    canon = fi.state_segments_from_points([(t, (b + a) / 2) for t, b, a in pts], gaps)
    ok(len(pair) == len(canon), "pair builder yields the canonical segment count")
    ok(all(abs(s0 - c0) < 1e-12 and abs(s1 - c1) < 1e-12
           and abs((b + a) / 2 - cm) < 1e-12
           for (s0, s1, b, a), (c0, c1, cm) in zip(pair, canon)),
       "pairs match canonical mid, start and end exactly")
    ok(state_segments_from_pairs([(1.0, 0.4, 0.44)], [(1.1, 2.0)]) == [],
       "a gap between receipt and maturity kills the quote")

    # 10-11: the boundary-quote guard, which two codebases got wrong.
    ok(state_segments_from_pairs([(1.0, 0.0, 1.0)], []) != [],
       "boundary quote 0/1 is RETAINED")
    ok(state_segments_from_pairs([(1.0, 0.5, 0.5)], []) == [],
       "crossed or equal quote is rejected")

    # 12-14: the maps. A book that is already calibrated must fit b = 1, a = 0.
    mids = [0.2, 0.35, 0.5, 0.65, 0.8] * 60
    y = [1 if (i % 100) / 100.0 < m else 0 for i, m in enumerate(mids)]
    beta = fit_logistic(design_for("affine", mids), y)
    ok(beta is not None, "affine fit converges")
    exact = fit_logistic(design_for("anchored", [0.3, 0.7] * 50),
                         [0] * 50 + [1] * 50)
    ok(exact is not None, "anchored fit converges")
    ok(len(design_for("two_slope", [0.3, 0.7])[0]) == 3, "two-slope has 3 columns")

    # 15: a perfectly calibrated synthetic book recovers b ~ 1 and a ~ 0.
    syn_m, syn_y = [], []
    for k in range(1, 20):
        m = k / 20.0
        n = 400
        syn_m += [m] * n
        syn_y += [1] * round(m * n) + [0] * (n - round(m * n))
    b2 = fit_logistic(design_for("affine", syn_m), syn_y)
    ok(b2 is not None and abs(b2[0]) < 0.05 and abs(b2[1] - 1.0) < 0.05,
       f"calibrated book recovers a=0,b=1 (got {b2})")

    # 16: and a book that is systematically UNDERconfident recovers b > 1.
    und_m, und_y = [], []
    for k in range(1, 20):
        m = k / 20.0
        true_p = expit(1.4 * logit(m))
        n = 400
        und_m += [m] * n
        und_y += [1] * round(true_p * n) + [0] * (n - round(true_p * n))
    b3 = fit_logistic(design_for("affine", und_m), und_y)
    ok(b3 is not None and abs(b3[1] - 1.4) < 0.08,
       f"underconfident book recovers b=1.4 (got {b3})")

    # 17-18: scoring.
    ok(abs(log_loss([0.5] * 4, [1, 0, 1, 0]) - math.log(2)) < 1e-9,
       "log-loss of a coin is log 2")
    ok(abs(brier([1.0, 0.0], [1, 0])) < 1e-12, "Brier of a perfect forecast is 0")

    # 19-20: THE DAY RULE. One day block is a refusal, never an interval.
    one = [([0.5], [0.6], [1])]
    ok(cluster_bootstrap_delta(one, 50, 1) is None,
       "a single cluster yields no interval")
    two = [([0.5], [0.6], [1]), ([0.5], [0.4], [0])]
    ok(cluster_bootstrap_delta(two, 50, 1) is not None,
       "two clusters do yield an interval")

    # 21: walk-forward must never score the first day -- it has no training set.
    rows = [{"slug": f"s{i}", "coin": "btc", "day": d, "elapsed": 30.0, "r": 270.0,
             "bid": 0.4, "ask": 0.42, "mid": 0.41, "spread": 0.02, "age_s": 0.1,
             "y": i % 2}
            for d in ("2026-08-20", "2026-08-21") for i in range(400)]
    wf = walk_forward(rows, n_boot=20)
    ok(all(r["day"] != "2026-08-20" or r.get("status") != "SCORED"
           for r in wf["per_day"]), "the first day is never scored")

    # 22: and with two days present, the day-clustered arm still refuses,
    # because only ONE day can be scored when the first is spent on training.
    scored_days = {r["day"] for r in wf["per_day"] if r.get("status") == "SCORED"}
    ok(len(scored_days) <= 1, "two present days give at most one scored day")

    # 23: isotonic is monotone by construction.
    iso = fit_isotonic_bins([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
                             0.85, 0.95] * 20,
                            ([0, 0, 1, 0, 1, 0, 1, 1, 1, 1] * 20))
    ok(all(iso[i] <= iso[i + 1] + 1e-12 for i in range(len(iso) - 1)),
       "isotonic bins are non-decreasing")

    # 24: the core domain is what the plan says it is.
    ok(abs(expit(CORE_MAX_ABS_LOGIT) - 0.9526) < 1e-3,
       "core domain edge is m = 0.9526")

    print(f"be_belief selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["run", "report"], default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=None,
                    help="cap windows per coin, spread ACROSS days (default: all)")
    ap.add_argument("--coins", type=str, default=None,
                    help="comma-separated subset, default all")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    coins = tuple(a.coins.split(",")) if a.coins else None
    if a.cmd == "report":            # re-render from the receipt; never re-fits
        # `coins` MUST be bound before this branch. It was not: today's
        # out_paths() change read it here and bound it eleven lines below, so
        # `report` raised UnboundLocalError -- and `report` is precisely the path
        # that re-renders a receipt WITHOUT re-fitting, i.e. the one needed to
        # compare two vintages. A scope bug introduced by the edit that fixed a
        # scope defect, in the only code path its own finding requires.
        oj, om = out_paths(coins)
        res = json.loads(oj.read_text())
        om.write_text("\n".join(report(res)) + "\n")
        print(f"re-rendered {om} from {oj}")
        return 0
    if a.cmd != "run":
        ap.print_help()
        return 1
    res = run(per_coin=a.per_coin, coins=coins, n_boot=a.boot)
    oj, om = out_paths(coins)
    oj.parent.mkdir(parents=True, exist_ok=True)
    oj.write_text(json.dumps(res, indent=1))
    lines = report(res)
    om.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nreceipt {oj}\nreport  {om}\nPOPULATION: {','.join(coins) if coins else 'ALL COINS (pooled)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
