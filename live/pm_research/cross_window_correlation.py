"""B6 -- cross-window same-coin correlation. DA measures, DE consumes.

Does holding positions in two windows of the same coin expose you to ONE risk
or two?  `DE_MODULE_PLAN` declares the `{Up,Down}` coupling ATOMIC and exact,
and the same-coin cross-window edge `SHARED_RISK` with the correlation
**unmeasured**.  This fills that number, and it is the open falsifier #2 in
`DA_INVENTORY_STATE_PLAN` -- "`net` proves not to be the risk-relevant state
... if correlated residuals across overlapping windows of the same coin
dominate the single-market residual".

**This probe contains no decision rule, no threshold and no verdict.**  That is
deliberate and it is the dispatch constraint: DA publishes the receipt, DE
reads it, and any rule the numbers imply goes to the coordinator rather than
being embedded here.  Contrast `inventory_walk.py`, which carries pre-registered
bars because its dispatch called for a verdict; this one must not.

Three measurements, THREE DIFFERENT POPULATIONS.  Read each figure with its own
denominator:

  1. CONCURRENCY   -- how many same-coin windows are quotable at once, from the
                      discovery grid.  Era-independent, all days.
  2. RESIDUAL      -- do settlement residuals of nearby windows covary?
                      Outcomes are era-independent and use every resolved
                      window; the mid(t0) variant needs the CLOB tape and is
                      therefore era-restricted.
  3. INVENTORY     -- do simulated positions covary, under the STANDARD
                      two-sided replay (`inventory_walk.simulate_window`,
                      imported unmodified)?  Era-restricted and sampled.

Reads only.  Modifies no other plane's file and writes nothing under
`data/pm_5min/tier1/`.

    python3 live/pm_research/cross_window_correlation.py --selftest
    python3 live/pm_research/cross_window_correlation.py run --per-coin-day 40
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:                 # inventory_walk imports flat
    sys.path.insert(0, str(_HERE))
if str(_REPO) not in sys.path:                 # tier1_pipeline imports by package
    sys.path.insert(0, str(_REPO))

import flow_intensity as fi                    # noqa: E402
import flow_fill_development as fd             # noqa: E402
import inventory_walk as iw                    # noqa: E402  (DE-owned; called, never edited)

PM = fi.PM
OUT = PM / "derived/cross_window_correlation_v1.json"

WINDOW_S = fi.WINDOW_S                         # 300 s
LAGS = (1, 2, 3, 6, 12)                        # windows: 5, 10, 15, 30, 60 minutes
N_BOOT = 2000
SEED = 20260823


# ---------------------------------------------------------------------------
# statistics -- correlation with an honest interval
# ---------------------------------------------------------------------------

def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


@dataclass(frozen=True)
class Pair:
    day: str
    x: float
    y: float


def correlate(pairs: Sequence[Pair], n_boot: int = N_BOOT,
              seed: int = SEED) -> dict[str, Any]:
    """Point estimate plus BOTH intervals, each labelled for what it ignores.

    The day-clustered interval is the honest one: adjacent windows of one coin
    on one day are the opposite of independent, which is the very thing being
    measured.  The pair-level interval is reported beside it precisely so the
    gap between them is visible rather than being a choice made silently.
    """
    if len(pairs) < 3:
        return {"n_pairs": len(pairs), "r": None, "reason": "insufficient pairs"}
    xs = [p.x for p in pairs]
    ys = [p.y for p in pairs]
    r = pearson(xs, ys)
    days = sorted({p.day for p in pairs})
    by_day: dict[str, list[Pair]] = collections.defaultdict(list)
    for p in pairs:
        by_day[p.day].append(p)

    rng = random.Random(seed)
    cluster: list[float] = []
    for _ in range(n_boot):
        picked: list[Pair] = []
        for _ in range(len(days)):
            picked.extend(by_day[days[rng.randrange(len(days))]])
        value = pearson([p.x for p in picked], [p.y for p in picked])
        if value is not None:
            cluster.append(value)
    naive: list[float] = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        value = pearson([p.x for p in sample], [p.y for p in sample])
        if value is not None:
            naive.append(value)

    def ci(values: list[float]) -> list[float] | None:
        if len(values) < 100:
            return None
        values = sorted(values)
        lo = values[int(0.025 * len(values))]
        hi = values[int(0.975 * len(values))]
        return [round(lo, 4), round(hi, 4)]

    return {
        "n_pairs": len(pairs),
        "n_day_clusters": len(days),
        "days": days,
        "r": None if r is None else round(r, 4),
        "ci95_day_clustered": ci(cluster),
        "ci95_pair_level_IGNORES_CLUSTERING": ci(naive),
    }


# ---------------------------------------------------------------------------
# 1. concurrency structure -- how much overlap actually exists
# ---------------------------------------------------------------------------

def _day_of(epoch_s: int) -> str:
    return datetime.fromtimestamp(epoch_s, timezone.utc).strftime("%Y%m%d")


def concurrency(markets: Mapping[str, Any]) -> dict[str, Any]:
    """Same-coin markets simultaneously quotable, from the discovery grid.

    LOWER BOUND, and the reason is worth stating: the interval used is
    [our discovery time, window_end].  The venue creates a market before we
    discover it, so true concurrency is at least this and possibly more.
    """
    by_coin: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for info in markets.values():
        by_coin[info.coin].append(
            (info.market_known_ns / 1e9, float(info.window_end_s))
        )

    out: dict[str, Any] = {}
    for coin, spans in sorted(by_coin.items()):
        events: list[tuple[float, int]] = []
        for start, end in spans:
            if end <= start:
                continue
            events.append((start, 1))
            events.append((end, -1))
        events.sort()
        live = 0
        counts: collections.Counter[int] = collections.Counter()
        seconds: collections.Counter[int] = collections.Counter()
        previous = None
        for when, delta in events:
            if previous is not None and when > previous and live > 0:
                counts[live] += 1                    # denominator: INTERVALS
                seconds[live] += when - previous     # denominator: TIME
            live += delta
            previous = when
        total_intervals = sum(counts.values()) or 1
        total_seconds = sum(seconds.values()) or 1.0
        ordered = sorted(counts.items())
        cumulative = 0
        p50 = p90 = maximum = 0
        for value, n in ordered:
            cumulative += n
            maximum = max(maximum, value)
            if p50 == 0 and cumulative >= 0.5 * total_intervals:
                p50 = value
            if p90 == 0 and cumulative >= 0.9 * total_intervals:
                p90 = value
        out[coin] = {
            "n_markets": len(spans),
            # TIME-weighted is the decision-relevant one: it answers "how often
            # am I actually holding two".  The interval count is reported beside
            # it because the two disagree -- transitions are short and numerous,
            # so counting intervals makes a permanent condition look occasional.
            "time_fraction_by_concurrency": {
                str(k): round(v / total_seconds, 4) for k, v in sorted(seconds.items())
            },
            "concurrent_max": maximum,
            "interval_count_p50_MISLEADING_DENOMINATOR": p50,
            "interval_count_p90_MISLEADING_DENOMINATOR": p90,
            "interval_distribution": {str(k): v for k, v in ordered},
        }
    return out


# ---------------------------------------------------------------------------
# 2. residual correlation -- settlement facts, no replay
# ---------------------------------------------------------------------------

def residual_series(
    markets: Mapping[str, Any],
    resolutions: Mapping[str, Any],
    mids: Mapping[str, float] | None = None,
) -> tuple[dict[str, list[tuple[int, str, float]]], dict[str, int]]:
    """(coin -> [(window_start, day, residual)]), plus the excluded census.

    `residual` is the settlement outcome minus the market's own prior.  With no
    mid it is `outcome - 0.5`, a demeaning that assumes nothing; with the mid at
    t0 it is `outcome - mid(t0)`, the market's forecast error.
    """
    series: dict[str, list[tuple[int, str, float]]] = collections.defaultdict(list)
    excluded: collections.Counter[str] = collections.Counter()
    for slug, info in markets.items():
        resolution = resolutions.get(slug)
        if resolution is None:
            excluded["unresolved"] += 1
            continue
        outcome = 1.0 if resolution.winner_up else 0.0
        if mids is None:
            prior = 0.5
        else:
            prior = mids.get(slug, float("nan"))
            if not (prior == prior):
                excluded["no_mid_on_tape"] += 1
                continue
        series[info.coin].append(
            (info.window_start_s, _day_of(info.window_start_s), outcome - prior)
        )
    for coin in series:
        series[coin].sort()
    return series, dict(excluded)


def lagged_pairs(
    series: Sequence[tuple[int, str, float]], lag: int
) -> list[Pair]:
    """Pairs separated by EXACTLY `lag` windows on the 300 s grid.

    Holes in the grid are skipped rather than bridged: pairing across a missing
    window would silently relabel a 10-minute separation as 5.
    """
    index = {start: (day, value) for start, day, value in series}
    out: list[Pair] = []
    step = int(WINDOW_S) * lag
    for start, day, value in series:
        other = index.get(start + step)
        if other is None:
            continue
        out.append(Pair(day, value, other[1]))
    return out


# ---------------------------------------------------------------------------
# 3. inventory correlation -- the STANDARD two-sided replay
# ---------------------------------------------------------------------------

def replay_window(task: tuple[str, str, str, str, list]) -> dict[str, Any] | None:
    slug, path, up, down, gaps = task
    result = iw.simulate_window(Path(path), up, down, gaps)
    if result is None:
        return None
    start = int(slug.rsplit("-", 1)[1])
    net_at_open = result.net[0] if result.net else 0.0
    return {
        "slug": slug,
        "coin": result.coin,
        "window_start": start,
        "day": _day_of(start),
        "terminal_net": result.terminal_net,
        "terminal_mid": result.terminal_mid,
        "terminal_cash_at_risk": result.terminal_cash_at_risk,
        "net_at_open": net_at_open,
        "n_fills": result.n_fills_buy + result.n_fills_sell,
    }


def select_runs(per_coin_day: int) -> tuple[list[tuple], dict[str, int]]:
    """CONTIGUOUS runs per (coin, day), so lagged pairs exist and days cluster.

    `inventory_walk.select` takes the earliest N slugs per coin, which gives one
    block on one day and no day clusters at all.  Correlation needs adjacency
    AND spread, so this selects a contiguous run inside each (coin, day).
    """
    paths = fi._archive_paths()
    tokens = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    covered = fi.covered_slugs(fi.ERA)
    excluded: collections.Counter[str] = collections.Counter()

    by_coin_day: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for slug in sorted(covered):
        coin = slug.split("-")[0]
        if slug not in paths:
            excluded["no_archive"] += 1
            continue
        if slug not in tokens:
            excluded["no_token_map"] += 1
            continue
        start = int(slug.rsplit("-", 1)[1])
        by_coin_day[(coin, _day_of(start))].append(slug)

    tasks: list[tuple] = []
    for (coin, day), slugs in sorted(by_coin_day.items()):
        slugs.sort(key=lambda s: int(s.rsplit("-", 1)[1]))
        take = slugs[:per_coin_day]
        excluded["beyond_per_coin_day_cap"] += len(slugs) - len(take)
        for slug in take:
            up, down = tokens[slug]
            tasks.append((slug, str(paths[slug]), up, down, gaps.get(slug, [])))
    return tasks, dict(excluded)


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def _series_pairs(
    rows: Sequence[Mapping[str, Any]], field_x: str, field_y: str, lag: int
) -> list[Pair]:
    index = {int(row["window_start"]): row for row in rows}
    step = int(WINDOW_S) * lag
    out: list[Pair] = []
    for row in rows:
        other = index.get(int(row["window_start"]) + step)
        if other is None:
            continue
        out.append(Pair(str(row["day"]), float(row[field_x]), float(other[field_y])))
    return out


def run(per_coin_day: int, workers: int = 14) -> dict[str, Any]:
    from concurrent.futures import ProcessPoolExecutor

    from live.pm_research import tier1_pipeline as tp

    fi.assert_days_current()
    markets, resolutions, _ = tp.load_market_metadata()

    report: dict[str, Any] = {
        "probe": "cross_window_correlation_v1",
        "claim_status": "MEASUREMENT_ONLY_NO_DECISION_RULE",
        "consumed_by": "DE-Allocator (SHARED_RISK coupling edge); DA falsifier #2",
        "lags_windows": list(LAGS),
        "window_seconds": WINDOW_S,
        # R-6: every receipt states which SP set it ran under, and only the
        # parameters that actually BIND it -- a decorative parameter dump would
        # imply dependencies this measurement does not have.
        "sp_params": {
            "set": "SP_PLANE_PLAN.md section 5, operative (user-ratified 2026-08-23)",
            "binding_on_this_measurement": {
                "quote_size_pin_shares": iw.QUOTE_SIZE,
                "policy": "JOIN_BBO two-sided",
                "state_lag_s": fd.STATE_LAG_S,
            },
            "not_binding": [
                "capital_budget", "kappa_usd", "ScenarioLossLimit",
                "gamma_ladder", "refuse_k",
            ],
            "invalidation_if_changed": {
                "quote_size_pin": (
                    "CLASS D under R-20 (was CLASS B). The `inventory` block "
                    "is conditioned on 5 shares "
                    "per side and does NOT carry over to another pin -- fills, "
                    "and therefore every `net` series, are a function of it. "
                    "The `concurrency` and `residual_outcome` blocks are NOT "
                    "conditioned on it and survive a change unchanged."
                ),
            },
        },
    }

    # --- 1. concurrency ---------------------------------------------------
    report["concurrency"] = {
        "definition": "same-coin markets whose [discovery, window_end] intervals overlap",
        "caveat": "LOWER BOUND -- venue creation precedes our discovery",
        "read_this_one": "time_fraction_by_concurrency",
        "per_coin": concurrency(markets),
    }

    # --- 2. residual, outcome-only, FULL population ------------------------
    series, excluded = residual_series(markets, resolutions)
    outcome_block: dict[str, Any] = {
        "estimand": "corr(outcome_k - 0.5, outcome_{k+lag} - 0.5)",
        "population": "every resolved window, ALL days, era-independent",
        "retained_windows": {c: len(v) for c, v in sorted(series.items())},
        "excluded": excluded,
        "per_coin": {},
    }
    for coin, rows in sorted(series.items()):
        outcome_block["per_coin"][coin] = {
            f"lag{lag}": correlate(lagged_pairs(rows, lag)) for lag in LAGS
        }
    pooled: dict[str, Any] = {}
    for lag in LAGS:
        every: list[Pair] = []
        for rows in series.values():
            every.extend(lagged_pairs(rows, lag))
        pooled[f"lag{lag}"] = correlate(every)
    outcome_block["pooled_across_coins"] = pooled
    report["residual_outcome"] = outcome_block

    # --- 3. inventory, STANDARD replay, era-restricted ---------------------
    tasks, replay_excluded = select_runs(per_coin_day)
    rows: list[dict[str, Any]] = []
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(replay_window, tasks, chunksize=2):
            if result is None:
                failed += 1
            else:
                rows.append(result)
    replay_excluded["replay_returned_none"] = failed

    # Join settlement onto the replay so the JOINT-LOSS object exists.  This is
    # the quantity the SHARED_RISK edge is actually about: not whether two
    # positions are similar, but whether they lose together.
    unsettled = 0
    for row in rows:
        resolution = resolutions.get(row["slug"])
        if resolution is None:
            row["settlement_residual"] = None
            row["settlement_pnl"] = None
            unsettled += 1
            continue
        outcome = 1.0 if resolution.winner_up else 0.0
        residual = outcome - row["terminal_mid"]
        row["settlement_residual"] = residual
        row["settlement_pnl"] = row["terminal_net"] * residual
    replay_excluded["replayed_but_unresolved"] = unsettled

    # `simulate_window` initialises its mark to 0.5 and only updates it when a
    # trade arrives with live book state, so a window that never traded returns
    # terminal_mid == 0.5 EXACTLY -- a default, not an observation.  A real mid
    # can also be exactly 0.50 (bid 0.49 / ask 0.51), so the two are not
    # separable from the returned result; the settlement block is therefore
    # reported on BOTH arms rather than silently mixing them.
    for row in rows:
        row["mark_is_default_or_exactly_half"] = row["terminal_mid"] == 0.5
    replay_excluded["mark_exactly_half_ambiguous"] = sum(
        1 for row in rows if row["mark_is_default_or_exactly_half"]
    )

    by_coin: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_coin[row["coin"]].append(row)
    for coin in by_coin:
        by_coin[coin].sort(key=lambda r: r["window_start"])

    inventory: dict[str, Any] = {
        "policy": "JOIN_BBO two-sided, inventory_walk.simulate_window UNMODIFIED",
        "quote_size_shares": iw.QUOTE_SIZE,
        "era": fi.ERA,
        "exposure_convention": (
            "the standard replay exposes t in [-60, +300] s relative to window "
            "start, so it can observe only 60 s of the concurrency that section "
            "1 measures structurally"
        ),
        "population": "contiguous run per (coin, day) inside the era",
        "retained_windows": {c: len(v) for c, v in sorted(by_coin.items())},
        "excluded": replay_excluded,
        "estimands": {
            "terminal_net_lagL": "corr of end-of-window position, L windows apart",
            "cash_at_risk_lagL": "corr of side-aware worst case, L windows apart",
            "simultaneous_terminal_net_vs_next_open": (
                "the only GENUINELY SIMULTANEOUS pair the standard replay can "
                "see: position at window k's settlement vs position window k+1 "
                "has already accumulated in its 60 s pre-start"
            ),
            "settlement_residual_lagL": "corr of (outcome - terminal mark)",
            "JOINT_LOSS_settlement_pnl_lagL": (
                "corr of net x (outcome - terminal mark) -- whether two held "
                "positions LOSE TOGETHER, which is what SHARED_RISK asks"
            ),
        },
        "per_coin": {},
    }
    for coin, coin_rows in sorted(by_coin.items()):
        entry: dict[str, Any] = {}
        for lag in LAGS:
            entry[f"terminal_net_lag{lag}"] = correlate(
                _series_pairs(coin_rows, "terminal_net", "terminal_net", lag)
            )
            entry[f"cash_at_risk_lag{lag}"] = correlate(
                _series_pairs(coin_rows, "terminal_cash_at_risk",
                              "terminal_cash_at_risk", lag)
            )
        entry["simultaneous_terminal_net_vs_next_open"] = correlate(
            _series_pairs(coin_rows, "terminal_net", "net_at_open", 1)
        )
        settled = [r for r in coin_rows if r.get("settlement_pnl") is not None]
        observed = [r for r in settled if not r["mark_is_default_or_exactly_half"]]
        entry["settlement_rows_all"] = len(settled)
        entry["settlement_rows_observed_mark"] = len(observed)
        entry["settlement_rows_excluded_ambiguous_mark"] = len(settled) - len(observed)
        for arm, source in (("ALL_ROWS", settled), ("OBSERVED_MARK", observed)):
            for lag in (1, 2, 3):
                entry[f"settlement_residual_lag{lag}_{arm}"] = correlate(
                    _series_pairs(source, "settlement_residual",
                                  "settlement_residual", lag)
                )
                entry[f"JOINT_LOSS_settlement_pnl_lag{lag}_{arm}"] = correlate(
                    _series_pairs(source, "settlement_pnl", "settlement_pnl", lag)
                )
        inventory["per_coin"][coin] = entry
    report["inventory"] = inventory

    report["provenance"] = fi.provenance(t[0] for t in tasks)
    return report


# ---------------------------------------------------------------------------

def selftest() -> int:
    checks = 0

    def ok(label: str, condition: bool) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(label)
        print(f"  PASS  {label}")

    ok("pearson recovers a perfect line",
       abs(pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-12)
    ok("pearson recovers perfect anticorrelation",
       abs(pearson([1, 2, 3, 4], [-1, -2, -3, -4]) + 1.0) < 1e-12)
    ok("a constant series has no correlation, not a crash",
       pearson([1, 1, 1, 1], [1, 2, 3, 4]) is None)

    series = [(0, "d1", 1.0), (300, "d1", 0.0), (600, "d1", 1.0), (1200, "d1", 0.0)]
    pairs = lagged_pairs(series, 1)
    ok("lag-1 pairs use the 300 s grid and SKIP the hole at 900",
       len(pairs) == 2 and {(p.x, p.y) for p in pairs} == {(1.0, 0.0), (0.0, 1.0)})
    ok("lag-2 pairs step exactly two windows: 0->600 and 600->1200",
       [(p.x, p.y) for p in lagged_pairs(series, 2)] == [(1.0, 1.0), (1.0, 0.0)])

    rng = random.Random(1)
    same = [Pair("d%d" % (i % 4), v := rng.gauss(0, 1), v) for i in range(200)]
    result = correlate(same, n_boot=300)
    ok("a perfectly correlated set reports r=1", result["r"] == 1.0)
    ok("both intervals are reported, and named for what they ignore",
       "ci95_day_clustered" in result
       and "ci95_pair_level_IGNORES_CLUSTERING" in result)
    ok("day clusters are counted, not assumed", result["n_day_clusters"] == 4)

    independent = [Pair("d%d" % (i % 4), rng.gauss(0, 1), rng.gauss(0, 1))
                   for i in range(400)]
    noise = correlate(independent, n_boot=300)
    ok("independent data gives an interval spanning zero",
       noise["ci95_day_clustered"][0] < 0 < noise["ci95_day_clustered"][1])

    class _M:
        def __init__(self, coin, start, end, known):
            self.coin, self.window_start_s = coin, start
            self.window_end_s, self.market_known_ns = end, known
    grid = {
        "a": _M("btc", 0, 300, 0),
        "b": _M("btc", 300, 600, 100 * 10**9),
    }
    conc = concurrency(grid)
    ok("overlapping same-coin markets are counted as concurrent",
       conc["btc"]["concurrent_max"] == 2)
    ok("concurrency is reported TIME-weighted, not only as an interval count",
       "time_fraction_by_concurrency" in conc["btc"])

    print(f"\n{checks} checks passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("run",))
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--per-coin-day", type=int, default=40)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if args.command != "run":
        parser.error("nothing to do: pass --selftest or `run`")

    report = run(args.per_coin_day, workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    slim = {k: v for k, v in report.items() if k not in ("concurrency",)}
    print(json.dumps(slim, indent=2, sort_keys=True)[:4000])
    print(f"\nreceipt: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
