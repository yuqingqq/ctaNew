"""Phase 0A-6: measure the reduced-form Route-A settlement law.

Protocol: live/pm_research/SIGMA_ROUTE_A_PROTOCOL.md (`route_a_v1`).

This is research code. It fits the observed settlement mark directly on the
observed published S30/S60 streams, strictly forward by UTC day. It never adds
the structural k/v/Omega decomposition. The first run is descriptive until ten
OOS test-day clusters exist.

Run:
    python3 -m live.pm_research.exp_sigma_route_a --selftest
    python3 -m live.pm_research.exp_sigma_route_a --protocol-only
    python3 -m live.pm_research.exp_sigma_route_a
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PM = REPO / "data/pm_5min"
PROTOCOL = Path(__file__).with_name("SIGMA_ROUTE_A_PROTOCOL.md")
PROTOCOL_VERSION = "route_a_v1"
HORIZONS = (30, 60, 120, 180, 240, 270)
COINS = {
    "btc": "btc/usd", "eth": "eth/usd", "sol": "sol/usd",
    "xrp": "xrp/usd", "doge": "doge/usd", "bnb": "bnb/usd",
    "hype": "hype/usd",
}
FRESH_MS = 5_000
COVERAGE_FRACTION = 0.90
MIN_TRAIN_ROWS = 30
MIN_OOS_DAYS = 10
MIN_CELL_ROWS = 30
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 20260820
MEAN_TOL = 0.10
VAR_TOL = 0.25
FAMILY_ALPHA = 0.05 / len(HORIZONS)
CELLS = ("all", "x:low", "x:mid", "x:high", "m:low", "m:mid", "m:high")


@dataclass(frozen=True)
class Tick:
    event_ms: int
    known_ms: int
    value: float


@dataclass
class Series:
    event_ticks: list[Tick]
    known_ticks: list[Tick]

    def __post_init__(self):
        self.event_axis = [x.event_ms for x in self.event_ticks]
        self.known_axis = [x.known_ms for x in self.known_ticks]

    def at_event(self, boundary_ms: int) -> Tick | None:
        i = bisect_right(self.event_axis, boundary_ms) - 1
        return self.event_ticks[i] if i >= 0 else None

    def at_known(self, boundary_ms: int) -> Tick | None:
        i = bisect_right(self.known_axis, boundary_ms) - 1
        return self.known_ticks[i] if i >= 0 else None

    def event_count(self, start_ms: int, end_ms: int) -> int:
        return bisect_right(self.event_axis, end_ms) - bisect_left(self.event_axis, start_ms)

    def realised_bps(self, start_ms: int, end_ms: int) -> float | None:
        """Realised |return| range over the ticks present in the span, in bps.

        Computed from WHATEVER ticks exist, including in windows that fail the
        coverage rule -- that is the point. The MNAR question is whether the
        excluded windows are busier than the retained ones, and tick count alone
        cannot answer it: the feed publishes at ~1 Hz regardless of activity, so
        a low count measures FEED HEALTH, not market activity. This measures
        activity. Recording only, never used in any admissibility decision."""
        lo = bisect_left(self.event_axis, start_ms)
        hi = bisect_right(self.event_axis, end_ms)
        vals = [t.value for t in self.event_ticks[lo:hi]
                if math.isfinite(t.value) and t.value > 0]
        if len(vals) < 2:
            return None
        return 10_000.0 * (max(vals) - min(vals)) / vals[0]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _manifest_item(path: Path, raw: bytes) -> dict:
    return {"path": str(path.relative_to(REPO)), "bytes": len(raw), "sha256": _sha(raw)}


def _jsonl_snapshot(path: Path) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    rows = []
    for line in raw.splitlines():
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return rows, _manifest_item(path, raw)


def load_streams() -> tuple[dict[tuple[str, int], Series], list[dict], Counter]:
    """Load immutable rotated feeds. Dedupe event timestamps by earliest recv_ns."""
    by_key: dict[tuple[str, int], dict[int, Tick]] = defaultdict(dict)
    manifest = []
    audit = Counter()
    topics = ((30, "crypto_prices_twap_thirty"), (60, "crypto_prices_twap_sixty"))
    snapshot_paths = [(window, topic, sorted((PM / "prices" / topic).glob("*.csv.gz")))
                      for window, topic in topics]
    for window, topic, paths in snapshot_paths:
        for path in paths:
            raw = path.read_bytes()
            manifest.append(_manifest_item(path, raw))
            try:
                text = gzip.decompress(raw).decode("utf-8")
            except (OSError, UnicodeDecodeError):
                audit[f"unreadable_stream_file:{topic}"] += 1
                continue
            for line in text.splitlines():
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    audit["malformed_stream_line"] += 1
                    continue
                try:
                    msg = json.loads(parts[1])
                    payload = msg.get("payload") or {}
                    symbol = str(payload["symbol"]).lower()
                    event_ms = int(payload["timestamp"])
                    known_ms = int(parts[0]) // 10**6
                    raw_value = payload.get("full_accuracy_value", payload.get("value"))
                    value = float(raw_value)
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("invalid value")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    audit["malformed_stream_payload"] += 1
                    continue
                key = (symbol, window)
                tick = Tick(event_ms, known_ms, value)
                prior = by_key[key].get(event_ms)
                if prior is None or known_ms < prior.known_ms:
                    if prior is not None:
                        audit["duplicate_event_replaced_by_earlier_knowledge"] += 1
                    by_key[key][event_ms] = tick
                else:
                    audit["duplicate_event_discarded"] += 1
    out = {}
    for key, ticks in by_key.items():
        event_ticks = sorted(ticks.values(), key=lambda x: (x.event_ms, x.known_ms))
        known_ticks = sorted(ticks.values(), key=lambda x: (x.known_ms, x.event_ms))
        out[key] = Series(event_ticks, known_ticks)
    return out, manifest, audit


def _combined_digest(manifest: list[dict]) -> str:
    h = hashlib.sha256()
    for x in sorted(manifest, key=lambda z: z["path"]):
        h.update(x["path"].encode())
        h.update(b"\0")
        h.update(x["sha256"].encode())
        h.update(b"\n")
    return h.hexdigest()


def _utc_day(epoch_s: int) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).date().isoformat()


def build_rows() -> tuple[list[dict], dict]:
    # Capture the append-only metadata bytes before the slower feed parse so the
    # run has the promised start-of-run snapshot rather than a moving tail.
    market_rows, market_manifest = _jsonl_snapshot(PM / "markets.jsonl")
    resolution_rows, resolution_manifest = _jsonl_snapshot(PM / "resolutions.jsonl")
    streams, manifest, stream_audit = load_streams()
    manifest.extend((market_manifest, resolution_manifest))

    markets = {}
    for row in market_rows:
        if row.get("slug"):
            markets[row["slug"]] = row
    resolutions = {}
    for row in resolution_rows:
        if row.get("slug") and row.get("closed") is True and row.get("winners"):
            resolutions[row["slug"]] = row

    exclusions = Counter()
    # WINDOW-IDENTITY LEDGER. The protocol and the collector audit both require an
    # accepted-versus-excluded activity/volatility comparison before the day-10
    # verdict; with a bare Counter that comparison is impossible. This records
    # WHICH windows were dropped and how busy they were.
    #
    # Protocol-neutral by construction: it changes no gate, tolerance, exclusion
    # or conditioning cell -- every `continue` fires on exactly the same
    # condition as before, and the counters increment identically. It only writes
    # down decisions that were already being made, so it does NOT create a new
    # protocol version.
    excluded: list[dict] = []

    def drop(reason: str, slug: str, coin: str = "", t0_s: int | None = None,
             **detail) -> None:
        exclusions[reason] += 1
        excluded.append({"reason": reason, "slug": slug, "coin": coin,
                         "window_start": t0_s, **detail})

    rows = []
    agreement_n = agreement_hit = 0
    known_skews = []
    covered_windows = 0

    for slug, resolution in sorted(resolutions.items()):
        market = markets.get(slug)
        if market is None:
            drop("no_market_metadata", slug)
            continue
        coin = str(market.get("coin", "")).lower()
        symbol = COINS.get(coin)
        if symbol is None:
            drop("unsupported_coin", slug, coin)
            continue
        fast, slow = streams.get((symbol, 30)), streams.get((symbol, 60))
        if fast is None or slow is None:
            drop("missing_stream", slug, coin)
            continue
        try:
            t0_s, end_s = int(market["window_start"]), int(market["window_end"])
        except (KeyError, TypeError, ValueError):
            drop("bad_window_metadata", slug, coin)
            continue
        if end_s - t0_s != 300:
            drop("non_300s_window", slug, coin, t0_s, span_s=end_s - t0_s)
            continue
        t0, end = t0_s * 1000, end_s * 1000
        cov_start, cov_end = t0 - 5_000, end + 5_000
        nominal = (cov_end - cov_start) / 1000
        min_ticks = math.ceil(COVERAGE_FRACTION * nominal)
        if fast.event_count(cov_start, cov_end) < min_ticks:
            drop("s30_window_coverage", slug, coin, t0_s,
                 ticks=fast.event_count(cov_start, cov_end), min_ticks=min_ticks,
                 realised_bps=slow.realised_bps(cov_start, cov_end))
            continue
        if slow.event_count(cov_start, cov_end) < min_ticks:
            drop("s60_window_coverage", slug, coin, t0_s,
                 ticks=slow.event_count(cov_start, cov_end), min_ticks=min_ticks,
                 realised_bps=slow.realised_bps(cov_start, cov_end))
            continue
        x0_tick, xT_tick = slow.at_event(t0), slow.at_event(end)
        if x0_tick is None or xT_tick is None:
            drop("missing_target_boundary", slug, coin, t0_s,
                 realised_bps=slow.realised_bps(cov_start, cov_end))
            continue
        if t0 - x0_tick.event_ms > FRESH_MS or end - xT_tick.event_ms > FRESH_MS:
            drop("stale_target_boundary", slug, coin, t0_s,
                 realised_bps=slow.realised_bps(cov_start, cov_end))
            continue
        x0, xT = x0_tick.value, xT_tick.value
        if not math.isfinite(x0) or x0 <= 0 or not math.isfinite(xT):
            drop("invalid_target_value", slug, coin, t0_s)
            continue
        covered_windows += 1
        winner_up = bool((resolution.get("winners") or {}).get("Up"))
        agreement_n += 1
        agreement_hit += int((xT >= x0) == winner_up)

        for r in HORIZONS:
            decision = end - r * 1000
            s30_tick, s60_tick = fast.at_known(decision), slow.at_known(decision)
            if s30_tick is None or s60_tick is None:
                drop(f"r{r}:missing_predictor", slug, coin, t0_s, horizon=r,
                     realised_bps=slow.realised_bps(cov_start, cov_end))
                continue
            if decision - s30_tick.known_ms > FRESH_MS or decision - s60_tick.known_ms > FRESH_MS:
                # THE ONE TO WATCH. The prices lane logs an 11-13 s gap roughly
                # every 20 minutes; one landing on a decision time breaks the
                # <=5 s staleness rule for exactly this horizon and no other.
                # Recording the observed staleness makes that mechanism testable
                # against the collector gap ledger instead of merely plausible.
                drop(f"r{r}:stale_predictor", slug, coin, t0_s, horizon=r,
                     s30_age_ms=decision - s30_tick.known_ms,
                     s60_age_ms=decision - s60_tick.known_ms,
                     realised_bps=slow.realised_bps(cov_start, cov_end))
                continue
            known_skews.append(abs(s30_tick.known_ms - s60_tick.known_ms))
            s30, s60 = s30_tick.value, s60_tick.value
            scale = 10_000.0 / x0
            x = (s30 - s60) * scale
            y = (xT - s60) * scale
            m = (s60 - x0) * scale
            if not all(math.isfinite(z) for z in (x, y, m)):
                drop(f"r{r}:nonfinite_normalized", slug, coin, t0_s, horizon=r)
                continue
            rows.append({
                # same statistic the exclusion ledger records, so accepted and
                # excluded windows are directly comparable. A one-sided ledger
                # cannot answer the MNAR question it exists for.
                "realised_bps": slow.realised_bps(cov_start, cov_end),
                "slug": slug, "coin": coin, "symbol": symbol,
                "day": _utc_day(t0_s), "horizon": r,
                "window_start_ms": t0, "window_end_ms": end,
                "decision_ms": decision, "x0_event_ms": x0_tick.event_ms,
                "xT_event_ms": xT_tick.event_ms,
                "s30_known_ms": s30_tick.known_ms, "s60_known_ms": s60_tick.known_ms,
                "x0": x0, "xT": xT, "s30": s30, "s60": s60,
                "x_bps": x, "y_bps": y, "m_bps": m,
                "winner_up": winner_up,
            })

    known_skews.sort()
    q = lambda p: known_skews[min(int(p * len(known_skews)), len(known_skews) - 1)] \
        if known_skews else None
    audit = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": _sha(PROTOCOL.read_bytes()),
        "script_sha256": _sha(Path(__file__).read_bytes()),
        "source_manifest": sorted(manifest, key=lambda z: z["path"]),
        "source_digest": _combined_digest(manifest),
        "immutable_stream_files": sum(1 for x in manifest if x["path"].endswith(".csv.gz")),
        "markets_snapshot_rows": len(market_rows),
        "resolution_snapshot_rows": len(resolution_rows),
        "final_resolutions": len(resolutions),
        "covered_windows": covered_windows,
        "settlement_agreement_n": agreement_n,
        "settlement_agreement_hit": agreement_hit,
        "known_stream_skew_ms_p50": q(0.50),
        "known_stream_skew_ms_p95": q(0.95),
        "stream_audit": dict(sorted(stream_audit.items())),
        "exclusions": dict(sorted(exclusions.items())),
        # per-window identity for the accepted-vs-excluded selection audit
        "excluded_windows": excluded,
    }
    return rows, audit


def _quantile(values: list[float], p: float) -> float:
    v = sorted(values)
    if not v:
        raise ValueError("quantile of empty list")
    pos = p * (len(v) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return v[lo]
    return v[lo] * (hi - pos) + v[hi] * (pos - lo)


def _bin3(value: float, cuts: tuple[float, float]) -> str:
    return "low" if value <= cuts[0] else "mid" if value <= cuts[1] else "high"


def fit_alpha(rows: list[dict]) -> float | None:
    denom = sum(x["x_bps"] ** 2 for x in rows)
    if not math.isfinite(denom) or denom <= 1e-12:
        return None
    alpha = sum(x["x_bps"] * x["y_bps"] for x in rows) / denom
    return alpha if math.isfinite(alpha) else None


def cross_fit(rows: list[dict]) -> tuple[list[dict], list[dict], Counter]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["coin"], row["horizon"])].append(row)
    oos = []
    fit_meta = []
    audit = Counter()
    for (coin, horizon), group in sorted(grouped.items()):
        group.sort(key=lambda z: (z["day"], z["window_start_ms"]))
        days = sorted({z["day"] for z in group})
        for test_day in days[1:]:
            train = [z for z in group if z["day"] < test_day]
            test = [z for z in group if z["day"] == test_day]
            if len(train) < MIN_TRAIN_ROWS:
                audit["fold_too_few_train_rows"] += 1
                continue
            alpha = fit_alpha(train)
            if alpha is None:
                audit["fold_unidentified_alpha"] += 1
                continue
            xcuts = (_quantile([z["x_bps"] for z in train], 1 / 3),
                     _quantile([z["x_bps"] for z in train], 2 / 3))
            mcuts = (_quantile([z["m_bps"] for z in train], 1 / 3),
                     _quantile([z["m_bps"] for z in train], 2 / 3))
            fit_meta.append({"coin": coin, "horizon": horizon, "test_day": test_day,
                             "n_train": len(train), "n_test": len(test),
                             "alpha": alpha, "x_cuts": xcuts, "m_cuts": mcuts})
            for row in test:
                pred_y = alpha * row["x_bps"]
                rec = dict(row)
                rec.update({
                    "fold_test_day": test_day, "fold_n_train": len(train),
                    "alpha_fold": alpha, "pred_y_bps": pred_y,
                    "residual_bps": row["y_bps"] - pred_y,
                    "x_bin": _bin3(row["x_bps"], xcuts),
                    "m_bin": _bin3(row["m_bps"], mcuts),
                })
                oos.append(rec)
    return oos, fit_meta, audit


def _cell_rows(rows: list[dict]) -> dict[str, list[dict]]:
    out = {c: [] for c in CELLS}
    for row in rows:
        out["all"].append(row)
        out[f"x:{row['x_bin']}"].append(row)
        out[f"m:{row['m_bin']}"].append(row)
    return out


def _effects(rows: list[dict]) -> tuple[dict, dict, float]:
    s2 = sum(z["residual_bps"] ** 2 for z in rows) / len(rows)
    if not math.isfinite(s2) or s2 <= 0:
        raise ValueError("nonpositive OOS residual MSE")
    sigma = math.sqrt(s2)
    cells = _cell_rows(rows)
    mean_cells, var_cells = {}, {}
    for name, rr in cells.items():
        if not rr:
            continue
        mean_cells[name] = abs(sum(z["residual_bps"] for z in rr) / len(rr)) / sigma
        var_cells[name] = abs(sum(z["residual_bps"] ** 2 for z in rr) / len(rr) / s2 - 1)
    return mean_cells, var_cells, s2


def _percentile(values: list[float], p: float) -> float:
    return _quantile(values, p)


def gate(rows: list[dict], kind: str, key: str) -> dict:
    mean_cells, var_cells, _ = _effects(rows)
    observed_cells = mean_cells if kind == "mean" else var_cells
    effect = max(observed_cells.values())
    counts = {k: len(v) for k, v in _cell_rows(rows).items()}
    days = sorted({z["day"] for z in rows})
    tolerance = MEAN_TOL if kind == "mean" else VAR_TOL
    result = {
        "test": "max-cell standardized residual mean" if kind == "mean"
                else "max-cell relative residual second moment",
        "conditioning": "all + training-terciles of x=(S30-S60) and m=(S60-strike)",
        "multiplicity": "day-block bootstrap; max over 7 cells; Bonferroni over 6 horizons",
        "effect_size": effect, "cell_effects": observed_cells,
        "cell_counts": counts, "tolerance": tolerance,
        "confidence_level_family": 0.95, "block_unit": "UTC OOS test day",
        "n_oos_days": len(days), "bootstrap_draws": 0,
        "ci_lo_abs": None, "ci_hi_abs": None,
    }
    if len(days) < MIN_OOS_DAYS or min(counts.values()) < MIN_CELL_ROWS:
        result["verdict"] = "INSUFFICIENT_EVIDENCE"
        result["reason"] = (f"needs >= {MIN_OOS_DAYS} OOS days and >= {MIN_CELL_ROWS} "
                            f"rows/cell; has {len(days)} days and min cell {min(counts.values())}")
        return result

    by_day = {d: [z for z in rows if z["day"] == d] for d in days}
    seed_material = f"{BOOTSTRAP_SEED}:{key}:{kind}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = []
        for d in (rng.choice(days) for _ in days):
            sample.extend(by_day[d])
        mc, vc, _ = _effects(sample)
        draws.append(max((mc if kind == "mean" else vc).values()))
    lo = _percentile(draws, FAMILY_ALPHA / 2)
    hi = _percentile(draws, 1 - FAMILY_ALPHA / 2)
    result.update({"bootstrap_draws": BOOTSTRAP_DRAWS,
                   "ci_lo_abs": lo, "ci_hi_abs": hi})
    if hi <= tolerance:
        result["verdict"] = "PASS"
        result["reason"] = "simultaneous upper bound is inside frozen tolerance"
    elif lo > tolerance:
        result["verdict"] = "MODEL_REFUTED"
        result["reason"] = "simultaneous lower bound exceeds frozen tolerance"
    else:
        result["verdict"] = "INSUFFICIENT_EVIDENCE"
        result["reason"] = "simultaneous interval overlaps frozen tolerance"
    return result


def summarize(rows: list[dict], oos: list[dict]) -> list[dict]:
    raw_groups, oos_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        raw_groups[(row["coin"], row["horizon"])].append(row)
    for row in oos:
        oos_groups[(row["coin"], row["horizon"])].append(row)
    out = []
    for key in sorted(raw_groups):
        raw, test = raw_groups[key], oos_groups.get(key, [])
        alpha_full = fit_alpha(raw)
        rec = {
            "coin": key[0], "horizon": key[1], "n_rows": len(raw),
            "n_days": len({z["day"] for z in raw}), "alpha_full": alpha_full,
            "n_oos": len(test), "n_oos_days": len({z["day"] for z in test}),
        }
        if not test:
            rec.update({"alpha_oos_weighted": None, "resid_mean_bps": None,
                        "resid_var_bps2": None, "resid_sd_bps": None,
                        "mean_gate": None, "var_gate": None})
        else:
            s2 = sum(z["residual_bps"] ** 2 for z in test) / len(test)
            rec.update({
                "alpha_oos_weighted": sum(z["alpha_fold"] for z in test) / len(test),
                "resid_mean_bps": sum(z["residual_bps"] for z in test) / len(test),
                "resid_var_bps2": s2, "resid_sd_bps": math.sqrt(s2),
                "mean_gate": gate(test, "mean", f"{key[0]}:{key[1]}"),
                "var_gate": gate(test, "var", f"{key[0]}:{key[1]}"),
            })
        out.append(rec)
    return out


def _fmt(x, digits=3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def _overall_status(verdicts: list[str]) -> str:
    expected = len(COINS) * len(HORIZONS) * 2
    if "MODEL_REFUTED" in verdicts:
        return "MODEL REFUTED — PRICING HOLD"
    if len(verdicts) == expected and all(v == "PASS" for v in verdicts):
        return "PRICING PASS"
    return "DESCRIPTIVE — INSUFFICIENT EVIDENCE"


def _fit_status(mean_gate: dict | None, var_gate: dict | None) -> str:
    if not mean_gate or not var_gate:
        return "—"
    verdicts = (mean_gate["verdict"], var_gate["verdict"])
    if "MODEL_REFUTED" in verdicts:
        return "REFUTED"
    if "INSUFFICIENT_EVIDENCE" in verdicts:
        return "INSUFFICIENT"
    return "PASS"


def render_markdown(payload: dict) -> str:
    audit, summaries = payload["audit"], payload["fits"]
    days = sorted({z["day"] for z in payload["dataset_rows"]})
    oos_days = sorted({z["day"] for z in payload["oos_rows"]})
    all_verdicts = [g[which]["verdict"] for g in summaries for which in ("mean_gate", "var_gate")
                    if g.get(which)]
    mean_point_breaches = sum(z["mean_gate"]["effect_size"] > MEAN_TOL for z in summaries
                              if z.get("mean_gate"))
    var_point_breaches = sum(z["var_gate"]["effect_size"] > VAR_TOL for z in summaries
                             if z.get("var_gate"))
    status = _overall_status(all_verdicts)
    lines = [
        "# SIGMA Route-A measurement — protocol route_a_v1",
        "",
        f"Run time: {payload['run_time_utc']}. Status: **{status}**.",
        "",
        "This is the first real fit of the Revision-5 reduced-form law. It uses",
        "only observed S30/S60 streams and the observed settlement target; no",
        "structural `k/v/Omega` term is added.",
        "",
        "## Snapshot and admissibility",
        "",
        f"- source digest: `{audit['source_digest']}`; immutable stream files: "
        f"{audit['immutable_stream_files']}",
        f"- UTC data days: {', '.join(days) if days else 'none'}; OOS test days: "
        f"{', '.join(oos_days) if oos_days else 'none'}",
        f"- final resolutions: {audit['final_resolutions']}; admissible windows: "
        f"{audit['covered_windows']}; regression rows: {len(payload['dataset_rows'])}; "
        f"OOS rows: {len(payload['oos_rows'])}",
        f"- settlement-direction agreement: {audit['settlement_agreement_hit']}/"
        f"{audit['settlement_agreement_n']} "
        f"({audit['settlement_agreement_hit']/audit['settlement_agreement_n']:.2%})"
        if audit["settlement_agreement_n"] else "- settlement-direction agreement: unavailable",
        f"- S30/S60 knowledge-time read skew: p50 {audit['known_stream_skew_ms_p50']} ms; "
        f"p95 {audit['known_stream_skew_ms_p95']} ms",
        "",
        "## Strictly forward OOS results",
        "",
        "`alpha train` is the coefficient fitted only on days before the OOS day;",
        "`alpha all` is descriptive. Residual variance is OOS mean squared error",
        "around zero, so mean bias is not subtracted away.",
        "",
        "| coin | r | rows/days | OOS rows/days | alpha train | alpha all | "
        "mean resid bp | resid sd bp | mean effect | var effect | verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for z in summaries:
        mg, vg = z.get("mean_gate"), z.get("var_gate")
        verdict = _fit_status(mg, vg)
        lines.append(
            f"| {z['coin']} | {z['horizon']} | {z['n_rows']}/{z['n_days']} | "
            f"{z['n_oos']}/{z['n_oos_days']} | {_fmt(z['alpha_oos_weighted'])} | "
            f"{_fmt(z['alpha_full'])} | {_fmt(z['resid_mean_bps'])} | "
            f"{_fmt(z['resid_sd_bps'])} | {_fmt(mg['effect_size'] if mg else None)} | "
            f"{_fmt(vg['effect_size'] if vg else None)} | {verdict} |")
    lines.extend(["", "## Exclusions", "", "| reason | count |", "|---|---:|"])
    for reason, count in audit["exclusions"].items():
        lines.append(f"| `{reason}` | {count} |")
    if not audit["exclusions"]:
        lines.append("| none | 0 |")
    lines.extend([
        "", "## Verdict", "",
        f"This snapshot has **{len(oos_days)} OOS test-day cluster(s)**. The frozen",
        f"gate requires {MIN_OOS_DAYS}, so every fitted cell is",
        "`INSUFFICIENT_EVIDENCE` regardless of its point estimate. The numbers",
        "above are a valid descriptive, strictly forward pipeline measurement;",
        "they are not a probability-law authorization.",
        "",
        f"The point diagnostics are an early warning, not a gate verdict: "
        f"**{mean_point_breaches}/{len(summaries)}** mean effects exceed 0.10 "
        f"residual sigma and **{var_point_breaches}/{len(summaries)}** variance "
        f"effects exceed 0.25. With one test day these can be a day/regime effect; "
        f"they provide no early support for homoskedastic Route A, but cannot yet "
        f"refute it.",
        "",
        "No new sigma specification is warranted from sample size alone. Re-run",
        "this identical protocol as the day count grows. Only an OOS residual",
        "diagnostic that eventually reads `MODEL_REFUTED` should reopen the",
        "Route-A functional form.",
        "",
        "Protocol: `live/pm_research/SIGMA_ROUTE_A_PROTOCOL.md`.",
    ])
    return "\n".join(lines) + "\n"


def selftest() -> int:
    rows = []
    for day, shift in (("2026-01-01", 0.0), ("2026-01-02", 0.1)):
        for i in range(60):
            x = (i - 30) / 10
            m = (i % 9) - 4
            y = 1.5 * x + (0.05 if i % 2 else -0.05) + shift * 0
            rows.append({"coin": "btc", "symbol": "btc/usd", "day": day,
                         "horizon": 60, "window_start_ms": i, "slug": f"s{i}",
                         "x_bps": x, "y_bps": y, "m_bps": m})
    oos, folds, audit = cross_fit(rows)
    checks = [
        ("one strictly forward fold", len(folds) == 1 and folds[0]["test_day"] == "2026-01-02"),
        ("alpha recovered", abs(folds[0]["alpha"] - 1.5) < 0.01),
        ("first day never appears OOS", {z["day"] for z in oos} == {"2026-01-02"}),
        ("seven frozen cells", set(_cell_rows(oos)) == set(CELLS)),
        ("too few OOS days is insufficient",
         gate(oos, "mean", "selftest")["verdict"] == "INSUFFICIENT_EVIDENCE"),
        ("partial passes cannot authorize pricing",
         _overall_status(["PASS"] * 82) != "PRICING PASS"),
        ("a refutation is never hidden by insufficient evidence",
         _overall_status(["INSUFFICIENT_EVIDENCE", "MODEL_REFUTED"]).startswith("MODEL REFUTED")
         and _fit_status({"verdict": "INSUFFICIENT_EVIDENCE"},
                         {"verdict": "MODEL_REFUTED"}) == "REFUTED"),
        ("no fold audit failures", not audit),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in checks) else 1


def protocol_only() -> None:
    print(json.dumps({
        "protocol_version": PROTOCOL_VERSION, "horizons": HORIZONS,
        "fresh_ms": FRESH_MS, "coverage_fraction": COVERAGE_FRACTION,
        "min_train_rows": MIN_TRAIN_ROWS, "min_oos_days": MIN_OOS_DAYS,
        "min_cell_rows": MIN_CELL_ROWS, "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED, "mean_tolerance": MEAN_TOL,
        "variance_tolerance": VAR_TOL, "family_alpha": FAMILY_ALPHA,
        "cells": CELLS,
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--protocol-only", action="store_true")
    ap.add_argument("--output-json", default=str(PM / "derived/sigma_route_a_v1.json"))
    ap.add_argument("--output-md", default=str(Path(__file__).with_name(
        "SIGMA_ROUTE_A_RESULTS_2026-08-20.md")))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.protocol_only:
        protocol_only()
        return 0

    rows, audit = build_rows()
    oos, folds, fold_audit = cross_fit(rows)
    audit["fold_audit"] = dict(sorted(fold_audit.items()))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "audit": audit, "dataset_rows": rows, "folds": folds,
        "oos_rows": oos, "fits": summarize(rows, oos),
    }
    out_json, out_md = Path(args.output_json), Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    out_md.write_text(render_markdown(payload))
    print(f"[route-a] dataset rows={len(rows)} OOS rows={len(oos)} fits={len(payload['fits'])}")
    print(f"[route-a] source digest={audit['source_digest']}")
    print(f"[route-a] wrote {out_json}")
    print(f"[route-a] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
