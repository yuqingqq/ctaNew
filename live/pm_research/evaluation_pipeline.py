"""Batch-gated Tier-2 evaluation artifacts for P-2026-003.

This offline pipeline consumes only a validated ``full`` Tier-1 measurement
batch and materializes two model-free/normalization datasets:

* ``markout_events``: one terminal maker-markout observation per parent trade;
* ``calib_panel``: one point-in-time book row per ``(slug, r_s)``.

It deliberately does not fit sigma, recalibrate probabilities, calculate a
confidence interval, or authorize a decision.  Incomplete Tier-1 partitions
and a non-PASS G-FF1 side-convention artifact are refused before derivation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from live.pm_research.coverage_ledger import canonical_json
from live.pm_research.daily_pipeline import validate_existing_partition
from live.pm_research.measurement_batch import (
    batch_plan_dict,
    discover_candidate_days,
    execute_batch,
    load_completed_batch,
    plan_batch,
)
from live.pm_research.tier1_pipeline import (
    COIN_SYMBOL,
    DEFAULT_OUTPUT_ROOT,
    PM,
    load_market_metadata,
    read_partition,
)


PIPELINE_VERSION = "evaluation_pipeline_v1"
DERIVER_VERSION = "evaluation_tier2_v1"
DEFAULT_TIER2_ROOT = DEFAULT_OUTPUT_ROOT.parent / "tier2"
DEFAULT_SIDE_EVIDENCE = PM / "derived" / "gff1_side_v3.json"
R_HORIZONS_S = (270, 240, 180, 120, 60, 30, 10, 5, 2)
ROW_GROUP_SIZE = 65_536


MARKOUT_SCHEMA = pa.schema(
    [
        ("t_known_ns", pa.int64()),
        ("t_known_err_ns", pa.int64()),
        ("t_known_prov", pa.string()),
        ("t_event_ms", pa.int64()),
        ("day", pa.string()),
        ("slug", pa.string()),
        ("coin", pa.string()),
        ("parent_id", pa.string()),
        ("transaction_hashes_json", pa.string()),
        ("trade_known_ns", pa.int64()),
        ("resolution_known_ns", pa.int64()),
        ("window_start_s", pa.int64()),
        ("window_end_s", pa.int64()),
        ("phase", pa.string()),
        ("phase_clock", pa.string()),
        ("token_side", pa.string()),
        ("q_up", pa.int8()),
        ("price_up", pa.float64()),
        ("size", pa.float64()),
        ("winner_up", pa.bool_()),
        ("outcome_up", pa.float64()),
        ("maker_edge_per_share", pa.float64()),
        ("maker_edge_cents", pa.float64()),
        ("maker_gross_cash", pa.float64()),
        ("fee_status", pa.string()),
        ("side_evidence_sha256", pa.string()),
        ("constituent_count", pa.int32()),
        ("clob_gap_count", pa.int32()),
        ("clob_gap_ms", pa.float64()),
        ("clob_gap_causes_json", pa.string()),
        ("clob_slow_consumer_gap", pa.bool_()),
    ]
)


CALIB_SCHEMA = pa.schema(
    [
        ("t_known_ns", pa.int64()),
        ("t_known_err_ns", pa.int64()),
        ("t_known_prov", pa.string()),
        ("t_event_ms", pa.int64()),
        ("day", pa.string()),
        ("slug", pa.string()),
        ("coin", pa.string()),
        ("r_s", pa.int16()),
        ("decision_time_ns", pa.int64()),
        ("winner_up", pa.bool_()),
        ("resolution_known_ns", pa.int64()),
        ("quote_status", pa.string()),
        ("quote_t_event_ms", pa.int64()),
        ("quote_t_known_ns", pa.int64()),
        ("quote_t_known_err_ns", pa.int64()),
        ("quote_staleness_ns", pa.int64()),
        ("bid_up", pa.float64()),
        ("ask_up", pa.float64()),
        ("spread", pa.float64()),
        ("tick_size", pa.float64()),
        ("p_book_raw", pa.float64()),
        ("p_book_clipped", pa.float64()),
        ("clip_floor", pa.float64()),
        ("pair_consistent", pa.bool_()),
        ("route_a_admissible", pa.bool_()),
        ("coverage_hash", pa.string()),
        ("admissibility_rule_id", pa.string()),
        ("admissibility_rule_hash", pa.string()),
        ("model_status", pa.string()),
        ("isotonic_status", pa.string()),
        ("clob_gap_count", pa.int32()),
        ("clob_gap_ms", pa.float64()),
        ("clob_gap_causes_json", pa.string()),
        ("clob_slow_consumer_gap", pa.bool_()),
    ]
)


SCHEMAS = {
    "markout_events": MARKOUT_SCHEMA,
    "calib_panel": CALIB_SCHEMA,
}


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(value: Any) -> str:
    return _sha256_bytes(canonical_json(value))


def _schema_hash(schema: pa.Schema) -> str:
    return _sha256_bytes(schema.serialize().to_pybytes())


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _plain_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a plain int")
    return value


def _unique_coins(coins: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for coin in coins:
        if coin not in COIN_SYMBOL:
            raise ValueError(f"unsupported coin {coin!r}")
        if coin not in result:
            result.append(coin)
    if not result:
        raise ValueError("evaluation requires at least one coin")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SideConventionEvidence:
    artifact_sha256: str
    protocol_version: str
    n_validated_tx: int
    agreement: float
    wilson_lo: float
    wilson_hi: float
    threshold: float
    verdict: str


def load_side_evidence(path: Path = DEFAULT_SIDE_EVIDENCE) -> SideConventionEvidence:
    """Load and fail-close the immutable G-FF1 direction result."""
    payload = json.loads(path.read_text())
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError(f"side evidence has no manifest: {path}")
    interval = payload.get("wilson95")
    if not isinstance(interval, list) or len(interval) != 2:
        raise RuntimeError(f"side evidence has invalid Wilson interval: {path}")
    result = SideConventionEvidence(
        artifact_sha256=_sha256_file(path),
        protocol_version=str(manifest.get("protocol", "")),
        n_validated_tx=_plain_int(payload.get("n_validated_tx"), "n_validated_tx"),
        agreement=_finite_float(payload.get("agreement"), "agreement"),
        wilson_lo=_finite_float(interval[0], "wilson_lo"),
        wilson_hi=_finite_float(interval[1], "wilson_hi"),
        threshold=_finite_float(payload.get("threshold"), "threshold"),
        verdict=str(payload.get("verdict", "")),
    )
    if not result.protocol_version:
        raise RuntimeError("side evidence has no protocol version")
    if result.verdict != "PASS":
        raise RuntimeError(f"side convention is not PASS: {result.verdict}")
    if result.n_validated_tx < 500:
        raise RuntimeError("side convention has fewer than 500 validated transactions")
    if not 0 <= result.agreement <= 1:
        raise RuntimeError("side convention agreement is outside [0,1]")
    if not 0 <= result.wilson_lo <= result.wilson_hi <= 1:
        raise RuntimeError("side convention Wilson interval is invalid")
    if result.wilson_lo < result.threshold:
        raise RuntimeError("side convention Wilson lower bound misses threshold")
    return result


def _window_index(
    windows: Sequence[Mapping[str, Any]], *, day: date, coin: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in windows:
        slug = str(row.get("slug", ""))
        if not slug or slug in result:
            raise RuntimeError(f"duplicate or empty window slug {slug!r}")
        if row.get("coin") != coin or row.get("closed") is not True:
            raise RuntimeError(f"invalid closed-window identity for {slug}")
        start_s = _plain_int(row.get("window_start_s"), "window_start_s")
        selected_day = datetime.fromtimestamp(start_s, timezone.utc).date()
        if selected_day != day:
            raise RuntimeError(f"window {slug} is outside {day}")
        if row.get("winner_up") not in {True, False}:
            raise RuntimeError(f"window {slug} has no binary outcome")
        result[slug] = row
    if len(result) != 288:
        raise RuntimeError(f"expected 288 closed windows, found {len(result)}")
    return result


def _coverage_index(
    coverage: Sequence[Mapping[str, Any]], window_slugs: set[str]
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in coverage:
        slug = str(row.get("slug", ""))
        if not slug or slug in result:
            raise RuntimeError(f"duplicate or empty coverage slug {slug!r}")
        result[slug] = row
    if set(result) != window_slugs:
        raise RuntimeError("coverage/window slug bijection failed")
    return result


def _composed_clock(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int, str]:
    if not rows:
        raise ValueError("cannot compose an empty clock set")
    known_ns = max(_plain_int(row.get("t_known_ns"), "t_known_ns") for row in rows)
    known_hi_ns = max(
        _plain_int(row.get("t_known_ns"), "t_known_ns")
        + _plain_int(row.get("t_known_err_ns"), "t_known_err_ns")
        for row in rows
    )
    provenances = {str(row.get("t_known_prov", "")) for row in rows}
    provenance = "OBSERVED" if provenances == {"OBSERVED"} else "MIXED"
    return known_ns, known_hi_ns - known_ns, provenance


def _knowledge_phase(trade: Mapping[str, Any], window: Mapping[str, Any]) -> str:
    known_ns = _plain_int(trade.get("t_known_ns"), "trade.t_known_ns")
    error_ns = _plain_int(trade.get("t_known_err_ns"), "trade.t_known_err_ns")
    known_hi_ns = known_ns + error_ns
    start_ns = (
        _plain_int(window.get("window_start_s"), "window_start_s")
        * 1_000_000_000
    )
    end_ns = _plain_int(window.get("window_end_s"), "window_end_s") * 1_000_000_000
    if known_hi_ns < start_ns:
        return "PRE_OPEN"
    if known_ns >= start_ns and known_hi_ns <= end_ns:
        return "IN_WINDOW"
    if known_ns > end_ns:
        return "POST_CLOSE"
    return "AMBIGUOUS_BOUNDARY"


def _markout_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    keys = sorted({(str(row["coin"]), str(row["phase"])) for row in rows})
    for coin, phase in keys:
        selected = [
            row for row in rows if row["coin"] == coin and row["phase"] == phase
        ]
        shares = sum(float(row["size"]) for row in selected)
        gross = sum(float(row["maker_gross_cash"]) for row in selected)
        result.append(
            {
                "coin": coin,
                "phase": phase,
                "n_fills": len(selected),
                "shares": shares,
                "per_fill_cents": sum(
                    float(row["maker_edge_cents"]) for row in selected
                )
                / len(selected),
                "share_weighted_cents": gross / shares * 100.0,
                "gross_cash": gross,
                "claim_status": "DESCRIPTIVE_POINT_ESTIMATE",
                "ci": {
                    "reason": "INSUFFICIENT_DAY_CLUSTERS",
                    "since": str(selected[0]["day"]),
                    "cause": None,
                },
            }
        )
    return result


def build_markout_rows(
    *,
    day: date,
    coin: str,
    trades: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    side_evidence: SideConventionEvidence,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive gross TO_RESOLUTION maker markout without any book input."""
    window_by_slug = _window_index(windows, day=day, coin=coin)
    rows: list[dict[str, Any]] = []
    seen_parent_ids: set[str] = set()
    late_winner_prices: list[float] = []
    for trade in trades:
        slug = str(trade.get("slug", ""))
        if slug not in window_by_slug:
            raise RuntimeError(f"trade references unknown window {slug!r}")
        parent_id = str(trade.get("parent_id", ""))
        if not parent_id or parent_id in seen_parent_ids:
            raise RuntimeError(f"duplicate or empty parent trade {parent_id!r}")
        seen_parent_ids.add(parent_id)
        q_up = _plain_int(trade.get("q_up"), "q_up")
        if q_up not in {-1, 1}:
            raise RuntimeError(f"parent trade {parent_id} has invalid q_up={q_up}")
        price_up = _finite_float(trade.get("price_up"), "price_up")
        size = _finite_float(trade.get("size"), "size")
        if not 0 <= price_up <= 1 or size <= 0:
            raise RuntimeError(f"invalid price/size for parent trade {parent_id}")
        window = window_by_slug[slug]
        winner_up = bool(window["winner_up"])
        outcome_up = 1.0 if winner_up else 0.0
        edge = q_up * (price_up - outcome_up)
        known_ns, known_err_ns, provenance = _composed_clock((trade, window))
        event_ms = _plain_int(trade.get("t_event_ms"), "trade.t_event_ms")
        start_s = _plain_int(window.get("window_start_s"), "window_start_s")
        end_s = _plain_int(window.get("window_end_s"), "window_end_s")
        if (end_s - 30) * 1000 <= event_ms <= end_s * 1000:
            late_winner_prices.append(price_up if winner_up else 1.0 - price_up)
        rows.append(
            {
                "t_known_ns": known_ns,
                "t_known_err_ns": known_err_ns,
                "t_known_prov": provenance,
                "t_event_ms": event_ms,
                "day": day.isoformat(),
                "slug": slug,
                "coin": coin,
                "parent_id": parent_id,
                "transaction_hashes_json": str(
                    trade.get("transaction_hashes_json", "[]")
                ),
                "trade_known_ns": int(trade["t_known_ns"]),
                "resolution_known_ns": int(window["resolution_known_ns"]),
                "window_start_s": start_s,
                "window_end_s": end_s,
                "phase": _knowledge_phase(trade, window),
                "phase_clock": "KNOWLEDGE_TIME_WITH_ERROR",
                "token_side": str(trade.get("token_side", "")),
                "q_up": q_up,
                "price_up": price_up,
                "size": size,
                "winner_up": winner_up,
                "outcome_up": outcome_up,
                "maker_edge_per_share": edge,
                "maker_edge_cents": edge * 100.0,
                "maker_gross_cash": edge * size,
                "fee_status": "GROSS_ONLY_NOT_APPLIED",
                "side_evidence_sha256": side_evidence.artifact_sha256,
                "constituent_count": int(trade.get("constituent_count", 1)),
                "clob_gap_count": int(window.get("clob_gap_count", 0)),
                "clob_gap_ms": float(window.get("clob_gap_ms", 0.0)),
                "clob_gap_causes_json": str(
                    window.get("clob_gap_causes_json", "{}")
                ),
                "clob_slow_consumer_gap": bool(
                    window.get("clob_slow_consumer_gap", False)
                ),
            }
        )
    if not rows:
        raise RuntimeError(f"no parent trades for {day} {coin}")
    rows.sort(key=lambda row: (row["t_known_ns"], row["slug"], row["parent_id"]))
    late_mean = (
        sum(late_winner_prices) / len(late_winner_prices)
        if late_winner_prices
        else None
    )
    mapping_status = (
        "INSUFFICIENT_SAMPLE"
        if len(late_winner_prices) < 30
        else "PASS"
        if late_mean is not None and late_mean > 0.5
        else "FAIL"
    )
    if mapping_status == "FAIL":
        raise RuntimeError("winning-token late-price mapping check failed")
    diagnostics = {
        "input_parent_trades": len(trades),
        "output_rows": len(rows),
        "unique_parent_ids": len(seen_parent_ids),
        "mapping_check": {
            "status": mapping_status,
            "n": len(late_winner_prices),
            "mean_winning_token_price": late_mean,
        },
        "phase_counts": _counts(row["phase"] for row in rows),
        "gap_touched_rows": sum(int(row["clob_gap_count"] > 0) for row in rows),
        "weighting_contract": "PER_FILL_AND_SHARE",
        "summaries": _markout_summaries(rows),
        "claim_status": "DESCRIPTIVE_POINT_ESTIMATE",
    }
    return rows, diagnostics


def _counts(values: Iterable[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _quote_status(quote: Mapping[str, Any]) -> tuple[str, dict[str, float | bool]]:
    bid = _finite_float(quote.get("bid_up"), "bid_up")
    ask = _finite_float(quote.get("ask_up"), "ask_up")
    tick = _finite_float(quote.get("tick_size"), "tick_size")
    pair_consistent = bool(quote.get("pair_consistent"))
    values: dict[str, float | bool] = {
        "bid": bid,
        "ask": ask,
        "tick": tick,
        "pair_consistent": pair_consistent,
    }
    if not 0 <= bid <= ask <= 1:
        return "INVALID_BOUNDS", values
    if not 0 < tick <= 1:
        return "INVALID_TICK", values
    if not pair_consistent:
        return "PAIR_INCONSISTENT", values
    return "AVAILABLE", values


def build_calibration_rows(
    *,
    day: date,
    coin: str,
    quotes: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    coverage: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the fixed nine-horizon point-in-time book scaffold."""
    window_by_slug = _window_index(windows, day=day, coin=coin)
    coverage_by_slug = _coverage_index(coverage, set(window_by_slug))
    quotes_by_slug: dict[str, list[Mapping[str, Any]]] = {
        slug: [] for slug in window_by_slug
    }
    for quote in quotes:
        slug = str(quote.get("slug", ""))
        if slug not in quotes_by_slug:
            raise RuntimeError(f"quote references unknown window {slug!r}")
        quotes_by_slug[slug].append(quote)
    for selected in quotes_by_slug.values():
        selected.sort(
            key=lambda row: (
                int(row["t_known_ns"]),
                int(row["t_event_ms"]),
                int(row["seq"]),
            )
        )

    rows: list[dict[str, Any]] = []
    for slug in sorted(window_by_slug):
        window = window_by_slug[slug]
        coverage_row = coverage_by_slug[slug]
        end_s = int(window["window_end_s"])
        for r_s in R_HORIZONS_S:
            decision_ns = (end_s - r_s) * 1_000_000_000
            candidates = [
                quote
                for quote in quotes_by_slug[slug]
                if int(quote["t_known_ns"]) + int(quote["t_known_err_ns"])
                <= decision_ns
            ]
            quote = (
                max(
                    candidates,
                    key=lambda row: (
                        int(row["t_event_ms"]),
                        int(row["t_known_ns"]),
                        int(row["seq"]),
                    ),
                )
                if candidates
                else None
            )
            status = "NO_ADMITTED_QUOTE"
            bid: float | None = None
            ask: float | None = None
            spread: float | None = None
            tick: float | None = None
            p_raw: float | None = None
            p_clipped: float | None = None
            clip_floor: float | None = None
            pair_consistent: bool | None = None
            quote_event_ms: int | None = None
            quote_known_ns: int | None = None
            quote_error_ns: int | None = None
            staleness_ns: int | None = None
            clock_inputs: list[Mapping[str, Any]] = [window]
            if quote is not None:
                status, values = _quote_status(quote)
                bid = float(values["bid"])
                ask = float(values["ask"])
                tick = float(values["tick"])
                pair_consistent = bool(values["pair_consistent"])
                spread = ask - bid
                quote_event_ms = int(quote["t_event_ms"])
                quote_known_ns = int(quote["t_known_ns"])
                quote_error_ns = int(quote["t_known_err_ns"])
                staleness_ns = decision_ns - quote_event_ms * 1_000_000
                if staleness_ns < 0:
                    raise RuntimeError(
                        f"admitted quote has future event time for {slug}"
                    )
                if status == "AVAILABLE":
                    p_raw = (bid + ask) / 2.0
                    clip_floor = max(tick / 2.0, 5e-4)
                    p_clipped = min(max(p_raw, clip_floor), 1.0 - clip_floor)
                clock_inputs.append(quote)
            known_ns, known_err_ns, provenance = _composed_clock(clock_inputs)
            rows.append(
                {
                    "t_known_ns": known_ns,
                    "t_known_err_ns": known_err_ns,
                    "t_known_prov": provenance,
                    "t_event_ms": end_s * 1000,
                    "day": day.isoformat(),
                    "slug": slug,
                    "coin": coin,
                    "r_s": r_s,
                    "decision_time_ns": decision_ns,
                    "winner_up": bool(window["winner_up"]),
                    "resolution_known_ns": int(window["resolution_known_ns"]),
                    "quote_status": status,
                    "quote_t_event_ms": quote_event_ms,
                    "quote_t_known_ns": quote_known_ns,
                    "quote_t_known_err_ns": quote_error_ns,
                    "quote_staleness_ns": staleness_ns,
                    "bid_up": bid,
                    "ask_up": ask,
                    "spread": spread,
                    "tick_size": tick,
                    "p_book_raw": p_raw,
                    "p_book_clipped": p_clipped,
                    "clip_floor": clip_floor,
                    "pair_consistent": pair_consistent,
                    "route_a_admissible": bool(coverage_row["admissible"]),
                    "coverage_hash": str(coverage_row["coverage_hash"]),
                    "admissibility_rule_id": str(coverage_row["rule_id"]),
                    "admissibility_rule_hash": str(coverage_row["rule_hash"]),
                    "model_status": "UNAVAILABLE_NOT_ATTACHED",
                    "isotonic_status": "UNAVAILABLE_REQUIRES_WALK_FORWARD_HISTORY",
                    "clob_gap_count": int(window.get("clob_gap_count", 0)),
                    "clob_gap_ms": float(window.get("clob_gap_ms", 0.0)),
                    "clob_gap_causes_json": str(
                        window.get("clob_gap_causes_json", "{}")
                    ),
                    "clob_slow_consumer_gap": bool(
                        window.get("clob_slow_consumer_gap", False)
                    ),
                }
            )
    expected = len(window_by_slug) * len(R_HORIZONS_S)
    keys = {(str(row["slug"]), int(row["r_s"])) for row in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError("calibration panel violates one-row grid")
    rows.sort(key=lambda row: (row["t_known_ns"], row["slug"], -row["r_s"]))
    diagnostics = {
        "windows": len(window_by_slug),
        "horizons_s": list(R_HORIZONS_S),
        "expected_rows": expected,
        "output_rows": len(rows),
        "unique_slug_horizon_keys": len(keys),
        "quote_status_counts": _counts(row["quote_status"] for row in rows),
        "route_a_admissible_rows": sum(
            int(row["route_a_admissible"]) for row in rows
        ),
        "model_status": "UNAVAILABLE_NOT_ATTACHED",
        "isotonic_status": "UNAVAILABLE_REQUIRES_WALK_FORWARD_HISTORY",
        "claim_status": "NORMALIZED_SCAFFOLD_ONLY",
    }
    return rows, diagnostics


@contextmanager
def evaluation_lock(tier2_root: Path) -> Iterator[Path]:
    lock_dir = tier2_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / "evaluation_pipeline.lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another evaluation pipeline holds {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f"{path.stem}-",
            suffix=".json.tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _partition_path(root: Path, dataset: str, day: date, coin: str) -> Path:
    return root / dataset / f"day={day.isoformat()}" / f"coin={coin}"


def _manifest_payload(
    *,
    dataset: str,
    day: date,
    coin: str,
    rows: int,
    output_path: Path,
    output_sha256: str,
    input_batch_hash: str,
    input_manifest_hashes: Sequence[str],
    side_evidence_sha256: str | None,
    diagnostics: Mapping[str, Any],
    written_at: str,
) -> dict[str, Any]:
    schema = SCHEMAS[dataset]
    source_descriptor = {
        "dataset": dataset,
        "day": day.isoformat(),
        "coin": coin,
        "distiller_version": DERIVER_VERSION,
        "distiller_code_sha256": _sha256_file(Path(__file__)),
        "schema_sha256": _schema_hash(schema),
        "input_batch_hash": input_batch_hash,
        "input_manifest_hashes": list(input_manifest_hashes),
        "side_evidence_sha256": side_evidence_sha256,
    }
    source_digest = _content_hash(source_descriptor)
    return {
        "dataset": dataset.upper(),
        "day": day.isoformat(),
        "coin": coin,
        "path": str(output_path),
        "rows": rows,
        "inputs": list(input_manifest_hashes),
        "distiller_version": DERIVER_VERSION,
        "distiller_code_sha256": _sha256_file(Path(__file__)),
        "schema_sha256": _schema_hash(schema),
        "era": _content_hash(
            {
                "source_digest": source_digest,
                "input_batch_hash": input_batch_hash,
                "side_evidence_sha256": side_evidence_sha256,
            }
        ),
        "written_at": written_at,
        "source_digest": source_digest,
        "output_sha256": output_sha256,
        "partial": False,
        "consumer_status": "ELIGIBLE_INPUT_ONLY",
        "input_batch_hash": input_batch_hash,
        "side_evidence_sha256": side_evidence_sha256,
        "diagnostics": dict(diagnostics),
    }


def _validate_derived_manifest(
    root: Path, dataset: str, day: date, coin: str
) -> dict[str, Any] | None:
    partition = _partition_path(root, dataset, day, coin)
    output_path = partition / "part-0.parquet"
    manifest_path = partition / "manifest.json"
    if not output_path.exists() and not manifest_path.exists():
        return None
    if not output_path.exists() or not manifest_path.exists():
        raise RuntimeError(f"incomplete Tier-2 partition {partition}")
    manifest = json.loads(manifest_path.read_text())
    payload = dict(manifest)
    declared = str(payload.pop("manifest_hash", ""))
    if not declared or declared != _content_hash(payload):
        raise RuntimeError(f"Tier-2 manifest hash mismatch at {manifest_path}")
    if manifest.get("partial") is not False:
        raise RuntimeError(f"Tier-2 consumer refuses partial partition {partition}")
    if manifest.get("distiller_version") != DERIVER_VERSION:
        raise RuntimeError(f"Tier-2 version mismatch at {partition}")
    if manifest.get("distiller_code_sha256") != _sha256_file(Path(__file__)):
        raise RuntimeError(f"Tier-2 code hash mismatch at {partition}")
    if manifest.get("schema_sha256") != _schema_hash(SCHEMAS[dataset]):
        raise RuntimeError(f"Tier-2 schema hash mismatch at {partition}")
    if manifest.get("output_sha256") != _sha256_file(output_path):
        raise RuntimeError(f"Tier-2 output hash mismatch at {output_path}")
    return manifest


def write_derived_partition(
    *,
    root: Path,
    dataset: str,
    day: date,
    coin: str,
    rows: Sequence[Mapping[str, Any]],
    input_batch_hash: str,
    input_manifest_hashes: Sequence[str],
    side_evidence_sha256: str | None,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    if dataset not in SCHEMAS:
        raise ValueError(f"unknown Tier-2 dataset {dataset}")
    if not rows:
        raise RuntimeError(f"Tier-2 refuses empty {dataset} for {day} {coin}")
    existing = _validate_derived_manifest(root, dataset, day, coin)
    partition = _partition_path(root, dataset, day, coin)
    output_path = partition / "part-0.parquet"
    expected_source = _manifest_payload(
        dataset=dataset,
        day=day,
        coin=coin,
        rows=len(rows),
        output_path=output_path,
        output_sha256=existing["output_sha256"] if existing else "PENDING",
        input_batch_hash=input_batch_hash,
        input_manifest_hashes=input_manifest_hashes,
        side_evidence_sha256=side_evidence_sha256,
        diagnostics=diagnostics,
        written_at=existing["written_at"] if existing else "PENDING",
    )
    if existing is not None:
        for key in (
            "source_digest",
            "input_batch_hash",
            "inputs",
            "side_evidence_sha256",
            "rows",
        ):
            if existing.get(key) != expected_source.get(key):
                raise RuntimeError(f"Tier-2 merge-never-overwrite mismatch: {key}")
        return existing

    partition.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=SCHEMAS[dataset])
    table = table.replace_schema_metadata(
        {
            b"distiller_version": DERIVER_VERSION.encode(),
            b"input_batch_hash": input_batch_hash.encode(),
            b"partial": b"false",
        }
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{partition.name}-", dir=partition.parent)
    )
    try:
        staged_output = staging / "part-0.parquet"
        pq.write_table(
            table,
            staged_output,
            compression="zstd",
            row_group_size=ROW_GROUP_SIZE,
            use_dictionary=True,
            write_statistics=True,
        )
        manifest = _manifest_payload(
            dataset=dataset,
            day=day,
            coin=coin,
            rows=len(rows),
            output_path=output_path,
            output_sha256=_sha256_file(staged_output),
            input_batch_hash=input_batch_hash,
            input_manifest_hashes=input_manifest_hashes,
            side_evidence_sha256=side_evidence_sha256,
            diagnostics=diagnostics,
            written_at=datetime.now(timezone.utc).isoformat(),
        )
        manifest["manifest_hash"] = _content_hash(manifest)
        _atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, partition)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _universe_id(coins: Sequence[str]) -> str:
    return _content_hash(list(coins))[:16]


def _run_path(root: Path, day: date, coins: Sequence[str]) -> Path:
    return (
        root
        / "runs"
        / f"day={day.isoformat()}"
        / f"universe={_universe_id(coins)}"
        / "run.json"
    )


def _load_hashed_json(path: Path, hash_field: str) -> dict[str, Any]:
    document = json.loads(path.read_text())
    declared = str(document.get(hash_field, ""))
    payload = dict(document)
    payload.pop(hash_field, None)
    if not declared or declared != _content_hash(payload):
        raise RuntimeError(f"{hash_field} mismatch at {path}")
    return document


def load_evaluation_run(
    *, tier2_root: Path, day: date, coins: Sequence[str]
) -> dict[str, Any] | None:
    path = _run_path(tier2_root, day, coins)
    if not path.exists():
        return None
    run = _load_hashed_json(path, "run_hash")
    if (
        run.get("status") != "COMPLETE"
        or run.get("target_day") != day.isoformat()
        or list(run.get("coins", [])) != list(coins)
        or run.get("source_batch_lane") != "full"
    ):
        raise RuntimeError(f"invalid evaluation run at {path}")
    expected_coins = set(coins)
    for field, dataset in (
        ("per_coin_markout_manifest_hashes", "markout_events"),
        ("per_coin_calib_manifest_hashes", "calib_panel"),
    ):
        if set(run.get(field, {})) != expected_coins:
            raise RuntimeError(f"evaluation coin binding mismatch: {field}")
        for coin in coins:
            manifest = _validate_derived_manifest(tier2_root, dataset, day, coin)
            if manifest is None or manifest.get("manifest_hash") != run[field][coin]:
                raise RuntimeError(f"evaluation manifest mismatch for {dataset} {coin}")
            if manifest.get("input_batch_hash") != run.get("source_batch_hash"):
                raise RuntimeError(f"evaluation batch binding mismatch for {coin}")
    return run


def _write_evaluation_run(
    *,
    tier2_root: Path,
    day: date,
    coins: Sequence[str],
    batch_hash: str,
    side_evidence_sha256: str,
    markout_manifests: Mapping[str, Mapping[str, Any]],
    calib_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(markout_manifests) != set(coins) or set(calib_manifests) != set(coins):
        raise ValueError("evaluation receipt requires both datasets for every coin")
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_code_sha256": _sha256_file(Path(__file__)),
        "target_day": day.isoformat(),
        "coins": list(coins),
        "source_batch_lane": "full",
        "source_batch_hash": batch_hash,
        "side_evidence_sha256": side_evidence_sha256,
        "per_coin_markout_manifest_hashes": {
            coin: str(markout_manifests[coin]["manifest_hash"]) for coin in coins
        },
        "per_coin_calib_manifest_hashes": {
            coin: str(calib_manifests[coin]["manifest_hash"]) for coin in coins
        },
        "claim_status": "DESCRIPTIVE_ARTIFACTS_ONLY",
        "status": "COMPLETE",
    }
    payload["run_hash"] = _content_hash(payload)
    path = _run_path(tier2_root, day, coins)
    if path.exists():
        existing = _load_hashed_json(path, "run_hash")
        if existing != payload:
            raise RuntimeError(f"evaluation receipt merge-never-overwrite at {path}")
        return existing
    _atomic_json(path, payload)
    return payload


def _tier1_manifests(
    *, tier1_root: Path, day: date, coin: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for dataset in ("quotes", "trades", "windows", "coverage"):
        manifest = validate_existing_partition(
            output_root=tier1_root,
            dataset=dataset,
            day=day,
            coin=coin,
        )
        if manifest is None:
            raise RuntimeError(f"full batch is missing {dataset} for {day} {coin}")
        result[dataset] = manifest
    return result


def execute_evaluation(
    *,
    day: date,
    coins: Sequence[str],
    tier1_root: Path,
    tier2_root: Path,
    side_evidence_path: Path = DEFAULT_SIDE_EVIDENCE,
) -> dict[str, Any]:
    """Resume Tier-2 partitions and publish the evaluation receipt last."""
    selected_coins = _unique_coins(coins)
    side_evidence = load_side_evidence(side_evidence_path)
    with evaluation_lock(tier2_root):
        batch = load_completed_batch(
            output_root=tier1_root,
            day=day,
            lane="full",
            coins=selected_coins,
        )
        if batch is None:
            raise RuntimeError("evaluation requires a completed full-lane batch")
        completed = load_evaluation_run(
            tier2_root=tier2_root, day=day, coins=selected_coins
        )
        if completed is not None:
            if completed.get("source_batch_hash") != batch.get("batch_hash"):
                raise RuntimeError("completed evaluation source batch changed")
            if completed.get("side_evidence_sha256") != side_evidence.artifact_sha256:
                raise RuntimeError("completed evaluation side evidence changed")
            return completed

        markout_manifests: dict[str, dict[str, Any]] = {}
        calib_manifests: dict[str, dict[str, Any]] = {}
        for coin in selected_coins:
            manifests = _tier1_manifests(
                tier1_root=tier1_root, day=day, coin=coin
            )
            windows = read_partition(tier1_root, "windows", day, coin).to_pylist()
            trades = read_partition(tier1_root, "trades", day, coin).to_pylist()
            quotes = read_partition(tier1_root, "quotes", day, coin).to_pylist()
            coverage = read_partition(tier1_root, "coverage", day, coin).to_pylist()
            markout_rows, markout_diagnostics = build_markout_rows(
                day=day,
                coin=coin,
                trades=trades,
                windows=windows,
                side_evidence=side_evidence,
            )
            calib_rows, calib_diagnostics = build_calibration_rows(
                day=day,
                coin=coin,
                quotes=quotes,
                windows=windows,
                coverage=coverage,
            )
            markout_manifests[coin] = write_derived_partition(
                root=tier2_root,
                dataset="markout_events",
                day=day,
                coin=coin,
                rows=markout_rows,
                input_batch_hash=str(batch["batch_hash"]),
                input_manifest_hashes=(
                    manifests["trades"]["manifest_hash"],
                    manifests["windows"]["manifest_hash"],
                ),
                side_evidence_sha256=side_evidence.artifact_sha256,
                diagnostics=markout_diagnostics,
            )
            calib_manifests[coin] = write_derived_partition(
                root=tier2_root,
                dataset="calib_panel",
                day=day,
                coin=coin,
                rows=calib_rows,
                input_batch_hash=str(batch["batch_hash"]),
                input_manifest_hashes=(
                    manifests["quotes"]["manifest_hash"],
                    manifests["windows"]["manifest_hash"],
                    manifests["coverage"]["manifest_hash"],
                ),
                side_evidence_sha256=None,
                diagnostics=calib_diagnostics,
            )
        return _write_evaluation_run(
            tier2_root=tier2_root,
            day=day,
            coins=selected_coins,
            batch_hash=str(batch["batch_hash"]),
            side_evidence_sha256=side_evidence.artifact_sha256,
            markout_manifests=markout_manifests,
            calib_manifests=calib_manifests,
        )


def _synthetic_rows(
    target: date,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    start_s = int(
        datetime(target.year, target.month, target.day, tzinfo=timezone.utc).timestamp()
    )
    windows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    quotes: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for index in range(288):
        ws = start_s + index * 300
        end = ws + 300
        slug = f"btc-updown-5m-{ws}"
        winner_up = index % 2 == 0
        resolution_known = (end + 10) * 1_000_000_000
        windows.append(
            {
                "t_known_ns": resolution_known,
                "t_known_err_ns": 1_000_000,
                "t_known_prov": "OBSERVED",
                "t_event_ms": ws * 1000,
                "slug": slug,
                "coin": "btc",
                "window_start_s": ws,
                "window_end_s": end,
                "resolution_known_ns": resolution_known,
                "closed": True,
                "winner_up": winner_up,
                "clob_gap_count": int(index == 0),
                "clob_gap_ms": 1000.0 if index == 0 else 0.0,
                "clob_gap_causes_json": '{"TEST": 1}' if index == 0 else "{}",
                "clob_slow_consumer_gap": False,
            }
        )
        trade_known = (ws + 281) * 1_000_000_000
        q_up = 1 if index % 3 else -1
        price_up = 0.9 if winner_up else 0.1
        trades.append(
            {
                "t_known_ns": trade_known,
                "t_known_err_ns": 1_000_000,
                "t_known_prov": "OBSERVED",
                "t_event_ms": (ws + 280) * 1000,
                "slug": slug,
                "coin": "btc",
                "token_side": "UP",
                "price_up": price_up,
                "size": 10.0,
                "q_up": q_up,
                "parent_id": f"parent-{index}",
                "transaction_hashes_json": f'["tx-{index}"]',
                "constituent_count": 1,
            }
        )
        quote_known = (ws + 2) * 1_000_000_000
        if index != 287:
            bid = 0.0 if index == 1 else 0.49
            ask = 0.01 if index == 1 else 0.51
            tick = 0.001 if index == 1 else 0.01
            quotes.append(
                {
                    "t_known_ns": quote_known,
                    "t_known_err_ns": 1_000_000,
                    "t_known_prov": "OBSERVED",
                    "t_event_ms": (ws + 1) * 1000,
                    "slug": slug,
                    "coin": "btc",
                    "bid_up": bid,
                    "ask_up": ask,
                    "spread": ask - bid,
                    "tick_size": tick,
                    "pair_consistent": True,
                    "seq": index,
                }
            )
        coverage.append(
            {
                "slug": slug,
                "admissible": index % 5 != 0,
                "coverage_hash": f"coverage-{index}",
                "rule_id": "A-TWAP-1",
                "rule_hash": "rule-hash",
            }
        )
    first_ws = start_s
    quotes.append(
        {
            "t_known_ns": (first_ws + 31) * 1_000_000_000,
            "t_known_err_ns": 1_000_000,
            "t_known_prov": "OBSERVED",
            "t_event_ms": (first_ws + 29) * 1000,
            "slug": f"btc-updown-5m-{first_ws}",
            "coin": "btc",
            "bid_up": 0.59,
            "ask_up": 0.61,
            "spread": 0.02,
            "tick_size": 0.01,
            "pair_consistent": True,
            "seq": 10_000,
        }
    )
    return windows, trades, quotes, coverage


def selftest() -> None:
    target = date(1970, 1, 2)
    with tempfile.TemporaryDirectory(prefix="pm-evaluation-test-") as tmp:
        root = Path(tmp)
        evidence_path = root / "gff1.json"
        evidence_payload = {
            "manifest": {"protocol": "gff1_test"},
            "n_validated_tx": 500,
            "agreement": 1.0,
            "wilson95": [0.992, 1.0],
            "threshold": 0.99,
            "verdict": "PASS",
        }
        _atomic_json(evidence_path, evidence_payload)
        evidence = load_side_evidence(evidence_path)
        windows, trades, quotes, coverage = _synthetic_rows(target)
        markout, markout_diag = build_markout_rows(
            day=target,
            coin="btc",
            trades=trades,
            windows=windows,
            side_evidence=evidence,
        )
        assert len(markout) == 288
        assert markout_diag["mapping_check"]["status"] == "PASS"
        first = next(row for row in markout if row["parent_id"] == "parent-0")
        assert math.isclose(first["maker_edge_per_share"], 0.1)
        assert first["fee_status"] == "GROSS_ONLY_NOT_APPLIED"
        assert markout_diag["weighting_contract"] == "PER_FILL_AND_SHARE"
        print("  PASS  terminal markout identity is model-free and dual-weighted")

        panel, panel_diag = build_calibration_rows(
            day=target,
            coin="btc",
            quotes=quotes,
            windows=windows,
            coverage=coverage,
        )
        assert len(panel) == 288 * len(R_HORIZONS_S)
        assert len({(row["slug"], row["r_s"]) for row in panel}) == len(panel)
        earliest = next(
            row
            for row in panel
            if row["slug"] == f"btc-updown-5m-{86_400}" and row["r_s"] == 270
        )
        assert earliest["p_book_raw"] == 0.5
        assert earliest["quote_t_known_ns"] == (86_400 + 2) * 1_000_000_000
        assert panel_diag["quote_status_counts"] == {
            "AVAILABLE": len(panel) - len(R_HORIZONS_S),
            "NO_ADMITTED_QUOTE": len(R_HORIZONS_S),
        }
        boundary_slug = f"btc-updown-5m-{86_400 + 300}"
        boundary = next(
            row
            for row in panel
            if row["slug"] == boundary_slug and row["r_s"] == 270
        )
        assert boundary["quote_status"] == "AVAILABLE"
        assert boundary["bid_up"] == 0.0 and boundary["p_book_raw"] == 0.005
        print("  PASS  calibration grid is unique and knowledge-time truncated")

        markout_manifest = write_derived_partition(
            root=root / "tier2",
            dataset="markout_events",
            day=target,
            coin="btc",
            rows=markout,
            input_batch_hash="batch-test",
            input_manifest_hashes=("trades", "windows"),
            side_evidence_sha256=evidence.artifact_sha256,
            diagnostics=markout_diag,
        )
        calib_manifest = write_derived_partition(
            root=root / "tier2",
            dataset="calib_panel",
            day=target,
            coin="btc",
            rows=panel,
            input_batch_hash="batch-test",
            input_manifest_hashes=("quotes", "windows", "coverage"),
            side_evidence_sha256=None,
            diagnostics=panel_diag,
        )
        repeated = write_derived_partition(
            root=root / "tier2",
            dataset="markout_events",
            day=target,
            coin="btc",
            rows=markout,
            input_batch_hash="batch-test",
            input_manifest_hashes=("trades", "windows"),
            side_evidence_sha256=evidence.artifact_sha256,
            diagnostics=markout_diag,
        )
        assert repeated == markout_manifest
        run = _write_evaluation_run(
            tier2_root=root / "tier2",
            day=target,
            coins=("btc",),
            batch_hash="batch-test",
            side_evidence_sha256=evidence.artifact_sha256,
            markout_manifests={"btc": markout_manifest},
            calib_manifests={"btc": calib_manifest},
        )
        loaded = load_evaluation_run(
            tier2_root=root / "tier2", day=target, coins=("btc",)
        )
        assert loaded == run
        print("  PASS  Tier-2 partitions and commit-last receipt are immutable")

        try:
            execute_evaluation(
                day=target,
                coins=("btc",),
                tier1_root=root / "missing-tier1",
                tier2_root=root / "isolated-tier2",
                side_evidence_path=evidence_path,
            )
        except RuntimeError as exc:
            assert "completed full-lane batch" in str(exc)
        else:
            raise AssertionError("Tier-2 derivation accepted no full batch")
        print("  PASS  Tier-2 derivation refuses without a completed full batch")

        evidence_payload["verdict"] = "INSUFFICIENT_EVIDENCE"
        _atomic_json(evidence_path, evidence_payload)
        try:
            load_side_evidence(evidence_path)
        except RuntimeError as exc:
            assert "not PASS" in str(exc)
        else:
            raise AssertionError("non-PASS side evidence was accepted")
        print("  PASS  non-PASS side evidence fails before derivation")


def _emit_run(run: Mapping[str, Any], json_output: bool) -> None:
    if json_output:
        print(json.dumps(run, sort_keys=True))
    else:
        print(
            f"EVALUATION_COMPLETE day={run['target_day']} "
            f"coins={len(run['coins'])} hash={str(run['run_hash'])[:12]} "
            f"claim={run['claim_status']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--day", help="target UTC day YYYY-MM-DD")
    target.add_argument("--latest", action="store_true", help="select today-2 UTC")
    target.add_argument(
        "--catch-up",
        action="store_true",
        help="run oldest candidate without a Tier-2 completion receipt",
    )
    parser.add_argument("--coin", action="append", choices=tuple(COIN_SYMBOL))
    parser.add_argument("--tier1-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tier2-root", type=Path, default=DEFAULT_TIER2_ROOT)
    parser.add_argument(
        "--side-evidence", type=Path, default=DEFAULT_SIDE_EVIDENCE
    )
    parser.add_argument("--as-of-day", help="planner clock override YYYY-MM-DD")
    parser.add_argument("--since", help="catch-up lower bound YYYY-MM-DD")
    parser.add_argument("--max-days", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="return success when blocked/idle so a timer can retry",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    if not (args.day or args.latest or args.catch_up):
        if args.selftest:
            return
        parser.error("choose --day, --latest, or --catch-up")
    if args.max_days <= 0:
        parser.error("--max-days must be positive")

    today = (
        date.fromisoformat(args.as_of_day)
        if args.as_of_day
        else datetime.now(timezone.utc).date()
    )
    coins = _unique_coins(args.coin or tuple(COIN_SYMBOL))
    markets, resolutions, _ = load_market_metadata()
    side_evidence = load_side_evidence(args.side_evidence)
    if args.day:
        target_days = [date.fromisoformat(args.day)]
    elif args.latest:
        target_days = [today - timedelta(days=2)]
    else:
        candidates = discover_candidate_days(
            markets=markets,
            coins=coins,
            today=today,
            since=date.fromisoformat(args.since) if args.since else None,
        )
        target_days = []
        for candidate in candidates:
            completed = load_evaluation_run(
                tier2_root=args.tier2_root, day=candidate, coins=coins
            )
            if (
                completed is not None
                and completed.get("side_evidence_sha256")
                != side_evidence.artifact_sha256
            ):
                raise RuntimeError(
                    "completed evaluation references different side evidence"
                )
            if completed is None:
                target_days.append(candidate)
            if len(target_days) >= args.max_days:
                break
        if not target_days:
            message = {
                "status": "IDLE",
                "coins": list(coins),
                "as_of_day": today.isoformat(),
                "side_evidence_sha256": side_evidence.artifact_sha256,
            }
            print(
                json.dumps(message, sort_keys=True)
                if args.json
                else "EVALUATION_IDLE"
            )
            return

    blocked = False
    for selected_day in target_days:
        plan = plan_batch(
            day=selected_day,
            coins=coins,
            lane="full",
            today=today,
            markets=markets,
            resolutions=resolutions,
        )
        plan_output = {
            "stage": "FULL_BATCH_PREFLIGHT",
            "side_evidence": asdict(side_evidence),
            "plan": batch_plan_dict(plan),
        }
        if args.json:
            print(json.dumps(plan_output, sort_keys=True))
        else:
            failed = {
                item.coin: [check.name for check in item.checks if not check.ready]
                for item in plan.per_coin
                if not item.ready
            }
            print(
                f"EVALUATION_PLAN day={selected_day} coins={len(coins)} "
                f"ready={str(plan.ready).lower()} side={side_evidence.verdict} "
                f"failed={failed or '-'}"
            )
        if not plan.ready:
            blocked = True
            break
        if args.plan_only:
            continue
        if args.verify:
            batch = load_completed_batch(
                output_root=args.tier1_root,
                day=selected_day,
                lane="full",
                coins=coins,
            )
            if batch is None:
                raise RuntimeError("evaluation verification requires its full batch")
            run = load_evaluation_run(
                tier2_root=args.tier2_root, day=selected_day, coins=coins
            )
            if run is None:
                raise RuntimeError("no completed evaluation run to verify")
            if run.get("source_batch_hash") != batch.get("batch_hash"):
                raise RuntimeError("evaluation/full-batch receipt mismatch")
            if run.get("side_evidence_sha256") != side_evidence.artifact_sha256:
                raise RuntimeError("evaluation/side-evidence receipt mismatch")
            _emit_run(run, args.json)
            continue
        execute_batch(plan, output_root=args.tier1_root)
        run = execute_evaluation(
            day=selected_day,
            coins=coins,
            tier1_root=args.tier1_root,
            tier2_root=args.tier2_root,
            side_evidence_path=args.side_evidence,
        )
        _emit_run(run, args.json)
    if blocked and not args.scheduled:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
