"""Deterministic Tier-0 -> Tier-1 distillation for P-2026-003.

This module is research data infrastructure.  It does not fit a model, score a
forecast, calculate markout, or authorize a trading decision.  It converts the
immutable collector landing zone into a compact, knowledge-time-preserving
Parquet spine:

* ``twap``: Chainlink price updates;
* ``quotes``: distinct reconstructed UP-token top-of-book states;
* ``trades``: token-normalized aggressive prints in UP coordinates; and
* ``windows``: market/resolution identity plus cause-aware gap facts; and
* ``coverage``: factual one-second coverage plus a separate hash-bound
  A-TWAP-1 decision.

Every partition is sorted by ``t_known_ns``, written temp-then-rename, and
paired with a content-addressed manifest.  Repeating an identical build reuses
the partition; a changed input set at the same partition key fails loudly.
Partial builds are supported only for smoke testing and are stamped
``partial=true`` so downstream readers can refuse them.

Examples::

    python3 -m live.pm_research.tier1_pipeline --selftest
    python3 -m live.pm_research.tier1_pipeline \
        --dataset twap --day 2026-08-20 --coin btc
    python3 -m live.pm_research.tier1_pipeline \
        --dataset clob --day 2026-08-20 --coin btc
    python3 -m live.pm_research.tier1_pipeline \
        --dataset coverage --day 2026-08-20 --coin btc

Coverage requires the previous, selected, and next UTC day's TWAP partitions.
This deliberately gives a complete daily coverage partition a one-day lag.

No active uncompressed collector file is read.  Only rotated ``*.gz`` inputs
are eligible for a non-partial partition.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from live.pm_research.coverage_ledger import (
    AdmissibilityRule,
    SourceRegistry,
    evaluate,
    load_rule,
    load_source_registry,
    measure_twap_coverage,
)


REPO = Path(__file__).resolve().parents[2]
PM = REPO / "data/pm_5min"
DEFAULT_OUTPUT_ROOT = PM / "tier1"
DISTILLER_VERSION = "tier1_v4_r12"
PRICE_SCALE = 1_000_000
ROW_GROUP_SIZE = 65_536
MAX_NS = (1 << 63) - 1
COIN_SYMBOL = {
    "btc": "btc/usd",
    "eth": "eth/usd",
    "sol": "sol/usd",
    "xrp": "xrp/usd",
    "doge": "doge/usd",
    "bnb": "bnb/usd",
    "hype": "hype/usd",
}
SYMBOL_COIN = {value: key for key, value in COIN_SYMBOL.items()}


TWAP_SCHEMA = pa.schema(
    [
        ("t_known_ns", pa.int64()),
        ("t_known_err_ns", pa.int64()),
        ("t_known_prov", pa.string()),
        ("t_event_ms", pa.int64()),
        ("t_publish_ms", pa.int64()),
        ("symbol", pa.string()),
        ("coin", pa.string()),
        ("topic", pa.string()),
        ("window_s", pa.int16()),
        ("value", pa.float64()),
        ("full_accuracy_value", pa.string()),
        ("seq", pa.int64()),
        ("duplicate_count", pa.int32()),
        ("source", pa.string()),
        ("source_profile_hash", pa.string()),
        ("collector_version", pa.string()),
        ("collector_era_coverage", pa.string()),
        ("source_file_id", pa.string()),
    ]
)

QUOTE_SCHEMA = pa.schema(
    [
        ("t_known_ns", pa.int64()),
        ("t_known_err_ns", pa.int64()),
        ("t_known_prov", pa.string()),
        ("t_event_ms", pa.int64()),
        ("slug", pa.string()),
        ("coin", pa.string()),
        ("bid_up", pa.float64()),
        ("ask_up", pa.float64()),
        ("bid_size", pa.float64()),
        ("ask_size", pa.float64()),
        ("spread", pa.float64()),
        ("tick_size", pa.float64()),
        ("pair_consistent", pa.bool_()),
        ("source_event_type", pa.string()),
        ("seq", pa.int64()),
        ("source", pa.string()),
        ("source_profile_hash", pa.string()),
        ("collector_version", pa.string()),
        ("collector_era_coverage", pa.string()),
        ("source_file_id", pa.string()),
    ]
)

TRADE_SCHEMA = pa.schema(
    [
        ("t_known_ns", pa.int64()),
        ("t_known_err_ns", pa.int64()),
        ("t_known_prov", pa.string()),
        ("t_event_ms", pa.int64()),
        ("slug", pa.string()),
        ("coin", pa.string()),
        ("asset_id", pa.string()),
        ("token_side", pa.string()),
        ("price_token", pa.float64()),
        ("price_up", pa.float64()),
        ("size", pa.float64()),
        ("side_raw", pa.string()),
        ("q_up", pa.int8()),
        ("transaction_hash", pa.string()),
        ("transaction_hashes_json", pa.string()),
        ("parent_id", pa.string()),
        ("constituent_count", pa.int32()),
        ("fee_rate_bps_raw", pa.float64()),
        ("fee_source_status", pa.string()),
        ("seq", pa.int64()),
        ("source", pa.string()),
        ("source_profile_hash", pa.string()),
        ("collector_version", pa.string()),
        ("collector_era_coverage", pa.string()),
        ("source_file_id", pa.string()),
    ]
)

WINDOW_SCHEMA = pa.schema(
    [
        ("t_known_ns", pa.int64()),
        ("t_known_err_ns", pa.int64()),
        ("t_known_prov", pa.string()),
        ("t_event_ms", pa.int64()),
        ("slug", pa.string()),
        ("coin", pa.string()),
        ("window_start_s", pa.int64()),
        ("window_end_s", pa.int64()),
        ("condition_id", pa.string()),
        ("up_asset", pa.string()),
        ("down_asset", pa.string()),
        ("market_known_ns", pa.int64()),
        ("resolution_known_ns", pa.int64()),
        ("closed", pa.bool_()),
        ("winner_up", pa.bool_()),
        ("clob_gap_count", pa.int32()),
        ("clob_gap_ms", pa.float64()),
        ("clob_gap_causes_json", pa.string()),
        ("clob_slow_consumer_gap", pa.bool_()),
        ("price_gap_count", pa.int32()),
        ("price_gap_ms", pa.float64()),
        ("price_gap_causes_json", pa.string()),
        ("gap_rule_status", pa.string()),
        ("seq", pa.int64()),
        ("source", pa.string()),
        ("source_profile_hash", pa.string()),
        ("source_file_id", pa.string()),
    ]
)

COVERAGE_SCHEMA = pa.schema(
    [
        ("t_known_ns", pa.int64()),
        ("t_known_err_ns", pa.int64()),
        ("t_known_prov", pa.string()),
        ("t_event_ms", pa.int64()),
        ("slug", pa.string()),
        ("coin", pa.string()),
        ("symbol", pa.string()),
        ("field", pa.string()),
        ("target_start_ns", pa.int64()),
        ("target_end_ns", pa.int64()),
        ("covered_json", pa.string()),
        ("gaps_json", pa.string()),
        ("weight_missing", pa.float64()),
        ("tail_deficit_ns", pa.int64()),
        ("observed_n", pa.int32()),
        ("expected_n", pa.int32()),
        ("duplicate_n", pa.int32()),
        ("max_gap_ns", pa.int64()),
        ("complete_frac", pa.float64()),
        ("strike_readable", pa.bool_()),
        ("protected_gap", pa.bool_()),
        ("coverage_hash", pa.string()),
        ("rule_id", pa.string()),
        ("rule_hash", pa.string()),
        ("admissible", pa.bool_()),
        ("failed_checks_json", pa.string()),
        ("evaluated_at_ns", pa.int64()),
        ("seq", pa.int64()),
        ("source", pa.string()),
        ("source_profile_hash", pa.string()),
    ]
)

SCHEMAS = {
    "twap": TWAP_SCHEMA,
    "quotes": QUOTE_SCHEMA,
    "trades": TRADE_SCHEMA,
    "windows": WINDOW_SCHEMA,
    "coverage": COVERAGE_SCHEMA,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an integer: {value!r}") from exc
    if result < 0:
        raise ValueError(f"{name} is negative")
    return result


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("day must be YYYY-MM-DD") from exc


def _compact_day(value: date) -> str:
    return value.strftime("%Y%m%d")


def _event_day(event_ms: int) -> date:
    return datetime.fromtimestamp(event_ms / 1000, timezone.utc).date()


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read()


@dataclass(frozen=True)
class InputSnapshot:
    path: str
    size: int
    sha256: str


def snapshot_file(path: Path, *, raw: bytes | None = None) -> InputSnapshot:
    payload = raw if raw is not None else _read_bytes(path)
    return InputSnapshot(
        path=str(path.relative_to(REPO) if path.is_relative_to(REPO) else path),
        size=len(payload),
        sha256=_sha256_bytes(payload),
    )


@dataclass(frozen=True)
class GapInterval:
    subject: str
    cause: str
    collector_version: str
    start_ns: int
    end_ns: int
    open_at_snapshot: bool

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


class CollectorLedger:
    """Byte-snapshotted collector versions and cause-aware gap intervals."""

    def __init__(self, path: Path, raw: bytes, *, subject_field: str) -> None:
        self.path = path
        self.raw = raw
        self.snapshot = snapshot_file(path, raw=raw)
        self.subject_field = subject_field
        self.events: list[dict[str, Any]] = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"malformed ledger JSON {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object ledger row {path}:{line_number}")
            self.events.append(row)
        self.events.sort(key=lambda row: int(row.get("recv_ns", 0)))
        self.snapshot_ns = max(
            (int(row.get("recv_ns", 0)) for row in self.events), default=0
        )
        self._eras = self._build_eras()
        self.gaps = self._build_gaps()

    @classmethod
    def load(cls, path: Path, *, subject_field: str) -> "CollectorLedger":
        raw = _read_bytes(path) if path.exists() else b""
        return cls(path, raw, subject_field=subject_field)

    def _build_eras(self) -> list[tuple[int, int, str]]:
        active: dict[str, tuple[int, str]] = {}
        eras: list[tuple[int, int, str]] = []
        for row in self.events:
            event = row.get("event")
            pid = row.get("pid")
            if pid is None:
                continue
            key = str(pid)
            recv_ns = _plain_int(row.get("recv_ns"), "ledger.recv_ns")
            if event == "collector_start":
                if key in active:
                    raise ValueError(f"duplicate active collector pid {key}")
                active[key] = (
                    recv_ns,
                    str(row.get("collector_version", "UNKNOWN")),
                )
            elif event == "collector_stop" and key in active:
                start_ns, version = active.pop(key)
                eras.append((start_ns, recv_ns, version))
        for start_ns, version in active.values():
            eras.append((start_ns, MAX_NS, version))
        return sorted(eras)

    def version_at(self, recv_ns: int) -> tuple[str, str]:
        versions = sorted(
            {
                version
                for start_ns, end_ns, version in self._eras
                if start_ns <= recv_ns <= end_ns
            }
        )
        if versions:
            return "+".join(versions), "LEDGER_ACTIVE"
        starts = [start for start, _, _ in self._eras]
        return (
            ("PRE_LEDGER", "PRE_LEDGER")
            if not starts or recv_ns < min(starts)
            else ("NO_ACTIVE_COLLECTOR", "LEDGER_INACTIVE")
        )

    def _build_gaps(self) -> list[GapInterval]:
        closed: dict[tuple[str, str, str, int], GapInterval] = {}
        for row in self.events:
            if row.get("event") != "gap_closed":
                continue
            subject = row.get(self.subject_field)
            if subject is None:
                continue
            start_ns = _plain_int(row.get("gap_start_ns"), "gap_start_ns")
            end_ns = _plain_int(row.get("gap_end_ns"), "gap_end_ns")
            if end_ns < start_ns:
                raise ValueError("gap end precedes gap start")
            key = (
                str(row.get("collector_version", "UNKNOWN")),
                str(subject),
                str(row.get("cause", "UNKNOWN")),
                start_ns,
            )
            closed[key] = GapInterval(
                subject=key[1],
                cause=key[2],
                collector_version=key[0],
                start_ns=start_ns,
                end_ns=end_ns,
                open_at_snapshot=False,
            )
        result = list(closed.values())
        for row in self.events:
            if row.get("event") != "gap_open":
                continue
            subject = row.get(self.subject_field)
            if subject is None:
                continue
            start_ns = _plain_int(row.get("gap_start_ns"), "gap_start_ns")
            key = (
                str(row.get("collector_version", "UNKNOWN")),
                str(subject),
                str(row.get("cause", "UNKNOWN")),
                start_ns,
            )
            if key in closed:
                continue
            result.append(
                GapInterval(
                    subject=key[1],
                    cause=key[2],
                    collector_version=key[0],
                    start_ns=start_ns,
                    end_ns=MAX_NS,
                    open_at_snapshot=True,
                )
            )
        return sorted(result, key=lambda gap: (gap.start_ns, gap.subject))

    def overlaps(self, subject: str, start_ns: int, end_ns: int) -> list[GapInterval]:
        return [
            gap
            for gap in self.gaps
            if gap.subject == subject
            and gap.start_ns <= end_ns
            and gap.end_ns >= start_ns
        ]


@dataclass(frozen=True)
class MarketInfo:
    slug: str
    coin: str
    window_start_s: int
    window_end_s: int
    condition_id: str
    up_asset: str
    down_asset: str
    market_known_ns: int
    source_file_id: str


@dataclass(frozen=True)
class ResolutionInfo:
    slug: str
    resolution_known_ns: int
    winner_up: bool


def _jsonl_rows(raw: bytes, source: str) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON {source}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSON row {source}:{line_number}")
        yield row


def load_market_metadata(
    markets_path: Path = PM / "markets.jsonl",
    resolutions_path: Path = PM / "resolutions.jsonl",
) -> tuple[
    dict[str, MarketInfo],
    dict[str, ResolutionInfo],
    list[InputSnapshot],
]:
    markets_raw = _read_bytes(markets_path)
    resolutions_raw = _read_bytes(resolutions_path)
    market_snapshot = snapshot_file(markets_path, raw=markets_raw)
    resolution_snapshot = snapshot_file(resolutions_path, raw=resolutions_raw)
    markets: dict[str, MarketInfo] = {}
    for row in _jsonl_rows(markets_raw, str(markets_path)):
        slug = str(row.get("slug", ""))
        if not slug:
            raise ValueError("market row has no slug")
        if slug in markets:
            continue  # earliest knowledge of immutable identity wins
        outcomes = row.get("outcomes")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        tokens = row.get("clobTokenIds")
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        if outcomes != ["Up", "Down"] or not isinstance(tokens, list) or len(tokens) != 2:
            raise ValueError(f"unexpected outcome/token mapping for {slug}")
        markets[slug] = MarketInfo(
            slug=slug,
            coin=str(row["coin"]),
            window_start_s=_plain_int(row["window_start"], "window_start"),
            window_end_s=_plain_int(row["window_end"], "window_end"),
            condition_id=str(row.get("conditionId", "")),
            up_asset=str(tokens[0]),
            down_asset=str(tokens[1]),
            market_known_ns=_plain_int(row["recv_ns"], "market.recv_ns"),
            source_file_id=market_snapshot.sha256[:16],
        )
    resolutions: dict[str, ResolutionInfo] = {}
    for row in _jsonl_rows(resolutions_raw, str(resolutions_path)):
        if row.get("closed") is not True or not isinstance(row.get("winners"), dict):
            continue
        slug = str(row.get("slug", ""))
        if not slug or slug in resolutions:
            continue
        resolutions[slug] = ResolutionInfo(
            slug=slug,
            resolution_known_ns=_plain_int(row["recv_ns"], "resolution.recv_ns"),
            winner_up=bool(row["winners"].get("Up")),
        )
    return markets, resolutions, [market_snapshot, resolution_snapshot]


@dataclass
class ParseStats:
    files: int = 0
    physical_lines: int = 0
    messages: int = 0
    blank_lines: int = 0
    partial_lines: int = 0
    malformed_lines: int = 0
    nonmonotone_recv: int = 0
    duplicate_rows: int = 0
    book_envelope_collapsed: int = 0
    collapsed_rows: int = 0
    collapse_groups: int = 0
    top_checks: int = 0
    top_matches: int = 0
    reconciled_levels: int = 0
    invalidated_books: int = 0
    unknown_assets: int = 0


def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


def _iter_wire_file(
    path: Path,
    stats: ParseStats,
    *,
    source_file_id: str,
) -> Iterator[tuple[int, int, str, dict[str, Any]]]:
    stats.files += 1
    previous_recv = -1
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            stats.physical_lines += 1
            if not line.strip():
                stats.blank_lines += 1
                continue
            prefix, separator, payload = line.partition("\t")
            if not separator:
                stats.partial_lines += 1
                continue
            try:
                recv_ns = _plain_int(prefix, "recv_ns")
                decoded = json.loads(payload)
            except (ValueError, json.JSONDecodeError):
                stats.malformed_lines += 1
                continue
            if recv_ns < previous_recv:
                stats.nonmonotone_recv += 1
            previous_recv = recv_ns
            messages = decoded if isinstance(decoded, list) else [decoded]
            for message_index, message in enumerate(messages):
                if not isinstance(message, dict):
                    stats.malformed_lines += 1
                    continue
                stats.messages += 1
                source_seq = (line_number << 16) + message_index
                yield recv_ns, source_seq, source_file_id, message


def _input_snapshots(paths: Sequence[Path]) -> list[InputSnapshot]:
    return [snapshot_file(path) for path in paths]


def discover_twap_files(day: date, topic: str) -> list[Path]:
    root = PM / "prices" / topic
    return sorted(root.glob(f"{_compact_day(day)}_*.csv.gz"))


def _slug_from_path(path: Path) -> str:
    return path.name.split(".jsonl", 1)[0]


def _shard_order(path: Path) -> tuple[str, int]:
    slug = _slug_from_path(path)
    suffix = path.name[len(slug) :]
    if suffix == ".jsonl.gz":
        return slug, 0
    if suffix.startswith(".jsonl.") and suffix.endswith(".gz"):
        middle = suffix[len(".jsonl.") : -len(".gz")]
        return slug, _plain_int(middle, "shard number") + 1
    raise ValueError(f"unsupported immutable shard name: {path.name}")


def discover_clob_files(day: date, coin: str) -> list[Path]:
    root = PM / "raw" / _compact_day(day)
    return sorted(root.glob(f"{coin}-updown-5m-*.jsonl*.gz"), key=_shard_order)


def _filter_inputs(
    paths: Sequence[Path], max_files: int | None
) -> tuple[list[Path], bool]:
    if max_files is None:
        return list(paths), False
    if max_files <= 0:
        raise ValueError("max_files must be positive")
    return list(paths[:max_files]), len(paths) > max_files


def normalize_twap(
    paths: Sequence[Path],
    *,
    day: date,
    coin: str,
    ledger: CollectorLedger,
    source_registry: SourceRegistry | None = None,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], ParseStats, bool]:
    source_registry = source_registry or load_source_registry()
    symbol = COIN_SYMBOL[coin]
    stats = ParseStats()
    candidates: list[dict[str, Any]] = []
    stopped = False
    seen_records = 0
    for path, snap in zip(paths, _input_snapshots(paths)):
        for recv_ns, source_seq, source_id, body in _iter_wire_file(
            path, stats, source_file_id=snap.sha256[:16]
        ):
            payload = body.get("payload")
            if not isinstance(payload, dict) or payload.get("symbol") != symbol:
                continue
            if body.get("topic") not in (
                "crypto_prices_twap_thirty",
                "crypto_prices_twap_sixty",
            ):
                continue
            event_ms = _plain_int(payload.get("timestamp"), "twap.timestamp")
            if _event_day(event_ms) != day:
                continue
            publish_ms = _plain_int(body.get("timestamp"), "twap.publish timestamp")
            if recv_ns < event_ms * 1_000_000:
                raise ValueError("TWAP recv_ns precedes payload event time")
            version, coverage = ledger.version_at(recv_ns)
            source = (
                "pm_twap30_ws"
                if body["topic"] == "crypto_prices_twap_thirty"
                else "pm_twap60_ws"
            )
            profile = source_registry.profile(source)
            candidates.append(
                {
                    "t_known_ns": recv_ns,
                    "t_known_err_ns": profile.clock_err_ns,
                    "t_known_prov": "OBSERVED",
                    "t_event_ms": event_ms,
                    "t_publish_ms": publish_ms,
                    "symbol": symbol,
                    "coin": coin,
                    "topic": str(body["topic"]),
                    "window_s": _plain_int(payload.get("window_s"), "window_s"),
                    "value": _finite_float(payload.get("value"), "twap.value"),
                    "full_accuracy_value": str(payload.get("full_accuracy_value", "")),
                    "seq": source_seq,
                    "duplicate_count": 0,
                    "source": source,
                    "source_profile_hash": source_registry.registry_hash,
                    "collector_version": version,
                    "collector_era_coverage": coverage,
                    "source_file_id": source_id,
                }
            )
            seen_records += 1
            if max_records is not None and seen_records >= max_records:
                stopped = True
                break
        if stopped:
            break
    candidates.sort(key=lambda row: (row["t_known_ns"], row["seq"]))
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in candidates:
        key = (
            row["topic"],
            row["symbol"],
            row["window_s"],
            row["t_event_ms"],
        )
        if key in deduped:
            stats.duplicate_rows += 1
            deduped[key]["duplicate_count"] += 1
            continue
        deduped[key] = row
    rows = sorted(deduped.values(), key=lambda row: (row["t_known_ns"], row["seq"]))
    for seq, row in enumerate(rows):
        row["seq"] = seq
    return rows, stats, stopped


class _BookState:
    def __init__(self, up_asset: str, down_asset: str) -> None:
        self.up_asset = up_asset
        self.down_asset = down_asset
        self.levels: dict[str, dict[str, dict[int, float]]] = defaultdict(
            lambda: {"BUY": {}, "SELL": {}}
        )
        self.valid = {up_asset: False, down_asset: False}
        self.tick_size = 0.01

    def apply_book(self, message: Mapping[str, Any]) -> None:
        asset = str(message.get("asset_id", ""))
        if asset not in (self.up_asset, self.down_asset):
            raise KeyError(asset)
        bids: dict[int, float] = {}
        asks: dict[int, float] = {}
        for row in message.get("bids") or []:
            price = round(_finite_float(row["price"], "bid.price") * PRICE_SCALE)
            size = _finite_float(row["size"], "bid.size")
            if size > 0:
                bids[price] = size
        for row in message.get("asks") or []:
            price = round(_finite_float(row["price"], "ask.price") * PRICE_SCALE)
            size = _finite_float(row["size"], "ask.size")
            if size > 0:
                asks[price] = size
        self.levels[asset] = {"BUY": bids, "SELL": asks}
        self.valid[asset] = bool(bids and asks)
        if message.get("tick_size") is not None:
            self.tick_size = _finite_float(message["tick_size"], "tick_size")

    def apply_change(self, change: Mapping[str, Any]) -> None:
        asset = str(change.get("asset_id", ""))
        if asset not in (self.up_asset, self.down_asset):
            raise KeyError(asset)
        side = str(change.get("side", ""))
        if side not in ("BUY", "SELL"):
            raise ValueError(f"unknown price_change side {side!r}")
        price = round(_finite_float(change["price"], "change.price") * PRICE_SCALE)
        size = _finite_float(change["size"], "change.size")
        book = self.levels[asset][side]
        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size

    def _best(self, asset: str) -> tuple[int, float, int, float] | None:
        bids = self.levels[asset]["BUY"]
        asks = self.levels[asset]["SELL"]
        if not bids or not asks:
            return None
        bid = max(bids)
        ask = min(asks)
        return bid, bids[bid], ask, asks[ask]

    def reconcile_change_top(
        self, change: Mapping[str, Any], stats: ParseStats
    ) -> None:
        """Reconcile one sequential change to its authoritative post-change top."""
        asset = str(change.get("asset_id", ""))
        if asset not in (self.up_asset, self.down_asset):
            raise KeyError(asset)
        if change.get("best_bid") is None or change.get("best_ask") is None:
            self.valid[asset] = False
            stats.invalidated_books += 1
            return
        bid = round(_finite_float(change["best_bid"], "best_bid") * PRICE_SCALE)
        ask = round(_finite_float(change["best_ask"], "best_ask") * PRICE_SCALE)
        if bid >= ask:
            raise ValueError("authoritative CLOB top is crossed")
        expected = self._best(asset)
        stats.top_checks += 1
        if expected is not None and (bid, ask) == (expected[0], expected[2]):
            stats.top_matches += 1

        # The feed can omit explicit zero-size removals when an incoming order
        # matches through resting levels.  The per-change best fields are the
        # authoritative post-change state, so discard levels that they prove
        # no longer exist.
        bids = self.levels[asset]["BUY"]
        asks = self.levels[asset]["SELL"]
        stale_bids = [price for price in bids if price > bid]
        stale_asks = [price for price in asks if price < ask]
        for price in stale_bids:
            del bids[price]
        for price in stale_asks:
            del asks[price]
        stats.reconciled_levels += len(stale_bids) + len(stale_asks)

        reconciled = self._best(asset)
        is_valid = (
            reconciled is not None
            and (reconciled[0], reconciled[2]) == (bid, ask)
        )
        if not is_valid:
            stats.invalidated_books += 1
        self.valid[asset] = is_valid

    def up_top(self) -> tuple[float, float, float, float, bool] | None:
        up = self._best(self.up_asset)
        down = self._best(self.down_asset)
        # Collector snapshots arrive as a two-token JSON batch.  The wire-file
        # iterator deliberately flattens that batch, so do not publish the
        # transient one-sided state between its UP and DOWN records.
        if (
            up is None
            or down is None
            or not self.valid[self.up_asset]
            or not self.valid[self.down_asset]
        ):
            return None
        bid, bid_size, ask, ask_size = up
        pair_consistent = (
            abs(up[0] + down[2] - PRICE_SCALE) <= 1
            and abs(up[2] + down[0] - PRICE_SCALE) <= 1
        )
        return (
            bid / PRICE_SCALE,
            ask / PRICE_SCALE,
            bid_size,
            ask_size,
            pair_consistent,
        )


BOOK_ENVELOPE_FIELDS = ("last_trade_price", "tick_size")


def _identity_digest(message: Mapping[str, Any]) -> str:
    """Digest of the identity-bearing content, for the duplicate check.

    R-12: for a `book` snapshot the venue `hash` identifies the book, so the
    optional envelope fields are not part of its identity and two deliveries
    that differ only there are ONE state seen twice.  Everything else digests
    whole -- in particular `bids`/`asks` disagreeing under one venue hash is a
    real conflict and still raises, which is the case the guard exists for
    (0 of 19 in the census, and we would want to hear about the first one).
    """
    if message.get("event_type") == "book":
        stripped = {
            key: value
            for key, value in message.items()
            if key not in BOOK_ENVELOPE_FIELDS
        }
        return _sha256_bytes(_canonical_json(stripped))
    return _sha256_bytes(_canonical_json(message))


def _raw_message_key(message: Mapping[str, Any]) -> tuple[Any, ...]:
    event_type = message.get("event_type")
    if event_type == "last_trade_price":
        transaction_hash = message.get("transaction_hash")
        if transaction_hash:
            return (
                event_type,
                transaction_hash,
                message.get("asset_id"),
                message.get("timestamp"),
            )
        return (
            event_type,
            message.get("asset_id"),
            message.get("timestamp"),
            message.get("price"),
            message.get("size"),
            message.get("side"),
        )
    if event_type == "price_change":
        # R-12: the resulting top-of-book is part of the EVENT IDENTITY.
        #
        # The venue `timestamp` is not unique: a corpus census over 264.8 M
        # records found 499 keys where two messages shared `timestamp` and every
        # change-row field yet sat up to 113 ms and 27 cents apart, each copy
        # internally consistent at bid(Up)+ask(Down)=1.0000 (1,996/1,996 sums).
        # They are two distinct successive book states, not duplicates -- so the
        # key is repaired and BOTH RECORDS ARE RETAINED.  Nothing is excluded:
        # dropping either copy destroys a top-of-book observation, and
        # `price_change.best_bid/ask` is what the standing rule says the whole
        # corpus must read book state from.
        changes = message.get("price_changes") or []
        return (
            event_type,
            message.get("timestamp"),
            tuple(
                (
                    row.get("asset_id"),
                    row.get("hash"),
                    row.get("price"),
                    row.get("size"),
                    row.get("side"),
                    row.get("best_bid"),
                    row.get("best_ask"),
                )
                for row in changes
            ),
        )
    if event_type == "book":
        # R-12: the venue `hash` identifies the book, so optional ENVELOPE
        # fields are not part of book identity.  All 19 conflicting book keys in
        # the census differed only in `last_trade_price`/`tick_size`, with the
        # same venue hash and no shared field disagreeing -- a genuine
        # re-delivery.  These collapse, and the collapsed count is published.
        return (event_type, message.get("asset_id"), message.get("hash"), message.get("timestamp"))
    return (event_type, message.get("timestamp"), _sha256_bytes(_canonical_json(message)))


def _collapse_trades(
    rows: Sequence[Mapping[str, Any]], stats: ParseStats
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["slug"]), int(row["t_event_ms"]), int(row["q_up"]))
        ].append(row)

    collapsed: list[dict[str, Any]] = []
    for key in sorted(grouped):
        members = sorted(
            grouped[key],
            key=lambda row: (
                int(row["t_known_ns"]),
                int(row["seq"]),
                str(row["transaction_hash"]),
            ),
        )
        total_size = sum(float(row["size"]) for row in members)
        if total_size <= 0:
            raise ValueError("parent trade has non-positive total size")
        hashes = sorted(
            {
                str(row["transaction_hash"])
                for row in members
                if row["transaction_hash"]
            }
        )
        token_sides = {str(row["token_side"]) for row in members}
        raw_sides = {str(row["side_raw"]) for row in members}
        assets = {str(row["asset_id"]) for row in members}
        versions = sorted({str(row["collector_version"]) for row in members})
        coverages = sorted(
            {str(row["collector_era_coverage"]) for row in members}
        )
        sources = sorted({str(row["source_file_id"]) for row in members})
        fee_statuses = {str(row["fee_source_status"]) for row in members}
        parent_descriptor = {
            "slug": key[0],
            "t_event_ms": key[1],
            "q_up": key[2],
            "transaction_hashes": hashes,
            "fallback_members": [
                [
                    str(row["asset_id"]),
                    float(row["price_token"]),
                    float(row["size"]),
                    str(row["side_raw"]),
                ]
                for row in members
                if not row["transaction_hash"]
            ],
        }
        representative = members[-1]
        collapsed.append(
            {
                **dict(representative),
                "t_known_ns": max(int(row["t_known_ns"]) for row in members),
                "asset_id": next(iter(assets)) if len(assets) == 1 else "MIXED",
                "token_side": (
                    next(iter(token_sides)) if len(token_sides) == 1 else "MIXED"
                ),
                "price_token": (
                    sum(
                        float(row["price_token"]) * float(row["size"])
                        for row in members
                    )
                    / total_size
                    if len(token_sides) == 1
                    else None
                ),
                "price_up": sum(
                    float(row["price_up"]) * float(row["size"])
                    for row in members
                )
                / total_size,
                "size": total_size,
                "side_raw": (
                    next(iter(raw_sides)) if len(raw_sides) == 1 else "MIXED"
                ),
                "transaction_hash": hashes[0] if len(hashes) == 1 else "",
                "transaction_hashes_json": json.dumps(hashes),
                "parent_id": _sha256_bytes(_canonical_json(parent_descriptor)),
                "constituent_count": len(members),
                "fee_rate_bps_raw": sum(
                    float(row["fee_rate_bps_raw"]) * float(row["size"])
                    for row in members
                )
                / total_size,
                "fee_source_status": (
                    next(iter(fee_statuses)) if len(fee_statuses) == 1 else "MIXED"
                ),
                "seq": max(int(row["seq"]) for row in members),
                "collector_version": "+".join(versions),
                "collector_era_coverage": "+".join(coverages),
                "source_file_id": "+".join(sources),
            }
        )
        if len(members) > 1:
            stats.collapse_groups += 1
            stats.collapsed_rows += len(members) - 1
    return collapsed


def normalize_clob(
    paths: Sequence[Path],
    *,
    day: date,
    coin: str,
    markets: Mapping[str, MarketInfo],
    ledger: CollectorLedger,
    source_registry: SourceRegistry | None = None,
    max_records: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], ParseStats, bool]:
    source_registry = source_registry or load_source_registry()
    source = "pm_clob_ws"
    profile = source_registry.profile(source)
    stats = ParseStats()
    by_slug: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_slug[_slug_from_path(path)].append(path)
    quote_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    stopped = False
    consumed = 0
    for slug in sorted(by_slug):
        market = markets.get(slug)
        if market is None:
            raise ValueError(f"no market metadata for {slug}")
        if market.coin != coin:
            raise ValueError(f"coin mismatch for {slug}: {market.coin} != {coin}")
        records: list[tuple[int, int, str, dict[str, Any]]] = []
        shard_paths = sorted(by_slug[slug], key=_shard_order)
        snapshots = _input_snapshots(shard_paths)
        for path, snap in zip(shard_paths, snapshots):
            for record in _iter_wire_file(
                path, stats, source_file_id=snap.sha256[:16]
            ):
                records.append(record)
                consumed += 1
                if max_records is not None and consumed >= max_records:
                    stopped = True
                    break
            if stopped:
                break
        records.sort(key=lambda row: (row[0], row[1], row[2]))
        unique: list[tuple[int, int, str, dict[str, Any]]] = []
        seen: dict[tuple[Any, ...], str] = {}
        for record in records:
            key = _raw_message_key(record[3])
            digest = _identity_digest(record[3])
            if key in seen:
                if digest != seen[key]:
                    raise ValueError(
                        f"duplicate identity has conflicting payload for {slug}"
                    )
                stats.duplicate_rows += 1
                if record[3].get("event_type") == "book" and _sha256_bytes(
                    _canonical_json(record[3])
                ) != digest:
                    # Collapsed on the venue hash despite a differing envelope.
                    stats.book_envelope_collapsed += 1
                continue
            seen[key] = digest
            unique.append(record)

        book = _BookState(market.up_asset, market.down_asset)
        last_top: tuple[Any, ...] | None = None
        for recv_ns, source_seq, source_id, message in unique:
            event_type = str(message.get("event_type", ""))
            event_ms = _plain_int(message.get("timestamp"), "CLOB.timestamp")
            if recv_ns < event_ms * 1_000_000:
                raise ValueError("CLOB recv_ns precedes payload event time")
            version, coverage = ledger.version_at(recv_ns)
            if event_type == "book":
                try:
                    book.apply_book(message)
                except KeyError:
                    stats.unknown_assets += 1
                    continue
            elif event_type == "price_change":
                changes = message.get("price_changes") or []
                if not isinstance(changes, list):
                    raise ValueError("price_changes must be a list")
                for change in changes:
                    try:
                        book.apply_change(change)
                        book.reconcile_change_top(change, stats)
                    except KeyError:
                        stats.unknown_assets += 1
            elif event_type == "last_trade_price":
                asset = str(message.get("asset_id", ""))
                if asset == market.up_asset:
                    token_side = "UP"
                elif asset == market.down_asset:
                    token_side = "DOWN"
                else:
                    stats.unknown_assets += 1
                    continue
                price_token = _finite_float(message.get("price"), "trade.price")
                size = _finite_float(message.get("size"), "trade.size")
                if not 0 < price_token < 1:
                    raise ValueError("trade price must be strictly inside (0, 1)")
                if size <= 0:
                    raise ValueError("trade size must be positive")
                side_raw = str(message.get("side", ""))
                if side_raw not in ("BUY", "SELL"):
                    raise ValueError(f"unknown trade side {side_raw!r}")
                price_up = price_token if token_side == "UP" else 1.0 - price_token
                q_up = 1 if (token_side == "UP") == (side_raw == "BUY") else -1
                tx_hash = str(message.get("transaction_hash", ""))
                parent_id = tx_hash or _sha256_bytes(
                    _canonical_json(
                        [slug, event_ms, asset, price_token, size, side_raw]
                    )
                )
                fee_raw = _finite_float(message.get("fee_rate_bps", 0), "fee_rate_bps")
                trade_rows.append(
                    {
                        "t_known_ns": recv_ns,
                        "t_known_err_ns": profile.clock_err_ns,
                        "t_known_prov": "OBSERVED",
                        "t_event_ms": event_ms,
                        "slug": slug,
                        "coin": coin,
                        "asset_id": asset,
                        "token_side": token_side,
                        "price_token": price_token,
                        "price_up": price_up,
                        "size": size,
                        "side_raw": side_raw,
                        "q_up": q_up,
                        "transaction_hash": tx_hash,
                        "transaction_hashes_json": json.dumps(
                            [tx_hash] if tx_hash else []
                        ),
                        "parent_id": parent_id,
                        "constituent_count": 1,
                        "fee_rate_bps_raw": fee_raw,
                        "fee_source_status": (
                            "UNPOPULATED_WS_ZERO" if fee_raw == 0 else "OBSERVED_NONZERO"
                        ),
                        "seq": source_seq,
                        "source": source,
                        "source_profile_hash": source_registry.registry_hash,
                        "collector_version": version,
                        "collector_era_coverage": coverage,
                        "source_file_id": source_id,
                    }
                )
                continue
            else:
                continue

            top = book.up_top()
            if top is None:
                continue
            bid, ask, bid_size, ask_size, pair_consistent = top
            top_key = (bid, ask, bid_size, ask_size, pair_consistent)
            if top_key == last_top:
                continue
            last_top = top_key
            quote_rows.append(
                {
                    "t_known_ns": recv_ns,
                    "t_known_err_ns": profile.clock_err_ns,
                    "t_known_prov": "OBSERVED",
                    "t_event_ms": event_ms,
                    "slug": slug,
                    "coin": coin,
                    "bid_up": bid,
                    "ask_up": ask,
                    "bid_size": bid_size,
                    "ask_size": ask_size,
                    "spread": ask - bid,
                    "tick_size": book.tick_size,
                    "pair_consistent": pair_consistent,
                    "source_event_type": event_type,
                    "seq": source_seq,
                    "source": source,
                    "source_profile_hash": source_registry.registry_hash,
                    "collector_version": version,
                    "collector_era_coverage": coverage,
                    "source_file_id": source_id,
                }
            )
        if stopped:
            break
    trade_rows = _collapse_trades(trade_rows, stats)
    quote_rows.sort(key=lambda row: (row["t_known_ns"], row["seq"], row["slug"]))
    trade_rows.sort(key=lambda row: (row["t_known_ns"], row["seq"], row["slug"]))
    for seq, row in enumerate(quote_rows):
        row["seq"] = seq
    for seq, row in enumerate(trade_rows):
        row["seq"] = seq
    return quote_rows, trade_rows, stats, stopped


def _overlap_ms(gaps: Sequence[GapInterval], start_ns: int, end_ns: int) -> float:
    return sum(
        max(0, min(gap.end_ns, end_ns) - max(gap.start_ns, start_ns)) / 1e6
        for gap in gaps
    )


def normalize_windows(
    *,
    day: date,
    coin: str,
    markets: Mapping[str, MarketInfo],
    resolutions: Mapping[str, ResolutionInfo],
    clob_ledger: CollectorLedger,
    price_ledger: CollectorLedger,
    source_registry: SourceRegistry | None = None,
) -> list[dict[str, Any]]:
    source_registry = source_registry or load_source_registry()
    rows: list[dict[str, Any]] = []
    price_topic = "crypto_prices_twap_sixty"
    for market in sorted(markets.values(), key=lambda item: item.slug):
        if market.coin != coin:
            continue
        if datetime.fromtimestamp(market.window_start_s, timezone.utc).date() != day:
            continue
        start_ns = market.window_start_s * 1_000_000_000
        end_ns = market.window_end_s * 1_000_000_000
        clob_gaps = clob_ledger.overlaps(market.slug, start_ns, end_ns)
        price_gaps = price_ledger.overlaps(
            price_topic, start_ns - 5_000_000_000, end_ns + 5_000_000_000
        )
        resolution = resolutions.get(market.slug)
        source = "pm_resolution_poll" if resolution else "pm_market_poll"
        profile = source_registry.profile(source)
        causes_clob = Counter(gap.cause for gap in clob_gaps)
        causes_price = Counter(gap.cause for gap in price_gaps)
        rows.append(
            {
                # This row includes the resolution, so its own knowledge time
                # cannot precede resolution knowledge.  Consumers needing only
                # identity use the separate market_known_ns field.
                "t_known_ns": max(
                    market.market_known_ns,
                    resolution.resolution_known_ns if resolution else 0,
                ),
                "t_known_err_ns": profile.clock_err_ns,
                "t_known_prov": "OBSERVED",
                "t_event_ms": market.window_start_s * 1000,
                "slug": market.slug,
                "coin": coin,
                "window_start_s": market.window_start_s,
                "window_end_s": market.window_end_s,
                "condition_id": market.condition_id,
                "up_asset": market.up_asset,
                "down_asset": market.down_asset,
                "market_known_ns": market.market_known_ns,
                "resolution_known_ns": (
                    resolution.resolution_known_ns if resolution else None
                ),
                "closed": resolution is not None,
                "winner_up": resolution.winner_up if resolution else None,
                "clob_gap_count": len(clob_gaps),
                "clob_gap_ms": _overlap_ms(clob_gaps, start_ns, end_ns),
                "clob_gap_causes_json": json.dumps(causes_clob, sort_keys=True),
                "clob_slow_consumer_gap": bool(causes_clob.get("SLOW_CONSUMER_1013")),
                "price_gap_count": len(price_gaps),
                "price_gap_ms": _overlap_ms(
                    price_gaps,
                    start_ns - 5_000_000_000,
                    end_ns + 5_000_000_000,
                ),
                "price_gap_causes_json": json.dumps(causes_price, sort_keys=True),
                "gap_rule_status": "FACTS_ONLY_RULE_SEPARATE",
                "seq": 0,
                "source": source,
                "source_profile_hash": source_registry.registry_hash,
                "source_file_id": market.source_file_id,
            }
        )
    rows.sort(key=lambda row: (row["t_known_ns"], row["slug"]))
    for seq, row in enumerate(rows):
        row["seq"] = seq
    return rows


def _partition_dir(root: Path, dataset: str, day: date, coin: str) -> Path:
    """Partition address carries the distiller generation (R-10's principle).

    The manifest binds `distiller_code_sha256`, so before this any edit to this
    file made every existing partition raise `distiller code mismatch` at a
    FIXED address -- the same collision between immutability and amendability
    that R-10 resolved for the run record.  R-12 changes `_raw_message_key`,
    which is exactly such an edit.  Versioning the address means the amended
    generation is written alongside and the superseded one is simply kept.

    Note the code binding is deliberately broad: it invalidates `twap` even
    though R-12 only touches CLOB keying.  That is the guard being conservative
    rather than wrong, and the cost is recompute, not data.
    """
    return (
        root
        / dataset
        / f"day={day.isoformat()}"
        / f"coin={coin}"
        / f"distiller={DISTILLER_VERSION}"
    )


def _manifest_payload(
    *,
    dataset: str,
    day: date,
    coin: str,
    rows: int,
    inputs: Sequence[InputSnapshot],
    output_path: Path,
    output_sha256: str,
    source_digest: str,
    era: str,
    partial: bool,
    diagnostics: Mapping[str, Any],
    written_at: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset.upper(),
        "day": day.isoformat(),
        "coin": coin,
        "path": str(output_path),
        "rows": rows,
        "inputs": [asdict(item) for item in inputs],
        "distiller_version": DISTILLER_VERSION,
        "distiller_code_sha256": _sha256_file(Path(__file__)),
        "era": era,
        "written_at": written_at,
        "source_digest": source_digest,
        "output_sha256": output_sha256,
        "partial": partial,
        "consumer_status": "REFUSE_PARTIAL" if partial else "ELIGIBLE_INPUT_ONLY",
        "diagnostics": dict(diagnostics),
    }


def write_partition(
    root: Path,
    dataset: str,
    day: date,
    coin: str,
    rows: Sequence[Mapping[str, Any]],
    inputs: Sequence[InputSnapshot],
    *,
    partial: bool,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    if dataset not in SCHEMAS:
        raise ValueError(f"unknown dataset {dataset}")
    partition = _partition_dir(root, dataset, day, coin)
    output_path = partition / "part-0.parquet"
    manifest_path = partition / "manifest.json"
    source_descriptor = {
        "dataset": dataset,
        "day": day.isoformat(),
        "coin": coin,
        "distiller_version": DISTILLER_VERSION,
        "distiller_code_sha256": _sha256_file(Path(__file__)),
        "partial": partial,
        "inputs": [asdict(item) for item in sorted(inputs, key=lambda item: item.path)],
    }
    source_digest = _sha256_bytes(_canonical_json(source_descriptor))
    era = _sha256_bytes(
        _canonical_json(
            {
                "source_digest": source_digest,
                "collector_versions": sorted(
                    {
                        str(row.get("collector_version"))
                        for row in rows
                        if row.get("collector_version") is not None
                    }
                ),
            }
        )
    )
    if output_path.exists() or manifest_path.exists():
        if not output_path.exists() or not manifest_path.exists():
            raise RuntimeError(f"partial existing partition at {partition}")
        existing = json.loads(manifest_path.read_text())
        if (
            existing.get("source_digest") != source_digest
            or bool(existing.get("partial")) != partial
            or existing.get("distiller_version") != DISTILLER_VERSION
            or existing.get("distiller_code_sha256")
            != _sha256_file(Path(__file__))
        ):
            raise RuntimeError(
                f"manifest mismatch at {partition}; merge-never-overwrite"
            )
        actual_output = _sha256_file(output_path)
        if actual_output != existing.get("output_sha256"):
            raise RuntimeError(f"output digest mismatch at {output_path}")
        return existing

    partition.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=SCHEMAS[dataset])
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"distiller_version": DISTILLER_VERSION.encode(),
            b"source_digest": source_digest.encode(),
            b"partial": str(partial).lower().encode(),
        }
    )
    table = table.replace_schema_metadata(metadata)
    parquet_tmp: Path | None = None
    manifest_tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="part-", suffix=".parquet.tmp", dir=partition, delete=False
        ) as handle:
            parquet_tmp = Path(handle.name)
        pq.write_table(
            table,
            parquet_tmp,
            compression="zstd",
            row_group_size=ROW_GROUP_SIZE,
            use_dictionary=True,
            write_statistics=True,
        )
        output_sha256 = _sha256_file(parquet_tmp)
        written_at = datetime.now(timezone.utc).isoformat()
        manifest = _manifest_payload(
            dataset=dataset,
            day=day,
            coin=coin,
            rows=len(rows),
            inputs=inputs,
            output_path=output_path,
            output_sha256=output_sha256,
            source_digest=source_digest,
            era=era,
            partial=partial,
            diagnostics=diagnostics,
            written_at=written_at,
        )
        manifest["manifest_hash"] = _sha256_bytes(_canonical_json(manifest))
        with tempfile.NamedTemporaryFile(
            prefix="manifest-", suffix=".json.tmp", dir=partition, delete=False
        ) as handle:
            manifest_tmp = Path(handle.name)
            handle.write(json.dumps(manifest, indent=2, sort_keys=True).encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(parquet_tmp, output_path)
        parquet_tmp = None
        os.replace(manifest_tmp, manifest_path)
        manifest_tmp = None
        return manifest
    finally:
        for temporary in (parquet_tmp, manifest_tmp):
            if temporary is not None and temporary.exists():
                temporary.unlink()


def read_partition(
    root: Path,
    dataset: str,
    day: date,
    coin: str,
    *,
    allow_partial: bool = False,
) -> pa.Table:
    partition = _partition_dir(root, dataset, day, coin)
    output_path = partition / "part-0.parquet"
    manifest_path = partition / "manifest.json"
    if not output_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(partition)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("partial") and not allow_partial:
        raise RuntimeError(f"consumer refuses partial partition {partition}")
    if _sha256_file(output_path) != manifest.get("output_sha256"):
        raise RuntimeError(f"partition digest mismatch {output_path}")
    # Read the physical file directly.  pq.read_table() may infer the Hive
    # parent fields and collide with the explicit day/coin columns.
    return pq.ParquetFile(output_path).read()


def _assert_closed_day(day: date, partial: bool) -> None:
    today = datetime.now(timezone.utc).date()
    if day >= today and not partial:
        raise ValueError("a non-partial partition requires a completed UTC day")


def build_twap_partition(
    *,
    day: date,
    coin: str,
    output_root: Path,
    max_files: int | None,
    max_records: int | None,
    partial_requested: bool,
) -> dict[str, Any]:
    paths, files_truncated = _filter_inputs(
        sorted(
            discover_twap_files(day, "crypto_prices_twap_sixty")
            + discover_twap_files(day, "crypto_prices_twap_thirty")
        ),
        max_files,
    )
    if not paths:
        raise FileNotFoundError(f"no immutable TWAP files for {day}")
    ledger = CollectorLedger.load(
        PM / "prices/collector_gaps.jsonl", subject_field="topic"
    )
    source_registry = load_source_registry()
    rows, stats, records_truncated = normalize_twap(
        paths,
        day=day,
        coin=coin,
        ledger=ledger,
        source_registry=source_registry,
        max_records=max_records,
    )
    partial = partial_requested or files_truncated or records_truncated
    _assert_closed_day(day, partial)
    if not rows and not partial:
        raise RuntimeError(f"non-partial TWAP partition is empty for {day} {coin}")
    inputs = _input_snapshots(paths) + [
        ledger.snapshot,
        snapshot_file(source_registry.path),
    ]
    return write_partition(
        output_root,
        "twap",
        day,
        coin,
        rows,
        inputs,
        partial=partial,
        diagnostics=asdict(stats),
    )


def build_clob_partitions(
    *,
    day: date,
    coin: str,
    output_root: Path,
    max_files: int | None,
    max_records: int | None,
    partial_requested: bool,
) -> list[dict[str, Any]]:
    paths, files_truncated = _filter_inputs(discover_clob_files(day, coin), max_files)
    if not paths:
        raise FileNotFoundError(f"no immutable CLOB files for {day} {coin}")
    markets, _, metadata_inputs = load_market_metadata()
    ledger = CollectorLedger.load(PM / "collector_gaps.jsonl", subject_field="slug")
    source_registry = load_source_registry()
    quotes, trades, stats, records_truncated = normalize_clob(
        paths,
        day=day,
        coin=coin,
        markets=markets,
        ledger=ledger,
        source_registry=source_registry,
        max_records=max_records,
    )
    partial = partial_requested or files_truncated or records_truncated
    _assert_closed_day(day, partial)
    if not quotes and not partial:
        raise RuntimeError(f"non-partial quote partition is empty for {day} {coin}")
    inputs = _input_snapshots(paths) + metadata_inputs[:1] + [
        ledger.snapshot,
        snapshot_file(source_registry.path),
    ]
    diagnostics = asdict(stats)
    return [
        write_partition(
            output_root,
            dataset,
            day,
            coin,
            rows,
            inputs,
            partial=partial,
            diagnostics=diagnostics,
        )
        for dataset, rows in (("quotes", quotes), ("trades", trades))
    ]


def build_windows_partition(
    *,
    day: date,
    coin: str,
    output_root: Path,
    partial_requested: bool,
) -> dict[str, Any]:
    markets, resolutions, inputs = load_market_metadata()
    clob_ledger = CollectorLedger.load(PM / "collector_gaps.jsonl", subject_field="slug")
    price_ledger = CollectorLedger.load(
        PM / "prices/collector_gaps.jsonl", subject_field="topic"
    )
    source_registry = load_source_registry()
    rows = normalize_windows(
        day=day,
        coin=coin,
        markets=markets,
        resolutions=resolutions,
        clob_ledger=clob_ledger,
        price_ledger=price_ledger,
        source_registry=source_registry,
    )
    _assert_closed_day(day, partial_requested)
    if not rows and not partial_requested:
        raise RuntimeError(f"non-partial windows partition is empty for {day} {coin}")
    return write_partition(
        output_root,
        "windows",
        day,
        coin,
        rows,
        inputs
        + [
            clob_ledger.snapshot,
            price_ledger.snapshot,
            snapshot_file(source_registry.path),
        ],
        partial=partial_requested,
        diagnostics={"windows": len(rows), "admissibility": "SEPARATE_DATASET"},
    )


def normalize_coverage(
    *,
    day: date,
    coin: str,
    markets: Mapping[str, MarketInfo],
    twap_rows: Sequence[Mapping[str, Any]],
    price_ledger: CollectorLedger,
    rule: AdmissibilityRule,
    source_registry: SourceRegistry,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    symbol = COIN_SYMBOL[coin]
    source = "pm_twap60_ws"
    profile = source_registry.profile(source)
    rows: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    admissible_count = 0
    for market in sorted(markets.values(), key=lambda item: item.slug):
        if market.coin != coin:
            continue
        if datetime.fromtimestamp(market.window_start_s, timezone.utc).date() != day:
            continue
        t0_ms = market.window_start_s * 1_000
        end_ms = market.window_end_s * 1_000
        external_gaps = price_ledger.overlaps(
            "crypto_prices_twap_sixty",
            (t0_ms - 5_000) * 1_000_000,
            (end_ms + 5_000) * 1_000_000,
        )
        facts = measure_twap_coverage(
            twap_rows,
            symbol=symbol,
            t0_ms=t0_ms,
            end_ms=end_ms,
            rule=rule,
            source_profile=profile,
            source_profile_hash=source_registry.registry_hash,
            external_gaps=external_gaps,
        )
        # Negative evidence is not knowable before the interval whose absence
        # it describes has ended.  A missing tail can otherwise make
        # ``facts.t_known_ns`` precede target_end_ns and launder future coverage
        # into an earlier StateView.
        known_ns = max(
            facts.t_known_ns,
            market.market_known_ns,
            facts.target_end_ns,
        )
        decision = evaluate(facts, rule, evaluated_at_ns=known_ns)
        admissible_count += int(decision.admissible)
        failure_counts.update(decision.failed_checks)
        rows.append(
            {
                "t_known_ns": known_ns,
                "t_known_err_ns": profile.clock_err_ns,
                "t_known_prov": "OBSERVED",
                "t_event_ms": t0_ms,
                "slug": market.slug,
                "coin": coin,
                "symbol": symbol,
                "field": facts.field,
                "target_start_ns": facts.target_start_ns,
                "target_end_ns": facts.target_end_ns,
                "covered_json": json.dumps(facts.covered),
                "gaps_json": json.dumps(
                    [asdict(gap) for gap in facts.gaps], sort_keys=True
                ),
                "weight_missing": facts.weight_missing,
                "tail_deficit_ns": facts.tail_deficit_ns,
                "observed_n": facts.observed_n,
                "expected_n": facts.expected_n,
                "duplicate_n": facts.duplicate_n,
                "max_gap_ns": facts.max_gap_ns,
                "complete_frac": facts.complete_frac,
                "strike_readable": facts.strike_readable,
                "protected_gap": facts.protected_gap,
                "coverage_hash": facts.coverage_hash,
                "rule_id": decision.rule,
                "rule_hash": decision.rule_hash,
                "admissible": decision.admissible,
                "failed_checks_json": json.dumps(decision.failed_checks),
                "evaluated_at_ns": decision.evaluated_at_ns,
                "seq": 0,
                "source": source,
                "source_profile_hash": source_registry.registry_hash,
            }
        )
    rows.sort(key=lambda row: (row["t_known_ns"], row["slug"]))
    for seq, row in enumerate(rows):
        row["seq"] = seq
    diagnostics = {
        "windows": len(rows),
        "admissible": admissible_count,
        "excluded": len(rows) - admissible_count,
        "failure_counts": dict(sorted(failure_counts.items())),
        "rule_id": rule.id,
        "rule_hash": rule.spec_hash,
        "source_profile_hash": source_registry.registry_hash,
    }
    return rows, diagnostics


def _load_twap_neighbors(
    *,
    output_root: Path,
    day: date,
    coin: str,
    allow_partial: bool,
) -> tuple[list[dict[str, Any]], list[InputSnapshot], bool]:
    rows: list[dict[str, Any]] = []
    inputs: list[InputSnapshot] = []
    incomplete = False
    for selected_day in (day - timedelta(days=1), day, day + timedelta(days=1)):
        partition = _partition_dir(output_root, "twap", selected_day, coin)
        manifest_path = partition / "manifest.json"
        if not manifest_path.exists():
            if allow_partial:
                incomplete = True
                continue
            raise FileNotFoundError(
                f"coverage requires TWAP neighbor partition {partition}"
            )
        manifest = json.loads(manifest_path.read_text())
        source_partial = bool(manifest.get("partial"))
        if source_partial and not allow_partial:
            raise RuntimeError(f"coverage refuses partial TWAP source {partition}")
        incomplete |= source_partial
        table = read_partition(
            output_root,
            "twap",
            selected_day,
            coin,
            allow_partial=allow_partial,
        )
        rows.extend(table.to_pylist())
        inputs.append(snapshot_file(manifest_path))
    rows.sort(key=lambda row: (row["t_known_ns"], row["seq"]))
    return rows, inputs, incomplete


def build_coverage_partition(
    *,
    day: date,
    coin: str,
    output_root: Path,
    partial_requested: bool,
) -> dict[str, Any]:
    twap_rows, twap_inputs, incomplete_sources = _load_twap_neighbors(
        output_root=output_root,
        day=day,
        coin=coin,
        allow_partial=partial_requested,
    )
    markets, _, metadata_inputs = load_market_metadata()
    price_ledger = CollectorLedger.load(
        PM / "prices/collector_gaps.jsonl", subject_field="topic"
    )
    rule = load_rule()
    source_registry = load_source_registry()
    rows, diagnostics = normalize_coverage(
        day=day,
        coin=coin,
        markets=markets,
        twap_rows=twap_rows,
        price_ledger=price_ledger,
        rule=rule,
        source_registry=source_registry,
    )
    partial = partial_requested or incomplete_sources
    _assert_closed_day(day, partial)
    if not rows and not partial:
        raise RuntimeError(f"non-partial coverage partition is empty for {day} {coin}")
    return write_partition(
        output_root,
        "coverage",
        day,
        coin,
        rows,
        twap_inputs
        + metadata_inputs[:1]
        + [
            price_ledger.snapshot,
            snapshot_file(rule.path),
            snapshot_file(source_registry.path),
        ],
        partial=partial,
        diagnostics=diagnostics,
    )


def _wire_line(recv_ns: int, body: Any) -> str:
    return f"{recv_ns}\t{json.dumps(body, separators=(',', ':'))}\n"


def selftest() -> None:
    with tempfile.TemporaryDirectory(prefix="pm-tier1-test-") as tmp:
        root = Path(tmp)
        ledger_path = root / "gaps.jsonl"
        ledger_raw = (
            json.dumps(
                {
                    "recv_ns": 1_000_000_000,
                    "collector_version": "test_v1",
                    "event": "collector_start",
                    "pid": 1,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "recv_ns": 1_500_000_000,
                    "collector_version": "test_v1",
                    "event": "gap_open",
                    "slug": "btc-updown-5m-1",
                    "cause": "SLOW_CONSUMER_1013",
                    "gap_start_ns": 1_400_000_000,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "recv_ns": 1_700_000_000,
                    "collector_version": "test_v1",
                    "event": "gap_closed",
                    "slug": "btc-updown-5m-1",
                    "cause": "SLOW_CONSUMER_1013",
                    "gap_start_ns": 1_400_000_000,
                    "gap_end_ns": 1_600_000_000,
                }
            )
            + "\n"
        ).encode()
        ledger_path.write_bytes(ledger_raw)
        ledger = CollectorLedger.load(ledger_path, subject_field="slug")
        assert ledger.version_at(1_200_000_000) == ("test_v1", "LEDGER_ACTIVE")
        assert len(ledger.overlaps("btc-updown-5m-1", 1_300_000_000, 1_500_000_000)) == 1
        print("  PASS  collector eras and closed gaps canonicalise")

        market = MarketInfo(
            slug="btc-updown-5m-1",
            coin="btc",
            window_start_s=1,
            window_end_s=301,
            condition_id="condition",
            up_asset="UP",
            down_asset="DOWN",
            market_known_ns=1_100_000_000,
            source_file_id="market",
        )
        raw_path = root / "btc-updown-5m-1.jsonl.gz"
        book = [
            {
                "asset_id": "UP",
                "timestamp": "1000",
                "hash": "u0",
                "bids": [{"price": "0.49", "size": "10"}],
                "asks": [{"price": "0.51", "size": "12"}],
                "tick_size": "0.01",
                "event_type": "book",
            },
            {
                "asset_id": "DOWN",
                "timestamp": "1000",
                "hash": "d0",
                "bids": [{"price": "0.49", "size": "12"}],
                "asks": [{"price": "0.51", "size": "10"}],
                "tick_size": "0.01",
                "event_type": "book",
            },
        ]
        change = {
            "timestamp": "1100",
            "event_type": "price_change",
            "price_changes": [
                {
                    "asset_id": "UP",
                    "price": "0.50",
                    "size": "8",
                    "side": "BUY",
                    "hash": "u1",
                    "best_bid": "0.50",
                    "best_ask": "0.51",
                },
                {
                    "asset_id": "DOWN",
                    "price": "0.50",
                    "size": "8",
                    "side": "SELL",
                    "hash": "d1",
                    "best_bid": "0.49",
                    "best_ask": "0.50",
                },
            ],
        }
        cross_change = {
            "timestamp": "1150",
            "event_type": "price_change",
            "price_changes": [
                {
                    "asset_id": "DOWN",
                    "price": "0.50",
                    "size": "9",
                    "side": "BUY",
                    "hash": "d2",
                    "best_bid": "0.50",
                    "best_ask": "0.51",
                },
                {
                    "asset_id": "UP",
                    "price": "0.50",
                    "size": "9",
                    "side": "SELL",
                    "hash": "u2",
                    "best_bid": "0.49",
                    "best_ask": "0.50",
                },
            ],
        }
        trade = {
            "asset_id": "DOWN",
            "price": "0.40",
            "size": "5",
            "fee_rate_bps": "0",
            "side": "BUY",
            "timestamp": "1200",
            "event_type": "last_trade_price",
            "transaction_hash": "tx1",
        }
        sibling_trade = {
            **trade,
            "asset_id": "UP",
            "price": "0.60",
            "size": "2",
            "side": "SELL",
            "transaction_hash": "tx2",
        }
        with gzip.open(raw_path, "wt") as handle:
            handle.write(_wire_line(1_010_000_000, book))
            handle.write(_wire_line(1_110_000_000, change))
            handle.write(_wire_line(1_160_000_000, cross_change))
            handle.write(_wire_line(1_210_000_000, trade))
            handle.write(_wire_line(1_215_000_000, sibling_trade))
            handle.write(_wire_line(1_220_000_000, trade))
        quotes, trades, stats, stopped = normalize_clob(
            [raw_path],
            day=date(1970, 1, 1),
            coin="btc",
            markets={market.slug: market},
            ledger=ledger,
        )
        assert not stopped and len(quotes) == 3 and len(trades) == 1
        assert quotes[-1]["bid_up"] == 0.49 and quotes[-1]["pair_consistent"]
        assert trades[0]["token_side"] == "MIXED"
        assert trades[0]["price_token"] is None
        assert trades[0]["price_up"] == 0.60 and trades[0]["q_up"] == -1
        assert trades[0]["size"] == 7 and trades[0]["constituent_count"] == 2
        assert json.loads(trades[0]["transaction_hashes_json"]) == ["tx1", "tx2"]
        assert stats.duplicate_rows == 1
        assert stats.collapse_groups == stats.collapsed_rows == 1
        assert stats.top_checks == 4 and stats.top_matches == 2
        assert stats.reconciled_levels == 2 and stats.invalidated_books == 0
        print("  PASS  CLOB replay, top reconciliation, dedup and parent collapse")

        price_ledger = CollectorLedger(root / "empty-price.jsonl", b"", subject_field="topic")
        resolution = ResolutionInfo(
            slug=market.slug,
            resolution_known_ns=400_000_000_000,
            winner_up=True,
        )
        window_rows = normalize_windows(
            day=date(1970, 1, 1),
            coin="btc",
            markets={market.slug: market},
            resolutions={market.slug: resolution},
            clob_ledger=ledger,
            price_ledger=price_ledger,
        )
        assert len(window_rows) == 1
        assert window_rows[0]["t_known_ns"] == resolution.resolution_known_ns
        assert window_rows[0]["clob_gap_count"] == 1
        assert window_rows[0]["clob_gap_ms"] == 200
        assert window_rows[0]["clob_slow_consumer_gap"]
        assert window_rows[0]["gap_rule_status"] == "FACTS_ONLY_RULE_SEPARATE"
        print("  PASS  windows preserve resolution knowledge and gap causes")

        coverage_market = MarketInfo(
            slug="btc-updown-5m-1000",
            coin="btc",
            window_start_s=1000,
            window_end_s=1300,
            condition_id="coverage-condition",
            up_asset="UP2",
            down_asset="DOWN2",
            market_known_ns=990_000_000_000,
            source_file_id="market",
        )
        grid_rows = []
        target_start_ms = coverage_market.window_start_s * 1000 - 5000
        for index in range(310):
            event_ms = target_start_ms + (index + 1) * 1000
            grid_rows.append(
                {
                    "symbol": "btc/usd",
                    "window_s": 60,
                    "t_event_ms": event_ms,
                    "t_known_ns": event_ms * 1_000_000 + 10_000_000,
                    "duplicate_count": 0,
                }
            )
        coverage_rows, coverage_diagnostics = normalize_coverage(
            day=date(1970, 1, 1),
            coin="btc",
            markets={coverage_market.slug: coverage_market},
            twap_rows=grid_rows,
            price_ledger=price_ledger,
            rule=load_rule(),
            source_registry=load_source_registry(),
        )
        assert len(coverage_rows) == 1 and coverage_rows[0]["admissible"]
        assert coverage_rows[0]["observed_n"] == 310
        assert coverage_diagnostics["admissible"] == 1
        print("  PASS  Tier-1 coverage row is hash-bound and admissible")

        tail_missing_rows, _ = normalize_coverage(
            day=date(1970, 1, 1),
            coin="btc",
            markets={coverage_market.slug: coverage_market},
            twap_rows=grid_rows[:250],
            price_ledger=price_ledger,
            rule=load_rule(),
            source_registry=load_source_registry(),
        )
        assert tail_missing_rows[0]["t_known_ns"] >= tail_missing_rows[0][
            "target_end_ns"
        ]
        print("  PASS  missing-tail coverage is not known before target end")

        twap_path = root / "twap.csv.gz"
        twap_body = {
            "payload": {
                "symbol": "btc/usd",
                "timestamp": 1000,
                "value": 100.0,
                "window_s": 60,
                "full_accuracy_value": "100000000000000000000",
            },
            "timestamp": 1050,
            "topic": "crypto_prices_twap_sixty",
        }
        with gzip.open(twap_path, "wt") as handle:
            handle.write(_wire_line(1_060_000_000, twap_body))
            handle.write(_wire_line(1_070_000_000, twap_body))
        twap_rows, twap_stats, _ = normalize_twap(
            [twap_path],
            day=date(1970, 1, 1),
            coin="btc",
            ledger=price_ledger,
        )
        assert len(twap_rows) == 1 and twap_stats.duplicate_rows == 1
        assert twap_rows[0]["t_event_ms"] == 1000
        assert twap_rows[0]["t_publish_ms"] == 1050
        print("  PASS  TWAP uses payload event time and earliest knowledge dedupe")

        inputs = [snapshot_file(twap_path)]
        manifest = write_partition(
            root / "tier1",
            "twap",
            date(1970, 1, 1),
            "btc",
            twap_rows,
            inputs,
            partial=True,
            diagnostics=asdict(twap_stats),
        )
        repeated = write_partition(
            root / "tier1",
            "twap",
            date(1970, 1, 1),
            "btc",
            twap_rows,
            inputs,
            partial=True,
            diagnostics=asdict(twap_stats),
        )
        assert manifest["manifest_hash"] == repeated["manifest_hash"]
        try:
            read_partition(root / "tier1", "twap", date(1970, 1, 1), "btc")
        except RuntimeError as exc:
            assert "refuses partial" in str(exc)
        else:
            raise AssertionError("partial partition was not refused")
        table = read_partition(
            root / "tier1",
            "twap",
            date(1970, 1, 1),
            "btc",
            allow_partial=True,
        )
        assert table.num_rows == 1
        print("  PASS  atomic manifest is idempotent and consumers refuse partial")

        changed_inputs = [
            InputSnapshot(inputs[0].path, inputs[0].size + 1, inputs[0].sha256)
        ]
        try:
            write_partition(
                root / "tier1",
                "twap",
                date(1970, 1, 1),
                "btc",
                twap_rows,
                changed_inputs,
                partial=True,
                diagnostics=asdict(twap_stats),
            )
        except RuntimeError as exc:
            assert "merge-never-overwrite" in str(exc)
        else:
            raise AssertionError("input-manifest mismatch did not fail")
        print("  PASS  changed inputs cannot overwrite an existing partition")

    # --- R-12: two mechanisms, two rules -----------------------------------
    pc_base = {
        "event_type": "price_change",
        "market": "0xm",
        "timestamp": "1000",
        "price_changes": [
            {"asset_id": "U", "hash": "h", "price": "0.74", "size": "0",
             "side": "BUY", "best_bid": "0.74", "best_ask": "0.84"},
        ],
    }
    pc_moved = json.loads(json.dumps(pc_base))
    pc_moved["price_changes"][0]["best_bid"] = "0.73"
    assert _raw_message_key(pc_base) != _raw_message_key(pc_moved)
    print("  PASS  R-12: same timestamp, moved top-of-book => DISTINCT keys, both kept")
    assert _raw_message_key(pc_base) == _raw_message_key(json.loads(json.dumps(pc_base)))
    print("  PASS  R-12: a true price_change re-delivery still collapses to one key")

    book_base = {
        "event_type": "book", "asset_id": "A", "hash": "h1", "timestamp": "1000",
        "bids": [{"price": "0.5", "size": "1"}], "asks": [],
    }
    book_envelope = dict(book_base, tick_size="0.01", last_trade_price="0.5")
    assert _raw_message_key(book_base) == _raw_message_key(book_envelope)
    assert _identity_digest(book_base) == _identity_digest(book_envelope)
    print("  PASS  R-12: book envelope fields are outside identity, so they collapse")
    book_real = dict(book_base, bids=[{"price": "0.9", "size": "1"}])
    assert _identity_digest(book_base) != _identity_digest(book_real)
    print("  PASS  R-12: a bids disagreement under one venue hash still conflicts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--dataset", choices=("twap", "clob", "windows", "coverage", "all")
    )
    parser.add_argument("--day", help="UTC day YYYY-MM-DD")
    parser.add_argument("--coin", choices=tuple(COIN_SYMBOL))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--partial",
        action="store_true",
        help="stamp output REFUSE_PARTIAL; required for truncated/open-day smoke runs",
    )
    args = parser.parse_args()
    if args.selftest:
        selftest()
    if args.dataset:
        if not args.day or not args.coin:
            parser.error("--dataset requires --day and --coin")
        selected_day = _day(args.day)
        manifests: list[dict[str, Any]] = []
        if args.dataset in ("twap", "all"):
            manifests.append(
                build_twap_partition(
                    day=selected_day,
                    coin=args.coin,
                    output_root=args.output_root,
                    max_files=args.max_files,
                    max_records=args.max_records,
                    partial_requested=args.partial,
                )
            )
        if args.dataset in ("clob", "all"):
            manifests.extend(
                build_clob_partitions(
                    day=selected_day,
                    coin=args.coin,
                    output_root=args.output_root,
                    max_files=args.max_files,
                    max_records=args.max_records,
                    partial_requested=args.partial,
                )
            )
        if args.dataset in ("windows", "all"):
            manifests.append(
                build_windows_partition(
                    day=selected_day,
                    coin=args.coin,
                    output_root=args.output_root,
                    partial_requested=args.partial,
                )
            )
        if args.dataset in ("coverage", "all"):
            manifests.append(
                build_coverage_partition(
                    day=selected_day,
                    coin=args.coin,
                    output_root=args.output_root,
                    partial_requested=args.partial,
                )
            )
        for manifest in manifests:
            print(
                f"{manifest['dataset']} day={manifest['day']} coin={manifest['coin']} "
                f"rows={manifest['rows']} partial={manifest['partial']} "
                f"source={manifest['source_digest'][:12]}"
            )
    if not args.selftest and not args.dataset:
        parser.error("choose --selftest and/or --dataset")


if __name__ == "__main__":
    main()
