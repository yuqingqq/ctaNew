"""Corpus-wide census of the duplicate-identity conflict that aborts normalize_clob.

`tier1_pipeline.normalize_clob` dedups raw CLOB messages by
``_raw_message_key`` and raises when two records share an identity but carry
different payloads.  One such key aborted the evaluation lane.  This probe
answers the population question the abort cannot: how many `(day, coin, slug)`
keys are affected across the whole raw tape, which payload fields disagree, and
which collector era wrote each copy.

It is a COVERAGE FACT producer.  It selects nothing, excludes nothing and
writes no admissibility rule -- that decision is coordinator-gated (R-ADMISS).
It never writes under ``data/pm_5min/tier1/``.

Run the pure checks with::

    python3 -m live.pm_research.da_duplicate_identity_scan --selftest

Run the census with::

    python3 -m live.pm_research.da_duplicate_identity_scan --scan

Then anatomize what the census found -- the census counts conflicts, the
anatomy says what KIND each one is::

    python3 -m live.pm_research.da_duplicate_identity_scan --anatomy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from live.pm_research import tier1_pipeline as tp


REPO = Path(__file__).resolve().parents[2]
PM = REPO / "data/pm_5min"
RAW = PM / "raw"
DERIVED = PM / "derived"
RECEIPT = DERIVED / "da_duplicate_identity_v1.json"

_DAY_DIR = re.compile(r"^\d{8}$")
_SLUG = re.compile(r"^(?P<coin>[a-z0-9]+)-updown-5m-(?P<start>\d+)$")


def source_days() -> list[str]:
    """Day list DERIVED from disk.  Never hardcode -- DAYS went stale 4x/3d."""
    return sorted(p.name for p in RAW.iterdir() if p.is_dir() and _DAY_DIR.match(p.name))


def _slug_of(path: Path) -> str:
    return tp._slug_from_path(path)


def discover_slugs(day: str) -> dict[str, list[Path]]:
    """(slug -> shard paths), grouped exactly as normalize_clob groups them."""
    root = RAW / day
    by_slug: dict[str, list[Path]] = defaultdict(list)
    for path in root.glob("*-updown-5m-*.jsonl*.gz"):
        by_slug[_slug_of(path)].append(path)
    return {
        slug: sorted(paths, key=tp._shard_order)
        for slug, paths in by_slug.items()
    }


def _digest(message: Mapping[str, Any]) -> str:
    """The pipeline's OWN identity digest, so the census measures what
    normalize_clob measures.  After R-12 that means a `book` snapshot digests
    without its optional envelope fields, matching the collapse rule."""
    return tp._identity_digest(message)


@dataclass(frozen=True, slots=True)
class Occurrence:
    recv_ns: int
    source_seq: int
    shard: str
    digest: str


def _iter_slug(shards: Sequence[Path]) -> Iterable[tuple[Path, int, int, dict[str, Any]]]:
    stats = tp.ParseStats()
    for path in shards:
        for recv_ns, source_seq, _source_id, message in tp._iter_wire_file(
            path, stats, source_file_id="scan"
        ):
            yield path, recv_ns, source_seq, message


def scan_slug(day: str, slug: str, shards: Sequence[Path]) -> dict[str, Any]:
    """First pass: detect conflicting keys.  Second pass: recover both payloads.

    Two passes rather than one because holding every message would cost ~1 GB
    per busy BTC window; conflicts are rare, so the second pass is cheap.
    """
    seen: dict[tuple[Any, ...], Occurrence] = {}
    conflicts: dict[tuple[Any, ...], list[Occurrence]] = {}
    records = 0
    exact_duplicates = 0

    for path, recv_ns, source_seq, message in _iter_slug(shards):
        records += 1
        key = tp._raw_message_key(message)
        digest = _digest(message)
        first = seen.get(key)
        if first is None:
            seen[key] = Occurrence(recv_ns, source_seq, path.name, digest)
            continue
        if first.digest == digest:
            exact_duplicates += 1
            continue
        bucket = conflicts.setdefault(key, [first])
        bucket.append(Occurrence(recv_ns, source_seq, path.name, digest))

    detail: list[dict[str, Any]] = []
    if conflicts:
        wanted = set(conflicts)
        payloads: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
        for _path, _recv_ns, _seq, message in _iter_slug(shards):
            key = tp._raw_message_key(message)
            if key in wanted:
                payloads[key][_digest(message)] = message
        for key, occurrences in conflicts.items():
            detail.append(
                _describe_conflict(day, slug, key, occurrences, payloads[key])
            )

    return {
        "day": day,
        "slug": slug,
        "coin": (_SLUG.match(slug).group("coin") if _SLUG.match(slug) else None),
        "shards": [p.name for p in shards],
        "records": records,
        "unique_keys": len(seen),
        "exact_duplicate_records": exact_duplicates,
        "conflicting_keys": len(conflicts),
        "conflicts": detail,
    }


def _describe_conflict(
    day: str,
    slug: str,
    key: tuple[Any, ...],
    occurrences: Sequence[Occurrence],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    variants = sorted({occ.digest for occ in occurrences})
    field_sets = {d: set(payloads[d]) for d in variants if d in payloads}
    shared = set.intersection(*field_sets.values()) if field_sets else set()
    union = set.union(*field_sets.values()) if field_sets else set()
    only_in_some = sorted(union - shared)
    differing_shared = sorted(
        field
        for field in shared
        if len({tp._canonical_json(payloads[d][field]) for d in variants if d in payloads}) > 1
    )
    return {
        "day": day,
        "slug": slug,
        "event_type": key[0] if key else None,
        "key": [str(part) for part in key],
        "n_occurrences": len(occurrences),
        "n_variants": len(variants),
        "occurrences": [
            {
                "recv_ns": occ.recv_ns,
                "source_seq": occ.source_seq,
                "shard": occ.shard,
                "digest12": occ.digest[:12],
            }
            for occ in occurrences
        ],
        "fields_present_in_some_only": only_in_some,
        "shared_fields_that_differ": differing_shared,
        "field_values_present_in_some_only": {
            field: {
                d[:12]: payloads[d].get(field, "<ABSENT>")
                for d in variants
                if d in payloads
            }
            for field in only_in_some
        },
    }


def _scan_one(task: tuple[str, str, list[str]]) -> dict[str, Any]:
    day, slug, shard_names = task
    return scan_slug(day, slug, [RAW / day / name for name in shard_names])


def scan_corpus(days: Sequence[str], workers: int = 14) -> dict[str, Any]:
    tasks: list[tuple[str, str, list[str]]] = []
    for day in days:
        for slug, shards in sorted(discover_slugs(day).items()):
            tasks.append((day, slug, [p.name for p in shards]))

    per_slug: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_scan_one, tasks, chunksize=4):
            per_slug.append(result)

    affected = [row for row in per_slug if row["conflicting_keys"]]
    by_coin: dict[str, int] = defaultdict(int)
    by_day: dict[str, int] = defaultdict(int)
    by_event: dict[str, int] = defaultdict(int)
    for row in affected:
        by_coin[row["coin"]] += row["conflicting_keys"]
        by_day[row["day"]] += row["conflicting_keys"]
        for conflict in row["conflicts"]:
            by_event[str(conflict["event_type"])] += 1

    return {
        "probe": "da_duplicate_identity_v1",
        "claim_status": "COVERAGE_FACTS_ONLY",
        "source_days": list(days),
        "slugs_scanned": len(per_slug),
        "records_scanned": sum(row["records"] for row in per_slug),
        "exact_duplicate_records": sum(
            row["exact_duplicate_records"] for row in per_slug
        ),
        "slugs_with_conflicts": len(affected),
        "conflicting_keys_total": sum(row["conflicting_keys"] for row in affected),
        "conflicting_keys_by_coin": dict(sorted(by_coin.items())),
        "conflicting_keys_by_day": dict(sorted(by_day.items())),
        "conflicting_keys_by_event_type": dict(sorted(by_event.items())),
        "affected": affected,
    }


ANATOMY_RECEIPT = DERIVED / "da_duplicate_identity_anatomy_v1.json"


def anatomize_pair(
    day: str,
    slug: str,
    first: tuple[int, int, Mapping[str, Any]],
    second: tuple[int, int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Say what KIND of conflict a pair is, without deciding what to do about it.

    Two mechanisms exist on this tape and they are not the same finding:
    a re-delivered `book` whose envelope differs is ONE state seen twice; two
    `price_change` messages whose resulting top-of-book differs are TWO states
    that happen to share a venue timestamp.
    """
    recv_a, seq_a, msg_a = first
    recv_b, seq_b, msg_b = second
    event_type = str(msg_a.get("event_type"))
    record: dict[str, Any] = {
        "day": day,
        "slug": slug,
        "event_type": event_type,
        "dt_us": (recv_b - recv_a) / 1e3,
        "same_recv_millisecond": (recv_a // 1_000_000) == (recv_b // 1_000_000),
        "wire_line_gap": (seq_b >> 16) - (seq_a >> 16),
        "same_market": msg_a.get("market") == msg_b.get("market"),
    }
    if event_type == "price_change":
        rows_a = msg_a.get("price_changes") or []
        rows_b = msg_b.get("price_changes") or []
        differing: set[str] = set()
        deltas: list[float] = []
        for row_a, row_b in zip(rows_a, rows_b):
            for field in set(row_a) | set(row_b):
                if row_a.get(field) != row_b.get(field):
                    differing.add(field)
                    if field in ("best_bid", "best_ask"):
                        try:
                            deltas.append(
                                round(abs(float(row_a[field]) - float(row_b[field])), 4)
                            )
                        except (TypeError, ValueError, KeyError):
                            pass
        record["row_count_equal"] = len(rows_a) == len(rows_b)
        record["differing_row_fields"] = sorted(differing)
        record["top_of_book_deltas"] = deltas
        record["one_book_sums"] = _one_book_sums(msg_a) + _one_book_sums(msg_b)
    else:
        fields_a, fields_b = set(msg_a), set(msg_b)
        record["fields_in_one_copy_only"] = sorted(fields_a ^ fields_b)
        record["shared_fields_that_differ"] = sorted(
            field for field in fields_a & fields_b if msg_a[field] != msg_b[field]
        )
    return record


def _one_book_sums(message: Mapping[str, Any]) -> list[float]:
    """bid(Up) + ask(Down) for a two-row price_change; 1.0 iff internally consistent."""
    rows = message.get("price_changes") or []
    if len(rows) != 2:
        return []
    try:
        return [
            round(float(rows[0]["best_bid"]) + float(rows[1]["best_ask"]), 4),
            round(float(rows[1]["best_bid"]) + float(rows[0]["best_ask"]), 4),
        ]
    except (TypeError, ValueError, KeyError):
        return []


def _anatomize_slug(task: tuple[str, str, list[str], list[list[str]]]) -> list[dict[str, Any]]:
    day, slug, shard_names, keys = task
    wanted = {tuple(key) for key in keys}
    variants: dict[tuple[str, ...], dict[str, tuple[int, int, dict[str, Any]]]] = defaultdict(dict)
    stats = tp.ParseStats()
    for name in shard_names:
        for recv_ns, source_seq, _sid, message in tp._iter_wire_file(
            RAW / day / name, stats, source_file_id="anatomy"
        ):
            key = tuple(str(part) for part in tp._raw_message_key(message))
            if key in wanted:
                variants[key][_digest(message)] = (recv_ns, source_seq, message)
    out: list[dict[str, Any]] = []
    for key, copies in variants.items():
        ordered = sorted(copies.values(), key=lambda t: (t[0], t[1]))
        if len(ordered) < 2:
            continue
        out.append(anatomize_pair(day, slug, ordered[0], ordered[1]))
    return out


def anatomize_corpus(census: Mapping[str, Any], workers: int = 14) -> dict[str, Any]:
    tasks = [
        (row["day"], row["slug"], row["shards"], [c["key"] for c in row["conflicts"]])
        for row in census["affected"]
    ]
    pairs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_anatomize_slug, tasks, chunksize=4):
            pairs.extend(result)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_type[pair["event_type"]].append(pair)

    summary: dict[str, Any] = {}
    for event_type, rows in by_type.items():
        entry: dict[str, Any] = {
            "n": len(rows),
            "same_market": sum(row["same_market"] for row in rows),
            "same_recv_millisecond": sum(row["same_recv_millisecond"] for row in rows),
            "dt_us_p50": sorted(row["dt_us"] for row in rows)[len(rows) // 2],
            "dt_us_max": max(row["dt_us"] for row in rows),
        }
        if event_type == "price_change":
            fields: dict[str, int] = defaultdict(int)
            deltas: dict[str, int] = defaultdict(int)
            sums: dict[str, int] = defaultdict(int)
            for row in rows:
                fields[",".join(row["differing_row_fields"])] += 1
                for delta in row["top_of_book_deltas"]:
                    deltas[f"{delta:.2f}"] += 1
                for value in row["one_book_sums"]:
                    sums[f"{value:.4f}"] += 1
            entry["differing_row_fields"] = dict(fields)
            entry["top_of_book_delta_counts"] = dict(sorted(deltas.items()))
            entry["one_book_sum_counts"] = dict(sorted(sums.items()))
            entry["row_count_equal_all"] = all(row["row_count_equal"] for row in rows)
        else:
            only: dict[str, int] = defaultdict(int)
            shared: dict[str, int] = defaultdict(int)
            for row in rows:
                only[",".join(row["fields_in_one_copy_only"])] += 1
                shared[",".join(row["shared_fields_that_differ"]) or "(none)"] += 1
            entry["fields_in_one_copy_only"] = dict(only)
            entry["shared_fields_that_differ"] = dict(shared)
        summary[event_type] = entry

    return {
        "probe": "da_duplicate_identity_anatomy_v1",
        "claim_status": "COVERAGE_FACTS_ONLY",
        "census_source_days": census["source_days"],
        "pairs_recovered": len(pairs),
        "by_event_type": summary,
        "pairs": pairs,
    }


def _synthetic(messages: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
    return [tp._raw_message_key(m) for m in messages]


def selftest() -> None:
    checks = 0

    def ok(label: str, condition: bool) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(label)
        print(f"  PASS  {label}")

    book_a = {
        "event_type": "book",
        "asset_id": "A",
        "hash": "h1",
        "timestamp": "1000",
        "bids": [{"price": "0.5", "size": "1"}],
        "asks": [],
    }
    book_b = dict(book_a, tick_size="0.01", last_trade_price="0.5")
    keys = _synthetic([book_a, book_b])
    ok("optional-field variants share one book identity key", keys[0] == keys[1])
    ok(
        "R-12: and they now share an IDENTITY digest, so they collapse",
        _digest(book_a) == _digest(book_b),
    )
    ok(
        "...though their full payloads still differ, which is what is collapsed",
        tp._canonical_json(book_a) != tp._canonical_json(book_b),
    )
    book_real = dict(book_a, bids=[{"price": "0.9", "size": "1"}])
    ok(
        "R-12: a real bids/asks disagreement under one venue hash still conflicts",
        _digest(book_a) != _digest(book_real),
    )

    book_c = dict(book_a, bids=[{"price": "0.6", "size": "1"}])
    described_c = _describe_conflict(
        "20260820",
        "btc-updown-5m-1",
        keys[0],
        [
            Occurrence(1, 1, "s.jsonl.gz", _digest(book_a)),
            Occurrence(2, 2, "s.jsonl.gz", _digest(book_c)),
        ],
        {_digest(book_a): book_a, _digest(book_c): book_c},
    )
    ok(
        "a real book disagreement IS reported as a differing shared field",
        described_c["shared_fields_that_differ"] == ["bids"],
    )

    pc_a = {
        "event_type": "price_change",
        "market": "0xm",
        "timestamp": "1000",
        "price_changes": [
            {"asset_id": "U", "hash": "h", "price": "0.74", "size": "0",
             "side": "BUY", "best_bid": "0.74", "best_ask": "0.84"},
            {"asset_id": "D", "hash": "g", "price": "0.26", "size": "0",
             "side": "SELL", "best_bid": "0.16", "best_ask": "0.26"},
        ],
    }
    pc_b = json.loads(json.dumps(pc_a))
    pc_b["price_changes"][0]["best_bid"] = "0.73"
    pc_b["price_changes"][1]["best_ask"] = "0.27"
    ok(
        "R-12: two top-of-book states are now SEPARATE price_change keys",
        tp._raw_message_key(pc_a) != tp._raw_message_key(pc_b),
    )
    ok(
        "R-12: an identical re-delivery still collapses to one key",
        tp._raw_message_key(pc_a) == tp._raw_message_key(
            json.loads(json.dumps(pc_a))
        ),
    )
    pair = anatomize_pair("20260819", "bnb-updown-5m-1", (10, 1 << 16, pc_a), (20, 2 << 16, pc_b))
    ok(
        "the anatomy names top-of-book as the only disagreement",
        pair["differing_row_fields"] == ["best_ask", "best_bid"],
    )
    ok(
        "both copies are internally consistent with bid(Up)+ask(Down)=1",
        pair["one_book_sums"] == [1.0, 1.0, 1.0, 1.0],
    )
    ok("the tick distance is reported", pair["top_of_book_deltas"] == [0.01, 0.01])

    days = source_days()
    ok("source days are derived from disk, not hardcoded", len(days) > 0)
    print(f"\n{checks} checks passed; days on disk: {', '.join(days)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--anatomy", action="store_true")
    parser.add_argument("--day", action="append")
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if args.anatomy:
        if not RECEIPT.exists():
            parser.error(f"run --scan first: {RECEIPT} does not exist")
        census = json.loads(RECEIPT.read_text())
        report = anatomize_corpus(census, workers=args.workers)
        out = args.out or ANATOMY_RECEIPT
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(
            {k: v for k, v in report.items() if k != "pairs"}, indent=2, sort_keys=True
        ))
        print(f"\nreceipt: {out}")
        return
    if not args.scan:
        parser.error("nothing to do: pass --selftest, --scan or --anatomy")

    days = args.day or source_days()
    report = scan_corpus(days, workers=args.workers)
    out = args.out or RECEIPT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {k: v for k, v in report.items() if k != "affected"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nreceipt: {out}")


if __name__ == "__main__":
    main()
