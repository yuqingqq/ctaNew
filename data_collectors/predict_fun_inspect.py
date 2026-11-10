"""Inspect a finalized recorder run; engineering diagnostics, no P&L or admission.

python3 -m data_collectors.predict_fun_inspect data/predict_fun/runs/<run_id>
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


from data_collectors import predict_fun_recorder as recorder
from data_collectors.predict_fun_recorder import book_diagnostics, parse_predict_frame, parse_rest_envelope, utc_timestamp


def inspect_run(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    if summary["run_id"] != manifest["run_id"] or summary.get("writer_error"):
        raise ValueError("run identity or writer failure")
    if hashlib.sha256((path / "collector_source.py").read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("recorder source identity mismatch")
    sources, events, http_statuses, topics, failures = Counter(), Counter(), Counter(), Counter(), Counter()
    market_ids, sequence, acked, requested = set(), set(), set(), set()
    first, last = {}, {}
    match_rows, second_aligned, timestamp_missing = 0, 0, 0
    match_transactions = set()
    stream_counts = Counter()
    book_frames = defaultdict(int)
    depth_checks = Counter()
    protocol_checks = Counter()
    input_hashes = {}
    for file in sorted(path.glob("*/*.jsonl.gz")):
        input_hashes[str(file.relative_to(path))] = hashlib.sha256(file.read_bytes()).hexdigest()
        for line in gzip.open(file, "rt"):
            row = json.loads(line)
            if row["run_id"] != manifest["run_id"] or type(row["seq"]) is not int or row["seq"] in sequence:
                raise ValueError("record identity or duplicate sequence")
            sequence.add(row["seq"])
            source = row["source"]
            sources[source] += 1
            first[source] = min(first.get(source, row["recv_ns"]), row["recv_ns"])
            last[source] = max(last.get(source, row["recv_ns"]), row["recv_ns"])
            if source == "events":
                events[row["kind"]] += 1
                if row["kind"] == "MARKET_DISCOVERED":
                    market_ids.add(row["market_id"])
                if row["kind"] == "WS_REQUEST":
                    requested.add((row["connection_id"], row["request_id"]))
                if row["kind"] in ("WS_FAILURE", "REST_CYCLE_FAILED", "MATCH_POLL_INCOMPLETE", "CATEGORY_REFUSED", "REST_RATE_LIMIT", "CATEGORY_FETCH_FAILED", "SNAPSHOT_FETCH_FAILED", "REST_UNAVAILABLE", "BOOK_OBSERVATION_REFUSED"):
                    failures["|".join(str(row.get(k, "")) for k in ("kind", "feed", "error", "reason", "status"))] += 1
            elif source == "rest":
                http_statuses[str(row["status"])] += 1
                if row["status"] != 200:
                    continue
                try:
                    payload = parse_rest_envelope(row["raw"])
                except (ValueError, TypeError):
                    protocol_checks["malformed_rest_envelopes"] += 1
                    continue
                if row["path"].endswith("/orderbook"):
                    protocol_checks["rest_books_checked"] += 1
                    try:
                        book_diagnostics(payload.get("data"), int(row["path"].split("/")[-2]))
                    except (ValueError, TypeError, OverflowError):
                        protocol_checks["malformed_rest_books"] += 1
                elif row["path"] == "/v1/orders/matches":
                    rows = payload.get("data")
                    if not isinstance(rows, list):
                        protocol_checks["malformed_match_pages"] += 1
                        continue
                    for match in rows:
                        match_rows += 1
                        if not isinstance(match, dict):
                            protocol_checks["malformed_match_rows"] += 1
                            timestamp_missing += 1
                            continue
                        market = match.get("market")
                        expected_id = (row.get("params") or {}).get("marketId")
                        valid_market = (isinstance(market, dict) and type(market.get("id")) is int
                                        and str(market["id"]) == str(expected_id))
                        if not valid_market:
                            protocol_checks["malformed_match_rows"] += 1
                        transaction = match.get("transactionHash")
                        if valid_market and isinstance(transaction, str) and transaction:
                            match_transactions.add((market["id"], transaction))
                        try:
                            ts = utc_timestamp(match.get("executedAt"))
                            second_aligned += ts % 1 == 0
                        except (TypeError, ValueError, OverflowError):
                            timestamp_missing += 1
            elif source == "predict_ws":
                try:
                    frame = parse_predict_frame(row["raw"])
                except (ValueError, TypeError):
                    protocol_checks["malformed_predict_frames"] += 1
                    continue
                topic = frame["topic"] if frame["type"] == "M" else "response"
                topics[topic] += 1
                if frame.get("type") == "R" and frame.get("success") is True:
                    acked.add((row["connection_id"], frame["requestId"]))
                if topic.startswith("predictOrderbook/"):
                    book_frames[topic.split("/")[1]] += 1
                    depth_checks["n_book_frames_checked"] += 1
                    try:
                        book = book_diagnostics(frame.get("data"), int(topic.split("/")[1]))
                    except (ValueError, TypeError, OverflowError):
                        depth_checks["malformed_book_frames"] += 1
                    else:
                        depth_checks["two_sided_book_frames"] += book["two_sided"]
                        depth_checks["empty_book_frames"] += book["empty"]
                        depth_checks["locked_or_crossed_book_frames"] += book["locked_or_crossed"]
            elif source in ("binance_spot", "binance_futures"):
                frame = json.loads(row["raw"])
                stream_counts[source + "/" + frame.get("stream", "UNKNOWN")] += 1
    expected = summary["enqueued"]
    if summary["persisted"] != expected or len(sequence) != expected or sequence != set(range(expected)):
        raise ValueError("record population does not reconcile with summary")
    return {
        "purpose": "ENGINEERING_SMOKE_ONLY",
        "data_quality_certified": False,
        "run_id": manifest["run_id"], "network": manifest.get("network"),
        "recorder_schema": manifest.get("schema"),
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "started_recv_ns": manifest["started"]["recv_ns"], "ended_recv_ns": summary["ended"]["recv_ns"],
        "recorder_source_sha256": manifest["source_sha256"],
        "inspector_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "book_validator_source_sha256": hashlib.sha256(Path(recorder.__file__).read_bytes()).hexdigest(),
        "input_file_sha256": input_hashes,
        "run_status": summary["status"], "n_records": expected,
        "source_counts": dict(sources), "source_first_recv_ns": first, "source_last_recv_ns": last,
        "n_discovered_markets": len(market_ids), "market_ids": sorted(market_ids),
        "http_status_counts": dict(http_statuses), "event_counts": dict(events),
        "failure_counts": dict(failures), "predict_topic_counts": dict(topics),
        "protocol_diagnostics": {name: protocol_checks[name] for name in (
            "malformed_predict_frames", "malformed_rest_envelopes", "malformed_match_pages",
            "malformed_match_rows", "rest_books_checked", "malformed_rest_books")},
        "n_subscription_requests": len(requested), "n_subscription_acks": len(acked),
        "n_requests_without_ack": len(requested - acked),
        "book_frame_counts_by_market": dict(book_frames), "binance_stream_counts": dict(stream_counts),
        "book_diagnostics": {name: depth_checks[name] for name in ["n_book_frames_checked", "malformed_book_frames", "two_sided_book_frames", "empty_book_frames", "locked_or_crossed_book_frames"]},
        "match_raw_rows_including_overlap": match_rows,
        "n_distinct_market_transaction_pairs": len(match_transactions),
        "match_rows_with_second_aligned_executedAt": second_aligned,
        "match_rows_with_unreadable_executedAt": timestamp_missing,
        "timing_note": "Second-aligned timestamps do not identify within-second order or establish matching-engine time.",
        "population_note": "Overlapping pages repeat executions; transaction pairs are not independent fills or actions. Counts are not strategy results.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, help="exclusive-create JSON receipt")
    args = parser.parse_args()
    receipt = inspect_run(args.run)
    text = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x") as handle:
            handle.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
