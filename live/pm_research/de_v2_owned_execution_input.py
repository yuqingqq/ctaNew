"""Gate-1f offline owned-execution input contract for P003 v2.

This module never connects to a venue and never places, cancels or signs an
order.  It admits only a pre-existing external account export whose order,
acknowledgement, fill and exact maker-fee identities are complete and bound by
content hashes.  The fixed repository audit path is
``data/pm_5min/owned_execution/manifest.json``.

An absent or invalid export preserves the Gate-1 stop and emits no economic
statistic.  A valid export is only an acquisition seam; it does not
retroactively clear Gate 1 or authorise Gate 2.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import re
import resource
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path


PROTOCOL = "P003_V2_GATE1F_OWNED_EXECUTION_SOURCE_AUDIT_V1"
EXPORT_PROTOCOL = "P003_OWNED_EXECUTION_EXPORT_V1"
CANDIDATE_RELATIVE_PATH = "data/pm_5min/owned_execution/manifest.json"
SUPERSEDED_AUDIT_RELATIVE_PATH = (
    "data/pm_5min/derived/"
    "p003_v2_gate1f_owned_source_audit__20260905T054848Z.json")
SUPERSEDED_AUDIT_SHA256 = (
    "bf3d01fa61ee799860ec8bbc764645b0e034162f1611a54879b146d99b292022")
SOURCE_MODE = "OWNED_ACCOUNT_OFFLINE_EXPORT"
MIN_COMPLETE_DAYS = 5
MIN_ORDERS = 200
MIN_FILLS = 200
MAX_RECORDS_PER_FILE = 500_000
SIDES = ("BUY_UP", "SELL_UP")
ACK_STATUSES = ("ACCEPTED", "REJECTED")
TERMINAL_STATUSES = (
    "FILLED", "PARTIALLY_FILLED_CANCELLED", "CANCELLED", "EXPIRED",
    "REJECTED")
FEE_COMPONENT = "VENUE_MAKER_FEE_EXCLUDING_REBATES_REWARDS"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
DAY_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
ORDER_FIELDS = (
    "reference_generation_id", "policy_action_seq", "client_order_id",
    "venue_order_id", "slug", "asset_id", "maker_side", "decision_ns",
    "submitted_ns", "ack_ns", "terminal_ns", "requested_price",
    "requested_shares", "ack_status", "terminal_status", "utc_day")
FILL_FIELDS = (
    "reference_generation_id", "client_order_id", "venue_order_id",
    "fill_id", "trade_id", "slug", "asset_id", "maker_side", "fill_ns",
    "price", "shares", "liquidity_role", "maker_fee_amount",
    "fee_currency", "fee_rate_bps", "fee_schedule_id", "fee_component",
    "utc_day")
MANIFEST_FIELDS = (
    "protocol", "source_mode", "venue", "account_id_sha256",
    "producer_identity", "export_as_of_utc", "pipeline_commit",
    "freeze_at_utc", "policy_path", "policy_sha256", "fee_schedule_path",
    "fee_schedule_sha256", "ownership_evidence_path",
    "ownership_evidence_sha256", "orders_path", "orders_sha256",
    "n_orders", "fills_path", "fills_sha256", "n_fills",
    "complete_utc_days", "orders_export_complete", "acks_export_complete",
    "fills_export_complete", "fees_export_complete",
    "maker_role_venue_asserted", "no_live_code_in_research_repo")


class OwnedExecutionRefused(RuntimeError):
    """The external export does not meet the prospective input contract."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def _iso(value: object, field: str) -> datetime.datetime:
    if not isinstance(value, str):
        raise OwnedExecutionRefused(f"{field} must be an ISO UTC string")
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OwnedExecutionRefused(f"{field} is not valid ISO time") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != datetime.timedelta(0):
        raise OwnedExecutionRefused(f"{field} must be UTC")
    return parsed


def _day_from_ns(value: object, field: str) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise OwnedExecutionRefused(f"{field} must be a positive integer ns")
    return datetime.datetime.fromtimestamp(
        value / 1e9, datetime.timezone.utc).date().isoformat()


def _relative_file(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OwnedExecutionRefused(f"{field} must be a relative file path")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 1:
        raise OwnedExecutionRefused(
            f"{field} must name one file beside the manifest")
    path = base / rel
    if not path.is_file():
        raise OwnedExecutionRefused(f"{field} file is missing: {value}")
    return path


def _require_fields(row: dict, required: tuple[str, ...], label: str) -> None:
    missing = sorted(set(required) - set(row))
    if missing:
        raise OwnedExecutionRefused(f"{label} missing fields: {missing}")


def _jsonl(path: Path, expected_count: int, label: str):
    if not isinstance(expected_count, int) or isinstance(expected_count, bool) \
            or not 0 <= expected_count <= MAX_RECORDS_PER_FILE:
        raise OwnedExecutionRefused(
            f"{label} count must be within 0..{MAX_RECORDS_PER_FILE}")
    seen = 0
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                raise OwnedExecutionRefused(
                    f"{label} has blank line {line_no}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OwnedExecutionRefused(
                    f"{label} line {line_no} is not JSON") from exc
            if not isinstance(row, dict):
                raise OwnedExecutionRefused(
                    f"{label} line {line_no} is not an object")
            seen += 1
            if seen > expected_count:
                raise OwnedExecutionRefused(
                    f"{label} has more rows than its manifest count")
            yield line_no, row
    if seen != expected_count:
        raise OwnedExecutionRefused(
            f"{label} count {seen} differs from manifest {expected_count}")


def _git_commit_exists(root: Path, commit: str) -> bool:
    if not re.fullmatch(r"[0-9a-f]{40}", commit or ""):
        return False
    return subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def validate_export(manifest_path: Path, *, repo_root: Path) -> dict:
    """Validate a bounded external export without reading credentials."""
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OwnedExecutionRefused("owned-execution manifest is unreadable") \
            from exc
    if not isinstance(manifest, dict):
        raise OwnedExecutionRefused("owned-execution manifest is not an object")
    _require_fields(manifest, MANIFEST_FIELDS, "manifest")
    if manifest["protocol"] != EXPORT_PROTOCOL:
        raise OwnedExecutionRefused("export protocol differs from declaration")
    if manifest["source_mode"] != SOURCE_MODE:
        raise OwnedExecutionRefused(
            "source mode is public/counterfactual rather than an owned export")
    for field in ("venue", "producer_identity"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise OwnedExecutionRefused(f"manifest {field} is empty")
    for field in ("account_id_sha256", "policy_sha256",
                  "fee_schedule_sha256", "ownership_evidence_sha256",
                  "orders_sha256", "fills_sha256"):
        if not isinstance(manifest[field], str) or not SHA_RE.fullmatch(
                manifest[field]):
            raise OwnedExecutionRefused(f"manifest {field} is not sha256")
    _iso(manifest["export_as_of_utc"], "export_as_of_utc")
    freeze = _iso(manifest["freeze_at_utc"], "freeze_at_utc")
    if not _git_commit_exists(repo_root, manifest["pipeline_commit"]):
        raise OwnedExecutionRefused(
            "pipeline freeze commit is absent or not a full git commit")
    for field in ("orders_export_complete", "acks_export_complete",
                  "fills_export_complete", "fees_export_complete",
                  "maker_role_venue_asserted",
                  "no_live_code_in_research_repo"):
        if manifest[field] is not True:
            raise OwnedExecutionRefused(f"manifest completeness flag {field} is not true")

    days = manifest["complete_utc_days"]
    if not isinstance(days, list) or len(days) < MIN_COMPLETE_DAYS \
            or len(set(days)) != len(days) or days != sorted(days) \
            or any(not isinstance(day, str) or not DAY_RE.fullmatch(day)
                   for day in days):
        raise OwnedExecutionRefused(
            f"complete_utc_days must contain >= {MIN_COMPLETE_DAYS} unique sorted days")
    freeze_day = freeze.date().isoformat()
    if any(day <= freeze_day for day in days):
        raise OwnedExecutionRefused(
            "complete UTC days must be strictly later than the freeze day")
    base = manifest_path.parent
    files = {}
    for stem in ("policy", "fee_schedule", "ownership_evidence", "orders",
                 "fills"):
        path = _relative_file(base, manifest[f"{stem}_path"], f"{stem}_path")
        digest = _sha256(path)
        if digest != manifest[f"{stem}_sha256"]:
            raise OwnedExecutionRefused(f"{stem} sha256 differs from manifest")
        files[stem] = path
    try:
        fee_schedule = json.loads(files["fee_schedule"].read_text())
    except json.JSONDecodeError as exc:
        raise OwnedExecutionRefused("fee schedule is not JSON") from exc
    if not isinstance(fee_schedule, dict) \
            or fee_schedule.get("account_id_sha256") \
            != manifest["account_id_sha256"] \
            or not isinstance(fee_schedule.get("fee_schedule_id"), str) \
            or not fee_schedule["fee_schedule_id"]:
        raise OwnedExecutionRefused(
            "fee schedule does not bind account identity and schedule id")
    schedule_id = fee_schedule["fee_schedule_id"]

    n_orders, n_fills = manifest["n_orders"], manifest["n_fills"]
    if not isinstance(n_orders, int) or isinstance(n_orders, bool) \
            or n_orders < MIN_ORDERS:
        raise OwnedExecutionRefused(
            f"owned order count must be at least {MIN_ORDERS}")
    if not isinstance(n_fills, int) or isinstance(n_fills, bool) \
            or n_fills < MIN_FILLS:
        raise OwnedExecutionRefused(
            f"owned maker fill count must be at least {MIN_FILLS}")

    by_client = {}
    by_venue = {}
    order_days = Counter()
    for line_no, row in _jsonl(files["orders"], n_orders, "orders"):
        _require_fields(row, ORDER_FIELDS, f"order line {line_no}")
        client = row["client_order_id"]
        venue = row["venue_order_id"]
        if not isinstance(client, str) or not client or client in by_client:
            raise OwnedExecutionRefused(
                f"order line {line_no} has duplicate/empty client_order_id")
        if row["ack_status"] not in ACK_STATUSES \
                or row["terminal_status"] not in TERMINAL_STATUSES:
            raise OwnedExecutionRefused(
                f"order line {line_no} has unknown lifecycle status")
        accepted = row["ack_status"] == "ACCEPTED"
        if accepted:
            if not isinstance(venue, str) or not venue or venue in by_venue:
                raise OwnedExecutionRefused(
                    f"order line {line_no} has duplicate/empty venue_order_id")
        elif venue is not None or row["terminal_status"] != "REJECTED":
            raise OwnedExecutionRefused(
                f"rejected order line {line_no} has a venue id/non-rejected terminal")
        if row["maker_side"] not in SIDES:
            raise OwnedExecutionRefused(f"order line {line_no} has unknown maker side")
        if not isinstance(row["policy_action_seq"], int) \
                or isinstance(row["policy_action_seq"], bool) \
                or row["policy_action_seq"] < 0:
            raise OwnedExecutionRefused(
                f"order line {line_no} has invalid policy_action_seq")
        for field in ("reference_generation_id", "slug", "asset_id"):
            if not isinstance(row[field], str) or not row[field]:
                raise OwnedExecutionRefused(f"order line {line_no} has empty {field}")
        times = [row[key] for key in (
            "decision_ns", "submitted_ns", "ack_ns", "terminal_ns")]
        if any(not isinstance(value, int) or isinstance(value, bool)
               for value in times) or times != sorted(times):
            raise OwnedExecutionRefused(
                f"order line {line_no} has decision/submission/ack/terminal inversion")
        for field in ("requested_price", "requested_shares"):
            if not _finite(row[field]) or float(row[field]) <= 0:
                raise OwnedExecutionRefused(
                    f"order line {line_no} has invalid {field}")
        day = _day_from_ns(row["ack_ns"], "ack_ns")
        if row["utc_day"] != day or day not in days:
            raise OwnedExecutionRefused(
                f"order line {line_no} day is outside complete UTC days")
        by_client[client] = row
        if accepted:
            by_venue[venue] = row
        order_days[day] += 1

    fill_ids, trade_ids = set(), set()
    fill_days = Counter()
    filled_shares = defaultdict(float)
    explicit_zero_fees = 0
    for line_no, row in _jsonl(files["fills"], n_fills, "fills"):
        _require_fields(row, FILL_FIELDS, f"fill line {line_no}")
        fill_id, trade_id = row["fill_id"], row["trade_id"]
        if not isinstance(fill_id, str) or not fill_id or fill_id in fill_ids:
            raise OwnedExecutionRefused(
                f"fill line {line_no} has duplicate/empty fill_id")
        if not isinstance(trade_id, str) or not trade_id or trade_id in trade_ids:
            raise OwnedExecutionRefused(
                f"fill line {line_no} has duplicate/empty trade_id")
        order = by_venue.get(row["venue_order_id"])
        if order is None or order["client_order_id"] != row["client_order_id"]:
            raise OwnedExecutionRefused(
                f"fill line {line_no} is orphaned from an acknowledged owned order")
        for field in ("reference_generation_id", "slug", "asset_id", "maker_side"):
            if row[field] != order[field]:
                raise OwnedExecutionRefused(
                    f"fill line {line_no} {field} differs from its order")
        if not isinstance(row["fill_ns"], int) or isinstance(row["fill_ns"], bool) \
                or not order["ack_ns"] <= row["fill_ns"] <= order["terminal_ns"]:
            raise OwnedExecutionRefused(
                f"fill line {line_no} occurs before ack or after terminal")
        if row["liquidity_role"] != "MAKER":
            raise OwnedExecutionRefused(
                f"fill line {line_no} is not venue-asserted MAKER")
        for field in ("price", "shares"):
            if not _finite(row[field]) or float(row[field]) <= 0:
                raise OwnedExecutionRefused(f"fill line {line_no} has invalid {field}")
        if not _finite(row["maker_fee_amount"]) \
                or float(row["maker_fee_amount"]) < 0:
            raise OwnedExecutionRefused(
                f"fill line {line_no} has missing/negative maker_fee_amount")
        if not _finite(row["fee_rate_bps"]) or float(row["fee_rate_bps"]) < 0:
            raise OwnedExecutionRefused(
                f"fill line {line_no} has missing/negative fee_rate_bps")
        if row["fee_component"] != FEE_COMPONENT \
                or row["fee_schedule_id"] != schedule_id \
                or not isinstance(row["fee_currency"], str) \
                or not row["fee_currency"]:
            raise OwnedExecutionRefused(
                f"fill line {line_no} fee identity differs from schedule/scope")
        day = _day_from_ns(row["fill_ns"], "fill_ns")
        if row["utc_day"] != day or day not in days:
            raise OwnedExecutionRefused(
                f"fill line {line_no} day is outside complete UTC days")
        filled_shares[row["venue_order_id"]] += float(row["shares"])
        if filled_shares[row["venue_order_id"]] \
                > float(order["requested_shares"]) + 1e-9:
            raise OwnedExecutionRefused(
                f"fill line {line_no} exceeds its owned order size")
        explicit_zero_fees += float(row["maker_fee_amount"]) == 0.0
        fill_ids.add(fill_id)
        trade_ids.add(trade_id)
        fill_days[day] += 1
    if any(order_days[day] == 0 or fill_days[day] == 0 for day in days):
        raise OwnedExecutionRefused(
            "each declared complete UTC day must contain owned orders and fills")
    return {
        "status": "ADMISSIBLE_OWNED_EXECUTION_SOURCE_PRESENT",
        "source_mode": SOURCE_MODE,
        "venue": manifest["venue"],
        "account_id_sha256": manifest["account_id_sha256"],
        "pipeline_commit": manifest["pipeline_commit"],
        "freeze_at_utc": manifest["freeze_at_utc"],
        "complete_utc_days": days,
        "n_complete_utc_days": len(days),
        "n_orders": n_orders,
        "n_accepted_orders": len(by_venue),
        "n_owned_maker_fills": n_fills,
        "n_explicit_zero_fee_fills": explicit_zero_fees,
        "all_fills_join_acknowledged_owned_orders": True,
        "all_fills_have_explicit_exact_maker_fee": True,
        "maker_rebates_and_liquidity_rewards_excluded": True,
        "economic_result": None,
        "gate1_retroactively_cleared": False,
        "gate2_authorized": False,
    }


def _public_surface_audit(root: Path) -> dict:
    collector = root / "live/pm_research/collect_pm.py"
    tier1 = root / "live/pm_research/tier1_pipeline.py"
    uncertainty = root / "live/pm_research/FLOW_UNCERTAINTY_LOOP.md"
    for path in (collector, tier1, uncertainty):
        if not path.is_file():
            raise OwnedExecutionRefused(f"audit source is missing: {path}")
    collector_text = collector.read_text()
    tier1_text = tier1.read_text()
    uncertainty_text = uncertainty.read_text()
    predicates = {
        "collector_uses_public_market_websocket":
            "ws-subscriptions-clob.polymarket.com/ws/market" in collector_text,
        "collector_records_public_last_trade_fee_field":
            "last_trade_price" in collector_text and "fee_rate_bps" in collector_text,
        "tier1_trade_schema_has_public_transaction_hash":
            '("transaction_hash", pa.string())' in tier1_text,
        "tier1_trade_schema_has_no_client_order_id":
            '("client_order_id",' not in tier1_text[
                tier1_text.find("TRADE_SCHEMA"):tier1_text.find("TRADE_SCHEMA") + 1800],
        "historical_audit_says_venue_ack_lag_not_observed_without_orders":
            "venue_ack_lag` (#11) are **not identifiable**" in uncertainty_text,
    }
    if not all(predicates.values()):
        raise OwnedExecutionRefused(
            f"public source audit predicate changed: {predicates}")
    raw_root = root / "data/pm_5min/raw"
    raw_days = sorted(path.name for path in raw_root.iterdir()
                      if path.is_dir() and re.fullmatch(r"20\d{6}", path.name)) \
        if raw_root.is_dir() else []
    tier_manifests = (root / "data/pm_5min/tier1/trades").glob(
        "day=*/coin=*/**/manifest.json")
    tier_days = sorted({
        next(part.removeprefix("day=") for part in path.parts
             if part.startswith("day="))
        for path in tier_manifests})
    return {
        "computed_predicates": predicates,
        "source_sha256": {
            str(path.relative_to(root)): _sha256(path)
            for path in (collector, tier1, uncertainty)},
        "raw_public_day_range": (
            {"first": raw_days[0], "last": raw_days[-1], "n": len(raw_days)}
            if raw_days else None),
        "tier1_public_trade_day_range": (
            {"first": tier_days[0], "last": tier_days[-1], "n": len(tier_days)}
            if tier_days else None),
        "conclusion": (
            "public market/trade/on-chain surfaces do not bind the simulated "
            "fill to an owned submitted-and-acknowledged order; additional "
            "public tape does not satisfy the owned execution contract"),
    }


def audit(root: Path) -> dict:
    root = root.resolve()
    candidate = root / CANDIDATE_RELATIVE_PATH
    public = _public_surface_audit(root)
    if candidate.is_file():
        admission = validate_export(candidate, repo_root=root)
        status = admission["status"]
    else:
        admission = {
            "status": "REFUSED_NO_OWNED_EXECUTION_SOURCE",
            "candidate": CANDIDATE_RELATIVE_PATH,
            "candidate_exists": False,
            "economic_result": None,
            "gate1_retroactively_cleared": False,
            "gate2_authorized": False,
        }
        status = admission["status"]
    superseded = root / SUPERSEDED_AUDIT_RELATIVE_PATH
    if not superseded.is_file() or _sha256(superseded) \
            != SUPERSEDED_AUDIT_SHA256:
        raise OwnedExecutionRefused(
            "superseded Gate-1f audit receipt is missing or changed")
    return {
        "protocol": PROTOCOL,
        "status": status,
        "as_of": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "declared_candidate": CANDIDATE_RELATIVE_PATH,
        "owned_execution_admission": admission,
        "public_surface_audit": public,
        "supersedes": {
            "path": SUPERSEDED_AUDIT_RELATIVE_PATH,
            "sha256": SUPERSEDED_AUDIT_SHA256,
            "correction": (
                "Tier-1 manifests are nested under distiller directories; "
                "the first path-only census reported a null Tier-1 range. "
                "The owned-source refusal and every gate field are unchanged"),
        },
        "decision_metric": None,
        "gate1_exit_cleared": False,
        "gate2_authorized": False,
        "interpretation": (
            "input-acquisition audit only; no economic field was computed or "
            "reinterpreted and no live-trading code is present"),
    }


def _fixture(root: Path, target: Path, mutate=None) -> Path:
    target.mkdir(parents=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        text=True, capture_output=True).stdout.strip()
    account = hashlib.sha256(b"synthetic-owned-account").hexdigest()
    policy = target / "policy.json"
    policy.write_text(json.dumps({"synthetic": True, "pipeline_commit": commit}))
    schedule = target / "fee_schedule.json"
    schedule.write_text(json.dumps({
        "fee_schedule_id": "synthetic-fee-v1",
        "account_id_sha256": account,
        "source_provenance": "SYNTHETIC_POSITIVE_CONTROL"}))
    ownership = target / "ownership_evidence.json"
    ownership.write_text(json.dumps({
        "account_id_sha256": account,
        "provenance": "SYNTHETIC_AUTHENTICATED_EXPORT"}))
    days = [f"2026-09-0{i}" for i in range(1, 6)]
    orders, fills = [], []
    for i in range(200):
        day = days[i % len(days)]
        base = int(datetime.datetime.fromisoformat(
            day + "T12:00:00+00:00").timestamp() * 1e9) + i * 10_000_000
        client, venue = f"client-{i}", f"venue-{i}"
        common = {"reference_generation_id": f"slug-{i // 2}|BUY_UP|{i}",
                  "client_order_id": client, "venue_order_id": venue,
                  "slug": f"btc-updown-5m-{i // 2}", "asset_id": "asset-up",
                  "maker_side": "BUY_UP", "utc_day": day}
        orders.append({**common, "policy_action_seq": i,
                       "decision_ns": base, "submitted_ns": base + 1_000_000,
                       "ack_ns": base + 2_000_000,
                       "terminal_ns": base + 4_000_000,
                       "requested_price": 0.50, "requested_shares": 1.0,
                       "ack_status": "ACCEPTED", "terminal_status": "FILLED"})
        fills.append({**common, "fill_id": f"fill-{i}",
                      "trade_id": f"trade-{i}", "fill_ns": base + 3_000_000,
                      "price": 0.50, "shares": 1.0,
                      "liquidity_role": "MAKER", "maker_fee_amount": 0.0,
                      "fee_currency": "USDC", "fee_rate_bps": 0.0,
                      "fee_schedule_id": "synthetic-fee-v1",
                      "fee_component": FEE_COMPONENT})
    state = {"orders": orders, "fills": fills, "days": days}
    if mutate is not None:
        mutate(state)
    order_path, fill_path = target / "orders.jsonl", target / "fills.jsonl"
    order_path.write_text("".join(json.dumps(row) + "\n" for row in orders))
    fill_path.write_text("".join(json.dumps(row) + "\n" for row in fills))
    manifest = {
        "protocol": EXPORT_PROTOCOL, "source_mode": SOURCE_MODE,
        "venue": "SYNTHETIC", "account_id_sha256": account,
        "producer_identity": "synthetic-positive-control",
        "export_as_of_utc": "2026-09-06T00:00:00Z",
        "pipeline_commit": commit, "freeze_at_utc": "2026-08-31T23:00:00Z",
        "policy_path": policy.name, "policy_sha256": _sha256(policy),
        "fee_schedule_path": schedule.name,
        "fee_schedule_sha256": _sha256(schedule),
        "ownership_evidence_path": ownership.name,
        "ownership_evidence_sha256": _sha256(ownership),
        "orders_path": order_path.name, "orders_sha256": _sha256(order_path),
        "n_orders": len(orders), "fills_path": fill_path.name,
        "fills_sha256": _sha256(fill_path), "n_fills": len(fills),
        "complete_utc_days": state["days"],
        "orders_export_complete": True, "acks_export_complete": True,
        "fills_export_complete": True, "fees_export_complete": True,
        "maker_role_venue_asserted": True,
        "no_live_code_in_research_repo": True,
    }
    if mutate is not None and "manifest_mutate" in state:
        state["manifest_mutate"](manifest)
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def selftest() -> int:
    root = Path(__file__).resolve().parents[2]
    checks = 0

    def ok(condition, label):
        nonlocal checks
        if not condition:
            raise SystemExit(f"[de_v2_owned_execution_input] FAIL: {label}")
        checks += 1
        print(f"  PASS  {label}")

    def refuses(mutate, label, needle):
        nonlocal checks
        with tempfile.TemporaryDirectory() as tmp:
            path = _fixture(root, Path(tmp) / "export", mutate)
            try:
                validate_export(path, repo_root=root)
            except OwnedExecutionRefused as exc:
                if needle not in str(exc):
                    raise SystemExit(
                        f"[de_v2_owned_execution_input] FAIL: {label}: {exc}")
                checks += 1
                print(f"  PASS  {label}")
                return
        raise SystemExit(
            f"[de_v2_owned_execution_input] FAIL (no refusal): {label}")

    with tempfile.TemporaryDirectory() as tmp:
        good = validate_export(
            _fixture(root, Path(tmp) / "export"), repo_root=root)
    ok(good["status"] == "ADMISSIBLE_OWNED_EXECUTION_SOURCE_PRESENT"
       and good["n_complete_utc_days"] == 5
       and good["n_orders"] == good["n_owned_maker_fills"] == 200,
       "positive export admits 200 owned maker fills over five post-freeze days")
    ok(good["n_explicit_zero_fee_fills"] == 200
       and good["all_fills_have_explicit_exact_maker_fee"],
       "explicit schedule-bound zero fees are data, never a missing-field default")

    refuses(lambda s: s.__setitem__("manifest_mutate", lambda m: m.__setitem__(
        "source_mode", "PUBLIC_MARKET_TRADES")),
        "known-bad public source mode refuses", "public/counterfactual")
    refuses(lambda s: s["fills"][0].__setitem__("venue_order_id", "orphan"),
            "known-bad orphan fill refuses", "orphaned")
    refuses(lambda s: s["fills"][0].__setitem__(
        "fill_ns", s["orders"][0]["ack_ns"] - 1),
        "known-bad pre-ack fill refuses", "before ack")
    refuses(lambda s: s["fills"][0].pop("maker_fee_amount"),
            "known-bad missing exact maker fee refuses", "missing fields")
    refuses(lambda s: s.__setitem__("manifest_mutate", lambda m: m.__setitem__(
        "orders_sha256", "0" * 64)),
        "known-bad export hash refuses", "orders sha256")
    refuses(lambda s: s.__setitem__("manifest_mutate", lambda m: m.__setitem__(
        "complete_utc_days", ["2026-08-31"] + m["complete_utc_days"][1:])),
        "known-bad freeze-day reuse refuses", "strictly later")
    refuses(lambda s: s["fills"][0].__setitem__("liquidity_role", "TAKER"),
            "known-bad taker fill refuses", "not venue-asserted MAKER")

    public = _public_surface_audit(root)
    ok(all(public["computed_predicates"].values()),
       "current public collector/Tier-1 schemas prove no owned-order join")
    ok(public["raw_public_day_range"] is not None
       and public["tier1_public_trade_day_range"] is not None,
       "path-only audit reports both raw and Tier-1 public date ranges")
    print(f"[de_v2_owned_execution_input] PASS -- {checks} checks")
    return 0


def _write_atomic(path: Path, payload: dict) -> None:
    if path.exists():
        raise OwnedExecutionRefused(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.audit or args.output is None:
        parser.error("repository audit requires --audit --output PATH")
    started = time.time()
    payload = audit(Path(__file__).resolve().parents[2])
    usage = resource.getrusage(resource.RUSAGE_SELF)
    payload["resource_observation"] = {
        "wall_seconds": time.time() - started,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "max_rss_kib": usage.ru_maxrss,
        "external_cap_required": True,
    }
    _write_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
