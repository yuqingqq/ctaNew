"""G-FF1 — is the WS `side` the taker's direction? Protocol `gff1_v2`.

Runs the frozen protocol in GFF1_PROTOCOL.md. Reads immutable window archives,
draws a stratified deterministic sample, joins each WS trade to the on-chain
OrdersMatched leg for the same token, VALIDATES the join on size and price
before using it, and reports agreement as GateEvidence with a Wilson interval.

    python3 live/pm_research/exp_gff1_side.py --selftest
    python3 live/pm_research/exp_gff1_side.py            # full frozen run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from da_feeds_polygon import (  # noqa: E402
    DecodeError,
    PolygonRPC,
    RpcError,
    orders_filled,
    orders_matched,
)

PROTOCOL = "gff1_v2"
REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data/pm_5min/raw"
OUT_JSON = REPO / "data/pm_5min/derived/gff1_side_v2.json"
OUT_MD = Path(__file__).with_name("GFF1_RESULTS.md")

# --- frozen protocol constants -------------------------------------------
DAYS = ("20260819", "20260820")
SEED = 20260821
TARGET_TX = 500
MIN_PER_COIN = 20
SIZE_TOL = 1e-6
PRICE_TOL = 5e-4
MAX_JOIN_MISMATCH_RATE = 0.05
MIN_COIN_AGREEMENT = 0.95
THRESHOLD = 0.99

# Inspected before the freeze to establish the decode/join method. Design data,
# excluded from the measurement sample by hash.
DESIGN_TX = "0x1c6a460820a70fa4cc80aa6c8e3137aaab2f4ba1735d4f34286ff98427bfd99c"

MONEYNESS = ((0.15, "p<0.15"), (0.35, "0.15-0.35"), (0.65, "0.35-0.65"),
             (0.85, "0.65-0.85"), (1.01, "p>=0.85"))

TRADE_MARK = b'"event_type":"last_trade_price"'
SLUG_RE = re.compile(r"^([a-z]+)-updown-5m-(\d+)$")


def moneyness_bucket(price: float) -> str:
    for hi, label in MONEYNESS:
        if price < hi:
            return label
    return MONEYNESS[-1][1]


def wilson(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Two-sided 95% Wilson interval. Defined at k=0 and k=n, unlike Wald."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True, slots=True)
class Leg:
    """One WS trade event: a (transaction, token) candidate."""

    tx_hash: str
    asset_id: str
    side: str
    size: float
    price: float
    coin: str
    slug: str
    recv_ns: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.tx_hash, self.asset_id)


@dataclass
class Manifest:
    protocol: str = PROTOCOL
    protocol_sha256: str = ""
    script_sha256: str = ""
    seed: int = SEED
    days: tuple[str, ...] = DAYS
    endpoints: tuple[str, ...] = ()
    source_files: list[dict[str, Any]] = field(default_factory=list)
    candidate_digest: str = ""
    n_candidate_legs: int = 0
    n_candidate_tx: int = 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_gz_lines(path: Path) -> Iterator[tuple[bytes, str]]:
    """Decompress and hash in ONE pass, so the manifest costs no extra read."""
    h = hashlib.sha256()
    dec = zlib.decompressobj(zlib.MAX_WBITS | 16)
    tail = b""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            try:
                out = dec.decompress(chunk)
            except zlib.error:
                break  # truncated archive: keep what parsed, never guess
            if not out:
                continue
            buf = tail + out
            *lines, tail = buf.split(b"\n")
            for line in lines:
                yield line, ""
    if tail:
        yield tail, ""
    yield b"", h.hexdigest()


def scan_day(day: str) -> tuple[list[Leg], list[dict[str, Any]]]:
    """Extract trade legs from every IMMUTABLE (gzipped, closed) window."""
    day_dir = RAW / day
    legs: list[Leg] = []
    files: list[dict[str, Any]] = []
    if not day_dir.is_dir():
        return legs, files

    for path in sorted(day_dir.glob("*.jsonl*.gz")):
        slug = path.name.split(".jsonl")[0]
        m = SLUG_RE.match(slug)
        coin = m.group(1) if m else "?"
        digest = ""
        n_here = 0
        for line, final_digest in _stream_gz_lines(path):
            if final_digest:
                digest = final_digest
                continue
            if TRADE_MARK not in line:
                continue  # ~97% of lines are price_change; skip before parsing
            parts = line.split(b"\t", 1)
            if len(parts) != 2:
                continue
            try:
                recv_ns = int(parts[0])
                payload = json.loads(parts[1])
            except (ValueError, json.JSONDecodeError):
                continue
            for msg in payload if isinstance(payload, list) else [payload]:
                if not isinstance(msg, dict):
                    continue
                if msg.get("event_type") != "last_trade_price":
                    continue
                tx = msg.get("transaction_hash")
                if not tx or tx.lower() == DESIGN_TX:
                    continue
                try:
                    legs.append(Leg(
                        tx_hash=tx.lower(),
                        asset_id=str(msg["asset_id"]),
                        side=str(msg["side"]).upper(),
                        size=float(msg["size"]),
                        price=float(msg["price"]),
                        coin=coin,
                        slug=slug,
                        recv_ns=recv_ns,
                    ))
                    n_here += 1
                except (KeyError, ValueError, TypeError):
                    continue
        files.append({"name": path.name, "bytes": path.stat().st_size,
                      "sha256": digest, "legs": n_here})
    return legs, files


def build_sample(legs: list[Leg]) -> tuple[list[str], dict[str, Any]]:
    """Deterministic stratified draw over TRANSACTIONS (a tx is wholly in/out)."""
    by_tx: dict[str, list[Leg]] = defaultdict(list)
    for leg in legs:
        by_tx[leg.tx_hash].append(leg)
    for tx in by_tx:
        by_tx[tx].sort(key=lambda l: (l.asset_id, l.recv_ns))

    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for tx in sorted(by_tx):
        head = by_tx[tx][0]
        strata[(head.coin, moneyness_bucket(head.price), head.side)].append(tx)

    rng = random.Random(SEED)
    for key in strata:
        strata[key].sort()
        rng.shuffle(strata[key])

    # Round-robin over sorted strata keys: even fill, no cell starved by order.
    picked: list[str] = []
    keys = sorted(strata)
    cursor = {k: 0 for k in keys}
    per_coin: Counter[str] = Counter()
    while len(picked) < TARGET_TX:
        advanced = False
        for k in keys:
            if len(picked) >= TARGET_TX and per_coin and min(
                    per_coin.get(c, 0) for c in {kk[0] for kk in keys}) >= MIN_PER_COIN:
                break
            i = cursor[k]
            if i >= len(strata[k]):
                continue
            picked.append(strata[k][i])
            per_coin[k[0]] += 1
            cursor[k] = i + 1
            advanced = True
        if not advanced:
            break  # tape exhausted; realised counts are reported as-is

    # Top up any coin short of its floor, where the tape allows.
    for coin in sorted({k[0] for k in keys}):
        while per_coin[coin] < MIN_PER_COIN:
            room = [k for k in keys if k[0] == coin and cursor[k] < len(strata[k])]
            if not room:
                break
            k = room[0]
            picked.append(strata[k][cursor[k]])
            cursor[k] += 1
            per_coin[coin] += 1

    design = {
        "strata_available": {"|".join(k): len(v) for k, v in sorted(strata.items())},
        "strata_drawn": {"|".join(k): cursor[k] for k in keys if cursor[k]},
        "per_coin_drawn": dict(sorted(per_coin.items())),
        "target_tx": TARGET_TX,
        "drawn_tx": len(picked),
    }
    return picked, design


def evaluate(sample_tx: list[str], legs_by_tx: dict[str, list[Leg]],
             rpc: PolygonRPC, verbose: bool = True) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    fees_nonzero = 0

    for n, tx in enumerate(sample_tx, 1):
        if verbose and n % 50 == 0:
            print(f"  {n}/{len(sample_tx)} tx  (rpc calls {rpc.calls}, "
                  f"cache {rpc.cache_hits})", flush=True)
        try:
            receipt = rpc.receipt(tx)
        except RpcError as exc:
            excluded["RECEIPT_UNAVAILABLE"] += 1
            rows.append({"tx": tx, "status": "RECEIPT_UNAVAILABLE", "detail": str(exc)})
            continue

        matched = orders_matched(receipt)
        leg_results: list[dict[str, Any]] = []
        for leg in legs_by_tx[tx]:
            cands = [om for om in matched if str(om.asset_id) == leg.asset_id]
            if len(cands) != 1:
                excluded["AMBIGUOUS_LEG"] += 1
                leg_results.append({"asset_id": leg.asset_id, "status": "AMBIGUOUS_LEG",
                                    "n_candidates": len(cands)})
                continue
            om = cands[0]
            try:
                # Direction comes from WHICH leg of the amount pair is USDC --
                # chain-only, independent of side_enum and of the WS field we
                # are testing. That is what keeps the comparison non-circular.
                chain_side = om.implied_direction
                chain_size = om.size
                chain_price = om.price
            except DecodeError as exc:
                excluded["DIRECTION_UNIDENTIFIED"] += 1
                leg_results.append({"asset_id": leg.asset_id,
                                    "status": "DIRECTION_UNIDENTIFIED",
                                    "detail": str(exc)})
                continue
            size_ok = abs(chain_size - leg.size) <= SIZE_TOL
            price_ok = abs(chain_price - leg.price) <= PRICE_TOL
            if not (size_ok and price_ok):
                excluded["JOIN_MISMATCH"] += 1
                leg_results.append({
                    "asset_id": leg.asset_id, "status": "JOIN_MISMATCH",
                    "ws_size": leg.size, "chain_size": chain_size,
                    "ws_price": leg.price, "chain_price": round(chain_price, 6),
                    "chain_side": chain_side, "side_enum": om.side_enum})
                continue
            leg_results.append({
                "asset_id": leg.asset_id, "status": "VALIDATED",
                "ws_side": leg.side, "chain_side": chain_side,
                "side_enum": om.side_enum,
                "agree": chain_side == leg.side,
                # secondary: does the uint8 enum carry the same label? This
                # LEARNS the enum mapping rather than assuming 0=BUY.
                "enum_agree": (om.side_enum == 0) == (leg.side == "BUY"),
                "taker": om.taker_order_maker, "coin": leg.coin,
                "moneyness": moneyness_bucket(leg.price)})

        validated = [r for r in leg_results if r["status"] == "VALIDATED"]
        if not validated:
            rows.append({"tx": tx, "status": "NO_VALIDATED_LEG", "legs": leg_results})
            continue
        fees_nonzero += any(of.fee for of in orders_filled(receipt))
        rows.append({
            "tx": tx, "status": "VALIDATED",
            "agree": all(r["agree"] for r in validated),
            "coin": validated[0]["coin"], "moneyness": validated[0]["moneyness"],
            "ws_side": validated[0]["ws_side"], "legs": leg_results})

    return {"rows": rows, "excluded": dict(excluded), "fees_nonzero_tx": fees_nonzero}


def verdict(agree: int, n: int, per_coin: dict[str, tuple[int, int]],
            mismatch_rate: float) -> tuple[str, list[str]]:
    lo, hi = wilson(agree, n)
    reasons: list[str] = []
    if n < TARGET_TX:
        reasons.append(f"only {n} validated tx-clusters, protocol requires {TARGET_TX}")
    if mismatch_rate > MAX_JOIN_MISMATCH_RATE:
        reasons.append(f"JOIN_MISMATCH rate {mismatch_rate:.3f} exceeds "
                       f"{MAX_JOIN_MISMATCH_RATE}")
    weak = {c: a / t for c, (a, t) in per_coin.items()
            if t and a / t < MIN_COIN_AGREEMENT}
    if weak:
        reasons.append("coin strata below "
                       f"{MIN_COIN_AGREEMENT}: " +
                       ", ".join(f"{c} {v:.3f}" for c, v in sorted(weak.items())))
    if hi < THRESHOLD:
        return "MODEL_REFUTED", reasons or [f"Wilson upper {hi:.4f} < {THRESHOLD}"]
    if reasons:
        return "INSUFFICIENT_EVIDENCE", reasons
    if lo >= THRESHOLD:
        return "PASS", []
    return "INSUFFICIENT_EVIDENCE", [f"Wilson lower {lo:.4f} < {THRESHOLD}"]


# --------------------------------------------------------------------------


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    ok(moneyness_bucket(0.10) == "p<0.15", "moneyness low")
    ok(moneyness_bucket(0.51) == "0.35-0.65", "moneyness atm")
    ok(moneyness_bucket(0.99) == "p>=0.85", "moneyness high")

    lo, hi = wilson(500, 500)
    ok(lo > 0.99 and hi == 1.0, f"wilson 500/500 lower {lo:.4f} must clear 0.99")
    lo2, _ = wilson(50, 50)
    ok(lo2 < 0.99, f"wilson 50/50 lower {lo2:.4f} must NOT clear 0.99 (n matters)")
    lo3, hi3 = wilson(0, 500)
    ok(hi3 < 0.01, "wilson 0/500 upper near zero")
    ok(wilson(0, 0) == (0.0, 1.0), "wilson n=0 is uninformative, not a crash")

    # verdict logic: failure to reject is not equivalence.
    v, r = verdict(500, 500, {"btc": (500, 500)}, 0.0)
    ok(v == "PASS", f"clean 500/500 should PASS, got {v} {r}")
    v, _ = verdict(100, 100, {"btc": (100, 100)}, 0.0)
    ok(v == "INSUFFICIENT_EVIDENCE", "n below target cannot PASS")
    v, _ = verdict(0, 500, {"btc": (0, 500)}, 0.0)
    ok(v == "MODEL_REFUTED", "inverted convention must refute, not sign-flip")
    v, _ = verdict(500, 500, {"btc": (500, 500)}, 0.20)
    ok(v == "INSUFFICIENT_EVIDENCE", "high mismatch rate blocks PASS")
    v, _ = verdict(500, 500, {"btc": (300, 320), "hype": (200, 180)}, 0.0)
    ok(v == "INSUFFICIENT_EVIDENCE", "a weak coin stratum blocks a pooled PASS")

    # sampling is deterministic and transaction-atomic.
    legs = [Leg(f"0x{i:064x}", str(1000 + i % 3), "BUY" if i % 2 else "SELL",
                1.0 + i, 0.1 + (i % 9) / 10, ["btc", "eth", "sol"][i % 3],
                "s", i) for i in range(400)]
    a, _ = build_sample(legs)
    b, _ = build_sample(legs)
    ok(a == b, "sample must be reproducible under the frozen seed")
    ok(len(set(a)) == len(a), "no duplicate transactions in the draw")

    print(f"exp_gff1_side selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--scan-only", action="store_true",
                    help="extract and stratify, no RPC")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    man = Manifest(
        protocol_sha256=sha256_file(Path(__file__).with_name("GFF1_PROTOCOL.md")),
        script_sha256=sha256_file(Path(__file__)),
    )

    print(f"[gff1] scanning immutable windows for {', '.join(DAYS)} ...", flush=True)
    all_legs: list[Leg] = []
    for day in DAYS:
        legs, files = scan_day(day)
        man.source_files.extend(files)
        all_legs.extend(legs)
        print(f"  {day}: {len(files)} archives, {len(legs)} trade legs", flush=True)

    if not all_legs:
        print("no trade legs found; nothing to do")
        return 1

    legs_by_tx: dict[str, list[Leg]] = defaultdict(list)
    for leg in all_legs:
        legs_by_tx[leg.tx_hash].append(leg)

    digest = hashlib.sha256()
    for leg in sorted(all_legs, key=lambda l: (l.tx_hash, l.asset_id, l.recv_ns)):
        digest.update(f"{leg.tx_hash}|{leg.asset_id}|{leg.side}|{leg.size}|"
                      f"{leg.price}".encode())
    man.candidate_digest = digest.hexdigest()
    man.n_candidate_legs = len(all_legs)
    man.n_candidate_tx = len(legs_by_tx)
    print(f"[gff1] {len(all_legs)} legs across {len(legs_by_tx)} transactions",
          flush=True)

    sample, design = build_sample(all_legs)
    print(f"[gff1] drew {len(sample)} transactions across "
          f"{len(design['strata_drawn'])} strata", flush=True)
    if args.scan_only:
        print(json.dumps(design, indent=2))
        return 0

    rpc = PolygonRPC()
    man.endpoints = rpc.endpoints
    print(f"[gff1] fetching receipts ...", flush=True)
    result = evaluate(sample, legs_by_tx, rpc)

    validated = [r for r in result["rows"] if r["status"] == "VALIDATED"]
    agree = sum(1 for r in validated if r["agree"])
    n = len(validated)
    lo, hi = wilson(agree, n)

    per_coin: dict[str, tuple[int, int]] = {}
    for r in validated:
        a, t = per_coin.get(r["coin"], (0, 0))
        per_coin[r["coin"]] = (a + (1 if r["agree"] else 0), t + 1)
    per_money: dict[str, tuple[int, int]] = {}
    for r in validated:
        a, t = per_money.get(r["moneyness"], (0, 0))
        per_money[r["moneyness"]] = (a + (1 if r["agree"] else 0), t + 1)

    n_legs_seen = sum(len(legs_by_tx[t]) for t in sample)
    mismatch_rate = result["excluded"].get("JOIN_MISMATCH", 0) / max(n_legs_seen, 1)
    v, reasons = verdict(agree, n, per_coin, mismatch_rate)

    payload = {
        "manifest": {**asdict(man), "source_files": man.source_files[:0] or
                     man.source_files},
        "sample_design": design,
        "excluded": result["excluded"],
        "n_sampled_tx": len(sample),
        "n_validated_tx": n,
        "agreement": agree / n if n else None,
        "wilson95": [lo, hi],
        "threshold": THRESHOLD,
        "verdict": v,
        "verdict_reasons": reasons,
        "per_coin": {c: {"agree": a, "n": t} for c, (a, t) in sorted(per_coin.items())},
        "per_moneyness": {m: {"agree": a, "n": t}
                          for m, (a, t) in sorted(per_money.items())},
        "fees_nonzero_tx": result["fees_nonzero_tx"],
        "rows": result["rows"],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1))

    write_report(payload)
    print(f"\n[gff1] agreement {agree}/{n} = "
          f"{(agree / n if n else float('nan')):.4f}  "
          f"Wilson95 [{lo:.4f}, {hi:.4f}]  ->  {v}")
    for r in reasons:
        print(f"        - {r}")
    print(f"[gff1] wrote {OUT_JSON} and {OUT_MD}")
    return 0


def write_report(p: dict[str, Any]) -> None:
    m = p["manifest"]
    lines = [
        f"# G-FF1 — WS `side` vs on-chain taker direction (`{PROTOCOL}`)",
        "",
        f"Verdict: **{p['verdict']}**. Threshold {p['threshold']} on the Wilson "
        "lower bound.",
        "",
        "## Sample",
        "",
        f"- protocol SHA-256: `{m['protocol_sha256']}`",
        f"- script SHA-256: `{m['script_sha256']}`",
        f"- candidate digest: `{m['candidate_digest']}`",
        f"- UTC days: {', '.join(m['days'])}; seed {m['seed']}",
        f"- source archives: {len(m['source_files'])}; candidate legs "
        f"{m['n_candidate_legs']}; candidate transactions {m['n_candidate_tx']}",
        f"- sampled transactions: {p['n_sampled_tx']}; validated: "
        f"{p['n_validated_tx']}",
        "",
        "## Result",
        "",
        f"Agreement **{p['agreement']:.4f}** "
        f"(Wilson 95% [{p['wilson95'][0]:.4f}, {p['wilson95'][1]:.4f}])"
        if p["agreement"] is not None else "No validated rows.",
        "",
        "| coin | agree | n | rate |",
        "|---|---:|---:|---:|",
    ]
    for c, d in p["per_coin"].items():
        lines.append(f"| {c} | {d['agree']} | {d['n']} | "
                     f"{d['agree'] / d['n']:.4f} |" if d["n"] else
                     f"| {c} | 0 | 0 | — |")
    lines += ["", "| moneyness | agree | n | rate |", "|---|---:|---:|---:|"]
    for mn, d in p["per_moneyness"].items():
        lines.append(f"| {mn} | {d['agree']} | {d['n']} | "
                     f"{d['agree'] / d['n']:.4f} |" if d["n"] else
                     f"| {mn} | 0 | 0 | — |")
    lines += ["", "## Excluded (reported beside the retained set)", "",
              "| reason | count |", "|---|---:|"]
    for k, vv in sorted(p["excluded"].items()):
        lines.append(f"| `{k}` | {vv} |")
    if not p["excluded"]:
        lines.append("| — | 0 |")
    if p["verdict_reasons"]:
        lines += ["", "## Why not PASS", ""]
        lines += [f"- {r}" for r in p["verdict_reasons"]]
    lines += ["", f"Protocol: `live/pm_research/GFF1_PROTOCOL.md`.", ""]
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
