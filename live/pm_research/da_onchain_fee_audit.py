"""DA: does the CHAIN say the maker fee is measured, bounded and signed?

THE QUESTION THIS ANSWERS, AND WHY IT IS THE THIRD CHECK. Gate 1f's terminal
stop rests on a negative existence claim -- that no owned-order per-fill maker
fee can be had.  The reviewer (`1aa9e4b`) found the claim was never checked
against the programme's own measurement; the coordinator verified the three
load-bearing cites at their artifacts.  Both read DOCUMENTS.  This reads the
RECEIPTS: 901 cached Polygon transaction receipts under
`data/pm_5min/onchain/receipts`, decoded here from raw hex.

R-235: this file does NOT import `da_feeds_polygon`.  It re-derives the event
topic0 values by keccak-256 from the signature strings, re-slices the log data
into words, and re-applies the leg classification.  A verifier that imports the
decoder it is verifying tests nothing.  The shipped constants are compared
against the recomputed ones as a fixture, and a mismatch is a finding.

WHAT IS RECOMPUTED, none of it read from a summary:
  * every OrderFilled leg at the exchange, its fee word, and its side
  * taker legs vs maker legs, split by counterparty via OrdersMatched
  * the charged/zero split on maker legs, and WHAT the charged ones are
  * the taker fee against C x 0.07 x p x (1-p), to 6 decimals

    python3 live/pm_research/da_onchain_fee_audit.py --selftest
    python3 live/pm_research/da_onchain_fee_audit.py --real --output P
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

PROTOCOL = "P003_DA_ONCHAIN_FEE_AUDIT_V1"
REPO = Path("/home/yuqing/ctaNew")
RECEIPTS = REPO / "data/pm_5min/onchain/receipts"
EXCHANGE = "0xe111180000d2663c0091e4f400237545b87b996b"
UNIT = 1_000_000
FEE_RATE = 0.07

ORDER_FILLED_SIG = (
    "OrderFilled(bytes32,address,address,uint8,uint256,uint256,uint256,"
    "uint256,bytes32,bytes32)"
)
ORDERS_MATCHED_SIG = (
    "OrdersMatched(bytes32,address,uint8,uint256,uint256,uint256)"
)
#: The values `da_feeds_polygon` ships. Compared against keccak here, never
#: trusted: a topic0 copied from a document is a claim about a document.
SHIPPED_ORDER_FILLED_TOPIC = (
    "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee")
SHIPPED_ORDERS_MATCHED_TOPIC = (
    "0x174b3811690657c217184f89418266767c87e4805d09680c39fc9c031c0cab7c")


class FeeAuditRefused(RuntimeError):
    """The receipts cannot support the question asked of them."""


def keccak256(data: bytes) -> str:
    """keccak-256, from pycryptodome if present else a local permutation.

    The local path exists so this file can prove the topic0 values on a box
    with no crypto library -- the whole point is not to take them on trust."""
    try:
        from Crypto.Hash import keccak as _k
        h = _k.new(digest_bits=256)
        h.update(data)
        return "0x" + h.hexdigest()
    except Exception:                                        # noqa: BLE001
        return "0x" + _keccak_pure(data)


_RC = (0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
       0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
       0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
       0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
       0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
       0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
       0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
       0x8000000000008080, 0x0000000080000001, 0x8000000080008008)
_ROT = ((0, 36, 3, 41, 18), (1, 44, 10, 45, 2), (62, 6, 43, 15, 61),
        (28, 55, 25, 21, 56), (27, 20, 39, 8, 14))


def _keccak_pure(msg: bytes, rate: int = 136) -> str:
    M = 0xFFFFFFFFFFFFFFFF

    def rol(x, n):
        return ((x << n) | (x >> (64 - n))) & M

    A = [[0] * 5 for _ in range(5)]
    pad = bytearray(msg) + b"\x01" + b"\x00" * ((-len(msg) - 1) % rate)
    pad[-1] ^= 0x80
    for off in range(0, len(pad), rate):
        blk = pad[off:off + rate]
        for i in range(rate // 8):
            A[i % 5][i // 5] ^= int.from_bytes(blk[i * 8:i * 8 + 8], "little")
        for rnd in range(24):
            C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4]
                 for x in range(5)]
            D = [C[(x - 1) % 5] ^ rol(C[(x + 1) % 5], 1) for x in range(5)]
            for x in range(5):
                for y in range(5):
                    A[x][y] ^= D[x]
            B = [[0] * 5 for _ in range(5)]
            for x in range(5):
                for y in range(5):
                    B[y][(2 * x + 3 * y) % 5] = rol(A[x][y], _ROT[x][y])
            for x in range(5):
                for y in range(5):
                    A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & M
                                         & B[(x + 2) % 5][y])
            A[0][0] ^= _RC[rnd]
    out = b""
    for i in range(4):
        out += A[i % 5][i // 5].to_bytes(8, "little")
    return out.hex()


def words(data: str) -> list[str]:
    body = data[2:] if data.startswith("0x") else data
    if len(body) % 64:
        raise FeeAuditRefused(
            f"log data is not a whole number of 32-byte words: {len(body)} hex")
    return [body[i:i + 64] for i in range(0, len(body), 64)]


def addr(topic: str) -> str:
    return "0x" + topic[-40:]


def decode_receipt(rec: dict) -> dict:
    """Every exchange leg in one receipt, re-decoded from raw hex."""
    filled, matched = [], []
    for log in rec.get("logs", []):
        if (log.get("address") or "").lower() != EXCHANGE:
            continue
        topics = log.get("topics") or []
        if not topics:
            continue
        t0 = topics[0].lower()
        if t0 == SHIPPED_ORDER_FILLED_TOPIC:
            w = words(log["data"])
            if len(w) != 7:
                raise FeeAuditRefused(
                    f"OrderFilled expects 7 data words, got {len(w)}")
            filled.append({
                "order_hash": topics[1],
                "maker": addr(topics[2]),
                "taker": addr(topics[3]),
                "maker_amount_filled": int(w[2], 16),
                "taker_amount_filled": int(w[3], 16),
                "fee": int(w[4], 16),
            })
        elif t0 == SHIPPED_ORDERS_MATCHED_TOPIC:
            matched.append({"taker_order_maker": addr(topics[2])})
    return {"order_filled": filled, "orders_matched": matched,
            "tx": rec.get("transactionHash"),
            "block": int(rec.get("blockNumber", "0x0"), 16)
            if isinstance(rec.get("blockNumber"), str)
            else rec.get("blockNumber"),
            "status": rec.get("status")}


def leg_economics(leg: dict) -> dict:
    """Size and price for one leg, from the two filled amounts alone.

    One side of every leg is USDC and the other is outcome tokens, both
    6-decimal. The token count is the LARGER of the two whenever p < 0.5 and
    the smaller whenever p > 0.5, so size is max/UNIT and price is min/max --
    which is what makes p land in (0,1) without needing the side flag."""
    a, b = leg["maker_amount_filled"], leg["taker_amount_filled"]
    lo, hi = (a, b) if a <= b else (b, a)
    if hi == 0:
        return {"size": 0.0, "price": None}
    return {"size": hi / UNIT, "price": lo / hi}


def classify(dec: dict) -> list[dict]:
    """Taker leg vs maker legs, split by counterparty.

    `OrdersMatched.takerOrderMaker` is the address that SUBMITTED the taker
    order. The OrderFilled leg whose `maker` field is that address is the
    taker's own leg; every other leg in the transaction is a resting maker."""
    takers = {m["taker_order_maker"] for m in dec["orders_matched"]}
    out = []
    for leg in dec["order_filled"]:
        role = "TAKER" if leg["maker"] in takers else "MAKER"
        e = leg_economics(leg)
        out.append({**leg, "role": role, "tx": dec["tx"],
                    "block": dec["block"], **e})
    return out


def predicted_fee_usdc(size: float, price: float) -> float:
    return size * FEE_RATE * price * (1.0 - price)


def audit(root: Path | None = None, limit: int | None = None) -> dict:
    d = Path(root) if root is not None else RECEIPTS
    if not d.is_dir():
        raise FeeAuditRefused(
            f"REFUSED: no receipt cache at {d}. An audit that cannot read the "
            f"chain must never report that the chain says nothing -- that is "
            f"the empty-set trap (rule 15).")
    files = sorted(p for p in d.rglob("*.json"))
    if not files:
        raise FeeAuditRefused(
            f"REFUSED: {d} holds ZERO receipts. A zero from a reader that "
            f"never fired is not a result.")
    if limit:
        files = files[:limit]
    legs, bad, n_tx_no_matched = [], [], 0
    for p in files:
        try:
            rec = json.loads(p.read_text())
        except Exception as e:                               # noqa: BLE001
            bad.append({"file": p.name, "error": repr(e)})
            continue
        dec = decode_receipt(rec)
        if dec["order_filled"] and not dec["orders_matched"]:
            n_tx_no_matched += 1
        legs.extend(classify(dec))

    taker = [x for x in legs if x["role"] == "TAKER"]
    maker = [x for x in legs if x["role"] == "MAKER"]
    maker_charged = [x for x in maker if x["fee"] > 0]
    maker_zero = [x for x in maker if x["fee"] == 0]
    taker_charged = [x for x in taker if x["fee"] > 0]

    # The formula, tested rather than assumed, on charged taker legs.
    exact6, resid = 0, []
    for x in taker_charged:
        if x["price"] is None:
            continue
        pred = predicted_fee_usdc(x["size"], x["price"])
        obs = x["fee"] / UNIT
        resid.append(abs(pred - obs))
        if abs(pred - obs) <= 1e-6:
            exact6 += 1
    resid.sort()

    # What ARE the ten? Address, market-ish key, time, amount.
    charged_detail = [{
        "tx": x["tx"], "block": x["block"], "maker": x["maker"],
        "taker": x["taker"], "fee_usdc": x["fee"] / UNIT,
        "size": x["size"], "price": x["price"],
        "fee_cents_per_share": (x["fee"] / UNIT) * 100.0 / x["size"]
        if x["size"] else None,
        "predicted_taker_fee_usdc": (predicted_fee_usdc(x["size"], x["price"])
                                     if x["price"] is not None else None),
    } for x in sorted(maker_charged, key=lambda z: (z["block"] or 0))]

    by_addr = collections.Counter(x["maker"] for x in maker_charged)
    by_tx = collections.Counter(x["tx"] for x in maker_charged)
    by_block = collections.Counter(x["block"] for x in maker_charged)
    charged_addr_total_legs = {
        a: sum(1 for x in maker if x["maker"] == a) for a in by_addr}

    return {
        "protocol": PROTOCOL,
        "n_receipt_files": len(files),
        "n_unreadable": len(bad),
        "unreadable": bad[:5],
        "n_tx_with_fills_but_no_ordersmatched": n_tx_no_matched,
        "n_legs": len(legs),
        "n_taker_legs": len(taker),
        "n_maker_legs": len(maker),
        "n_taker_legs_charged": len(taker_charged),
        "taker_all_charged": len(taker_charged) == len(taker) and bool(taker),
        "n_maker_legs_charged": len(maker_charged),
        "n_maker_legs_zero": len(maker_zero),
        "maker_charged_share": (len(maker_charged) / len(maker)
                                if maker else None),
        "formula": "C * 0.07 * p * (1-p)",
        "n_taker_charged_matching_formula_to_1e6": exact6,
        "taker_formula_match_share": (exact6 / len(taker_charged)
                                      if taker_charged else None),
        "taker_residual_usdc": {
            "max": resid[-1] if resid else None,
            "p50": resid[len(resid) // 2] if resid else None,
        },
        "maker_charged_detail": charged_detail,
        "maker_charged_by_address": dict(by_addr),
        "maker_charged_address_total_maker_legs": charged_addr_total_legs,
        "maker_charged_distinct_tx": len(by_tx),
        "maker_charged_distinct_blocks": len(by_block),
        "role": "REPORTED, NOT ENFORCED (rule 14). This measures what the "
                "chain records; it promotes nothing and clears no gate.",
        "limits": [
            "the 901 receipts are a SAMPLE of our own recorded trades, not a "
            "population; incidence here bounds observed volume only",
            "leg roles come from OrdersMatched.takerOrderMaker; a transaction "
            "carrying fills with no OrdersMatched is counted and reported",
            "fees are read from the OrderFilled fee word, never from the "
            "websocket fee_rate_bps field, which is unpopulated",
        ],
    }


def selftest() -> int:
    fails = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    # ---- keccak proves itself on the published empty-string vector -------
    ok(keccak256(b"") == "0x" + "c5d2460186f7233c927e7db2dcc703c0"
                                "e500b653ca82273b7bfad8045d85a470",
       "KECCAK: matches the published keccak-256('') vector, so the digest "
       "used to derive the topics is the right function")

    # ---- and the topic0 values are DERIVED, not copied -------------------
    of = keccak256(ORDER_FILLED_SIG.encode())
    om = keccak256(ORDERS_MATCHED_SIG.encode())
    ok(of == SHIPPED_ORDER_FILLED_TOPIC,
       f"TOPIC0 OrderFilled DERIVED from the signature = the shipped "
       f"constant ({of[:18]}...) -- the ABI is confirmed, not assumed")
    ok(om == SHIPPED_ORDERS_MATCHED_TOPIC,
       f"TOPIC0 OrdersMatched DERIVED = shipped ({om[:18]}...)")

    # ---- POSITIVE CONTROL: a synthetic receipt must decode exactly -------
    def w(n):
        return f"{n:064x}"
    tx = {"transactionHash": "0xdead", "blockNumber": "0x10", "status": "0x1",
          "logs": [
              {"address": EXCHANGE,
               "topics": [SHIPPED_ORDERS_MATCHED_TOPIC, "0x" + w(1),
                          "0x" + "0" * 24 + "aa" * 20],
               "data": "0x" + w(0) + w(0) + w(0) + w(0)},
              {"address": EXCHANGE,
               "topics": [SHIPPED_ORDER_FILLED_TOPIC, "0x" + w(1),
                          "0x" + "0" * 24 + "aa" * 20,
                          "0x" + "0" * 24 + "bb" * 20],
               "data": "0x" + w(0) + w(0) + w(47_190_000)
                       + w(94_380_000) + w(1_651_650) + w(0) + w(0)},
              {"address": EXCHANGE,
               "topics": [SHIPPED_ORDER_FILLED_TOPIC, "0x" + w(2),
                          "0x" + "0" * 24 + "cc" * 20,
                          "0x" + "0" * 24 + "bb" * 20],
               "data": "0x" + w(0) + w(0) + w(47_190_000)
                       + w(94_380_000) + w(0) + w(0) + w(0)},
              {"address": "0x" + "ee" * 20,          # a foreign contract
               "topics": [SHIPPED_ORDER_FILLED_TOPIC, "0x" + w(3),
                          "0x" + "0" * 24 + "dd" * 20,
                          "0x" + "0" * 24 + "bb" * 20],
               "data": "0x" + w(0) + w(0) + w(1) + w(1) + w(999) + w(0) + w(0)},
          ]}
    legs = classify(decode_receipt(tx))
    ok(len(legs) == 2,
       "FOREIGN-ADDRESS CONTROL: a log with the right topic0 at a DIFFERENT "
       "contract is excluded -- 2 legs, not 3")
    roles = {x["role"]: x for x in legs}
    ok(set(roles) == {"TAKER", "MAKER"},
       "CLASSIFY: the leg whose maker is OrdersMatched.takerOrderMaker is the "
       "TAKER leg; the other is a resting MAKER")
    ok(roles["TAKER"]["fee"] == 1_651_650 and roles["MAKER"]["fee"] == 0,
       "DECODE: fee word[4] read as 1,651,650 on the taker leg and 0 on the "
       "maker leg")
    e = leg_economics(roles["TAKER"])
    ok(abs(e["size"] - 94.38) < 1e-9 and abs(e["price"] - 0.5) < 1e-9,
       f"ECONOMICS: size {e['size']} and price {e['price']} recovered from the "
       f"two filled amounts alone")
    pred = predicted_fee_usdc(e["size"], e["price"])
    ok(abs(pred - 1.651650) < 1e-9,
       f"FORMULA: C*0.07*p*(1-p) = {pred:.6f} reproduces the ITER1_M worked "
       f"example (94.38 shares at p=0.50 -> $1.651650)")

    # ---- KNOWN-BAD, both directions --------------------------------------
    try:
        words("0x" + "ab" * 33)
        ok(False, "KNOWN-BAD: accepted a ragged data field -- must refuse")
    except FeeAuditRefused:
        ok(True, "KNOWN-BAD: refuses log data that is not whole words")
    try:
        bad = json.loads(json.dumps(tx))
        bad["logs"][1]["data"] = "0x" + w(0) * 3
        decode_receipt(bad)
        ok(False, "KNOWN-BAD: accepted a 3-word OrderFilled -- must refuse")
    except FeeAuditRefused:
        ok(True, "KNOWN-BAD: refuses an OrderFilled without 7 data words")
    try:
        audit(root=Path("/nonexistent/receipts"))
        ok(False, "KNOWN-BAD: reported on an absent cache -- must refuse")
    except FeeAuditRefused:
        ok(True, "KNOWN-BAD: an absent receipt cache REFUSES rather than "
                 "reporting that the chain says nothing")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        try:
            audit(root=Path(td))
            ok(False, "KNOWN-BAD: reported on an EMPTY cache -- must refuse")
        except FeeAuditRefused:
            ok(True, "KNOWN-BAD: an empty cache refuses -- a zero from a "
                     "reader that never fired is not a result")

    # ---- FALSIFIER: the audit must FIND a charged maker leg when one -----
    # ---- exists, or its count of ten is a check that cannot fire. --------
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.json"
        planted = json.loads(json.dumps(tx))
        planted["logs"][2]["data"] = ("0x" + w(0) + w(0) + w(47_190_000)
                                      + w(94_380_000) + w(4_242) + w(0) + w(0))
        p.write_text(json.dumps(planted))
        got = audit(root=Path(td))
    ok(got["n_maker_legs_charged"] == 1
       and abs(got["maker_charged_detail"][0]["fee_usdc"] - 0.004242) < 1e-9,
       "FALSIFIER: a PLANTED charged maker leg is found and priced -- the "
       "charged-maker count is a check that can fire, so a low count is a "
       "measurement rather than a silent miss")

    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.real:
        out = audit(limit=a.limit)
        txt = json.dumps(out, indent=2, sort_keys=True)
        if a.output:
            a.output.write_text(txt)
        print(txt[:4000])
        return 0
    ap.error("choose --selftest or --real")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
