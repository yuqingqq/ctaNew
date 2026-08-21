"""DA-Feeds / PolygonRPC — read-only on-chain receipt source.

Fills the `PolygonRPC` named seam in PM_ARCHITECTURE v12 section 12. Scope is
deliberately narrow: fetch a transaction receipt by hash, decode the exchange
events we have verified, and cache. It performs no address analysis, holds no
state across runs beyond the receipt cache, and never writes to chain.

Event signatures are verified by keccak-256 at import (`_assert_signatures`),
not trusted from a document. The classic Polymarket CTF Exchange ABI does NOT
apply to this tape: the exchange at 0xe111180000d2663c0091e4f400237545b87b996b
replaces (makerAssetId, takerAssetId) with one asset id plus a uint8 side enum.

Self-test:  python3 live/pm_research/da_feeds_polygon.py --selftest
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from Crypto.Hash import keccak

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data/pm_5min/onchain/receipts"

# Public full nodes. Receipts are not state, so archive depth is irrelevant and
# no API key is required. Ordered by observed reliability; failover is linear.
ENDPOINTS: tuple[str, ...] = (
    "https://polygon-bor-rpc.publicnode.com",
    "https://polygon.drpc.org",
)

EXCHANGE = "0xe111180000d2663c0091e4f400237545b87b996b"

USER_AGENT = "ctaNew-pm-research/1.0 (research; read-only receipt fetch)"

ORDER_FILLED_SIG = (
    "OrderFilled(bytes32,address,address,uint8,uint256,uint256,uint256,"
    "uint256,bytes32,bytes32)"
)
ORDERS_MATCHED_SIG = "OrdersMatched(bytes32,address,uint8,uint256,uint256,uint256)"

ORDER_FILLED_TOPIC = (
    "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee"
)
ORDERS_MATCHED_TOPIC = (
    "0x174b3811690657c217184f89418266767c87e4805d09680c39fc9c031c0cab7c"
)

# USDC and the CTF outcome tokens are both 6-decimal on Polymarket.
UNIT = 1_000_000


class RpcError(RuntimeError):
    """Transport or JSON-RPC level failure, after retries across endpoints."""


class DecodeError(ValueError):
    """A log did not decode under the verified ABI. Never silently tolerated."""


def keccak256_text(text: str) -> str:
    h = keccak.new(digest_bits=256)
    h.update(text.encode())
    return "0x" + h.hexdigest()


def _assert_signatures() -> None:
    """Fail at import if the hard-coded topics ever drift from their source."""
    for sig, topic in (
        (ORDER_FILLED_SIG, ORDER_FILLED_TOPIC),
        (ORDERS_MATCHED_SIG, ORDERS_MATCHED_TOPIC),
    ):
        got = keccak256_text(sig)
        if got != topic:
            raise AssertionError(f"topic drift for {sig}: {got} != {topic}")


_assert_signatures()


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


def _words(data: str) -> list[str]:
    body = data[2:] if data.startswith("0x") else data
    if len(body) % 64:
        raise DecodeError(f"log data is not a whole number of words: {len(body)} hex")
    return [body[i : i + 64] for i in range(0, len(body), 64)]


def _addr(topic: str) -> str:
    if len(topic) != 66:
        raise DecodeError(f"malformed topic {topic!r}")
    return "0x" + topic[-40:]


@dataclass(frozen=True, slots=True)
class OrdersMatched:
    """The taker order's aggregate fill. One per matched taker order."""

    taker_order_hash: str
    taker_order_maker: str  # the address that submitted the TAKER order
    side_enum: int  # 0 / 1; the enum-to-label mapping is a hypothesis, not a fact
    asset_id: int
    maker_amount_filled: int
    taker_amount_filled: int
    log_index: int

    @property
    def implied_direction(self) -> str:
        """BUY or SELL, derived from WHICH leg of the amount pair is USDC.

        An order's `makerAmount` is what its creator GIVES and `takerAmount` is
        what it RECEIVES, so the pair is ordered differently by direction:

            BUY   taker gives USDC, receives tokens -> price = maker/taker
            SELL  taker gives tokens, receives USDC -> price = taker/maker

        A prediction-market price lies in (0, 1], so exactly one reading is in
        range. That identifies direction from the amounts ALONE -- without
        trusting `side_enum` and without consulting any off-chain field, which
        is what keeps the G-FF1 comparison non-circular.
        """
        if self.maker_amount_filled <= 0 or self.taker_amount_filled <= 0:
            raise DecodeError("non-positive fill amount; direction undefined")
        if self.maker_amount_filled == self.taker_amount_filled:
            # price 1.0 both ways: genuinely ambiguous, never guessed.
            raise DecodeError("equal fill amounts; direction not identified")
        return "BUY" if self.maker_amount_filled < self.taker_amount_filled else "SELL"

    @property
    def size(self) -> float:
        """Token quantity, whichever leg of the pair holds it."""
        if self.implied_direction == "BUY":
            return self.taker_amount_filled / UNIT
        return self.maker_amount_filled / UNIT

    @property
    def price(self) -> float:
        """Price per share in (0, 1)."""
        if self.implied_direction == "BUY":
            return self.maker_amount_filled / self.taker_amount_filled
        return self.taker_amount_filled / self.maker_amount_filled


@dataclass(frozen=True, slots=True)
class OrderFilled:
    """One maker leg of a match. A transaction may carry several."""

    order_hash: str
    maker: str
    taker: str
    side_enum: int
    asset_id: int
    maker_amount_filled: int
    taker_amount_filled: int
    fee: int
    log_index: int


def decode_orders_matched(log: Mapping[str, Any]) -> OrdersMatched:
    topics = log["topics"]
    if len(topics) != 3 or topics[0] != ORDERS_MATCHED_TOPIC:
        raise DecodeError("not an OrdersMatched log")
    w = _words(log["data"])
    if len(w) != 4:
        raise DecodeError(f"OrdersMatched expects 4 data words, got {len(w)}")
    return OrdersMatched(
        taker_order_hash=topics[1],
        taker_order_maker=_addr(topics[2]),
        side_enum=int(w[0], 16),
        asset_id=int(w[1], 16),
        maker_amount_filled=int(w[2], 16),
        taker_amount_filled=int(w[3], 16),
        log_index=int(log["logIndex"], 16),
    )


def decode_order_filled(log: Mapping[str, Any]) -> OrderFilled:
    topics = log["topics"]
    if len(topics) != 4 or topics[0] != ORDER_FILLED_TOPIC:
        raise DecodeError("not an OrderFilled log")
    w = _words(log["data"])
    if len(w) != 7:
        raise DecodeError(f"OrderFilled expects 7 data words, got {len(w)}")
    return OrderFilled(
        order_hash=topics[1],
        maker=_addr(topics[2]),
        taker=_addr(topics[3]),
        side_enum=int(w[0], 16),
        asset_id=int(w[1], 16),
        maker_amount_filled=int(w[2], 16),
        taker_amount_filled=int(w[3], 16),
        fee=int(w[4], 16),
        log_index=int(log["logIndex"], 16),
    )


def _exchange_logs(receipt: Mapping[str, Any], topic: str) -> Iterator[Mapping[str, Any]]:
    for log in receipt.get("logs", ()):
        if log["address"].lower() != EXCHANGE.lower():
            continue
        if log["topics"] and log["topics"][0] == topic:
            yield log


def orders_matched(receipt: Mapping[str, Any]) -> list[OrdersMatched]:
    return [decode_orders_matched(g) for g in _exchange_logs(receipt, ORDERS_MATCHED_TOPIC)]


def orders_filled(receipt: Mapping[str, Any]) -> list[OrderFilled]:
    return [decode_order_filled(g) for g in _exchange_logs(receipt, ORDER_FILLED_TOPIC)]


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


class PolygonRPC:
    """Receipt reader with an on-disk cache and linear endpoint failover.

    The cache is keyed by transaction hash and is the reason a rerun of any
    probe is byte-reproducible without re-querying the chain.
    """

    def __init__(
        self,
        endpoints: Sequence[str] = ENDPOINTS,
        cache_dir: Path = CACHE,
        min_interval_s: float = 0.12,
        timeout_s: float = 20.0,
        max_attempts: int = 4,
    ) -> None:
        if not endpoints:
            raise ValueError("at least one endpoint is required")
        self.endpoints = tuple(endpoints)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = float(min_interval_s)
        self.timeout_s = float(timeout_s)
        self.max_attempts = int(max_attempts)
        self._last_call = 0.0
        self.calls = 0
        self.cache_hits = 0

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, tx_hash: str) -> Path:
        h = tx_hash.lower()
        return self.cache_dir / h[2:4] / f"{h}.json"

    def cached(self, tx_hash: str) -> dict[str, Any] | None:
        p = self._cache_path(tx_hash)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None  # a torn cache entry is refetched, never trusted

    def _store(self, tx_hash: str, receipt: Mapping[str, Any]) -> None:
        p = self._cache_path(tx_hash)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp-{time.time_ns()}")
        tmp.write_text(json.dumps(receipt))
        tmp.replace(p)  # atomic; a reader never sees a partial receipt

    # -- transport -----------------------------------------------------------

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _post(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._throttle()
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            # Public nodes 403 urllib's default User-Agent; curl gets through.
            headers={"content-type": "application/json", "user-agent": USER_AGENT},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode())
        self.calls += 1
        if "error" in body:
            raise RpcError(f"{endpoint}: {body['error']}")
        return body

    def _call(self, method: str, params: list[Any]) -> Any:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            endpoint = self.endpoints[attempt % len(self.endpoints)]
            try:
                return self._post(endpoint, payload)["result"]
            except (urllib.error.URLError, TimeoutError, RpcError, OSError,
                    json.JSONDecodeError) as exc:
                last = exc
                time.sleep(min(2.0 ** attempt * 0.25, 4.0))
        raise RpcError(f"{method} failed after {self.max_attempts} attempts: {last}")

    # -- API -----------------------------------------------------------------

    def block_number(self) -> int:
        return int(self._call("eth_blockNumber", []), 16)

    def receipt(self, tx_hash: str, use_cache: bool = True) -> dict[str, Any]:
        """Fetch one receipt. Raises RpcError if the node returns null."""
        if use_cache:
            hit = self.cached(tx_hash)
            if hit is not None:
                self.cache_hits += 1
                return hit
        result = self._call("eth_getTransactionReceipt", [tx_hash])
        if result is None:
            raise RpcError(f"no receipt for {tx_hash} (unknown or not yet mined)")
        self._store(tx_hash, result)
        return result


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------

# Captured from the design-data transaction. Kept inline so decoding is tested
# without a network call, and so a change in the ABI breaks the test loudly.
_FIXTURE_ORDERS_MATCHED = {
    "address": EXCHANGE,
    "logIndex": "0x15",
    "topics": [
        ORDERS_MATCHED_TOPIC,
        "0xfc9b42e2fd6aff6d57f90f32c5c3102f13b215af005e0e7bbad7e8424bd7e49e",
        "0x000000000000000000000000e9678b6f9830a5b41f941581b1137711af234b05",
    ],
    "data": "0x"
    "0000000000000000000000000000000000000000000000000000000000000000"
    "89524c37efc0b6c2bfcbd9085ab6f6d858f9eee7f54afa8907ac201d61a33039"
    "00000000000000000000000000000000000000000000000000000000002c0bc8"
    "0000000000000000000000000000000000000000000000000000000000565d60",
}

_FIXTURE_ORDER_FILLED = {
    "address": EXCHANGE,
    "logIndex": "0xe",
    "topics": [
        ORDER_FILLED_TOPIC,
        "0xd233e4ae23fd4e9e45363c03b18267aec23d8deecff0d1594acbf2f02af6a3d1",
        "0x0000000000000000000000000abc8aa0247074b3281cf82f9657ab5a7d23a05f",
        "0x000000000000000000000000e9678b6f9830a5b41f941581b1137711af234b05",
    ],
    "data": "0x"
    "0000000000000000000000000000000000000000000000000000000000000000"
    "6ffdb99913f4b22e0a53ea2b70b61eea1c4558727be65a8e8e03feb316982553"
    "000000000000000000000000000000000000000000000000000000000025d528"
    "00000000000000000000000000000000000000000000000000000000004d35a0"
    "0000000000000000000000000000000000000000000000000000000000000000"
    "0000000000000000000000000000000000000000000000000000000000000000"
    "0000000000000000000000000000000000000000000000000000000000000000",
}


def _expect(label: str, exc: type[BaseException], fn) -> None:
    try:
        fn()
    except exc:
        return
    except BaseException as other:  # noqa: BLE001
        raise AssertionError(f"{label}: expected {exc.__name__}, got {other!r}") from other
    raise AssertionError(f"{label}: expected {exc.__name__}, nothing raised")


def selftest() -> int:
    checks = 0

    def ok(cond: bool, label: str) -> None:
        nonlocal checks
        if not cond:
            raise AssertionError(label)
        checks += 1

    # 1-2: signatures are derived, not asserted from a doc.
    ok(keccak256_text(ORDER_FILLED_SIG) == ORDER_FILLED_TOPIC, "OrderFilled topic")
    ok(keccak256_text(ORDERS_MATCHED_SIG) == ORDERS_MATCHED_TOPIC, "OrdersMatched topic")

    # 3: a control -- a wrong signature must NOT produce the topic. Without
    # this, check 1 would pass against any hash function that returned it.
    ok(keccak256_text(ORDER_FILLED_SIG + " ") != ORDER_FILLED_TOPIC, "topic control")

    # 4-9: OrdersMatched decodes to the values read off the design transaction.
    om = decode_orders_matched(_FIXTURE_ORDERS_MATCHED)
    ok(om.taker_order_maker == "0xe9678b6f9830a5b41f941581b1137711af234b05", "om taker")
    ok(om.side_enum == 0, "om side enum")
    ok(
        om.asset_id
        == 62112267755987661111025482512929736133058926416708079136710252533642621562937,
        "om asset id",
    )
    ok(om.taker_amount_filled == 5_660_000, "om taker amount")
    ok(abs(om.size - 5.66) < 1e-9, "om size")
    ok(abs(om.price - 0.51) < 5e-4, "om price")
    ok(om.implied_direction == "BUY", "om direction from amounts")

    # The SELL leg orders the amount pair the OTHER way round. Decoding it with
    # the BUY reading yields size 0.0042 at price 4.76 -- a price above 1, which
    # is impossible in a prediction market. That impossibility is exactly what
    # identifies direction, and getting it wrong silently cost 274 of 500 rows
    # on the gff1_v1 run.
    sell = OrdersMatched("0x0", "0xabc", 1, 99, 20_000, 4_200, 0)
    ok(sell.implied_direction == "SELL", "sell direction from amounts")
    ok(abs(sell.size - 0.02) < 1e-9, f"sell size, got {sell.size}")
    ok(abs(sell.price - 0.21) < 5e-4, f"sell price, got {sell.price}")
    _expect("equal amounts are ambiguous", DecodeError,
            lambda: OrdersMatched("0x0", "0xabc", 0, 1, 500, 500, 0).implied_direction)
    _expect("zero amount refuses", DecodeError,
            lambda: OrdersMatched("0x0", "0xabc", 0, 1, 0, 500, 0).implied_direction)

    # 10-13: OrderFilled names maker and taker separately -- the whole point.
    of = decode_order_filled(_FIXTURE_ORDER_FILLED)
    ok(of.maker == "0x0abc8aa0247074b3281cf82f9657ab5a7d23a05f", "of maker")
    ok(of.taker == "0xe9678b6f9830a5b41f941581b1137711af234b05", "of taker")
    ok(of.maker != of.taker, "of maker != taker")
    ok(of.fee == 0, "of fee")

    # 14-15: the two events describe different legs of one transaction. If a
    # future refactor made them agree, the leg-selection rule would be untested.
    ok(of.asset_id != om.asset_id, "legs carry different assets")
    ok(of.taker == om.taker_order_maker, "OrderFilled.taker is the OrdersMatched maker")

    # 16-19: malformed input REFUSES rather than returning a plausible number.
    _expect("odd data", DecodeError, lambda: _words("0x1234"))
    _expect(
        "wrong topic",
        DecodeError,
        lambda: decode_orders_matched({**_FIXTURE_ORDERS_MATCHED,
                                       "topics": [ORDER_FILLED_TOPIC, "0x00", "0x00"]}),
    )
    _expect(
        "short data",
        DecodeError,
        lambda: decode_orders_matched({**_FIXTURE_ORDERS_MATCHED, "data": "0x" + "00" * 32}),
    )
    _expect(
        "zero taker amount",
        DecodeError,
        lambda: OrdersMatched("0x0", "0x0", 0, 1, 5, 0, 0).price,
    )

    # 20-21: log filtering is address-scoped, so an unrelated contract emitting
    # the same topic cannot enter the sample.
    receipt = {"logs": [_FIXTURE_ORDERS_MATCHED, _FIXTURE_ORDER_FILLED]}
    ok(len(orders_matched(receipt)) == 1 and len(orders_filled(receipt)) == 1, "filter")
    foreign = {"logs": [{**_FIXTURE_ORDERS_MATCHED, "address": "0x" + "11" * 20}]}
    ok(orders_matched(foreign) == [], "foreign address excluded")

    # 22: an empty endpoint list is a construction error, not a runtime surprise.
    _expect("no endpoints", ValueError, lambda: PolygonRPC(endpoints=()))

    print(f"da_feeds_polygon selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--receipt", help="fetch and summarise one transaction hash")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.receipt:
        rpc = PolygonRPC()
        rec = rpc.receipt(args.receipt)
        print(f"block {int(rec['blockNumber'], 16)} status {rec['status']} "
              f"logs {len(rec['logs'])}")
        for om in orders_matched(rec):
            print(f"  OrdersMatched  side={om.side_enum} asset={om.asset_id} "
                  f"size={om.size} price={om.price:.4f} taker={om.taker_order_maker}")
        for of in orders_filled(rec):
            print(f"  OrderFilled    side={of.side_enum} maker={of.maker} "
                  f"taker={of.taker} fee={of.fee}")
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
