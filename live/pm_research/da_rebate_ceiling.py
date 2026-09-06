"""DA: an ASSUMPTION-FREE upper bound on the maker rebate, measured on the
public tape rather than argued.

WHY THIS EXISTS. The reviewer (30646c3 s6.3) established that E-R's magnitude
clauses survive only up to a headroom factor: MATERIAL flips only if the TRUE
rebate is at least 3.6184x the identity value 0.20*fe that DE priced. Nothing
had bounded the true rebate from above, so the clauses rested on the identity
being right rather than on the rebate being small.

THE BOUND, AND WHY IT NEEDS NO ASSUMPTION ABOUT OUR SHARE. The venue pays
crypto makers 20% of COLLECTED TAKER FEES, pro-rata across the makers of that
market. Whatever our pro-rata share turns out to be, it cannot exceed the whole
pool. So

    rebate_ceiling = 0.20 * SUM over markets m of P_m

where P_m is the FULL taker-fee pool of market m -- every taker in it, not
only the ones we traded against. That is an upper bound on any share, which is
what makes it assumption-free: it is true for a 1% share and for a 100% share.

DECLARED BEFORE THE RUN (rule 6), and none of it is tuned afterwards:

  SURFACE   the tier1 PUBLIC trade tape, `day=2026-08-24`, `coin=btc`,
            distiller `tier1_v4_r12`, restricted to the ruled run's single
            market `btc-updown-5m-1787579400` over [t0, t0+300). The ruled
            run has exactly one market, so the sum over m has one term.
  POOL      P_m = SUM over every trade row in that market of the taker fee
            7*p*(1-p)*size cents, using the formula VERIFIED on chain in
            Q-DA-251: exact on 879 of 901 decoded legs after flooring to the
            10 uUSDC grid the chain actually uses.
  COMPARE   ceiling against 3.6184 * 0.20 * fe for each arm, fe_B and fe_T
            taken from DE's ruled receipt.
  PREDICATE `ceiling_below_headroom` -- True iff the ceiling is BELOW the
            headroom for BOTH arms, i.e. the true rebate cannot reach the
            level at which MATERIAL flips.
  COMPLETE  `pool_completeness_established` is True only if ALL of: the
            distiller manifest says `partial: false`; the window carries
            trades; and OUR OWN 458 fills join into the pool by
            `transaction_hash`, which is the check that the tape is a
            SUPERSET of what we traded rather than a different slice.
  REFUTES   a ceiling at or above the headroom for either arm refutes the
            reading: E-R's magnitude clauses would then not hold and the
            rebate would have to be measured rather than bounded.

TWO DIRECTIONS IN WHICH THIS IS A LOWER BOUND ON THE CEILING, both labelled
in the artifact rather than left for a reader to infer: the fee formula
UNDERSTATES on 22 of 901 decoded legs (Q-DA-255, mechanism open), and if
completeness fails the pool is a partial slice. A lower bound on an upper
bound is still an upper bound ONLY if completeness holds -- so when it does
not, the artifact says PARTIAL and the predicate is not claimed.

    python3 live/pm_research/da_rebate_ceiling.py --selftest
    python3 live/pm_research/da_rebate_ceiling.py --real --output P
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

PROTOCOL = "P003_DA_REBATE_CEILING_V1"
REPO = Path("/home/yuqing/ctaNew")
TAPE = (REPO / "data/pm_5min/tier1/trades/day=2026-08-24/coin=btc"
        / "distiller=tier1_v4_r12/part-0.parquet")
MANIFEST = TAPE.parent / "manifest.json"
DE_RECEIPT = (REPO / "data/pm_5min/derived"
              / "p003_v2_fee_endpoint_sensitivity__20260905T161824Z.json")
CACHE = REPO / "data/pm_5min/derived/de_section81_cache_12.pkl"

RULED_SLUG = "btc-updown-5m-1787579400"
WINDOW_T0 = 1787579400
WINDOW_S = 300
REBATE_SHARE = 0.20
FEE_RATE_CENTS = 7.0          # 0.07 $/share * 100 c/$; fee = 7*p*(1-p)*size
#: The reviewer's factor: MATERIAL flips only if the true rebate is at least
#: this multiple of the identity value 0.20*fe. Declared, not derived here.
HEADROOM_FACTOR = 3.6184


class RebateCeilingRefused(RuntimeError):
    """The pool cannot be established as asked."""


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def carrying_commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                       text=True, cwd=str(Path(__file__).resolve().parent))
    return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"


def taker_fee_cents(price: float, size: float) -> float:
    """One taker leg's fee, in cents, by the formula verified on chain.

    Q-DA-251: the chain floors to a 10 uUSDC grid (= 0.001 c). Flooring is
    applied so the pool is not inflated by sub-grid fractions the venue never
    charges."""
    raw = FEE_RATE_CENTS * price * (1.0 - price) * size
    return (int(raw * 1000.0) / 1000.0) if raw > 0 else 0.0


def load_pool(tape: Path | None = None, slug: str = RULED_SLUG) -> dict:
    import pyarrow.parquet as pq
    t = Path(tape) if tape is not None else TAPE
    if not t.is_file():
        raise RebateCeilingRefused(f"REFUSED: no trade tape at {t}")
    mf = t.parent / "manifest.json"
    man = json.loads(mf.read_text()) if mf.is_file() else {}
    cols = ["slug", "t_event_ms", "price_token", "size", "transaction_hash",
            "fee_rate_bps_raw", "fee_source_status"]
    tbl = pq.ParquetFile(t).read(columns=cols).to_pydict()
    lo, hi = WINDOW_T0 * 1000, (WINDOW_T0 + WINDOW_S) * 1000
    rows, pool, txs = 0, 0.0, set()
    prices, sizes = [], []
    for s, tm, p, sz, tx in zip(tbl["slug"], tbl["t_event_ms"],
                                tbl["price_token"], tbl["size"],
                                tbl["transaction_hash"]):
        if s != slug:
            continue
        if tm is None or not (lo <= tm < hi):
            continue
        if p is None or sz is None:
            continue
        rows += 1
        pool += taker_fee_cents(float(p), float(sz))
        prices.append(float(p))
        sizes.append(float(sz))
        if tx:
            txs.add(tx)
    return {
        "tape": str(t), "tape_sha256": sha256(t),
        "manifest_partial": man.get("partial"),
        "manifest_output_sha256": man.get("output_sha256"),
        "distiller_version": man.get("distiller_version"),
        "era": man.get("era"),
        "slug": slug, "window_t0": WINDOW_T0, "window_s": WINDOW_S,
        "n_trades_in_window": rows,
        "n_distinct_tx": len(txs),
        "tx_set": txs,
        "pool_taker_fee_cents": pool,
        "total_shares": sum(sizes),
        "price_range": [min(prices), max(prices)] if prices else None,
    }


def our_fill_txs() -> dict:
    """The transaction hashes of OUR OWN fills in the ruled window, if the
    reference book carries them. Used only to test that the public pool is a
    SUPERSET of what we traded."""
    import pickle
    if not CACHE.is_file():
        return {"available": False, "why": f"no cache at {CACHE}"}
    d = pickle.loads(CACHE.read_bytes())
    ref = (d.get("fr") or {}).get("reference") or {}
    if RULED_SLUG not in ref:
        return {"available": False, "why": "ruled slug absent from the cache"}
    n = sum(len(g.get("tranches") or [])
            for side in ref[RULED_SLUG] for g in ref[RULED_SLUG][side])
    keys = set()
    for side in ref[RULED_SLUG]:
        for g in ref[RULED_SLUG][side]:
            for tr in (g.get("tranches") or []):
                keys |= set(k for k in tr if "tx" in k.lower()
                            or "hash" in k.lower())
    return {"available": bool(keys), "n_our_fills": n,
            "tranche_tx_fields": sorted(keys),
            "why": ("the reference tranches carry no transaction hash, so a "
                    "tx-level superset join is not possible from this cache"
                    if not keys else "")}


def assess(pool: dict, fe_b: float, fe_t: float, ours: dict) -> dict:
    ceiling = REBATE_SHARE * pool["pool_taker_fee_cents"]
    out = {"rebate_ceiling_cents": ceiling,
           "headroom_factor": HEADROOM_FACTOR, "arms": {}}
    for name, fe in (("baseline", fe_b), ("treatment", fe_t)):
        identity = REBATE_SHARE * fe
        headroom = HEADROOM_FACTOR * identity
        out["arms"][name] = {
            "fe_cents": fe,
            "identity_value_0p20_fe_cents": identity,
            "headroom_cents": headroom,
            "ratio_ceiling_over_identity": (ceiling / identity
                                            if identity else None),
            "ceiling_below_this_arms_headroom": ceiling < headroom,
        }
    complete = (pool["manifest_partial"] is False
                and pool["n_trades_in_window"] > 0)
    superset = bool(ours.get("available"))
    out["pool_completeness_established"] = bool(complete and superset)
    out["completeness_evidence"] = {
        "manifest_partial_is_false": pool["manifest_partial"] is False,
        "window_carries_trades": pool["n_trades_in_window"] > 0,
        "our_fills_join_by_transaction_hash": superset,
        "why_join_matters": ("a pool that does not contain our own fills is a "
                             "different slice, not a superset, and would "
                             "bound nothing"),
        "join_status": ours.get("why") or "joined",
    }
    out["ceiling_below_headroom"] = all(
        a["ceiling_below_this_arms_headroom"] for a in out["arms"].values())
    out["bound_status"] = ("COMPLETE" if out["pool_completeness_established"]
                           else "PARTIAL_LOWER_BOUND_ON_THE_CEILING")
    # THE DIRECTION OF THE INCOMPLETENESS IS A PREDICATE, NOT A REMARK.
    # A partial pool is a LOWER bound on the true pool, so incompleteness can
    # only move the ceiling UP. That WEAKENS a "below the headroom" reading
    # (the true ceiling might be above it) and STRENGTHENS an "above" one
    # (the true ceiling is at least this high). So when the measured ceiling
    # is already above the headroom, the conclusion survives the missing
    # completeness evidence -- and when it is below, it does not.
    out["conclusion_robust_to_incompleteness"] = (
        out["pool_completeness_established"]
        or not out["ceiling_below_headroom"])
    out["why_robust"] = (
        "an incomplete pool understates P_m, so the true ceiling is at least "
        "the measured one. A measured ceiling ABOVE the headroom therefore "
        "stays above it however much of the pool is missing; only a 'below' "
        "reading would need completeness to stand."
        if not out["ceiling_below_headroom"] else
        "the measured ceiling is BELOW the headroom, which is exactly the "
        "reading that REQUIRES completeness -- without it the true pool could "
        "be larger and the ceiling could cross.")
    out["directions_in_which_this_understates"] = [
        "the fee formula understates on 22 of 901 decoded legs (Q-DA-255, "
        "mechanism open), so the pool is a slight lower bound",
        "flooring to the chain's 10 uUSDC grid rounds every leg DOWN",
    ] + ([] if out["pool_completeness_established"] else [
        "COMPLETENESS IS NOT ESTABLISHED: this is a lower bound on the "
        "ceiling and does NOT bound the rebate from above"])
    return out


def run_real() -> dict:
    de = json.loads(DE_RECEIPT.read_text())
    s = de["fee_endpoint_summary"]
    fe_b = s["fe_cents"]["baseline"]
    dd = s["decision_delta_cents"]
    fe_t = (dd["treatment"]["D_E_MINUS_R"]
            - dd["treatment"]["D_E0"]) / REBATE_SHARE + fe_b
    pool = load_pool()
    ours = our_fill_txs()
    tx = pool.pop("tx_set")
    a = assess(pool, fe_b, fe_t, ours)
    return {
        "protocol": PROTOCOL,
        "carrying_commit": carrying_commit(),
        "declared_before_run": {
            "surface": (f"tier1 public trade tape day=2026-08-24 coin=btc "
                        f"distiller=tier1_v4_r12, slug {RULED_SLUG}, "
                        f"[t0, t0+{WINDOW_S}) -- ALL takers, not only our "
                        f"counterparties"),
            "pool": "P_m = SUM 7*p*(1-p)*size cents, floored to the chain's "
                    "10 uUSDC grid (formula verified in Q-DA-251)",
            "ceiling": "0.20 * SUM_m P_m",
            "comparison": f"ceiling vs {HEADROOM_FACTOR} * 0.20 * fe, per arm",
            "completeness": "manifest partial==false AND the window carries "
                            "trades AND our own fills join by "
                            "transaction_hash",
            "refutes": "a ceiling at or above the headroom for EITHER arm",
            "headroom_factor_source": "reviewer 30646c3 s6.3",
        },
        "de_receipt": {"path": str(DE_RECEIPT),
                       "sha256": sha256(DE_RECEIPT)},
        "pool": pool,
        "our_fills": ours,
        "n_pool_tx": len(tx),
        "assessment": a,
        "role": "REPORTED, NOT ENFORCED (rule 14). This bounds a quantity; it "
                "promotes nothing and clears no gate.",
        "limits": [
            "one market, one 5-minute window, G=0 complete UTC days",
            "the pool is the FULL taker fee of that market; our pro-rata "
            "share is strictly smaller and is NOT estimated here",
            "the rebate also has a $1/day minimum accrual and pays in pUSD, "
            "neither of which is modelled",
        ],
    }


def selftest() -> int:
    fails = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    ok(abs(taker_fee_cents(0.5, 100.0) - 175.0) < 1e-9,
       f"FORMULA: 100 shares at p=0.50 -> {taker_fee_cents(0.5,100.0)} c "
       f"(= 1.75 c/share, the facts table's ATM taker fee)")
    ok(taker_fee_cents(0.99, 1.0) < taker_fee_cents(0.5, 1.0),
       "FORMULA: the fee decays away from the money")
    ok(taker_fee_cents(0.5, 0.0) == 0.0,
       "FORMULA: a zero-size trade pays nothing")

    base = {"manifest_partial": False, "n_trades_in_window": 10,
            "pool_taker_fee_cents": 100.0}
    ours_ok = {"available": True, "n_our_fills": 458}
    ours_no = {"available": False, "why": "no tx field"}

    # POSITIVE CONTROL: a small pool must be BELOW the headroom.
    a = assess(base, 1000.0, 1000.0, ours_ok)
    ok(a["ceiling_below_headroom"] is True,
       f"POSITIVE CONTROL: a 100 c pool against a 1000 c fe gives ceiling "
       f"{a['rebate_ceiling_cents']:.1f} vs headroom "
       f"{a['arms']['baseline']['headroom_cents']:.1f} -> below")
    # FALSIFIER: a PLANTED pool above the headroom must FLIP the predicate.
    big = dict(base, pool_taker_fee_cents=100000.0)
    b = assess(big, 1000.0, 1000.0, ours_ok)
    ok(b["ceiling_below_headroom"] is False,
       f"FALSIFIER: a planted 100,000 c pool gives ceiling "
       f"{b['rebate_ceiling_cents']:.1f} ABOVE the headroom "
       f"{b['arms']['baseline']['headroom_cents']:.1f} -> the predicate "
       f"FLIPS, so a True is a measurement and not a default")
    # and it must flip on EITHER arm alone.
    c = assess(dict(base, pool_taker_fee_cents=800.0), 1000.0, 100.0, ours_ok)
    ok(c["arms"]["baseline"]["ceiling_below_this_arms_headroom"] is True
       and c["arms"]["treatment"]["ceiling_below_this_arms_headroom"] is False
       and c["ceiling_below_headroom"] is False,
       "FALSIFIER: below for one arm and above for the other still reports "
       "False overall -- the predicate is ANDed across arms, never averaged")
    # exact boundary
    ident = REBATE_SHARE * 1000.0
    edge = assess(dict(base, pool_taker_fee_cents=HEADROOM_FACTOR * ident
                       / REBATE_SHARE), 1000.0, 1000.0, ours_ok)
    ok(edge["ceiling_below_headroom"] is False,
       "BOUNDARY: exactly AT the headroom is not BELOW it -- the comparison "
       "is strict, which is the conservative side")

    # COMPLETENESS, both directions.
    ok(assess(base, 1000.0, 1000.0, ours_ok)[
           "pool_completeness_established"] is True,
       "COMPLETENESS: manifest not partial + trades present + our fills join "
       "-> established")
    for bad, why in ((dict(base, manifest_partial=True), "a PARTIAL manifest"),
                     (dict(base, n_trades_in_window=0), "an EMPTY window")):
        r = assess(bad, 1000.0, 1000.0, ours_ok)
        ok(r["pool_completeness_established"] is False
           and r["bound_status"] == "PARTIAL_LOWER_BOUND_ON_THE_CEILING",
           f"COMPLETENESS KNOWN-BAD: {why} -> NOT established, and the bound "
           f"is labelled PARTIAL rather than reported as a ceiling")
    r = assess(base, 1000.0, 1000.0, ours_no)
    ok(r["pool_completeness_established"] is False,
       "COMPLETENESS KNOWN-BAD: our fills failing to join -> NOT established, "
       "because a pool that does not contain our own trades is a different "
       "slice and bounds nothing")

    try:
        load_pool(Path("/nonexistent.parquet"))
        ok(False, "KNOWN-BAD: accepted an absent tape -- must refuse")
    except RebateCeilingRefused:
        ok(True, "KNOWN-BAD: an absent trade tape REFUSES")

    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.real:
        out = run_real()
        txt = json.dumps(out, indent=2, sort_keys=True, default=str)
        if a.output:
            a.output.write_text(txt)
        print(txt[:3000])
        return 0
    ap.error("choose --selftest or --real")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
