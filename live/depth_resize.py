"""Depth-aware (slippage-budget) position resizer — the capacity optimization, wired for the LIVE forward test.

WHY: the +4.22 backtest is a small-size artifact. At real AUM the strategy concentrates in THIN names whose taker
impact (median ~25 bps/leg, up to 140+ on back-loaded books) crushes realistic Sharpe to ~+1.2. The fix is to size
each name to what its LIVE Hyperliquid book can absorb cheaply, keeping the full cross-section (the alpha is
illiquidity-bound — can't filter to liquid names) and staying dollar-neutral.

HOW (runs between `convexity_paper_bot --decide` and the realfill, env-gated by the caller):
  1. read decision.json (net_after = signed weights, turnover = the trade)
  2. per leg, fetch the live HL L2 book and compute its SLIPPAGE BUDGET = the largest notional whose walk-the-book
     taker slippage <= DEPTH_CAP_BPS (this is liquidity-PLACEMENT aware; total-depth fractions over-trade back-loaded
     books — SOPH books $10k at 26bps, so its honest budget is tiny, not 20% of a fat-but-deep book).
  3. desired_notional = |net_after| * DEPTH_AUM ; scale = min(1, budget/desired) ; capped_net_after = net_after*scale
  4. BALANCE: scale each side to the thinner side's total so Sigma long == Sigma short (stay neutral — per-leg capping
     alone would leave net directional exposure, e.g. long 10k / short 8k = +2k).
  5. rewrite net_after (capped+balanced) and turnover (= capped_net_after - prev_agg, prev recovered from the file).

PIT-correct by construction (the current book IS known at decision time). Cannot be backtested (no historical L2) →
this is the live-forward instrument; realfill measures the realized post-slippage PnL. FAIL-SAFE: any error leaves
decision.json byte-unchanged. OFF unless DEPTH_AWARE_SIZING=1.

  DEPTH_AWARE_SIZING=1 DEPTH_AUM=1000000 DEPTH_CAP_BPS=10 python3 live/depth_resize.py --state live/state/convexity_v2/state
"""
import argparse, json, os, sys, logging
import concurrent.futures as cf
from pathlib import Path
sys.path.insert(0, "/home/yuqing/ctaNew")
from live.convexity_slippage import fetch_hl_l2_book, simulate_taker_fill, HL_INFO
import requests

log = logging.getLogger("depth_resize")
DEPTH_AUM   = float(os.environ.get("DEPTH_AUM", "1000000"))     # deploy size to model the cap at ($)
DEPTH_CAP_BPS = float(os.environ.get("DEPTH_CAP_BPS", "10"))    # per-leg taker-slippage cap (bps)
DEPTH_MAX_NOTIONAL = float(os.environ.get("DEPTH_MAX_NOTIONAL", "2000000"))  # search ceiling per name


def budget_from_book(book: dict, side: str, cap_bps: float, hi: float = DEPTH_MAX_NOTIONAL) -> float:
    """Largest notional whose taker walk-the-book slippage <= cap_bps (binary search over the tested fill sim)."""
    lo = 100.0
    s_lo = abs(simulate_taker_fill(book, side, lo)["slippage_bps"])
    if not (s_lo == s_lo):          # NaN book
        return 0.0
    if s_lo > cap_bps:              # even a tiny clip already too costly
        return 0.0
    if abs(simulate_taker_fill(book, side, hi)["slippage_bps"]) <= cap_bps:
        return hi                   # book deeper than our ceiling
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        s = abs(simulate_taker_fill(book, side, mid)["slippage_bps"])
        if s == s and s <= cap_bps: lo = mid
        else: hi = mid
    return lo


def b2hl(sym: str, hl_coins: set) -> str | None:
    base = sym[:-4] if sym.endswith("USDT") else sym
    if base.startswith("1000"):
        for cand in ("k" + base[4:], base[4:]):
            if cand in hl_coins: return cand
    return base if base in hl_coins else None


def depth_aware_resize(net_after: dict, turnover: dict, budgets: dict, aum: float) -> tuple[dict, dict, list]:
    """Pure core (no I/O): cap each leg to its $ budget at `aum`, then balance sides. Returns (net_after', turnover', diag).
    budgets[sym] = max |notional| ($) for that name; missing/None => uncapped (fail-safe)."""
    prev = {s: float(net_after.get(s, 0.0)) - float(turnover.get(s, 0.0)) for s in set(net_after) | set(turnover)}
    capped, diag = {}, []
    for s, w in net_after.items():
        w = float(w)
        if abs(w) < 1e-12:
            capped[s] = w; continue
        b = budgets.get(s)
        desired = abs(w) * aum
        if b is None or desired <= 0:
            capped[s] = w; diag.append((s, desired, b, 1.0)); continue
        scale = min(1.0, b / desired)
        capped[s] = w * scale
        diag.append((s, desired, b, scale))
    # balance sides to the thinner side's gross (stay dollar-neutral)
    L = sum(v for v in capped.values() if v > 0)
    S = -sum(v for v in capped.values() if v < 0)
    if L > 1e-12 and S > 1e-12:
        g = min(L, S)
        for s, v in capped.items():
            if v > 0:   capped[s] = v * g / L
            elif v < 0: capped[s] = v * g / S
    new_turn = {s: round(capped.get(s, 0.0) - prev.get(s, 0.0), 6)
                for s in set(capped) | set(prev) if abs(capped.get(s, 0.0) - prev.get(s, 0.0)) > 1e-9}
    return capped, new_turn, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="dir holding decision.json")
    ap.add_argument("--dry-run", action="store_true", help="compute + print, do NOT rewrite decision.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dpath = Path(a.state) / "decision.json"
    try:
        dec = json.loads(dpath.read_text())
        net_after = dec.get("net_after", {})
        turnover = dec.get("turnover", {})
        legs = {s: float(w) for s, w in net_after.items() if abs(float(w)) > 1e-12}
        if not legs:
            log.info("depth_resize: no legs to size — leaving decision.json unchanged"); return
        meta = requests.post(HL_INFO, json={"type": "meta"}, timeout=20).json()
        hl_coins = {x["name"] for x in meta["universe"]}
        coins = {s: b2hl(s, hl_coins) for s in legs}

        def budget(item):
            s, w = item
            coin = coins.get(s)
            if coin is None: return s, None
            try:
                bk = fetch_hl_l2_book(coin)
                side = "buy" if w > 0 else "sell"          # entry side
                return s, budget_from_book(bk, side, DEPTH_CAP_BPS)
            except Exception as e:
                log.info(f"  book fetch fail {s}->{coin}: {str(e)[:40]} — leaving uncapped"); return s, None

        with cf.ThreadPoolExecutor(max_workers=16) as ex:
            budgets = dict(ex.map(budget, legs.items()))

        new_net, new_turn, diag = depth_aware_resize(net_after, turnover, budgets, DEPTH_AUM)
        gross0 = sum(abs(float(w)) for w in net_after.values()) * DEPTH_AUM / 2
        gross1 = sum(abs(w) for w in new_net.values()) * DEPTH_AUM / 2
        log.info(f"depth_resize @ AUM ${DEPTH_AUM:,.0f}, cap {DEPTH_CAP_BPS:.0f}bps: per-side gross "
                 f"${gross0:,.0f} -> ${gross1:,.0f} ({gross1/gross0*100 if gross0 else 0:.0f}%); "
                 f"{sum(1 for *_ ,sc in diag if sc < 0.999)}/{len(diag)} legs capped")
        for s, des, b, sc in sorted(diag, key=lambda x: x[3])[:8]:
            log.info(f"    {s.replace('USDT',''):8s} desired ${des:>10,.0f}  budget ${(b or 0):>10,.0f}  scale {sc:.2f}")
        if a.dry_run:
            log.info("  (dry-run — decision.json NOT modified)"); return
        # write back, keeping every other field; snapshot the pre-resize file for audit
        (Path(a.state) / "decision_predepth.json").write_text(json.dumps(dec, indent=2))
        dec["net_after"] = {s: round(w, 8) for s, w in new_net.items()}
        dec["turnover"] = new_turn
        dec["depth_resized"] = dict(aum=DEPTH_AUM, cap_bps=DEPTH_CAP_BPS,
                                    gross_per_side_before=round(gross0, 2), gross_per_side_after=round(gross1, 2))
        dpath.write_text(json.dumps(dec, indent=2))
        log.info(f"  decision.json rewritten ({len(new_turn)} legs to execute). pre-resize -> decision_predepth.json")
    except Exception as e:                                  # FAIL-SAFE: never break the live cycle
        log.warning(f"depth_resize FAILED ({type(e).__name__}: {str(e)[:80]}) — decision.json left UNCHANGED")
        return


if __name__ == "__main__":
    main()
