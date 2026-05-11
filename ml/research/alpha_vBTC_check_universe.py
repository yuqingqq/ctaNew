"""Check the full available universe on Binance USDM perp and Hyperliquid.

Goal: identify tokens that exist on both exchanges (training on Binance,
execution on HL) but aren't in our local 39-symbol data dir.

Output: candidate names ranked by 24h volume, with availability flags.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path

import requests
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import list_universe

OUT_DIR = REPO / "outputs/vBTC_check_universe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_binance_usdm_perps() -> pd.DataFrame:
    """Get all live USDM perpetual contracts + 24h volume.

    Uses CoinGecko's binance_futures endpoint (Binance fapi is geo-blocked
    from many regions). The endpoint returns up-to-date USDM perp tickers
    with 24h converted USD volume.
    """
    print("Fetching Binance USDM perps via CoinGecko (fapi geo-blocked)...")
    url = "https://api.coingecko.com/api/v3/derivatives/exchanges/binance_futures?include_tickers=all"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    tickers = data.get("tickers", [])
    rows = []
    for t in tickers:
        if t.get("contract_type") != "perpetual": continue
        if t.get("target") != "USDT": continue
        if t.get("expired_at") is not None: continue   # delisted
        sym = t.get("symbol", "")
        # CoinGecko already returns the symbol in BASEUSDT form (matches Binance's form)
        cv = t.get("converted_volume", {})
        # converted_volume.usd is sometimes a string
        usd_vol = float(cv.get("usd", 0)) if cv else 0
        rows.append({
            "symbol": sym,
            "base": t.get("base"),
            "volume_usdt": usd_vol,
            "open_interest_usd": float(t.get("open_interest_usd") or 0),
            "funding_rate": float(t.get("funding_rate") or 0),
            "bid_ask_spread": float(t.get("bid_ask_spread") or 0),
        })
    df = pd.DataFrame(rows).sort_values("volume_usdt", ascending=False)
    print(f"  Binance USDM perps (CoinGecko, non-delisted): {len(df)}")
    return df


def fetch_hyperliquid_perps() -> pd.DataFrame:
    """Get all HL perp markets + 24h volume."""
    print("Fetching Hyperliquid metadata...")
    meta = requests.post("https://api.hyperliquid.xyz/info",
                          json={"type": "metaAndAssetCtxs"}, timeout=15).json()
    universe_meta = meta[0]["universe"]
    asset_ctxs = meta[1]
    rows = []
    for u, ctx in zip(universe_meta, asset_ctxs):
        coin = u["name"]
        # HL uses bare names (BTC, ETH, etc.)
        # Map to USDT-equivalent (BTC → BTCUSDT for matching)
        rows.append({
            "hl_coin": coin,
            "hl_symbol": coin + "USDT",
            "max_leverage": u.get("maxLeverage", 0),
            "day_volume_usd": float(ctx.get("dayNtlVlm", 0)),
            "open_interest_usd": float(ctx.get("openInterest", 0)) * float(ctx.get("markPx", 0)),
            "is_delisted": u.get("isDelisted", False),
        })
    df = pd.DataFrame(rows)
    # Drop xyz: prefixed entries (US equities) — only crypto perps for our purpose
    df = df[~df["hl_coin"].str.startswith("xyz:")].copy()
    df = df[~df["is_delisted"]].copy()
    df = df.sort_values("day_volume_usd", ascending=False)
    return df


def main():
    # 1. Local universe
    local_universe = sorted(list_universe(min_days=200))
    print(f"\nLocal data universe: {len(local_universe)} symbols")

    # 2. Binance USDM
    binance_df = fetch_binance_usdm_perps()
    print(f"  Binance USDM perps total: {len(binance_df)}")

    # 3. Hyperliquid
    hl_df = fetch_hyperliquid_perps()
    print(f"  Hyperliquid perps (crypto only, non-delisted): {len(hl_df)}")

    # 4. Cross-reference
    print("\n" + "=" * 90)
    print("CROSS-REFERENCE")
    print("=" * 90)
    binance_syms = set(binance_df["symbol"])
    hl_syms = set(hl_df["hl_symbol"])
    local_set = set(local_universe)

    in_both = binance_syms & hl_syms
    in_binance_only = binance_syms - hl_syms
    in_hl_only = hl_syms - binance_syms
    print(f"\n  On BOTH Binance USDM and Hyperliquid: {len(in_both)} symbols")
    print(f"  Binance USDM ∩ HL ∩ local data:        {len(in_both & local_set)}")
    print(f"  Binance USDM ∩ HL minus local data:    {len(in_both - local_set)}  ← potential additions")

    # 5. Top candidates: on both exchanges but not in local data
    candidates = sorted(in_both - local_set)
    candidate_df = binance_df[binance_df["symbol"].isin(candidates)].merge(
        hl_df.rename(columns={"hl_symbol": "symbol"})[["symbol", "day_volume_usd", "max_leverage"]],
        on="symbol", how="left"
    )
    candidate_df = candidate_df.sort_values("volume_usdt", ascending=False)

    print("\n" + "=" * 90)
    print(f"TOP 40 CANDIDATES (on both Binance + HL, NOT in our local data)")
    print("=" * 90)
    print(f"  {'symbol':<14} {'binance_24h_vol_$M':>20} {'hl_24h_vol_$M':>15} "
          f"{'hl_max_lev':>10}")
    for _, r in candidate_df.head(40).iterrows():
        print(f"  {r['symbol']:<14} {r['volume_usdt']/1e6:>20.1f} "
              f"{r['day_volume_usd']/1e6:>15.1f} {r['max_leverage']:>10.0f}")

    # 6. Also report what's in our local data but maybe not optimal
    local_in_both = sorted(local_set & in_both)
    local_in_both_df = binance_df[binance_df["symbol"].isin(local_in_both)].sort_values("volume_usdt", ascending=False)
    print("\n" + "=" * 90)
    print(f"OUR LOCAL UNIVERSE — coverage on both exchanges")
    print("=" * 90)
    print(f"  Local symbols on both Binance + HL: {len(local_in_both)}/{len(local_universe)}")
    in_local_not_hl = sorted(local_set - hl_syms)
    if in_local_not_hl:
        print(f"  Local symbols NOT on HL (can't trade): {in_local_not_hl}")
    in_local_not_binance = sorted(local_set - binance_syms)
    if in_local_not_binance:
        print(f"  Local symbols NOT on Binance: {in_local_not_binance}")

    # 7. Recommended liquidity floor analysis
    print("\n" + "=" * 90)
    print("CANDIDATES BY LIQUIDITY FLOOR")
    print("=" * 90)
    for vol_floor_m in [500, 200, 100, 50, 30, 10]:
        n = (candidate_df["volume_usdt"] / 1e6 >= vol_floor_m).sum()
        n_hl = ((candidate_df["volume_usdt"] / 1e6 >= vol_floor_m) &
                (candidate_df["day_volume_usd"] / 1e6 >= 5)).sum()
        print(f"  >= ${vol_floor_m}M binance + >= $5M HL: {n_hl} candidates "
              f"(>= ${vol_floor_m}M binance only: {n})")

    # Save
    candidate_df.to_csv(OUT_DIR / "candidates_to_add.csv", index=False)
    binance_df.to_csv(OUT_DIR / "all_binance_usdm.csv", index=False)
    hl_df.to_csv(OUT_DIR / "all_hyperliquid.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
