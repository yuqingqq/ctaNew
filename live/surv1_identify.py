"""surv1_identify (pre-reg addendum 54, step 1): identify TRUE delistings among the 585 USDT perps
absent from the survivor panel. Metadata-only pass over Binance Vision klines listings — first/last
trade month per symbol, classify delisted (last trade << panel-end 2026-06) vs still-trading-unselected
(illiquid, fell below our gate but data exists, e.g. FTTUSDT). No kline downloads here; step 2 downloads
ONLY the delisted-and-would-qualify subset. Concurrent listing, monthly primary + daily fallback for
short-lived names.
"""
import re, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
HOST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
PANEL_END_MO = "2026-05"   # last-trade >= this => still-trading (reached panel end)

def _list(prefix, pat):
    url = f"{HOST}?delimiter=/&prefix={prefix}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            x = r.read().decode()
    except Exception:
        return []
    return sorted(set(re.findall(pat, x)))

def date_range(sym):
    # monthly 1h first (light); fall back to daily 1h for sub-month lifespans
    mo = _list(f"data/futures/um/monthly/klines/{sym}/1h/", rf"{re.escape(sym)}-1h-(\d{{4}}-\d{{2}})\.zip</Key>")
    if mo:
        return sym, mo[0], mo[-1], len(mo), "monthly"
    dd = _list(f"data/futures/um/daily/klines/{sym}/1h/", rf"{re.escape(sym)}-1h-(\d{{4}}-\d{{2}})-\d{{2}}\.zip</Key>")
    if dd:
        return sym, dd[0], dd[-1], len(set(dd)), "daily"
    return sym, None, None, 0, "none"

def main():
    syms = [s.strip() for s in open(SD / "absent_nolocal.txt") if s.strip()]
    print(f"probing {len(syms)} absent USDT perps for trade-date range...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=24) as ex:
        for i, r in enumerate(ex.map(date_range, syms), 1):
            rows.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(syms)}", flush=True)
    delisted, still, empty = [], [], []
    for sym, fst, lst, n, src in rows:
        if src == "none" or lst is None:
            empty.append(sym); continue
        (still if lst >= PANEL_END_MO else delisted).append((sym, fst, lst, n))
    delisted.sort(key=lambda t: t[2])
    print(f"\n===== classification of {len(syms)} absent USDT perps =====")
    print(f"  TRUE DELISTINGS (last trade < {PANEL_END_MO}): {len(delisted)}")
    print(f"  still-trading-but-unselected (illiquid, data exists): {len(still)}")
    print(f"  no archive found (skip): {len(empty)}")
    # delisting-year distribution
    from collections import Counter
    yr = Counter(lst[:4] for _, _, lst, _ in delisted)
    print(f"  delisting-year dist: {dict(sorted(yr.items()))}")
    # write full table
    with open(SD / "delisted_table.csv", "w") as f:
        f.write("symbol,first_month,last_month,n_months,status\n")
        for sym, fst, lst, n in delisted:
            f.write(f"{sym},{fst},{lst},{n},delisted\n")
        for sym, fst, lst, n in still:
            f.write(f"{sym},{fst},{lst},{n},still_trading\n")
    open(SD / "delisted_syms.txt", "w").write("\n".join(s for s, *_ in delisted))
    print(f"\n  wrote {SD/'delisted_table.csv'} and delisted_syms.txt ({len(delisted)} delisted)")
    # show the longest-lived delistings (most likely to have been in-universe & have real crash history)
    longlived = sorted(delisted, key=lambda t: -t[3])[:25]
    print(f"\n  longest-lived delistings (n_months desc) — top in-universe candidates:")
    for sym, fst, lst, n in longlived:
        print(f"    {sym:<18} {fst} -> {lst}  ({n} mo)")
    print("SURV1DONE")

if __name__ == "__main__":
    main()
