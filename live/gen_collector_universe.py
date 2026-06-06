"""Generate live/collector_universe.txt — the canonical symbol list the exec-server WS collector should feed.

Why the FULL universe (not just the traded low-vol book): bars_since_high_xs_rank is ranked over the whole
panel (175-XS, validated 2026-06-06), so the live cross-section must see every symbol or the live feature
drifts from the backtest. We drop only DURABLY-DEAD names (delisted/halted) — a dead symbol has no WS feed to
collect anyway, and its absence from the cross-section is immaterial (1/175). Trade-gate liveness is faster
(7d, in convexity_paper_bot); this collector list uses a more conservative 14d window so a brief halt doesn't
churn the feed list.

Regenerate at each monthly retrain (kept in git so the exec box git-pulls it instead of hand-editing).
Usage: python3 live/gen_collector_universe.py
"""
import sys; from pathlib import Path
import pandas as pd
REPO = Path("/home/yuqing/ctaNew"); sys.path.insert(0, str(REPO))
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OUT = REPO/"live/collector_universe.txt"
FLAT_WIN_DAYS = 14
FLAT_MAX_FRAC = 0.85    # >85% flat-price days over trailing window = durably dead/halted -> not feedable

def main():
    pan = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    end = pan["open_time"].max()
    recent = pan[pan["open_time"] >= end - pd.Timedelta(days=FLAT_WIN_DAYS)]
    flat_frac = recent.groupby("symbol")["return_pct"].apply(lambda s: (s.abs() < 1e-12).mean())
    allsyms = sorted(pan["symbol"].unique())
    dead = sorted(s for s in allsyms if flat_frac.get(s, 0.0) > FLAT_MAX_FRAC)
    alive = [s for s in allsyms if s not in dead]
    OUT.write_text("\n".join(alive) + "\n")
    print(f"panel {len(allsyms)} syms | DROPPED dead/halted ({len(dead)}): {dead}")
    print(f"wrote {OUT.name}: {len(alive)} live-feed symbols (as-of {end.date()})")

if __name__ == "__main__":
    main()
