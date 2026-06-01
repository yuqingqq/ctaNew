"""Two-book combiner for the convexity champion forward test.

Reads two per-book cycles.csv (BookA = flow model on flow-syms; BookB = price model on
non-flow-syms), 50/50 PnL-combines them, and writes the combined equity/summary.

Combine semantics (matches the validated +3.71 reproduction, 2026-05-31):
  - 'both'  : combine only cycles where BOTH books are active (the documented +3.71 figure;
              the optimistic bound — assumes capital is never idle).
  - 'fill0' : 0.5*A + 0.5*B with absent book = 0 (the conservative bound, ~+2.9; models
              idle capital when a book has no position that cycle).
Report both so the live read is honestly bracketed.

Usage:
  python3 live/convexity_twobook_combine.py \
      --book-a live/state/convexity_bookA/cycles.csv \
      --book-b live/state/convexity_bookB/cycles.csv \
      --out    live/state/convexity_twobook
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

ANN = np.sqrt(6 * 365)  # 6 cycles/day, annualized


def _sharpe(pnl_bps: pd.Series) -> float:
    p = np.asarray(pnl_bps, float) / 1e4
    return float(p.mean() / p.std() * ANN) if p.std() > 0 else float("nan")


def _maxdd(pnl_bps: pd.Series) -> float:
    c = pd.Series(np.asarray(pnl_bps, float)).cumsum()
    return float((c - c.cummax()).min()) if len(c) else float("nan")


def combine(book_a: Path, book_b: Path, out: Path) -> dict:
    A = pd.read_csv(book_a); B = pd.read_csv(book_b)
    A["open_time"] = pd.to_datetime(A["open_time"]); B["open_time"] = pd.to_datetime(B["open_time"])
    m = (A[["open_time", "pnl_bps"]].merge(
            B[["open_time", "pnl_bps"]], on="open_time", suffixes=("_a", "_b"), how="outer")
         .sort_values("open_time"))
    both = m.dropna()
    comb_both = 0.5 * both["pnl_bps_a"] + 0.5 * both["pnl_bps_b"]
    comb_fill = 0.5 * m["pnl_bps_a"].fillna(0) + 0.5 * m["pnl_bps_b"].fillna(0)
    corr = float(np.corrcoef(both["pnl_bps_a"], both["pnl_bps_b"])[0, 1]) if len(both) > 2 else float("nan")

    out.mkdir(parents=True, exist_ok=True)
    eq = m[["open_time"]].copy()
    eq["pnl_bps_combined_fill0"] = comb_fill.values
    eq["equity_fill0"] = 10_000.0 * (1 + comb_fill.values / 1e4).cumprod()
    eq.to_csv(out / "twobook_equity.csv", index=False)

    summary = dict(
        cycles_total=int(len(m)), cycles_both_active=int(len(both)),
        book_pnl_corr=corr,
        sharpe_both_active=_sharpe(comb_both), totPnL_both_active=float(comb_both.sum()),
        maxDD_both_active=_maxdd(comb_both),
        sharpe_fill0=_sharpe(comb_fill), totPnL_fill0=float(comb_fill.sum()),
        maxDD_fill0=_maxdd(comb_fill),
        sharpe_bookA=_sharpe(A["pnl_bps"]), sharpe_bookB=_sharpe(B["pnl_bps"]),
        start=str(m["open_time"].min()), end=str(m["open_time"].max()),
    )
    (out / "twobook_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--book-a", required=True)
    ap.add_argument("--book-b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s = combine(Path(args.book_a), Path(args.book_b), Path(args.out))
    print(f"TWO-BOOK combined: both-active Sharpe={s['sharpe_both_active']:+.3f} "
          f"(totPnL {s['totPnL_both_active']:+.0f}bps, DD {s['maxDD_both_active']:.0f}) | "
          f"fill0 Sharpe={s['sharpe_fill0']:+.3f} | "
          f"bookA={s['sharpe_bookA']:+.2f} bookB={s['sharpe_bookB']:+.2f} corr={s['book_pnl_corr']:.2f}")
