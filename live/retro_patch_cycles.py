"""One-shot retro-patch for cycles.csv: rebuild gross_pnl_bps and net_bps
from hourly_pnl.csv for cycles affected by the last_marked_mid clobber bug
(fixed in paper_bot.py via last_cycle_mid).

For each cycle row with had_prev_positions=1, gross is recomputed as the
sum of hourly_pnl_bps over the prior cycle's holding window. Backup of
the original is written alongside.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

CYCLES_PATH = Path("live/state/cycles.csv")
HOURLY_PATH = Path("live/state/hourly_pnl.csv")
BACKUP_PATH = CYCLES_PATH.with_suffix(".csv.bak")


def main():
    cy = pd.read_csv(CYCLES_PATH)
    hr = pd.read_csv(HOURLY_PATH)
    cy["t"] = pd.to_datetime(cy["decision_time_utc"])
    hr["ts"] = pd.to_datetime(hr["ts_utc"])
    cy = cy.sort_values("t").reset_index(drop=True)

    if BACKUP_PATH.exists():
        print(f"Backup already exists at {BACKUP_PATH} — leaving untouched")
    else:
        shutil.copy(CYCLES_PATH, BACKUP_PATH)
        print(f"Backed up original cycles.csv -> {BACKUP_PATH}")

    print("\n=== Retro-patching ===")
    for i in range(len(cy)):
        row = cy.iloc[i]
        if row.get("had_prev_positions", 0) != 1:
            print(f"  cycle {i} ({row['decision_time_utc']}): skip (no prev positions)")
            continue
        prev_t = cy["t"].iloc[i - 1]
        cur_t = row["t"]
        win = hr[(hr["ts"] > prev_t) & (hr["ts"] < cur_t)]
        if win.empty:
            print(f"  cycle {i}: no hourly snapshots in window — skip")
            continue
        true_gross = float(win["hourly_pnl_bps"].sum())
        old_gross = float(row["gross_pnl_bps"])
        new_net = true_gross - float(row["fees_bps"]) - float(row["slippage_bps"]) - float(row["funding_bps"])
        old_net = float(row["net_bps"])
        cy.loc[i, "gross_pnl_bps"] = true_gross
        cy.loc[i, "net_bps"] = new_net
        print(f"  cycle {i} ({row['decision_time_utc']}, {len(win)} hourly ticks):")
        print(f"      gross: {old_gross:+.2f} -> {true_gross:+.2f} bps  "
              f"(Δ {true_gross - old_gross:+.2f})")
        print(f"      net:   {old_net:+.2f} -> {new_net:+.2f} bps  "
              f"(Δ {new_net - old_net:+.2f})")

    cy.drop(columns=["t"]).to_csv(CYCLES_PATH, index=False)
    print(f"\nWrote patched {CYCLES_PATH}")


if __name__ == "__main__":
    sys.exit(main() or 0)
