"""Rank long-tail sweep candidates against the baseline, applying the acceptance criteria.

Baseline (frozen v3): ALL sh 2.829 tot 22380 dd -4937 | BULL dd -4937 tot 3662 |
                      BEAR_GRIND tot -1391 sh -1.541 dd -2977 | BEAR_DEEP tot 4440 sh 14.089 | SIDE tot 15669

A candidate is INTERESTING for:
  squeeze : reduces |BULL_dd| (and ALL_dd) meaningfully, with ALL_sh not materially worse (>= base-0.10)
            and ALL_tot not crushed (>= base-1500).
  grind   : raises BEAR_GRIND_tot / BEAR_GRIND_sh, WITHOUT hurting BEAR_DEEP (tot >= base-500) and
            ALL_sh >= base-0.10.
Prints a ranked table with delta columns and a PASS/interest flag. Robustness (placebo/nested-OOS)
is applied only to the flagged survivors in the next stage.
"""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/home/yuqing/ctaNew/live/state/longtail")
BASE = dict(ALL_sh=2.829, ALL_tot=22380, ALL_dd=-4937, BULL_dd=-4937, BULL_tot=3662,
            BEAR_GRIND_tot=-1391, BEAR_GRIND_sh=-1.541, BEAR_DEEP_tot=4440, SIDE_tot=15669)

def main():
    led = pd.read_csv(ROOT / "ledger.csv")
    led = led[led["ok"] == 1].drop_duplicates("tag", keep="last")
    for k, v in BASE.items():
        led[f"d_{k}"] = led[k] - v
    # interest flags
    squeeze = ((led["ALL_dd"] - BASE["ALL_dd"] > 400) &        # DD shallower by >400 bps
               (led["d_ALL_sh"] >= -0.10) & (led["d_ALL_tot"] >= -1500))
    grind = ((led["d_BEAR_GRIND_tot"] > 200) & (led["d_BEAR_DEEP_tot"] >= -500) &
             (led["d_ALL_sh"] >= -0.10))
    led["flag"] = np.where(squeeze & grind, "BOTH", np.where(squeeze, "squeeze",
                    np.where(grind, "grind", "")))
    cols = ["tag", "wave", "ALL_sh", "d_ALL_sh", "ALL_tot", "ALL_dd", "d_ALL_dd",
            "BULL_dd", "BEAR_GRIND_tot", "d_BEAR_GRIND_tot", "BEAR_GRIND_sh",
            "BEAR_DEEP_tot", "d_BEAR_DEEP_tot", "SIDE_tot", "flag"]
    led = led.rename(columns={"d_ALL_dd": "d_ALL_dd"})
    led["d_ALL_dd"] = led["ALL_dd"] - BASE["ALL_dd"]
    show = led[[c for c in cols if c in led.columns]].copy()
    print(f"BASELINE  ALL_sh {BASE['ALL_sh']:+.3f}  ALL_tot {BASE['ALL_tot']}  ALL_dd {BASE['ALL_dd']}  "
          f"BULL_dd {BASE['BULL_dd']}  GRIND_tot {BASE['BEAR_GRIND_tot']}  DEEP_tot {BASE['BEAR_DEEP_tot']}\n")
    # sort: flagged first, then by ALL_sh
    show["_rank"] = show["flag"].map({"BOTH": 0, "squeeze": 1, "grind": 1, "": 2})
    show = show.sort_values(["_rank", "ALL_sh"], ascending=[True, False]).drop(columns="_rank")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(show.to_string(index=False))
    flagged = show[show["flag"] != ""]
    print(f"\n{len(flagged)} flagged candidate(s): {list(flagged['tag'])}")

if __name__ == "__main__":
    main()
