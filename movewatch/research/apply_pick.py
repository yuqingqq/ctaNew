"""Write a replay's picked thresholds into config/movewatch.yaml (the `rules:` block only),
with a provenance comment, so the live config always traces to a calibration run.

  python3 -m research.apply_pick data/replay/cal_v4/summary.json              # the declared pick
  python3 -m research.apply_pick data/replay/cal_v4/summary.json --budget 80  # best August point
                                                                              # within 80 msgs/day
--budget re-picks mechanically from the run's grid.csv (calibration columns only).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "movewatch.yaml"


def rules_block(summary: dict, src: str) -> str:
    m = summary["multipliers"]
    lines = [f"rules:                        # from {src}: floor x{m['floor']}, z x{m['z']}, v x{m['v']},"
             f" liq >= ${summary['liq_min_usd'] / 1e6:.0f}M (pick on calibration only; docs/CALIBRATION.md)",
             f"  liq_min_usd: {int(summary['liq_min_usd'])}",
             "  market_liq_usd: 5000000     # instruments counted in the market median",
             "  horizons:                   # floor_pct: |move net of market| ; zmin: |z| ; vmin: volume multiple"]
    for h in ("1m", "5m", "15m", "1h", "4h", "24h"):
        r = summary["rules"][h]
        lines.append(f"    {h + ':':4s} {{floor_pct: {r['floor_pct']:.2f}, zmin: {r['zmin']:.2f}, vmin: {r['vmin']:.2f}}}")
    lines.append("  market_floor_pct: {15m: 2.5, 1h: 4.0, 4h: 6.0}")
    return "\n".join(lines) + "\n\n"


def repick(summary: dict, grid_csv: Path, budget: float) -> dict:
    from movewatch.signals import DEFAULT_RULES, H_NAME
    g = pd.read_csv(grid_csv)
    ok = g[g["cal_msgs_per_day"] <= budget]
    if ok.empty:
        raise SystemExit(f"no grid point within {budget} msgs/day")
    b = ok.sort_values(["cal_good", "cal_msgs_per_day"], ascending=[False, True]).iloc[0]
    rules = DEFAULT_RULES.scaled(float(b["floor_mult"]), float(b["z_mult"]), float(b["v_mult"]))
    out = dict(summary)
    out["multipliers"] = {"floor": float(b["floor_mult"]), "z": float(b["z_mult"]), "v": float(b["v_mult"])}
    out["liq_min_usd"] = float(b["liq_min"])
    out["rules"] = {H_NAME[h]: asdict(r) for h, r in rules.horizons.items()}
    print(f"re-picked within {budget} msgs/day:", b.to_dict())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--budget", type=float, default=None, help="messages/day cap for a mechanical re-pick")
    args = ap.parse_args()
    src = args.summary
    summary = json.loads(Path(src).read_text())
    if args.budget is not None:
        summary = repick(summary, Path(src).resolve().parent / "grid.csv", args.budget)
    text = CFG.read_text()
    new = re.sub(r"^rules:.*?(?=^\S)", rules_block(summary, str(Path(src).resolve().parent.relative_to(ROOT))),
                 text, count=1, flags=re.S | re.M)
    if new == text:
        raise SystemExit("rules: block not found or unchanged")
    CFG.write_text(new)
    print(rules_block(summary, src))


if __name__ == "__main__":
    main()
