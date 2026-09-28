"""Moved from the session scratchpad on 2026-09-28 so every quoted number has its producing
code in the repo (CLAUDE.md rules 10/12). Run from movewatch/: python3 -m research.compare_v1_v2"""
"""Before (v1) vs after (v2, bugs fixed) for all three studies. Read-only."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "data" / "replay"
ld = lambda p: json.loads((R / p).read_text())  # noqa: E731


def ci(x, scale=1.0, nd=3):
    if not x or x.get("ci") is None:
        return "n/a" if not x else f"{x.get('ic', x.get('mean', float('nan')))*scale:+.{nd}f}"
    v = x.get("ic", x.get("mean"))
    return f"{v*scale:+.{nd}f} [{x['ci'][0]*scale:+.{nd}f},{x['ci'][1]*scale:+.{nd}f}]"


print("=" * 100, "\n1) POST-SIGNAL (live strict signals, Binance USDM) — v1 = mean LOG return x100; v2 = arithmetic mean of SIMPLE returns %")
a, b = ld("post_signal/summary.v1.json"), ld("post_signal/summary.json")
for key in ("delay1|aug|NEW|pump", "delay1|sep01_19|NEW|pump", "delay1|aug|NEW|dump", "delay1|sep01_19|NEW|dump",
            "delay1|aug|REVERSAL|pump", "delay1|sep01_19|REVERSAL|pump"):
    for h in ("r60", "r1440"):
        print(f"  {key:28s} {h:6s} v1 {ci(a[key][h], 100, 2):26s} v2 {ci(b[key][h], 100, 2):26s} n {a[key]['n']}->{b[key]['n']}"
              f"  hit1h {a[key]['hit_rate_1h']}->{b[key]['hit_rate_1h']}")
    print(f"  {'':28s} v2 medians: 1h {100*(b[key].get('r60_median') or float('nan')):+.1f}%  24h {100*(b[key].get('r1440_median') or float('nan')):+.1f}%")

print("=" * 100, "\n2) SIGNAL FEATURES vs 4h outcome — IC (Aug) / IC (Sep 1-19)")
for tag in ("signal_features", "signal_features_strict"):
    a, b = ld(f"{tag}/results.v1.json"), ld(f"{tag}/results.json")
    print(f"  [{tag}] n {a['n']} -> {b['n']} (tradfi excluded {b.get('tradfi_excluded')})")
    for pop in ("all", "pumps"):
        if f"aug|{pop}" not in a:
            continue
        nulls = (a[f"aug|{pop}"].get("null_max_abs_ic_95"), b[f"aug|{pop}"].get("null_max_abs_ic_95"))
        print(f"   {pop}: family null Aug {nulls[0]} -> {nulls[1]}")
        for f in ("oi_chg_1h", "oi_chg_4h", "top_ls", "top_ls_chg_4h", "acct_ls", "taker_ls_1h", "mkt_breadth", "premium", "extension", "log_liq"):
            va, vb = a[f"aug|{pop}"]["features"][f]["x240"], b[f"aug|{pop}"]["features"][f]["x240"]
            sa, sb = a[f"sep01_19|{pop}"]["features"][f]["x240"], b[f"sep01_19|{pop}"]["features"][f]["x240"]
            print(f"     {f:15s} Aug {ci(va):24s} -> {ci(vb):24s} | Sep {ci(sa):24s} -> {ci(sb)}")

print("=" * 100, "\n3) PERSISTENCE (pumps, 72h)")
a, b = ld("persistence/results.v1.json"), ld("persistence/results.json")
print("  status v2:", json.dumps(b["status"]))
for w in ("discovery", "holdout"):
    print(f"  {w:9s} pumps v1 {a[f'describe_{w}_pumps']}")
    print(f"  {w:9s} pumps v2 {b[f'describe_{w}_pumps']}")
    print(f"  {w:9s} dumps v2 {b[f'describe_{w}_dumps']}")
print("  nulls v1 static", a["pumps"]["null_static"], "| v2", b["pumps"]["nulls"])
print(f"  {'feature':18s} {'v1 disc':>22s} {'v2 disc':>22s} {'v1 hold':>22s} {'v2 hold':>22s}  verdict v1 -> v2")
for f in a["pumps"]["table"]:
    va, vb = a["pumps"]["table"][f], b["pumps"]["table"][f]
    print(f"  {f:18s} {ci(va['discovery']):>22s} {ci(vb['discovery']):>22s} {ci(va['holdout']):>22s} {ci(vb['holdout']):>22s}  {va['verdict']} -> {vb['verdict']}")
for f in ("vol_persist", "hold_4h", "runup_72h", "trigger_h", "oi_chg_24h", "liq_pre"):
    vb = b["pumps"]["table"][f]
    print(f"\n  [{f}] groups holdout: {json.dumps(vb['groups_holdout'])}")
    print(f"  [{f}] groups discovery: {json.dumps(vb['groups_discovery'])}")
    if vb.get("robustness_holdout"):
        print(f"  [{f}] robustness holdout: {json.dumps(vb['robustness_holdout'])}")
        print(f"  [{f}] robustness discovery: {json.dumps(vb['robustness_discovery'])}")
print("\n  combined features v2:", b.get("combined_features"), "| holdout:", json.dumps(b.get("combined_holdout")))
