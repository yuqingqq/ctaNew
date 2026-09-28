"""Moved from the session scratchpad on 2026-09-28 so every quoted number has its producing
code in the repo (CLAUDE.md rules 10/12). Run from movewatch/: python3 -m research.recheck_conclusions"""
"""Recheck all conclusions on the fixed (v3) outputs; verdicts computed from each study's declared criterion."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "data" / "replay"
ld = lambda p: json.loads((R / p).read_text())  # noqa: E731
excl0 = lambda x: bool(x and x.get("ci") and (x["ci"][0] > 0 or x["ci"][1] < 0))  # noqa: E731


def fm(x, s=1.0, nd=3):
    if not x:
        return "n/a"
    v = x.get("ic", x.get("mean"))
    if v is None or v != v:
        return "n/a"
    return f"{v*s:+.{nd}f} [{x['ci'][0]*s:+.{nd}f},{x['ci'][1]*s:+.{nd}f}]{'*' if excl0(x) else ''}" if x.get("ci") else f"{v*s:+.{nd}f}"


print("1) AFTER A LIVE-SETTING SIGNAL (entry 1 min after the alert; direction-adjusted; % simple returns; * = CI excl 0)")
b = ld("post_signal/summary.json")
for w in ("aug", "sep01_19"):
    for g in ("NEW|pump", "NEW|dump", "REVERSAL|pump", "REVERSAL|dump"):
        k = f"delay1|{w}|{g}"
        t = b[k]
        print(f"  {w:8s} {g:14s} n={t['n']:3d} hit1h={t['hit_rate_1h']}  1h mean {fm(t['r60'],100,2):24s} median {100*(t['r60_median'] or 0):+6.2f}%"
              f"  | 24h mean {fm(t['r1440'],100,2):26s} median {100*(t['r1440_median'] or 0):+6.2f}%")

print("\n2) EXTRA METRICS vs 4h OUTCOME — declared criterion: Aug |IC| > family null AND CI excl 0, then Sep 1-19 same sign (CI excl 0 = replicated)")
for tag in ("signal_features", "signal_features_strict"):
    r = ld(f"{tag}/results.json")
    for pop in ("all", "pumps"):
        null = r[f"aug|{pop}"].get("null_max_abs_ic_95")
        hits = []
        for f, v in r[f"aug|{pop}"]["features"].items():
            a, s = v["x240"], r[f"sep01_19|{pop}"]["features"][f]["x240"]
            if null is not None:
                passes = excl0(a) and abs(a["ic"]) > null
            else:
                passes = excl0(a)            # strict population: no null computed -> nominal only
            if passes:
                same = s["ic"] == s["ic"] and a["ic"] * s["ic"] > 0
                verdict = "REPLICATED" if same and excl0(s) else ("same sign, n.s." if same else "sign flips")
                hits.append(f"{f} Aug {fm(a)} Sep {fm(s)} -> {verdict}")
        label = f"null {null:.3f}" if null else "no null (nominal)"
        print(f"  [{tag} · {pop} · {label}]")
        for h in hits or ["(nothing passes the August step)"]:
            print("     ", h)

print("\n3) PERSISTENCE (72h)")
p = ld("persistence/results.json")
print("  status:", json.dumps(p["status"]))
for w in ("discovery", "holdout"):
    print(f"  {w:9s} pumps {p[f'describe_{w}_pumps']}")
    print(f"  {w:9s} dumps {p[f'describe_{w}_dumps']}")
print("  nulls:", {k: round(v, 3) for k, v in p["pumps"]["nulls"].items()})
for f, v in p["pumps"]["table"].items():
    if v["verdict"] != "no":
        print(f"  {f:14s} disc {fm(v['discovery']):26s} hold {fm(v['holdout']):26s} -> {v['verdict']}")
v = p["pumps"]["table"]["vol_persist"]
print("  vol_persist holdout groups:", json.dumps(v["groups_holdout"]))
print("  vol_persist discovery groups:", json.dumps(v["groups_discovery"]))
print("  vol_persist robustness holdout:", json.dumps(v.get("robustness_holdout")))
print("  combined:", p.get("combined_features"), json.dumps(p.get("combined_holdout")))
