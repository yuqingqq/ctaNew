"""Cent-exact comparator for the Phase-0 reproduction gate.

WHY A COMPARATOR AND NOT A DIFF. BE's R-156(2) work added three ADDITIVE keys
to `evaluate_policy`'s output (`n_actions`, `rows_per_action`, `unit`). No
computed number changed -- verified against f1ceec9 -- but a whole-JSON diff of
a fresh receipt against the frozen one WILL show differences, and none of them
are numeric. Eyeballing that diff at the moment of truth is how a structural
change gets mistaken for a reproduction failure, or worse, how a real cent
difference gets waved through as "just the new fields".

So the gate compares the NAMED NUMBERS the manifest pins, field by field, and
says CENT_EXACT only when every one matches to the cent.

Rule 15: `selftest` proves it FLAGS a one-cent difference and PASSES identity.
A comparator that has never rejected anything is not a gate.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

MANIFEST = Path("/home/yuqing/ctaNew/data/pm_5min/derived/"
                "harmful_candidate_manifest_v1.json")
FIELDS = (("auc", "auc", 1e-9),
          ("n_generations", "n_generations", 0),
          ("net_cents_5pct", "net_cents", 0.005),
          ("harm_avoided_cents_5pct", "harm_avoided_cents", 0.005))


def extract(receipt: dict, coin: str, arm: str = "PM_PLUS_FINE") -> dict:
    a = receipt["paired_arms"][coin][arm]
    g = a.get("gate", {})
    b = (g.get("budgets") or {}).get("5%", {})
    return {"auc": a.get("auc"), "n_generations": g.get("n_generations"),
            "net_cents": b.get("net_cents"),
            "harm_avoided_cents": b.get("harm_avoided_cents")}


def compare(fresh: dict, targets: dict) -> dict:
    """targets: manifest's target_scores_to_reproduce block."""
    rows, worst = [], 0.0
    for coin, key in (("btc", "btc_PM_PLUS_FINE"), ("eth", "eth_PM_PLUS_FINE")):
        tgt = targets[key]
        got = extract(fresh, coin)
        for tname, gname, tol in FIELDS:
            t, g = tgt.get(tname), got.get(gname)
            if t is None or g is None:
                rows.append({"coin": coin, "field": tname, "target": t,
                             "got": g, "ok": False, "reason": "MISSING"})
                continue
            d = abs(float(g) - float(t))
            ok = d <= tol
            worst = max(worst, d if "cents" in tname else 0.0)
            rows.append({"coin": coin, "field": tname, "target": t, "got": g,
                         "abs_diff": d, "tol": tol, "ok": ok})
    return {"rows": rows, "all_ok": all(r["ok"] for r in rows),
            "worst_cent_diff": worst,
            "verdict": "CENT_EXACT" if all(r["ok"] for r in rows)
                       else "NOT_REPRODUCED"}


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    tgt = {"btc_PM_PLUS_FINE": {"auc": 0.692310, "n_generations": 171452,
                                "net_cents_5pct": 2492.200082000001,
                                "harm_avoided_cents_5pct": 9217.5027415},
           "eth_PM_PLUS_FINE": {"auc": 0.731839, "n_generations": 231721,
                                "net_cents_5pct": 131.69754650000002,
                                "harm_avoided_cents_5pct": 1878.7572594999995}}

    def mk(btc_net, extra=False):
        def arm(auc, n, net, harm):
            g = {"n_generations": n,
                 "budgets": {"5%": {"net_cents": net, "harm_avoided_cents": harm}}}
            if extra:            # the additive R-156(2) keys
                g.update({"n_actions": n, "rows_per_action": 1.7, "unit": "ACTION"})
            return {"auc": auc, "gate": g}
        return {"paired_arms": {
            "btc": {"PM_PLUS_FINE": arm(0.692310, 171452, btc_net, 9217.5027415)},
            "eth": {"PM_PLUS_FINE": arm(0.731839, 231721, 131.69754650000002,
                                        1878.7572594999995)}}}

    r = compare(mk(2492.200082000001), tgt)
    ok(r["verdict"] == "CENT_EXACT", "KNOWN-GOOD: identical numbers reproduce")

    # THE ADDITIVE KEYS MUST NOT AFFECT THE VERDICT -- the whole point
    r2 = compare(mk(2492.200082000001, extra=True), tgt)
    ok(r2["verdict"] == "CENT_EXACT",
       "the three additive R-156(2) keys do NOT break the comparison -- a "
       "structural change must never read as a reproduction failure")

    # POSITIVE CONTROL: one cent must be caught
    r3 = compare(mk(2492.210082000001), tgt)
    ok(r3["verdict"] == "NOT_REPRODUCED",
       "POSITIVE CONTROL: a ONE-CENT difference is FLAGGED, not absorbed")
    ok(any(x["field"] == "net_cents_5pct" and not x["ok"] for x in r3["rows"]),
       "and the failing field is NAMED, not just a boolean")

    # a missing field is a failure, never a silent pass
    r4 = compare({"paired_arms": {"btc": {"PM_PLUS_FINE": {"auc": 0.692310,
                  "gate": {}}}, "eth": {"PM_PLUS_FINE": {"auc": 0.731839,
                  "gate": {}}}}}, tgt)
    ok(r4["verdict"] == "NOT_REPRODUCED",
       "KNOWN-BAD REFUSED: absent numbers fail rather than compare equal")
    ok(any(x.get("reason") == "MISSING" for x in r4["rows"]),
       "and absence is reported as MISSING, distinct from a mismatch")

    print(f"repro_compare selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    if len(sys.argv) < 2:
        print("usage: repro_compare.py <fresh_receipt.json> | --selftest")
        return 2
    fresh = json.loads(Path(sys.argv[1]).read_text())
    tgt = json.loads(MANIFEST.read_text())["target_scores_to_reproduce"]
    r = compare(fresh, tgt)
    for x in r["rows"]:
        mark = "ok " if x["ok"] else "FAIL"
        print(f"  {mark} {x['coin']:<4} {x['field']:<24} "
              f"target={x['target']!r} got={x['got']!r}")
    print(f"\n  VERDICT: {r['verdict']}  worst cent diff {r['worst_cent_diff']:.6f}")
    return 0 if r["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
