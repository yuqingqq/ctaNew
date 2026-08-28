"""Fresh settlement-convention audit on RECENT windows.

PRE-REGISTERED AT Q-DA-145 BEFORE ANY RECENT-WINDOW NUMBER EXISTED: population,
grid, boundary reader, tie rule, gate, and the three readings were all declared
first. Authorisation: coordinator dispatch, which is what R-110's operative
clause requires (`COORDINATION.md:13423-13424`).

WHY THIS EXISTS. Amendment A2 adopted `S60(T) >= S60(t0)` on the strength of
`EXP_RESULTS_2026-08-20.md:10-17` (n=1,465, 99.8%), against a market description
that reads as a FULL-RANGE average. I could adopt the reconstruction on evidence
but could not EXPLAIN the disagreement, and two of the three candidate
explanations are testable on fresh data:

  (i)  endpoint passes, full-window fails -> A2 confirmed, description still
       unexplained but the convention is stable;
  (ii) endpoint FAILS -> the convention changed after 08-20, A2's escape hatch
       fires and the estimand is re-derived before any freeze;
  (iii) full-window PASSES -> the 08-20 population was unrepresentative.

WHAT IS SHARED AND WHAT IS NOT, on purpose. The declared BOUNDARY READER
(`load_streams`, `read_at`, `mean_over`) is imported from `exp_m6_settlement`
unchanged: the audit must read the boundary the same way or it is auditing a
different convention, and that script is not my surface to edit. The POPULATION
SELECTION and the TALLY are implemented here, because those are what a day-split
audit changes -- and sharing them would make the equivalence control vacuous.

EQUIVALENCE CONTROL, run BEFORE the recent split is read: on the IDENTICAL full
population this harness must reproduce the original script's table. Agreement is
evidence precisely because the tally is not shared.

NOTHING HERE IS A CHALLENGER SCORE. No Brier, no skill, no comparison against
`Identity`. This is a settlement-convention audit of the same kind EXP-M6 was.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from live.pm_research.exp_m6_settlement import (      # noqa: E402
    COINS, PM, load_streams, mean_over, read_at)

# ---- everything below was fixed at pre-registration (Q-DA-145) -------------
CONVENTIONS = [
    ("S60(T) vs S60(t0)", 60, "point", 60),
    ("S30(T) vs S30(t0)", 30, "point", 30),
    ("S60(T) vs S30(t0)", 60, "point", 30),
    ("meanS60[t0,T] vs S60(t0)", 60, "mean", 60),
]
RECENT_DAYS = ("2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27")
GATE_POOLED = 0.99
GATE_BIG_MARGIN = 0.995
BIG_MARGIN_BP = 0.5
MIN_POWERED = 400


def load_population():
    markets, res = {}, {}
    for ln in open(PM / "markets.jsonl"):
        try:
            m = json.loads(ln)
            markets[m["slug"]] = m
        except Exception:
            pass
    for ln in open(PM / "resolutions.jsonl"):
        try:
            r = json.loads(ln)
            if r.get("closed") is True and r.get("winners"):
                res[r["slug"]] = r["winners"]
        except Exception:
            pass
    return markets, res


def day_of(m):
    return dt.datetime.fromtimestamp(
        m["window_start"], dt.timezone.utc).strftime("%Y-%m-%d")


def audit(markets, res, streams, days=None):
    """Tally the declared grid over a population. `days=None` is ALL days --
    the equivalence control; a day tuple is the pre-registered recent split."""
    ev = {c[0]: {"n": 0, "hit": 0, "margins": []} for c in CONVENTIONS}
    kn = {c[0]: {"n": 0, "hit": 0} for c in CONVENTIONS}
    used, skipped = set(), 0
    for slug, winners in sorted(res.items()):
        m = markets.get(slug)
        if not m:
            continue
        if days is not None and day_of(m) not in days:
            continue
        sym = COINS.get(m["coin"])
        t0, T = m["window_start"] * 1000, m["window_end"] * 1000
        up_won = bool(winners.get("Up"))
        counted = False
        for name, wT, mode, w0 in CONVENTIONS:
            sT, s0 = streams.get((sym, wT)), streams.get((sym, w0))
            if not sT or not s0:
                continue
            for by_known, store in ((False, ev), (True, kn)):
                xT = (mean_over(sT, t0, T, by_known) if mode == "mean"
                      else read_at(sT, T, by_known)[0])
                x0 = read_at(s0, t0, by_known)[0]
                if xT is None or x0 is None:
                    if not by_known:
                        skipped += 1
                    continue
                pred_up = xT >= x0
                store[name]["n"] += 1
                store[name]["hit"] += int(pred_up == up_won)
                if not by_known:
                    counted = True
                    store[name]["margins"].append(
                        (abs(xT - x0) / x0 * 1e4, pred_up == up_won))
        if counted:
            used.add(slug)
    return ev, kn, len(used), skipped


def table(ev, kn, label):
    print(f"\n=== {label} ===")
    print(f"{'convention':<28} {'n':>6} {'agree':>8} {'agree|>0.5bp':>13} "
          f"{'knowledge-time':>15}  gate")
    out = {}
    for name, _, _, _ in CONVENTIONS:
        t = ev[name]
        if not t["n"]:
            continue
        acc = t["hit"] / t["n"]
        big = [ok for mg, ok in t["margins"] if mg > BIG_MARGIN_BP]
        accb = (sum(big) / len(big)) if big else float("nan")
        k = kn[name]
        kacc = (k["hit"] / k["n"]) if k["n"] else float("nan")
        # GATE EVALUATED, NEVER PRINTED AS A CONCLUSION (rule 10).
        passed = (acc >= GATE_POOLED and (accb == accb)
                  and accb >= GATE_BIG_MARGIN and t["n"] >= MIN_POWERED)
        print(f"{name:<28} {t['n']:>6} {acc:>7.1%} {accb:>12.1%} "
              f"{kacc:>14.1%}  {'PASS' if passed else 'fail'}")
        out[name] = {"n": t["n"], "agree": acc, "agree_big": accb,
                     "knowledge_time_agree": kacc, "gate_pass": passed,
                     "n_big_margin": len(big)}
    return out


def main() -> int:
    markets, res = load_population()
    streams = load_streams()
    print(f"[audit] markets={len(markets)} resolved={len(res)} "
          f"streams={len(streams)}")

    # ---- EQUIVALENCE CONTROL FIRST, before any recent number is read ------
    ev_all, kn_all, n_all, skip_all = audit(markets, res, streams, days=None)
    full = table(ev_all, kn_all,
                 f"EQUIVALENCE CONTROL — ALL days, windows={n_all}, "
                 f"skipped={skip_all} (must match exp_m6_settlement)")

    ev_r, kn_r, n_r, skip_r = audit(markets, res, streams, days=RECENT_DAYS)
    recent = table(ev_r, kn_r,
                   f"PRE-REGISTERED RECENT SPLIT — {RECENT_DAYS[0]}.."
                   f"{RECENT_DAYS[-1]}, windows={n_r}, skipped={skip_r}")

    ep, fw = "S60(T) vs S60(t0)", "meanS60[t0,T] vs S60(t0)"
    r_ep, r_fw = recent.get(ep), recent.get(fw)
    reading = "INDETERMINATE"
    if r_ep and r_fw:
        if r_ep["gate_pass"] and not r_fw["gate_pass"]:
            reading = "(i) A2 CONFIRMED on fresh data"
        elif not r_ep["gate_pass"] and not r_fw["gate_pass"]:
            reading = "(ii) ENDPOINT FAILS — A2's escape hatch fires"
        elif r_fw["gate_pass"] and not r_ep["gate_pass"]:
            reading = "(iii) FULL-WINDOW passes — 08-20 population unrepresentative"
        else:
            reading = "(iv) BOTH pass — the grid does not discriminate here"
    print(f"\nPRE-REGISTERED READING: {reading}")
    print("(Q-DA-145 declared all four before any of these numbers existed.)")
    Path(PM / "derived" / "da_settlement_audit_v1.json").write_text(
        json.dumps({"pre_registration": "Q-DA-145",
                    "recent_days": list(RECENT_DAYS),
                    "gate": {"pooled": GATE_POOLED, "big": GATE_BIG_MARGIN,
                             "big_margin_bp": BIG_MARGIN_BP,
                             "min_powered": MIN_POWERED},
                    "equivalence_control_all_days": full,
                    "recent": recent, "n_windows_all": n_all,
                    "n_windows_recent": n_r, "reading": reading},
                   indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
