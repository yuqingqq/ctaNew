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
class EquivalenceFailed(RuntimeError):
    """Raised instead of proceeding. The audit stops."""


CONVENTIONS = [
    ("S60(T) vs S60(t0)", 60, "point", 60),
    ("S30(T) vs S30(t0)", 30, "point", 30),
    ("S60(T) vs S30(t0)", 60, "point", 30),
    ("meanS60[t0,T] vs S60(t0)", 60, "mean", 60),
]
RECENT_DAYS = ("2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27")
PRIMARY = "S60(T) vs S60(t0)"
ORIGINAL = "live.pm_research.exp_m6_settlement"
GATE_POOLED = 0.99
GATE_BIG_MARGIN = 0.995
BIG_MARGIN_BP = 0.5
MIN_POWERED = 400


def load_population(root: Path | None = None):
    """Loader EXCLUSIONS ARE COUNTED, never silent (rule 4).

    Both loops were bare `except: pass`. A malformed-JSON line vanished with no
    trace, so a corrupted or truncated feed would have shrunk the population
    invisibly and the audit would have reported a clean n over a quietly
    reduced denominator -- the same silent-drop shape this programme keeps
    finding, in the loader of the instrument built to check for it.
    """
    markets, res = {}, {}
    skips = {"markets_unparseable": 0, "markets_no_slug": 0,
             "resolutions_unparseable": 0, "resolutions_open_or_no_winner": 0}
    root = PM if root is None else root
    for ln in open(root / "markets.jsonl"):
        if not ln.strip():
            continue
        try:
            m = json.loads(ln)
        except Exception:
            skips["markets_unparseable"] += 1
            continue
        if "slug" not in m:
            skips["markets_no_slug"] += 1
            continue
        markets[m["slug"]] = m
    for ln in open(root / "resolutions.jsonl"):
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except Exception:
            skips["resolutions_unparseable"] += 1
            continue
        if r.get("closed") is True and r.get("winners") and "slug" in r:
            res[r["slug"]] = r["winners"]
        else:
            skips["resolutions_open_or_no_winner"] += 1
    return markets, res, skips


def day_of(m):
    return dt.datetime.fromtimestamp(
        m["window_start"], dt.timezone.utc).strftime("%Y-%m-%d")


def audit(markets, res, streams, days=None):
    """Tally the declared grid over a population. `days=None` is ALL days --
    the equivalence control; a day tuple is the pre-registered recent split."""
    ev = {c[0]: {"n": 0, "hit": 0, "margins": []} for c in CONVENTIONS}
    kn = {c[0]: {"n": 0, "hit": 0} for c in CONVENTIONS}
    used, skipped = set(), 0
    # ENDPOINT STALENESS AS A STATUS. `read_at` returns the last sample AT OR
    # BEFORE the boundary and says nothing about how old it is -- a sample from
    # 4s earlier reads exactly like one from 40ms earlier. The audit therefore
    # scored some windows against a reference the feed had not refreshed, and
    # reported them beside fresh ones with no way to tell them apart.
    stale = {"gt_1s": 0, "gt_2s": 0, "gt_1s_disagree": 0, "gt_2s_disagree": 0,
             "fresh_n": 0, "fresh_hit": 0}
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
                if mode == "mean":
                    xT, tT = mean_over(sT, t0, T, by_known), None
                else:
                    xT, tT = read_at(sT, T, by_known)
                x0, t0s = read_at(s0, t0, by_known)
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
                    if name == PRIMARY and mode == "point":
                        ageT = (T - tT) / 1000.0 if tT is not None else None
                        age0 = (t0 - t0s) / 1000.0 if t0s is not None else None
                        worst = max([a for a in (ageT, age0) if a is not None],
                                    default=None)
                        agree = pred_up == up_won
                        if worst is not None and worst > 2.0:
                            stale["gt_2s"] += 1
                            stale["gt_2s_disagree"] += int(not agree)
                        elif worst is not None and worst > 1.0:
                            stale["gt_1s"] += 1
                            stale["gt_1s_disagree"] += int(not agree)
                        else:
                            stale["fresh_n"] += 1
                            stale["fresh_hit"] += int(agree)
        if counted:
            used.add(slug)
    return ev, kn, len(used), skipped, stale


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


def snapshot_inputs() -> Path:
    """Freeze the population BOTH sides read.

    THE TAPE GROWS DURING MEASUREMENT (CLAUDE.md rule 8). Between two runs
    minutes apart the market count moved 18,098 -> 18,126, so a live
    side-by-side comparison would confound POPULATION DRIFT with INSTRUMENT
    DISAGREEMENT -- and the confound points the flattering way, since a
    mismatch could always be waved off as "the tape grew".

    So both sides read one snapshot: the two small JSONL files are COPIED and
    the large price directory is SYMLINKED (append-only per-hour files, and
    copying 216 of them to prove a point would be its own kind of dishonesty
    about cost). The equivalence question is about the TALLY, and the tally
    reads the copied files.
    """
    import shutil
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="da_m6_snapshot_"))
    for f in ("markets.jsonl", "resolutions.jsonl"):
        shutil.copy2(PM / f, d / f)
    (d / "prices").symlink_to(PM / "prices")
    return d


def original_table(snap: Path) -> dict[str, tuple[int, str, str]]:
    """Run the ORIGINAL script and parse its grid. LIVE, not pinned.

    THE CONTROL WAS DECORATIVE. It printed "(must match exp_m6_settlement)" and
    never compared anything: Codex monkeypatched my table to a wrong 999-row
    schema and `main` still returned 0 and wrote "reading (i) A2 CONFIRMED".
    I did compare the two tables -- by hand, once, in a throwaway script -- so
    the RESULT stood, but the INSTRUMENT could not have caught its own failure,
    and anyone re-running it would get a green-looking output with no check
    behind it. That is the decorative-anchor class, in the audit harness I
    wrote the same day I filed about that class.

    Run live rather than pinned so it cannot go stale: if either side changes,
    the comparison fails and says so.
    """
    import re
    import subprocess
    # Drive the ORIGINAL unmodified, with its input root redirected at the
    # snapshot. Redirecting a module attribute in a subprocess is not editing
    # the script -- the file on disk is untouched and its logic is the logic
    # under test.
    prog = (f"import pathlib, live.pm_research.exp_m6_settlement as M\n"
            f"M.PM = pathlib.Path({str(snap)!r})\n"
            f"M.main()\n")
    out = subprocess.run([sys.executable, "-u", "-c", prog],
                         cwd=str(Path(__file__).resolve().parents[2]),
                         capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        raise EquivalenceFailed(
            f"the original script exited {out.returncode}; equivalence cannot "
            f"be established and the audit STOPS.\n{out.stderr[-400:]}")
    txt = out.stdout
    i = txt.find("=== E-M6 convention grid")
    j = txt.find("=== same grid", i + 1)
    if i < 0 or j < 0:
        raise EquivalenceFailed("cannot locate the original's grid in its "
                                "output; refusing to assume equivalence")
    table_ = {}
    for ln in txt[i:j].splitlines():
        m = re.match(r"^(\S.*?)\s{2,}(\d+)\s+([\d.]+)%\s+([\d.]+)%", ln)
        if m:
            table_[m.group(1).strip()] = (int(m.group(2)), m.group(3),
                                          m.group(4))
    if not table_:
        raise EquivalenceFailed("the original's grid parsed to ZERO rows -- an "
                                "empty comparison is not an equivalence")
    return table_


def assert_equivalent(mine: dict, theirs: dict) -> dict:
    """REFUSE on any disagreement. An empty intersection also refuses: two
    tables that share no convention agree vacuously."""
    fmt = {k: (v["n"], f"{v['agree'] * 100:.1f}", f"{v['agree_big'] * 100:.1f}")
           for k, v in mine.items()}
    shared = sorted(set(fmt) & set(theirs))
    if not shared:
        raise EquivalenceFailed(
            f"no shared conventions between this harness {sorted(fmt)} and the "
            f"original {sorted(theirs)}: an empty comparison is not agreement")
    diffs = [(k, fmt[k], theirs[k]) for k in shared if fmt[k] != theirs[k]]
    missing = sorted(set(theirs) - set(fmt)) + sorted(set(fmt) - set(theirs))
    if diffs or missing:
        raise EquivalenceFailed(
            f"EQUIVALENCE FAILED -- the audit STOPS and no recent-window "
            f"number is read. differing={diffs} unmatched={missing}")
    return {"conventions_compared": shared, "all_match": True}


def main() -> int:
    snap = snapshot_inputs()
    markets, res, load_skips = load_population(snap)
    streams = load_streams()
    print(f"[audit] markets={len(markets)} resolved={len(res)} "
          f"streams={len(streams)}")

    # ---- EQUIVALENCE CONTROL FIRST, before any recent number is read ------
    ev_all, kn_all, n_all, skip_all, stale_all = audit(
        markets, res, streams, days=None)
    full = table(ev_all, kn_all,
                 f"EQUIVALENCE CONTROL — ALL days, windows={n_all}, "
                 f"skipped={skip_all} (must match exp_m6_settlement)")

    # THE GATE: compare, or stop. Nothing below runs if this raises.
    orig = original_table(snap)
    eq = assert_equivalent(full, orig)
    print(f"\nEQUIVALENCE ENFORCED: {len(eq['conventions_compared'])} "
          f"conventions compared against a LIVE run of {ORIGINAL}, all match.")

    ev_r, kn_r, n_r, skip_r, stale_r = audit(markets, res, streams,
                                             days=RECENT_DAYS)
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
    fresh = (stale_r["fresh_hit"] / stale_r["fresh_n"]
             if stale_r["fresh_n"] else float("nan"))
    print(f"\nENDPOINT STALENESS (recent, {PRIMARY}): "
          f">2s {stale_r['gt_2s']} ({stale_r['gt_2s_disagree']} disagree) · "
          f">1s {stale_r['gt_1s']} ({stale_r['gt_1s_disagree']} disagree) · "
          f"fresh {stale_r['fresh_n']} at {fresh:.4%}")
    Path(PM / "derived" / "da_settlement_audit_v1.json").write_text(
        json.dumps({"pre_registration": "Q-DA-145",
                    "as_of_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "equivalence": eq, "original_module": ORIGINAL,
                    "snapshot_dir": str(snap),
                    "snapshot_note": "both sides read one frozen copy of "
                                     "markets/resolutions; the tape grows "
                                     "during measurement (rule 8)",
                    "original_table": {k: list(v) for k, v in orig.items()},
                    "loader_exclusions": load_skips,
                    "staleness_recent": stale_r,
                    "staleness_all": stale_all,
                    "fresh_subset_agree": fresh,
                    "recent_days": list(RECENT_DAYS),
                    "gate": {"pooled": GATE_POOLED, "big": GATE_BIG_MARGIN,
                             "big_margin_bp": BIG_MARGIN_BP,
                             "min_powered": MIN_POWERED},
                    "equivalence_control_all_days": full,
                    "recent": recent, "n_windows_all": n_all,
                    "n_windows_recent": n_r, "reading": reading},
                   indent=2, sort_keys=True), encoding="utf-8")
    return 0


def _selftests() -> int:
    """The checker's own falsifier and positive control (rule 15).

    An equivalence gate that has never refused is exactly what the previous
    version was, so this proves it CAN fail before it is trusted to pass.
    """
    checks, fails = 0, []

    def ok(c, label):
        nonlocal checks
        checks += 1
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    def raises(fn, needle):
        try:
            fn()
        except EquivalenceFailed as e:
            return needle in str(e)
        return False

    mine = {"A": {"n": 10, "agree": 0.999, "agree_big": 1.0},
            "B": {"n": 10, "agree": 0.856, "agree_big": 0.873}}
    same = {"A": (10, "99.9", "100.0"), "B": (10, "85.6", "87.3")}
    ok(assert_equivalent(mine, same)["all_match"] is True,
       "POSITIVE CONTROL: matching tables pass the equivalence gate")
    ok(raises(lambda: assert_equivalent(
           mine, dict(same, A=(999, "99.9", "100.0"))), "EQUIVALENCE FAILED"),
       "FALSIFIER: a WRONG ROW COUNT (999) refuses. Codex monkeypatched "
       "exactly this and the old harness returned 0 and wrote 'A2 CONFIRMED' "
       "-- it PRINTED 'must match' and never compared")
    ok(raises(lambda: assert_equivalent(
           mine, dict(same, B=(10, "85.7", "87.3"))), "EQUIVALENCE FAILED"),
       "FALSIFIER: a one-tenth-of-a-point disagreement refuses too")
    ok(raises(lambda: assert_equivalent(mine, {"Z": (1, "1.0", "1.0")}),
              "empty comparison is not agreement"),
       "VACUITY: tables sharing NO convention refuse -- two tables that "
       "compare nothing agree vacuously")
    ok(raises(lambda: assert_equivalent(mine, dict(same, C=(1, "1.0", "1.0"))),
              "unmatched"),
       "an EXTRA convention on either side refuses: coverage must match, not "
       "merely overlap")
    print(f"\n{'AUDIT CHECKER GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing, {checks} checks")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftests())
    raise SystemExit(main())
