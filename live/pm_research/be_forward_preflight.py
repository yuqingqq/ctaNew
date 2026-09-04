"""GO / NO-GO FOR SCORING A FORWARD DAY, BEFORE THE 28-MINUTE RUN.

Everything that went badly today went badly at a boundary nobody had
rehearsed: a 2 GB cap sized from a smaller day, a verdict that predated the
ruling it needed, a mask the run could not see because a worktree mirror was
stale, and a governed day whose mask had not landed yet. Each was discovered
INSIDE a run, after 25 minutes or 4 OOM kills.

This asks every precondition first, cheaply, and answers GO or NO-GO with the
reason. It CHANGES NOTHING except refreshing this worktree's `derived` mirror,
which is the one failure that is purely local and purely mechanical.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

MAIN_DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
LOCAL_DERIVED = HERE.parents[1] / "data" / "pm_5min" / "derived"


def refresh_local_mirror() -> dict:
    """Symlink anything the main tree has and this worktree does not.

    `data/pm_5min/derived` holds TRACKED files, so it cannot simply BE a
    symlink -- replacing the directory makes git report every tracked file
    deleted, which I did and undid. Per-entry symlinks keep git's view intact
    and still let a new artifact appear."""
    if not LOCAL_DERIVED.is_dir() or LOCAL_DERIVED.is_symlink():
        return {"applicable": False,
                "why": f"{LOCAL_DERIVED} is not a real directory; nothing to "
                       f"refresh (running from the main tree, or already "
                       f"linked)"}
    have = {p.name for p in LOCAL_DERIVED.iterdir()}
    want = {p.name for p in MAIN_DERIVED.iterdir()}
    added = []
    for name in sorted(want - have):
        try:
            (LOCAL_DERIVED / name).symlink_to(MAIN_DERIVED / name)
            added.append(name)
        except OSError:
            pass
    return {"applicable": True, "n_missing_before": len(want - have),
            "n_linked": len(added), "linked": added[:12],
            "still_missing": sorted((want - have) - set(added))[:5]}


def preflight(day: str) -> dict:
    """Every precondition, cheaply, before anything expensive runs."""
    import be_forward_day as BFD
    import de_admissible_windows as AW
    import harmful_forward_scorer as FS

    out: dict = {"day": day, "checks": {}, "blockers": []}

    def note(name, ok_, detail):
        out["checks"][name] = {"ok": bool(ok_), "detail": detail}
        if not ok_:
            out["blockers"].append(f"{name}: {detail}")

    out["mirror_refresh"] = refresh_local_mirror()

    v = FS.read_day_verdict(day)
    note("verdict_exists", bool(v),
         f"{FS.DERIVED / f'da_dayverdict_{day}.json'}"
         if v else "ABSENT — DA has not written it yet")
    if v:
        note("day_closed_calendar", v.get("day_closed_calendar") is True,
             f"day_closed_calendar={v.get('day_closed_calendar')!r}")
        wr = str(v.get("write_reason") or "")
        sched = wr.startswith(BFD.SCHEDULED_PREFIX)
        adm = BFD.USER_ADMISSIONS_BY_DAY.get(day)
        note("attribution", sched or adm is not None,
             "scheduled-unit prefix present" if sched else
             (f"NO prefix, but an admission is declared "
              f"({adm.get('filed_at')})" if adm else
              "NO scheduled-unit prefix AND NO admission — this is the "
              "09-03 shape and needs a USER admission before it can score"))
        out["accrues"] = v.get("race_accrual_eligible")
        out["verdict_as_of"] = v.get("as_of_utc")

    gov = AW.is_governed(day)
    mp = AW.mask_path(day)
    note("mask_present_if_governed", (not gov) or mp.exists(),
         f"governed={gov}; mask at {mp} "
         f"{'EXISTS' if mp.exists() else 'MISSING — a governed day REFUSES '
                                          'without it'}")

    sel = BFD.ratification_for(day)
    note("ratification", bool(sel.get("ref")),
         f"{sel['ref']} / {sel['kind']}")

    try:
        BFD.assert_day_closed_and_attributed(
            day,
            verdict=(BFD.admitted_verdict(day) or {}).get("verdict"),
            admission=(BFD.admitted_verdict(day) or {}).get("record"))
        note("gate_1_would_pass", True, "day_closed_and_attributed passes")
    except Exception as e:                            # noqa: BLE001
        note("gate_1_would_pass", False, f"{type(e).__name__}: {str(e)[:150]}")

    out["verdict"] = "GO" if not out["blockers"] else "NO-GO"
    out["recommended_memory"] = "8G (race days; 2G OOM-killed two runs)"
    out["expected_wall"] = "~1,600-1,700 s on the three days scored so far"
    return out


EXPECTED_CHECKS = 11


def selftest() -> int:
    checks = 0
    fails = []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    r = preflight("20260903")
    ok(r["verdict"] == "GO" and not r["blockers"],
       f"POSITIVE CONTROL: 09-03, a day that DID score, reads GO "
       f"({len(r['checks'])} checks, no blockers)")
    ok(r["checks"]["attribution"]["ok"]
       and "admission" in r["checks"]["attribution"]["detail"],
       "and its attribution passes VIA THE ADMISSION, named — not via a "
       "scheduled prefix it does not have")
    b = preflight("21000101")
    ok(b["verdict"] == "NO-GO" and b["blockers"],
       f"KNOWN-BAD: the pinned non-day 21000101 reads NO-GO with blockers "
       f"{[x.split(':')[0] for x in b['blockers']]} — a preflight that "
       f"cannot say NO-GO is not a preflight")
    ok(b["checks"]["verdict_exists"]["ok"] is False,
       "and it names the FIRST missing precondition rather than dying on it")
    m = r["mirror_refresh"]
    ok("applicable" in m,
       f"the local mirror refresh reports what it did "
       f"({m.get('n_linked', 0)} linked, {m.get('n_missing_before', 0)} "
       f"missing before) — the 09-03 failure was a stale mirror and it is "
       f"mechanical, so it is done rather than warned about")
    ok(preflight("20260902")["checks"]["attribution"]["detail"]
       == "scheduled-unit prefix present",
       "CONTRAST: 09-02 passes attribution on its OWN prefix, so the "
       "admission path is not doing the work for days that do not need it")

    # (2) THE RUNNER'S THREE REFUSAL PATHS, DRIVEN. A path whose first real
    # execution is the night it matters is untested.
    import subprocess, tempfile, os
    sh = str(HERE / "be_score_forward_day.sh")
    env = {**os.environ, "SCORE_DEADLINE_S": "1", "SCORE_RETRY_INTERVAL_S": "1"}
    with tempfile.TemporaryDirectory() as td:
        full = Path(td) / "full"
        full.mkdir()
        (full / "already_here.json").write_text("{}")
        r = subprocess.run([sh, "20260903", str(full)], capture_output=True,
                           text=True, env=env, timeout=120)
        ok(r.returncode == 2 and "is not empty" in (r.stdout + r.stderr),
           f"RUNNER PATH 1 DRIVEN: a NON-EMPTY outdir refuses with rc="
           f"{r.returncode} and says why -- reusing one is how a half-written "
           f"feed gets read as a day")
        r2 = subprocess.run([sh, "20260904", str(Path(td) / "new")],
                            capture_output=True, text=True, env=env,
                            timeout=180)
        out2 = r2.stdout + r2.stderr
        ok(r2.returncode == 3,
           f"RUNNER PATH 2 DRIVEN: a day that is NOT ready exits 3 at the "
           f"deadline (got {r2.returncode}) -- it does NOT wait silently and "
           f"it does NOT score")
        ok("mask" in out2.lower() and "NO-GO" in out2,
           "RUNNER PATH 3 DRIVEN: the MASK blocker is NAMED on every attempt, "
           "so a watcher sees which blocker clears and which persists")
        ok("NOTHING WAS SCORED" in out2,
           "and the give-up line SAYS nothing was scored -- a run that waits "
           "through the night and reports nothing is indistinguishable from "
           "one that never started")
        ok(not (Path(td) / "new").exists()
           or not any((Path(td) / "new").iterdir()),
           "and it wrote NO artifact for the day it refused")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--day" in argv:
        i = argv.index("--day")
        if i + 1 >= len(argv):
            print("REFUSED: --day needs a token")
            return 2
        r = preflight(argv[i + 1])
        print(json.dumps(r, indent=1, sort_keys=True, default=str))
        return 0 if r["verdict"] == "GO" else 1
    print("usage: be_forward_preflight.py --selftest | --day <YYYYMMDD>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
