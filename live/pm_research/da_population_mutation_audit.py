"""DA: does anything actually FAIL if `da_population_audit` is broken?

THE SELF-INDICTMENT THIS CLOSES, filed at R-531(E) and open since:
"NOBODY HAS VERIFIED MY INSTRUMENTS EXCEPT ME. `da_population_audit` was
imported unchanged by DE and used to certify NOTHING_EXCLUDED -- if it has a
defect, that certification is worthless AND IT HAS ALREADY PROPAGATED INTO
ANOTHER SEAT'S ARTIFACT."

A suite that passes proves nothing about a suite that CAN fail.  So this
breaks the instrument on purpose, three ways, and requires the break to be
CAUGHT -- not only by the instrument's own suite but by
`de_section81_mid_census`, the consumer that imported it unchanged.  A mutant
that survives BOTH is a hole in the certification DE relies on; a mutant that
survives only the consumer is a hole in the consumer.

NOT THE SAME THING AS `da_mutation_audit.py`, AND I LEARNED THAT THE HARD WAY.
That file already existed (`bdecb8f`, under the R-347 grant) and is a GENERAL
refusal-DELETION harness with four controls: it walks a module's AST for
`raise` sites and deletes each in turn.  I wrote this one straight over it
without looking, destroying 284 lines, and caught it only when the commit stat
read `M` where it should have read `A`.  The original is restored and untouched;
this lives at its own path.  The two are complementary and both are needed --
deleting a REFUSAL and corrupting a STATISTIC are different mutations, and
neither harness performs the other's.  Look for the instrument before writing
the instrument.

THE THREE SURFACES, chosen because they are the three things the certification
actually rests on:
  * THE STATISTIC  -- `_tvd`, the total-variation distance that decides
                      whether the excluded set looks like the retained one
  * THE NULL       -- the permutation loop that turns that statistic into a p
  * THE STATUS     -- the `NOTHING_EXCLUDED` branch, which is the exact string
                      DE's artifact cites

EVERY MUTANT IS RESTORED BYTE-EXACT AND THE sha256 IS RE-CHECKED, so this file
cannot leave a damaged instrument behind even if it is interrupted: the
restore runs in a `finally`, and the digest is compared before and after.

    python3 live/pm_research/da_population_mutation_audit.py --selftest
    python3 live/pm_research/da_population_mutation_audit.py --real --output P
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RECEIPT_VERSION = 3
old_basename = 'p003_da_population_mutation_audit_v2__20260906T024619Z.json'
old_sha = '8164d495d211ff556c07fef5fbc2b58074dbc9f8a9fc17dc3e6a0082cdac4dcf'


def carrying_commit():
    import subprocess as _sp
    r=_sp.run(['git','rev-parse','HEAD'],capture_output=True,text=True,cwd=str(HERE))
    return r.stdout.strip() if r.returncode==0 else 'UNKNOWN'


PROTOCOL = f"P003_DA_POPULATION_MUTATION_AUDIT_V{RECEIPT_VERSION}"
HERE = Path(__file__).resolve().parent
TARGET = HERE / "da_population_audit.py"
#: The suites that must catch a broken instrument: its own, and the consumer
#: that imported it unchanged to certify NOTHING_EXCLUDED.
SUITES = ("da_population_audit", "de_section81_mid_census")

#: (name, what it breaks, exact source substring, replacement, why it matters)
MUTANTS = (
    ("STATISTIC_returns_zero", "statistic",
     "    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)",
     "    return 0.0 * sum(abs(a[k] / na - b[k] / nb) for k in keys)",
     "TVD always 0: every exclusion looks perfectly representative, so "
     "NOTHING is ever flagged selective -- the always-PASS direction"),
    ("STATISTIC_drops_abs", "statistic",
     "    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)",
     "    return 0.5 * sum((a[k] / na - b[k] / nb) for k in keys)",
     "signed differences cancel to ~0 for any distribution, so a real "
     "imbalance reads as agreement -- silent and plausible"),
    ("NULL_does_not_permute", "null",
     "            rng.shuffle(shuf)",
     "            pass  # MUTANT: labels not permuted",
     "the null draws become the observed split every time, so the p-value "
     "collapses and every exclusion looks extreme or none does"),
    ("STATUS_always_nothing_excluded", "status",
     "    if not excluded:",
     "    if True:  # MUTANT: always take the NOTHING_EXCLUDED branch",
     "the exact string DE's artifact cites is emitted unconditionally -- the "
     "certification would be produced for a population that WAS filtered"),
)


class MutationRefused(RuntimeError):
    """The audit cannot be run safely or its target is not as expected."""


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_suite(mod: str, timeout: int = 900) -> dict:
    # ROUND 55: PYTHONDONTWRITEBYTECODE IS NOT ENOUGH, AND MY ROUND-54 RUN
    # RELIED ON IT ALONE. The variable stops the interpreter WRITING a cache;
    # it does NOT stop it READING one that is already on disk. The reviewer
    # demonstrated a stale `.pyc` being read under the variable (A-6) and the
    # coordinator reproduced it. A same-length mutant leaves mtime and size
    # unchanged, so the cached bytecode still validates and the ORIGINAL code
    # runs while the source on disk is mutated -- the mutant reads green and
    # the harness reports a survivor that never ran.
    #
    # So the cache is DELETED before every child, which is BE's pattern at
    # `be_forward_day.py:2903` (R-446). Belt and braces: the variable stays
    # so nothing new is written, `-B` is passed on the command line so the
    # flag cannot be lost through the environment, and the directory is
    # removed so nothing old can be read.
    shutil.rmtree(HERE / "__pycache__", ignore_errors=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", str(HERE / f"{mod}.py"),
                        "--selftest"],
                       capture_output=True, text=True, cwd=str(HERE),
                       timeout=timeout, env=env)
    tail = (r.stdout or "")[-4000:]
    fails = [ln.strip() for ln in tail.splitlines()
             if ln.strip().startswith(("FAIL", "FAILED"))
             or " FAIL" in ln or "SELFTEST FAILED" in ln]
    # ROUND 54, ITEM 4: A KILL IS CLASSIFIED BY CAUSE, AND A RED RUN MUST
    # ALWAYS BE ABLE TO SAY WHY. The STATUS mutant sent this suite red with
    # `named_failures: []` -- red, and unable to say what caught it. Measured:
    # the mutant makes `compare` return the NOTHING_EXCLUDED dict, which has
    # no `excluded_fraction`, so the suite dies with a KeyError BEFORE any
    # FAIL line is printed. That is a CRASH-KILL, not an ASSERTION-KILL.
    #
    # The distinction is not cosmetic and it is not mine: the pre-existing
    # `da_mutation_audit` harness draws exactly this line in its own
    # docstring -- a crash-kill is still a detection, but it does NOT show
    # the defect is ASSERTED, and a defensive check added elsewhere later
    # would silently delete that coverage. My round-51 row said "red BY NAME"
    # for all four; for the STATUS mutant that was wrong, and this is the
    # correction.
    exc = ""
    for ln in reversed((r.stderr or "").splitlines()):
        t = ln.strip()
        if t and not ln.startswith((" ", "\t")) and ":" in t:
            exc = t
            break
    if r.returncode == 0:
        cause = "NOT_KILLED"
    elif fails:
        cause = "ASSERTION_KILL"
    elif exc:
        cause = "CRASH_KILL"
    else:
        cause = "RED_WITHOUT_A_REASON"
    return {"module": mod, "rc": r.returncode, "green": r.returncode == 0,
            "named_failures": fails[:6],
            "kill_cause": cause,
            "why_red": (exc if cause == "CRASH_KILL"
                        else (fails[0] if fails else "")),
            "stderr_tail": (r.stderr or "")[-400:] if r.returncode else ""}


def audit(mutants=MUTANTS, suites=SUITES, target: Path | None = None) -> dict:
    t = Path(target) if target is not None else TARGET
    if not t.is_file():
        raise MutationRefused(f"REFUSED: no target at {t}")
    original = t.read_bytes()
    before = hashlib.sha256(original).hexdigest()

    baseline = {m: run_suite(m) for m in suites}
    if not all(v["green"] for v in baseline.values()):
        raise MutationRefused(
            f"REFUSED: the baseline suites are not green, so a red mutant "
            f"would prove nothing: "
            f"{ {k: v['rc'] for k, v in baseline.items()} }")

    results = []
    try:
        for name, surface, old, new, why in mutants:
            text = original.decode()
            if text.count(old) != 1:
                results.append({
                    "mutant": name, "surface": surface,
                    "status": "ANCHOR_NOT_UNIQUE",
                    "n_occurrences": text.count(old),
                    "why_it_matters": why,
                    "note": "the mutation could not be applied unambiguously; "
                            "reported rather than applied to a guessed site"})
                continue
            mutated = text.replace(old, new, 1).encode()
            t.write_bytes(mutated)
            # WHICH BYTES DID THE INTERPRETER ACTUALLY EXECUTE? Recording the
            # digest of the source ON DISK at the moment the child ran is the
            # only way to say the mutant was the thing under test, and it is
            # what a stale-.pyc run would have made a lie: the source would
            # hash to the mutant while the interpreter ran the original.
            # Paired with the cache removal in `run_suite`, the pair is
            # checkable rather than asserted.
            shutil.rmtree(HERE / "__pycache__", ignore_errors=True)
            executed = digest(t)
            caught = {m: run_suite(m) for m in suites}
            t.write_bytes(original)
            shutil.rmtree(HERE / "__pycache__", ignore_errors=True)
            if digest(t) != before:
                raise MutationRefused(
                    "REFUSED: restore did not reproduce the original digest")
            results.append({
                "mutant": name, "surface": surface, "status": "APPLIED",
                "why_it_matters": why,
                "source_sha256_while_the_child_ran": executed,
                "source_differs_from_original": executed != before,
                "pycache_removed_before_child": True,
                "caught_by": {m: {"went_red": not v["green"], "rc": v["rc"],
                                  "kill_cause": v["kill_cause"],
                              "why_red": v["why_red"],
                              "named_failures": v["named_failures"]}
                              for m, v in caught.items()},
                "survived_in": sorted(m for m, v in caught.items()
                                      if v["green"]),
                "caught_everywhere": all(not v["green"]
                                         for v in caught.values()),
            })
    finally:
        t.write_bytes(original)
    after = digest(t)

    applied = [r for r in results if r["status"] == "APPLIED"]
    survivors = [r for r in applied if r["survived_in"]]
    return {
        "protocol": PROTOCOL,
        "supersedes": {
            "path": "data/pm_5min/derived/"
                    + old_basename,
            "sha256": old_sha,
            "what_changed": (
                "v3: THE STALE-BYTECODE HAZARD IS CLOSED PROPERLY. v2 set "
                "PYTHONDONTWRITEBYTECODE=1 and nothing else, which stops the "
                "interpreter WRITING a cache but NOT reading one already on "
                "disk (reviewer A-6, reproduced by the coordinator) -- so "
                "v2's 4/4 was obtained under a harness that could in "
                "principle have run cached bytecode. The cache is now REMOVED "
                "before every child and every restore (BE's pattern at "
                "be_forward_day.py:2903), the -B flag is passed on the "
                "command line so it cannot be lost through the environment, "
                "and every mutant records the source digest that was ON DISK "
                "while its child ran. "
                "v2 also carried, and v3 keeps: the consumer predicate "
                "RENAMED to say SELFTEST, because the old name said "
                "'consumer' while what ran was DE's SELFTEST -- and at "
                "c476d0f that selftest asserted nothing on PA.compare's "
                "output (3 PA. sites), so the 0/4 was structurally "
                "guaranteed before any mutant was written. The limit was in "
                "limits[2] and the NAME did not carry it, and R-541(F) was "
                "written from the name. Re-run at HEAD (DE's e67252d, 12 PA. "
                "sites, consumer-side falsifier landed): 4 of 4, every one "
                "an ASSERTION_KILL. Kills are classified by CAUSE, which "
                "corrects my round-51 claim that all four went red BY NAME: "
                "the STATUS mutant is a CRASH_KILL."),
            "correction_is_in_band": (
                f"rule 13: this is v{RECEIPT_VERSION}, a superseding "
                f"receipt; v{RECEIPT_VERSION - 1} is not edited and stands "
                f"as provenance, as does every earlier link"),
        },
        "carrying_commit": carrying_commit(),
        "target": (str(t.relative_to(HERE.parents[1]))
                   if HERE.parents[1] in t.parents else str(t)),
        "target_sha256_before": before,
        "target_sha256_after": after,
        "target_restored_byte_exact": before == after,
        "suites": list(suites),
        "baseline_all_green": True,
        "baseline": baseline,
        "n_mutants": len(mutants),
        "n_applied": len(applied),
        "results": results,
        "computed_predicates": {
            "every_applied_mutant_caught_by_its_own_suite": all(
                r["caught_by"]["da_population_audit"]["went_red"]
                for r in applied),
            # ROUND 54, B-2: RENAMED, AND THE RENAME IS THE CORRECTION.
            # This field was `every_applied_mutant_caught_by_the_consumer`,
            # which NAMES DE's census. What actually ran was DE's SELFTEST,
            # and at c476d0f that selftest asserted nothing about
            # `PA.compare`'s output -- so 0/4 was structurally guaranteed
            # before any mutant was written. My own limits[2] said exactly
            # that and the FIELD NAME did not, and the coordinator wrote
            # R-541(F) from the name. A limit that only a careful reader
            # reaches is not a limit; it belongs in the predicate.
            "every_applied_mutant_caught_by_the_consumers_SELFTEST": all(
                r["caught_by"]["de_section81_mid_census"]["went_red"]
                for r in applied),
            "what_the_consumer_number_measures": (
                "the consumer's SELFTEST, not its production census. A "
                "selftest that never asserts on the imported function's "
                "output cannot go red when that function is broken, so a 0 "
                "here is a fact about the SUITE's coverage and NOT evidence "
                "that the census is unprotected"),
            "every_applied_mutant_caught_by_both_suites": all(
                r["caught_everywhere"] for r in applied),
            "surviving_mutants": [r["mutant"] for r in survivors],
            "n_surviving": len(survivors),
            # HOW each suite caught each mutant, not merely THAT it did.
            "kill_cause_by_mutant": {
                r["mutant"]: {m: v["kill_cause"]
                              for m, v in r["caught_by"].items()}
                for r in applied},
            "n_assertion_kills_own_suite": sum(
                1 for r in applied
                if r["caught_by"]["da_population_audit"]["kill_cause"]
                == "ASSERTION_KILL"),
            "n_crash_kills_own_suite": sum(
                1 for r in applied
                if r["caught_by"]["da_population_audit"]["kill_cause"]
                == "CRASH_KILL"),
            # The hardening, as a predicate: every mutant must have been
            # ON DISK and DIFFERENT from the original while its child ran.
            "every_mutant_was_on_disk_while_its_child_ran": all(
                r["source_differs_from_original"] for r in applied),
            "bytecode_cache_removed_before_every_child": all(
                r["pycache_removed_before_child"] for r in applied),
            "every_red_run_can_say_why": all(
                bool(v["why_red"]) for r in applied
                for v in r["caught_by"].values() if v["went_red"]),
        },
        "role": "REPORTED, NOT ENFORCED (rule 14). A surviving mutant is a "
                "hole in a certification another seat relies on; this names "
                "it and decides nothing.",
        "limits": [
            "four mutants on three surfaces is a SAMPLE of the ways this "
            "instrument could be wrong, never a proof that it is right",
            "a mutant caught by a suite says the suite discriminates on that "
            "line, not that the line is correct",
            "the consumer is run at its own selftest, which is not the same "
            "as the production census it performs on real data -- this limit "
            "is now carried in the predicate's own field name and in "
            "`what_the_consumer_number_measures`, because as a limits entry "
            "it was read past (R-541(F))",
            "the stale-bytecode hazard is closed by DELETING the cache "
            "before every child and every restore (BE's pattern at "
            "be_forward_day.py:2903), plus `-B` and "
            "PYTHONDONTWRITEBYTECODE=1. The variable ALONE does NOT close it "
            "-- it stops the interpreter WRITING a cache, not READING one "
            "already on disk (reviewer A-6), and my round-54 re-run relied "
            "on the variable alone",
        ],
    }


def selftest() -> int:
    fails = []

    def ok(c, m):
        print(("ok   " if c else "FAIL ") + m)
        if not c:
            fails.append(m)

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"
        p.write_text("X = 1\n")
        b = digest(p)
        try:
            audit(mutants=(("m", "s", "X = 1", "X = 2", "w"),),
                  suites=(), target=p)
        except Exception:                                    # noqa: BLE001
            pass
        ok(digest(p) == b,
           "RESTORE: the target's digest is unchanged after a run, and the "
           "restore sits in a `finally` so an interrupted run cannot leave a "
           "damaged instrument behind")

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.py"
        p.write_text("A = 1\nA = 1\n")
        r = audit(mutants=(("dup", "s", "A = 1", "A = 2", "w"),),
                  suites=(), target=p)
        ok(r["results"][0]["status"] == "ANCHOR_NOT_UNIQUE"
           and r["results"][0]["n_occurrences"] == 2,
           "AMBIGUOUS ANCHOR: a mutation whose site is not unique is "
           "REPORTED, never applied to a guessed line")
    try:
        audit(target=Path("/nonexistent.py"))
        ok(False, "KNOWN-BAD: accepted an absent target -- must refuse")
    except MutationRefused:
        ok(True, "KNOWN-BAD: an absent target REFUSES")

    ok(digest(TARGET) is not None and TARGET.is_file(),
       f"TARGET present at {TARGET.name}, sha256 {digest(TARGET)[:16]}")
    ok(len(MUTANTS) >= 3
       and {m[1] for m in MUTANTS} >= {"statistic", "null", "status"},
       f"COVERAGE: {len(MUTANTS)} mutants across "
       f"{sorted({m[1] for m in MUTANTS})} -- the three surfaces the "
       f"certification rests on")

    print(f"\n{'selftest OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(fails)} failure(s)")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.real:
        out = audit()
        txt = json.dumps(out, indent=2, sort_keys=True)
        if a.output:
            a.output.write_text(txt)
        print(txt[:3500])
        return 0
    ap.error("choose --selftest or --real")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
