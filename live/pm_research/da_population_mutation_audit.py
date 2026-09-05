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
import subprocess
import sys
from pathlib import Path

PROTOCOL = "P003_DA_POPULATION_MUTATION_AUDIT_V1"
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
    r = subprocess.run([sys.executable, str(HERE / f"{mod}.py"), "--selftest"],
                       capture_output=True, text=True, cwd=str(HERE),
                       timeout=timeout)
    tail = (r.stdout or "")[-4000:]
    fails = [ln.strip() for ln in tail.splitlines()
             if ln.strip().startswith(("FAIL", "FAILED"))
             or " FAIL" in ln or "SELFTEST FAILED" in ln]
    return {"module": mod, "rc": r.returncode, "green": r.returncode == 0,
            "named_failures": fails[:6],
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
            t.write_bytes(text.replace(old, new, 1).encode())
            caught = {m: run_suite(m) for m in suites}
            t.write_bytes(original)
            if digest(t) != before:
                raise MutationRefused(
                    "REFUSED: restore did not reproduce the original digest")
            results.append({
                "mutant": name, "surface": surface, "status": "APPLIED",
                "why_it_matters": why,
                "caught_by": {m: {"went_red": not v["green"], "rc": v["rc"],
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
            "every_applied_mutant_caught_by_the_consumer": all(
                r["caught_by"]["de_section81_mid_census"]["went_red"]
                for r in applied),
            "every_applied_mutant_caught_everywhere": all(
                r["caught_everywhere"] for r in applied),
            "surviving_mutants": [r["mutant"] for r in survivors],
            "n_surviving": len(survivors),
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
            "as the production census it performs on real data",
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
