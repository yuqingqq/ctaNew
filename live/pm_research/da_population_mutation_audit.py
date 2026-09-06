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

RECEIPT_VERSION = 4
old_basename = 'p003_da_population_mutation_audit_v3__20260906T025432Z.json'
old_sha = '1077875e6b90a16db72f706a162429e1be93b2a3ecf072612a3cbb3697a69efe'

#: ROUND 57, REVIEWER B-1: `source_sha256_while_the_child_ran` DIGESTS THE
#: DISK, NOT WHAT THE INTERPRETER EXECUTED. v3 wrote the mutant, hashed the
#: file, and ran the child -- which establishes that the mutant was ON DISK
#: and nothing at all about which bytes the interpreter loaded. Under a stale
#: `.pyc` those are different answers, and that is the whole hazard: the disk
#: digest reads as the mutant while the original code runs, so a "survivor"
#: never ran and a "kill" proves nothing about the mutation.
#:
#: THIS PROBE ASKS THE CHILD. After importing the module it reports, from
#: inside the interpreter that loaded it:
#:   * `mod.__file__`         -- WHICH file was resolved (a shadowing copy
#:                               earlier on sys.path is a different defect
#:                               with the same symptom)
#:   * the loader's own source digest -- the bytes the import system read
#:   * sha256(marshal.dumps(loader.get_code(name)))  -- the WHOLE MODULE's
#:     code object as the import system produced it, INCLUDING module-level
#:     statements. Under a valid cache this is the CACHED bytecode, which is
#:     exactly the signal the disk digest cannot carry.
#:   * sha256 over marshal.dumps of every function/method actually present in
#:     the imported module -- these objects ARE what ran.
#: The harness compares them to a BASELINE probe of the unmutated module. A
#: differential needs no cross-process assumption about marshal stability: if
#: the mutant's digests equal the original's, the mutation did not reach the
#: interpreter, whatever the disk says.
PROBE_SRC = r'''
import hashlib, importlib, importlib.util, json, marshal, os, sys
d, name = sys.argv[1], sys.argv[2]
sys.path.insert(0, d)
out = {"module": name}
try:
    spec = importlib.util.find_spec(name)
    cp = getattr(spec, "cached", None)
    out["cache_path"] = cp
    out["pre_cache_existed"] = bool(cp and os.path.exists(cp))
    mod = importlib.import_module(name)
    ldr = mod.__loader__
    out["module_file"] = getattr(mod, "__file__", None)
    out["cached"] = getattr(mod, "__cached__", None)
    try:
        s = ldr.get_source(name)
        out["loader_source_sha256"] = (
            hashlib.sha256(s.encode()).hexdigest() if s is not None else None)
    except Exception as e:
        out["loader_source_sha256"] = "ERR:" + repr(e)
    try:
        out["loader_code_sha256"] = hashlib.sha256(
            marshal.dumps(ldr.get_code(name))).hexdigest()
    except Exception as e:
        out["loader_code_sha256"] = "ERR:" + repr(e)
    h, n = hashlib.sha256(), 0
    for k in sorted(vars(mod)):
        v = vars(mod)[k]
        c = getattr(v, "__code__", None)
        if c is not None and getattr(v, "__module__", None) == name:
            h.update(k.encode()); h.update(marshal.dumps(c)); n += 1
        elif isinstance(v, type) and getattr(v, "__module__", None) == name:
            for mk in sorted(vars(v)):
                mc = getattr(vars(v)[mk], "__code__", None)
                if mc is not None:
                    h.update((k + "." + mk).encode())
                    h.update(marshal.dumps(mc)); n += 1
    out["functions_code_sha256"] = h.hexdigest()
    out["n_code_objects"] = n
    out["python"] = sys.version.split()[0]
    out["ok"] = True
except Exception as e:
    out["ok"] = False; out["error"] = repr(e)
print(json.dumps(out))
'''


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


def probe(target: Path, clear_cache: bool = True, timeout: int = 300) -> dict:
    """What did the INTERPRETER load? Asked of the child, not of the disk.

    `clear_cache` is the ONE variable that separates the good path from the
    known-bad: `-B` is passed either way, because `-B` stops the interpreter
    WRITING a cache and has never stopped it READING one. Isolating the
    variable is the point -- v3's mistake was to treat a flag about writing as
    a control on reading.
    """
    d = target.resolve().parent
    name = target.stem
    if clear_cache:
        shutil.rmtree(d / "__pycache__", ignore_errors=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-c", PROBE_SRC, str(d), name],
                       capture_output=True, text=True, cwd=str(d),
                       timeout=timeout, env=env)
    if r.returncode != 0:
        return {"ok": False, "rc": r.returncode,
                "error": (r.stderr or "")[-400:]}
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:                                   # noqa: BLE001
        return {"ok": False, "error": f"unparseable probe output: {e!r}",
                "stdout_tail": (r.stdout or "")[-400:]}
    out["clear_cache"] = clear_cache
    return out


def interpreter_saw_the_mutant(base: dict, mut: dict,
                               disk_sha: str) -> dict:
    """Did the mutation reach the bytes the interpreter executed?

    Both directions are computed and reported; the caller refuses on the
    negative. `reached` is deliberately an OR over two independent digests:
    a module whose mutation lies outside every function would move
    `loader_code_sha256` and not `functions_code_sha256`, and a module with no
    functions at all would leave the second constant -- which, taken alone,
    would be a control that cannot fire.
    """
    ok = bool(base.get("ok")) and bool(mut.get("ok"))
    same_file = base.get("module_file") == mut.get("module_file")
    code_moved = (ok and base.get("loader_code_sha256")
                  != mut.get("loader_code_sha256"))
    fn_moved = (ok and base.get("functions_code_sha256")
                != mut.get("functions_code_sha256"))
    loader_src = mut.get("loader_source_sha256")
    loader_matches_disk = (loader_src == disk_sha)
    return {
        "probes_ok": ok,
        "module_file_unchanged": same_file,
        "module_file": mut.get("module_file"),
        "loader_code_sha256_moved": code_moved,
        "functions_code_sha256_moved": fn_moved,
        "n_code_objects": mut.get("n_code_objects"),
        "loader_source_sha256": loader_src,
        "disk_sha256": disk_sha,
        "loader_source_matches_disk": loader_matches_disk,
        "cache_existed_before_import": mut.get("pre_cache_existed"),
        "reached_the_interpreter": bool(ok and (code_moved or fn_moved)),
        "what_this_adds_over_the_disk_digest": (
            "the disk digest says the mutant was on disk. These say the "
            "interpreter produced DIFFERENT code objects for it -- which a "
            "stale .pyc would make false while the disk digest stayed true"),
    }


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


def audit(mutants=MUTANTS, suites=SUITES, target: Path | None = None,
          clear_cache: bool = True) -> dict:
    t = Path(target) if target is not None else TARGET
    if not t.is_file():
        raise MutationRefused(f"REFUSED: no target at {t}")
    original = t.read_bytes()
    before = hashlib.sha256(original).hexdigest()

    # THE BASELINE PROBE. Every mutant's "did the interpreter see it" is a
    # DIFFERENCE against this, so it is taken once, from the unmutated module,
    # before anything is written.
    base_probe = probe(t, clear_cache=clear_cache)
    if not base_probe.get("ok"):
        raise MutationRefused(
            f"REFUSED: the baseline probe could not import {t.stem}: "
            f"{base_probe.get('error')}")
    if base_probe.get("loader_source_sha256") != before:
        raise MutationRefused(
            f"REFUSED: the interpreter resolved a DIFFERENT source than the "
            f"file under test. loader read "
            f"{base_probe.get('loader_source_sha256')} at "
            f"{base_probe.get('module_file')}; disk holds {before}. A "
            f"shadowing module on sys.path would make every result below a "
            f"statement about the wrong file.")

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
            if clear_cache:
                shutil.rmtree(HERE / "__pycache__", ignore_errors=True)
            executed = digest(t)
            # ROUND 57, B-1: ASK THE INTERPRETER, DO NOT ASK THE DISK.
            # `executed` above is the disk digest v3 relied on. It is kept --
            # it is still the right answer to "was the mutant written?" -- and
            # it is no longer the answer to "did the mutant RUN?".
            mprobe = probe(t, clear_cache=clear_cache)
            reach = interpreter_saw_the_mutant(base_probe, mprobe, executed)
            if not reach["reached_the_interpreter"]:
                raise MutationRefused(
                    f"REFUSED: mutant {name} is ON DISK ({executed[:16]}) but "
                    f"the interpreter produced the SAME code objects as the "
                    f"original -- it did not run. "
                    f"loader_code moved={reach['loader_code_sha256_moved']}, "
                    f"functions moved={reach['functions_code_sha256_moved']}, "
                    f"cache existed before import="
                    f"{reach['cache_existed_before_import']}. A suite result "
                    f"gathered here would describe the ORIGINAL code.")
            if not reach["loader_source_matches_disk"]:
                raise MutationRefused(
                    f"REFUSED: mutant {name} -- the loader read source "
                    f"{reach['loader_source_sha256']} while the disk holds "
                    f"{executed}. The module under test is not the file being "
                    f"mutated.")
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
                "pycache_removed_before_child": clear_cache,
                # B-1: what the INTERPRETER loaded, reported by the child.
                "interpreter_evidence": reach,
                "probe_baseline": {
                    "loader_code_sha256":
                        base_probe.get("loader_code_sha256"),
                    "functions_code_sha256":
                        base_probe.get("functions_code_sha256"),
                    "module_file": base_probe.get("module_file")},
                "probe_under_mutant": {
                    "loader_code_sha256": mprobe.get("loader_code_sha256"),
                    "functions_code_sha256":
                        mprobe.get("functions_code_sha256"),
                    "module_file": mprobe.get("module_file")},
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

    def _over(suite: str, key: str):
        """A predicate over one suite, or None when that suite was not run.

        `all([])` is True, and a True that was never measured is exactly
        SEAT_PROTOCOL 16's control that cannot fail. None says "not
        measured" instead -- and this shape was found by the v4 selftest,
        which is the first caller ever to reach these predicates with an
        applied mutant and no suites.
        """
        vals = [r["caught_by"][suite][key] for r in applied
                if suite in r["caught_by"]]
        return all(vals) if vals else None
    survivors = [r for r in applied if r["survived_in"]]
    return {
        "protocol": PROTOCOL,
        "supersedes": {
            "path": "data/pm_5min/derived/"
                    + old_basename,
            "sha256": old_sha,
            "what_changed": (
                "v4: THE HARNESS NOW PROVES WHICH BYTES THE INTERPRETER RAN, "
                "NOT WHICH BYTES WERE ON DISK (reviewer B-1, filed at "
                "3563fd2 and accepted). v3's "
                "`source_sha256_while_the_child_ran` digests the FILE after "
                "writing the mutant; that establishes the mutant was WRITTEN "
                "and nothing about which code the child executed. Under a "
                "valid stale `.pyc` the two answers differ -- the disk reads "
                "as the mutant while the ORIGINAL bytecode runs -- so a "
                "'survivor' would be a mutant that never ran and a 'kill' "
                "would be evidence about the wrong code. v4 adds a PROBE "
                "CHILD that imports the module and reports, from inside the "
                "interpreter that loaded it: the resolved `__file__`, the "
                "loader's source digest, sha256(marshal.dumps(get_code)) for "
                "the whole module, and sha256 over marshal.dumps of every "
                "function/method object actually present. Each mutant's "
                "probe is compared to a BASELINE probe of the unmutated "
                "module, and a mutant whose code objects are IDENTICAL to the "
                "original REFUSES THE AUDIT rather than reporting a result. "
                "A loader that read a different file from the one on disk "
                "also refuses. The disk digest is RETAINED -- it is still the "
                "right answer to 'was the mutant written?' -- and is no "
                "longer used as the answer to 'did it run?'. "
                "v3's cache deletion, `-B` and PYTHONDONTWRITEBYTECODE all "
                "stand; the known-bad below shows they were necessary and, on "
                "their own, unprovable: only the code-object digest can tell "
                "a cleared cache from an uncleared one AFTER the fact. "
                "v3 also carried, and v4 keeps: the consumer predicate named "
                "SELFTEST rather than 'consumer', and kills classified by "
                "CAUSE (the STATUS mutant is a CRASH_KILL)."),
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
        "baseline_probe": base_probe,
        "cache_cleared_before_every_child": clear_cache,
        "n_mutants": len(mutants),
        "n_applied": len(applied),
        "results": results,
        "computed_predicates": {
            "every_applied_mutant_caught_by_its_own_suite":
                _over("da_population_audit", "went_red"),
            # ROUND 54, B-2: RENAMED, AND THE RENAME IS THE CORRECTION.
            # This field was `every_applied_mutant_caught_by_the_consumer`,
            # which NAMES DE's census. What actually ran was DE's SELFTEST,
            # and at c476d0f that selftest asserted nothing about
            # `PA.compare`'s output -- so 0/4 was structurally guaranteed
            # before any mutant was written. My own limits[2] said exactly
            # that and the FIELD NAME did not, and the coordinator wrote
            # R-541(F) from the name. A limit that only a careful reader
            # reaches is not a limit; it belongs in the predicate.
            "every_applied_mutant_caught_by_the_consumers_SELFTEST":
                _over("de_section81_mid_census", "went_red"),
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
                if r["caught_by"].get("da_population_audit", {}).get(
                    "kill_cause") == "ASSERTION_KILL"),
            "n_crash_kills_own_suite": sum(
                1 for r in applied
                if r["caught_by"].get("da_population_audit", {}).get(
                    "kill_cause") == "CRASH_KILL"),
            "suites_actually_run": list(suites),
            # The hardening, as a predicate: every mutant must have been
            # ON DISK and DIFFERENT from the original while its child ran.
            "every_mutant_was_on_disk_while_its_child_ran": all(
                r["source_differs_from_original"] for r in applied),
            "bytecode_cache_removed_before_every_child": all(
                r["pycache_removed_before_child"] for r in applied),
            "every_red_run_can_say_why": all(
                bool(v["why_red"]) for r in applied
                for v in r["caught_by"].values() if v["went_red"]),
            # B-1. The disk predicate above says the mutant was WRITTEN.
            # These say the INTERPRETER produced different code for it.
            "every_mutant_REACHED_THE_INTERPRETER": all(
                r["interpreter_evidence"]["reached_the_interpreter"]
                for r in applied),
            "every_mutant_moved_the_loader_code_object": all(
                r["interpreter_evidence"]["loader_code_sha256_moved"]
                for r in applied),
            "every_mutant_moved_a_loaded_function_code_object": all(
                r["interpreter_evidence"]["functions_code_sha256_moved"]
                for r in applied),
            "loader_read_the_file_on_disk_for_every_mutant": all(
                r["interpreter_evidence"]["loader_source_matches_disk"]
                for r in applied),
            "the_same_module_file_was_resolved_every_time": all(
                r["interpreter_evidence"]["module_file_unchanged"]
                for r in applied),
            "what_the_interpreter_predicates_add": (
                "`every_mutant_was_on_disk_while_its_child_ran` is a "
                "statement about the FILESYSTEM and was v3's whole answer to "
                "the stale-bytecode hazard. These are statements about the "
                "CODE OBJECTS the child actually held. A stale .pyc makes the "
                "first true and the second false."),
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
            "the interpreter evidence is a DIFFERENTIAL against the "
            "baseline probe. It proves the mutant's code objects are not the "
            "original's; it does not prove they are the code objects the "
            "MUTATION intended -- a mutation that changed something else in "
            "the file would also move the digest",
            "marshal.dumps of a code object is compared only WITHIN one "
            "interpreter version, never across; the receipt records the "
            "child's python version so a cross-version comparison of two "
            "receipts cannot be made by accident",
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

    # ------------------------------------------------------------------
    # B-1: THE STALE-CACHE KNOWN-BAD, PLANTED ON PURPOSE, BOTH DIRECTIONS.
    #
    # v3's `source_sha256_while_the_child_ran` cannot tell these two apart
    # and v4 must. Control A reproduces the reviewer's exact scenario --
    # compile the ORIGINAL to a timestamp-validated `.pyc`, then mutate the
    # source SAME-LENGTH with the mtime restored, so the cache still
    # validates and the interpreter runs the original while the disk reads
    # as the mutant.
    # ------------------------------------------------------------------
    import py_compile
    import tempfile as _tf
    with _tf.TemporaryDirectory() as d:
        p = Path(d) / "stale_target.py"
        ORIG = "def f(x):\n    return 0.5 * x\n"
        MUT = "def f(x):\n    return 0.0 * x\n"          # SAME LENGTH
        p.write_text(ORIG)
        orig_sha, orig_stat = digest(p), p.stat()
        py_compile.compile(str(p), doraise=True)          # cache from ORIGINAL
        base = probe(p, clear_cache=False)
        ok(base.get("ok") and base.get("pre_cache_existed"),
           "STALE CACHE setup: a timestamp-validated .pyc compiled from the "
           "ORIGINAL is on disk and the probe sees it")

        p.write_text(MUT)
        os.utime(p, (orig_stat.st_atime, orig_stat.st_mtime))
        mut_sha = digest(p)
        ok(len(MUT) == len(ORIG) and p.stat().st_size == orig_stat.st_size
           and p.stat().st_mtime == orig_stat.st_mtime and mut_sha != orig_sha,
           "STALE CACHE setup: the mutant is byte-different but SAME SIZE and "
           "SAME MTIME, so the cached bytecode still validates -- the precise "
           "shape the reviewer described")

        stale = probe(p, clear_cache=False)
        ev = interpreter_saw_the_mutant(base, stale, mut_sha)
        ok(mut_sha != orig_sha and ev["loader_source_matches_disk"],
           "KNOWN-BAD, v3's PREDICATE: the disk digest says the mutant is "
           "there and the loader's SOURCE agrees -- v3 would have reported "
           "`source_differs_from_original: true` and called the run good")
        ok(ev["reached_the_interpreter"] is False
           and not ev["loader_code_sha256_moved"]
           and not ev["functions_code_sha256_moved"],
           "KNOWN-BAD, v4's PREDICATE: the interpreter produced the SAME code "
           "objects as the original -- the mutant NEVER RAN. This is the "
           "distinction the disk digest cannot carry, and it is the whole of "
           "reviewer B-1")

        fresh = probe(p, clear_cache=True)
        ev2 = interpreter_saw_the_mutant(base, fresh, mut_sha)
        ok(ev2["reached_the_interpreter"] is True
           and ev2["loader_code_sha256_moved"]
           and ev2["functions_code_sha256_moved"],
           "POSITIVE CONTROL: with the cache DELETED the identical mutation "
           "DOES reach the interpreter -- so the refusal above is about the "
           "cache, not about a mutation this probe cannot see")

    # Control B: the refusal must fire through `audit()` itself, not only at
    # the probe. An UNCHECKED_HASH cache is used without validating the source
    # at all, so it survives audit()'s own write of the mutant -- the same
    # class of defect, reachable through the real entry point.
    with _tf.TemporaryDirectory() as d:
        p = Path(d) / "unchecked_target.py"
        p.write_text("def f(x):\n    return 0.5 * x\n")
        py_compile.compile(
            str(p), doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
        muts = (("m", "statistic", "return 0.5 * x", "return 0.0 * x", "w"),)
        try:
            audit(mutants=muts, suites=(), target=p, clear_cache=False)
            ok(False, "KNOWN-BAD: audit() accepted a mutant that never "
                      "reached the interpreter -- it must REFUSE")
        except MutationRefused as e:
            ok("did not run" in str(e),
               "KNOWN-BAD through the REAL ENTRY POINT: audit() REFUSES when "
               "the interpreter's code objects are unchanged, instead of "
               "reporting a survivor or a kill that describes the original "
               "code (an UNCHECKED_HASH cache ignores mtime and size, so it "
               "survives audit()'s own write)")
        r = audit(mutants=muts, suites=(), target=p, clear_cache=True)
        ok(r["results"][0]["interpreter_evidence"]["reached_the_interpreter"]
           and r["computed_predicates"]["every_mutant_REACHED_THE_INTERPRETER"],
           "POSITIVE CONTROL through the REAL ENTRY POINT: with the cache "
           "cleared the same audit ADMITS and the predicate is TRUE -- the "
           "refusal discriminates rather than always firing")

    # A SHADOWING MODULE is the other way the disk digest lies: the loader
    # resolves a DIFFERENT file from the one being mutated.
    with _tf.TemporaryDirectory() as d:
        p = Path(d) / "shadow_target.py"
        p.write_text("def f():\n    return 1\n")
        pr = probe(p)
        ok(pr["ok"] and Path(pr["module_file"]).resolve() == p.resolve()
           and pr["loader_source_sha256"] == digest(p)
           and pr["n_code_objects"] >= 1,
           "PROBE IDENTITY: the child reports the resolved __file__, the "
           "loader's source digest and a non-zero code-object count -- a "
           "zero count would make the function digest a control that cannot "
           "fire")

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
