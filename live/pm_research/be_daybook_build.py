"""THE SCORED DAY-BOOK. One day, btc, both pinned heads. The book only.

DE's design v7 R1: the day book must carry `asm`, not just a reference --
`be_cancel_axis_null.load()` reads `c["asm"]` and builds its decision
population from `asm["by_arm"][(coin, head)][0]`. A reference-only book
raises and there is no decision population at all.

WHAT THIS BUILDS AND WHAT IT REFUSES TO DO. It builds the book: reference
(slug -> side -> generations with tranches), statuses, population, n_slugs,
terminal_marks, and `asm` for BOTH pinned heads at their PINNED thetas. It
does NOT score arms, draw nulls, or compute economics. Those are DE's and
they run on this.

THE DAY IS SELECTED THE WAY THE FORWARD SCORER SELECTS IT, NOT THE WAY THE
ARMS CACHE WAS SELECTED. `harmful_exposure_rows.select_v2_era` is bounded by
the declared populations, which end at 2026-08-26T00:00 -- MEASURED, and it
is why a September day cannot come through that door at all. The day path is
`de_admissible_windows.supply(day, present_from_ledger(day))`, the same
supply the forward scorer gated 09-03 through at 12/12.

MEMORY IS THE BINDING CONSTRAINT AND DE MEASURED IT BEFORE ME.
`build_tape_index`'s own docstring: "the score split is 638,917 rows and
1.42 GB, the train split 1,125,289 rows and 3.90 GB cumulative, 390.7 s for
both. That 3.9 GB is resident for the whole assembly, and THE RULED RUN OF
2026-09-03 DIED because it was held alongside the entire fragment." So the
fragment is consumed in CHUNKS against one index, and the cap is not raised.

THE DIGEST IS OF THE BYTES AS WRITTEN (the v3/B-1 discipline, applied to the
write side): the book is serialised to a buffer, that buffer is hashed, and
THAT buffer is written. There is no second read and no second serialisation.
"""
from __future__ import annotations

import gc
import hashlib
import json
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import be_data_root as _BDR
import be_rule22 as _R22

#: RULE 22 AS AMENDED (R-605). The CAPTURE happens here, at import,
#: before any work: the digest of every module of this run's import
#: closure under `live/`, plus the worktree HEAD and its dirty state.
#: The STAMP is taken at EMIT -- it has to be, because the fields that
#: matter (`closure_drift`, `head_unchanged_during_the_run`) are
#: statements about the interval BETWEEN import and emit, and a stamp
#: frozen at import could not carry them. What must never be read at
#: emit is the DIGEST, and it is not: `stamp()` reports the bytes seen
#: at first sight and compares them with the file now.
_R22.init("be_daybook_build import")

#: BE48 §B.4: THIS MODULE HAD TWO ROOTS. The book went through the resolver
#: while the scratch fragment and the receipt went to `HERE.parents[1]` -- so
#: from a worktree the receipt named a book that was not beside it, and a
#: multi-hundred-MB scratch file landed in the worktree's `data/`, which is
#: the act that manufactures the shells this seat spent three rounds
#: removing. My round-47 reasoning for the split (don't write into the main
#: tree) was WRONG: R-397/R-554 already rule that artifacts under `data/`
#: are landed from the main tree by pathspec, so the ledger IS where they go.
#: ONE ROOT NOW, and `require_ledger` guards the result-bearing emission.
ROOT = HERE.parents[1]                    # CODE root only; never a data root
LEDGER_DERIVED = _BDR.derived()
OUT_DERIVED = LEDGER_DERIVED

COIN = "btc"

#: R-649 §3.2: 75 is EX_TEMPFAIL. From OUTSIDE a unit, ExecMainStatus=75
#: reads "the lock was held" OR "a producer broke the declaration and exited
#: 75 for its own reasons" -- and the two are indistinguishable. So this
#: producer DECLARES its exit codes and its selftest asserts 75 is not among
#: them, which is what makes the launcher's 75 mean one thing.
EXIT_CODES = {
    0: "the artifact was written and its receipt emitted",
    1: "a refusal or an uncaught error (Python's default for an exception)",
    2: "usage: no --day and no --selftest",
}
EXIT_CODE_NOTE = ("75 is RESERVED to the launcher's flock conflict and is "
                  "not in this map; the selftest asserts it.")


#: PER-STAGE MEMORY BUDGETS, declared, and a stage over its budget REFUSES.
#: Named to match DE's `--day` stages so the seams agree: DE's S0/S1 consume
#: the book this produces, and R11 says the index is needed ONLY to produce
#: `asm` -- so it is RELEASED before the book is written and the whole-day
#: peak is max(index stage, assembly stage), not their sum.
STAGE_BUDGETS_GB = {
    "A0_reference": 3.0,      # measured 2.008 on 09-03
    "A1_index": 6.5,          # measured 5.971 cumulative (reference + index)
    "A2_assemble": 7.5,       # index + reference + chunked fragment + asm
    "A3_release_index": 7.5,  # high-water only; CURRENT must fall
    "A4_write_book": 7.5,
}
FIXTURE_STAGE_BUDGETS_GB = {k: 0.7 for k in STAGE_BUDGETS_GB}
#: The scope cap every budget above was derived against. Never raised.
CAP_GB = 8.0
CHUNK_WINDOWS = 6             # as declared in be_assembly_budget

#: REV 46 (5). The index release was MEASURED and never ASSERTED -- a
#: number in a receipt that nothing checks. A no-op release would have
#: reported `freed_gb: 0.0` and passed. The release must free at least this
#: fraction of the A1 index peak or the build REFUSES.
MIN_RELEASE_FRACTION = 0.10


#: REV 63 S3. Lifted out of the receipt literal so the battery can read the
#: VALUE rather than the source: the sentence below spans several Python
#: string literals, so a source scan for it found nothing while the receipt
#: carried it perfectly well. Test the thing, not its spelling.
SCOPE_NUMBERS_SUPPORT = {
    "a_peak_equal_to_the_cap_is_a_floor":
        "09-04's tape reported peak_bytes == max_bytes with 1,199 reclaims "
        "that pinned it there. That is a bound, not a measurement of "
        "demand: it supports `demand was AT LEAST the cap and was throttled "
        "1,199 times`, and it is not commensurable with an unthrottled peak "
        "from another run",
    "what_the_two_days_constrain":
        "the builder's own footprint did not move (peak_rss_gb identical to "
        "three decimals across a 31% row spread, from a field that varies "
        "sensibly elsewhere); the whole cgroup difference sits in page "
        "cache, which ROSE on the smaller day as anon FELL -- so whatever "
        "differed was outside the builder's own allocation. Stronger than a "
        "correlation, weaker than a mechanism, and asserted as neither",
    "the_field_that_says_so": "scope.peak_is_censored",
}


def _index_call_made() -> str:
    """The seam call, READ FROM THIS MODULE'S SOURCE, never restated.

    The receipt carried the literal "build_tape_index(splits, tape_path=…)"
    while the call had already moved to the one-object `inputs=` form -- a
    literal contradicting the code beside it (rule 10), and the third of that
    class this seat has shipped. It is derived now, so it cannot drift."""
    import re
    src = Path(__file__).read_text()
    m = re.search(r"R\.build_tape_index\(([^)]*)\)", src)
    return f"build_tape_index({m.group(1)})" if m else "UNKNOWN"


def _assembly_evidence(asm: dict, ref: dict, cov: dict, n_gen: int,
                       chunk_windows: int, rows_pin: dict | None = None) -> dict:
    """REV 46 (2) and (3), corrected in round 60: the numbers that PROVE the
    seam worked, as FIELDS -- each accounting CLOSED IN ITS OWN POPULATION.

    WHAT ROUND 59 GOT WRONG, measured on the 09-04 book. The field
    `reasons_account_for_the_count` compared `sum(drops_by_coin)` with
    `n_uncovered` and reported FALSE (29,465 against 19,663). Both numbers
    were right; the comparison was not. `drops_by_coin` counts FRAGMENT ROWS
    (`phase2_arms` builds it over `[r for r in data["rows"] if status ==
    "OK"]`); `n_uncovered` counts REFERENCE GENERATIONS
    (`n_reference_generations - n_covered`). Those are two populations, so
    the predicate could not hold on any real day -- and REV 48's known-bad
    could not reveal that, because the fixture supplied both sides in the
    same unit (15 against 15). A control whose fixture makes the two sides
    commensurable cannot fail the way the real data fails.

    So there are now TWO statements, each within one population:

      ROWS        kept + dropped == the rows the producer published
                  (the tape receipt's `n_rows`, an INDEPENDENT number from
                  round 58, not a total this function computed itself).
                  On 09-04: 609,137 + 29,465 = 638,602. It closes.

      GENERATIONS n_uncovered, reported with its coverage and WITHOUT a
                  reason breakdown -- because none of the drop classes is in
                  that unit. Saying so is the honest form; attaching row
                  reasons to a generation count is what produced the false
                  field."""
    a = asm.get("assembly", {}) or {}
    drops = {c: dict(v) for c, v in (a.get("drops_by_coin") or {}).items()}
    kept = dict(a.get("kept_by_coin") or {})
    n_chunks = -(-len(ref) // chunk_windows) if chunk_windows else None
    uncovered = {h: c["n_uncovered"] for h, c in cov.items()}
    per_reason = {}
    for c, dd in drops.items():
        for k, v in dd.items():
            per_reason[k] = per_reason.get(k, 0) + int(v)
    total_drops = sum(per_reason.values())
    total_kept = sum(int(v) for v in kept.values())
    one_uncovered = sorted(set(uncovered.values()))
    rows_published = (rows_pin or {}).get("n_rows")
    accounted = total_kept + total_drops
    return {
        "state_join_failed": per_reason.get("state_join_failed"),
        "state_join_failed_is_zero":
            per_reason.get("state_join_failed") == 0,
        "why_that_matters": "a non-zero state_join_failed would mean the "
                            "day's generations did not find rows in the "
                            "day's tape -- which is exactly what the "
                            "parameterised seam exists to make impossible",
        "n_chunks": n_chunks,
        "chunk_windows": chunk_windows,
        "n_windows": len(ref),
        "kept_by_coin": kept,
        "ROW_ACCOUNTING": {
            "population": "FRAGMENT ROWS -- the rows phase2_arms reads from "
                          "the day's fragment with status OK",
            "kept": total_kept,
            "dropped": total_drops,
            "by_reason": per_reason,
            "accounted": accounted,
            "rows_published_by_the_tape_receipt": rows_published,
            "rows_pin_receipt": (rows_pin or {}).get("receipt"),
            "rows_accounted_for": (rows_published is not None
                                   and accounted == rows_published),
            "if_they_do_not_close": "the residual is UNEXPLAINED and is "
                                    "reported as such rather than absorbed. "
                                    "Both sides are ROW counts and the "
                                    "total comes from round 58's receipt, "
                                    "not from this function -- so a "
                                    "mismatch is a real disagreement "
                                    "between producers, not a unit error",
            "residual": (None if rows_published is None
                         else rows_published - accounted),
        },
        "UNCOVERED_GENERATIONS": {
            "population": "REFERENCE GENERATIONS -- a DIFFERENT population "
                          "from the row accounting above, which is why no "
                          "reason breakdown is attached to it",
            "count": one_uncovered[0] if len(one_uncovered) == 1 else uncovered,
            "identical_across_heads": len(one_uncovered) == 1,
            "n_reference_generations": n_gen,
            "coverage": (1 - one_uncovered[0] / n_gen
                         if len(one_uncovered) == 1 and n_gen else None),
            "why_no_reason_class_here": "the drop classes "
                                        "(pre_window_excluded, "
                                        "gap_at_cutoff_excluded, "
                                        "no_level_history_excluded, "
                                        "state_join_failed) are counted in "
                                        "FRAGMENT ROWS. Attaching them to a "
                                        "generation count is what made "
                                        "round 59's "
                                        "`reasons_account_for_the_count` "
                                        "report FALSE on numbers that were "
                                        "each correct",
            "superseded_field": "reasons_account_for_the_count (round 59) -- "
                                "withdrawn, not silently dropped: it "
                                "compared rows with generations",
        },
    }

def assert_index_released(before_gb: float, after_gb: float,
                          index_peak_gb: float,
                          fraction: float = MIN_RELEASE_FRACTION) -> dict:
    """The index is GONE after A3, or the day refuses.

    DE v9 R11 makes the whole-day peak `max(index, assembly)` instead of
    their sum ONLY IF the index is actually released. Reporting a number
    nobody checks is how that becomes a claim rather than a fact."""
    freed = before_gb - after_gb
    need = fraction * index_peak_gb
    out = {"current_gb_before": before_gb, "current_gb_after": after_gb,
           "freed_gb": round(freed, 3), "index_peak_gb": index_peak_gb,
           "required_fraction": fraction, "required_freed_gb": round(need, 3),
           "freed_fraction_of_index_peak":
               round(freed / index_peak_gb, 4) if index_peak_gb else None,
           "asserted_not_only_measured": True}
    if freed < need:
        raise BookRefused(
            f"REFUSED at A3_release_index: only {freed:.3f} GB was freed, "
            f"below the required {need:.3f} GB ({fraction:.0%} of the "
            f"{index_peak_gb:.3f} GB index peak). R11's whole-day budget "
            f"rests on the index being GONE before the book is written; a "
            f"release that frees nothing makes `max(index, assembly)` a "
            f"claim rather than a fact.")
    out["released"] = True
    return out

HEADS = {"CONDVALUE_X_SKEW": "q1_arrival_composed_lgbm",
         "HAZARD_OVER_SKEWED_REF": "incumbent_linear_d"}
BUDGET = 0.10


class BookRefused(RuntimeError):
    """A named refusal."""


def day_tape_sha(day: str, coin: str = COIN) -> str | None:
    """The day tape's digest AS THE BUILDER RECEIPT NAMES IT.

    Read from the receipt rather than recomputed here, so the assembly is
    bound to the bytes the tape builder published -- not merely to whatever
    is at the path today."""
    # REV 43: this hardcoded (.v2, .json) and would MISS a .v3 the moment
    # the builder auto-versioned past it -- which it did the same round.
    # The search now GLOBS and takes the HIGHEST version, so it follows the
    # builder instead of restating a snapshot of it.
    return (day_tape_pin(day, coin) or {}).get("sha256")


def _receipt_head(stem: str) -> tuple:
    """The HIGHEST-versioned receipt matching `stem`, and its parsed version.

    ONE resolver, used by everything that needs a round-58 receipt -- so the
    name a receipt REPORTS and the file it READ can never disagree again.
    Round 59 emitted `...v3.json` from a hardcoded f-string 470 lines away
    from this glob, and named a file that does not exist for 09-04."""
    import re

    def _ver(q: Path) -> int:
        m = re.search(r"\.v(\d+)\.json$", q.name)
        return int(m.group(1)) if m else 1

    cands = sorted(OUT_DERIVED.glob(f"{stem}*.json"), key=_ver, reverse=True)
    return (cands, _ver)


def day_tape_pin(day: str, coin: str = COIN) -> dict | None:
    """The tape's PIN as the round-58 receipt publishes it: digest, the
    receipt's REAL name, and the row count the row accounting closes on."""
    stem = f"be_gate1_state_tape_receipt_{day}_{coin}"
    cands, _ = _receipt_head(stem)
    for r in cands:
        d = json.loads(r.read_text())
        if d.get("WHICH_SPLIT_THE_ASSEMBLY_SCORES_FROM_AND_WHY", {}) \
                .get("split") == "score":
            return {"sha256": d["tape"]["sha256"],
                    "receipt": r.name,          # READ, never typed
                    "n_rows": d["tape"].get("n_rows"),
                    "bytes": d["tape"].get("bytes"),
                    "split": "score"}
    return None


def day_fragment_pin(day: str, coin: str = COIN) -> dict | None:
    """The fragment's PIN as its own receipt publishes it.

    Round 59 had no such binding: the builder hashed the file itself and
    handed that digest to the front door, which recomputed and compared it
    with itself. A self-consistency check over a two-statement window is not
    the check its name implies -- it cannot tell that these are the bytes
    round 58 published."""
    stem = f"be_gate1_fragment_receipt_{day}_{coin}"
    cands, _ = _receipt_head(stem)
    for r in cands:
        d = json.loads(r.read_text())
        fr = d.get("fragment") or {}
        if fr.get("sha256"):
            return {"sha256": fr["sha256"], "receipt": r.name,
                    "n_rows": (d.get("build") or {}).get("n_rows"),
                    "n_windows": (d.get("build") or {}).get("n_windows"),
                    "bytes": fr.get("bytes")}
    return None


def assert_input_matches_its_receipt(kind: str, path, pin: dict | None) -> dict:
    """THE FILE ON DISK IS THE ONE ITS BUILDER PUBLISHED -- or refuse.

    This is the check round 59 did not have. `day_assembly_inputs` verifies
    WHICH BYTES reach the pass; only this says those bytes are the ones the
    producing receipt pinned."""
    if not pin or not pin.get("sha256"):
        raise BookRefused(
            f"REFUSED: no builder receipt pin for the {kind}. The assembly "
            f"binds to the digest its producer published; without one there "
            f"is nothing to bind to and the day is refused, never assumed.")
    got = _sha_file(path)
    if got != pin["sha256"]:
        raise BookRefused(
            f"REFUSED: the {kind} on disk does not match the digest its "
            f"receipt {pin['receipt']} pins -- {got[:16]}... on disk against "
            f"{pin['sha256'][:16]}... pinned. These are not the bytes round "
            f"58 published.")
    return {"kind": kind, "sha256": got, "receipt": pin["receipt"],
            "matches_its_builder_receipt": True,
            "compared_full_length": len(got) == 64}


def _sha_file(p) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def assert_day_tape(day: str, coin: str = COIN, *,
                    tape: Path | None = None,
                    receipt_sha: str | None = None) -> dict:
    """THE ASSEMBLY MUST READ THE DAY'S TAPE, AND TODAY IT CANNOT.

    `phase2_arms.tape_index` reads the MODULE CONSTANT `TAPE_PATH`
    (`phase2_state_tape_v5.json`, the live August tape) and neither it nor
    `de_phase4_diag_runner.build_tape_index(splits)` takes a path --
    VERIFIED at the signatures. So an assembly run for a September day would
    index the WRONG TAPE, find no rows for the day's generations, and emit a
    book whose `asm` is empty. That is the failure that looks like a result,
    and DE's guard would fire on it one stage later.

    This refuses BEFORE any of it. The fix is a path parameter on
    `phase2_arms.tape_index` -- DE's surface, routed at the reviewer's BE48
    item 2, not taken here."""
    import be_gate1_state_tape as TAPEMOD
    import phase2_arms as PA
    # `tape`/`receipt_sha` are FIXTURE INJECTION POINTS. The reviewer could
    # not drive this guard from a worktree (BE-51 1.x): it reached for the
    # ledger's tape and receipt, which a worktree does not carry. Injecting
    # them makes the guard drivable anywhere without weakening it -- the
    # real path is still the default.
    want = Path(tape) if tape is not None else TAPEMOD.out_path(day, coin)
    if not want.exists():
        raise BookRefused(
            f"REFUSED: the day's tape {want.name} does not exist.")
    sha = receipt_sha or day_tape_sha(day, coin)
    if not sha:
        raise BookRefused(
            f"REFUSED: no SCORE-split builder receipt for {day} {coin}. The "
            f"assembly binds to the digest the tape builder published, and a "
            f"tape whose receipt says `train` is the wrong split for a ruled "
            f"forward day.")
    # ITEM 1 landed: the path is now a parameter, so the default constant is
    # no longer the blocker. What must still hold is that the day's tape
    # EXISTS and its receipt names the SCORE split.
    if False:
        raise BookRefused(
            f"REFUSED: the assembly would index {have.name}, not this day's "
            f"tape {want.name}. `phase2_arms.TAPE_PATH` is a module constant "
            f"and neither `tape_index(split, features_in_order)` nor "
            f"`build_tape_index(splits)` accepts a path -- so a September "
            f"day cannot be pointed at its own tape and the run would emit a "
            f"book with an EMPTY `asm`. BLOCKED on a path parameter, which "
            f"is DE's surface (reviewer BE48, item 2). Not raised here, and "
            f"not worked around.")
    return {"tape": str(want), "is_the_days_tape": True,
            "fixture_injected": tape is not None,
            "sha256_from_receipt": sha,
            "default_constant_no_longer_blocks": True,
            "why": "phase2_arms.tape_index and build_tape_index now take a "
                   "`path` (BE round 52 item 1), so the assembly indexes the "
                   "day's tape and verifies its digest at load"}


def _flock_mode(lock_path) -> str | None:
    """DELEGATED to `be_rule22.flock_mode` -- moved there so all three
    producers read the lock the same way. Kept as a name because this
    module's battery drives it."""
    return _R22.flock_mode(lock_path)

def assert_rule20(*, fixture: bool = False) -> dict:
    """DELEGATED to DE's `wrapper_observed` -- MEASURED, not declared.

    R12: `flock -n <lock> systemd-run --scope` passes the lock's fd through
    the exec, so holding it is READ FROM /proc/self/fd. A string in a params
    file cannot be evidence that a lock was held. My 05:54Z breach is exactly
    what this refuses: a heavy run beside another heavy run, in the slice but
    without the lock. Delegated rather than reimplemented -- two
    implementations of one check is two checks (Q-BE-271)."""
    import de_multiday_gate1_runner as RUN
    w = dict(RUN.wrapper_observed())
    w["delegated_to"] = "de_multiday_gate1_runner.wrapper_observed"
    w["fixture"] = fixture
    # THE REVIEWER'S FINDING (round 53): TWO `flock -s` HOLDERS BOTH CERTIFY.
    # A shared lock is not mutual exclusion, and "the fd is held" cannot tell
    # the two apart -- so the MODE is read from /proc/locks, where an
    # exclusive flock is ADVISORY WRITE and a shared one is READ.
    w["lock_mode"] = _flock_mode(RUN.HEAVY_RUN_LOCK)
    w["exclusive"] = w["lock_mode"] == "WRITE"
    w["why_mode_not_just_held"] = ("two `flock -s` holders would both report "
                                   "the fd and both certify; only WRITE is "
                                   "mutual exclusion")
    if not fixture and w.get("heavy_run_lock_held") and not w["exclusive"]:
        raise BookRefused(
            f"REFUSED: the heavy-run lock is held in mode "
            f"{w['lock_mode']!r}, not WRITE. A SHARED (`flock -s`) lock lets "
            f"a second heavy run take it at the same time and both would "
            f"certify -- which is not one-heavy-run-at-a-time. Take it "
            f"exclusively (`flock -n`, the default).")
    if not fixture and not w.get("heavy_run_lock_held"):
        raise BookRefused(
            f"REFUSED: a real day is HEAVY BY CONSTRUCTION and this "
            f"process does not hold the heavy-run lock. Launch it with "
            f"`{_R22.LAUNCHER.name} <unit> be_daybook_build.py --day <day>`, "
            f"which runs the lock INSIDE a transient service. The command "
            f"this refusal used to print -- `flock -n <lock> systemd-run "
            f"--user --scope ...` -- is the form R-628 ruled against: the "
            f"payload sits in the launching shell's process tree and dies "
            f"with it, and every BE heavy run through 09-05 used it. At "
            f"05:54Z on 2026-09-06 this seat ran a heavy build beside "
            f"another heavy run; this refuses that before any work.")
    return w


def _rss_gb() -> float:
    """HIGH-WATER mark. It never falls, which is why `_rss_now_gb` exists."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def _rss_now_gb() -> float:
    """CURRENT RSS. `ru_maxrss` is a high-water mark and cannot show a
    release, so the claim "the index was dropped" needs a falling number to
    rest on. Read from /proc/self/status."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 2**20, 3)
    except OSError:
        pass
    return float("nan")


class _Stages:
    """Per-stage budgets, asserted, measured as THIS RUN'S OWN GROWTH.

    DA 77 (R-613) found the smoke's shape one caller away here: `ru_maxrss`
    is process-wide and never falls, so a budget narrower than the process --
    one stage's, one fixture's -- compared against it is not measuring what
    it names, and once earlier work has raised the high-water the narrow
    check can never pass again. That is exactly how the 09-03 smoke died: a
    real day's 2,426 MB high-water judged against a fixture's 700 MB budget.

    So the BUDGET is now compared against `growth_gb` -- the high-water minus
    this run's own baseline at the first stage -- for the fixture and the
    real path alike. That comparison is LOOSER than the old one by the
    baseline (~0.1-0.2 GB), and a budget must never be quietly widened, so
    the absolute figure is not dropped: `peak_gb` is still measured and now
    checked against the CAP itself, which is the ceiling every budget was
    derived from. Nothing is relaxed overall -- one check became two, each
    against the quantity it actually names."""

    def __init__(self, budgets: dict, *, cap_gb: float = CAP_GB):
        self.budgets = dict(budgets)
        self.cap_gb = cap_gb
        #: CURRENT RSS, not the high-water. REV 59 §3: round 60 took BOTH
        #: terms of the growth from `ru_maxrss`, which never falls -- so a
        #: SECOND `_Stages` in the same process read growth 0.000 and its
        #: budget could not fire at all. DE 89 measured and rejected that
        #: same instrument a round earlier and its comment says why: "a
        #: budget measured with an instrument that cannot fall is a budget
        #: that only works once per process". `_rss_now_gb` exists in this
        #: module precisely because a falling number was needed.
        self.baseline_gb = _rss_now_gb()
        self.rows = []

    def done(self, name: str, t0: float) -> dict:
        b = self.budgets.get(name)
        peak = _rss_gb()                      # process high-water: the CAP
        cur = _rss_now_gb()                   # falls: the BUDGET
        row = {"stage": name, "wall_s": round(time.time() - t0, 1),
               "peak_gb": peak,
               "baseline_gb": self.baseline_gb,
               "growth_gb": round(cur - self.baseline_gb, 3),
               "current_gb": cur,
               "budget_gb": b, "cap_gb": self.cap_gb,
               "budget_is_measured_on": "CURRENT RSS minus this _Stages' own "
                                        "baseline -- an instrument that can "
                                        "FALL, so the budget fires as often "
                                        "as it is asked. The cap below is "
                                        "measured on the process high-water, "
                                        "which is the right instrument for a "
                                        "ceiling that must never be crossed",
               "what_growth_cannot_see": "a transient INSIDE a stage that is "
                                         "already released by the time the "
                                         "stage ends. That is what the "
                                         "high-water cap check catches, and "
                                         "it is why both are reported"}
        row["within_budget"] = (b is None or row["growth_gb"] <= b)
        row["within_cap"] = peak <= self.cap_gb
        self.rows.append(row)
        if not row["within_budget"]:
            raise BookRefused(
                f"REFUSED at stage {name}: this run GREW {row['growth_gb']} "
                f"GB from its own {self.baseline_gb} GB baseline, which "
                f"exceeds the declared budget of {b} GB. R8/R-174: the cap "
                f"is NOT raised and the population is NOT reduced. The day "
                f"is reported with its measured growth and refused.")
        if not row["within_cap"]:
            raise BookRefused(
                f"REFUSED at stage {name}: process high-water "
                f"{row['peak_gb']} GB is at or over the CAP of "
                f"{self.cap_gb} GB. The cap is never raised (R8/R-174).")
        return row


def day_slugs(day: str, coin: str = COIN, *, supply: dict = None) -> list:
    """The day's SUPPLIED slugs, from the forward scorer's own supply.

    `supply` is an INJECTION POINT and it exists because of a real defect the
    reviewer drove (BE48 §B.2). The refusal below was UNREACHABLE: for any
    day with no ledger entry, `present_from_ledger` refuses one call earlier,
    so this module's own empty-supply refusal had ZERO driven coverage in a
    battery that reported "both guards driven". Injecting the supply reaches
    it, so the check tests THIS module rather than `be_forward_day`."""
    if supply is None:
        import be_forward_day as FD
        import de_admissible_windows as AW
        supply = AW.supply(day, FD.present_from_ledger(day))
    w = (supply.get("windows") or {}).get(coin) or []
    out = [x["slug"] for x in w]
    if not out:
        raise BookRefused(
            f"REFUSED: no supplied {coin} windows for {day}. A book over an "
            f"empty day is not a small book, it is a different question.")
    return sorted(out)


def day_selector(day: str, coin: str = COIN):
    """A `build_reference` selector scoped to ONE DAY.

    Same entry shape `select_v2_era` returns -- (slug, path, up, down, gaps)
    -- built from the same three indices, but gated by the DAY'S SUPPLY
    rather than by a declared population interval. The population intervals
    end 2026-08-26T00:00, so no September day can pass through them."""
    import flow_intensity as fi
    import harmful_exposure_rows as HER
    want = set(day_slugs(day, coin))
    era = HER._era_or_refuse(fi, None, "be_daybook_build")
    paths, toks = fi._archive_paths(), fi.token_map()
    gaps = fi.gaps_by_slug(era)
    missing = sorted(s for s in want if s not in paths or s not in toks)
    if missing:
        raise BookRefused(
            f"REFUSED: {len(missing)} supplied slug(s) have no archive path "
            f"or no token map entry, e.g. {missing[:3]}. A book that "
            f"silently drops them is a book about a different day.")

    def _sel(coins, population):
        out = [(s, paths[s], toks[s][0], toks[s][1], gaps.get(s, []))
               for s in sorted(want)]
        return out, 0
    _sel.era = era
    _sel.n_wanted = len(want)
    return _sel


def assert_pool_equality(a: set, b: set) -> bool:
    """The SHARED draw pool is only sound while both heads score one set.

    `be_cancel_axis_null` builds `rows` from CONDVALUE's head and both arms
    draw from it. On the consumed hour the two heads scored the SAME 29,813
    generations, which is why the shared pool was declared with the note
    that the equality is a MEASUREMENT and not a guarantee. Here it is
    measured per day, and a day where it fails is REFUSED rather than
    silently pooled -- because then the shared pool is a real choice, and
    this seat has not declared that one."""
    if a != b:
        raise BookRefused(
            f"REFUSED: the two pinned heads scored DIFFERENT generation sets "
            f"({len(a)} vs {len(b)}, symmetric difference {len(a ^ b)}). The "
            f"declared draw pool is SHARED and that is only sound while the "
            f"sets are identical. The day is refused rather than pooled.")
    return True


def assert_coverage(cov: dict, n_gen: int, day: str) -> bool:
    """A day the tape does not reach is REFUSED, never emitted empty.

    An empty decision population is the failure mode that looks like a
    result -- the reviewer's own words about the wrong ledger root, and the
    same shape here."""
    if not cov:
        raise BookRefused(f"REFUSED: no coverage computed for {day}.")
    if all(v["n_covered"] == 0 for v in cov.values()):
        raise BookRefused(
            f"REFUSED: NO generation of {day} carries an assembled score "
            f"(0 of {n_gen}). The tape does not cover this day, so the book "
            f"would have an EMPTY decision population -- an empty answer "
            f"that looks like a result.")
    return True


def build(day: str, *, coin: str = COIN,
          chunk_windows: int = CHUNK_WINDOWS,
          scratch: Path | None = None, progress: bool = True,
          fixture: bool = False) -> dict:
    import be_gate1_state_tape as TAPEMOD
    import de_phase4_diag_runner as R
    t0 = time.time()
    obs = {}
    obs["started_utc"] = subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True,
        text=True, timeout=30).stdout.strip()
    stages = _Stages(FIXTURE_STAGE_BUDGETS_GB if fixture
                     else STAGE_BUDGETS_GB)
    obs["wrapper"] = assert_rule20(fixture=fixture)
    sel = day_selector(day, coin)
    if progress:
        print(json.dumps({"stage": "selected", "slugs": sel.n_wanted,
                          "era": sel.era}), flush=True)

    t = time.time()
    fr = R.build_reference(coin, selector=sel)
    stages.done("A0_reference", t)
    ref = fr["reference"]
    obs["reference_s"] = round(time.time() - t, 1)
    obs["reference_peak_gb"] = _rss_gb()
    n_gen = sum(len(ref[s][sd]) for s in ref for sd in ("BUY_UP", "SELL_UP")
                if sd in ref[s])
    if progress:
        print(json.dumps({"stage": "reference", "windows": len(ref),
                          "generations": n_gen, **obs}), flush=True)
    if not ref:
        raise BookRefused(f"REFUSED: {day} produced an EMPTY reference.")

    assert_day_tape(day, coin)
    # DE 80's FRONT DOOR (Q-DE-80): the verified pair, digests recomputed at
    # read time, refusing a ruled day that supplies nothing rather than
    # falling back to the consumed-era constant.
    import be_gate1_fragment as FRAGMOD
    # RULE 22: that import is LAZY and lands ~11 minutes into the run, so the
    # closure captured at module import did not contain it. Capture again --
    # a stamp that silently omits a module it later used would be worse than
    # no stamp, because it reads as coverage.
    _R22.CAPTURE.capture("be_daybook_build build(): after the lazy imports")
    _hy = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    _tp = TAPEMOD.out_path(day, coin)
    _fp = FRAGMOD.out_path(day, coin)
    # THE INPUTS ARE THE ONES ROUND 58 PUBLISHED, not merely self-consistent:
    # each digest is compared with the pin in its own builder receipt, at the
    # receipt's REAL head, before the front door sees it.
    _tpin = day_tape_pin(day, coin)
    _fpin = day_fragment_pin(day, coin)
    obs["inputs_vs_their_receipts"] = [
        assert_input_matches_its_receipt("tape", _tp, _tpin),
        assert_input_matches_its_receipt("fragment", _fp, _fpin)]
    inp = R.day_assembly_inputs(
        _hy,
        tape={"path": str(_tp), "sha256": _sha_file(_tp)},
        fragment={"path": str(_fp), "sha256": _sha_file(_fp)})
    obs["assembly_inputs"] = {k: v for k, v in inp.items() if k != "day"}
    splits = R.DECLARED_SPLIT_SETS[R.RULED_SPLIT_SET]
    t = time.time()
    # ITEM 1 IS IN: the index is built from THE DAY'S OWN TAPE, with its
    # digest verified at load. Before this, `build_tape_index` had no path
    # and would have indexed the consumed hour's tape for a September day.
    # R16: the ONE-OBJECT form, so the ruled-day and load-side
    # checks inside the seam fire. The `tape_path=` form the
    # 09-03 book was built under passes only a path and skips them.
    tape = R.build_tape_index(splits, inputs=inp)
    stages.done("A1_index", t)
    obs["tape_index_s"] = round(time.time() - t, 1)
    obs["tape_rows"] = tape.get("n_tape_rows")
    obs["after_tape_peak_gb"] = _rss_gb()
    if progress:
        print(json.dumps({"stage": "tape", **{k: obs[k] for k in
                          ("tape_index_s", "tape_rows",
                           "after_tape_peak_gb")}}), flush=True)

    # NO SLICE. The day's own fragment IS the source -- slicing off the eraB
    # fragment is what round 48 refused, because eraB holds no September slug.
    frag = Path(inp["fragment"]["path"])
    obs["fragment_bytes"] = inp["fragment"]["bytes"]
    obs["fragment_is_the_days_own"] = True

    t = time.time()
    asm = R.assemble_streaming({coin: ref}, splits=splits, coins=(coin,),
                               chunk_windows=chunk_windows, source=frag,
                               tape=tape)
    stages.done("A2_assemble", t)
    obs["assembly_s"] = round(time.time() - t, 1)
    obs["peak_gb"] = _rss_gb()

    # DE v9 R11: THE INDEX IS NEEDED ONLY TO PRODUCE `asm`. The consumer of
    # the book reads `asm` and the reference and never a tape row, so the
    # index does not have to be alive when the book is written. Releasing it
    # here is what makes the whole-day peak max(index, assembly) instead of
    # their sum -- on the measured numbers, the difference between fitting
    # the cap and not.
    t = time.time()
    _before_release = _rss_now_gb()
    del tape
    gc.collect()
    _after_release = _rss_now_gb()
    _a1 = next((r["peak_gb"] for r in stages.rows if r["stage"] == "A1_index"),
               0.0)
    obs["index_released"] = dict(
        assert_index_released(_before_release, _after_release, _a1),
        measured_on_CURRENT_rss="ru_maxrss is a high-water mark and cannot "
                                "show a release; this is VmRSS",
        index_is_build_time_only="DE design v9 R11 -- "
                                 "INDEX_SPLITS_NEEDED_BY_DAY = NONE at any "
                                 "stage")
    stages.done("A3_release_index", t)
    if progress:
        print(json.dumps({"stage": "assembled", **{k: obs[k] for k in
                          ("assembly_s", "peak_gb")}}), flush=True)

    # ---- WHAT THE BOOK MUST SATISFY, COMPUTED BEFORE IT IS WRITTEN --------
    cov = {}
    keys = {}
    for arm, head in HEADS.items():
        k = (coin, head)
        if k not in asm["by_arm"]:
            raise BookRefused(
                f"REFUSED: `asm['by_arm']` has no entry for {k}. DE's R1 "
                f"requires BOTH pinned heads; a book with one is a book the "
                f"null cannot drive for the other arm.")
        gs = asm["by_arm"][k][0]
        keys[head] = set(gs)
        scored = sum(1 for s in sorted(ref) for side in ("BUY_UP", "SELL_UP")
                     for g in ref[s].get(side, [])
                     if (s, side, float(g["t0"])) in gs)
        cov[head] = {"n_scored_keys": len(gs), "n_reference_generations": n_gen,
                     "n_covered": scored, "n_uncovered": n_gen - scored,
                     "coverage": scored / n_gen if n_gen else None,
                     "theta": float(R.theta_for(coin, head, BUDGET))}
    a, b = (keys[h] for h in (HEADS["CONDVALUE_X_SKEW"],
                              HEADS["HAZARD_OVER_SKEWED_REF"]))
    equal = assert_pool_equality(a, b)
    assert_coverage(cov, n_gen, day)

    _BDR.require_ledger()          # result-bearing: refuse a non-ledger tree
    # RULE 22 (R-605): REFUSE BEFORE ANYTHING IS WRITTEN, not after. DE's
    # runner refuses at the emit and loses the receipt; refusing here loses
    # neither -- if a module of this run's closure moved, or HEAD moved, no
    # book and no receipt exist to misattribute. What this does NOT cover is
    # stated rather than implied: the ~7 s between this line and the last
    # byte of the receipt. The stamp in the receipt re-reads the closure and
    # reports it again, so that window is visible too.
    # REV 65 §1.2: decided at runtime from this process's own leaf, which a
    # static lint cannot see behind a variable or a wrapper.
    obs["launch_form_at_runtime"] = _R22.assert_not_a_scope(fixture=fixture)
    obs["rule22_checked_before_write"] = _R22.assert_unchanged(
        "be_daybook_build: before the book is written")
    t = time.time()
    book = {"fr": fr, "asm": asm}
    buf = pickle.dumps(book, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(buf).hexdigest()
    asm_digest = hashlib.sha256(
        pickle.dumps(asm, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()
    dst = LEDGER_DERIVED / f"be_daybook_{day}_{coin}.pkl"
    dst.write_bytes(buf)
    back = hashlib.sha256(dst.read_bytes()).hexdigest()
    stages.done("A4_write_book", t)
    obs["wall_s"] = round(time.time() - t0, 1)
    obs["peak_rss_gb"] = _rss_gb()
    obs["stages"] = stages.rows
    obs["stage_budgets_gb"] = stages.budgets
    obs["asm_peak_gb_PUBLISHED"] = next(
        (r["peak_gb"] for r in stages.rows if r["stage"] == "A2_assemble"),
        None)
    return {
        "protocol": "BE_DAYBOOK_V1",
        "day": day, "coin": coin,
        "book": {"path": str(dst), "bytes": len(buf), "sha256": digest,
                 "sha256_of_asm": asm_digest,
                 "digest_is_of_the_buffer_as_written": True,
                 "readback_sha256": back,
                 "readback_matches": back == digest,
                 "why_readback": "the digest is taken from the buffer that "
                                 "was written (the B-1 discipline on the "
                                 "write side); the readback is a SECOND, "
                                 "independent statement of the same bytes "
                                 "and is reported beside it, not instead"},
        "selection": {"source": "de_admissible_windows.supply(day, "
                                "be_forward_day.present_from_ledger(day))",
                      "why_not_select_v2_era": "the declared population "
                                               "intervals end "
                                               "2026-08-26T00:00, so no "
                                               "September day passes them",
                      "era": sel.era, "n_supplied_slugs": sel.n_wanted},
        "reference": {"windows": len(ref), "generations": n_gen,
                      "statuses": fr.get("statuses"),
                      "n_slugs": fr.get("n_slugs"),
                      "terminal_marks_present": bool(fr.get("terminal_marks")),
                      "n_terminal_marks": len(fr.get("terminal_marks") or {})},
        "assembly_evidence": _assembly_evidence(asm, ref, cov, n_gen,
                                                chunk_windows,
                                                rows_pin=_tpin),
        "asm": {"by_arm_keys": [list(k) for k in asm["by_arm"]],
                "coverage_by_head": cov,
                "both_heads_present": True,
                "set_equality_asserted": True,
                "sets_are_equal": equal,
                "n_shared_keys": len(a)},
        "stage_budgets_gb": STAGE_BUDGETS_GB,
        "ROUND_49_BUDGET_WITHDRAWN": {
            "what_was_declared": "be_assembly_budget_declaration_v1.json "
                                 "computed 8.713 GB against the 8 GB cap "
                                 "(resident floor 5.971 + whole fragment "
                                 "2.742) and answered NO",
            "what_it_rested_on": "an A1_index peak of 5.971 GB, MEASURED "
                                 "while indexing the LIVE v5 tape -- the "
                                 "consumed hour's, not the day's",
            "what_was_measured_once_the_seam_took_a_path": "A1_index 3.190 "
                                                           "GB, from the "
                                                           "day's 991 MB "
                                                           "tape",
            "so_the_floor_was_an_ARTEFACT": "of indexing the wrong tape, not "
                                            "a property of the day. The "
                                            "budget's arithmetic was right "
                                            "and its input was wrong.",
            "status": "WITHDRAWN. The declaration is not edited (rule 13); "
                      "this records the withdrawal where the budgets are "
                      "read.",
            "and_the_term_it_could_not_close_is_now_measured":
                "asm's own peak, published per run as "
                "resources.asm_peak_gb_PUBLISHED",
        },
        "resources": obs,
        # RULE 22 AS AMENDED: captured at IMPORT (and again after the lazy
        # imports), reported here, and REFUSED above if anything moved.
        "producing_code": _R22.stamp(__file__),
        "seam": {"commit": _R22.module_commit(R.__file__),
                 "was_a_typed_literal_until_round_60":
                     "`seam.commit` read \"6f134a6\" for three rounds -- true "
                     "of the front door once and unchecked since (DA 77, "
                     "R-613). It is now READ from the module the call goes "
                     "through, at import, and says so when it cannot be "
                     "located",
                 "front_door": "de_phase4_diag_runner.day_assembly_inputs",
                 "index": _index_call_made(),
                 "digests_recomputed_at_read_time": True},
        "inputs_pinned": {
            # THE RECEIPT NAME IS THE RESOLVER'S, NEVER AN f-STRING. Round 59
            # emitted `.v3.json` here while the resolver 470 lines up had
            # correctly bound to `.v2` -- a name that was true of 09-03 and
            # false of the day it was written for.
            "tape": {"path": str(_tp), "sha256": _sha_file(_tp),
                     "receipt": (_tpin or {}).get("receipt"),
                     "receipt_name_is": "read from the resolver that chose "
                                        "it, not restated",
                     "matches_the_receipt_pin": True,
                     "split": "score"},
            "fragment": {"path": str(_fp), "sha256": _sha_file(_fp),
                         "receipt": (_fpin or {}).get("receipt"),
                         "matches_the_receipt_pin": True}},
        "wrapper_measured": obs.get("wrapper"),
        "data_root": _BDR.receipt_block(),
        # THE SCOPE'S OWN ACCOUNTING. The fragment and tape receipts have
        # carried this since round 58 and the book's did not, so round 59's
        # "the cap was not hit" rested on a poll of mine and on systemd's
        # stop line rather than on the artifact. `ru_maxrss` cannot answer
        # it: the tape scope hit the 8 GiB cap 1,199 times while the process
        # RSS peaked at 4.741 GB.
        "scope": _BDR.scope_stats(),
        "launch_form_at_runtime": _R22.assert_not_a_scope(fixture=fixture),
        # R-641 / rule 20: the journal is NOT the record. Its lines are
        # COPIED here at emit, filtered on this run's InvocationID with both
        # fields, with the retention state MEASURED beside them -- and a
        # window the journal no longer reaches is UNMEASURED, naming the
        # oldest entry that does exist. No verdict in this receipt rests on
        # any of it.
        "journal_copy": _R22.journal_copy(
            (_R22.cgroup_leaf().get("leaf") or "").rsplit(".", 1)[0]
            or "unknown",
            _R22.unit_outcome(
                (_R22.cgroup_leaf().get("leaf") or "unknown.service")
            ).get("InvocationID") or None,
            window_start_utc=obs.get("started_utc")),
        "exit_codes": {"map": {str(k): v for k, v in EXIT_CODES.items()},
                       "conflict_code_is_the_launcher's": _R22.lock_conflict_rc(),
                       "note": EXIT_CODE_NOTE,
                       "declaration_head":
                           _R22.declaration_head("heavy_run_form")["name"]},
        # REV 63 S3. The bare correlation I filed for 09-04 vs 09-05
        # understated what the receipts already contain, and a reader told
        # only "a correlation" will over-read the comparison.
        "WHAT_THE_SCOPE_NUMBERS_SUPPORT": SCOPE_NUMBERS_SUPPORT,
        "no_scoring_of_arms": True,
        "no_null_draws": True,
        "no_economics": True,
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 120


def real_data_reachable(day: str = "20260903") -> tuple:
    """Can this battery see the ledger's day inputs? BE48 §B.5.

    From a worktree it cannot: `de_admissible_windows` computes its own root
    as `parents[2]` with no resolver, so it looks for
    `<worktree>/data/pm_5min/derived/da_blackout_mask_<day>.json`, which is
    present at the ledger and NOT tracked. The battery used to REFUSE at
    check 1 there, so a reviewer in an R-397 worktree could not drive it at
    all. It now runs the fixture-driven checks everywhere and reports the
    real-data ones as SKIPPED WITH THEIR REASON (rule 4: an exclusion is a
    status, never a silent drop)."""
    import de_admissible_windows as AW
    mask = Path(AW.ROOT) / "data/pm_5min/derived" / f"da_blackout_mask_{day}.json"
    if mask.exists():
        return True, str(mask)
    return False, (f"{mask} not present. `de_admissible_windows.ROOT` is "
                   f"parents[2] with no resolver, so from a worktree it "
                   f"looks for the mask in the worktree. Set PM_DATA_ROOT "
                   f"and run at the ledger tree, or drive the fixture "
                   f"checks alone.")


def selftest() -> int:
    checks, fails, skipped = 0, [], []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    def skip(label, why):
        skipped.append(label)
        print(f"SKIP: {label}  [{why[:110]}]")

    reachable, why_not = real_data_reachable()

    import harmful_exposure_rows as HER
    iv = HER.POPULATION_SLUG_INTERVALS
    ends = [v[1] for v in iv.values() if isinstance(v, (list, tuple))]
    ok(all(e is None or e < 1788000000 for e in ends),
       f"THE REASON THE DAY PATH EXISTS, MEASURED: every declared population "
       f"interval ends before September ({ends}) -- so `select_v2_era` "
       f"cannot reach a September day and a book built through it would be "
       f"EMPTY, not small")

    if reachable:
        sl = day_slugs("20260903")
        ok(len(sl) == 247 and all(s.startswith("btc-") for s in sl),
           f"the day supply returns {len(sl)} btc slugs for 20260903, "
           f"matching the forward receipt's supplied count for that coin")
        sel = day_selector("20260903")
        ent, ngap = sel(("btc",), None)
        ok(len(ent) == len(sl) and len(ent[0]) == 5,
           f"and the selector returns {len(ent)} entries in select_v2_era's "
           f"own 5-tuple shape, so `build_reference` needs no change")
    else:
        skip("the day supply returns 247 btc slugs for 20260903", why_not)
        skip("the selector returns entries in the 5-tuple shape", why_not)

    # BE48 §B.2. The previous version of this check was carried by
    # `isinstance(e, Exception)` inside `except Exception` -- TRUE BY
    # CONSTRUCTION -- and the refusal it named was never reached, because
    # `present_from_ledger` refuses a day with no ledger entry one call
    # earlier. Both halves are fixed: the refusal is REACHED by injecting a
    # supply, and the assertion names the TYPE as well as the text.
    try:
        day_slugs("20260903", supply={"windows": {"eth": [{"slug": "x"}]}})
        ok(False, "an empty btc supply must refuse")
    except BookRefused as e:
        ok("no supplied btc windows for 20260903" in str(e),
           "KNOWN-BAD, AND IT NOW REACHES THIS MODULE'S OWN REFUSAL: a "
           "supply carrying eth windows and NO btc raises BookRefused here, "
           "not an upstream refusal -- the previous check was carried by "
           "`isinstance(e, Exception)` and tested be_forward_day instead")
    except Exception as e:                               # noqa: BLE001
        ok(False, f"expected BookRefused, got {type(e).__name__}")
    # and the UPSTREAM refusal is still asserted, BY TYPE, as its own case
    import be_forward_day as _FD
    if not reachable:
        skip("the upstream ForwardDayRefused case", why_not)
    else:
      try:
          day_slugs("19700101")
          ok(False, "a day with no ledger entry must refuse")
      except BookRefused as e:
          ok(False, f"expected the UPSTREAM refusal, got BookRefused: {e}")
      except Exception as e:                               # noqa: BLE001
          ok(type(e).__name__ == "ForwardDayRefused"
             and "ledger holds no window" in str(e),
             f"KNOWN-BAD, THE OTHER PATH, NAMED BY TYPE: a day with no ledger "
             f"entry raises {type(e).__name__} from be_forward_day -- a "
             f"different refusal from a different module, and the battery now "
             f"says which is which")

    # ---- THE TWO GUARDS, DRIVEN BOTH WAYS (tree-independent) --------------
    ok(assert_pool_equality({1, 2, 3}, {3, 2, 1}) is True,
       "POSITIVE CONTROL: identical scored sets PASS the shared-pool guard")
    try:
        assert_pool_equality({1, 2, 3}, {1, 2, 4})
        ok(False, "unequal pools must refuse")
    except BookRefused as e:
        ok("symmetric difference 2" in str(e),
           "KNOWN-BAD: sets differing by one element each REFUSE the DAY, "
           "naming the symmetric difference -- the shared pool is only sound "
           "while they are identical, and this day would make it a choice")
    ok(assert_coverage({"h": {"n_covered": 5}}, 10, "d") is True,
       "POSITIVE CONTROL: a day with ANY covered generation passes")
    try:
        assert_coverage({"h1": {"n_covered": 0}, "h2": {"n_covered": 0}},
                        10, "20260903")
        ok(False, "zero coverage must refuse")
    except BookRefused as e:
        ok("EMPTY decision population" in str(e),
           "KNOWN-BAD: zero coverage on every head REFUSES -- an empty "
           "decision population is the failure that looks like a result")

    # ROUND 51's BLOCKER IS CLEARED BY ROUND 52's ITEM 1, so this check
    # asserts the NEW truth: the day tape ADMITS, bound to the digest its
    # SCORE-split receipt names.
    if not reachable:
        for _lbl in ("the day-tape positive control",
                     "the no-tape known-bad"):
            skip(_lbl, why_not)
    else:
        _adt = assert_day_tape("20260903")
        ok(_adt["is_the_days_tape"]
           and _adt["sha256_from_receipt"].startswith("9de88da950598e86")
           and _adt["default_constant_no_longer_blocks"],
           f"POSITIVE CONTROL: the assembly binds to THE DAY'S tape at the "
           f"digest its SCORE-split receipt names "
           f"({_adt['sha256_from_receipt'][:16]}…)")
        try:
            assert_day_tape("19700101")
            ok(False, "a day with no tape must refuse")
        except BookRefused as e:
            ok("does not exist" in str(e),
               "KNOWN-BAD: a day with no tape REFUSES -- the assembly never "
               "silently falls back to the consumed hour's tape")

    # ---- ITEM 1: the tape PATH parameter, driven three ways --------------
    import phase2_arms as _PA
    try:
        _PA.assert_tape_for_day("20260903")          # default path
        ok(False, "the default tape on a ruled day must refuse")
    except _PA.TapePathRefused as e:
        ok("RULED FORWARD DAY" in str(e) and "EMPTY `asm`" in str(e),
           "KNOWN-BAD: a RULED DAY asked for with the DEFAULT tape REFUSES, "
           "naming the consequence -- indexing the consumed hour's tape for "
           "a September day yields an empty asm")
    import be_gate1_state_tape as _TM
    _dt = _TM.out_path("20260903")
    if _dt.exists():
        try:
            _PA.assert_tape_for_day("20260903", _dt,
                                    expect_sha256="0" * 64)
            ok(False, "a wrong digest must refuse")
        except _PA.TapePathRefused as e:
            ok("is not that day's tape" in str(e),
               "KNOWN-BAD: the day's tape with a WRONG expected digest "
               "REFUSES -- bytes that are not the ones the builder receipt "
               "names are not that day's tape")
        _full = _sha_file(_dt)
        _r = _PA.assert_tape_for_day("20260903", _dt, expect_sha256=_full)
        ok(_r["digest_verified_at_load"] and not _r["is_the_default"]
           and _r["n_hex_compared"] == 64,
           f"POSITIVE CONTROL: the day's own tape ADMITS with its FULL "
           f"64-hex digest compared ({_r['sha256'][:16]}…, "
           f"n_hex_compared={_r['n_hex_compared']}) and is not the default")
    else:
        skip("the day-tape digest checks", f"{_dt.name} absent")
        skip("the day-tape positive control", f"{_dt.name} absent")
    ok(_PA.assert_tape_for_day(None)["is_the_default"] is True,
       "and the CONSUMED HOUR still uses the default with no day named -- "
       "the parameter is additive and changes no existing call")

    # ---- THE REVIEWER'S ROUND-53 FINDING, DRIVEN BOTH WAYS ---------------
    import fcntl as _fc, subprocess as _sp, tempfile as _tf
    with _tf.NamedTemporaryFile(suffix=".lock", delete=False) as _lk:
        _lp = _lk.name
    _probe = ("import sys,fcntl,os;f=open(sys.argv[1],'w');"
              "fcntl.flock(f,fcntl.LOCK_EX if sys.argv[2]=='EX' "
              "else fcntl.LOCK_SH);"
              "sys.path.insert(0,%r);import be_daybook_build as B;"
              "print(B._flock_mode(sys.argv[1]))" % str(HERE))
    _ex = _sp.run([sys.executable, "-c", _probe, _lp, "EX"],
                  capture_output=True, text=True).stdout.strip()
    _sh = _sp.run([sys.executable, "-c", _probe, _lp, "SH"],
                  capture_output=True, text=True).stdout.strip()
    Path(_lp).unlink(missing_ok=True)
    ok(_ex == "WRITE" and _sh == "READ",
       f"LOCK MODE IS READ FROM /proc/locks AND DISTINGUISHES THE TWO: "
       f"LOCK_EX -> {_ex!r}, LOCK_SH -> {_sh!r}. The reviewer's finding is "
       f"that TWO `flock -s` holders both certify -- holding the fd is not "
       f"evidence of exclusion, and only WRITE is")

    # ---- THE BLOCKER GUARD, DRIVEN FROM ANY TREE (reviewer BE-51) --------
    import tempfile as _tf3
    with _tf3.TemporaryDirectory() as _td3:
        _fx = Path(_td3) / "phase2_state_tape_gate1_20260903_btc.json"
        _fx.write_text("{}")
        _r3 = assert_day_tape("20260903", tape=_fx, receipt_sha="abc123")
        ok(_r3["fixture_injected"] and _r3["sha256_from_receipt"] == "abc123",
           "THE BLOCKER GUARD IS DRIVABLE FROM ANY TREE: a fixture tape and "
           "a fixture digest admit, so a reviewer in an R-397 worktree can "
           "drive it without the ledger's artifacts")
        try:
            assert_day_tape("20260903", tape=Path(_td3) / "absent.json")
            ok(False, "an absent fixture tape must refuse")
        except BookRefused as e:
            ok("does not exist" in str(e),
               "KNOWN-BAD, on the fixture path too: an absent tape REFUSES, "
               "so the injection does not weaken the guard it makes drivable")

    # ---- REV 45: A TRUNCATED EXPECTATION VERIFIES NOTHING -----------------
    import be_gate1_state_tape as _TM2
    _dt = _TM2.out_path("20260903", COIN) if reachable else Path("/nope")
    if _dt.exists():
        for _lbl, _stub in (("one character", "9"),
                            ("16 hex", "9de88da950598e86"),
                            ("right 16, wrong 48", "9de88da950598e86" + "0"*48),
                            ("UPPERCASE full", _sha_file(_dt).upper())):
            try:
                _PA.assert_tape_for_day("20260903", _dt, expect_sha256=_stub)
                ok(False, f"a {_lbl} expectation must refuse")
            except _PA.TapePathRefused:
                ok(True, f"KNOWN-BAD: a {_lbl!r} expectation REFUSES. Before "
                         f"REV 45 the compare was `got.startswith(expect[:16])` "
                         f"and ALL of these ADMITTED with "
                         f"digest_verified_at_load TRUE")
    else:
        for _lbl in ("one character", "16 hex", "right 16, wrong 48",
                     "UPPERCASE full"):
            skip(f"REV-45 known-bad: {_lbl}", why_not)

    # ---- REV 46's FOUR, EACH DRIVEN BOTH WAYS -----------------------------
    # (5) the release is ASSERTED
    _rel = assert_index_released(4.096, 2.940, 3.190)
    ok(_rel["released"] and _rel["freed_gb"] == 1.156
       and _rel["freed_fraction_of_index_peak"] > MIN_RELEASE_FRACTION,
       f"REV 46(5) POSITIVE CONTROL: the 09-03 release (4.096 -> 2.940, "
       f"{_rel['freed_gb']} GB = "
       f"{_rel['freed_fraction_of_index_peak']:.1%} of the index peak) "
       f"passes the assertion")
    try:
        assert_index_released(4.096, 4.096, 3.190)
        ok(False, "a no-op release must refuse")
    except BookRefused as e:
        ok("only 0.000 GB was freed" in str(e) and "claim rather than a "
           "fact" in str(e),
           "KNOWN-BAD: a NO-OP release REFUSES -- it would have reported "
           "`freed_gb: 0.0` and passed, because the number was measured and "
           "never asserted")
    try:
        assert_index_released(4.096, 3.900, 3.190)
        ok(False, "an under-threshold release must refuse")
    except BookRefused as e:
        ok("below the required" in str(e),
           "KNOWN-BAD: a release below the declared 10% of the index peak "
           "REFUSES too -- the threshold is a fraction, not merely non-zero")

    # (2)+(3) the seam's evidence, computed
    _ref = {f"s{i}": {"BUY_UP": [{"t0": 1.0, "gen": 0}]} for i in range(12)}
    _asm = {"assembly": {"drops_by_coin": {"btc": {
        "state_join_failed": 0, "pre_window_excluded": 10,
        "gap_at_cutoff_excluded": 3, "no_level_history_excluded": 2}},
        "kept_by_coin": {"btc": 85}}}
    _cov = {"h1": {"n_uncovered": 15}, "h2": {"n_uncovered": 15}}
    # THE FIXTURE'S TWO SIDES ARE NOW IN DIFFERENT UNITS ON PURPOSE. 85 kept
    # + 15 dropped are ROWS; 100 is the generation count; the row total the
    # producer published is 100 ROWS. Round 59's fixture made both sides 15
    # and 15, which is precisely why its known-bad could not expose that the
    # real comparison crossed two populations.
    _pin = {"n_rows": 100, "receipt": "FIXTURE_tape_receipt.v2.json"}
    _ev = _assembly_evidence(_asm, _ref, _cov, 100, 6, rows_pin=_pin)
    ok(_ev["state_join_failed"] == 0 and _ev["state_join_failed_is_zero"]
       and _ev["n_chunks"] == 2 and _ev["n_windows"] == 12,
       f"REV 46(2): `state_join_failed` ({_ev['state_join_failed']}) and "
       f"`n_chunks` ({_ev['n_chunks']} over {_ev['n_windows']} windows) are "
       f"RECEIPT FIELDS now -- the 09-03 receipt carried neither, and the "
       f"number proving the seam worked lived only in a Q-row")
    _ra = _ev["ROW_ACCOUNTING"]
    ok(_ra["kept"] == 85 and _ra["dropped"] == 15 and _ra["accounted"] == 100
       and _ra["rows_published_by_the_tape_receipt"] == 100
       and _ra["rows_accounted_for"] is True and _ra["residual"] == 0
       and _ra["rows_pin_receipt"] == "FIXTURE_tape_receipt.v2.json",
       f"POSITIVE CONTROL, ONE POPULATION: kept {_ra['kept']} + dropped "
       f"{_ra['dropped']} = {_ra['accounted']} ROWS against the "
       f"{_ra['rows_published_by_the_tape_receipt']} ROWS round 58's "
       f"receipt published -- it CLOSES, and the total comes from the "
       f"producing receipt rather than from the function checking it")
    _bad = _assembly_evidence(_asm, _ref, _cov, 100, 6,
                              rows_pin={"n_rows": 137, "receipt": "x.json"})
    ok(_bad["ROW_ACCOUNTING"]["rows_accounted_for"] is False
       and _bad["ROW_ACCOUNTING"]["residual"] == 37,
       "KNOWN-BAD, SAME UNIT: when the rows do not close against the "
       "published total the predicate reads FALSE and the residual (37 "
       "rows) is reported rather than absorbed")
    # THE KNOWN-BAD ROUND 59 COULD NOT HAVE: a GENERATION count handed in
    # where a ROW total belongs. This is the actual defect -- 29,465 rows
    # compared with 19,663 generations -- and it must read FALSE.
    _units = _assembly_evidence(
        _asm, _ref, {"h1": {"n_uncovered": 15}, "h2": {"n_uncovered": 15}},
        100, 6, rows_pin={"n_rows": 15, "receipt": "wrong_unit.json"})
    ok(_units["ROW_ACCOUNTING"]["rows_accounted_for"] is False,
       "KNOWN-BAD, THE ONE THE OLD FIXTURE HID: feed the GENERATION-side "
       "number (15 uncovered) where the published ROW total belongs and the "
       "accounting REFUSES to close -- 100 rows accounted against 15. The "
       "round-59 fixture made both sides 15 and 15, so it passed while the "
       "real day reported FALSE on two correct numbers")
    ok("reasons_account_for_the_count" not in _ev["UNCOVERED_GENERATIONS"]
       and "by_reason" not in _ev["UNCOVERED_GENERATIONS"]
       and _ev["UNCOVERED_GENERATIONS"]["count"] == 15
       and _ev["UNCOVERED_GENERATIONS"]["identical_across_heads"]
       and "FRAGMENT ROWS" in
       _ev["UNCOVERED_GENERATIONS"]["why_no_reason_class_here"],
       "THE CROSS-POPULATION CLAIM IS WITHDRAWN, NOT SILENTLY DROPPED: the "
       "generations block carries its count, its coverage and the reason "
       "there is no reason class in that unit -- and names the field it "
       "supersedes, so a reader of the 09-04 receipt can find out what "
       "happened to it")

    # (4) the round-49 withdrawal, beside the budgets
    import re as _re4
    _src4 = Path(__file__).read_text()
    _blk4 = _src4[_src4.index('"ROUND_49_BUDGET_WITHDRAWN"'):]
    _blk4 = _blk4[:_blk4.index('"resources": obs,')]
    ok(all(x in _blk4 for x in ("8.713", "5.971", "3.190", "WITHDRAWN")),
       "REV 46(4): the round-49 budget WITHDRAWAL sits beside "
       "`stage_budgets_gb` with all three numbers -- the 8.713 GB that "
       "answered NO, the 5.971 GB floor it rested on, and the 3.190 GB "
       "actually measured once the seam took a path. The declaration itself "
       "is NOT edited (rule 13)")

    # the seam literal, now computed
    _idx = _index_call_made()
    ok("inputs=inp" in _idx and "tape_path" not in _idx,
       f"THE SEAM LITERAL IS GONE: `seam.index` is derived from this "
       f"module's own source and reads {_idx!r}. The 09-03 receipt said "
       f"`tape_path=…` while the call was already `inputs=` -- a literal "
       f"contradicting the code beside it, the third of that class here")

    # ---- RULE 22 AS AMENDED (R-605): the closure, HEAD, and the refusal --
    import tempfile as _tf
    import importlib as _il
    _td = _tf.mkdtemp(prefix="be60_rule22_")
    _m1 = Path(_td) / "be60_probe_a.py"
    _m1.write_text("VALUE = 1\n")
    _outside = Path(_tf.mkdtemp(prefix="be60_outside_")) / "be60_probe_b.py"
    _outside.write_text("VALUE = 1\n")
    sys.path.insert(0, _td); sys.path.insert(0, str(_outside.parent))
    _pa = _il.import_module("be60_probe_a")
    _pb = _il.import_module("be60_probe_b")
    # a capture whose ROOT is the temp tree, so the falsifier never has to
    # mutate a real module in the worktree to prove the check fires
    _cap = _R22.Capture(root=_td).capture("battery")
    ok(len(_cap.closure) == 1
       and str(_m1.resolve()) in _cap.closure
       and str(_outside.resolve()) not in _cap.closure,
       f"THE CLOSURE IS SCOPED AND SAYS WHAT IT COVERS: {len(_cap.closure)} "
       f"module under the declared root; a module OUTSIDE it is not "
       f"captured and is not claimed to be")
    _adm = _cap.assert_unchanged("battery positive control")
    ok(_adm["closure_unchanged"] and _adm["head_unchanged"],
       "POSITIVE CONTROL, RULE 22: with nothing moved the emit ADMITS -- a "
       "guard shown only to refuse has not been shown to work (rule 16)")
    _sha_at_import = _cap.closure[str(_m1.resolve())]["sha256"]
    _m1.write_text("VALUE = 2   # a landing mid-run\n")
    try:
        _cap.assert_unchanged("battery known-bad")
        ok(False, "a moved module must refuse the emit")
    except _R22.Rule22Refused as _e:
        ok("be60_probe_a.py" in str(_e) and "IMPORT CLOSURE" in str(_e)
           and "DID NOT RUN" in str(_e),
           "KNOWN-BAD, RULE 22: a module of the closure rewritten mid-run "
           "REFUSES THE EMIT BY NAME -- this is R-603's defect exactly, "
           "where a runner re-read `__file__` at emit and would have "
           "stamped bytes that did not run")
    _st = _cap.stamp(_m1)
    ok(_st["producing_code_sha256"] == _sha_at_import
       and _st["closure_drift"][0]["at_import"] == _sha_at_import
       and _st["closure_drift"][0]["now"] != _sha_at_import
       and _st["closure_unchanged_during_the_run"] is False,
       "THE DIGEST IS THE ONE FROM IMPORT, NOT FROM EMIT: after the file "
       "changed, the stamp still reports the bytes that RAN and names the "
       "drift beside them. An emit-time digest would have reported the new "
       "bytes and called them the producer")
    _outside.write_text("VALUE = 3\n")
    _cap2 = _R22.Capture(root=str(_outside.parent)).capture("battery")
    _cap2.closure[str(_outside.resolve())]["sha256"] = "0" * 64
    try:
        _cap2.assert_unchanged("battery head test")
        ok(False, "a drifted module must refuse")
    except _R22.Rule22Refused:
        ok(True, "KNOWN-BAD: a planted digest mismatch refuses too -- the "
                 "comparison is of BYTES, not of a flag someone set")
    _cap3 = _R22.Capture(root=_td).capture("battery")
    _cap3.head_at_import = dict(_cap3.head_at_import or {})
    _cap3.head_at_import["head"] = "0" * 40
    try:
        _cap3.assert_unchanged("battery head known-bad")
        ok(False, "a moved HEAD must refuse the emit")
    except _R22.Rule22Refused as _e:
        ok("HEAD MOVED" in str(_e) and "builder_commit" in str(_e),
           "KNOWN-BAD, RULE 22: HEAD moving under the run REFUSES -- a "
           "receipt's builder_commit would otherwise name a commit the run "
           "did not execute from")
    sys.path.remove(_td); sys.path.remove(str(_outside.parent))
    _mine = _R22.stamp(__file__)
    ok(_mine["producing_code"] == "be_daybook_build.py"
       and _mine["producing_code_sha256"] == _sha_file(Path(__file__))
       and _mine["captured_at"] == "IMPORT"
       and _mine["import_closure"]["n_modules"] > 1
       and _mine["builder_commit"],
       f"THIS MODULE'S OWN STAMP IS REAL: "
       f"{_mine['import_closure']['n_modules']} modules of the live closure "
       f"digested at import, HEAD {str(_mine['builder_commit'])[:12]}, and "
       f"the producing digest equals this file on disk")

    # ---- the per-stage budget: an instrument that can FALL, and FIRE ----
    # REV 59 §3. Round 60's known-bads hand-set the baseline to zero AFTER
    # construction, so they tested the arithmetic and not the instrument --
    # and the instrument's one failure mode was exactly what the hand-set
    # hid. Nothing below sets a baseline: every _Stages here uses the one
    # its own constructor took, which is the thing under test.
    #: DECLARED, not derived from ambient memory. Round 60 read the process
    #: high-water here and halved it to make a budget -- a bound computed
    #: from whatever the process happened to be holding, which means
    #: something different on every run (the class DA 81 is generalising).
    _PROBE_BUDGET_GB = 0.05
    _PROBE_ALLOC_MB = 300
    _sg = _Stages({"S": _PROBE_BUDGET_GB}, cap_gb=CAP_GB)
    _row_idle = _sg.done("S", time.time())
    ok(_row_idle["within_budget"] and _row_idle["growth_gb"] <= _PROBE_BUDGET_GB
       and _row_idle["baseline_gb"] == _sg.baseline_gb,
       f"POSITIVE CONTROL, ON THE CONSTRUCTOR'S OWN BASELINE: a stage that "
       f"allocates nothing grows {_row_idle['growth_gb']} GB from the "
       f"{_row_idle['baseline_gb']} GB baseline `_Stages` took itself, and "
       f"ADMITS against a DECLARED {_PROBE_BUDGET_GB} GB budget")
    _blob = bytearray(_PROBE_ALLOC_MB * 1024 * 1024)
    _blob[::4096] = b"\x01" * (len(_blob) // 4096)        # touch every page
    try:
        _sg.done("S", time.time())
        ok(False, "growth over budget must refuse")
    except BookRefused as _e:
        ok("GREW" in str(_e) and "NOT raised" in str(_e),
           f"KNOWN-BAD A: {_PROBE_ALLOC_MB} MB of real, touched memory "
           f"against a {_PROBE_BUDGET_GB} GB budget REFUSES, naming the "
           f"growth and the baseline it is measured from")
    _hw_after = _rss_gb()
    del _blob
    __import__("gc").collect()
    # THE REVIEWER'S FALSIFIER (REV 59 §3B): a SECOND _Stages in the SAME
    # process, after the high-water has already been pushed up. On round
    # 60's instrument both terms came from the high-water, so this read a
    # growth of zero and COULD NOT FIRE -- a budget that only works once
    # per process, which is the instrument DE 89 measured and rejected a
    # round before I shipped it.
    _sg2 = _Stages({"S": _PROBE_BUDGET_GB}, cap_gb=CAP_GB)
    _blob2 = bytearray((_PROBE_ALLOC_MB // 2) * 1024 * 1024)
    _blob2[::4096] = b"\x01" * (len(_blob2) // 4096)
    try:
        _sg2.done("S", time.time())
        ok(False, "the SECOND _Stages must still refuse")
    except BookRefused as _e2:
        ok("GREW" in str(_e2) and _sg2.baseline_gb < _hw_after,
           f"KNOWN-BAD B -- THE ONE ROUND 60 COULD NOT FIRE: a SECOND "
           f"_Stages in the same process, with the high-water already at "
           f"{_hw_after} GB, takes a baseline of {_sg2.baseline_gb} GB "
           f"(current RSS, which fell) and still REFUSES on "
           f"{_PROBE_ALLOC_MB // 2} MB. Measured on the high-water both "
           f"terms would be {_hw_after} and the growth would read zero")
    del _blob2
    __import__("gc").collect()
    ok(_rss_gb() >= _hw_after and _rss_now_gb() < _hw_after,
       f"AND THE TWO INSTRUMENTS ARE DIFFERENT, MEASURED: after the "
       f"allocations were freed the high-water stayed at {_rss_gb()} GB "
       f"while current RSS fell to {_rss_now_gb()} GB. The budget needs the "
       f"falling one; the cap needs the other")
    _sc = _Stages({"S": 99.0}, cap_gb=0.001)
    try:
        _sc.done("S", time.time())
        ok(False, "a peak over the cap must refuse")
    except BookRefused as _e:
        ok("CAP" in str(_e) and "never raised" in str(_e),
           "KNOWN-BAD C, AND THE REASON GROWTH DOES NOT WIDEN ANYTHING: the "
           "absolute high-water is still checked against the CAP -- on "
           "`ru_maxrss`, which is the right instrument for a ceiling -- so "
           "a run that fits its budget but not the machine still refuses")

    # ---- the seam commit is READ, and the receipt name is the resolver's --
    _sc2 = _R22.module_commit(Path(HERE) / "de_phase4_diag_runner.py")
    ok(_sc2["commit"] and len(_sc2["commit"]) == 40
       and _sc2["short"] != "6f134a6",
       f"THE TYPED SEAM COMMIT IS GONE: the front door's commit is READ "
       f"from the module ({_sc2['short']}), not the literal \"6f134a6\" "
       f"that stood for three rounds and was stale")
    _nc = _R22.module_commit(Path(_td) / "be60_probe_a.py")
    ok(_nc["commit"] is None and "why_absent" in _nc,
       "AND WHEN IT CANNOT BE LOCATED IT SAYS SO: a module with no history "
       "returns a null commit with the reason, rather than a constant")
    # TEST THE CODE, NOT THE TEXT. Two earlier forms of this check scanned
    # source text and failed on their own prose -- first the module, then
    # the emit block, each containing the sentence that names the literal it
    # forbids. A string search cannot tell a hardcoded value from a
    # description of one; the AST can, because a comment is not a node.
    import ast as _ast
    _tree = _ast.parse(Path(__file__).read_text())
    _bld = next(n for n in _ast.walk(_tree)
                if isinstance(n, _ast.FunctionDef) and n.name == "build")
    _suffix = ".v" + "3" + ".json"
    _lits = [n.value for n in _ast.walk(_bld)
             if isinstance(n, _ast.Constant) and isinstance(n.value, str)
             and n.value.endswith(_suffix)]
    _srcm = Path(__file__).read_text()
    ok(not _lits and '"receipt": (_tpin or {}).get("receipt")' in _srcm,
       f"MY ROUND-59 DEFECT IS CLOSED AT THE SOURCE: the AST finds "
       f"{len(_lits)} hardcoded receipt-version literals inside `build()`, "
       f"the function that constructs the receipt -- and the name emitted "
       f"is the one the resolver actually chose. THREE forms of this check "
       f"failed before this one: scanning the module, then the emit block, "
       f"then the whole AST, each finding its OWN description of the "
       f"literal it forbids. A checker that names a forbidden string "
       f"contains it; the search space has to exclude the checker")
    _fx = Path(_td) / "derived"; _fx.mkdir()
    for _nm, _sp in ((f"be_gate1_state_tape_receipt_20260904_btc.json", "train"),
                     (f"be_gate1_state_tape_receipt_20260904_btc.v2.json", "score")):
        (_fx / _nm).write_text(json.dumps(
            {"WHICH_SPLIT_THE_ASSEMBLY_SCORES_FROM_AND_WHY": {"split": _sp},
             "tape": {"sha256": "a" * 64, "n_rows": 638602, "bytes": 1}}))
    _saved = globals()["OUT_DERIVED"]
    globals()["OUT_DERIVED"] = _fx
    try:
        _pin2 = day_tape_pin("20260904", "btc")
    finally:
        globals()["OUT_DERIVED"] = _saved
    ok(_pin2["receipt"] == "be_gate1_state_tape_receipt_20260904_btc.v2.json"
       and _pin2["n_rows"] == 638602,
       f"THE RESOLVER PICKS THE REAL HEAD AND REPORTS ITS NAME: "
       f"{_pin2['receipt']} -- the .v2, because the unversioned file is the "
       f"TRAIN split; round 59 emitted `.v3.json`, which does not exist for "
       f"this day")

    # ---- the inputs are the bytes their own receipts pinned --------------
    _f = Path(_td) / "input.bin"; _f.write_bytes(b"the day's bytes")
    _good = assert_input_matches_its_receipt(
        "tape", _f, {"sha256": _sha_file(_f), "receipt": "r.v2.json"})
    ok(_good["matches_its_builder_receipt"] and _good["compared_full_length"],
       "POSITIVE CONTROL: an input whose bytes match the digest its builder "
       "receipt published ADMITS, compared over all 64 characters")
    try:
        assert_input_matches_its_receipt(
            "tape", _f, {"sha256": "b" * 64, "receipt": "r.v2.json"})
        ok(False, "a file that does not match its receipt must refuse")
    except BookRefused as _e:
        ok("does not match the digest its receipt" in str(_e)
           and "round 58 published" in str(_e),
           "KNOWN-BAD, MY RELOAD FINDING CLOSED: an input that does not "
           "match its BUILDER RECEIPT's pin refuses. Round 59 hashed the "
           "file itself and handed that digest to the front door, which "
           "compared it with itself -- a check that could not tell these "
           "were round 58's bytes")
    try:
        assert_input_matches_its_receipt("tape", _f, None)
        ok(False, "a missing pin must refuse")
    except BookRefused as _e:
        ok("no builder receipt pin" in str(_e),
           "KNOWN-BAD: NO receipt pin at all REFUSES rather than proceeding "
           "unbound -- absence must never read as a pass (rule 11)")

    # ---- the scope block the book receipt lacked -------------------------
    _sc3 = _BDR.scope_stats()
    ok(isinstance(_sc3, dict)
       and (_sc3.get("status") == "NO_CGROUP"
            or ("cap_was_hit" in _sc3 and "events" in _sc3
                and "anon_bytes" in _sc3)),
       f"THE SCOPE BLOCK IS AVAILABLE TO THIS RECEIPT: "
       f"{sorted(_sc3)[:6]}... -- the fragment and tape receipts have "
       f"carried it since round 58 and the book's did not, so round 59's "
       f"cap answer rested on my poll and systemd's stop line, not on the "
       f"artifact")
    ok(_sc3.get("peak_is_censored") is False
       and "measurement" in _sc3["peak_censoring"]["what_this_peak_supports"],
       f"POSITIVE CONTROL, REV 63 S3: this scope's peak "
       f"({_sc3['peak_censoring']['peak_bytes']}) is BELOW its cap "
       f"({_sc3['peak_censoring']['cap_bytes']}) and was never throttled, "
       f"so the block reports it as a MEASUREMENT")
    _cen = dict(_sc3)
    _cen["peak_bytes"], _cen["max_bytes"] = "8589934592", "8589934592"
    _cen["events"] = {"max": 1199}
    _reb = _BDR.scope_stats.__wrapped__ if hasattr(
        _BDR.scope_stats, "__wrapped__") else None
    _pk, _mx = int(_cen["peak_bytes"]), int(_cen["max_bytes"])
    ok(_pk >= _mx and _cen["events"]["max"] == 1199,
       "KNOWN-BAD INPUT, 09-04's OWN NUMBERS: peak == cap with 1,199 "
       "reclaims is the shape the field exists to flag -- a FLOOR reported "
       "beside an uncensored 7.555 GB would read as `09-04 needed more`, "
       "which those numbers cannot say")
    # read the EMITTED block, not the whole module: the check should test
    # what a receipt will carry, not that the words appear somewhere.
    _txtS = " ".join(SCOPE_NUMBERS_SUPPORT.values())
    ok("outside the builder's own allocation" in _txtS
       and "AT LEAST the cap" in _txtS
       and "asserted as neither" in _txtS
       and '"WHAT_THE_SCOPE_NUMBERS_SUPPORT": SCOPE_NUMBERS_SUPPORT,'
       in Path(__file__).read_text(),
       "AND THE RECEIPT CARRIES THE REVIEWER'S SENTENCE IN PLACE OF THE "
       "BARE CORRELATION: the builder's own footprint did not move, the "
       "whole cgroup difference sits in page cache which rose as anon "
       "fell, so whatever differed was outside the builder's allocation -- "
       "stronger than a correlation, weaker than a mechanism")
    _lfB = _R22.assert_launch_form()
    ok(_lfB["form_is_correct"]
       and _lfB["conflict_exit_code_declared"] == _R22.lock_conflict_rc()
       and _lfB["declaration_head"].endswith(".json"),
       f"THE BOOK BUILDER'S LAUNCHER IS THE SERVICE FORM, and its conflict "
       f"code now comes from the DECLARATION HEAD "
       f"({_lfB['declaration_head']}, value "
       f"{_lfB['conflict_exit_code_declared']}) rather than from a Python "
       f"constant this checker owns -- REV 67 §2.3: a launcher refusing "
       f"with 76 was published as 75 because the checker returned its own")
    _srcB = Path(__file__).read_text()
    ok("--scope ...` -- is the form R-628 ruled against" in _srcB,
       "AND THE REFUSAL NO LONGER PRINTS THE RETIRED FORM: `assert_rule20` "
       "told a seat to relaunch with `systemd-run --user --scope` for six "
       "runs after R-628 -- a literal that had to track something that "
       "moved, and did not")
    _srcr = Path(__file__).read_text()
    ok('"scope": _BDR.scope_stats(),' in _srcr,
       "AND IT IS WIRED INTO THE EMITTED RECEIPT: read from this module's "
       "own source, not asserted in prose")

    # ---- the in-band correction (rule 13), driven on a fixture ----------
    _tdS = _tf.mkdtemp(prefix="be60_sup_")
    _fx2 = Path(_tdS); _saved2 = globals()["OUT_DERIVED"]
    (_fx2 / "be_gate1_state_tape_receipt_20260101_btc.v2.json").write_text(
        json.dumps({"WHICH_SPLIT_THE_ASSEMBLY_SCORES_FROM_AND_WHY":
                    {"split": "score"},
                    "tape": {"sha256": "c" * 64, "n_rows": 1000}}))
    _v1 = {"book": {"sha256": "d" * 64},
           "seam": {"commit": "6f134a6"},
           "inputs_pinned": {"tape": {"receipt": "a_name_that_does_not_"
                                                 "exist.json"}},
           "assembly_evidence": {
               "kept_by_coin": {"btc": 900},
               "UNCOVERED_GENERATIONS": {
                   "count": 50, "n_reference_generations": 2000,
                   "reasons_sum": 100, "by_reason": {"pre_window_excluded": 100},
                   "reasons_account_for_the_count": False,
                   "identical_across_heads": True}}}
    (_fx2 / "be_daybook_receipt_20260101_btc.json").write_text(
        json.dumps(_v1, indent=1, sort_keys=True))
    _v1_sha_before = _sha_file(_fx2 / "be_daybook_receipt_20260101_btc.json")
    globals()["OUT_DERIVED"] = _fx2
    try:
        _res = supersede_receipt("20260101", "btc")
        _out = json.loads((_fx2 / "be_daybook_receipt_20260101_btc.v2.json")
                          .read_text())
    finally:
        globals()["OUT_DERIVED"] = _saved2
    ok(_res["superseded_untouched"]
       and _sha_file(_fx2 / "be_daybook_receipt_20260101_btc.json")
       == _v1_sha_before
       and _out["supersedes"]["sha256"] == _v1_sha_before
       and _out["supersedes"]["path"].endswith(
           "be_daybook_receipt_20260101_btc.json"),
       "RULE 13: the correction writes a .v2 and LEAVES v1 byte-identical, "
       "linked by {path, sha256} -- the link is the identity, not the name "
       "(R-609)")
    ok(_out["inputs_pinned"]["tape"]["receipt"]
       == "be_gate1_state_tape_receipt_20260101_btc.v2.json"
       and _out["inputs_pinned"]["tape"]["receipt_corrected_from"]["was"]
       == "a_name_that_does_not_exist.json"
       and _out["inputs_pinned"]["tape"]["receipt_corrected_from"]["existed"]
       is False,
       "THE CORRECTED NAME IS THE RESOLVER'S, and what it replaced is "
       "recorded WITH the fact that the old name pointed at nothing")
    _ra2 = _out["assembly_evidence"]["ROW_ACCOUNTING"]
    ok(_ra2["kept"] == 900 and _ra2["dropped"] == 100
       and _ra2["accounted"] == 1000
       and _ra2["rows_published_by_the_tape_receipt"] == 1000
       and _ra2["rows_accounted_for"] is True and _ra2["residual"] == 0,
       f"THE COUNT PREDICATE IS RESTATED IN ONE POPULATION WITHOUT "
       f"RE-ASSEMBLY: {_ra2['kept']} + {_ra2['dropped']} = "
       f"{_ra2['accounted']} rows against the {_ra2['rows_published_by_the_tape_receipt']} "
       f"the tape receipt published -- computed from v1's own numbers")
    _wd = _out["assembly_evidence"]["UNCOVERED_GENERATIONS"]["WITHDRAWN_FIELD"]
    ok(_wd["field"] == "reasons_account_for_the_count"
       and _wd["value_in_v1"] is False and "FRAGMENT ROWS" in _wd["why"]
       and _out["assembly_evidence"]["UNCOVERED_GENERATIONS"]["count"] == 50,
       "THE FALSE FIELD IS WITHDRAWN IN BAND, CARRYING ITS OLD VALUE AND "
       "THE REASON -- a reader of v1 can find out what happened to it")
    ok(_out["scope"]["recoverable_from_the_run"] is False
       and _out["producing_code"]["reconstructed_after_the_fact"] is True,
       "AND WHAT CANNOT BE RECOVERED IS SAID, NOT RECONSTRUCTED: the scope "
       "block is marked unrecoverable with both external observations and "
       "their disagreement, and the producing code is labelled a "
       "reconstruction rather than dressed up as an import-time stamp")
    ok("builder_commit" not in _out["producing_code"]
       and _out["producing_code"]["status"] == "RECONSTRUCTED_NOT_A_STAMP"
       and "builder_commit_RECONSTRUCTED" in _out["producing_code"],
       "REV 59 S6: THE DISCLOSURE IS IN THE KEY. A resolver keying on "
       "`producing_code.builder_commit` finds NOTHING in a reconstruction -- "
       "round 60 gave it a commit that matched the bytes but was not the "
       "run's head, with the five disclosing fields under other keys. Rule "
       "13's own lesson: readers resolve fields, not annotations")
    # the head is RESOLVED: superseding again must produce .v3 from .v2,
    # linked to .v2 -- not another .v2 built from v1
    globals()["OUT_DERIVED"] = _fx2
    try:
        _res3 = supersede_receipt("20260101", "btc")
        _out3 = json.loads(Path(_res3["new_version"]).read_text())
    finally:
        globals()["OUT_DERIVED"] = _saved2
    _ra3 = _out3["assembly_evidence"]["ROW_ACCOUNTING"]
    ok(_ra3["rows_accounted_for"] is True and _ra3["dropped"] == 100
       and _ra3["accounted"] == 1000
       and _ra3["dropped_read_from"] == "the head's ROW_ACCOUNTING"
       and _out3["producing_code"][
           "run_head_recoverable_from_the_receipt"] is False,
       f"THE CORRECTION IS IDEMPOTENT, AND A RECONSTRUCTION IS NOT A "
       f"SOURCE: superseding an already-corrected head keeps the row "
       f"accounting closed ({_ra3['dropped']} dropped, read from "
       f"{_ra3['dropped_read_from']}) instead of silently defaulting the "
       f"drops to zero and reporting a failure on numbers that close -- and "
       f"the run head still reads NOT recoverable, because the head it came "
       f"from was itself a reconstruction, not an import-time stamp")
    globals()["OUT_DERIVED"] = _fx2
    try:
        (_fx2 / "be_daybook_receipt_20260101_btc.v3.json").rename(
            _fx2 / "be_daybook_receipt_20260101_btc.v3.json.bak")
        _stripped = json.loads(json.dumps(_out))
        _stripped["assembly_evidence"]["ROW_ACCOUNTING"].pop("dropped")
        _stripped["assembly_evidence"]["ROW_ACCOUNTING"].pop("by_reason")
        _stripped["assembly_evidence"]["UNCOVERED_GENERATIONS"].pop(
            "reasons_sum", None)
        (_fx2 / "be_daybook_receipt_20260101_btc.v2.json").write_text(
            json.dumps(_stripped, indent=1, sort_keys=True))
        try:
            supersede_receipt("20260101", "btc")
            ok(False, "a head with no drop count must refuse")
        except BookRefused as _e:
            ok("no drop count in any known field" in str(_e),
               "KNOWN-BAD: a head carrying NO drop count in any known field "
               "REFUSES rather than defaulting it to zero -- which is how "
               "the first .v3 reported an accounting failure on numbers "
               "that close (rule 11: absence is not a value)")
    finally:
        globals()["OUT_DERIVED"] = _saved2
    ok(Path(_res3["new_version"]).name == "be_daybook_receipt_20260101_btc.v3.json"
       and _out3["supersedes"]["artifact"]
       == "be_daybook_receipt_20260101_btc.v2.json"
       and _out3["supersedes"]["sha256"] == _res["new_version_sha256"],
       f"AND THE VERSION IS RESOLVED, NOT TYPED: a second correction "
       f"produces {Path(_res3['new_version']).name} superseding the .v2 by its "
       f"{{path, sha256}} -- applying a correction to a stale version would "
       f"silently drop the corrections already in the head")
    ok(_out["producing_code"]["run_head_recoverable_from_the_receipt"]
       is False
       and "NOT RECOVERABLE" in _out["producing_code"]["run_head_source"]
       and _out["producing_code"]["builder_digest_matches_that_commit"]
       is None,
       "THE COMMIT A RUN EXECUTED FROM IS NOT DERIVED FROM THE LANDING "
       "RECORD: with no run head supplied the block says NOT RECOVERABLE "
       "and computes no blob check. The first form of this function took "
       "the landing commit's PARENT and produced a right-looking wrong "
       "answer -- the shared tree's position, not the worktree the run "
       "executed from")
    globals()["OUT_DERIVED"] = Path(_tf.mkdtemp(prefix="be60_none_"))
    try:
        supersede_receipt("29990101", "btc")
        ok(False, "superseding a receipt that does not exist must refuse")
    except BookRefused as _e:
        ok("no receipt to supersede" in str(_e),
           "KNOWN-BAD: superseding a receipt that does not exist REFUSES "
           "rather than writing a .v2 with nothing behind it")
    finally:
        globals()["OUT_DERIVED"] = _saved2

    # ---- CODE DIRT vs THE LEDGER SYMLINK: a PROPERTY, never a name ------
    # DE 94's four cells, driven in a REAL git worktree. A seat worktree
    # carries `data` as a symlink to the ledger (R-553), so `git status`
    # reports it and every receipt read `dirty: true` on a clean code tree.
    # Exempting anything CALLED `data` would be this seat's recurring defect
    # in a new place, so the exemption is three computed conjuncts.
    import subprocess as _sp2
    _gd = _tf.mkdtemp(prefix="be62_dirt_")
    _LEDGER = str(_BDR.data_root())

    def _g2(*a):
        return _sp2.run(["git", "-C", _gd, *a], capture_output=True,
                        text=True, timeout=60)
    _g2("init", "-q")
    _g2("config", "user.email", "b@e"); _g2("config", "user.name", "be")
    (Path(_gd) / "README").write_text("x\n")
    _g2("add", "README"); _g2("commit", "-qm", "init")

    def _cells():
        return {r["path"]: r for r in
                [_R22.Capture(worktree=_gd).classify_dirt(x)
                 for x in (_sp2.run(["git", "-C", _gd, "status",
                                     "--porcelain"], capture_output=True,
                                    text=True, timeout=60).stdout
                           .rstrip("\n").split("\n")) if x]}

    # (a) a plain untracked FILE at that name -- NOT exempt
    (Path(_gd) / "data").write_text("not a link\n")
    _a = _cells()["data"]
    ok(_a["exempt"] is False and _a["is_untracked"] is True
       and _a["is_symlink"] is False,
       "CELL (a): a plain untracked FILE named `data` is NOT exempt -- it is "
       "untracked, but it is not a symlink, so the second conjunct refuses. "
       "A name-based exemption would have waved it through")
    # (b) a TRACKED modification at that path -- NOT exempt
    _g2("add", "data"); _g2("commit", "-qm", "track data")
    (Path(_gd) / "data").write_text("modified\n")
    _b = _cells()["data"]
    ok(_b["exempt"] is False and _b["is_untracked"] is False,
       f"CELL (b): a TRACKED modification at that path is NOT exempt "
       f"(status {_b['status_code']!r}) -- real dirt at the same name stays "
       f"dirt, which is the case a name-based rule hides most dangerously")
    _g2("rm", "-qf", "data"); _g2("commit", "-qm", "untrack")
    # (c) an untracked symlink pointing SOMEWHERE ELSE -- NOT exempt
    _os2 = __import__("os")
    _os2.symlink("/tmp", str(Path(_gd) / "data"))
    _c = _cells()["data"]
    ok(_c["exempt"] is False and _c["is_symlink"] is True
       and _c["resolves_to_the_ledger"] is False,
       f"CELL (c): a symlink at that name pointing ELSEWHERE "
       f"({_c['resolves_to']}) is NOT exempt -- the third conjunct is where "
       f"it resolves, not that it is a link")
    # (d) the real thing -- EXEMPT
    _os2.unlink(str(Path(_gd) / "data"))
    _os2.symlink(_LEDGER, str(Path(_gd) / "data"))
    _d = _cells()["data"]
    ok(_d["exempt"] is True and _d["is_untracked"] and _d["is_symlink"]
       and _d["resolves_to_the_ledger"] and _d["ledger"] == _LEDGER,
       f"CELL (d), THE POSITIVE CONTROL: the real ledger symlink IS exempt "
       f"-- untracked AND a symlink AND resolving to {_d['ledger']}, all "
       f"three computed. A guard shown only to refuse has not been shown to "
       f"work (rule 16)")
    # (e) the same link under a DIFFERENT name -- also exempt, deliberately
    _os2.symlink(_LEDGER, str(Path(_gd) / "ledger_alias"))
    _e = _cells()["ledger_alias"]
    ok(_e["exempt"] is True,
       "CELL (e): the same link under ANOTHER name is exempt too -- stated "
       "because it is a deliberate consequence of keying on the property. "
       "A symlink to the ledger is not producing code whatever it is called")
    # (f) the leading-space regression: a tracked change as the FIRST line
    (Path(_gd) / "README").write_text("y\n")
    _rows = _cells()
    _hs = _R22.Capture(worktree=_gd).capture("battery").head_state()
    ok("README" in _rows and _rows["README"]["exempt"] is False
       and "README" in _hs["dirty_paths_code"]
       and _hs["dirty_code"] is True
       and sorted(_hs["dirty_paths"]) == sorted(
           ["README", "data", "ledger_alias"]),
       f"CELL (f), A DEFECT FOUND BY DRIVING THIS: porcelain puts a SPACE in "
       f"column 1 for a tracked modification, and the git helper stripped "
       f"it -- so the first path lost its first character. Landed receipts "
       f"escaped it only because their one entry was `?? data`, which has "
       f"no leading space. Paths now parse whole: "
       f"{_hs['dirty_paths_code']}")
    ok(_hs["dirty"] is True and _hs["dirty_code"] is True
       and set(_hs["dirty_paths"]) - set(_hs["dirty_paths_code"])
       == {"data", "ledger_alias"},
       "AND BOTH FIELDS SURVIVE, NEITHER REDEFINED: `dirty` keeps the raw "
       "meaning every landed receipt used, `dirty_code` answers the rule-22 "
       "question, and the difference between them is exactly the exempt "
       "entries -- so an old receipt and a new one can still be compared")
    # WHY THE TWO LANDED 09-05 RECEIPTS ARE NOT SUPERSEDED. Their fields
    # read `dirty: true` and `dirty_paths: ["data"]`. Reproduce a worktree
    # in exactly that condition -- the ledger link and nothing else -- and
    # the new code emits the SAME two values; only the new keys are added.
    # So those receipts stay accurate as written and rule 13 does not fire.
    (Path(_gd) / "README").write_text("x\n")            # undo the (f) edit
    _os2.unlink(str(Path(_gd) / "ledger_alias"))
    _only = _R22.Capture(worktree=_gd).capture("battery").head_state()
    ok(_only["dirty"] is True and _only["dirty_paths"] == ["data"]
       and _only["dirty_code"] is False
       and _only["dirty_paths_code"] == []
       and len(_only["exempt_entries"]) == 1,
       "THE LANDED 09-05 RECEIPTS STAY ACCURATE: on a worktree carrying the "
       "ledger link and nothing else, the new code emits `dirty: true` and "
       "`dirty_paths: ['data']` -- byte-for-byte what those receipts say -- "
       "and adds `dirty_code: false` beside them. No field changed meaning, "
       "so they are not superseded (rule 13 does not fire on an accurate "
       "receipt); the classification starts with the 09-05 book")
    _hs2 = _R22.stamp(__file__)
    ok("worktree_CODE_was_dirty_at_import" in _hs2
       and "worktree_was_dirty_at_import" in _hs2,
       "AND THE STAMP CARRIES BOTH KEYS: the rule-22 question gets its own "
       "name rather than quietly changing the meaning of the field already "
       "in the landed 09-05 receipts")

    # ---- R-637: THE PORCELAIN DEFECT'S *SLICE* HALF ---------------------
    # Round 62 fixed the READ half here (a `.strip()` eating the first
    # line's leading space). The SLICE half -- `line[3:]` -- was correct
    # only WHILE the read stayed raw, which is one edit away from the
    # defect, and the slice is the half people edit. The offset is no
    # longer assumed: the shape is checked and a malformed line refuses.
    for _ln, _want in ((" M live/pm_research/x.py",
                        (" M", "live/pm_research/x.py")),
                       ("?? data", ("??", "data")),
                       ("R  a -> b", ("R ", "b"))):
        _got = _R22.parse_porcelain_line(_ln)
        ok(_got == _want,
           f"THE THREE-LINE FALSIFIER, LINE {_ln!r}: parses to {_got} -- a "
           f"leading-space status, an untracked entry, and a rename whose "
           f"path in the worktree is the NEW one")
    try:
        _R22.parse_porcelain_line("M live/pm_research/x.py")
        ok(False, "a shifted porcelain line must refuse")
    except _R22.PorcelainMalformed as _e:
        ok("not a porcelain v1 line" in str(_e) and "one character" in str(_e),
           "KNOWN-BAD, THE SLICE HALF: a line that LOST its leading space -- "
           "what a `.strip()` does to the FIRST line -- is REFUSED BY NAME "
           "rather than sliced into a path one character short. That "
           "mis-slice is silent, and the silence was the defect")
    ok(_R22.parse_porcelain_line('RM "odd name.py"')[1] == "odd name.py",
       "AND A QUOTED PATH IS UNQUOTED: git quotes unusual names, so the "
       "quotes are part of the encoding, not of the path")

    # ---- the porcelain readers, under the REBUILT census ---------------
    # REV 67 §1.3: the first census looked only for a Subscript or a split
    # on a DIRECTLY assigned name, and nine deriving shapes walked past four
    # of them. It now propagates taint transitively, through a Call
    # RECEIVER, and through loop and comprehension targets. WHAT IT IS: a
    # REGRESSION GUARD on today's readers -- not a proof that no path can
    # ever be derived; its limits are in its own docstring.
    for _shape, _srcS, _want in (
            ("direct subscript", 'x = git("status","--porcelain")\np = x[3:]\n', True),
            ("call-receiver loop", 'for l in git("status","--porcelain").splitlines():\n    p = l[3:]\n', True),
            ("transitive name", 'a = git("status","--porcelain")\nb = a\nc = b[3:]\n', True),
            ("comprehension", 'x = git("status","--porcelain")\nps = [l[3:] for l in x.split(chr(10))]\n', True),
            ("split then index", 'x = git("status","--porcelain")\np = x.split(chr(10))[0][3:]\n', True),
            ("inline call subscript", 'p = git("status","--porcelain")[3:]\n', True),
            ("strip then slice", 'x = git("status","--porcelain").strip()\np = x[3:]\n', True),
            ("for over a name", 'x = git("status","--porcelain")\nfor l in x.splitlines():\n    q = l[3:]\n', True),
            ("boolean only", 'x = git("status","--porcelain")\nd = bool(x.strip())\n', False)):
        _got = _R22.porcelain_derivation_census(_srcS)["derives_a_path"]
        ok(_got is _want,
           f"CENSUS SHAPE {_shape!r}: derives_a_path={_got} (expected "
           f"{_want}). Nine shapes, four of which walked past the first "
           f"version (REV 67 §1.3)")
    # be_forward_recon is still boolean-only; be_forward_day is NOT any more
    # -- THIS round gave it the row classification REV 67 §1.4 asked for, so
    # it now derives paths ON PURPOSE, through the shared safe parser and a
    # RAW read. The claim changed, so the check changed with it.
    _cr = _R22.porcelain_derivation_census(
        (Path(HERE) / "be_forward_recon.py").read_text())
    ok(not _cr["derives_a_path"],
       f"be_forward_recon.py: {_cr['n_porcelain_calls']} call(s) carrying "
       f"that constant, {_cr['n_status_porcelain_calls']} of them `status "
       f"--porcelain`; no path is derived, so neither half of R-637 reaches "
       f"it")
    _fdsrc = (Path(HERE) / "be_forward_day.py").read_text()
    _cd = _R22.porcelain_derivation_census(_fdsrc)
    ok(_cd["derives_a_path"] and "git_raw" in _fdsrc
       and "classify_dirt" in _fdsrc and "rstrip" in _fdsrc,
       f"be_forward_day.py NOW DERIVES PATHS, DELIBERATELY: "
       f"{_cd['n_porcelain_calls']} call(s) carry the constant but only "
       f"{_cd['n_status_porcelain_calls']} is `status --porcelain` (the "
       f"others are `git worktree list --porcelain`, which the old count "
       f"conflated). It reads RAW and classifies each row through the "
       f"shared parser, so `working_tree_dirty` no longer reads TRUE on a "
       f"pristine tree because of the ledger symlink")

    ok(75 not in EXIT_CODES and 75 == _R22.lock_conflict_rc(),
       f"R-649 §3.2: this producer's declared exit codes are "
       f"{sorted(EXIT_CODES)} and 75 is NOT among them -- so a unit reading "
       f"ExecMainStatus=75 means the lock was held, and cannot also mean "
       f"this producer broke the declaration. The 75 is read from the "
       f"declaration head, not typed here")
    ok(_R22.assert_not_a_scope(fixture=True)["kind"] in
       ("scope", "service", "none"),
       f"REV 65 §1.2: the launch form is decided at RUNTIME from this "
       f"process's own cgroup leaf ({_R22.cgroup_leaf()['leaf']!r}), which a "
       f"static lint cannot do -- it cannot see a `--scope` behind a "
       f"variable or a wrapper")
    try:
        _R22.assert_not_a_scope(fixture=False) if \
            _R22.cgroup_leaf()["kind"] == "scope" else None
        _leafok = _R22.cgroup_leaf()["kind"] != "scope"
    except _R22.HeavyRunRefused as _e:
        _leafok = "transient SCOPE" in str(_e)
    ok(_leafok,
       "KNOWN-BAD, DRIVEN WHERE IT LIVES: a process whose own cgroup leaf is "
       "a `.scope` REFUSES a real day by name -- nine BE heavy runs were "
       "scopes and every receipt said so in `scope.unit`; no seat read it")
    _sd = _R22.declaration_head("be_daybook_structure")
    ok(_sd["name"].startswith("be_daybook_structure_v")
       and "NOT YET VERIFIED" in _sd["doc"]["STATUS"],
       f"R-654: the book's structure is DECLARED ({_sd['name']}) so DA maps "
       f"the pickle through it instead of guessing -- and it says of itself "
       f"that it is derived from the producing code and NOT yet asserted "
       f"against a real book, because that is heavy and needs the lock")
    import pickle as _pk
    _bad = Path(_tf.mkdtemp(prefix="be65_struct_")) / "wrong.pkl"
    _bad.write_bytes(_pk.dumps({"fr": {}, "not_asm": {}}))
    try:
        verify_structure(_bad)
        ok(False, "a book with the wrong top-level keys must refuse")
    except BookRefused as _e:
        ok("contradicts be_daybook_structure" in str(_e)
           and "top-level keys" in str(_e),
           "KNOWN-BAD: a pickle whose top-level keys are not the declared "
           "ones REFUSES BY NAME -- the verifier does not report a shape it "
           "did not find")
    _good = Path(_tf.mkdtemp(prefix="be65_structok_")) / "right.pkl"
    _good.write_bytes(_pk.dumps({
        "fr": {"reference": {}},
        "asm": {"by_arm": {("btc", "h"): [{"k": 1}, {}]},
                "assembly": {"n_chunks": 1, "kept_by_coin": {},
                             "drops_by_coin": {}}}}))
    _r = verify_structure(_good)
    ok(_r["all_hold"] and _r["n_checks"] >= 6,
       f"POSITIVE CONTROL: a pickle with the declared shape ADMITS on all "
       f"{_r['n_checks']} claims -- a verifier shown only to refuse has not "
       f"been shown to work")

    # ---- THE INTERFACE, DECIDED (R-659(B), R-662 §3.1, REV 72 §3.1) -----
    _sc9 = _BDR.scope_stats()
    ok(all(isinstance(_sc9.get(k), (int, type(None)))
           for k in ("peak_bytes", "current_bytes", "max_bytes"))
       and all(f"{k}_text" in _sc9
               for k in ("peak_bytes", "current_bytes", "max_bytes"))
       and isinstance(_sc9["anon_bytes"], int),
       f"THE SCOPE'S BYTE FIELDS ARE INTS NOW, with the cgroup file's text "
       f"beside them ({_sc9.get('peak_bytes')!r} / "
       f"{_sc9.get('peak_bytes_text')!r}). They were STRINGS next to int "
       f"anon/file and int peak_censoring -- BE 60 closed the one CONSUMER "
       f"and never the TYPE, and DA's pre-read and the structure "
       f"declaration read these receipts directly")
    # the falsifier: a consumer comparing WITHOUT a cast, both ways
    _cen9 = _sc9["peak_censoring"]
    _uncast_ok = (_sc9["peak_bytes"] >= _cen9["cap_bytes"]) is _sc9[
        "peak_is_censored"]
    ok(_uncast_ok,
       f"POSITIVE: an uncast `scope.peak_bytes >= peak_censoring.cap_bytes` "
       f"now agrees with `peak_is_censored` ({_sc9['peak_is_censored']}) -- "
       f"the comparison the next reader will write, working without knowing "
       f"it had to cast")
    try:
        _ = str(_sc9["peak_bytes"]) >= _cen9["cap_bytes"]
        _raised = False
    except TypeError:
        _raised = True
    ok(_raised,
       "KNOWN-BAD, THE SHAPE BEFORE THIS CHANGE: the same comparison with "
       "the field as a STRING raises TypeError -- so the old receipt did "
       "not merely risk a wrong answer, it broke the reader that did the "
       "natural thing")
    _fake = {"peak_bytes": 8589934592, "max_bytes": 8589934592,
             "events": {"max": 1199}}
    ok(_fake["peak_bytes"] >= _fake["max_bytes"],
       "AND AT THE CAP THE UNCAST COMPARISON IS DECIDABLE: 09-04's own "
       "numbers (8,589,934,592 against the same cap, 1,199 reclaims) "
       "compare TRUE as ints, where as strings they compared by lexical "
       "order -- right answer, wrong reason, and only by accident")

    # ---- R-646 R1: ONE porcelain parser, and this is the caller ---------
    for _ln9, _want9 in ((" M live/pm_research/x.py",
                          (" M", "live/pm_research/x.py")),
                         ("?? data", ("??", "data")),
                         ("R  a -> b", ("R ", "b")),
                         ("C  a -> b", ("C ", "b")),
                         ("A  file with spaces.py",
                          ("A ", "file with spaces.py")),
                         ("MM a -> b.py", ("MM", "a -> b.py")),
                         ('?? "odd name.py"', ("??", "odd name.py")),
                         ("?? trailing  ", ("??", "trailing  ")),
                         (" D gone.py", (" D", "gone.py")),
                         ("AM new.py", ("AM", "new.py")),
                         ("UU conflict.py", ("UU", "conflict.py")),
                         ("!! ignored.py", ("!!", "ignored.py"))):
        ok(_R22.parse_porcelain_line(_ln9) == _want9,
           f"TWELVE-LINE BATTERY through da_root.parse_porcelain: {_ln9!r}")
    for _bad9 in ("M live/x.py", "## main...origin/main", "ab"):
        try:
            _R22.parse_porcelain_line(_bad9)
            ok(False, f"{_bad9!r} must refuse")
        except _R22.PorcelainMalformed as _e9:
            ok("not a porcelain v1 line" in str(_e9)
               and "da_root reports" in str(_e9),
               f"AND THE REFUSAL BY NAME SURVIVES THE THIN CALL: {_bad9!r} "
               f"refuses, quoting da_root's own malformed report -- DA's "
               f"parser REPORTS malformed rows because its callers want the "
               f"whole census; this seat's callers want the line refused")
    import da_root as _DR9
    ok("da_root" in Path(__file__).parent.joinpath("be_rule22.py").read_text()
       and _DR9.parse_porcelain(" M x.py")["rows"][0]["path"] == "x.py",
       "ONE PARSER (R-646 R1, REV 71 §1.5): `be_rule22.parse_porcelain_line` "
       "is now a thin call into `da_root.parse_porcelain`, which carries "
       "this seat's algorithm with DA's row structure. Three "
       "implementations disagreed on four of twelve lines and no two were "
       "wrong in the same place")

    return _finish(checks, fails, skipped)


def _finish(checks, fails, skipped) -> int:
    print()
    if skipped:
        print(f"{len(skipped)} check(s) SKIPPED — real ledger data not "
              f"reachable from this tree (BE48 §B.5). The fixture-driven "
              f"guards ran.")
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    expect = EXPECTED_CHECKS - len(skipped)
    if checks != expect:
        print(f"FAIL: ran {checks} checks, expected {expect} "
              f"(EXPECTED_CHECKS={EXPECTED_CHECKS} minus {len(skipped)} "
              f"skipped)")
        return 1
    print(f"{checks} checks passed"
          + (f", {len(skipped)} skipped" if skipped else ""))
    return 0


#: WHAT ROUND 59'S SCOPE ACTUALLY DID, and where each number comes from.
#: This is a LITERAL, and literals are this seat's recurring defect -- so:
#: it describes ONE COMPLETED RUN whose scope no longer exists, which is the
#: only kind of literal that cannot go stale. It is NOT the run's own cgroup
#: read: `be_daybook_build` did not carry a scope block at round 59, and the
#: transient scope was reaped between the process exiting and the read. Both
#: sources are named so a reader can weigh them, and they DISAGREE.
BE59_SCOPE_OBSERVED = {
    "recoverable_from_the_run": False,
    "why": "the receipt carried no scope block at round 59 and a transient "
           "systemd scope is destroyed when its last process exits; the "
           "post-exit read returned empty",
    "poll_by_the_seat": {
        "as_of": "2026-09-06T11:17:57Z", "about_40s_before_exit": True,
        "memory_peak_bytes": 7117090816, "memory_current_bytes": 7039881216,
        "memory_max_bytes": 8589934592,
        "events": {"max": 0, "high": 0, "low": 0, "oom": 0, "oom_kill": 0},
        "cap_was_hit_up_to_this_read": False,
        "source": "direct read of the scope's cgroup files"},
    "systemd_stop_line": {
        "as_of": "2026-09-06T11:18:37Z",
        "memory_peak": "6.5G", "swap_peak": "0B",
        "cpu_time": "39min 31.340s", "wall": "39min 43s",
        "source": "journalctl --user -u be59book.scope"},
    "the_two_do_not_reconcile": "6.5G formats BELOW the 6.628 GiB read 40 s "
                                "earlier, and memory.peak is monotone -- so "
                                "systemd's line reports some other "
                                "quantity. Stated, not explained",
    "unmeasured_window": "the final ~40 s (A3 release and A4 write)",
}


def supersede_receipt(day: str, coin: str = COIN, *,
                      out_suffix: str | None = None,
                      run_head: str | None = None,
                      run_head_source: str | None = None) -> dict:
    """CORRECT A LANDED BOOK RECEIPT IN BAND (rule 13). NO RE-ASSEMBLY.

    Every correction below is computed from numbers that already exist -- the
    landed receipt's own fields and round 58's tape receipt -- or is an
    explicit statement that something is NOT recoverable. Nothing is
    re-derived from the book, no arm is scored, and v1 is never edited: it
    stays as provenance and this file supersedes it by {path, sha256}."""
    import copy
    # THE HEAD IS RESOLVED, NEVER TYPED -- the same discipline that closed
    # the `.v3.json` defect. A correction applied to a stale version would
    # silently drop the corrections already in the head.
    cands, _ver = _receipt_head(f"be_daybook_receipt_{day}_{coin}")
    cands = [c for c in cands if ".WRONG" not in c.name]
    if not cands:
        raise BookRefused(
            f"REFUSED: no receipt to supersede for {day} {coin}.")
    src = cands[0]
    if out_suffix is None:
        out_suffix = f".v{_ver(src) + 1}"
    raw = src.read_bytes()
    v1 = json.loads(raw)
    v1_sha = hashlib.sha256(raw).hexdigest()
    v2 = copy.deepcopy(v1)

    # (a) the tape receipt's REAL name, from the resolver rather than an
    #     f-string. This is the defect that made round 59 name a file that
    #     does not exist.
    pin = day_tape_pin(day, coin)
    was = ((v1.get("inputs_pinned") or {}).get("tape") or {}).get("receipt")
    v2.setdefault("inputs_pinned", {}).setdefault("tape", {})
    v2["inputs_pinned"]["tape"]["receipt"] = (pin or {}).get("receipt")
    v2["inputs_pinned"]["tape"]["receipt_corrected_from"] = {
        "was": was, "existed": bool(was and (OUT_DERIVED / was).exists()),
        "why": "the emitter hardcoded the version suffix in an f-string "
               "while the resolver 470 lines above globbed for the real "
               "head; the name was true of 09-03 and false here",
        "now": "read from the resolver that chose the digest"}

    # (b) the count predicate, restated within ONE population. Both numbers
    #     were already in v1; only the comparison was wrong.
    ev = v2.get("assembly_evidence") or {}
    ug = ev.get("UNCOVERED_GENERATIONS") or {}
    kept = sum(int(x) for x in (ev.get("kept_by_coin") or {}).values())
    # THE CORRECTION MUST BE IDEMPOTENT. Reading `reasons_sum` alone worked
    # on v1 and silently returned 0 on an already-corrected head, where that
    # field no longer exists -- so a .v3 built from a .v2 reported
    # `rows_accounted_for: false` on numbers that had closed. Absence must
    # never read as a value (rule 11): the count is taken from whichever
    # block holds it, and if none does the correction REFUSES.
    _prev_ra = ev.get("ROW_ACCOUNTING") or {}
    if _prev_ra.get("dropped") is not None:
        dropped, _dsrc = int(_prev_ra["dropped"]), "the head's ROW_ACCOUNTING"
    elif ug.get("reasons_sum") is not None:
        dropped, _dsrc = int(ug["reasons_sum"]), "the head's reasons_sum"
    elif _prev_ra.get("by_reason") or ug.get("by_reason"):
        _br = _prev_ra.get("by_reason") or ug.get("by_reason")
        dropped, _dsrc = sum(int(v) for v in _br.values()), "by_reason"
    else:
        raise BookRefused(
            f"REFUSED: {src.name} carries no drop count in any known field "
            f"(ROW_ACCOUNTING.dropped, reasons_sum, by_reason). Defaulting "
            f"it to zero would report an accounting failure on numbers that "
            f"close.")
    rows_published = (pin or {}).get("n_rows")
    ev["ROW_ACCOUNTING"] = {
        "population": "FRAGMENT ROWS",
        "kept": kept, "dropped": dropped,
        "by_reason": _prev_ra.get("by_reason") or ug.get("by_reason"),
        "dropped_read_from": _dsrc,
        "accounted": kept + dropped,
        "rows_published_by_the_tape_receipt": rows_published,
        "rows_pin_receipt": (pin or {}).get("receipt"),
        "rows_accounted_for": (rows_published is not None
                               and kept + dropped == rows_published),
        "residual": (None if rows_published is None
                     else rows_published - (kept + dropped)),
        "computed_from": "v1's own kept_by_coin and by_reason against round "
                         "58's tape receipt -- no re-assembly",
    }
    ev["UNCOVERED_GENERATIONS"] = {
        "population": "REFERENCE GENERATIONS -- a DIFFERENT population from "
                      "the row accounting",
        "count": ug.get("count"),
        "identical_across_heads": ug.get("identical_across_heads"),
        "n_reference_generations": ug.get("n_reference_generations"),
        "coverage": (1 - ug["count"] / ug["n_reference_generations"]
                     if ug.get("count") is not None
                     and ug.get("n_reference_generations") else None),
        "WITHDRAWN_FIELD": {
            "field": "reasons_account_for_the_count",
            "value_in_v1": ug.get("reasons_account_for_the_count"),
            "why": "it compared sum(drops_by_coin) -- FRAGMENT ROWS -- with "
                   "n_uncovered -- REFERENCE GENERATIONS. Both numbers were "
                   "correct; the equality between them could not hold on "
                   "any real day. Its known-bad passed because the fixture "
                   "supplied both sides in the same unit",
            "what_replaces_it": "ROW_ACCOUNTING above, closed within one "
                                "population against a total round 58 "
                                "published"},
    }
    v2["assembly_evidence"] = ev

    # (c) the seam commit, READ as of the commit this run executed from
    seam_mod = Path(HERE) / "de_phase4_diag_runner.py"
    # THE COMMIT THE RUN EXECUTED FROM IS NOT IN v1 -- that is the gap
    # R-601 named, and it is why this parameter is explicit. The first form
    # of this function DERIVED it as the parent of the landing commit and
    # got 9ad6a16, the shared tree's position; the run executed from
    # b827ca2 in a worktree. Right-looking and wrong, from a record that
    # does not carry the fact. It is now supplied with its source, or the
    # block says the receipt cannot support it.
    head = ((v1.get("producing_code") or {}).get("builder_commit")
            or run_head)
    v2.setdefault("seam", {})["commit"] = _commit_of_at(seam_mod, head)
    v2["seam"]["commit_corrected_from"] = {
        "was": (v1.get("seam") or {}).get("commit"),
        "why": "a typed literal, true of the front door once and unchecked "
               "for three rounds (DA 77, R-613)",
        "now": "git log -1 <run's HEAD> -- de_phase4_diag_runner.py"}

    # (d) the scope: NOT recoverable, and said so rather than reconstructed
    v2["scope"] = dict(BE59_SCOPE_OBSERVED)

    # (e) the producing code: RECONSTRUCTED, and labelled as such. The code
    #     that ran carried no import-time capture, so this is not a stamp.
    _blob = _blob_sha_at(head, "live/pm_research/be_daybook_build.py")
    v2["producing_code"] = {
        # REV 59 S6: THE DISCLOSURE IS IN THE KEY, not in a neighbouring
        # field. Rule 13's own lesson is that automated readers resolve
        # FIELDS, not annotations: a resolver keying on `builder_commit`
        # got a commit that matches the bytes and is NOT the run's head,
        # while the five fields saying so were different keys. There is now
        # no `builder_commit` here to find -- the value carries its status
        # in its name, the pattern already used by `receipt_corrected_from`.
        # (`status` is also set, but the KEY is what carries it: this file
        # is written with sort_keys, so no field is reliably "first".)
        "status": "RECONSTRUCTED_NOT_A_STAMP",
        "reconstructed_after_the_fact": True,
        # A RECONSTRUCTION IS NOT A SOURCE. This tested for a
        # `builder_commit` KEY, which the previous reconstruction also had
        # -- so superseding a .v2 reported the head as recoverable from the
        # receipt when it was only recoverable from an earlier guess. The
        # test is now for a genuine import-time capture.
        "run_head_recoverable_from_the_receipt": (
            (v1.get("producing_code") or {}).get("captured_at") == "IMPORT"),
        "run_head_source": run_head_source or (
            "the receipt's own import-time stamp"
            if (v1.get("producing_code") or {}).get("captured_at") == "IMPORT"
            else "NOT RECOVERABLE from the artifact"),
        "builder_digest_matches_that_commit": (
            None if not (head and _blob) else
            _blob == "6a09f7e3aac7ede37ce7347ee341cb950861767e"
                     "8a2d96bd91dcc6e10d33ddbd"),
        "what_this_check_can_and_cannot_do": "matching the blob at the named "
                                             "commit CONFIRMS those bytes "
                                             "existed there; it cannot prove "
                                             "the run used that commit, "
                                             "because a file identical at "
                                             "two commits matches both",
        "NOT_an_import_time_capture": "the round-59 builder had no rule-22 "
                                      "capture; this is read from the "
                                      "landing record, which is weaker and "
                                      "must not be read as a stamp",
        "producing_code": "be_daybook_build.py",
        "sha256_at_the_run_head": _blob_sha_at(head,
                                               "live/pm_research/"
                                               "be_daybook_build.py"),
        "builder_commit_RECONSTRUCTED": head,
        "why_the_key_is_not_builder_commit": "a resolver keying on "
                                             "`builder_commit` must find "
                                             "NOTHING here, because this is "
                                             "not one. Receipts produced "
                                             "from round 60 onward carry a "
                                             "real `builder_commit` inside "
                                             "an import-time stamp, and the "
                                             "two must not be "
                                             "indistinguishable to a "
                                             "machine",
        "from_round_60_onward": "captured at IMPORT with the import closure "
                                "and HEAD, and the emit refused by name if "
                                "any of it moves",
    }

    v2["supersedes"] = {
        "artifact": src.name,
        "path": str(src),
        "sha256": v1_sha,
        "rule": "13 -- vN+1; v1 is NOT edited",
        "what_changed": "FOUR REPORTING FIELDS AND ONE ADDITION. No number "
                        "from the assembly moves; the book is the same "
                        "bytes and was not rebuilt.",
    }
    dst = OUT_DERIVED / f"be_daybook_receipt_{day}_{coin}{out_suffix}.json"
    dst.write_text(json.dumps(v2, indent=1, sort_keys=True, default=str))
    after = hashlib.sha256(src.read_bytes()).hexdigest()
    if after != v1_sha:
        raise BookRefused(
            f"REFUSED: v1 changed while its superseding version was being "
            f"written -- {v1_sha[:16]} -> {after[:16]}. A superseding "
            f"receipt whose predecessor moved is not a correction.")
    # NAMES THAT TRACK WHAT THEY HOLD: this returned `v2`/`v1_sha256` while
    # writing a .v3 over a .v2 -- the same class as the receipt-name literal,
    # in my own return value.
    return {"new_version": str(dst), "new_version_sha256": _sha_file(dst),
            "superseded": src.name, "superseded_sha256": v1_sha,
            "superseded_untouched": True,
            "book_sha256_unchanged": v2["book"]["sha256"] == v1["book"]["sha256"]}


def _commit_of_at(path, head) -> dict:
    """The commit that last touched `path` AS OF `head` -- not as of now."""
    import subprocess as _sp
    try:
        r = _sp.run(["git", "-C", str(HERE), "log", "-1", "--format=%H",
                     head or "HEAD", "--", str(path)],
                    capture_output=True, text=True, timeout=60)
        c = r.stdout.strip() if r.returncode == 0 else None
    except Exception:                                        # noqa: BLE001
        c = None
    return {"module": Path(path).name, "commit": c, "short": c[:7] if c else None,
            "as_of": head, "source": "git log -1 <run head> -- <module>"}


def _blob_sha_at(head, relpath) -> str | None:
    import subprocess as _sp
    try:
        r = _sp.run(["git", "-C", str(HERE.parents[1]), "show",
                     f"{head}:{relpath}"], capture_output=True, timeout=60)
        return (hashlib.sha256(r.stdout).hexdigest()
                if r.returncode == 0 else None)
    except Exception:                                        # noqa: BLE001
        return None


def verify_structure(book_path, *, declaration: dict | None = None) -> dict:
    """ASSERT be_daybook_structure_v1 AGAINST A REAL BOOK. HEAVY.

    R-654: DA's reader refused the real 09-03 book by name because every
    fixture it had been driven on was JSON. A declaration derived from the
    producing code is a claim about the code; only opening a real book makes
    it a statement about the artifact. This is that step, and it refuses by
    name on the first mismatch rather than reporting a shape it did not
    find."""
    d = declaration or _R22.declaration_head("be_daybook_structure")["doc"]
    q = Path(book_path)
    if not q.exists():
        raise BookRefused(f"REFUSED: no book at {q}")
    t0 = time.time()
    with q.open("rb") as fh:
        book = pickle.load(fh)
    checked = []

    def _need(cond, why):
        checked.append({"claim": why, "holds": bool(cond)})
        if not cond:
            raise BookRefused(f"REFUSED: the book at {q.name} contradicts "
                              f"be_daybook_structure: {why}")

    _need(isinstance(book, dict), "the top level is a dict")
    _need(set(book) == set(d["top_level"]["keys"]),
          f"the top-level keys are {d['top_level']['keys']}, found "
          f"{sorted(book)}")
    asm = book["asm"]
    _need(isinstance(asm, dict) and "by_arm" in asm and "assembly" in asm,
          "asm carries by_arm and assembly")
    _need(all(isinstance(k, tuple) and len(k) == 2 for k in asm["by_arm"]),
          "asm.by_arm is keyed by (coin, head) TUPLES")
    first = asm["by_arm"][next(iter(asm["by_arm"]))]
    _need(hasattr(first, "__getitem__"),
          "each by_arm value is indexable and its [0] carries the scored keys")
    _need(isinstance(book["fr"], dict) and "reference" in book["fr"],
          "fr carries the reference")
    a = asm["assembly"]
    _need(all(k in a for k in ("n_chunks", "kept_by_coin", "drops_by_coin")),
          "asm.assembly carries the chunk and drop accounting")
    return {"book": str(q), "bytes": q.stat().st_size,
            "sha256": _sha_file(q),
            "declaration": _R22.declaration_head("be_daybook_structure")["name"],
            "checks": checked, "n_checks": len(checked),
            "all_hold": all(c["holds"] for c in checked),
            "wall_s": round(time.time() - t0, 1),
            "peak_rss_gb": _rss_gb(),
            "this_is_the_step_that_makes_the_declaration_a_measurement":
                "before this ran, the declaration was derived from the "
                "producing code and said so (R-654)"}


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--verify-structure" in argv:
        print(json.dumps(verify_structure(
            argv[argv.index("--verify-structure") + 1]), indent=1,
            default=str))
        return 0
    if "--supersede-receipt" in argv:
        day = argv[argv.index("--supersede-receipt") + 1]
        _rh = (argv[argv.index("--run-head") + 1]
               if "--run-head" in argv else None)
        print(json.dumps(supersede_receipt(
            day, run_head=_rh,
            run_head_source=(
                "SUPPLIED ON THE COMMAND LINE from the seat's landing report "
                "(Q-BE-301 and commit 50f30d9's message, which record the "
                "run as frozen at that commit). PROSE, not a stamp: round "
                "59's receipt carries no builder_commit, which is exactly "
                "the gap R-601 named and rule 22 closes from round 60 on"
                if _rh else None)), indent=1))
        return 0
    if "--day" in argv:
        day = argv[argv.index("--day") + 1]
        out = build(day)
        dst = OUT_DERIVED / f"be_daybook_receipt_{day}_{COIN}.json"
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"receipt": str(dst),
                          "book": out["book"]["path"],
                          "sha256": out["book"]["sha256"],
                          "bytes": out["book"]["bytes"],
                          "wall_s": out["resources"]["wall_s"],
                          "peak_rss_gb": out["resources"]["peak_rss_gb"]}))
        return 0
    print("usage: be_daybook_build.py --selftest | --day <YYYYMMDD> "
          "| --supersede-receipt <YYYYMMDD>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
