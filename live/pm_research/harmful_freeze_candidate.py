"""Build the frozen reduced-fine candidate. RUNS ONLY ON THE USER'S YES.

AUTHORISATION (R-126, in-file): R-166(2) instructs BE to draft the freeze
receipt as an ASK. The FREEZE DECISION IS THE USER'S (R-162(3)); this module
exists so the ASK describes something concrete and so the freeze act is one
reviewable command rather than an improvisation after the yes.

WHY A SEPARATE MODULE. `harmful_hazard_model.py` just passed a cent-exact
reproduction gate. Adding a freeze mode to it would change the file whose
hash the manifest pins and whose output was verified to the cent. So the
freeze builder imports the gated pipeline and does not edit it.

WHAT THE REFIT IS, AND WHAT IT IS NOT (R-166(3)):
  IS  : one fit over the CONSUMED FRAGMENT ONLY -- both days, through slug
        1787650200 -- producing the coefficient vector that would be deployed.
        `run --fine` fits on days[:-1] (08-24) and scores 08-25; a deployed
        candidate should use every consumed row it is allowed to use.
  NOT : any use of the R-145(3) top-up. That tape is Phase 2's untouched
        development-test surface. Training on it here would consume it and
        change the incumbent mid-race.

NO IN-SAMPLE VERDICT IS STORED. The refit's own fit-quality numbers are not
evidence of anything -- they are in-sample by construction. The evidence for
this candidate is the ALREADY-COMPLETED paired comparison; the freeze does not
re-argue it. Rule 14: this estimates, it does not decide.
"""
from __future__ import annotations

import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"
LAST_CONSUMED_SLUG_T0 = 1787650200          # R-166(3) boundary, inclusive
OUT = DERIVED / "harmful_reduced_fine_candidate_v1.json"


class TopUpLeak(RuntimeError):
    """A row outside the consumed fragment reached the freeze refit."""


def assert_consumed_fragment_only(slug_t0s) -> None:
    """REFUSE if any row post-dates the consumed fragment.

    The top-up is Phase 2's development-test surface. If it leaked into the
    freeze refit, the incumbent would have been trained on the tape it is about
    to be tested against, and nobody would see it in the numbers -- the fit
    would simply look slightly better. So this refuses loudly instead."""
    late = sorted({t for t in slug_t0s if t > LAST_CONSUMED_SLUG_T0})
    if late:
        raise TopUpLeak(
            f"{len(late)} slug(s) after the consumed fragment reached the "
            f"freeze refit (first {late[0]}, last {late[-1]}; boundary "
            f"{LAST_CONSUMED_SLUG_T0}). R-166(3) excludes the top-up from the "
            f"freeze: training on Phase 2's test surface would consume it and "
            f"move the incumbent mid-race.")


def atomic_write_json(path: Path, obj: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True); raise


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    assert_consumed_fragment_only([1787579400, LAST_CONSUMED_SLUG_T0])
    ok(True, "KNOWN-GOOD: the consumed fragment's own slugs pass, boundary "
             "inclusive")
    try:
        assert_consumed_fragment_only([1787579400, LAST_CONSUMED_SLUG_T0 + 300])
        ok(False, "a top-up slug must be REFUSED")
    except TopUpLeak as e:
        ok("consume it" in str(e),
           "POSITIVE CONTROL: a single post-boundary slug is REFUSED, and the "
           "message names WHY it matters (consuming Phase 2's test surface), "
           "not merely that a bound was crossed")
    try:
        assert_consumed_fragment_only([1787702100])
        ok(False, "the declared top-up's last slug must be refused")
    except TopUpLeak:
        ok(True, "the declared top-up range is refused wholesale")
    ok(OUT.name.endswith("_v1.json"),
       "the candidate artifact is versioned, so a correction supersedes as v2 "
       "rather than editing a frozen file (rule 13)")
    print(f"harmful_freeze_candidate selftest: {checks} checks OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    print("REFUSED: the freeze is the USER's decision (R-162(3)).\n"
          "This builder runs only after an explicit yes, via:\n"
          "  python3 harmful_freeze_candidate.py --user-approved-freeze")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
