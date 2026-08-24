"""Pin a review target's hash at dispatch and verify it at report.

THE DEFECT THIS EXISTS FOR, committed twice by DA in consecutive iterations of
one loop: the charter says the target is FROZEN for the duration of an
iteration, and the author edited it mid-run anyway -- once while a conformance
reviewer was reading, once while a citation reviewer was.  Both reviewers
detected it themselves and re-verified; neither should have had to.

A DECLARED FREEZE IS PROSE.  The loop charter asserted it, the author agreed to
it in writing one turn before breaching it the second time, and nothing in the
process could tell.  That is the same shape this programme rules against
everywhere else -- a rule whose text reads correctly while nothing evaluates it
-- arriving in the review method itself, which is exactly where R-61 found the
moving lens set.

So: record the sha256 at dispatch, re-check at report, and let the ITERATION
fail rather than the reviewer notice.  A reviewer catching the author's breach
is a lucky outcome, not a control: it depends on the reviewer having taken a
hash for their own reasons, which is how both of these were actually caught.

Consequence, deliberately harsh: a breached iteration is NOT streak-eligible
under R-61, because the reviewers did not review one document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PIN_SUFFIX = ".freeze-pin.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def pin(target: Path, iteration: str, pin_dir: Path) -> dict:
    """Record the target's hash IMMEDIATELY BEFORE dispatching reviewers.

    Not at the end of the previous iteration.  DA pinned for iteration 6 while
    finishing iteration 5's follow-up, then edited the banner and the loop log,
    and `verify` reported FREEZE_BREACHED -- correctly, but for a workflow
    defect rather than a review breach: no reviewer had read a moving document.
    The verdict is right either way, and that is the point of a mechanism over a
    promise; but a pin taken early converts ordinary between-iteration edits into
    breach reports, and a checker that cries wolf is one nobody reads.
    """
    rec = {
        "target": str(target),
        "iteration": iteration,
        "sha256": sha256_of(target),
        "bytes": target.stat().st_size,
    }
    out = pin_dir / (target.name + PIN_SUFFIX)
    out.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    return rec


def verify(target: Path, pin_dir: Path) -> dict:
    """Re-check at report time.  Returns a verdict; never guesses."""
    p = pin_dir / (target.name + PIN_SUFFIX)
    if not p.exists():
        # Fail CLOSED.  A missing pin is not a passing freeze -- it is an
        # iteration whose freeze cannot be certified, which is the same thing
        # as a broken one for streak purposes.
        return {"verdict": "NO_PIN", "streak_eligible": False,
                "detail": f"no pin recorded at {p}"}
    rec = json.loads(p.read_text(encoding="utf-8"))
    now = sha256_of(target)
    if now == rec["sha256"]:
        return {"verdict": "FREEZE_HELD", "streak_eligible": True,
                "sha256": now, "iteration": rec["iteration"]}
    return {"verdict": "FREEZE_BREACHED", "streak_eligible": False,
            "iteration": rec["iteration"],
            "pinned": rec["sha256"], "now": now,
            "detail": "the target moved during the iteration; reviewers did "
                      "not review one document, so this iteration cannot "
                      "advance the streak"}


def _selftests() -> int:
    import tempfile
    checks = 0

    def ok(c, label):
        nonlocal checks
        checks += 1
        if not c:
            raise AssertionError(f"selftest failed: {label}")

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        t = d / "PLAN.md"
        t.write_text("revision 1\n", encoding="utf-8")

        pin(t, "it-1", d)
        ok(verify(t, d)["verdict"] == "FREEZE_HELD", "unchanged target holds")
        ok(verify(t, d)["streak_eligible"], "and is streak-eligible")

        # THE ACTUAL DEFECT: an edit during the iteration.
        t.write_text("revision 1\nedited mid-iteration\n", encoding="utf-8")
        v = verify(t, d)
        ok(v["verdict"] == "FREEZE_BREACHED", "a mid-iteration edit is detected")
        ok(not v["streak_eligible"], "and the iteration cannot advance the streak")
        ok(v["pinned"] != v["now"], "both hashes are reported, not just a verdict")

        # A WHITESPACE-ONLY edit is still a breach: reviewers cite line numbers.
        t.write_text("revision 1\n", encoding="utf-8")
        pin(t, "it-2", d)
        t.write_text("revision 1\n\n", encoding="utf-8")
        ok(verify(t, d)["verdict"] == "FREEZE_BREACHED",
           "a whitespace-only edit is still a breach -- line numbers move")

        # FAIL CLOSED: no pin is not a pass.
        ok(verify(d / "OTHER.md" if (d / "OTHER.md").exists() else t,
                  d / "nowhere")["verdict"] == "NO_PIN",
           "a missing pin fails closed rather than certifying")
        ok(not verify(t, d / "nowhere")["streak_eligible"],
           "and is not streak-eligible")

    print(f"da_freeze_pin selftests: {checks} checks passed")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--target")
    ap.add_argument("--iteration", default="?")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent / ".freeze"))
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftests())
    d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    t = Path(a.target)
    rec = pin(t, a.iteration, d) if a.pin else verify(t, d)
    print(json.dumps(rec, indent=2, sort_keys=True))
    raise SystemExit(0 if rec.get("verdict") in (None, "FREEZE_HELD") else 1)


if __name__ == "__main__":
    main()
