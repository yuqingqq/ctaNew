"""ADVERSARIAL contract test: does the fit stage read DA's verdict correctly?

SURFACE AUTHORISATION (R-126): R-203(3) rules the verdict contract and assigns
DA the falsifier for it. This is a TEST OF A CONSUMER, not an implementation of
one -- DA checks, DA does not specify (R-185). BE owns the fix; these are the
assertions it must satisfy.

WHY IT EXISTS. User audit #5 finding 3: `phase2_arms.assert_gate_passed`
accepted any `{"verdict": "PASS"}` and REFUSED DA's real emission, whose
`verdict` field carries the SCHEMA NAME `da_tape_gate_verdict_v1`. So the gate
that is supposed to authorise fitting could be satisfied by a two-line file
anyone could write, while the genuine artifact was rejected. Both directions
wrong at once.

THE CONTRACT (R-203(3)), as assertions:
  1. REFUSE a fabricated {"verdict": "PASS"} -- no predicate table, no subject.
  2. ACCEPT DA's real emission when all_pass is true AND the subject matches.
  3. REFUSE DA's real emission when all_pass is false.
  4. RECOMPUTE all_pass from the predicate table -- a header claiming PASS over
     a table containing a failure must be REFUSED. (DA's writer already
     recomputes; R-203 makes that mandatory on the READ side too, because a
     reader that trusts a headline can be handed a doctored one.)
  5. REFUSE when the predicate table is absent.
  6. REFUSE when the subject does not match the tape about to be used --
     tape_path / sha prefix / bytes -- so a stale PASS cannot be replayed
     against a different artifact.

    python3 live/pm_research/da_verdict_contract_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[2]


def _real_emission(tape: Path, all_pass: bool = True) -> dict:
    """A genuine DA verdict, produced by DA's own writer."""
    import da_state_tape_verify as G
    preds = [
        {"predicate": "tape_non_empty", "pass": True, "applicable": True,
         "detail": "1 rows"},
        {"predicate": "gap_count_matches_expected", "pass": all_pass,
         "applicable": True, "detail": "289 vs 289" if all_pass else "0 vs 289"},
        {"predicate": "embargo_respected", "pass": False, "applicable": False,
         "detail": "ENFORCED-DOWNSTREAM"},
    ]
    rep = {"predicates": preds, "n_rows": 1, "schema_family": "PRED_STATE_V1",
           "tape_header_pins": {}, "not_applicable": ["embargo_respected"]}
    out = tape.parent / "verdict.json"
    return G.write_verdict(rep, tape, out), out


def main() -> int:
    import phase2_arms as ARMS
    fn = getattr(ARMS, "assert_gate_passed", None)
    if fn is None:
        print("REFUSED: phase2_arms.assert_gate_passed not found -- the "
              "contract cannot be tested against a consumer that is absent.")
        return 2

    results = []

    def case(label, path_written, expect_accept, tape=None):
        """Point the consumer at `path_written`; did it accept?"""
        orig = ARMS.DA_VERDICT
        try:
            ARMS.DA_VERDICT = path_written
            try:
                fn()
                accepted = True
            except Exception:
                accepted = False
        finally:
            ARMS.DA_VERDICT = orig
        ok = accepted == expect_accept
        results.append(ok)
        verb = "ACCEPTED" if accepted else "REFUSED"
        want = "accept" if expect_accept else "REFUSE"
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}: {verb} (must {want})")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        tape = d / "phase2_state_tape_v5.json"
        tape.write_text(json.dumps({"rows": []}), encoding="utf-8")

        # 1. FABRICATION -- the two-line file anyone could write
        fab = d / "fab.json"
        fab.write_text(json.dumps({"verdict": "PASS"}), encoding="utf-8")
        case("fabricated {'verdict':'PASS'}", fab, False)

        # 2. DA's REAL emission, passing
        real, rp = _real_emission(tape, all_pass=True)
        case("DA's real emission, all_pass true", rp, True)

        # 3. DA's REAL emission, failing
        realf, rpf = _real_emission(tape, all_pass=False)
        case("DA's real emission, all_pass FALSE", rpf, False)

        # 4. DOCTORED headline over a failing table
        doc = json.loads(rpf.read_text())
        doc["all_pass"] = True
        dp = d / "doctored.json"
        dp.write_text(json.dumps(doc), encoding="utf-8")
        case("headline PASS over a FAILING table", dp, False)

        # 5. table absent
        noc = json.loads(rp.read_text())
        noc.pop("predicates", None)
        np_ = d / "notable.json"
        np_.write_text(json.dumps(noc), encoding="utf-8")
        case("predicate table ABSENT", np_, False)

        # 6. subject mismatch -- a stale PASS replayed on other bytes
        mis = json.loads(rp.read_text())
        mis["tape_sha256_prefix"] = "0" * 16
        mp = d / "mismatch.json"
        mp.write_text(json.dumps(mis), encoding="utf-8")
        case("subject sha MISMATCH vs the tape in use", mp, False)

    n_ok = sum(results)
    print(f"\nverdict contract: {n_ok}/{len(results)} satisfied")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
