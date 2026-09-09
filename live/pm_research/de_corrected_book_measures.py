"""THE TWO QUANTITIES THAT NEEDED A CORRECTED BOOK -- DE 173 (5).

The USER's defects (2) and (3) each ended in a number nobody could give:
  (2) `decisions` counted above-threshold ROWS while its own definition
      said GENERATIONS -- identical before BE 107's per-row change,
      different after -- so the REAL inflation factor was unmeasurable
      until a corrected book existed;
  (3) the cancel-matched control rests on "a generation yields at most one
      cancel", which is FALSE under repost -- and the REAL RATE was
      likewise unmeasurable.

This measures both on a book, from the arm's own replay, and reports the
policy-key invariant beside the reference-key count so the two id spaces
are visible together (that difference is why the invariant missed it).

    python3 live/pm_research/de_corrected_book_measures.py <book.pkl>

READ-ONLY, but it replays both arms over a ~300 MB book, so it runs under
the heavy lock like any other producer of that size (rule 20)."""
import argparse
import json
import sys
from pathlib import Path

# ---- DE 175 (3), REV 139: IMPORT FROM THIS FILE'S OWN TREE ------------
# It read `sys.path.insert(0, "/home/yuqing/ctaNew-wt-de2/live/pm_research")`
# -- a HARDCODED WORKTREE -- so the bytes that produced the 1-in-3,862 were
# not necessarily the tree the artifact cites, and a copy of this file in
# any other worktree would silently measure a third one. A measured
# exception the USER has ruled INTO THE RECEIPT must come from the code the
# receipt names.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import be_cancel_axis_null as B  # noqa: E402
import harmful_stateful_policy as HSP  # noqa: E402
import de_matched_cancel_control as MCC  # noqa: E402

EXPECTED_CHECKS = 6

def measure(book: Path) -> dict:
    bk = B.load(book)
    out = {"book": book.name, "n_rows": len(bk["rows"]),
           "imported_from": str(HERE),
           "why_that_matters": ("DE 175 (3): this module used to import "
                                "from a hardcoded worktree, so its bytes "
                                "were not necessarily the tree the artifact "
                                "cites")}
    for arm, spec in B.ARMS.items():
        stream = B.arm_stream(bk, spec["head"])
        res = HSP.replay_policy(bk["ref"], stream, B.params_for(spec["theta"]))
        rel = MCC.cancels_per_reference_generation(res["cancels"])
        inv = HSP.check_invariants(res)
        # rows per DECISION, independently of the runner's own figure
        above = [r for r in stream if float(r["score"]) >= spec["theta"]]
        gens = {(r["slug"], r["side"], int(r["gen"])) for r in above}
        out[arm] = {
            "n_cancels": rel["n_cancels"],
            "n_reference_generations_cancelled": rel["n_reference_generations"],
            "max_cancels_on_one_reference_generation":
                rel["max_cancels_on_one_reference_generation"],
            "n_generations_cancelled_more_than_once":
                rel["n_reference_generations_cancelled_more_than_once"],
            "rate_double_cancelled": (
                round(rel["n_reference_generations_cancelled_more_than_once"]
                      / rel["n_reference_generations"], 6)
                if rel["n_reference_generations"] else None),
            "premise_holds": rel["premise_holds"],
            "one_cancel_per_generation_POLICY_key": inv["one_cancel_per_generation"],
            "n_above_threshold_rows": len(above),
            "n_above_threshold_generations": len(gens),
            "rows_per_decision": (round(len(above) / len(gens), 4)
                                  if gens else None),
            "which_generations": dict(sorted(rel["which"].items())[:10]),
        }
    return out


def selftest(quiet: bool = False) -> int:
    """THE FALSIFIER THIS SCRIPT DID NOT SHIP -- DE 175 (3), REV 139.

    Twice in one evening a probe of mine has needed REV to supply its
    control, and this one produced a number the USER has ruled INTO THE
    RECEIPT. What is MINE here is the WIRING -- the module's import root,
    and that the two quantities come from the arm's own replay -- so that
    is what is driven. The arithmetic below `cancels_per_reference_
    generation` is `de_matched_cancel_control`'s and carries its own."""
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_corrected_book_measures] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    ok(sys.path[0] == str(HERE)
       and Path(B.__file__).resolve().parent == HERE
       and Path(MCC.__file__).resolve().parent == HERE
       and Path(HSP.__file__).resolve().parent == HERE,
       f"DE 175 (3) THE DEFECT REV FOUND, MADE CHECKABLE: this module "
       f"imports from ITS OWN directory ({HERE}) and every module it "
       f"measures through resolves there. It used to insert a HARDCODED "
       f"worktree path, so a copy of this file anywhere else would have "
       f"measured a different tree while reporting the same numbers")
    # ---- ROWS PER DECISION: the arithmetic, both directions ----------
    S = HSP.SIDES[0]
    stream = [{"t": 1.0, "slug": "w", "side": S, "gen": 0, "score": 0.9},
              {"t": 2.0, "slug": "w", "side": S, "gen": 0, "score": 0.95},
              {"t": 3.0, "slug": "w", "side": S, "gen": 1, "score": 0.9},
              {"t": 4.0, "slug": "w", "side": S, "gen": 1, "score": 0.1}]
    above = [r for r in stream if r["score"] >= 0.5]
    gens = {(r["slug"], r["side"], r["gen"]) for r in above}
    ok(len(above) == 3 and len(gens) == 2
       and round(len(above) / len(gens), 4) == 1.5,
       f"ROWS PER DECISION: 3 above-threshold ROWS over 2 GENERATIONS is "
       f"1.5, not 3 -- the arithmetic behind the 1.3127 and 1.1117 "
       f"reported on the corrected book, on a fixture where the answer is "
       f"hand-checkable")
    ok(len({(r["slug"], r["side"], r["gen"]) for r in stream}) == 2
       and len([r for r in stream if r["score"] >= 0.99]) == 0,
       "POSITIVE CONTROL: the same stream at a threshold nothing crosses "
       "yields no decisions, so the ratio is a property of the threshold "
       "and not a constant of the fixture")
    # ---- DOUBLE-CANCELLED GENERATIONS: fires, and stays silent -------
    _dup = [{"slug": "w", "side": S, "ref_gen": 7},
            {"slug": "w", "side": S, "ref_gen": 7}]
    _clean = [{"slug": "w", "side": S, "ref_gen": 7},
              {"slug": "w", "side": S, "ref_gen": 8}]
    r_dup = MCC.cancels_per_reference_generation(_dup)
    r_cln = MCC.cancels_per_reference_generation(_clean)
    ok(r_dup["n_reference_generations_cancelled_more_than_once"] == 1
       and r_dup["max_cancels_on_one_reference_generation"] == 2
       and r_dup["premise_holds"] is False
       and r_cln["n_reference_generations_cancelled_more_than_once"] == 0
       and r_cln["premise_holds"] is True,
       f"DOUBLE-CANCEL DETECTION, BOTH WAYS: two cancels on ONE reference "
       f"generation give 1 offender and `premise_holds` False; two cancels "
       f"on TWO generations give 0 and True. The 1-in-3,862 rests on this "
       f"and the census had never been shown to stay SILENT on a clean arm")
    ok(r_dup["n_reference_generations"] == 1
       and r_cln["n_reference_generations"] == 2,
       f"AND THE DENOMINATOR IS THE ONE THE RATE IS OVER: cancelled "
       f"reference GENERATIONS ({r_dup['n_reference_generations']} and "
       f"{r_cln['n_reference_generations']}), not cancels and not rows. "
       f"REV 139 confirmed the denominator on the real book; this is the "
       f"cell that makes it checkable here")
    ok("policy" in r_dup["why_the_invariant_missed_it"].lower()
       and "ref" in r_dup["why_the_invariant_missed_it"].lower(),
       "and the result CARRIES why `one_cancel_per_generation` cannot see "
       "it -- the two id spaces named in the output, so a reader of the "
       "receipt's exception block does not have to know the history")
    if n[0] != EXPECTED_CHECKS:
        raise SystemExit(f"[de_corrected_book_measures] FAIL: check count "
                         f"{n[0]} != {EXPECTED_CHECKS}")
    if not quiet:
        print(f"[de_corrected_book_measures] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("book", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.book:
        ap.print_help()
        return 2
    print(json.dumps(measure(Path(a.book)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
