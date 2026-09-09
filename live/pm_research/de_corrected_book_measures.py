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
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/yuqing/ctaNew-wt-de2/live/pm_research")
import be_cancel_axis_null as B, harmful_stateful_policy as HSP
import de_matched_cancel_control as MCC

BOOK = Path(sys.argv[1] if len(sys.argv) > 1 else
            "/home/yuqing/ctaNew/data/pm_5min/derived/"
            "be_daybook_20260903_btc__L250ms__EV21.pkl")
bk = B.load(BOOK)
out = {"book": BOOK.name, "n_rows": len(bk["rows"])}
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
print(json.dumps(out, indent=2, sort_keys=True))
