"""THE 1,309: WHERE 31,122 BECOMES 29,813, ON BE'S FILE, BY DIGEST.

DE explained the gap as a filter stage in one pipeline (Q-DE-67) and then
noted (0fe061b) that the argument rests on COUNTS OVER DE's CACHE, not on
the two files being identical. So the two derivations cannot be compared
until each side publishes its own -- WHICH FILE, WHICH STAGE, WHICH COUNT.
This is BE's side. It asserts nothing about DE's file.

THE DERIVATION, in the two stages the loader actually has:

  STAGE 1  every generation in the reference, both sides, all slugs
  STAGE 2  those with an ASSEMBLED SCORE, keyed (slug, side, t0)

  1,309 = STAGE 1 - STAGE 2.

`be_cancel_axis_null.load()` builds its decision population by exactly that
membership test, so the 1,309 are generations the null CANNOT draw -- they
have no score to threshold. `de_section81_arms.stream()` names the cause in
its own docstring: the scorer refuses a generation whose rows the feature
pass dropped, "since scoring it would be scoring from nothing", and counts
the drop as `excluded_no_assembled_score` (rule 4).

THE FACT THAT DECIDES WHETHER THIS IS A HEAD-SPECIFIC FILTER: the count is
computed for BOTH pinned heads separately. If they agree, the drop is a
property of the ASSEMBLY, not of either model.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import be_data_root as _BDR

ROOT = HERE.parents[1]
#: READ root resolves through the shared helper (R-559(C));
#: WRITES stay in this seat's worktree -- the ledger's
#: derived/ is the MAIN TREE's checkout.
_DATA_ROOT = Path(_BDR.resolve(ROOT)["data_root"])
DERIVED = _BDR.derived()  # BE48 B.4: one root, from the resolver. This was `ROOT / 'data/...'` -- a data root built on a CODE root, which the first version of `audit_derived_roots` could not see because the value holds no `parents` and no literal.          # WRITE (this seat)
CACHE = _DATA_ROOT / "pm_5min/derived/de_section81_cache_12.pkl"  # READ (ledger)
HEADS = ("q1_arrival_composed_lgbm", "incumbent_linear_d")
COIN = "btc"


class CountRefused(RuntimeError):
    """A named refusal."""


def derive(path: Path | None = None) -> dict:
    import harmful_stateful_policy as HSP
    p = Path(path) if path is not None else CACHE
    if not p.exists():
        raise CountRefused(f"REFUSED: no cache at {p}.")
    body = p.read_bytes()
    c = pickle.loads(body)
    ref = c["fr"]["reference"]
    asm = c["asm"]
    stage1 = sum(len(ref[s][sd]) for s in ref for sd in HSP.SIDES)
    per_head = {}
    for head in HEADS:
        key = (COIN, head)
        if key not in asm["by_arm"]:
            per_head[head] = {"status": "NOT_ASSEMBLED_IN_THIS_CACHE"}
            continue
        gs = asm["by_arm"][key][0]
        kept = sum(1 for s in sorted(ref) for sd in HSP.SIDES
                   for g in ref[s][sd] if (s, sd, float(g["t0"])) in gs)
        per_head[head] = {
            "len_gen_scores": len(gs),
            "stage2_rows_kept": kept,
            "dropped": stage1 - kept,
            "membership_test": "(slug, side, float(g['t0'])) in gen_scores",
        }
    got = {h: v.get("dropped") for h, v in per_head.items()
           if "dropped" in v}
    vals = sorted(set(got.values()))
    return {
        "protocol": "BE_GENERATION_COUNT_DERIVATION_V2",
        "supersedes": {
            "artifact": "data/pm_5min/derived/"
                        "be_generation_count_derivation_v1.json",
            "rule": "13 -- vN+1; v1 is NOT edited and stays at its bytes",
            "what_changed": "PROVENANCE ONLY: the resolved data root and the "
                            "branch taken now travel in the receipt "
                            "(R-559(C)). Every count is unchanged and the "
                            "checker asserts them.",
        },
        "data_root": _BDR.receipt_block(),
        "as_of_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "this_is_BEs_derivation_only": (
            "it publishes WHICH FILE, WHICH STAGE and WHICH COUNT so BE's and "
            "DE's derivations can be compared by digest and count. It makes "
            "NO claim about DE's cache and does not assert the two files are "
            "identical -- that is the thing 0fe061b said was not established."),
        "file": {"path": str(p), "sha256": hashlib.sha256(body).hexdigest(),
                 "bytes": len(body)},
        "stage_1": {"count": stage1,
                    "what": "every generation in the reference, both sides, "
                            "all slugs",
                    "expression": "sum(len(ref[slug][side]) for slug, side)",
                    "n_slugs": len(ref), "sides": list(HSP.SIDES)},
        "stage_2_by_head": per_head,
        "THE_GAP": {
            "value": vals[0] if len(vals) == 1 else None,
            "identical_across_both_pinned_heads": len(vals) == 1,
            "per_head": got,
            "reading": ("IDENTICAL across both heads, so the drop is a "
                        "property of the ASSEMBLY and not of either model"
                        if len(vals) == 1 else
                        "DIFFERS by head -- the drop would then be "
                        "head-specific and this reading does not hold"),
        },
        "cause_named_at_its_own_source": {
            "module": "de_section81_arms.stream",
            "docstring": "the scorer refuses a generation whose rows the "
                         "feature pass dropped -- correctly, since scoring "
                         "it would be scoring from nothing -- so the "
                         "exclusion happens HERE, before scoring, and is "
                         "COUNTED (rule 4)",
            "status_field": "excluded_no_assembled_score",
            "quoted_not_paraphrased": True,
        },
        "consequence_for_the_null": (
            "be_cancel_axis_null.load() builds its decision population by "
            "this exact membership test, so these generations cannot be "
            "drawn: they have no score to threshold."),
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 6


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    d = derive()
    s1 = d["stage_1"]["count"]
    ok(s1 == 31122,
       f"STAGE 1 is {s1:,} reference generations -- the same number the filed "
       f"arms artifact carries as `population.generations`")
    ok(all(v.get("len_gen_scores") == v.get("stage2_rows_kept")
           for v in d["stage_2_by_head"].values() if "dropped" in v),
       "and for each head the assembled-score COUNT equals the rows the "
       "membership test KEEPS -- so the gap is not an artefact of the test "
       "disagreeing with the map it reads")
    ok(d["THE_GAP"]["value"] == 1309 and d["THE_GAP"]["identical_across_both_pinned_heads"],
       f"THE GAP IS {d['THE_GAP']['value']:,} AND IT IS IDENTICAL ACROSS BOTH "
       f"PINNED HEADS ({d['THE_GAP']['per_head']}) -- a property of the "
       f"assembly, not of either model")
    ok(d["file"]["sha256"].startswith("ee1f150b263c1c9a"),
       f"the derivation names the FILE it ran on by digest "
       f"({d['file']['sha256'][:16]}…), which is the whole point: DE's "
       f"counts are over DE's cache")
    ok(s1 - d["THE_GAP"]["value"] == 29813,
       f"and the arithmetic closes: {s1:,} - {d['THE_GAP']['value']:,} = "
       f"29,813, the decision population the null draws from")
    try:
        derive(Path("/nonexistent/cache.pkl"))
        ok(False, "a missing cache must refuse")
    except CountRefused as e:
        ok("no cache at" in str(e),
           "KNOWN-BAD: a missing cache REFUSES rather than reporting a gap "
           "of zero over an empty reference")

    print()
    if fails:
        print(f"{len(fails)} FAILURES of {checks} checks")
        return 1
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}")
        return 1
    print(f"{checks} checks passed")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--emit" in argv:
        out = derive()
        dst = DERIVED / "be_generation_count_derivation_v2.json"
        dst.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(json.dumps({"written": str(dst),
                          "stage_1": out["stage_1"]["count"],
                          "gap": out["THE_GAP"]["value"],
                          "same_both_heads":
                              out["THE_GAP"]["identical_across_both_pinned_heads"]}))
        return 0
    print("usage: be_generation_count_derivation.py --selftest | --emit")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
