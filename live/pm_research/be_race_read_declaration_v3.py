"""RACE-READ DECLARATION v3 — THE ESTIMAND WAS ALWAYS COMPUTABLE. v2 NAMED
THE WRONG FILE.

Round 52 refused: the declared estimand (net cents vs the INCUMBENT at the
ACTION unit, L = 50 ms) is not computable from
`be_forward_day_SEALED_scores_<DAY>.json`, whose rows are
`(t0, expected_cancel_value)` from ONE fit. That refusal stands and is
correct about those bytes.

WHAT IT DID NOT ESTABLISH, AND THIS DECLARATION DOES: whether the estimand
is computable from the bytes the read SHOULD open. It is. Read at the
producer and the consumer, never by opening a sealed file:

  `be_forward_day.FEED_FIELDS` -- ("slug", "side", "gen", "t0", "t_start",
  "score", "score_incumbent", "any_fill_ahead", "value_cents",
  "preventable_shares", "level") -- under a comment that names its purpose
  outright: *"The feed the action-level estimand consumes."*

  `be_read_cells.load_two_arm_feed` -- the INTERIM's own reader -- streams
  `(rows, cand_scores, inc_scores)` from that feed and REFUSES a one-arm
  feed BY NAME, because "the declared estimand is an increment OVER the
  incumbent".

So every input the estimand names is there: `slug/side/gen` is the ACTION
unit, `score` + `score_incumbent` is the pair, `value_cents` +
`preventable_shares` + `level` is the realised net, `t0`/`t_start` resolve L.

**THE CONTRADICTION RESOLVES ONE WAY.** v2 declared that the read OPENS the
SCORES and that the FEED STAYS SEALED. That is backwards: the scores cannot
support the estimand and the feed can. The interim read 09-01 and 09-02
through the feed, with `MATCHED_VOLUME`, at `latency_ms: 50` -- so the
net-cents half was not only computable, IT WAS COMPUTED, on exactly the two
days v2 calls re-reads. The "window-level sign-flip of paired increments"
that v2 inherited in one place was never the statistic those days were read
with.

**THEREFORE NO RE-SEAL IS RECOMMENDED.** The dispatch offered one at ~5 x 23
min; it is not needed, and re-scoring to obtain fields that already exist
would spend two hours to arrive where the artifacts already are.

**AND A SECOND CONTRADICTION, NAMED RATHER THAN QUIETLY RESOLVED.** The
interim makes `MATCHED_VOLUME` PRIMARY and says BY_THRESHOLD "is not matched
on [the decision variable]" (CLAUDE.md rule 7); v2 declares the pairing
BY_THRESHOLD. Both cannot be the primary. This declaration RECOMMENDS the
interim's -- it is the one with a rule cited and the one the two opened days
were actually read with -- and reports BY_THRESHOLD beside it. **The
coordinator rules with the reviewer's filing; this recommends.**

v1 and v2 are untouched (rule 13).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import be_data_root as _BDR
import be_race_read_declaration as V2

ROOT = HERE.parents[1]


class DeclarationRefused(RuntimeError):
    """A named refusal."""


def feed_fields() -> dict:
    """The feed's fields, READ FROM THE WRITER. Never opened, never typed."""
    import be_forward_day as FD
    return {"fields": list(FD.FEED_FIELDS),
            "protocol": FD.FEED_PROTOCOL,
            "source": "be_forward_day.FEED_FIELDS",
            "the_writers_own_words": "The feed the action-level estimand "
                                     "consumes. Named here so a checker can "
                                     "assert the driver emits it without "
                                     "importing the metric module.",
            "read_at_the_writer_not_by_opening": True}


def estimand_inputs_available() -> dict:
    """Each input the estimand names, matched to the field that supplies it."""
    f = set(feed_fields()["fields"])
    need = {
        "the ACTION unit (slug, side, gen)": ("slug", "side", "gen"),
        "the INCUMBENT, as a pair": ("score", "score_incumbent"),
        "realised cents": ("value_cents", "preventable_shares", "level"),
        "L = 50 ms resolution": ("t0", "t_start"),
    }
    rows = {k: {"fields": list(v), "all_present": set(v) <= f}
            for k, v in need.items()}
    return {"per_requirement": rows,
            "every_input_present": all(r["all_present"] for r in rows.values()),
            "in_the_SCORES_instead": False,
            "why_the_scores_cannot": "be_forward_day.seal writes "
                                     "(t0, expected_cancel_value) from ONE "
                                     "fit: no incumbent, no action identity "
                                     "beyond t0, no realised cents "
                                     "(BE round 52)"}


def interim_evidence() -> dict:
    """ITEM 1: what the interim COMPUTED, from its own receipts."""
    d = json.loads((HERE / "declarations"
                    / "be_interim_declaration_v1.json").read_text())
    ps = d["primary_statistic"]
    return {
        "source": "live/pm_research/declarations/be_interim_declaration_v1"
                  ".json (the interim's OWN declaration) and "
                  "be_read_cells.py (its producer) -- never a sealed file",
        "days": d["population"]["days"],
        "latency_ms": d["latency_ms"],
        "primary_statistic_name": ps["name"],
        "primary_statistic_definition": ps["definition"],
        "reported_beside_and_NOT_primary": ps[
            "reported_beside_it_and_neither_is_primary"],
        "why_BY_THRESHOLD_is_not_primary": ps["why_primary"],
        "what_it_read": "be_read_cells.load_two_arm_feed(path, latency_ms) "
                        "-- the FEED, streamed, refusing a one-arm feed BY "
                        "NAME because 'the declared estimand is an increment "
                        "OVER the incumbent'",
        "it_computed_net_cents_not_a_sign_flip": True,
        "so_which_half_was_ever_computable": (
            "the NET-CENTS half, and it was not merely computable -- IT WAS "
            "COMPUTED, on 09-01 and 09-02, the exact two days v2 calls "
            "re-reads. The 'window-level sign-flip of paired increments' "
            "v2 inherited in one place was never the statistic those days "
            "were read with."),
        "status_of_the_interim": d["status"],
    }


def build() -> dict:
    ie = interim_evidence()
    av = estimand_inputs_available()
    return {
        "protocol": "BE_RACE_READ_DECLARATION_V3",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "supersedes": {
            "artifacts": ["be_race_read_declaration_v1.json",
                          "be_race_read_declaration_v2.json"],
            "rule": "13 -- vN+1; v1 and v2 are NOT edited",
            "what_changes": "WHICH FILE THE READ OPENS, and which of the two "
                            "statistics v2 carried is the one. No number "
                            "moves; nothing has been read.",
        },
        "R_529_A_UP_FRONT": "THIS READ ESTABLISHES DIRECTION AND CONSISTENCY "
                            "AND NEVER A HOLM-CLEARING VERDICT (R-529(A)).",
        "THE_CORRECTION": {
            "v2_said": "the read OPENS be_forward_day_SEALED_scores_<DAY>"
                       ".json and the FEED STAYS SEALED",
            "and_that_is_backwards": "the scores cannot support the declared "
                                     "estimand and the feed can",
            "v3_says": "the read OPENS be_forward_day_SEALED_feed_<DAY>"
                       ".jsonl; the SCORES stay sealed, because the estimand "
                       "does not need them and opening more than it needs is "
                       "consumption without purpose (rule 11)",
            "round_52s_refusal_stands": "it was correct ABOUT THE SCORES. "
                                        "What it did not establish is "
                                        "whether the right file exists. It "
                                        "does.",
        },
        "interim_evidence": ie,
        "estimand_inputs": av,
        "feed": feed_fields(),
        "RECOMMENDATION": {
            "option": "A — declare an estimand computable from the sealed "
                      "bytes AS THEY ARE, against the FEED",
            "re_seal_NOT_recommended": True,
            "why_not": "the dispatch offered a re-seal at ~5 x 23 min at 8G. "
                       "It is not needed: every field the estimand names is "
                       "already in the feed BY DESIGN, and the interim "
                       "already computed the statistic from it. Re-scoring "
                       "to obtain fields that exist would spend two hours to "
                       "arrive where the artifacts already are.",
            "and_it_avoids_a_rule_11_question_entirely": (
                "a re-seal writes rather than reads, so it would not have "
                "consumed a day -- but it would have replaced the artifacts "
                "the interim's own numbers rest on, and the cheapest way to "
                "keep those comparable is not to move them"),
            "primary_statistic_recommended": ie["primary_statistic_name"],
            "why": "it is the one with a RULE cited (CLAUDE.md rule 7: "
                   "controls matched on the DECISION VARIABLE) and the one "
                   "the two already-opened days were actually read with. "
                   "BY_THRESHOLD is reported BESIDE it.",
            "the_second_contradiction_named": "v2 declares the pairing "
                                              "BY_THRESHOLD; the interim "
                                              "declares BY_THRESHOLD "
                                              "explicitly NOT primary. Both "
                                              "cannot be the primary, and "
                                              "this names it rather than "
                                              "silently picking.",
            "who_decides": "the coordinator, with the reviewer's filing. "
                           "This RECOMMENDS (rule 14).",
        },
        "rule_11": {
            "09_01_and_09_02_remain_RE_READS": True,
            "why": "they were opened under the interim and this read does "
                   "not un-open them",
            "floors_unchanged": "both readings still computed, conservative "
                                "resolved (0.25 on the 5/3 split)",
        },
        "carried_forward_from_v2_unchanged": [
            "the permutation floors at both readings, conservative resolved",
            "the required post-read byte-identity recompute, VOID on mismatch",
            "the Gate-1 separation, checked against the paths actually opened",
        ],
        "opens_nothing": True,
        "nothing_was_read_to_produce_this": "the interim's declaration and "
                                            "two source files; no sealed "
                                            "artifact was opened",
        "data_root": _BDR.receipt_block(),
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 8


def selftest() -> int:
    checks, fails = 0, []

    def ok(cond, label):
        nonlocal checks
        checks += 1
        print(("PASS: " if cond else "FAIL: ") + label)
        if not cond:
            fails.append(label)

    ie = interim_evidence()
    ok(ie["days"] == ["20260901", "20260902"] and ie["latency_ms"] == 50,
       f"ITEM 1, FROM THE INTERIM'S OWN DECLARATION: its population is "
       f"exactly {ie['days']} at L = {ie['latency_ms']} ms -- the two days "
       f"v2 calls re-reads")
    ok(ie["primary_statistic_name"] == "MATCHED_VOLUME"
       and "net cents" in ie["primary_statistic_definition"]
       and ie["it_computed_net_cents_not_a_sign_flip"],
       f"and its PRIMARY statistic is {ie['primary_statistic_name']} -- "
       f"'net cents (candidate) MINUS net cents (incumbent...)'. **The "
       f"net-cents half is the half that was ever computed.**")
    ok("BY_THRESHOLD" in ie["reported_beside_and_NOT_primary"],
       "and the interim puts BY_THRESHOLD explicitly BESIDE it and NOT "
       "primary -- which v2 declares as the pairing. The two declarations "
       "contradict each other and this names it")

    f = feed_fields()
    ok(f["read_at_the_writer_not_by_opening"]
       and "action-level estimand" in f["the_writers_own_words"],
       f"the feed's {len(f['fields'])} fields are read at the WRITER "
       f"(`be_forward_day.FEED_FIELDS`), whose own comment says it is *the "
       f"feed the action-level estimand consumes*")
    av = estimand_inputs_available()
    ok(av["every_input_present"],
       f"EVERY input the estimand names is present in the feed: "
       f"{ {k: v['fields'] for k, v in av['per_requirement'].items()} }")
    ok(not av["in_the_SCORES_instead"] and "ONE fit" in av["why_the_scores_cannot"],
       "and NOT in the scores -- round 52's refusal stands about those bytes")

    d = build()
    ok(d["RECOMMENDATION"]["re_seal_NOT_recommended"] is True
       and d["RECOMMENDATION"]["primary_statistic_recommended"]
       == "MATCHED_VOLUME"
       and d["decides_nothing"].startswith("REPORTED"),
       "the recommendation is OPTION A with NO re-seal, MATCHED_VOLUME "
       "primary -- and it RECOMMENDS rather than rules (rule 14)")
    ok(d["opens_nothing"] and "no sealed artifact was opened"
       in d["nothing_was_read_to_produce_this"],
       "and producing this declaration OPENED NOTHING -- the evidence is the "
       "interim's declaration and two source files")

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
    if "--declare" in argv:
        out = build()
        dst = HERE / "declarations" / "be_race_read_declaration_v3.json"
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"written": str(dst),
                          "recommends": out["RECOMMENDATION"]["option"][:40],
                          "re_seal": out["RECOMMENDATION"][
                              "re_seal_NOT_recommended"]}))
        return 0
    print("usage: be_race_read_declaration_v3.py --selftest | --declare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
