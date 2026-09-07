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


class StaleHistoryPin(RuntimeError):
    """A named refusal: the bytes a PAST act consumed are not these bytes."""


#: THE PINS `build_v4` WAS BUILT FROM, AS A PAIR (REV 86 §8 / R-729: an
#: instrument touching a PAST act resolves every declaration BY THE PAIR THE
#: ACT RECORDED, never by the head -- the head is for writers).
#:
#: BE 91 classified `:235` at the code. It was a bare filename passed to
#: `json.loads`, which is DA 94's own separating property for a stale pin:
#: bytes consumed by a DIGEST cannot make code behave as if it were under
#: old bars, bytes consumed by `json.loads` can. But the answer is NOT to
#: resolve the head. v4 is LANDED, and its G, its floor and its
#: READ_BUT_UNRECOVERABLE set are all DERIVED from these pins' `exists`
#: flags -- so the head would silently restate a landed declaration:
#: measured, the pins head (v2, 09-06 pinned) has FOUR readable days, which
#: gives G = 4 and a floor of 0.125 against v4's landed G = 3 and 0.25.
#: The literal's INTENT was right and its SHAPE was wrong.
#:
#: AND v4 RECORDED NO DIGEST. It names the file in prose
#: (`population.the_pins_say_so_too`) and nothing more, so the pair cannot
#: be RECOVERED from v4 -- it is pinned here, once, against the landed v1,
#: which rule 20 makes IMMUTABLE. That is exactly why the refusal below is
#: the point rather than a formality: v1's bytes cannot legitimately move,
#: so if they ever do, this builder must STOP rather than recompute a
#: landed declaration from bytes it never saw.
PINS_AS_V4_RECORDED = {
    "path": "be_race_read_feed_pins_v1.json",
    "sha256": "2cca55c64ffca8e78937533da617a52501a5af1f7418f4b4be8e538a683f9b04",
    "recorded_by": "NOT v4 -- v4 names the file without a digest; pinned at "
                   "BE 91 against the landed v1 (rule 20: immutable)",
}


def pins_as_v4_recorded(declarations=None) -> dict:
    """The pins `build_v4` consumed, BY THE PAIR, or a refusal BY NAME."""
    import hashlib
    d = Path(declarations) if declarations else (HERE / "declarations")
    q = d / PINS_AS_V4_RECORDED["path"]
    if not q.exists():
        raise StaleHistoryPin(
            f"HISTORY_PIN_ABSENT: {PINS_AS_V4_RECORDED['path']} is not under "
            f"{d}. A check that depends on a declaration FAILS when it is "
            f"gone; it does not skip (R-649) -- and it does not fall back to "
            f"the head, which is a DIFFERENT population.")
    got = hashlib.sha256(q.read_bytes()).hexdigest()
    if got != PINS_AS_V4_RECORDED["sha256"]:
        raise StaleHistoryPin(
            f"HISTORY_PIN_MOVED: {PINS_AS_V4_RECORDED['path']} hashes "
            f"{got[:16]}… but v4 was built from "
            f"{PINS_AS_V4_RECORDED['sha256'][:16]}…. A landed version is "
            f"IMMUTABLE (rule 20), so these bytes moving IS the defect. "
            f"Refusing rather than rebuilding a landed declaration from "
            f"bytes it never saw.")
    return json.loads(q.read_text())


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


def build_v4(declarations=None) -> dict:
    """v4: the population is the five NAMED days, and only three are readable.

    R-549 / REV 44 B / DA 65: 09-01 and 09-02 were opened under the interim
    and their surviving receipts are SEAL-RELOCATION receipts, which carry no
    economics. No interim OUTPUT artifact exists. So those two days are
    READ-BUT-UNRECOVERABLE: consumed under rule 11 and unavailable as
    numbers. The read is therefore stated at G = 3 -- which is v2's own
    PESSIMISTIC branch, not a new choice -- with the floor 0.25.

    NO RECEIPT IS CITED FOR A NUMBER IT DOES NOT CARRY."""
    import hashlib
    # `declarations` is a parameter ONLY so the SEAM can be driven: a cell
    # that tests `pins_as_v4_recorded` alone tests the helper, not that THIS
    # builder uses it (REV 84 §3.1 -- an importer's own cell belongs at the
    # seam). Production callers pass nothing.
    _decl = Path(declarations) if declarations else (HERE / "declarations")
    v3p = _decl / "be_race_read_declaration_v3.json"
    pins = pins_as_v4_recorded(_decl)   # BY THE PAIR, never the head (R-729)
    per = pins["per_day"]
    readable = sorted(d for d, v in per.items() if v.get("exists"))
    unrec = sorted(d for d, v in per.items() if not v.get("exists"))
    m = 2
    floor_g3 = m / 2 ** len(readable)
    ie = interim_evidence()
    return {
        "protocol": "BE_RACE_READ_DECLARATION_V4",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "supersedes": {
            "artifact": "be_race_read_declaration_v3.json",
            "sha256": hashlib.sha256(v3p.read_bytes()).hexdigest()
                      if v3p.exists() else None,
            "rule": "13 -- vN+1; v3 is NOT edited",
            "what_changes": "the POPULATION and G. The estimand, the "
                            "statistic and the no-re-seal ruling are "
                            "unchanged.",
        },
        "R_529_A_UP_FRONT": "THIS READ ESTABLISHES DIRECTION AND CONSISTENCY "
                            "AND NEVER A HOLM-CLEARING VERDICT (R-529(A)).",
        "population": {
            "the_five_named_days": sorted(per),
            "READABLE": readable,
            "READ_BUT_UNRECOVERABLE": unrec,
            "why_unrecoverable": (
                "09-01 and 09-02 were OPENED under the interim (R-549), so "
                "they are consumed under rule 11 -- but their surviving "
                "receipts are SEAL-RELOCATION receipts, which carry no "
                "economics, and NO interim OUTPUT artifact exists. Verified "
                "independently by REV 44 §B and DA 65. Consumed and "
                "unavailable are different facts and both hold."),
            "the_pins_say_so_too": "be_race_read_feed_pins_v1.json marks "
                                   "both `exists: false` with no digest; the "
                                   "reader REFUSES such a day BY NAME rather "
                                   "than skipping it",
        },
        "G": len(readable),
        "why_G_is_3": "it is v2's OWN `PESSIMISTIC_only_the_three_first_"
                      "openings_are_fresh` branch, now the only branch the "
                      "artifacts support -- not a choice made after seeing "
                      "anything",
        "permutation_floor": {
            "G": len(readable), "multiplicity": m,
            "best_possible_adjusted_p": floor_g3,
            "clears_0_05": floor_g3 <= 0.05,
            "computed_here_not_quoted": True,
        },
        "statistic": {
            "primary": ie["primary_statistic_name"],
            "definition": ie["primary_statistic_definition"],
            "latency_ms": ie["latency_ms"],
            "BY_THRESHOLD": "REPORTED, never primary (rule 7, as the interim "
                            "states it)",
            "unchanged_from_v3": True,
        },
        "re_seal": {"recommended": False,
                    "why": "every field the estimand names is already in the "
                           "feed by design (v3); and a re-seal could not "
                           "recover 09-01/02's economics, because the "
                           "interim's OUTPUT was never written"},
        "cites_no_receipt_for_a_number_it_does_not_carry": True,
        "opens_nothing": True,
        "decides_nothing": "REPORTED (rule 14).",
    }


EXPECTED_CHECKS = 17      # BE 91: +5 for the `:235` history-pin classification (4 + the seam)


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

    v4 = build_v4()
    ok(v4["G"] == 3 and v4["population"]["READ_BUT_UNRECOVERABLE"]
       == ["20260901", "20260902"],
       f"v4: G = {v4['G']} and the two READ-BUT-UNRECOVERABLE days are "
       f"{v4['population']['READ_BUT_UNRECOVERABLE']} -- taken from the "
       f"PINS' own `exists: false`, not typed")
    ok(v4["permutation_floor"]["best_possible_adjusted_p"] == 0.25
       and not v4["permutation_floor"]["clears_0_05"],
       f"and the floor is COMPUTED at "
       f"{v4['permutation_floor']['best_possible_adjusted_p']} -- v2's own "
       f"PESSIMISTIC branch, which is now the only branch the artifacts "
       f"support")
    ok(v4["statistic"]["primary"] == "MATCHED_VOLUME"
       and "never primary" in v4["statistic"]["BY_THRESHOLD"]
       and v4["re_seal"]["recommended"] is False,
       "MATCHED_VOLUME stays primary, BY_THRESHOLD reported never primary, "
       "and no re-seal -- unchanged from v3")
    ok(v4["supersedes"]["sha256"] is not None
       and v4["cites_no_receipt_for_a_number_it_does_not_carry"],
       f"v4 supersedes v3 BY SHA256 ({str(v4['supersedes']['sha256'])[:16]}…) "
       f"and cites no receipt for a number it does not carry")

    # ---- BE 91: `:235` CLASSIFIED AT THE CODE, AND RESOLVED ------------
    # It was `json.loads` on a bare filename -- DA 94's own separating
    # property for a stale pin (hashed -> not a pin; INTERPRETED -> refused).
    # Classified: a READER OF HISTORY whose SHAPE was wrong, not a head
    # consumer. The cells below drive the refusal both ways on SCRATCH
    # declarations, and measure what the HEAD would have done.
    import hashlib as _hl, json as _js, tempfile as _tfP
    _pd = Path(_tfP.mkdtemp(prefix="be91_pins_"))
    _real = json.loads((HERE / "declarations"
                        / PINS_AS_V4_RECORDED["path"]).read_text())
    (_pd / PINS_AS_V4_RECORDED["path"]).write_text(json.dumps(_real))
    try:
        pins_as_v4_recorded(_pd); _moved = "NOT REFUSED"
    except StaleHistoryPin as _e:
        _moved = str(_e).split(":")[0]
    ok(_moved == "HISTORY_PIN_MOVED",
       f"KNOWN-BAD A: the SAME pins content re-serialised (same days, one "
       f"byte of whitespace different) is REFUSED BY NAME ({_moved}) -- the "
       f"pin is the PAIR, so `it is still v1` is not the question and a "
       f"filename could never have asked it")
    _pd2 = Path(_tfP.mkdtemp(prefix="be91_pinsabs_"))
    try:
        pins_as_v4_recorded(_pd2); _abs = "NOT REFUSED"
    except StaleHistoryPin as _e:
        _abs = str(_e).split(":")[0]
    ok(_abs == "HISTORY_PIN_ABSENT",
       f"KNOWN-BAD B: an ABSENT pins file REFUSES under its OWN name "
       f"({_abs}) rather than skipping or falling back to the head -- the "
       f"two faults are not the same fault (R-649)")
    _got = pins_as_v4_recorded()
    ok(sorted(_got["per_day"]) == ["20260901", "20260902", "20260903",
                                   "20260904", "20260905"]
       and _hl.sha256((HERE / "declarations"
                       / PINS_AS_V4_RECORDED["path"]).read_bytes()).hexdigest()
       == PINS_AS_V4_RECORDED["sha256"],
       f"POSITIVE CONTROL ON THE REAL PIN: the pair resolves and returns the "
       f"FIVE days v4 was built from -- a guard shown only to refuse has not "
       f"been shown to work (rule 16). v4's G = {v4['G']} and floor "
       f"{v4['permutation_floor']['best_possible_adjusted_p']} are computed "
       f"from THESE bytes")
    # WHY NOT THE HEAD -- measured on a SCRATCH chain, never the real one.
    import declaration_chain as _dc
    _hd = Path(_tfP.mkdtemp(prefix="be91_head_"))
    _f1 = {"per_day": {"20990101": {"exists": True}, "20990102": {"exists": True},
                       "20990103": {"exists": False}}, "supersedes": None}
    (_hd / "fx_pins_v1.json").write_text(_js.dumps(_f1, indent=1, sort_keys=True))
    _p1 = {"path": str(_hd / "fx_pins_v1.json"),
           "sha256": _hl.sha256((_hd / "fx_pins_v1.json").read_bytes()).hexdigest()}
    _f2 = {"per_day": dict(_f1["per_day"], **{"20990103": {"exists": True}}),
           "supersedes": _p1}
    (_hd / "fx_pins_v2.json").write_text(_js.dumps(_f2, indent=1, sort_keys=True))
    _head = _dc.resolve_head(_hd, "fx_pins")
    _n_pair = sum(1 for v in _f1["per_day"].values() if v.get("exists"))
    _n_head = sum(1 for v in json.loads(Path(_head["path"]).read_text())
                  ["per_day"].values() if v.get("exists"))
    ok(_head["name"] == "fx_pins_v2.json" and _n_pair == 2 and _n_head == 3
       and (2 / 2 ** _n_pair) != (2 / 2 ** _n_head),
       f"AND THE HEAD IS THE WRONG RESOLUTION HERE, COMPUTED ON A SCRATCH "
       f"CHAIN: the pinned version has {_n_pair} readable days (floor "
       f"{2 / 2 ** _n_pair}) and the HEAD has {_n_head} (floor "
       f"{2 / 2 ** _n_head}). A pins family GROWS one version per day close, "
       f"so resolving the head would restate a LANDED declaration's G and "
       f"floor from a population it never saw -- which is why R-729 sends a "
       f"reader of history to the pair and only writers to the head. The "
       f"real chain is not touched by this cell")
    try:
        build_v4(_pd); _seam = "NOT REFUSED"
    except StaleHistoryPin as _e:
        _seam = str(_e).split(":")[0]
    ok(_seam == "HISTORY_PIN_MOVED",
       f"AND THE SEAM, NOT JUST THE HELPER: `build_v4` ITSELF refuses "
       f"({_seam}) when the pins bytes have moved. A cell that drove only "
       f"`pins_as_v4_recorded` would pass with the builder still calling "
       f"`json.loads` on the bare filename -- the unreachable duplicate "
       f"REV 84 §3.1 names. Before this round `build_v4(_pd)` would have "
       f"read those bytes and rebuilt a LANDED declaration from them")

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
    if "--declare-v4" in argv:
        out = build_v4()
        dst = HERE / "declarations" / "be_race_read_declaration_v4.json"
        dst.write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
        print(json.dumps({"written": str(dst), "G": out["G"],
                          "floor": out["permutation_floor"][
                              "best_possible_adjusted_p"],
                          "unrecoverable": out["population"][
                              "READ_BUT_UNRECOVERABLE"]}))
        return 0
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
