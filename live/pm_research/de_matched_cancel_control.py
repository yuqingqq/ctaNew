#!/usr/bin/env python3
"""The matched control for the DAY RUN, matched on CANCELS — USER ruling B.

WHAT WAS RULED AND WHY. DE 155 (1) re-timed the decision: each scored row
is emitted at its own `t_start` and the FIRST crossing of theta cancels.
`bk["rows"]` is also the null's sampling unit, so per-row rows changed what
a draw draws while the policy still issues ONE CANCEL PER GENERATION -- and
"matched on the action count" stopped naming one quantity
(`drafts/DE158_null_sampling_unit_QUESTION.md`, three options, none chosen
by DE). **The USER ruled B: match on CANCELS.** Under the ruled rule a
generation yields at most one cancel, so the cancel IS the action, and B is
the only option that matches on the DECISION variable (rule 7).

THE CONTRACT IS NOT INVENTED HERE. `de_matched_random_control` already
declares it for the Phase-4 lane -- "an ACTING arm that cancels
generations chosen UNIFORMLY AT RANDOM inside each (side, hour) stratum,
where the number it cancels in a stratum is DETERMINED BY THE TREATED ARM's
action count and is never a caller-chosen number" -- with four rules, each
a refusal. This module applies that same contract at the DAY RUN's
granularity, which is the GENERATION rather than the slug (a day has many
generations per slug), and CITES it rather than restating it.

THE COST THE USER ACCEPTED, AND WHAT REPLACES IT. Matching on cancels makes
the draw data-dependent, so a fixed seed no longer reproduces a draw the
way it does when the draw size is a constant. **The reproduction property
is not abandoned, it is MOVED: the drawn control set is PERSISTED as an
artifact and every replay reproduces FROM THE ARTIFACT, not from a seed.**
The battery proves that element by element, with a can-fail control -- a
different persisted set must NOT reproduce, or the comparison establishes
nothing.

RULE 6, DECLARED HERE AND BEFORE ANY RESULT: the design is above, and the
minimum sample is `MIN_DRAWS = 200`. A draw count below it REFUSES. An
under-sampled correct null flatters as much as a wrong one.

WHAT THIS MODULE DOES NOT DO. It does not value anything and it does not
choose a metric. The comparison is on the DECISION metric -- net value and
rho = adverse/spread -- never a proxy like harm share (rule 7); that lives
with the estimator, and this module only says WHICH generations the control
cancels and WHEN.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROTOCOL = "P003_DE_MATCHED_CANCEL_CONTROL_V1"
#: Rule 6: the minimum sample, declared before any result.
MIN_DRAWS = 200
CONTRACT_CITED = "de_matched_random_control (R-459 §3(iii))"

DEMAND_SUPPLIED = "MATCHED_CONTROL_DEMAND_WAS_SUPPLIED_NOT_READ"
STRATUM_TOO_SMALL = "MATCHED_CONTROL_STRATUM_SMALLER_THAN_DEMAND"
NOT_RANDOM = "MATCHED_CONTROL_REPRODUCES_THE_TREATED_ARM"
NO_HOUR = "MATCHED_CONTROL_SLUG_CARRIES_NO_HOUR"
TOO_FEW_DRAWS = "MATCHED_CONTROL_BELOW_DECLARED_MIN_DRAWS"
ARTIFACT_DIGEST = "MATCHED_CONTROL_ARTIFACT_DIGEST_MISMATCH"
#: DE 166 (3), the USER's finding. Ruling B was recommended on the premise
#: that "under the ruled first-crossing rule a generation yields at most
#: one cancel, so the cancel IS the action". **The premise does not hold
#: as stated.** Under the production repost settings
#: (`theta_repost = theta_cancel / 2`, `repost_dwell_s = 2.0`) a side that
#: cancels, holds, becomes repost-eligible and re-places on the SAME
#: reference generation can cancel it AGAIN -- and
#: `harmful_stateful_policy.check_invariants`' `one_cancel_per_generation`
#: DID NOT SEE IT, because it keys on the POLICY generation (`7`, `7.r1`)
#: and not on the reference one. REV 110 and the coordinator both cited
#: that invariant as proof the cancel is the action; it was checking the
#: wrong id space.
#:
#: THE CONSEQUENCE FOR THIS MODULE, driven both ways (see the battery):
#:   * `demand_from_arm` counts CANCELS, and `draw_one` samples DISTINCT
#:     reference generations without replacement -- so when the arm
#:     cancels one generation twice the demand is 2 and the control draws
#:     TWO generations. The control acts on a population the arm did not.
#:   * If the stratum happens to be saturated, `draw_one` refuses
#:     STRATUM_TOO_SMALL -- loud, but for the WRONG REASON: it reads as
#:     "the stratum is too small" when the truth is "the demand counts a
#:     different unit than the pool supplies".
#:   * `assert_random_wrt_arm` compares SETS of reference generations, so
#:     it collapses the duplicate and cannot see it either.
#: **The matching unit is the USER's to rule and is NOT re-ruled here.**
#: What this module does is refuse to draw a control whose premise it can
#: see is violated, so a day cannot quietly produce a mis-matched null
#: while the ruling is pending.
PREMISE_VIOLATED = "MATCHED_CONTROL_PREMISE_ONE_CANCEL_PER_GENERATION_FALSE"


def cancels_per_reference_generation(arm_cancels) -> dict:
    """THE MEASURED RELATION between cancels and REFERENCE generations.

    Reported whether or not it violates anything, because "how many
    cancels does a reference generation get" is the question ruling B's
    premise answers with "at most one" and nobody had measured."""
    per: dict = {}
    for c in arm_cancels:
        k = (c["slug"], c["side"], int(c["ref_gen"]))
        per[k] = per.get(k, 0) + 1
    multi = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in per.items() if v > 1}
    return {
        "n_cancels": sum(per.values()),
        "n_reference_generations": len(per),
        "max_cancels_on_one_reference_generation": (max(per.values())
                                                    if per else 0),
        "n_reference_generations_cancelled_more_than_once": len(multi),
        "which": dict(sorted(multi.items())[:20]),
        "premise_holds": not multi,
        "premise": ("USER ruling B was recommended on 'a generation yields "
                    "at most one cancel, so the cancel IS the action'"),
        "why_the_invariant_missed_it": (
            "`check_invariants.one_cancel_per_generation` keys on "
            "(slug, side, POLICY_gen) -- `7` and `7.r1` are two keys with "
            "one cancel each -- while this counts (slug, side, REF_gen)"),
    }


class MatchedControlRefused(RuntimeError):
    """Refuses rather than drawing a control it cannot defend."""


_EPOCH = re.compile(r"(\d{9,})$")


def hour_of(slug: str) -> int:
    """The slug's hour stratum, PARSED from its own epoch suffix.

    A slug that carries no epoch REFUSES: an unstratifiable generation
    silently pooled with every other hour is a control matched on less
    than rule 7 requires."""
    m = _EPOCH.search(str(slug))
    if not m:
        raise MatchedControlRefused(
            f"{NO_HOUR}: {slug!r} carries no epoch suffix, so its (side, "
            f"hour) stratum cannot be formed. Rule 7 matches on side AND "
            f"hour; pooling it across hours would match on less.")
    return int(m.group(1)) // 3600


def build_pool(scored: dict) -> dict:
    """(side, hour) -> the generations eligible to be cancelled, each with
    the row times at which it COULD be cancelled.

    Eligibility is having a scored row: a generation the feature pass
    dropped cannot be cancelled by the arm either, so including it would
    give the control a population the arm never had."""
    pool: dict = {}
    for (slug, side, t), v in scored.items():
        st = (side, hour_of(slug))
        key = (slug, side, int(v["gen"]))
        pool.setdefault(st, {}).setdefault(key, []).append(float(t))
    for st in pool:
        for k in pool[st]:
            pool[st][k].sort()
    return pool


def build_pool_from_rows(rows) -> dict:
    """The same pool, built from the book's DECISION ROWS.

    Since BE 107 `bk["rows"]` is per scored row and carries slug, side,
    gen and t -- everything a stratum and a cancel time need -- so the
    day-run path builds the pool from the very stream the arm decided
    over, rather than from a second view of it."""
    pool: dict = {}
    for r in rows:
        st = (r["side"], hour_of(r["slug"]))
        key = (r["slug"], r["side"], int(r["gen"]))
        pool.setdefault(st, {}).setdefault(key, []).append(float(r["t"]))
    for st in pool:
        for k in pool[st]:
            pool[st][k].sort()
    return pool


def demand_from_arm(arm_cancels) -> dict:
    """(side, hour) -> how many cancels the control owes, READ OFF the arm.

    `arm_cancels` is the treated arm's own cancel records. A caller-chosen
    count is refused by `draw` -- a control whose size is chosen is a
    control that can be tuned (the cited contract's first rule)."""
    # ---- DE 166 (3): THE PREMISE IS CHECKED, NOT ASSUMED --------------
    rel = cancels_per_reference_generation(arm_cancels)
    if not rel["premise_holds"]:
        raise MatchedControlRefused(
            f"{PREMISE_VIOLATED}: {rel['n_cancels']} cancels fall on "
            f"{rel['n_reference_generations']} reference generations, and "
            f"{rel['n_reference_generations_cancelled_more_than_once']} of "
            f"them carry more than one (up to "
            f"{rel['max_cancels_on_one_reference_generation']}; "
            f"{rel['which']}). Under repost the SAME reference generation "
            f"is re-placed as a new POLICY generation and can be cancelled "
            f"again -- `check_invariants` keys on the policy id and does "
            f"not see it. This demand counts CANCELS while `draw_one` "
            f"samples DISTINCT reference generations, so the control would "
            f"act on more generations than the arm did. REFUSED rather "
            f"than drawn: the matching unit is the USER's to rule "
            f"(DE 166 (3)) and a control drawn on a premise known to be "
            f"false is not a control.")
    d: dict = {}
    for c in arm_cancels:
        st = (c["side"], hour_of(c["slug"]))
        d[st] = d.get(st, 0) + 1
    return d


def draw_one(pool: dict, demand: dict, rng) -> list:
    """ONE control draw: which generations it cancels, and WHEN.

    The generation is drawn uniformly without replacement inside its
    stratum; the row is then drawn uniformly from that generation's own
    scored rows, so the control's cancel TIMES are random within the
    generation rather than pinned to its start. Pinning them to `t0` would
    give the control a systematically earlier cancel than the arm's first
    crossing and would be a difference in timing, not in selection."""
    out = []
    for st in sorted(demand):
        avail = sorted(pool.get(st, {}))
        want = demand[st]
        if want > len(avail):
            raise MatchedControlRefused(
                f"{STRATUM_TOO_SMALL}: stratum {st} owes {want} cancels "
                f"and has {len(avail)} eligible generations. REFUSED, "
                f"never clamped -- a clamp silently answers an easier "
                f"question than the one the treated arm poses.")
        for key in rng.sample(avail, want):
            times = pool[st][key]
            out.append({"slug": key[0], "side": key[1], "gen": key[2],
                        "t": times[rng.randrange(len(times))]})
    return sorted(out, key=lambda a: (a["t"], a["slug"], a["side"], a["gen"]))


def draw_many(pool: dict, demand: dict, *, n_draws: int, seed: int) -> list:
    """`n_draws` control sets. Rule 6's minimum is enforced HERE."""
    if n_draws < MIN_DRAWS:
        raise MatchedControlRefused(
            f"{TOO_FEW_DRAWS}: {n_draws} draws against a declared minimum "
            f"of {MIN_DRAWS}. A null max is an extreme-order statistic; an "
            f"under-sampled correct null flatters as much as a wrong one "
            f"(rule 6). The cap is never lowered (R-174).")
    rng = random.Random(seed)
    return [draw_one(pool, demand, rng) for _ in range(n_draws)]


def assert_random_wrt_arm(draws: list, arm_cancels) -> dict:
    """A control that reproduces the treated arm's own actions is refused
    BY IDENTITY -- it is the arm under test wearing the null's name."""
    arm = {(c["slug"], c["side"], int(c["ref_gen"])) for c in arm_cancels}
    same = [i for i, d in enumerate(draws)
            if {(a["slug"], a["side"], a["gen"]) for a in d} == arm]
    if same:
        raise MatchedControlRefused(
            f"{NOT_RANDOM}: draw(s) {same[:5]} select exactly the "
            f"generations the treated arm cancelled. With a real pool that "
            f"is astronomically unlikely and means the draw is not random "
            f"with respect to the arm.")
    return {"n_draws": len(draws), "n_identical_to_the_arm": 0,
            "why": "a control that reproduces the arm is the arm"}


# --------------------------------------------------- the artifact

def write_control_set(path, draws: list, *, day: str, arm: str,
                      demand: dict, seed: int) -> dict:
    """PERSIST THE DRAWN SET. This is what replaces seed reproducibility.

    Matching on cancels makes the draw data-dependent, so the seed is no
    longer sufficient to reproduce it. The bytes are, and they are what
    every later replay reads."""
    path = Path(path)
    if path.exists():
        raise MatchedControlRefused(
            f"control set already exists: {path} -- a drawn control is "
            f"never overwritten.")
    with gzip.open(path, "wt") as fh:
        fh.write(json.dumps({
            "row": "HEADER", "protocol": PROTOCOL, "day": day, "arm": arm,
            "n_draws": len(draws), "min_draws_declared": MIN_DRAWS,
            "seed_that_produced_it": seed,
            "seed_is_NOT_sufficient_to_reproduce": (
                "the draw is data-dependent because it is matched on "
                "CANCELS (USER ruling B). These BYTES are the "
                "reproduction, not the seed."),
            "demand_by_stratum": {f"{s}|{h}": n for (s, h), n in
                                  sorted(demand.items())},
            "contract": CONTRACT_CITED}, sort_keys=True) + "\n")
        for i, d in enumerate(draws):
            fh.write(json.dumps({"row": "DRAW", "i": i, "cancels": d},
                                sort_keys=True) + "\n")
    return {"path": str(path), "sha256": _sha(path), "n_draws": len(draws)}


def read_control_set(path, *, expect_sha256: str | None = None) -> dict:
    path = Path(path)
    got = _sha(path)
    if expect_sha256 and got != expect_sha256:
        raise MatchedControlRefused(
            f"{ARTIFACT_DIGEST}: {path.name} digests {got[:16]} and the "
            f"caller named {expect_sha256[:16]}. The control this replay "
            f"claims to reproduce is not the control on disk.")
    header, draws = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["row"] == "HEADER":
                header = r
            else:
                draws.append(r["cancels"])
    return {"header": header, "draws": draws, "sha256": got}


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flags_for(draw: list, row_index: dict) -> list:
    """The row indices `flagged_stream` must flag for ONE control draw."""
    out = []
    for a in draw:
        k = (a["slug"], a["side"], float(a["t"]))
        if k not in row_index:
            raise MatchedControlRefused(
                f"the control names a row {k} that is not in the book's "
                f"decision stream; the control and the book disagree.")
        out.append(row_index[k])
    return sorted(out)


# ------------------------------------------------------- the battery

EXPECTED_CHECKS = 9


def selftest(quiet: bool = False) -> int:
    import tempfile
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_matched_cancel_control] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    SIDE = "BUY_UP"
    base = 1788480000
    scored = {}
    for s in range(12):
        slug = f"btc-updown-5m-{base + s * 300}"
        for g in range(3):
            for r in range(2):
                scored[(slug, SIDE, float(g * 10 + r))] = {
                    "score": 0.1, "gen": g, "t0": float(g * 10)}
    pool = build_pool(scored)
    arm = [{"slug": f"btc-updown-5m-{base}", "side": SIDE, "ref_gen": 0},
           {"slug": f"btc-updown-5m-{base + 300}", "side": SIDE,
            "ref_gen": 1},
           {"slug": f"btc-updown-5m-{base + 600}", "side": SIDE,
            "ref_gen": 2}]
    demand = demand_from_arm(arm)
    ok(sum(demand.values()) == 3 and len(pool) >= 1
       and all(isinstance(k, tuple) and len(k) == 2 for k in pool),
       f"THE DEMAND IS READ OFF THE ARM, NEVER SUPPLIED: the treated arm "
       f"cancelled {sum(demand.values())} generations and the demand is "
       f"exactly that, per (side, hour) stratum {sorted(demand)}")

    draws = draw_many(pool, demand, n_draws=MIN_DRAWS, seed=11)
    ok(len(draws) == MIN_DRAWS
       and all(len(d) == sum(demand.values()) for d in draws)
       and all(len({(a["slug"], a["side"], a["gen"]) for a in d}) == len(d)
               for d in draws),
       f"MATCHED ON CANCELS: every one of {len(draws)} draws selects "
       f"exactly {sum(demand.values())} DISTINCT generations, which is the "
       f"arm's cancel count on THIS fixture -- where each cancel does fall "
       f"on its own reference generation. **THE SENTENCE THAT USED TO END "
       f"THIS CELL -- 'under the ruled first-crossing rule a generation "
       f"yields at most one cancel, so the cancel IS the action' -- IS "
       f"WITHDRAWN: it is false under repost, and the next cell drives it "
       f"on the real engine (DE 166 (3), the USER's finding).**")


    # ---- DE 166 (3): THE PREMISE, DRIVEN ON THE REAL ENGINE -----------
    # Ruling B was recommended on "a generation yields at most one cancel".
    # This does not argue about it: it runs `harmful_stateful_policy` with
    # the PRODUCTION repost shape (theta_repost = theta_cancel / 2,
    # repost_dwell_s = 2.0 -- `de_phase4_diag_runner.cell_params`) and
    # reads what comes out.
    import harmful_stateful_policy as _H166
    _S166 = _H166.SIDES[0]
    _sl166 = f"btc-updown-5m-{base}"
    _ref166 = {_sl166: {_S166: [_H166._gen(7, 0.0, 40.0,
                                           [(3.0, 1.0, -5.0),
                                            (25.0, 1.0, -9.0)])],
                        _H166.SIDES[1]: []}}
    _sc166 = [{"t": t, "slug": _sl166, "side": _S166, "gen": 7, "score": v}
              for t, v in ((1.0, 0.99), (4.0, 0.0), (10.0, 0.0),
                           (14.0, 0.0), (20.0, 0.99), (30.0, 0.0))]
    _p166 = _H166._params(theta_cancel=0.8, theta_repost=0.4,
                          repost_dwell_s=2.0,
                          cancel_effective_latency_ms=1000.0,
                          protection_mode="ALL_ORDERS_OVERRIDE")
    _out166 = _H166.replay_policy(_ref166, _sc166, _p166)
    _inv166 = _H166.check_invariants(_out166)
    _rel166 = cancels_per_reference_generation(_out166["cancels"])
    ok(_rel166["n_cancels"] == 2
       and _rel166["n_reference_generations"] == 1
       and _rel166["max_cancels_on_one_reference_generation"] == 2
       and _rel166["premise_holds"] is False
       and _inv166["one_cancel_per_generation"] is True
       and all(_inv166.values())
       and sorted(c["policy_gen"] for c in _out166["cancels"]) == ["7", "7.r1"],
       f"DE 166 (3) THE PREMISE IS FALSE, AND THE INVARIANT CANNOT SEE IT: "
       f"one reference generation takes "
       f"{_rel166['max_cancels_on_one_reference_generation']} cancels "
       f"({[c['policy_gen'] for c in _out166['cancels']]}) while "
       f"`one_cancel_per_generation` reads "
       f"{_inv166['one_cancel_per_generation']} and EVERY invariant holds "
       f"-- because it keys on the POLICY generation and the two reposts "
       f"are two keys with one cancel each. The mechanism is the ordinary "
       f"one: cancel, hold, dwell below theta_repost, re-place on the same "
       f"reference generation, cross again")
    # AND WHAT IT DOES TO THE CONTROL, both branches, measured.
    _rows166 = [{"t": e["t"], "slug": e["slug"], "side": e["side"],
                 "gen": e["gen"]} for e in _sc166]
    _pool166 = build_pool_from_rows(_rows166)
    _sat = None
    try:
        demand_from_arm(_out166["cancels"])
        _sat = "ADMITTED"
    except MatchedControlRefused as _e:
        _sat = str(_e).split(":")[0]
    # the SILENT branch: the same arm in a stratum with room. Built by
    # hand from the driven cancels so the control's own arithmetic -- not
    # the engine -- is what this half measures.
    _wide = {**_pool166}
    _st166 = (_S166, hour_of(_sl166))
    for _i in range(1, 6):
        _wide[_st166][(f"btc-updown-5m-{base + 300 * _i}", _S166, 7)] = [1.0]
    _demand_raw = {}
    for _c in _out166["cancels"]:
        _k = (_c["side"], hour_of(_c["slug"]))
        _demand_raw[_k] = _demand_raw.get(_k, 0) + 1
    _dr = draw_many(_wide, _demand_raw, n_draws=MIN_DRAWS, seed=3)
    _ctrl_gens = {len({(a["slug"], a["side"], a["gen"]) for a in d})
                  for d in _dr}
    ok(_sat == PREMISE_VIOLATED
       and _demand_raw == {_st166: 2}
       and _ctrl_gens == {2}
       and _rel166["n_reference_generations"] == 1,
       f"AND THE CONTROL WOULD HAVE ACTED ON A POPULATION THE ARM DID NOT: "
       f"the raw demand is {list(_demand_raw.values())[0]} (it counts "
       f"CANCELS) and every draw selects {sorted(_ctrl_gens)[0]} DISTINCT "
       f"reference generations, where the arm acted on "
       f"{_rel166['n_reference_generations']}. Matched on the COUNT, not "
       f"on the UNIT. `demand_from_arm` now refuses it by name "
       f"(`{_sat}`) instead of drawing it -- and in a SATURATED stratum "
       f"the old code refused `{STRATUM_TOO_SMALL}`, which is loud for the "
       f"WRONG REASON: it reads as 'the stratum is too small' when the "
       f"truth is 'the demand counts a different unit than the pool "
       f"supplies'. THE UNIT IS THE USER'S TO RULE; this refuses in the "
       f"meantime rather than drawing a control it cannot defend")
    # POSITIVE CONTROL: the honest arm still draws.
    ok(sum(demand_from_arm(arm).values()) == 3
       and cancels_per_reference_generation(arm)["premise_holds"] is True,
       f"POSITIVE CONTROL: an arm whose cancels fall one per reference "
       f"generation still passes and still draws -- the refusal above is "
       f"about the duplicate and not about the module having been made "
       f"unable to build a demand")

    # ---- (2) REPRODUCTION FROM THE ARTIFACT, ELEMENT BY ELEMENT --------
    d = Path(tempfile.mkdtemp(prefix="mcc_"))
    w = write_control_set(d / "cs.jsonl.gz", draws, day="2026-09-03",
                          arm="A", demand=demand, seed=11)
    back = read_control_set(d / "cs.jsonl.gz", expect_sha256=w["sha256"])
    ok(back["draws"] == draws and back["header"]["n_draws"] == MIN_DRAWS
       and back["header"]["min_draws_declared"] == MIN_DRAWS,
       f"THE ARTIFACT REPRODUCES THE DRAW ELEMENT BY ELEMENT: all "
       f"{len(draws)} sets, every generation and every cancel TIME, read "
       f"back identical. This is what replaces seed reproducibility -- "
       f"matching on cancels makes the draw data-dependent, so the BYTES "
       f"are the reproduction and the seed is not")

    # ---- AND THE CAN-FAIL CONTROL -------------------------------------
    other = draw_many(pool, demand, n_draws=MIN_DRAWS, seed=12)
    w2 = write_control_set(d / "cs2.jsonl.gz", other, day="2026-09-03",
                           arm="A", demand=demand, seed=12)
    back2 = read_control_set(d / "cs2.jsonl.gz")
    dig = None
    try:
        read_control_set(d / "cs2.jsonl.gz", expect_sha256=w["sha256"])
    except MatchedControlRefused as e:
        dig = str(e).split(":")[0]
    ok(back2["draws"] != draws and w2["sha256"] != w["sha256"]
       and dig == ARTIFACT_DIGEST,
       f"AND THE COMPARISON CAN FAIL: a DIFFERENT drawn set does NOT read "
       f"back as the first ({w2['sha256'][:12]} against "
       f"{w['sha256'][:12]}), and reading it while naming the first's "
       f"digest refuses `{dig}`. Without this the reproduction above "
       f"would establish nothing")

    # ---- THE FOUR REFUSALS OF THE CITED CONTRACT ----------------------
    small = None
    try:
        draw_one({("BUY_UP", hour_of(f"btc-updown-5m-{base}")): {
            ("s", SIDE, 0): [0.0]}}, {("BUY_UP", hour_of(
                f"btc-updown-5m-{base}")): 5}, random.Random(1))
    except MatchedControlRefused as e:
        small = str(e).split(":")[0]
    few = None
    try:
        draw_many(pool, demand, n_draws=MIN_DRAWS - 1, seed=1)
    except MatchedControlRefused as e:
        few = str(e).split(":")[0]
    nohour = None
    try:
        hour_of("btc-updown-5m-nope")
    except MatchedControlRefused as e:
        nohour = str(e).split(":")[0]
    ok(small == STRATUM_TOO_SMALL and few == TOO_FEW_DRAWS
       and nohour == NO_HOUR,
       f"THE CONTRACT'S REFUSALS: a stratum smaller than its demand "
       f"refuses `{small}` and is NEVER clamped -- a clamp answers an "
       f"easier question than the arm poses; a draw count below the "
       f"declared minimum of {MIN_DRAWS} refuses `{few}` (rule 6); and a "
       f"slug with no epoch refuses `{nohour}` rather than being pooled "
       f"across hours, which would match on less than rule 7 requires")

    # ---- THE CONTROL MAY NOT BE THE ARM -------------------------------
    armset = [{"slug": a["slug"], "side": a["side"], "gen": a["ref_gen"],
               "t": 0.0} for a in arm]
    notrand = None
    try:
        assert_random_wrt_arm([armset], arm)
    except MatchedControlRefused as e:
        notrand = str(e).split(":")[0]
    okrand = assert_random_wrt_arm(draws, arm)
    ok(notrand == NOT_RANDOM and okrand["n_identical_to_the_arm"] == 0,
       f"AND A CONTROL THAT REPRODUCES THE ARM IS REFUSED BY IDENTITY "
       f"(`{notrand}`) -- it is the arm under test wearing the null's "
       f"name -- while {okrand['n_draws']} real draws select none of them")

    import shutil
    shutil.rmtree(d, ignore_errors=True)
    if n[0] != EXPECTED_CHECKS:
        raise SystemExit(f"[de_matched_cancel_control] FAIL: check count "
                         f"{n[0]} != {EXPECTED_CHECKS}")
    if not quiet:
        print(f"[de_matched_cancel_control] PASS -- {n[0]} checks, "
              f"n_disarmed 0, n_skipped 0")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
