#!/usr/bin/env python3
"""THE DECLARED CELL FAMILY, AND THE REFUSAL THAT KEEPS IT HONEST.

WHY A MODULE FOR A COUNT. Holm's denominator is the multiplicity argument.
Iteration 011 turned on 0.0479 against 0.1199 -- the difference between a
surviving set and none -- and the denominator is the one input a reader cannot
recompute from the reported cells, because cells that were never reported are
invisible. R-497 (F)(4) rules BOTH pairing conventions reported, and states the
cost: the family grows and the denominator grows with it. **That cost is
declared here, as an integer, in a committed artifact, before any forward
score exists**, and `require_declared_count` REFUSES when a run would report a
different number of cells than it declared.

THE COUNT IS ENUMERATED, NOT MULTIPLIED. A product of factors is how a family
silently gains or loses a dimension; this module lists the cell identities and
takes `len()`. That is also how it discovered that the ruling's word "DOUBLES"
does not describe this family: the by-COUNT convention has NO operating point
-- kk determines its cutoff -- so it does not twin the sensitivity arm, and
the growth is 1.5x rather than 2x. Reported as a computed disagreement with
the framing, never silently reconciled to it (rule 10).

TWO FACTORS ARE NOT MINE AND ARE MARKED OPEN. Whether the sensitivity
operating-point arm consumes alpha, and whether the forward read covers both
coins of the freeze or btc alone as iteration 011 did, each change the integer.
Both are enumerated here with the count under every alternative computed, so a
USER ruling is one word and the artifact already carries its consequence.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import be_forward_metric as FM
import harmful_forward_scorer as FS
import phase2_declaration as PD

REPO = Path("/home/yuqing/ctaNew")


class FamilyRefused(RuntimeError):
    """A named refusal."""


#: The operating-point arms the ruling names. PRIMARY is adjudicated;
#: SENSITIVITY is reported beside it and never selected.
OPERATING_POINT_ARMS = ("FROZEN_FROM_TRAIN_QUANTILE",
                        "FROZEN_FROM_A_CONSUMED_DAY")

#: Which conventions carry an operating point at all. by-COUNT does not: its
#: cutoff is kk, so pairing it with a theta arm would enumerate cells that
#: cannot differ. Computed against the metric module's own registry below.
CONVENTION_TAKES_OPERATING_POINT = {"BY_THRESHOLD": True, "BY_COUNT": False}

#: Factors this module cannot rule. Each carries the alternatives so the
#: consequence of either answer is already computed in the artifact.
OPEN_FACTORS = {
    "sensitivity_arm_in_family": {
        "question": ("does the FROZEN_FROM_A_CONSUMED_DAY sensitivity arm "
                     "consume alpha, or is it reported outside the family?"),
        "alternatives": [True, False],
        "who": "the USER (rule 14)",
        # RULED 2026-09-04. Kept as an OPEN_FACTORS entry rather than deleted
        # because the alternatives and their counts are what make the ruling
        # legible -- a reader must be able to see what was decided AGAINST.
        "STATUS": "RULED — the arm stays IN the family at 18 cells",
        "ruled_by": "USER, 2026-09-04, relayed in the dispatch of BE round 38",
        "ruled_what": (
            "the operating-point fence stays pinned to ONE canonical "
            "verification artifact, so no FROZEN_FROM_A_CONSUMED_DAY "
            "operating point can pass it and the arm's six cells stay "
            "permanently NOT COMPUTED"),
        "consequence_for_multiplicity": (
            "the denominator of 18 KNOWINGLY carries six cells that cannot be "
            "filled. Every Holm correction against it is therefore "
            "CONSERVATIVE -- it divides alpha further than the reportable "
            "cells require -- and it is NOT wrong. This was DECIDED, not "
            "overlooked"),
        "the_arm_is_built_and_inert": (
            "the operating point EXISTS at declarations/"
            "be_operating_point_declaration_SENSITIVITY_v1.json, derived from "
            "08-29's consumed feed. It is committed and INERT: the fence "
            "refuses it by name. If the USER ever reverses the pin ruling the "
            "family completes IMMEDIATELY, with no rebuild and no new day"),
        "why_the_pin_was_not_widened": (
            "widening the admissibility guard to complete the family that "
            "needs it is the move the pin exists to prevent; BE declined to "
            "take it and the USER agreed"),
        "note": ("R-497 (F)(2) calls it a declared sensitivity arm and says "
                 "all cells are reported with Holm over the declared count; "
                 "whether 'all cells' includes a never-selected arm is not "
                 "settled by that sentence."),
    },
    "coins": {
        "question": ("does the forward read cover both coins the freeze "
                     "carries fits for, or btc alone as iteration 011 did?"),
        "alternatives": [["btc", "eth"], ["btc"]],
        "who": "the USER (rule 14)",
        "note": ("the frozen candidate carries fits for btc and eth only; "
                 "windows of the other five coins are supplied, replayed and "
                 "counted and produce NO score (measured 1,269 on 09-02)."),
    },
}


def frozen_coins() -> list:
    """The coins the freeze actually covers — READ from the artifact."""
    return sorted(json.loads(Path(FS.CANDIDATE).read_text())["fits"])


def enumerate_family(coins, budgets=None, sensitivity_arm_in_family=True,
                     conventions=None) -> list:
    """The cell identities, listed. `len()` of this is the declared count.

    Enumerated rather than multiplied, because a product cannot express that
    one convention has no operating-point dimension."""
    budgets = tuple(budgets if budgets is not None else PD.BUDGETS)
    conventions = tuple(conventions if conventions is not None
                        else sorted(FM.PAIRING_CONVENTIONS))
    arms = (OPERATING_POINT_ARMS if sensitivity_arm_in_family
            else OPERATING_POINT_ARMS[:1])
    cells = []
    for conv in conventions:
        if CONVENTION_TAKES_OPERATING_POINT.get(conv, True):
            for coin, b, arm in itertools.product(coins, budgets, arms):
                cells.append(f"{conv}/{arm}/{coin}/{int(b * 100)}%")
        else:
            for coin, b in itertools.product(coins, budgets):
                cells.append(f"{conv}/NO_OPERATING_POINT/{coin}/"
                             f"{int(b * 100)}%")
    return sorted(cells)


def _git(*a):
    r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                       text=True, timeout=30)
    return r.stdout.strip() if r.returncode == 0 else None


def declare(coins=None, sensitivity_arm_in_family=True) -> dict:
    """The declaration artifact. Carries the integer AND its enumeration.

    Also carries the count under every alternative of every OPEN factor, so a
    ruling does not require a new round to price."""
    coins = list(coins if coins is not None else frozen_coins())
    cells = enumerate_family(coins, sensitivity_arm_in_family=sensitivity_arm_in_family)
    by_conv: dict = {}
    for c in cells:
        by_conv[c.split("/")[0]] = by_conv.get(c.split("/")[0], 0) + 1
    single = enumerate_family(coins,
                              sensitivity_arm_in_family=sensitivity_arm_in_family,
                              conventions=("BY_THRESHOLD",))
    alternatives = {}
    for sens in OPEN_FACTORS["sensitivity_arm_in_family"]["alternatives"]:
        for cs in OPEN_FACTORS["coins"]["alternatives"]:
            k = f"sensitivity_arm_in_family={sens},coins={','.join(cs)}"
            alternatives[k] = len(enumerate_family(
                cs, sensitivity_arm_in_family=sens))
    return {
        "protocol": "BE_FORWARD_FAMILY_DECLARATION_V1",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "ruling": "R-497 (F)(4): BOTH pairing conventions reported",
        "declared_cell_count": len(cells),
        "cells": cells,
        "cells_by_convention": dict(sorted(by_conv.items())),
        "factors": {
            "coins": coins,
            "coins_source": "the frozen candidate's own `fits` keys",
            "budgets": [f"{int(b * 100)}%" for b in PD.BUDGETS],
            "budgets_source": ("phase2_declaration.BUDGETS, cited by sha "
                               "rather than retyped"),
            "budgets_module_sha256_prefix": hashlib.sha256(
                Path(PD.__file__).read_bytes()).hexdigest()[:16],
            "operating_point_arms": list(
                OPERATING_POINT_ARMS if sensitivity_arm_in_family
                else OPERATING_POINT_ARMS[:1]),
            "conventions": sorted(FM.PAIRING_CONVENTIONS),
            "convention_takes_operating_point":
                dict(CONVENTION_TAKES_OPERATING_POINT),
        },
        "growth_vs_single_convention": {
            "single_convention_count": len(single),
            "both_conventions_count": len(cells),
            "ratio": len(cells) / len(single) if single else None,
            "ruling_word": "DOUBLES",
            "ratio_is_2x": (len(cells) == 2 * len(single)),
            "why_not": ("BY_COUNT carries NO operating point -- kk sets its "
                        "cutoff -- so it does not twin the sensitivity arm. "
                        "Computed and reported as a disagreement with the "
                        "framing rather than reconciled to it (rule 10)."),
        },
        "open_factors": OPEN_FACTORS,
        "count_under_each_alternative": alternatives,
        "holm_denominator": len(cells),
        "holm_note": ("the denominator is the multiplicity argument and is "
                      "the one input a reader cannot recompute from the "
                      "reported cells, because unreported cells are "
                      "invisible. Declared here before any forward score."),
        # NAMED FOR WHAT IT IS: the tree this artifact was GENERATED from.
        # The commit that CARRIES it is necessarily a child of this one, and a
        # reader wanting that asks git for the file's history rather than
        # trusting a field that could not have known it.
        "generated_at_head": _git("rev-parse", "HEAD"),
        "carrying_commit_note": ("the commit that carries this file is a "
                                 "descendant of generated_at_head; ask git "
                                 "for this path's log rather than reading it "
                                 "from a field written before the commit "
                                 "existed"),
        "selects_nothing": True,
    }


#: Where the committed declaration lives. IN THE REPOSITORY, not under
#: `data/`: the count is a protocol fact that must travel with the code and be
#: readable at any commit, and `data/` is gitignored.
DECLARATION_PATH = (Path(__file__).resolve().parent / "declarations"
                    / "be_forward_family_declaration_v1.json")


def load_declaration(path: Path = None) -> dict:
    """The COMMITTED declaration, read from disk -- never recomputed.

    The distinction is the whole point. A run that recomputed the count would
    always agree with itself; the denominator has to be the one fixed before
    the read, so it is read back from a file that predates it."""
    f = Path(path or DECLARATION_PATH)
    if not f.exists():
        raise FamilyRefused(
            f"REFUSED: no family declaration at {f}. The Holm denominator is "
            f"declared before the first forward score, not derived beside it.")
    return json.loads(f.read_text())


def verify_declaration(path: Path = None) -> dict:
    """Does the committed declaration still describe what this code enumerates?

    A later edit to `enumerate_family` is legitimate; silently disagreeing with
    a committed denominator is not. This re-enumerates under the artifact's OWN
    recorded factors and compares identities, so drift is visible rather than
    absorbed."""
    d = load_declaration(path)
    f = d.get("factors") or {}
    sens = len(f.get("operating_point_arms") or ()) > 1
    now = enumerate_family(f.get("coins") or [],
                           sensitivity_arm_in_family=sens)
    same = now == list(d.get("cells") or ())
    return {
        "declaration_path": str(Path(path or DECLARATION_PATH)),
        "declared_cell_count": d.get("declared_cell_count"),
        "recomputed_count": len(now),
        "identities_unchanged": same,
        "factors_used": {"coins": f.get("coins"),
                         "sensitivity_arm_in_family": sens},
        "why": ("the committed denominator is compared with what this code "
                "would enumerate TODAY under the artifact's own factors. A "
                "code change that moves the family without moving the "
                "declaration turns this False."),
    }


def require_declared_count(declaration: dict, cells_to_report) -> dict:
    """REFUSE when a run would report a family it did not declare.

    Both directions matter: reporting FEWER cells than declared is a family
    that quietly shrank after the read, and reporting MORE is one that grew.
    Neither is visible in a p-value."""
    if not isinstance(declaration, dict):
        raise FamilyRefused(
            f"REFUSED: the family declaration is {type(declaration).__name__}, "
            f"not a mapping.")
    dec = declaration.get("declared_cell_count")
    if not isinstance(dec, int) or isinstance(dec, bool) or dec < 1:
        raise FamilyRefused(
            f"REFUSED: declared_cell_count is {dec!r}; the Holm denominator "
            f"must be a positive integer fixed before the read.")
    got = list(cells_to_report)
    if len(got) != dec:
        raise FamilyRefused(
            f"REFUSED: the run would report {len(got)} cells against a "
            f"declared family of {dec}. "
            f"{'A family that grew after the read inflates nothing a reader can see; ' if len(got) > dec else 'A family that shrank after the read makes every surviving p look stronger; '}"
            f"either way the Holm denominator no longer describes what was "
            f"tested.")
    declared = set(declaration.get("cells") or ())
    if declared and set(got) != declared:
        raise FamilyRefused(
            f"REFUSED: the run reports the declared NUMBER of cells but not "
            f"the declared cells. Missing {sorted(declared - set(got))[:4]}; "
            f"unexpected {sorted(set(got) - declared)[:4]}. A substituted "
            f"cell keeps the denominator and changes the family.")
    return {"declared_cell_count": dec, "reported": len(got),
            "identities_match": True,
            "holm_denominator": dec}


# ---------------------------------------------------------------------------
# SELFTEST. Rule 15 / SEAT_PROTOCOL rule 16: the refusal fires on a family
# that moved, and ADMITS one that did not.
# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 23


def selftest() -> int:
    import traceback
    checks = 0
    fails = []

    def ok(c, label):
        nonlocal checks
        checks += 1
        print(f"PASS: {label}" if c else f"FAIL: {label}")
        if not c:
            fails.append(label)

    def refuses(fn, want, label):
        nonlocal checks
        checks += 1
        try:
            fn()
        except FamilyRefused as e:
            if want in str(e):
                print(f"PASS: {label}")
                return
            fails.append(f"{label} [wrong cause: {str(e)[:110]}]")
            print(f"FAIL: {label} -- refused, not for {want!r}")
            return
        except Exception as e:                        # noqa: BLE001
            fails.append(f"{label} [{type(e).__name__}]")
            print(f"FAIL: {label} -- {type(e).__name__}: {str(e)[:110]}")
            print(traceback.format_exc()[-300:])
            return
        fails.append(f"{label} [ACCEPTED]")
        print(f"FAIL: {label} -- the known-bad was ACCEPTED")

    coins = frozen_coins()
    ok(coins == ["btc", "eth"],
       f"the coin set is READ from the frozen candidate's own fits ({coins}), "
       f"never typed here")
    d = declare()
    cells = d["cells"]

    # ---- the enumeration, and the structure the ruling's word misses ----
    ok(len(cells) == len(set(cells)) == d["declared_cell_count"],
       f"the declared count ({d['declared_cell_count']}) is len() of a list of "
       f"DISTINCT cell identities, not a product of factors")
    ok(d["cells_by_convention"]["BY_THRESHOLD"] == 12
       and d["cells_by_convention"]["BY_COUNT"] == 6,
       f"the two conventions contribute UNEQUALLY "
       f"({d['cells_by_convention']}) -- by-COUNT has no operating-point "
       f"dimension, so it does not twin the sensitivity arm")
    g = d["growth_vs_single_convention"]
    ok(g["ratio_is_2x"] is False and abs(g["ratio"] - 1.5) < 1e-12,
       f"COMPUTED DISAGREEMENT WITH THE RULING'S FRAMING: reporting both "
       f"conventions grows the family {g['single_convention_count']} -> "
       f"{g['both_conventions_count']} = {g['ratio']}x, not the 2x the word "
       f"'DOUBLES' implies -- reported, never reconciled to the framing")
    ok(all("/" in c and c.split("/")[0] in FM.PAIRING_CONVENTIONS
           for c in cells),
       "every cell identity CARRIES its pairing convention as its first "
       "component -- a reader cannot hold one without it")
    ok(all("NO_OPERATING_POINT" in c for c in cells
           if c.startswith("BY_COUNT")),
       "every BY_COUNT cell says NO_OPERATING_POINT in its own identity")
    ok(d["holm_denominator"] == d["declared_cell_count"],
       "the Holm denominator IS the declared count, in one field")
    ok(d["selects_nothing"] is True,
       "the declaration selects nothing (rule 14)")

    # ---- the open factors are priced, not merely named ------------------
    alts = d["count_under_each_alternative"]
    ok(len(alts) == 4 and len(set(alts.values())) > 1,
       f"every alternative of both OPEN factors is PRICED as an integer "
       f"({alts}) -- a ruling is one word and its consequence is already here")
    ok(set(d["open_factors"]) == {"sensitivity_arm_in_family", "coins"},
       "the two factors this module cannot rule are named as OPEN")

    # ---- the refusal, BOTH directions -----------------------------------
    ok(require_declared_count(d, cells)["identities_match"] is True,
       "POSITIVE CONTROL: reporting exactly the declared cells is ADMITTED")
    refuses(lambda: require_declared_count(d, cells[:-1]),
            "shrank after the read",
            "KNOWN-BAD: reporting FEWER cells than declared REFUSES -- a "
            "shrunken family makes every surviving p look stronger")
    refuses(lambda: require_declared_count(d, cells + ["EXTRA/x/btc/5%"]),
            "grew after the read",
            "KNOWN-BAD: reporting MORE cells than declared REFUSES")
    swapped = cells[:-1] + ["BY_THRESHOLD/FROZEN_FROM_TRAIN_QUANTILE/doge/5%"]
    refuses(lambda: require_declared_count(d, swapped),
            "not the declared cells",
            "KNOWN-BAD: a SUBSTITUTED cell keeps the count and changes the "
            "family -- refused on identity, not only on length")
    refuses(lambda: require_declared_count({"declared_cell_count": 0}, []),
            "positive integer",
            "KNOWN-BAD: a zero declared count REFUSES")
    refuses(lambda: require_declared_count("18", cells),
            "not a mapping",
            "KNOWN-BAD: a declaration that is not a mapping REFUSES")

    # ---- the factors move the count in the direction claimed ------------
    ok(len(enumerate_family(["btc"])) < len(enumerate_family(["btc", "eth"])),
       "dropping a coin REDUCES the family -- the enumeration responds to its "
       "factors rather than returning a constant")
    ok(len(enumerate_family(coins, sensitivity_arm_in_family=False))
       < len(cells),
       "dropping the sensitivity arm REDUCES the family, and only the "
       "BY_THRESHOLD half of it")
    ok(len([c for c in enumerate_family(coins, sensitivity_arm_in_family=False)
            if c.startswith("BY_COUNT")])
       == d["cells_by_convention"]["BY_COUNT"],
       "and the BY_COUNT half is UNCHANGED by that factor -- which is the "
       "whole reason the growth is not 2x")
    ok(d["factors"]["budgets"] == ["5%", "10%", "15%"]
       and len(d["factors"]["budgets_module_sha256_prefix"]) == 16,
       "the grid is taken from phase2_declaration and cited BY SHA rather "
       "than retyped")

    # ---- the COMMITTED declaration, and drift against it ---------------
    if DECLARATION_PATH.exists():
        v = verify_declaration()
        ok(v["identities_unchanged"] is True
           and v["declared_cell_count"] == v["recomputed_count"],
           f"POSITIVE CONTROL: the COMMITTED declaration "
           f"({v['declared_cell_count']} cells) still matches what this code "
           f"enumerates today under the artifact's own factors")
        ok(load_declaration()["declared_cell_count"] == d["declared_cell_count"],
           "the committed count is READ BACK from the file rather than "
           "recomputed beside the run -- a run that recomputed would always "
           "agree with itself")
    else:
        ok(False, "the committed family declaration is absent from "
                  f"{DECLARATION_PATH}")
        ok(False, "and so the committed count cannot be read back")
    refuses(lambda: load_declaration(Path("/nonexistent/decl.json")),
            "no family declaration at",
            "KNOWN-BAD: a missing declaration REFUSES -- the denominator is "
            "never derived beside the read")

    print(f"\n{checks} checks passed" if not fails
          else f"\n{len(fails)} FAILURES of {checks} checks")
    for f in fails:
        print(f"  - {f}")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--declare" in argv:
        print(json.dumps(declare(), indent=1, sort_keys=True, default=str))
        return 0
    print("usage: be_forward_family.py --selftest | --declare")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
