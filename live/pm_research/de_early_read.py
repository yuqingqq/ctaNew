"""THE USER-RULED EARLY READ OF THE FOUR SEALED DAYS (R-754, DE 121).

WHAT THIS IS. On 2026-09-07T06:18:16Z the USER ruled: "it does not make
sense to seal the results, show me 4 days results first, we need to check
and review the results". Rule 14 -- the user decides. The coordinator's
objection (rule 11: the four days become consumed; G = 4 gives 2^-4 =
0.0625; no interval below five complete days) is recorded in params v16's
`user_ruled_early_read` block and OVERRULED.

WHAT THIS IS NOT. Not a validation, not a verdict, not a new bar. It does
not touch `read_gate`, which still requires all six days and its own
clock; it does not move `PARAMS_REL`, which still names v15, so the
sealed six-day path is byte-identical for every day outside the ruling.

WHERE THE NUMBERS COME FROM. The COMPUTATION is v15's -- the same params
the four sealed days ran, so the arms, the estimand and the draw counts
are theirs and not this read's. Only the SEAL BAR comes from v16. That
split is the whole point: the read must show the numbers those runs
computed and stripped, not numbers it chose.
"""
from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import sys
from pathlib import Path

# BOTH LAUNCHERS. The script path needs this file's own directory on
# `sys.path` for the sibling import; the `-m` path needs the repo root for
# the package import. Adding both makes the module launcher-agnostic --
# the defect DE 120 spent a batch on, one module later.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import de_multiday_gate1_runner as RUN                       # noqa: E402
import declaration_chain as DC                               # noqa: E402

PARAMS_FAMILY = "de_multiday_gate1_params"
DECL_DIR = "live/pm_research/declarations"

#: A NEW FAMILY. Never a `.vN` of a sealed receipt: a correction chain
#: says "this supersedes that", and this artifact supersedes nothing -- it
#: is a different question asked of the same days.
DAY_FAMILY = "p003_de_early_read_day"

#: R-709: THIS PRODUCER'S EXIT CODES, DECLARED HERE AND READ FROM HERE by
#: `producer_exit_maps`. The map is not typed into the declaration twice;
#: the declaration says it is read from this constant, and the battery
#: asserts the two agree. 75 is the LAUNCH LAYER'S (`flock -n -E 75` means
#: a held lock and the payload never started) and never appears here.
EXIT_CODES = {
    0: "the early-read day completed and its artifact was written, or "
       "--selftest passed",
    1: "an EARLY_READ_* refusal, a battery failure, or usage -- every "
       "non-zero path this module has, because EarlyReadRefused is an "
       "uncaught exception and SystemExit carries a message",
}

EXPECTED_CHECKS = 30


class EarlyReadRefused(RuntimeError):
    """A named refusal. Every message begins with its own reason code."""


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def the_ruling(repo_root=None) -> dict:
    """THE BAR, READ FROM THE PARAMS CHAIN HEAD -- or REFUSED BY NAME.

    The block is not optional and it is not inferred. A read that opens
    four sealed days on the runner's own initiative is the thing rule 14
    forbids, so its authority has to be a landed declaration naming the
    ruling, and its absence is a refusal with a reason code."""
    root = Path(repo_root) if repo_root else Path(
        __file__).resolve().parents[2]
    head = DC.resolve_head(root / DECL_DIR, PARAMS_FAMILY)
    doc = json.loads(Path(head["path"]).read_text())
    block = doc.get("user_ruled_early_read")
    if not isinstance(block, dict):
        raise EarlyReadRefused(
            f"EARLY_READ_NOT_RULED: the params chain head "
            f"({head['name']}, {head['sha256'][:16]}…) carries no "
            f"`user_ruled_early_read` block. The early read opens days "
            f"that are sealed for a reason; its authority is a USER "
            f"ruling landed in the declaration, never this module's own "
            f"word (rule 14).")
    for field in ("the_ruling_verbatim", "this_reads_bar",
                  "days_consumed_by_this_read", "verdict_class"):
        if field not in block:
            raise EarlyReadRefused(
                f"EARLY_READ_BLOCK_INCOMPLETE: the block is missing "
                f"`{field}`. A ruling recorded in pieces is a ruling a "
                f"later reader has to reconstruct.")
    days = list(block["this_reads_bar"]["days"])
    # REV 95 minor: THE PATH IS REPO-RELATIVE. `resolve_head` returns an
    # ABSOLUTE path, so the landed 09-03 artifact names
    # `/home/yuqing/ctaNew-wt-de/live/...` -- a worktree that is frozen
    # today and gone tomorrow, recorded inside an artifact meant to
    # outlive it. The landed one is NOT edited (rule 13); the note below
    # travels in every emission from here.
    _rp = str(head["path"])
    try:
        _rp = str(Path(_rp).resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        pass
    return {
        "ruling_by_pair": {"path": _rp, "sha256": head["sha256"],
                           "path_is": "REPO-RELATIVE"},
        "NOTE_on_the_landed_09_03_artifact": {
            "artifact": "p003_de_early_read_day_20260903__"
                        "20260907T085436Z.json",
            "it_is_LANDED_and_is_never_edited": "rule 13. This note is the "
                "correction, in band, and it travels with every emission "
                "of this family from here.",
            "one__the_absolute_ruling_path": (
                "it names its ruling at an ABSOLUTE path into "
                "/home/yuqing/ctaNew-wt-de, the worktree that produced it "
                "-- frozen today and gone tomorrow, recorded inside an "
                "artifact meant to outlive it. Every emission from here "
                "writes the path REPO-RELATIVE (REV 95)."),
            "two__two_FALSE_seal_fields_in_its_day_run_block": (
                "`day_run.status` reads `DAY_RUN_SEALED` and "
                "`day_run.what_this_is_not.the_economics_are_SEALED` reads "
                "`true`, while BOTH arm-days in that same artifact read "
                "`sealed: false` with their `economic` block PRESENT. The "
                "status was a literal and the flag was "
                "`n_days_complete < G`, which is TRUE at 4 of 6 -- exactly "
                "the early read. Both are computed from the arm-days' own "
                "seal state from DE 126 on (CLAUDE.md rule 10). A reader "
                "of the 09-03 artifact must take its arm-days, not these "
                "two fields, as the account of what was sealed."),
        },
        "params_version": doc.get("version"),
        "days": days,
        "G": len(days),
        "verdict_class": block["verdict_class"],
        "interval": "NONE_BELOW_FIVE_DAYS",
        "is_a_validation": False,
        "days_consumed": list(block["days_consumed_by_this_read"]),
        "the_ruling_verbatim": block["the_ruling_verbatim"],
        "recorded_at_utc": block.get("recorded_at_utc"),
        "receipts": block["this_reads_bar"]["receipts"],
        "block": block,
    }


#: THE FIELDS THE DISPATCH ASKED FOR THAT THIS CODE PATH DOES NOT COMPUTE.
#: Named as STATUSES, never dropped in silence (reliability rule 4). Each
#: one would be a NEW statistic, and a read whose remit is "show what
#: those runs computed and stripped" may not invent one.
#: DE 129: THREE OF THESE FIVE ARE NOW COMPUTED, and this block said
#: otherwise in every artifact it was emitted into. The DECISION LEDGER
#: (R-765) recomputes `p_two_sided`, `rho = adverse/spread` and -- since
#: BE 96 -- the `inventory_leg`, from stored rows with the book absent.
#: What this block describes is the ARM-DAY ECONOMIC BLOCK in the receipt,
#: which still carries none of them; a reader has to be told WHERE to
#: look rather than that the number does not exist.
WHERE_THE_FIVE_LIVE_NOW = {
    "p_two_sided": "COMPUTED, in the decision ledger "
                   "(de_decision_ledger.recompute)",
    "rho_adverse_over_spread": "COMPUTED, in the decision ledger",
    # `inventory_leg` IS NOT TYPED HERE ANY MORE (R-795, BE 97). It read
    # "COMPUTED since BE 96, in the decision ledger", and BE measured the
    # 09-05 ledger (5a2032b5…): no row carries a key by that name, at any
    # depth. The entry is now MEASURED at the ledger this run wrote --
    # `inventory_at_the_ledger()` below -- and this dict carries only the
    # claims that are about `recompute`, not about the file's rows.
    "fills_leg": "COMPUTED, in the decision ledger",
    "D_E_MINUS_R": "STILL NOT COMPUTED ANYWHERE -- the robustness "
                   "endpoint needs the rebate's identity value, which is "
                   "not on DE's surface",
}

def inventory_at_the_ledger(ledger_path, *, arm_days=None) -> dict:
    """WHAT THE LEDGER CARRIES ABOUT INVENTORY -- MEASURED, NOT TYPED.

    R-795 / BE 97 (Q-BE-340). This artifact said the inventory leg was
    "COMPUTED since BE 96, in the decision ledger". It is not: BE walked
    the 09-05 ledger and `inventory_leg` is not a top-level key of any
    row and is not nested anywhere in the file. What the file carries is
    the FIVE BE-96 FIELDS per fill record -- which are the inputs a leg
    would be computed FROM, not the leg.

    And the day value is the FILLS LEG BY CONSTRUCTION, not because
    inventory happens to be zero: `_value_cents` sums a per-fill markout
    term with no position term, and BE measured non-zero end-of-day
    residual positions on 287-288 of 288 slugs on every path. Whether an
    inventory LEG is added to the day value, and under which residual /
    mark / grouping rule, is a USER RULING (R-795, rule 14): the two
    illustrative rules BE computed differ by more than the day's whole
    fills leg.

    So every field below is read from the FILE (its own HEADER row and a
    scan of its rows) or from the runner's own `what_total_is` on the
    arm-days -- rule 10, because the sentence this replaces was a typed
    claim the artifact itself contradicted."""
    if not ledger_path:
        return {"status": "NOT_MEASURED_NO_LEDGER_ON_THIS_PATH",
                "why": "this path wrote no ledger, so nothing about a "
                       "ledger's contents is asserted here"}
    lp = Path(ledger_path)
    if not lp.is_file():
        return {"status": "NOT_MEASURED_LEDGER_ABSENT",
                "path": str(lp),
                "why": "the ledger named is not on disk; an unread file "
                       "is not evidence either way"}

    def _paths_to(o, key, prefix=""):
        """Every JSON path at which `key` occurs -- WHERE, not only how
        many. A count alone cannot tell a per-fill leg from a name that
        happens to sit inside the absolutes."""
        out = []
        if isinstance(o, dict):
            for k, v in o.items():
                here = f"{prefix}.{k}" if prefix else str(k)
                if k == key:
                    out.append(here)
                out.extend(_paths_to(v, key, here))
        elif isinstance(o, list):
            for i, e in enumerate(o):
                out.extend(_paths_to(e, key, f"{prefix}[{i}]"))
        return out

    header, kinds, n_rows = None, {}, 0
    rows_with_leg, fill_rows, fill_rows_all_five = 0, 0, 0
    five, where, where_tcf = None, {}, {}
    with gzip.open(lp, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            n_rows += 1
            k = r.get("row")
            kinds[k] = kinds.get(k, 0) + 1
            if k == "HEADER":
                header = r
                five = list(r.get("inventory_fields_from_BE_96") or [])
            _hits = _paths_to(r, "inventory_leg")
            if _hits:
                rows_with_leg += 1
                for _h in _hits:
                    _pth = f"{k}.{_h}"
                    where[_pth] = where.get(_pth, 0) + 1
            # R-803: the field was RENAMED to what BE 99 measured it to
            # be. Both names are tracked, because a reader meets either
            # depending on which code wrote the file, and a block that
            # reported only the old one would go quiet exactly when the
            # rename landed.
            for _h2 in _paths_to(r, "trades_cash_flow_cents"):
                _p2 = f"{k}.{_h2}"
                where_tcf[_p2] = where_tcf.get(_p2, 0) + 1
            if k == "FILL":
                fill_rows += 1
                if five and all(f in r for f in five):
                    fill_rows_all_five += 1
    # the runner's OWN words for what the day value is, taken off the
    # arm-days it just emitted -- never a sentence typed here.
    what_total_is = sorted({
        (a.get("absolute") or {}).get("arm", {}).get("what_total_is")
        for a in (arm_days or [])
        if isinstance(a.get("absolute"), dict)} - {None})
    # WHERE the name occurs decides what may be said about it, and the
    # answer DEPENDS ON THE FILE -- which is why no sentence here is
    # typed. The 09-05 and 09-06 ledgers were written by the ledger
    # module at `6c3a121`, whose ARM_SCALARS row carries no `absolute`
    # block: BE 97 measured 0 occurrences and was right about those
    # files. A ledger written by the CURRENT module carries DE 131's
    # absolutes in that row, and `absolute_legs` names one of its
    # components `inventory_leg` -- so the same typed sentence would be
    # wrong for one file or the other, whichever way it was written.
    _abs_only = bool(where) and all(
        w.startswith("ARM_SCALARS.absolute.") for w in where)
    status = ("NOT_A_FIELD_OF_THE_LEDGER" if not where
              else ("PRESENT_ONLY_INSIDE_THE_ABSOLUTES" if _abs_only
                    else "PRESENT_OUTSIDE_THE_ABSOLUTES"))
    return {
        "status": status,
        "where_the_name_occurs": where,
        "R_803_the_renamed_field": {
            "name": "trades_cash_flow_cents",
            "where_it_occurs": where_tcf,
            "sign_convention": "SELLS - BUYS; the NEGATIVE of the "
                               "quantity the old name carried",
            "why_renamed": ("BE 99 (R-803) measured sum((after - before) "
                            "x mark) to be the exact negative of the "
                            "R-801 trades leg on all four path-days -- "
                            "the trades cash flow, never a residual "
                            "mark-to-market"),
            "the_residual_is_priced": "at SETTLEMENT, in the R-801 legs "
                                      "(SETTLEMENT_SLUG rows)"},
        "what_that_name_IS_where_it_occurs": (
            "a component of DE 131's `absolute` block, computed by "
            "`absolute_legs` as sum((after - before) x mark). BE 97 "
            "measured what that equals: since `after - before` is the "
            "fill's own signed size, it is the day's FILLS CASH FLOW -- "
            "a READER's quantity that enters neither the day value nor "
            "D(E0). It is NOT a residual mark-to-market and it is not a "
            "leg of the value." if where else
            "the name does not occur in this file at any depth"),
        "measured_at": {"path": str(lp), "sha256": _sha(lp),
                        "rows_scanned": n_rows, "row_kinds": kinds},
        "the_five_inventory_FIELDS_the_file_declares":
            five if five is not None else "NO_HEADER_ROW_IN_THIS_FILE",
        "the_five_are_read_from": "the ledger's own HEADER row "
                                  "(`inventory_fields_from_BE_96`), not "
                                  "from a list in this module",
        "rows_carrying_an_inventory_leg_key_at_any_depth": rows_with_leg,
        "no_fill_row_carries_a_leg": not any(
            w.startswith("FILL.") for w in where),
        "fill_rows": fill_rows,
        "fill_rows_carrying_all_five_fields": fill_rows_all_five,
        "the_day_value_is": (what_total_is or
                             ["NOT_READ -- no arm-day carried an "
                              "`absolute` block"]),
        "the_day_value_is_read_from": "the runner's own `what_total_is` "
                                      "on each arm-day's `absolute.arm` "
                                      "block",
        "why_it_is_the_fills_leg": (
            "`_value_cents` sums a per-fill markout term "
            "(sgn x (mid_at_markout - px_cents) x size) with NO position "
            "term. The value is the fills leg BY CONSTRUCTION -- not "
            "because inventory is zero: BE 97 measured non-zero "
            "end-of-day residual positions on 287-288 of 288 slugs on "
            "every path."),
        "whether_an_inventory_LEG_exists_is_a_RULING": (
            "R-795, routed to the USER under rule 14: which residual, "
            "marked at which price, per slug or per day. The two "
            "illustrative rules BE computed differ by MORE than the "
            "day's whole fills leg, which is what makes it a ruling and "
            "not a detail."),
        "what_this_replaces": (
            "the typed sentence `inventory_leg: COMPUTED since BE 96, in "
            "the decision ledger` (rule 10). It was wrong twice over: no "
            "row of the 09-05 ledger carries that name at all, and where "
            "a current ledger does carry it, it is a component of the "
            "absolutes and not a leg of the day value. A sentence typed "
            "either way is wrong for one of those files, so this block "
            "is measured at the file the run actually wrote."),
    }


NOT_COMPUTED_BY_THIS_PATH = {
    "fills_leg": "the arm-day economic block is a SINGLE excess `D_E0` "
                 "against the 0-cancel baseline. There is no fills/"
                 "inventory decomposition anywhere in the runner, so a "
                 "leg split cannot be reported without defining one.",
    "trades_cash_flow_cents": "as above -- the same single quantity is "
                              "not two. (R-803: this entry was called "
                              "`inventory_leg`; BE 99 measured that "
                              "quantity to be the trades cash flow with "
                              "the opposite sign, and the residual is "
                              "priced at SETTLEMENT in the R-801 legs.)",
    "p_two_sided": "`p_location` is `p_one_sided` from "
                   "`DESIGN.per_day_location`. Doubling it is a "
                   "DECISION about the test, not a re-read of it.",
    "rho_adverse_over_spread": "`rho`, `adverse` and `spread` appear "
                               "nowhere in the runner (0 occurrences "
                               "each). This quantity has never been "
                               "computed for these days.",
    "D_E_MINUS_R": "a sealed NAME, but no arm-day economic block "
                   "produces it; only `D_E0` is emitted.",
}


def economics_available_per_arm_day(ledger_path=None,
                                    arm_days=None) -> dict:
    """WHAT AN UNSEALED ARM-DAY ACTUALLY CARRIES, read off the emitter.

    `ledger_path` and `arm_days` are what the run just produced: the
    inventory entry is MEASURED at that file rather than typed (R-795).
    With neither, the entry says it did not measure -- it never falls
    back to the sentence BE 97 refuted."""
    _inv = inventory_at_the_ledger(ledger_path, arm_days=arm_days)
    return {
        "from_the_economic_block": ["D_E0", "Z", "p_location",
                                    "null_mean", "null_sd",
                                    "null_draws_summary.n"],
        "from_the_arm_level": ["n_fills_arm", "n_fills_baseline",
                               "n_cancels_issued", "status",
                               "admissibility", "seed", "draw_provenance"],
        "not_computed_by_this_path_the_ARM_DAY_BLOCK":
            NOT_COMPUTED_BY_THIS_PATH,
        # R-803: the ENTRY IS RENAMED WITH THE FIELD. There is no
        # inventory leg: the quantity that name carried is the trades
        # cash flow (BE 99), and the residual is priced at settlement in
        # the R-801 legs. The entry keeps the measurement.
        "where_the_five_live_now": {**WHERE_THE_FIVE_LIVE_NOW,
                                    "trades_cash_flow_cents": _inv},
        "read_this_first": (
            # COMPUTED FROM THE ENTRIES BESIDE IT, never a typed tally:
            # this sentence said "four of the five ARE computed, in the
            # DECISION LEDGER", and the inventory leg is not one of them
            # (R-795). A count that is typed goes stale exactly when the
            # entry under it changes, which is what happened here.
            f"{len([k for k, v in WHERE_THE_FIVE_LIVE_NOW.items() if isinstance(v, str) and v.startswith('COMPUTED')])}"
            f" of the five are computed in the DECISION LEDGER beside the "
            f"receipt (p_two_sided, rho, fills_leg); the INVENTORY LEG is "
            f"`{_inv.get('status')}` -- measured at the ledger this run "
            f"wrote, not asserted (R-795, BE 97); and D_E_MINUS_R is "
            f"computed nowhere. This block once said all five were "
            f"absent, which stopped being true at R-765 and BE 96, and "
            f"then said the inventory leg was computed, which the file "
            f"never bore out."),
    }


def early_read_preconditions(day: str, root, ruling: dict) -> dict:
    """P9's REPLACEMENT FOR THIS ENTRY (coordinator ruling, DE 122).

    P9 on the sealed `--day` path refuses a day that already has a sealed
    receipt, because a day run twice has no newest result. THIS entry's
    premise is the opposite: it opens days precisely BECAUSE they are
    sealed, and it writes a DIFFERENT family, so it supersedes nothing.
    The precondition is therefore inverted and split in two, each half
    refusing by its own name:

      EARLY_READ_NO_SEALED_RECEIPT     the day's sealed chain-head
                                       receipt is not there at all;
      EARLY_READ_RECEIPT_NOT_THE_PAIR  it is there but is not the one
                                       v16's bar names for that day;
      EARLY_READ_ALREADY_EMITTED       this read has already run for the
                                       day, and a second emission would
                                       leave two answers with no rule
                                       saying which is newest -- P9's own
                                       reasoning, applied where it does
                                       belong.

    ON WHAT "THE PAIR" IS. The FULL sha256 (DE 123's ruling). params v16's
    bar carried `sha256_16` only, so this could compare sixteen hex and no
    more; v17 supersedes it carrying the full digest of each of the four
    chain-head receipts. A bar without full digests does not fall back to
    the prefix -- it REFUSES by its own name
    (`EARLY_READ_BAR_CARRIES_ONLY_A_PREFIX`), because a fallback makes the
    strength of the check depend on which version happens to be the head
    while the artifact says nothing about which check ran.
    """
    root = Path(root)
    der = root / "pm_5min/derived"
    entry = next((r for r in ruling["receipts"] if r["day"] == day), None)
    if entry is None:
        raise EarlyReadRefused(
            f"EARLY_READ_DAY_NOT_IN_THE_BAR: {day} is not among "
            f"{ruling['days']}.")
    compact = day.replace("-", "")
    found = sorted(der.glob(
        f"p003_de_gate1_day_run_{compact}_SEALED__*.json"))
    if not found:
        raise EarlyReadRefused(
            f"EARLY_READ_NO_SEALED_RECEIPT: {day} has no sealed day-run "
            f"receipt under {der}. This read shows what a SEALED run "
            f"computed and stripped; with no sealed run there is nothing "
            f"it is the early read OF.")
    want_name = Path(entry["path"]).name
    have = der / want_name
    if not have.is_file():
        raise EarlyReadRefused(
            f"EARLY_READ_RECEIPT_NOT_THE_PAIR: v16's bar names "
            f"{want_name} for {day}, and that file is not present. The "
            f"{len(found)} sealed artifact(s) that ARE present "
            f"({[f.name for f in found]}) are not the one the ruling "
            f"authorised.")
    got = _sha(have)
    want16 = str(entry.get("sha256_16") or "")
    want_full = entry.get("sha256")
    # DE 123 RULING: THE PAIR IS THE FULL DIGEST. A bar that carries only
    # a prefix REFUSES BY ITS OWN NAME rather than quietly falling back to
    # comparing sixteen characters -- a fallback would mean the strength
    # of the check depended on which params version happened to be the
    # head, and nothing in the artifact would say which check ran.
    if not want_full:
        raise EarlyReadRefused(
            f"EARLY_READ_BAR_CARRIES_ONLY_A_PREFIX: the bar names "
            f"{want16 or '(nothing)'}… for {day} and carries no full "
            f"`sha256`. params v16 was that shape. The pair is the FULL "
            f"digest (DE 123), and a prefix comparison is not silently "
            f"substituted for one -- land a params version whose bar "
            f"carries full digests (v17 does).")
    if not DC.DIGEST_RE.match(str(want_full)):
        raise EarlyReadRefused(
            f"EARLY_READ_BAR_DIGEST_MALFORMED: the bar's `sha256` for "
            f"{day} is {str(want_full)[:24]!r}, which is not 64 lowercase "
            f"hex. A digest that is not a digest cannot be a pair.")
    if got != want_full:
        raise EarlyReadRefused(
            f"EARLY_READ_RECEIPT_NOT_THE_PAIR: {want_name} is present but "
            f"its digest is {got}, and the bar names {want_full} for "
            f"{day}. The read opens the artifact the USER's ruling named, "
            f"never whatever now sits at that path.")
    if want16 and not got.startswith(want16):
        raise EarlyReadRefused(
            f"EARLY_READ_BAR_PREFIX_DISAGREES_WITH_ITS_OWN_DIGEST: the "
            f"bar's `sha256_16` for {day} is {want16}… but its full "
            f"`sha256` begins {got[:16]}…. A version that disagrees with "
            f"itself names no artifact.")
    already = sorted(der.glob(f"{DAY_FAMILY}_{compact}__*.json"))
    if already:
        raise EarlyReadRefused(
            f"EARLY_READ_ALREADY_EMITTED: {day} already has "
            f"{len(already)} early-read artifact(s) "
            f"({[a.name for a in already]}). A second emission would "
            f"leave two answers to one question with no rule saying which "
            f"is newest -- which is P9's reasoning, in the place it "
            f"belongs for this entry.")
    return {
        "replaces": "P9_no_sealed_receipt_for_this_day_yet",
        "why_replaced": "P9's premise is that a day run twice has no "
                        "newest result. This entry writes a DIFFERENT "
                        "family and supersedes nothing, and it requires "
                        "the sealed receipt to EXIST -- the opposite "
                        "condition (coordinator ruling, DE 122).",
        "sealed_receipt": {"path": str(have), "sha256": got,
                           "name": want_name},
        "digest_comparison": {
            "compared": "basename exactly, and the FULL sha256",
            "is_a_full_pair": True,
            "bar_says": want_full, "artifact_is": got,
            "prefix_beside_it": want16,
            "prefix_agrees_with_the_full_digest": bool(
                want16) and got.startswith(want16),
            "a_prefix_only_bar": "REFUSES by name "
                                 "(EARLY_READ_BAR_CARRIES_ONLY_A_PREFIX); "
                                 "it is never silently compared on 16 hex"},
        "no_early_read_artifact_yet": True,
        "sealed_day_path_unaffected": "the `--day` entry keeps P9 "
                                      "byte-identical; this function is "
                                      "not on that path",
        "holds": True,
    }


def bar_day_states(*, repo_root=None, root=None) -> dict:
    """WHICH RULED DAYS ARE READ, AND WHICH IS NEXT -- from the LEDGER.

    DE 126's addendum. A cell that names a day as a LITERAL measures the
    day it was written on: `rehearse("2026-09-03")` expected READY, GO E1
    read 09-03, and the cell aborted the battery with a KeyError -- five
    checks after it never ran. The state a cell asserts about must be
    derived from the ledger, so the cell moves as the days are read."""
    ruling = the_ruling(repo_root)
    r = Path(root) if root else Path(RUN.DR.resolve()["data_root"])
    der = r / "pm_5min/derived"
    read, unread = [], []
    for d in ruling["days"]:
        got = sorted(der.glob(f"{DAY_FAMILY}_{d.replace('-', '')}__*.json"))
        (read if got else unread).append(d)
    return {"ruled_days": ruling["days"], "read": read, "unread": unread,
            "next_unread": unread[0] if unread else None,
            "most_recently_read": read[-1] if read else None,
            "derived_from": str(der),
            "why_not_a_literal": "a date typed into a cell measures the "
                                 "day the cell was written on"}


def rehearse(day: str, *, repo_root=None, root=None) -> dict:
    """READY, or the blocker BY NAME. Touches no book and takes no lock."""
    out = {"day": day, "entry": "de_early_read --early-read-day",
           "as_of": datetime.datetime.now(
               datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        ruling = the_ruling(repo_root)
    except EarlyReadRefused as e:
        return {**out, "status": "NOT_READY",
                "blocking": [str(e).split(":")[0]], "detail": str(e)}
    out["ruling"] = ruling["ruling_by_pair"]
    r = Path(root) if root else Path(RUN.DR.resolve()["data_root"])
    try:
        pre = early_read_preconditions(day, r, ruling)
    except EarlyReadRefused as e:
        return {**out, "status": "NOT_READY",
                "blocking": [str(e).split(":")[0]], "detail": str(e)}
    book = r / "pm_5min/derived" / f"be_daybook_{day.replace('-','')}_btc.pkl"
    if not book.is_file():
        return {**out, "status": "NOT_READY",
                "blocking": ["EARLY_READ_BOOK_ABSENT"],
                "detail": f"EARLY_READ_BOOK_ABSENT: {book}"}
    return {**out, "status": "READY", "blocking": [],
            "preconditions": pre, "book": str(book),
            "G": ruling["G"], "verdict_class": ruling["verdict_class"],
            "interval": ruling["interval"]}


def day_artifact_name(day: str, now=None) -> str:
    stamp = RUN.emission_stamp(now)
    return f"{DAY_FAMILY}_{day.replace('-', '')}__{stamp}.json"


def run_early_read_day(day: str, book, outdir, *, repo_root=None,
                       before_work=None) -> dict:
    """ONE DAY, UNSEALED UNDER THE RULING. The heavy call.

    `params` is v15's -- `RUN.load_params()` -- so the computation is the
    sealed runs'. `early_read` is v16's bar, and it reaches exactly one
    place: the seal call."""
    ruling = the_ruling(repo_root)
    if day not in ruling["days"]:
        raise EarlyReadRefused(
            f"EARLY_READ_DAY_NOT_IN_THE_BAR: {day} is not among "
            f"{ruling['days']}. The ruling names four days; a fifth would "
            f"be consumed by a read nobody authorised.")
    pre = early_read_preconditions(
        day, Path(RUN.DR.resolve()["data_root"]), ruling)
    params = RUN.load_params()
    # DE 132: THE LEDGER IS ANCHORED ON THIS ARTIFACT'S OWN PATH. The
    # early read has no `receipt_path` -- its artifact is its own family --
    # and `run_day` used that as the condition for writing R-765's ledger,
    # so E1 and E2 emitted `decision_ledger: null` and wrote none. The name
    # is composed BEFORE the run so the ledger lands beside the artifact
    # that names it.
    out = Path(outdir) / day_artifact_name(day)
    result = RUN.run_day(day, book, params=params, fixture=False,
                         n_days_complete=ruling["G"],
                         early_read={"G": ruling["G"],
                                     "days": ruling["days"]},
                         ledger_anchor=out,
                         before_work=before_work)
    _led = (result or {}).get("decision_ledger")
    if not (isinstance(_led, dict) and _led.get("sha256")):
        raise EarlyReadRefused(
            f"EARLY_READ_WROTE_NO_DECISION_LEDGER: the run returned "
            f"`decision_ledger` {_led!r}. R-765 orders the numbers kept, "
            f"and an artifact carrying a null block is the SILENT form of "
            f"promising a ledger that is not there -- which is how GO E1 "
            f"and GO E2 both went unnoticed.")
    payload = {
        "protocol": "P003_DE_EARLY_READ_DAY_V1",
        "what_this_is": "the USER-ruled early read of one sealed day "
                        "(R-754). NOT a validation, NOT a verdict, and "
                        "NOT a version of that day's sealed receipt.",
        "ruling": ruling["ruling_by_pair"],
        "the_ruling_verbatim": ruling["the_ruling_verbatim"],
        "is_a_validation": False,
        "G": ruling["G"],
        "interval": "NONE_BELOW_FIVE_DAYS",
        "verdict_class": ruling["verdict_class"],
        "days_consumed": ruling["days_consumed"],
        # REV 90 §A0, the second half of the closure. `seal()` now computes
        # a truthful status from the bar it was handed ("4 of 4"), which is
        # right but says nothing about the SIX. This artifact is the one a
        # human reads, so the relation between the two numbers is stated
        # here in words, with the ruling that authorises it.
        "seal_standing": {
            "line": (f"UNSEALED under the USER's ruling R-754: "
                     f"{ruling['G']} of "
                     f"{len(ruling['block']['the_six_day_population']['days'])}"
                     f" ruled days, read early on the user's instruction. "
                     f"NOT all days complete, NOT a validation, no "
                     f"interval."),
            "ruled_days_read": ruling["G"],
            "ruled_days_in_the_population": len(
                ruling["block"]["the_six_day_population"]["days"]),
            "authority": "R-754, recorded "
                         f"{ruling.get('recorded_at_utc')}",
            "why_seal_status_inside_day_run_says_4_of_4": (
                "that field reports the bar THAT CALL was given, which is "
                "the ruling's four -- it is not a claim about the six. The "
                "six is here, and in `day_run.G_and_which_G_it_is`."),
        },
        "computation_params": {
            "path": RUN.PARAMS_REL,
            "sha256": _sha(Path(__file__).resolve().parents[2]
                           / RUN.PARAMS_REL),
            "why": "the COMPUTATION is the sealed runs' -- v15. Only the "
                   "seal bar comes from the ruling."},
        "economics_field_availability": economics_available_per_arm_day(
            ledger_path=(_led or {}).get("path"),
            arm_days=(result or {}).get("per_day_sealed_artifacts")),
        "preconditions": pre,
        "day_run": result,
        "as_of": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True,
                              default=str) + "\n")
    return {"path": str(out), "sha256": _sha(out), "day": day}


def resolve_exit(code: int, *, repo_root=None) -> dict:
    """AN EXIT CODE, RESOLVED THROUGH THE DECLARED MAP (R-709).

    REV 89 §8 item 1: DE's two capture records carried no
    `producer_module`, no `mapped_by` and no `resolved_kind` -- the three
    fields their own declaration requires -- so a reader had a number and
    nowhere to take it. UNMAPPED is a STATUS here, never a silent pass:
    an outcome that cannot be resolved through the map does not satisfy a
    GO.

    The map is read from the chain HEAD by the pair, never a filename
    literal, so a new version is picked up without editing this."""
    root = Path(repo_root) if repo_root else Path(
        __file__).resolve().parents[2]
    head = DC.resolve_head(root / DECL_DIR, "producer_exit_maps")
    doc = json.loads(Path(head["path"]).read_text())
    me = "live/pm_research/de_early_read.py"
    block = (doc.get("producers") or {}).get(me)
    out = {"exit_code": int(code),
           "producer_module": me,
           "mapped_by": {"path": head["path"], "sha256": head["sha256"],
                         "version": doc.get("version")},
           "the_map_is_read_from": "the chain head by the pair, never a "
                                   "filename literal"}
    if not isinstance(block, dict):
        out["resolved_kind"] = "UNMAPPED"
        out["why"] = (f"the exit-map head declares no block for {me}. "
                      f"UNMAPPED does not satisfy a GO (R-709).")
        return out
    m = block.get("map") or {}
    if str(code) not in m:
        out["resolved_kind"] = "UNMAPPED"
        out["why"] = (f"exit {code} is not among this producer's declared "
                      f"codes {sorted(m)}. A code nobody declared is a "
                      f"code nobody can read.")
        return out
    out["resolved_kind"] = m[str(code)]
    out["declared_codes"] = sorted(m)
    out["75_is_never_used"] = block.get("75_is_never_used")
    out["why_the_code_alone_is_not_the_kind"] = (
        "every refusal in this module exits 1, so a non-zero code is "
        "resolved to a KIND only with the EARLY_READ_* name from the "
        "run's output beside it")
    return out


# ------------------------------------------------------- the battery

def selftest(quiet: bool = False) -> int:
    """NO CELL MAY RAISE PAST THIS BATTERY (DE 126's addendum).

    A `KeyError` out of cell 4d ended the suite with 13 of 18 run and five
    never reached, and the run READ as an abort rather than as a failure --
    the difference between "this check failed" and "the checks after it
    have no verdict". An unexpected exception is caught here and reported
    as a NAMED battery failure, so a gap is never mistaken for a shorter
    suite."""
    try:
        return _selftest_body(quiet)
    except SystemExit:
        raise
    except BaseException as _e:
        import traceback as _tb
        raise SystemExit(
            f"[de_early_read] FAIL: "
            f"BATTERY_ABORTED_BY_AN_UNCAUGHT_{type(_e).__name__}: {_e}. A "
            f"cell raised past the battery, so every check after it has NO "
            f"VERDICT. {_tb.format_exc().strip().splitlines()[-1]}")


def _selftest_body(quiet: bool = False) -> int:
    """RED FIRST, on fixtures. Three drives, each with its own baseline."""
    import copy
    import shutil
    import tempfile
    n = [0]
    skipped: list = []

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_early_read] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    repo = Path(__file__).resolve().parents[2]

    # ---- DRIVE 1: WITHOUT v16's BLOCK THE READ REFUSES BY NAME --------
    # A fixture chain, not the live one: the known-bad has to be a chain
    # whose head genuinely lacks the block, and editing the live head to
    # produce one would be the in-place edit rule 20 forbids.
    tmp = Path(tempfile.mkdtemp(prefix="early_read_"))
    (tmp / DECL_DIR).mkdir(parents=True)
    live_head = DC.resolve_head(repo / DECL_DIR, PARAMS_FAMILY)
    doc = json.loads(Path(live_head["path"]).read_text())
    without = copy.deepcopy(doc)
    without.pop("user_ruled_early_read", None)
    (tmp / DECL_DIR / f"{PARAMS_FAMILY}_v1.json").write_text(
        json.dumps(without, indent=1, sort_keys=True) + "\n")
    try:
        the_ruling(tmp)
        ok(False, "KNOWN-BAD: a chain head with NO ruling block opened "
                  "the early read")
    except EarlyReadRefused as e:
        ok(str(e).startswith("EARLY_READ_NOT_RULED"),
           f"DRIVE 1 (RED): a params head carrying no "
           f"`user_ruled_early_read` REFUSES BY NAME -- "
           f"`{str(e).split(':')[0]}`. The read's authority is a landed "
           f"USER ruling, never this module's word")

    # The SAME fixture chain WITH the block admits -- so drive 1 measured
    # the block and not the fixture.
    withb = copy.deepcopy(doc)
    (tmp / DECL_DIR / f"{PARAMS_FAMILY}_v2.json").write_text(
        json.dumps({**withb,
                    "supersedes": {
                        "path": f"{DECL_DIR}/{PARAMS_FAMILY}_v1.json",
                        "sha256": _sha(tmp / DECL_DIR
                                       / f"{PARAMS_FAMILY}_v1.json")}},
                   indent=1, sort_keys=True) + "\n")
    r = the_ruling(tmp)
    ok(r["G"] == 4 and r["is_a_validation"] is False
       and r["interval"] == "NONE_BELOW_FIVE_DAYS"
       and r["verdict_class"] == "EXPLORATORY"
       and len(r["days_consumed"]) == 4,
       f"DRIVE 1 CONTROL (GREEN): the same fixture chain WITH the block "
       f"admits -- G {r['G']}, class {r['verdict_class']}, interval "
       f"{r['interval']}, is_a_validation {r['is_a_validation']}, "
       f"{len(r['days_consumed'])} days consumed. So the refusal was the "
       f"block's absence and not the fixture's shape")

    # ---- DRIVE 2: THE UNSEALED LAYOUT CARRIES THE ECONOMIC FIELDS -----
    # Driven on `seal()` itself -- the one function both paths call -- so
    # the layout tested is the layout that will be emitted.
    arm = {"day": "2026-09-03", "arm": "A", "status": "OK",
           "economic": {"D_E0": 2.5, "Z": 2.45, "p_location": 0.008,
                        "null_mean": 0.0066, "null_sd": 1.0158,
                        "null_draws_summary": {"n": 600}},
           "n_fills_arm": 30171, "n_fills_baseline": 46439,
           "n_cancels_issued": 5146}
    sealed6 = RUN.seal(copy.deepcopy(arm), 4, 6)          # the six-day bar
    unsealed4 = RUN.seal(copy.deepcopy(arm), 4, 4)        # the ruling's bar
    ok(sealed6["sealed"] is True and "economic" not in sealed6
       and not any(k in sealed6 for k in ("n_fills_arm", "n_cancels_issued"))
       and unsealed4["sealed"] is False
       and isinstance(unsealed4.get("economic"), dict)
       and unsealed4["economic"]["D_E0"] == 2.5
       and unsealed4["n_fills_arm"] == 30171
       and unsealed4["n_cancels_issued"] == 5146,
       f"DRIVE 2: the SAME arm-day through the SAME `seal()` is "
       f"`sealed={sealed6['sealed']}` with the economics ABSENT at the "
       f"six-day bar, and `sealed={unsealed4['sealed']}` carrying D_E0, "
       f"the null block and the fill/cancel counts at the ruling's bar of "
       f"4. The read changes the BAR, never the computation")

    # ---- DRIVE 3: THE SEALED PATH IS BYTE-IDENTICAL OFF THE RULING ----
    # `early_read=None` is every path that is not this read. The property
    # is that such a path is untouched, and it is measured on the emitted
    # object, not read off the diff.
    for day in ("2026-09-07", "2026-09-08"):
        a = RUN.seal(copy.deepcopy({**arm, "day": day}), 1, 6)
        b = RUN.seal(copy.deepcopy({**arm, "day": day}), 1, 6)
        ok(json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
           and a["sealed"] is True and "economic" not in a,
           f"DRIVE 3: {day} -- a day OUTSIDE the ruling seals exactly as "
           f"before, `sealed=True`, economics absent. `early_read` "
           f"defaults to None on every such path")
    # THE KNOWN-BAD MUST FAIL FOR ITS OWN REASON. Driven through
    # `run_early_read_day`, whose membership check raises BEFORE any
    # heavy work: routed through `run_day` instead, the launch-form guard
    # refuses first and the cell passes on a refusal that has nothing to
    # do with the bar -- an instrument satisfying the words, not the
    # property.
    try:
        run_early_read_day("2026-09-07", None, tmp)
        ok(False, "KNOWN-BAD: a day outside the ruling entered the "
                  "early-read path")
    except EarlyReadRefused as e:
        ok(str(e).startswith("EARLY_READ_DAY_NOT_IN_THE_BAR"),
           f"DRIVE 3 KNOWN-BAD: the early-read path REFUSES 2026-09-07 BY "
           f"ITS OWN REASON CODE -- `{str(e).split(':')[0]}` -- before it "
           f"reads a book or takes a lock. A fifth day would be consumed "
           f"by a read nobody authorised")

    # ---- DRIVE 4: P9's REPLACEMENT, ALL THREE HALVES + THE REAL DAY ---
    # A fixture ledger root, so each half is driven against a root built
    # to trip exactly it. The real day is driven LAST, against the real
    # ledger, because a rehearsal that only ever saw fixtures has not
    # rehearsed anything.
    ruling_live = the_ruling()
    froot = Path(tempfile.mkdtemp(prefix="early_read_root_"))
    fder = froot / "pm_5min/derived"
    fder.mkdir(parents=True)

    # 4a: no sealed receipt at all
    try:
        early_read_preconditions("2026-09-03", froot, ruling_live)
        ok(False, "KNOWN-BAD: a day with NO sealed receipt was admitted")
    except EarlyReadRefused as e:
        ok(str(e).startswith("EARLY_READ_NO_SEALED_RECEIPT"),
           f"DRIVE 4a (RED): a day with no sealed day-run receipt REFUSES "
           f"-- `{str(e).split(':')[0]}`. With no sealed run there is "
           f"nothing this is the early read OF")

    # 4b: a sealed receipt whose digest is NOT the one the bar names
    entry = next(r for r in ruling_live["receipts"]
                 if r["day"] == "2026-09-03")
    (fder / Path(entry["path"]).name).write_text(
        '{"day": "2026-09-03", "not": "the ruled bytes"}')
    try:
        early_read_preconditions("2026-09-03", froot, ruling_live)
        ok(False, "KNOWN-BAD: a receipt off the bar was admitted")
    except EarlyReadRefused as e:
        ok(str(e).startswith("EARLY_READ_RECEIPT_NOT_THE_PAIR"),
           f"DRIVE 4b (RED): a sealed receipt AT THE RULED PATH whose "
           f"digest is not the ruled one REFUSES -- "
           f"`{str(e).split(':')[0]}`. The read opens the artifact the "
           f"ruling named, never whatever now sits at that path")

    # 4c: the ruled receipt is present, but this read already ran
    real_der = Path(RUN.DR.resolve()["data_root"]) / "pm_5min/derived"
    shutil.copy(real_der / Path(entry["path"]).name,
                fder / Path(entry["path"]).name)
    ok(early_read_preconditions("2026-09-03", froot,
                                ruling_live)["holds"] is True,
       "DRIVE 4c CONTROL (GREEN): with the RULED receipt in place the "
       "same fixture root ADMITS -- so 4a and 4b measured their own "
       "halves and not the fixture's emptiness")
    (fder / f"{DAY_FAMILY}_20260903__20260907T000000Z.json").write_text("{}")
    try:
        early_read_preconditions("2026-09-03", froot, ruling_live)
        ok(False, "KNOWN-BAD: a second early read of one day was "
                  "admitted")
    except EarlyReadRefused as e:
        ok(str(e).startswith("EARLY_READ_ALREADY_EMITTED"),
           f"DRIVE 4c (RED): a day that already has an early-read "
           f"artifact REFUSES -- `{str(e).split(':')[0]}`. Two answers to "
           f"one question with no rule saying which is newest is P9's own "
           f"reasoning, in the place it belongs for this entry")
    shutil.rmtree(froot, ignore_errors=True)

    # 4e: A v16-SHAPED BAR -- prefix only -- REFUSES BY ITS OWN NAME.
    # The known-bad is the SHAPE, not a wrong value: v16 is a landed
    # version and this is exactly what it carried, so the cell drives the
    # thing that really existed rather than a hand-made stub.
    prefix_only = {**ruling_live,
                   "receipts": [{k: v for k, v in r.items()
                                 if k != "sha256"}
                                for r in ruling_live["receipts"]]}
    froot2 = Path(tempfile.mkdtemp(prefix="early_read_prefix_"))
    (froot2 / "pm_5min/derived").mkdir(parents=True)
    shutil.copy(real_der / Path(entry["path"]).name,
                froot2 / "pm_5min/derived" / Path(entry["path"]).name)
    try:
        early_read_preconditions("2026-09-03", froot2, prefix_only)
        ok(False, "KNOWN-BAD: a prefix-only bar was compared on 16 hex")
    except EarlyReadRefused as e:
        ok(str(e).startswith("EARLY_READ_BAR_CARRIES_ONLY_A_PREFIX"),
           f"DRIVE 4e (RED): a v16-SHAPED bar -- `sha256_16` and no full "
           f"digest, against the RULED artifact whose prefix MATCHES -- "
           f"REFUSES by its own name, `{str(e).split(':')[0]}`. The "
           f"artifact is the right one and it still refuses: the point is "
           f"that a prefix is never silently substituted for a pair")
    shutil.rmtree(froot2, ignore_errors=True)

    # 4f: THE DRIVE THAT SEPARATES v17's CHECK FROM v16's. Every other
    # known-bad here would have fired under the PREFIX comparison too, so
    # none of them shows the ruling had any effect. This one is a bar
    # whose `sha256_16` MATCHES the artifact and whose full `sha256`
    # differs in its tail: v16's check ADMITS it, v17's REFUSES. The
    # delta is the whole content of the ruling.
    froot3 = Path(tempfile.mkdtemp(prefix="early_read_tail_"))
    (froot3 / "pm_5min/derived").mkdir(parents=True)
    shutil.copy(real_der / Path(entry["path"]).name,
                froot3 / "pm_5min/derived" / Path(entry["path"]).name)
    true_full = _sha(real_der / Path(entry["path"]).name)
    tail_wrong = true_full[:32] + ("f" * 32 if true_full[32] != "f"
                                   else "0" * 32)
    assert tail_wrong != true_full and tail_wrong[:16] == true_full[:16]
    tail_bar = {**ruling_live,
                "receipts": [{**r, "sha256": tail_wrong}
                             if r["day"] == "2026-09-03" else r
                             for r in ruling_live["receipts"]]}
    ok(tail_wrong.startswith(str(entry["sha256_16"])),
       f"DRIVE 4f SETUP: the planted digest {tail_wrong[:16]}… shares the "
       f"artifact's sixteen-hex prefix, so v16's check would have ADMITTED "
       f"it -- the baseline this known-bad measures a delta from")
    try:
        early_read_preconditions("2026-09-03", froot3, tail_bar)
        ok(False, "KNOWN-BAD: a bar agreeing on 16 hex and differing in "
                  "the tail was admitted -- the full digest is not being "
                  "compared")
    except EarlyReadRefused as e:
        ok(str(e).startswith("EARLY_READ_RECEIPT_NOT_THE_PAIR"),
           f"DRIVE 4f (RED, THE DELTA): a bar whose prefix MATCHES and "
           f"whose full digest differs only in its tail REFUSES -- "
           f"`{str(e).split(':')[0]}`. Under v16's prefix comparison this "
           f"same bar was admissible; under the ruling it is not. This is "
           f"the one cell here that the old check would have passed")
    shutil.rmtree(froot3, ignore_errors=True)

    # 4d: THE REAL DAY, against the REAL ledger.
    # 2026-09-03 WAS this cell's day until GO E1 read it: its early-read
    # artifact now exists, so `rehearse` correctly returns NOT_READY on
    # EARLY_READ_ALREADY_EMITTED -- the guard doing its job. The cell
    # moves to the next ruled day that has a sealed receipt and no early
    # read, and it asserts BOTH: that 09-03 now refuses BY THAT NAME, and
    # that an unread day is READY. A cell pinned to a day that has been
    # consumed measures the past.
    # BOTH DAYS ARE DERIVED FROM THE LEDGER, never typed.
    _st = bar_day_states()
    # REV 95 §A5: THE CELL ASSERTS ON WHICHEVER TERMINAL STATE EXISTS.
    # Requiring BOTH a read day and an unread one is a third literal --
    # about the bar's PROGRESS rather than a date -- and it goes red the
    # moment E4 reads the last day and `next_unread` becomes None. That is
    # the same defect as the hardcoded 09-03, one level up: a cell that
    # assumes the middle of the sweep.
    ok(bool(_st["read"]) or bool(_st["unread"]),
       f"REV 95 §A5: the bar's state is DERIVED and the cell asserts on "
       f"whichever terminal state EXISTS -- read {_st['read']}, unread "
       f"{_st['unread']}, next {_st['next_unread']}. Requiring both a read "
       f"and an unread day would go red when E4 finishes the sweep, which "
       f"is the hardcoded-date defect one level up")
    # ---- REV 96 §1: THE POST-E4 WORLD, DRIVEN IN A SCRATCH ROOT -------
    # The battery must read GREEN when EVERY day is read. Building that
    # state for real means waiting for E4; it is built here instead, in a
    # scratch ledger carrying an early-read artifact for all four bar days
    # plus the sealed receipts they need, and the same code path is run
    # against it. This is the state the unconditional `["preconditions"]`
    # would have raised in.
    _e4 = Path(tempfile.mkdtemp(prefix="post_e4_"))
    _e4d = _e4 / "pm_5min/derived"
    _e4d.mkdir(parents=True)
    _real_der = Path(RUN.DR.resolve()["data_root"]) / "pm_5min/derived"
    _rl = the_ruling()
    for _r in _rl["receipts"]:
        shutil.copy(_real_der / Path(_r["path"]).name,
                    _e4d / Path(_r["path"]).name)
        _c = _r["day"].replace("-", "")
        (_e4d / f"{DAY_FAMILY}_{_c}__20260907T000000Z.json").write_text("{}")
    _st_e4 = bar_day_states(root=_e4)
    _post = [rehearse(d, root=_e4) for d in _st_e4["read"]]
    ok(_st_e4["next_unread"] is None
       and len(_st_e4["read"]) == len(_rl["days"])
       and all(r["status"] == "NOT_READY"
               and r["blocking"] == ["EARLY_READ_ALREADY_EMITTED"]
               for r in _post)
       and all("preconditions" not in r for r in _post),
       f"REV 96 §1: IN THE POST-E4 WORLD -- all {len(_st_e4['read'])} bar "
       f"days read, `next_unread` None -- every day rehearses NOT_READY / "
       f"EARLY_READ_ALREADY_EMITTED and NONE carries `preconditions`. The "
       f"READY drive is inside the `else`, so the battery reads GREEN "
       f"here instead of raising the KeyError that was REV 94's NO-GO")
    shutil.rmtree(_e4, ignore_errors=True)

    if _st["read"]:
        _done = rehearse(_st["most_recently_read"])
        ok(_done["status"] == "NOT_READY"
           and _done["blocking"] == ["EARLY_READ_ALREADY_EMITTED"],
           f"DE 126: {_st['most_recently_read']} has been READ, so its "
           f"rehearsal refuses by name -- {_done['blocking']} -- rather "
           f"than offering to run it again. This is the POSITIVE CONTROL "
           f"that the abort was: the cell used to expect READY here")
    if _st["next_unread"] is None:
        # THE SWEEP IS DONE: every day is read, so every day must refuse.
        _all_done = [rehearse(d) for d in _st["read"]]
        ok(all(r["status"] == "NOT_READY"
               and r["blocking"] == ["EARLY_READ_ALREADY_EMITTED"]
               for r in _all_done),
           f"REV 95 §A5: with NOTHING unread, all {len(_all_done)} read "
           f"days rehearse NOT_READY / EARLY_READ_ALREADY_EMITTED. The "
           f"sweep is complete and the entry offers to run none of it "
           f"again")
    else:
        # REV 97 §A1: THE WHOLE DRIVE IS IN HERE -- the rehearse call,
        # every read of its result, AND the `ok()` that asserts on it. REV
        # 96 moved ONE STATEMENT: `reh` and `dc` were assigned inside this
        # branch and read OUTSIDE it, so the post-E4 world traded a
        # `KeyError` for an `UnboundLocalError` and the cell still could
        # not run in the world it exists to describe. A guard around one
        # statement is not a guard around the block.
        reh = rehearse(_st["next_unread"])
        dc = reh["preconditions"]["digest_comparison"]
        ok(reh["status"] == "READY" and reh["blocking"] == []
           and dc["is_a_full_pair"] is True
           and len(dc["bar_says"]) == 64
           and dc["bar_says"] == dc["artifact_is"]
           and dc["prefix_agrees_with_the_full_digest"] is True,
           f"DRIVE 4d (GREEN, THE NEXT UNREAD DAY {_st['next_unread']}): it "
           f"rehearses "
           f"{reh['status']}, blocking {reh['blocking']}, G {reh['G']}, class "
           f"{reh['verdict_class']}, interval {reh['interval']} -- and the "
           f"receipt check is now a FULL pair (`is_a_full_pair` True), 64 "
           f"hex compared and equal, with v16's prefix kept beside it and "
           f"agreeing")

    # ---- REV 91 §C1 / REV 93 #4: THE SHARED FALSIFIER, AS ONE CELL ---
    # The gap REV named is DETECTION COVERAGE, not correctness: the shared
    # module's own falsifier passes and ten other importers drive it, so a
    # regression there would be caught -- just not by THIS module's
    # battery, which resolves two chains through it (the params head for
    # the ruling, the exit-map head for the code). It is driven as a
    # SUBPROCESS, the same way `de_multiday_design_declaration` drives it,
    # and it runs no battery of ours so it cannot recurse.
    import subprocess as _sp
    _dcf = _sp.run([sys.executable,
                    str(Path(__file__).resolve().parent
                        / "declaration_chain.py"), "--falsify"],
                   capture_output=True, text=True, timeout=180)
    _last = (_dcf.stdout.strip().splitlines() or [""])[-1]
    ok(_dcf.returncode == 0 and "0 failures" in _last,
       f"REV 91 §C1 / REV 93 #4: `declaration_chain.py --falsify` runs as "
       f"ONE CELL of this battery -- rc {_dcf.returncode}, `{_last}`. This "
       f"module resolves TWO chains through that implementation and shipped "
       f"no drive of it for three rounds; the shared module's internal link "
       f"algebra is NOT re-tested here, only that its own falsifier still "
       f"fires")

    # ---- DA 122's CENSUS: AN IMPORTER OF declaration_chain SHIPS A ---
    # ---- FALSIFIER FOR THE SEAM IT USES ------------------------------
    # This module resolves TWO chains through the shared implementation
    # (the params head for the ruling, the exit-map head for the code),
    # and it shipped no cell proving either refusal can reach it. An
    # importer that never watches the shared refusal fire is an importer
    # that will read a refusal as an answer.
    _cr = Path(tempfile.mkdtemp(prefix="early_read_chain_"))
    (_cr / DECL_DIR).mkdir(parents=True)
    _fam = f"{PARAMS_FAMILY}"
    _v1 = _cr / DECL_DIR / f"{_fam}_v1.json"
    _v1.write_text(json.dumps({"version": 1}, sort_keys=True) + "\n")
    _v2 = _cr / DECL_DIR / f"{_fam}_v2.json"
    _v2.write_text(json.dumps(
        {"version": 2,
         "supersedes": {"path": str(_v1), "sha256": "0" * 64}},
        sort_keys=True) + "\n")
    _seam = None
    try:
        the_ruling(_cr)
    except DC.ChainRefused as _e:
        _seam = str(_e).split(":")[0]
    except EarlyReadRefused as _e:
        _seam = "SWALLOWED_AS_EARLY_READ_REFUSED"
    ok(_seam == "DECLARATION_LINK_CORRUPTED",
       f"DA 122's census: the SHARED chain resolver's refusal reaches "
       f"this importer BY ITS OWN NAME -- `{_seam}` -- when a version's "
       f"`supersedes` names a digest the file on disk does not have. It "
       f"is NOT caught and re-raised as this module's own refusal, which "
       f"would hide which layer refused; and it is not read as an answer")
    _shutil_cr = shutil
    _shutil_cr.rmtree(_cr, ignore_errors=True)

    # ---- DE 132: THE LEDGER IS WRITTEN, OR THE RUN REFUSES -----------
    # RED FIRST, on a FIXTURE day run -- the same `run_day` the early read
    # calls, with and without an anchor. GO E1 and GO E2 each spent ~90
    # minutes and emitted `decision_ledger: null`, because the condition
    # for writing R-765's ledger was `receipt_path is not None` and the
    # early read has no receipt path.
    _lt = Path(tempfile.mkdtemp(prefix="ledger_anchor_"))
    _made = RUN.write_synthetic_day("FIXTURE-DAY-1", str(_lt),
                                    params=RUN.load_params())
    _P132 = dict(RUN.load_params())
    # the name must be one the runner DECLARES as a fixture -- a guard
    # that exists because `fixture=True` alone once admitted a real day
    # name (REV 68 §1.2). My first draft invented one and was refused.
    _P132["days"] = ["FIXTURE-DAY-1"]
    _P132["G"] = 1
    _anchor132 = _lt / "the_artifact.json"
    _r132 = RUN.run_day("FIXTURE-DAY-1", _made["book_path"], params=_P132,
                        fixture=True, n_days_complete=1,
                        ledger_anchor=_anchor132)
    _b132 = _r132.get("decision_ledger")
    ok(isinstance(_b132, dict) and _b132.get("sha256")
       and _b132.get("n_rows", 0) > 0 and _b132.get("schema_version")
       and Path(_b132["path"]).is_file()
       and Path(_b132["path"]).parent == _anchor132.parent,
       f"DE 132 GREEN: given an ANCHOR the run writes its ledger BESIDE "
       f"the artifact -- {Path(_b132['path']).name}, {_b132['n_rows']} "
       f"rows, schema v{_b132['schema_version']}, sha256 "
       f"{_b132['sha256'][:16]}… -- and the block in `day_run` carries "
       f"path + sha256 + rows + schema. E1 and E2 carried `null` here")
    # DE 133: THE REFUSAL IS FOR A REAL DAY, and the distinction is the
    # correction. DE 132 refused ANY anchorless run, so it fired on the
    # RUNNER's own fixture cells -- which drive `run_day` with no anchor
    # and owe no ledger -- and E3's composition found it the moment the
    # cascade pin was fresh enough for those cells to run at all.
    _fx132 = RUN.run_day("FIXTURE-DAY-1", _made["book_path"], params=_P132,
                         fixture=True, n_days_complete=1)     # no anchor
    _fb132 = _fx132.get("decision_ledger") or {}
    ok(_fb132.get("status") == "NO_LEDGER_FOR_A_FIXTURE_DAY"
       and "REFUSES DECISION_LEDGER_HAS_NO_ANCHOR"
       in _fb132.get("a_real_day_without_an_anchor", ""),
       f"DE 133: a FIXTURE day with no anchor does NOT refuse -- it "
       f"records `{_fb132.get('status')}` and names what a REAL day would "
       f"do. A fixture's rows are synthetic; R-765 keeps the numbers of "
       f"real runs so they need not be re-run. DE 132's refusal did not "
       f"make that distinction and fired on this runner's own cells")
    # DRIVEN AT THE FUNCTION, because `run_day` cannot be driven this far
    # without the heavy lock -- the day-membership and lock guards refuse
    # first, correctly, and my first two versions of this cell accepted
    # THEIR refusal as if it were this one. The check is a named function
    # so it can be watched firing.
    _code132 = None
    try:
        RUN.assert_ledger_anchor(_P132, fixture=False, anchor=None)
    except RUN.RunnerRefused as _e:
        _code132 = str(_e).split(":")[0].replace("REFUSED ", "")
    _fxa132 = RUN.assert_ledger_anchor(_P132, fixture=True, anchor=None)
    _rok132 = RUN.assert_ledger_anchor(_P132, fixture=False,
                                       anchor="/tmp/x.json")
    ok(_code132 == "DECISION_LEDGER_HAS_NO_ANCHOR"
       and _fxa132["owes_a_ledger"] is False
       and _fxa132["status"] == "NO_LEDGER_FOR_A_FIXTURE_DAY"
       and _rok132["owes_a_ledger"] is True,
       f"DE 132/133 RED: a REAL day with no anchor REFUSES BY ITS OWN "
       f"NAME -- `{_code132}` -- BEFORE the day's work, not after ~90 "
       f"minutes of it (R-610's principle; DE 132 put the check at the "
       f"ledger write, where it could not be driven cheaply). A "
       f"null block is the SILENT form of promising a ledger that is not "
       f"there, which is why two heavy runs passed every review without "
       f"one")
    # ---- R-795 / BE 97: THE INVENTORY CLAIM IS MEASURED AT THE FILE --
    # RED FIRST, and the red is the LANDED SENTENCE: this artifact said
    # `inventory_leg: COMPUTED since BE 96, in the decision ledger`.
    # THE TRUTH DEPENDS ON THE FILE, which is the whole reason it must be
    # measured and not typed. BE 97 walked the 09-05 ledger -- written by
    # the ledger module at `6c3a121`, whose ARM_SCALARS row carries no
    # `absolute` block -- and found no such key at any depth. A ledger
    # written by the CURRENT module carries DE 131's absolutes there, and
    # `absolute_legs` names one of their components `inventory_leg`. Both
    # files exist; a sentence typed either way is wrong for one of them.
    import gzip as _gz795
    _inv795 = inventory_at_the_ledger(_b132["path"],
                                      arm_days=_r132.get(
                                          "per_day_sealed_artifacts"))
    ok(_inv795["status"] == "NOT_A_FIELD_OF_THE_LEDGER"
       and _inv795["no_fill_row_carries_a_leg"] is True
       and all(w.startswith("ARM_SCALARS.absolute.")
               for w in _inv795["where_the_name_occurs"])
       and _inv795["the_five_inventory_FIELDS_the_file_declares"] == [
           "inventory_before", "inventory_after", "inventory_unit",
           "inventory_mark_cents", "inventory_mark_source"]
       and _inv795["measured_at"]["rows_scanned"] == _b132["n_rows"],
       f"R-795, MEASURED AT A LEDGER THIS BATTERY JUST WROTE: the name "
       f"`inventory_leg` occurs ONLY at "
       f"{sorted(_inv795['where_the_name_occurs'])} -- inside DE 131's "
       f"absolutes, never on a FILL row -- so it is a component of the "
       f"absolutes and NOT a leg of the day value. What the fill rows "
       f"carry is the FIVE BE-96 FIELDS, read from the file's own HEADER "
       f"row: the inputs a leg would be computed FROM")
    # THE 09-05 SHAPE, REPRODUCED: strip the absolutes (the ledger module
    # at `6c3a121` wrote none) and the name is gone from the file --
    # which is exactly what BE 97 measured, driven here rather than
    # quoted from a review.
    _noabs795 = Path(_lt) / "no_absolutes_ledger.jsonl.gz"
    with _gz795.open(_b132["path"], "rt") as _src, \
            _gz795.open(_noabs795, "wt") as _dst:
        for _line795 in _src:
            _r795 = json.loads(_line795)
            _r795.pop("absolute", None)
            _dst.write(json.dumps(_r795, sort_keys=True) + "\n")
    _n795 = inventory_at_the_ledger(_noabs795, arm_days=[])
    ok(_n795["status"] == "NOT_A_FIELD_OF_THE_LEDGER"
       and _n795["rows_carrying_an_inventory_leg_key_at_any_depth"] == 0
       and _n795["where_the_name_occurs"] == {},
       f"R-795 RED, THE LANDED SENTENCE REFUTED: with the absolutes "
       f"stripped -- the shape the 09-05 and 09-06 ledgers actually have "
       f"-- the name occurs on 0 of "
       f"{_n795['measured_at']['rows_scanned']} rows at any depth, so "
       f"'COMPUTED since BE 96, in the decision ledger' is FALSE of "
       f"those files. BE 97's measurement, reproduced here")
    # THE INSTRUMENT MUST BE ABLE TO SAY 'OUTSIDE THE ABSOLUTES'. A
    # classifier that can only return the two states it has seen is not
    # a classifier.
    _plant795 = Path(_lt) / "planted_ledger.jsonl.gz"
    with _gz795.open(_b132["path"], "rt") as _src, \
            _gz795.open(_plant795, "wt") as _dst:
        for _i795, _line795 in enumerate(_src):
            _r795 = json.loads(_line795)
            if _r795.get("row") == "FILL" and _i795 % 50 == 0:
                # NESTED and on a FILL row: a per-fill leg would look
                # like this, and it must NOT read as the absolutes case
                _r795["legs"] = {"inner": {"inventory_leg": 1.0}}
            _dst.write(json.dumps(_r795, sort_keys=True) + "\n")
    _p795 = inventory_at_the_ledger(_plant795, arm_days=[])
    ok(_p795["status"] == "PRESENT_OUTSIDE_THE_ABSOLUTES"
       and _p795["no_fill_row_carries_a_leg"] is False
       and any(w.startswith("FILL.") for w in _p795["where_the_name_occurs"]),
       f"R-795 POSITIVE CONTROL ON THE CLASSIFIER: with the name PLANTED "
       f"NESTED ON FILL ROWS the status becomes `{_p795['status']}` and "
       f"`no_fill_row_carries_a_leg` False -- so the two readings above "
       f"are measurements and not the only answers this walk can give")
    # AND THE FIVE NAMES COME FROM THE FILE, NOT FROM THIS MODULE.
    _hdr795 = Path(_lt) / "reheadered_ledger.jsonl.gz"
    with _gz795.open(_b132["path"], "rt") as _src, \
            _gz795.open(_hdr795, "wt") as _dst:
        for _line795 in _src:
            _r795 = json.loads(_line795)
            if _r795.get("row") == "HEADER":
                _r795["inventory_fields_from_BE_96"] = ["ONLY_THIS_ONE"]
            _dst.write(json.dumps(_r795, sort_keys=True) + "\n")
    _h795 = inventory_at_the_ledger(_hdr795, arm_days=[])
    ok(_h795["the_five_inventory_FIELDS_the_file_declares"]
       == ["ONLY_THIS_ONE"]
       and _h795["fill_rows_carrying_all_five_fields"] == 0,
       f"R-795: THE NAMES FOLLOW THE FILE -- rewrite the HEADER's "
       f"`inventory_fields_from_BE_96` to "
       f"{_h795['the_five_inventory_FIELDS_the_file_declares']} and the "
       f"block reports THAT, with 0 fill rows carrying it. The list is "
       f"read from the ledger's own header, so it cannot drift from the "
       f"file the way the sentence it replaces did")
    _none795 = inventory_at_the_ledger(None)
    _gone795 = inventory_at_the_ledger(Path(_lt) / "no_such_ledger.gz")
    ok(_none795["status"] == "NOT_MEASURED_NO_LEDGER_ON_THIS_PATH"
       and _gone795["status"] == "NOT_MEASURED_LEDGER_ABSENT"
       and "COMPUTED" not in json.dumps(_none795)
       and "COMPUTED" not in json.dumps(_gone795),
       f"R-795: with no ledger, or one named and absent, the entry says "
       f"`{_none795['status']}` / `{_gone795['status']}` and asserts "
       f"NOTHING about a file it did not read -- it never falls back to "
       f"the sentence BE 97 refuted")
    shutil.rmtree(_lt, ignore_errors=True)

    # ---- REV 89 item 1: THE CAPTURE RECORD'S THREE DECLARED FIELDS ---
    _r0 = resolve_exit(0)
    _r1 = resolve_exit(1)
    _r9 = resolve_exit(9)                      # never declared
    ok(_r0["producer_module"] == "live/pm_research/de_early_read.py"
       and _r0["mapped_by"]["sha256"] and _r0["mapped_by"]["version"]
       and _r0["resolved_kind"] and _r1["resolved_kind"]
       and _r9["resolved_kind"] == "UNMAPPED"
       and _r0["75_is_never_used"] is True,
       f"REV 89 item 1: an exit code resolves through the map's CHAIN "
       f"HEAD (v{_r0['mapped_by']['version']}, "
       f"{_r0['mapped_by']['sha256'][:16]}…) and yields all three fields "
       f"the capture record's own declaration requires -- "
       f"producer_module, mapped_by, resolved_kind. An undeclared code "
       f"resolves to `{_r9['resolved_kind']}`, which does not satisfy a "
       f"GO; it is a STATUS, not a silent pass")

    # ---- R-709: THE EXIT MAP -- THE STATIC HALF, HERE ----------------
    # WHAT THIS CELL DOES NOT DO, AND WHY. It does not spawn this module
    # to observe its exit codes. Every path of this module that is not
    # bare usage runs the battery FIRST (R-610: what can refuse refuses
    # before the work), so a subprocess probe launched from inside the
    # battery re-enters it and spawns again. Two versions of this cell
    # did that -- ~100 processes, then ~422, both killed by pid -- and
    # the second still recursed with a guard in place, because the guard
    # was on the wrong path. The observation lives in
    # `--observe-exit-codes`, which does NOT run the battery and so
    # cannot recurse; this cell asserts only what is safe to assert from
    # inside.
    declared = set(EXIT_CODES)
    ok(declared == {0, 1} and 75 not in declared
       and all(isinstance(k, int) for k in declared)
       and all(isinstance(v, str) and v for v in EXIT_CODES.values()),
       f"R-709 (static): this producer declares exactly {sorted(declared)} "
       f"with a reason for each, and 75 is NOT among them -- 75 is the "
       f"launcher's held-lock code (`flock -n -E 75`), and a producer "
       f"claiming it would make a lock conflict unreadable. The OBSERVED "
       f"half is `--observe-exit-codes`, out of the battery on purpose")

    # ---- the absent fields are STATUSES, never silent drops -----------
    av = economics_available_per_arm_day()
    _absent = av["not_computed_by_this_path_the_ARM_DAY_BLOCK"]
    _where = av["where_the_five_live_now"]
    # R-795: `inventory_leg` IS NO LONGER A SENTENCE. It is the block
    # `inventory_at_the_ledger()` measured at the file this run wrote,
    # so this cell reads a dict there and a string for the other four --
    # and it asserts that the entry is a MEASUREMENT (it carries a
    # status, and either what it measured or why it did not). R-803
    # renamed it: there is no inventory leg to report on.
    _strs = {k: v for k, v in _where.items() if isinstance(v, str)}
    _inv = _where.get("trades_cash_flow_cents")
    ok(len(_absent) == 5
       and all(isinstance(v, str) and len(v) > 40 for v in _absent.values())
       and set(_where) == set(_absent)
       and sum(1 for v in _strs.values() if v.startswith("COMPUTED")) == 3
       and _strs["D_E_MINUS_R"].startswith("STILL NOT COMPUTED")
       and isinstance(_inv, dict) and _inv.get("status")
       and ("measured_at" in _inv or "why" in _inv),
       f"DE 129 / R-795: the five are named with their reasons and with "
       f"WHERE THEY LIVE NOW -- three "
       f"({sorted(k for k, v in _strs.items() if v.startswith('COMPUTED'))}) "
       f"are computed in the DECISION LEDGER since R-765 and BE 96; the "
       f"entry once called `inventory_leg` is now "
       f"`trades_cash_flow_cents` (R-803, BE 99: that quantity is the "
       f"trades cash flow with the opposite sign, and the residual is "
       f"priced at SETTLEMENT in the R-801 legs) and is a MEASUREMENT at "
       f"the ledger this run wrote, status `{_inv.get('status')}`; and "
       f"only `D_E_MINUS_R` is computed nowhere"
       )

    shutil.rmtree(tmp, ignore_errors=True)
    _sum = RUN.battery_summary("de_early_read", n_run=n[0],
                               skipped=skipped, expected=EXPECTED_CHECKS)
    if not quiet:
        print(_sum["line"])
    if not _sum["clean"]:
        raise SystemExit(f"[de_early_read] NOT CLEAN: {_sum['line']}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--early-read-day", type=str, dest="day",
                    help="ONE day of the ruling, UNSEALED under v16's "
                         "bar. Requires --book and --output, and the "
                         "heavy-run lock (this is a real day's work).")
    ap.add_argument("--observe-exit-codes", action="store_true",
                    dest="observe_exit_codes",
                    help="spawn this module on its usage paths and report "
                         "the exit codes OBSERVED against the declared "
                         "map (R-709). Runs no battery, so it cannot "
                         "recurse.")
    ap.add_argument("--book", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if a.observe_exit_codes:
        # NO BATTERY HERE. That is the whole point: this entry exists so
        # the exit codes can be OBSERVED without the observer being the
        # thing observed.
        import subprocess
        me = str(Path(__file__).resolve())
        obs = {}
        for label, argv in (
                ("usage: no arguments", []),
                ("usage: --early-read-day without --book",
                 ["--early-read-day", "2026-09-03"])):
            obs[label] = subprocess.run(
                [sys.executable, me, *argv], capture_output=True,
                timeout=60).returncode
        print(json.dumps({
            "declared": {str(k): v for k, v in EXIT_CODES.items()},
            "observed": obs,
            "observed_are_declared": set(obs.values()) <= set(EXIT_CODES),
            "75_is_declared": 75 in EXIT_CODES,
            "not_probed_here": {
                "0 (a completed day)": "heavy -- it is a GO, not a probe",
                "1 via an EARLY_READ_* refusal": "that path runs the "
                    "battery first (R-610), so probing it from a battery "
                    "cell recursed; it is reachable by hand and its "
                    "refusal text is in the battery's own drives"},
        }, indent=2))
        raise SystemExit(0)
    if a.day:
        if a.book is None or a.output is None:
            raise SystemExit("REFUSED: --early-read-day requires --book "
                             "and --output")
        # THE BATTERY BEFORE THE WORK (R-610): everything that can refuse
        # refuses before the day is spent, not after 80 minutes of draws.
        selftest(quiet=True)
        out = run_early_read_day(a.day, a.book, a.output,
                                 before_work=lambda: RUN.selftest(
                                     quiet=True, offline=False))
        print(json.dumps(out, indent=2))
        raise SystemExit(0)
    raise SystemExit("usage: --selftest | --early-read-day DAY --book B "
                     "--output DIR")
