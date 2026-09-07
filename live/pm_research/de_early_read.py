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

EXPECTED_CHECKS = 16


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
    return {
        "ruling_by_pair": {"path": head["path"], "sha256": head["sha256"]},
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
NOT_COMPUTED_BY_THIS_PATH = {
    "fills_leg": "the arm-day economic block is a SINGLE excess `D_E0` "
                 "against the 0-cancel baseline. There is no fills/"
                 "inventory decomposition anywhere in the runner, so a "
                 "leg split cannot be reported without defining one.",
    "inventory_leg": "as above -- the same single quantity is not two.",
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


def economics_available_per_arm_day() -> dict:
    """WHAT AN UNSEALED ARM-DAY ACTUALLY CARRIES, read off the emitter."""
    return {
        "from_the_economic_block": ["D_E0", "Z", "p_location",
                                    "null_mean", "null_sd",
                                    "null_draws_summary.n"],
        "from_the_arm_level": ["n_fills_arm", "n_fills_baseline",
                               "n_cancels_issued", "status",
                               "admissibility", "seed", "draw_provenance"],
        "not_computed_by_this_path": NOT_COMPUTED_BY_THIS_PATH,
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
    result = RUN.run_day(day, book, params=params, fixture=False,
                         n_days_complete=ruling["G"],
                         early_read={"G": ruling["G"],
                                     "days": ruling["days"]},
                         before_work=before_work)
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
        "computation_params": {
            "path": RUN.PARAMS_REL,
            "sha256": _sha(Path(__file__).resolve().parents[2]
                           / RUN.PARAMS_REL),
            "why": "the COMPUTATION is the sealed runs' -- v15. Only the "
                   "seal bar comes from the ruling."},
        "economics_field_availability": economics_available_per_arm_day(),
        "preconditions": pre,
        "day_run": result,
        "as_of": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = Path(outdir) / day_artifact_name(day)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True,
                              default=str) + "\n")
    return {"path": str(out), "sha256": _sha(out), "day": day}


# ------------------------------------------------------- the battery

def selftest(quiet: bool = False) -> int:
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

    # 4d: THE REAL DAY, against the REAL ledger, under v17's bar
    reh = rehearse("2026-09-03")
    dc = reh["preconditions"]["digest_comparison"]
    ok(reh["status"] == "READY" and reh["blocking"] == []
       and dc["is_a_full_pair"] is True
       and len(dc["bar_says"]) == 64 and dc["bar_says"] == dc["artifact_is"]
       and dc["prefix_agrees_with_the_full_digest"] is True,
       f"DRIVE 4d (GREEN, THE REAL DAY): 2026-09-03 rehearses "
       f"{reh['status']}, blocking {reh['blocking']}, G {reh['G']}, class "
       f"{reh['verdict_class']}, interval {reh['interval']} -- and the "
       f"receipt check is now a FULL pair (`is_a_full_pair` True), 64 hex "
       f"compared and equal, with v16's prefix kept beside it and "
       f"agreeing")

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
    ok(len(av["not_computed_by_this_path"]) == 5
       and all(isinstance(v, str) and len(v) > 40
               for v in av["not_computed_by_this_path"].values()),
       f"the {len(av['not_computed_by_this_path'])} fields the dispatch "
       f"asked for that this path does not compute are named with their "
       f"reasons -- {sorted(av['not_computed_by_this_path'])} -- not "
       f"dropped in silence (reliability rule 4)")

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
