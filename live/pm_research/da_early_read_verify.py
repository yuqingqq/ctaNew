#!/usr/bin/env python3
"""DA -- THE INDEPENDENT READER OF THE EARLY-READ FAMILY (R-754, R-757).

WHAT THIS IS FOR. The USER ruled at R-754 that the four sealed Gate-1 days be
read NOW -- *"it does not make sense to seal the results, show me 4 days
results first, we need to check and review the results"*. DE emits that read as
a NEW family, `p003_de_early_read_day_<D>__<stamp>.json`. This module is the
SECOND implementation that stands between those artifacts and the coordinator's
table: it re-derives nothing DE computed and re-implements every CHECK.

R-235, DO-NOT-HARMONIZE. It imports no module of DE's. The ruling, the bar, the
labels and the field lists are read as DOCUMENTS from the landed declaration and
from the artifact; the vocabulary this reader judges against is declared HERE,
with its source named beside it. Two implementations that agree are evidence;
one implementation checking itself is not.

THE ANTI-ECHO CENSUS IS INVERTED FOR THIS FAMILY, and that inversion is the
whole reason this reader exists. Everywhere else in this programme an economic
name in an emission is a LEAK. Here the economics are what the USER ruled may be
seen -- so the census asks the opposite question: does the arm-day block carry
EXACTLY the six computed economic fields plus the three counts, and nothing
else economic? A field that appears without having been computed would be an
invention, and a field that quietly disappears would be a table missing a
column nobody notices.

WHAT IT WILL NOT DO.
  * It never opens an artifact of the SEALED family. Those receipts are sealed
    and this reader has no business quoting from them; fed one, it REFUSES BY
    NAME (`SEALED_FAMILY_ARTIFACT_REFUSED`).
  * It resolves the ruling BY THE PAIR THE ARTIFACT RECORDS -- never by the
    family head. The head is for writers; a reader of a past act resolves what
    that act recorded (R-729, REV 86 S8). An artifact written under v16 stays
    checkable after v17 lands, and it is checked against v16.
  * It computes no economic quantity of its own, and it invents nothing: the
    five fields the runs never produced are STATUSES here as they are there,
    and a status carrying a number is a refusal.

    python3 live/pm_research/da_early_read_verify.py --selftest
    python3 live/pm_research/da_early_read_verify.py --verify <artifact> [--print]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

PROTOCOL = "P003_DA_EARLY_READ_VERIFIER_V1"

#: The family this reader accepts, and the one it must never open.
EARLY_FAMILY = "p003_de_early_read_day"
SEALED_FAMILY_MARKERS = ("p003_de_gate1_day_run_", "_SEALED__")
EARLY_PROTOCOL = "P003_DE_EARLY_READ_DAY_V1"

#: The ruling lives in the params family. RESOLVED BY THE PAIR THE ARTIFACT
#: RECORDS, never by this constant and never by the head -- the family name is
#: here only so the recorded path can be checked to BE a params version.
RULING_FAMILY = "de_multiday_gate1_params"
DECL_DIR = "live/pm_research/declarations"

#: THE LABELS THE RULING FIXES (R-754 as landed in the block). Each is checked
#: for PRESENCE and for VALUE, and a difference is refused by name -- a read
#: that quietly relabelled itself a validation would be the whole failure.
REQUIRED_LABELS = {
    "is_a_validation": False,
    "G": 4,
    "verdict_class": "EXPLORATORY",
    "interval": "NONE_BELOW_FIVE_DAYS",
}
DAYS_CONSUMED = ["2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"]

#: THE FIVE FIELDS THE RUNS NEVER COMPUTED (R-757 ruling (2)). They are named
#: here, in this reader's own words, because a checker that read the list off
#: the artifact it is checking would pass any artifact that shortened it.
NOT_COMPUTED_KEYS = ("fills_leg", "inventory_leg", "p_two_sided",
                     "rho_adverse_over_spread", "D_E_MINUS_R")

#: WHAT AN ARM-DAY MAY CARRY. Six computed economic fields and three counts.
ECON_REQUIRED = ("D_E0", "Z", "p_location", "null_mean", "null_sd",
                 "null_draws_summary")
COUNTS_REQUIRED = ("n_fills_arm", "n_fills_baseline", "n_cancels_issued")
#: AND THE VOCABULARY IT MAY NOT. Sources, named: the ELEVEN-name seal scope of
#: 09-04/05/06 and the EIGHT-name scope of 09-03 (read off those receipts'
#: `sealed_field_names`), plus the five never-computed names above and the
#: words R-754's dispatch asked for. A name in this vocabulary that is not in
#: the allowed set is an INVENTION or a field from another computation.
ECON_VOCAB = set(ECON_REQUIRED) | set(COUNTS_REQUIRED) | {
    "D_E_MINUS_R", "sd_over_abs_mean", "rho", "adverse", "spread",
    "fills_leg", "inventory_leg", "p_two_sided", "p_two_sided_location",
    "adverse_over_spread", "rho_adverse_over_spread", "eff_RT",
    "value_cents", "net_cents", "D_E0_MINUS_R",
}
ALLOWED_ECON = set(ECON_REQUIRED) | set(COUNTS_REQUIRED)

#: The label line that rides on EVERY print (R-754's consequences, carried
#: with the numbers rather than in a covering note nobody reads).
LABEL_LINE = ("EXPLORATORY, G 4, point estimates, NO INTERVAL, days consumed")

DIGEST64 = re.compile(r"^[0-9a-f]{64}$")

#: WHERE THE THREE COUNTS CAME FROM, PER DAY (REV 90 S A2, recorded in the
#: ruling's own `blindness_notes`). 2026-09-03 was sealed under an
#: EIGHT-name scope, so `n_fills_arm`, `n_fills_baseline` and
#: `n_cancels_issued` have been READABLE IN THE OPEN since its receipt
#: landed; 09-04, 09-05 and 09-06 were sealed under the ELEVEN-name scope
#: that includes them, so for those days this read is what unseals them.
#: ***The table says which, per day***: a reader comparing four days'
#: counts is comparing three that this read opened with three that anyone
#: could have seen while the later days were being run, and that is a fact
#: about BLINDNESS, not about the numbers.
COUNTS_PROVENANCE = {
    "2026-09-03": ("VISIBLE IN THE OPEN since 2026-09-06T14:01Z -- 09-03 "
                   "was sealed under the EIGHT-name scope, which does not "
                   "include the three counts"),
    "_default": ("unsealed BY THIS READ -- sealed under the ELEVEN-name "
                 "scope, which includes the three counts"),
}


def counts_provenance(day: str) -> str:
    return COUNTS_PROVENANCE.get(str(day), COUNTS_PROVENANCE["_default"])


class EarlyReadVerifyRefused(RuntimeError):
    """A named refusal. Every message begins with its own reason code."""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------- the reads
def load_artifact(path, *, repo_root=None) -> dict:
    """The artifact, or a REFUSAL BY NAME. Never a sealed-family file."""
    p = Path(path)
    name = p.name
    if any(m in name for m in SEALED_FAMILY_MARKERS):
        raise EarlyReadVerifyRefused(
            f"SEALED_FAMILY_ARTIFACT_REFUSED: {name} is a SEALED Gate-1 day "
            f"receipt, not an early-read artifact. This reader exists to read "
            f"the family the USER ruled OPEN; the sealed family is sealed, and "
            f"a reader that would quote from it on being handed one is a way "
            f"of unsealing it by mistake.")
    if not p.is_file():
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_ARTIFACT_ABSENT: no artifact at {p}. The artifact is "
            f"written by DE's read; its EXISTENCE is what says the read has "
            f"happened, and this reader will not be told that it did.")
    try:
        doc = json.loads(p.read_bytes())
    except json.JSONDecodeError as e:
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_ARTIFACT_UNREADABLE: {name} is not readable JSON "
            f"({e.msg} at line {e.lineno}). Present and unparseable is not the "
            f"same as absent.")
    if doc.get("protocol") != EARLY_PROTOCOL:
        raise EarlyReadVerifyRefused(
            f"NOT_AN_EARLY_READ_ARTIFACT: {name} carries protocol "
            f"{doc.get('protocol')!r}, not {EARLY_PROTOCOL!r}.")
    if not name.startswith(EARLY_FAMILY):
        raise EarlyReadVerifyRefused(
            f"NOT_AN_EARLY_READ_ARTIFACT: {name} is not of the family "
            f"{EARLY_FAMILY}_<day>__<stamp>.json.")
    return doc


def the_ruling_by_the_pair(doc: dict, *, repo_root=None) -> dict:
    """THE RULING THE ARTIFACT RECORDS, resolved BY ITS PAIR.

    The head is consulted only to SAY whether the recorded version is the
    current head or a superseded one; both are fine and the record states
    which. An artifact written under v16 is checked against v16 after v17
    lands -- that is what "the pair the act recorded" means (R-729).
    """
    root = Path(repo_root) if repo_root else HERE.parents[1]
    pair = doc.get("ruling")
    if not isinstance(pair, dict) or not pair.get("path") \
            or not DIGEST64.match(str(pair.get("sha256", ""))):
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_RULING_NOT_THE_PAIR: the artifact's `ruling` is "
            f"{pair!r}, which is not a {{path, sha256}} pair with a 64-hex "
            f"digest. A ruling cited by name alone is an address that "
            f"verifies nothing.")
    name = Path(str(pair["path"])).name
    if not name.startswith(RULING_FAMILY + "_v"):
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_RULING_NOT_THE_PAIR: the artifact names {name}, "
            f"which is not a version of {RULING_FAMILY}. The authority for "
            f"this read is a USER ruling landed in that family and nothing "
            f"else.")
    decl_dir = root / DECL_DIR
    f = decl_dir / name
    if not f.is_file():
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_RULING_NOT_THE_PAIR: {name} is not present under "
            f"{decl_dir}. A check that depends on a declaration FAILS when it "
            f"is gone; it does not skip (R-649).")
    got = _sha(f)
    if got != pair["sha256"]:
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_RULING_NOT_THE_PAIR: {name} digests {got[:16]}… and "
            f"the artifact records {str(pair['sha256'])[:16]}…. The bytes the "
            f"read cited are not the bytes on disk, so what it was authorised "
            f"to do cannot be established.")
    doc_r = json.loads(f.read_text())
    block = doc_r.get("user_ruled_early_read")
    if not isinstance(block, dict):
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_RULING_NOT_THE_PAIR: {name} carries no "
            f"`user_ruled_early_read` block, so it is not the ruling this "
            f"read claims.")
    head = None
    try:
        import declaration_chain as CHAIN                     # noqa: PLC0415
        h = CHAIN.resolve_head(decl_dir, RULING_FAMILY)
        head = {"name": h["name"], "sha256": h["sha256"]}
    except Exception as e:                                    # noqa: BLE001
        head = {"name": None, "why": f"{type(e).__name__}: {e}"}
    return {"name": name, "sha256": got, "block": block,
            "version": doc_r.get("version"),
            "is_the_current_head": bool(head and head.get("sha256") == got),
            "the_current_head": head,
            "resolved_by": "the pair the artifact recorded, verified here",
            "why_not_the_head": (
                "the head is for writers; a reader of a past act resolves "
                "what that act recorded (R-729, REV 86 S8). This artifact "
                "stays checkable after a later version lands")}


def the_bar_for_the_day(ruling: dict, day: str) -> dict:
    """The day's row in `this_reads_bar`, with the FULL digest or a refusal."""
    bar = (ruling["block"].get("this_reads_bar") or {})
    rows = bar.get("receipts") or []
    row = next((r for r in rows if str(r.get("day")) == str(day)), None)
    if row is None:
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_DAY_NOT_IN_THE_BAR: {day} is not among "
            f"{[r.get('day') for r in rows]}. The ruling names the days it "
            f"opens; a day outside it was consumed by no ruling.")
    full = row.get("sha256")
    if not (isinstance(full, str) and DIGEST64.match(full)):
        raise EarlyReadVerifyRefused(
            f"PREFIX_ONLY: {ruling['name']}'s bar carries no full sha256 for "
            f"{day} -- it has {list(row)} and the comparable digest is "
            f"{row.get('sha256_16')!r}, a SIXTEEN-HEX PREFIX. A prefix is not "
            f"the pair, and this reader does not silently compare on 16 hex: "
            f"the strength of a check must not depend on which version "
            f"happens to be cited. Cite a ruling version whose bar carries "
            f"full digests.")
    return {"day": day, "path": row.get("path"), "sha256": full,
            "prefix_beside_it": row.get("sha256_16"),
            "is_the_chain_head_says_the_bar": row.get("is_the_chain_head"),
            "digest_is_full": True}


def check_receipt_against_the_bar(doc: dict, bar_row: dict,
                                  *, data_root=None) -> dict:
    """The sealed receipt the ARTIFACT names is the one the BAR names.

    The receipt itself is NOT opened -- it is sealed and this reader has no
    business inside it. Its NAME and its DIGEST are what is compared, and
    where the file is on disk the digest is re-derived from the bytes rather
    than taken from either document.
    """
    pre = (doc.get("preconditions") or {})
    got = (pre.get("sealed_receipt") or {})
    name, sha = Path(str(got.get("path", ""))).name, got.get("sha256")
    want_name, want_sha = Path(str(bar_row["path"])).name, bar_row["sha256"]
    out = {"artifact_names": {"name": name, "sha256": sha},
           "the_bar_names": {"name": want_name, "sha256": want_sha},
           "the_receipt_was_not_opened": True,
           "why_not_opened": ("it is a SEALED receipt; this reader compares "
                              "its identity and never its contents")}
    if name != want_name or sha != want_sha:
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_RECEIPT_NOT_THE_BAR: the artifact names "
            f"{name} at {str(sha)[:16]}… and the ruling's bar names "
            f"{want_name} at {str(want_sha)[:16]}…. A read of a day is a read "
            f"of ONE receipt, and which one is not a detail.")
    if data_root:
        f = Path(data_root) / "pm_5min/derived" / want_name
        if f.is_file():
            on_disk = _sha(f)
            out["on_disk_sha256"] = on_disk
            out["on_disk_agrees"] = (on_disk == want_sha)
            if on_disk != want_sha:
                raise EarlyReadVerifyRefused(
                    f"EARLY_READ_RECEIPT_NOT_THE_BAR: {want_name} on disk "
                    f"digests {on_disk[:16]}…, not the bar's "
                    f"{want_sha[:16]}…. The day's chain head moved under the "
                    f"read that cited it.")
        else:
            out["on_disk_sha256"] = None
            out["on_disk_agrees"] = None
            out["why_no_disk_check"] = (
                "the receipt is not under this data root; the identity check "
                "above stands and is said to be the only one that ran")
    return out


def check_labels(doc: dict, ruling: dict) -> dict:
    """The labels R-754 fixed. Missing and DIFFERENT are separate refusals."""
    out = {}
    for k, want in REQUIRED_LABELS.items():
        if k not in doc:
            raise EarlyReadVerifyRefused(
                f"EARLY_READ_LABEL_MISSING: the artifact carries no `{k}`. "
                f"The labels are what stop this read being read as something "
                f"it is not, and an absent label is not a default.")
        if doc[k] != want:
            raise EarlyReadVerifyRefused(
                f"EARLY_READ_LABEL_DIFFERS: `{k}` is {doc[k]!r} and the "
                f"ruling fixes {want!r}.")
        out[k] = doc[k]
    days = doc.get("days_consumed")
    if days is None:
        raise EarlyReadVerifyRefused(
            "EARLY_READ_LABEL_MISSING: the artifact carries no "
            "`days_consumed`. Rule 11: the days this read spends are part of "
            "the result, not a footnote.")
    if list(days) != DAYS_CONSUMED:
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_LABEL_DIFFERS: `days_consumed` is {list(days)} and "
            f"the ruling consumes {DAYS_CONSUMED}.")
    ruled = list(ruling["block"].get("days_consumed_by_this_read") or [])
    if ruled != DAYS_CONSUMED:
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_LABEL_DIFFERS: the RULING's own "
            f"`days_consumed_by_this_read` is {ruled}, not {DAYS_CONSUMED}. "
            f"This reader's constant and the declaration disagree, and it "
            f"refuses rather than preferring either.")
    out["days_consumed"] = list(days)
    out["checked_against"] = ("this reader's own constants AND the ruling "
                              "block, which must agree")
    return out


def check_not_computed(doc: dict) -> dict:
    """The five uncomputed fields: NAMED STATUSES with reasons, never numbers."""
    blk = ((doc.get("economics_field_availability") or {})
           .get("not_computed_by_this_path"))
    if not isinstance(blk, dict):
        raise EarlyReadVerifyRefused(
            "EARLY_READ_STATUS_MISSING: the artifact carries no "
            "`economics_field_availability.not_computed_by_this_path`. The "
            "five fields R-754 asked for were never computed for these days; "
            "their ABSENCE is a status this artifact owes (rule 4).")
    out = {}
    for k in NOT_COMPUTED_KEYS:
        if k not in blk:
            raise EarlyReadVerifyRefused(
                f"EARLY_READ_STATUS_MISSING: `{k}` is not among the named "
                f"statuses ({sorted(blk)}). A field that is neither computed "
                f"nor named has simply gone quiet.")
        v = blk[k]
        if _is_number(v) or (isinstance(v, dict) and any(
                _is_number(x) for x in v.values())):
            raise EarlyReadVerifyRefused(
                f"EARLY_READ_STATUS_CARRIES_A_NUMBER: `{k}` is {v!r}. These "
                f"five were NEVER COMPUTED for these days; a number here "
                f"would be an invention wearing a status's name.")
        if not isinstance(v, str) or len(v.strip()) < 20:
            raise EarlyReadVerifyRefused(
                f"EARLY_READ_STATUS_WITHOUT_A_REASON: `{k}` is {v!r}. A "
                f"status without a reason is a shrug.")
        out[k] = v
    return out


def census_arm_day(doc: dict) -> dict:
    """THE ANTI-ECHO CENSUS, INVERTED FOR THIS FAMILY.

    Everywhere else an economic name in an emission is a leak. Here the
    economics are what the USER ruled may be seen, so the question is the
    opposite: EXACTLY the six computed fields and the three counts, and
    nothing else economic.
    """
    day_run = doc.get("day_run") or {}
    arms = day_run.get("per_day_sealed_artifacts")
    if not isinstance(arms, list) or not arms:
        raise EarlyReadVerifyRefused(
            "EARLY_READ_NO_ARM_BLOCKS: `day_run.per_day_sealed_artifacts` is "
            "empty or absent. An empty read is a FAILURE, not a day with "
            "nothing in it.")
    per_arm, extras, missing = {}, [], []
    for blk in arms:
        arm = blk.get("arm")
        keys = set(blk)
        for k in ECON_REQUIRED + COUNTS_REQUIRED:
            if k not in keys:
                missing.append(f"{arm}.{k}")
        for k in sorted(keys & ECON_VOCAB - ALLOWED_ECON):
            extras.append(f"{arm}.{k}")
        nds = blk.get("null_draws_summary")
        n_draws = nds.get("n") if isinstance(nds, dict) else None
        per_arm[arm] = {
            "D_E0": blk.get("D_E0"), "Z": blk.get("Z"),
            "p_location": blk.get("p_location"),
            "null_mean": blk.get("null_mean"), "null_sd": blk.get("null_sd"),
            "n_draws": n_draws,
            "n_fills_arm": blk.get("n_fills_arm"),
            "n_fills_baseline": blk.get("n_fills_baseline"),
            "n_cancels_issued": blk.get("n_cancels_issued"),
            "status": blk.get("status"),
        }
        if n_draws is None:
            missing.append(f"{arm}.null_draws_summary.n")
    if missing:
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_ECONOMIC_FIELD_MISSING: {missing}. The USER ruled "
            f"these visible; a column that quietly disappears is a table with "
            f"a hole nobody sees.")
    if extras:
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_EXTRA_ECONOMIC_KEY: {extras}. The arm-day block "
            f"carries an economic name outside the set these runs computed. "
            f"A field that appears without having been computed is an "
            f"INVENTION, and this census is the inverted one: it asks what is "
            f"here that should not be.")
    return {"n_arms": len(per_arm), "per_arm": per_arm,
            "allowed": sorted(ALLOWED_ECON),
            "vocabulary_checked_against": sorted(ECON_VOCAB - ALLOWED_ECON),
            "the_census_is_INVERTED_for_this_family": (
                "elsewhere an economic name is a LEAK; here it is the point, "
                "and the census asks for EXACTLY the computed set")}


def verify(path, *, repo_root=None, data_root=None) -> dict:
    """One artifact, end to end. Every failure raises BY NAME."""
    doc = load_artifact(path, repo_root=repo_root)
    day = doc.get("day_run", {}).get("day") or doc.get("day")
    ruling = the_ruling_by_the_pair(doc, repo_root=repo_root)
    bar = the_bar_for_the_day(ruling, day)
    receipt = check_receipt_against_the_bar(doc, bar, data_root=data_root)
    labels = check_labels(doc, ruling)
    statuses = check_not_computed(doc)
    census = census_arm_day(doc)
    return {
        "protocol": PROTOCOL, "artifact": str(path),
        "artifact_sha256": _sha(Path(path)), "day": day,
        "IS_A_VERIFICATION": True,
        "verdict": "VERIFIED",
        "ruling": {"name": ruling["name"], "sha256": ruling["sha256"],
                   "version": ruling["version"],
                   "is_the_current_head": ruling["is_the_current_head"],
                   "the_current_head": ruling["the_current_head"],
                   "resolved_by": ruling["resolved_by"]},
        "bar": bar, "sealed_receipt_identity": receipt,
        "labels": labels, "not_computed_statuses": statuses,
        "census": census,
        "label_line": LABEL_LINE,
        "counts_provenance": {
            "day": day, "says": counts_provenance(day),
            "why_it_is_said": (
                "REV 90 S A2 and the ruling's own blindness_notes: 09-03's "
                "three counts were readable in the open while the later "
                "days were being run, and the other three days' counts are "
                "unsealed by this read. A table that showed four days' "
                "counts without saying which is which would invite a "
                "comparison across two different blindness states")},
        "this_reader_computed_nothing": (
            "every number below is DE's, read from the artifact and checked "
            "for presence and shape. This reader re-derives no economic "
            "quantity and opens no book"),
    }


def print_table(res: dict) -> str:
    """The coordinator's table. THE LABEL LINE RIDES ON EVERY PRINT."""
    lines = [f"EARLY READ -- day {res['day']} -- {LABEL_LINE}",
             f"  ruling {res['ruling']['name']} "
             f"{res['ruling']['sha256'][:16]}… "
             f"(head today: {res['ruling']['the_current_head'].get('name')})",
             f"  sealed receipt {res['sealed_receipt_identity']['the_bar_names']['name']} "
             f"{str(res['bar']['sha256'])[:16]}…",
             "  arm                     D_E0         Z   p(1-sided)"
             "   null_mean    null_sd   n_draws   fills_arm  fills_base"
             "   cancels"]
    for arm, v in sorted(res["census"]["per_arm"].items()):
        lines.append(
            f"  {arm:<20} {str(v['D_E0']):>10} {str(v['Z']):>9} "
            f"{str(v['p_location']):>12} {str(v['null_mean']):>11} "
            f"{str(v['null_sd']):>10} {str(v['n_draws']):>9} "
            f"{str(v['n_fills_arm']):>11} {str(v['n_fills_baseline']):>11} "
            f"{str(v['n_cancels_issued']):>9}")
    lines.append(f"  p is ONE-SIDED (p_location). {LABEL_LINE}.")
    lines.append(f"  the three COUNTS on this day: "
                 f"{counts_provenance(res['day'])}")
    lines.append("  NOT COMPUTED for these days, as named statuses: "
                 + ", ".join(NOT_COMPUTED_KEYS))
    return "\n".join(lines)


# --------------------------------------------------------------- the battery
def _fixture_artifact(d: Path, ruling_name: str, ruling_sha: str,
                      bar_row: dict, day="2026-09-03") -> dict:
    """A well-formed early-read artifact, built here and nowhere else."""
    return {
        "protocol": EARLY_PROTOCOL,
        "ruling": {"path": f"{DECL_DIR}/{ruling_name}", "sha256": ruling_sha},
        "the_ruling_verbatim": "show me 4 days results first",
        "is_a_validation": False, "G": 4,
        "interval": "NONE_BELOW_FIVE_DAYS",
        "verdict_class": "EXPLORATORY",
        "days_consumed": list(DAYS_CONSUMED),
        "economics_field_availability": {
            "not_computed_by_this_path": {
                k: f"{k} was never computed by this path, and the reason is "
                   f"recorded rather than a number being supplied"
                for k in NOT_COMPUTED_KEYS}},
        "preconditions": {"sealed_receipt": {
            "path": f"data/pm_5min/derived/{Path(bar_row['path']).name}",
            "sha256": bar_row["sha256"]}},
        "day_run": {"day": day, "per_day_sealed_artifacts": [
            {"arm": "CONDVALUE_X_SKEW", "day": day, "status": "OK",
             "D_E0": -1234.5, "Z": -0.87, "p_location": 0.19,
             "null_mean": -900.1, "null_sd": 380.4,
             "null_draws_summary": {"n": 500},
             "n_fills_arm": 30171, "n_fills_baseline": 46439,
             "n_cancels_issued": 5146},
            {"arm": "HAZARD_OVER_SKEWED_REF", "day": day, "status": "OK",
             "D_E0": 210.75, "Z": 0.41, "p_location": 0.66,
             "null_mean": 12.0, "null_sd": 480.9,
             "null_draws_summary": {"n": 500},
             "n_fills_arm": 44895, "n_fills_baseline": 46439,
             "n_cancels_issued": 700}]},
        "as_of": "2026-01-01T00:00:00Z",
    }


def selftest() -> tuple:                                      # noqa: C901
    checks, fails = [], 0

    def ck(label, cond, detail=""):
        nonlocal fails
        checks.append({"check": label, "pass": bool(cond), "detail": detail})
        if not cond:
            fails += 1
        print(("ok   " if cond else "FAIL ") + label)
        if detail:
            print("       " + detail)

    root = HERE.parents[1]
    decl = root / DECL_DIR
    #: The two ruling versions are read HERE, from the ledger, because the
    #: fixtures must cite a pair that really resolves -- a fixture citing an
    #: invented digest would test nothing about the pair rule.
    v16 = decl / f"{RULING_FAMILY}_v16.json"
    v17 = decl / f"{RULING_FAMILY}_v17.json"
    have16, have17 = v16.is_file(), v17.is_file()
    ck("THE TWO RULING VERSIONS ARE PRESENT AND ARE READ AS DOCUMENTS: v16 "
       "carries the ruling with a SIXTEEN-HEX prefix bar, v17 supersedes it "
       "with the FULL digests. A fixture citing a digest nothing has would "
       "test nothing about the pair",
       have16 and have17,
       f"v16 {have16} ({_sha(v16)[:16] if have16 else '-'}…), v17 {have17} "
       f"({_sha(v17)[:16] if have17 else '-'}…)")
    sha16, sha17 = _sha(v16), _sha(v17)
    b17 = json.loads(v17.read_text())["user_ruled_early_read"]
    row17 = next(r for r in b17["this_reads_bar"]["receipts"]
                 if r["day"] == "2026-09-03")

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        good = _fixture_artifact(t, v17.name, sha17, row17)
        gp = t / f"{EARLY_FAMILY}_20260903__20260101T000000Z.json"
        gp.write_text(json.dumps(good))
        res = verify(gp, repo_root=root)
        ck("A WELL-FORMED ARTIFACT VERIFIES AND PRINTS: the ruling resolves "
           "by the pair, the receipt matches the bar's FULL digest, the four "
           "labels hold, the five statuses carry reasons, and the inverted "
           "census finds exactly the computed set",
           res["verdict"] == "VERIFIED" and res["census"]["n_arms"] == 2
           and res["labels"]["verdict_class"] == "EXPLORATORY"
           and res["bar"]["digest_is_full"] is True,
           f"{res['verdict']}: ruling {res['ruling']['name']} "
           f"(head today {res['ruling']['the_current_head']['name']}), "
           f"{res['census']['n_arms']} arms")
        table = print_table(res)
        ck("AND THE LABEL LINE RIDES ON THE PRINT ITSELF: every table this "
           "reader emits carries EXPLORATORY, G 4, point estimates, NO "
           "INTERVAL and days consumed -- on the header AND under the rows, "
           "so a screenshot of the middle of the table still carries them",
           table.count(LABEL_LINE) >= 2 and "p is ONE-SIDED" in table
           and "NOT COMPUTED" in table,
           f"label line appears {table.count(LABEL_LINE)}x; "
           f"{len(table.splitlines())} lines")

        def refuses(mutate, where):
            bad = json.loads(json.dumps(good))
            mutate(bad)
            bp = t / f"{EARLY_FAMILY}_20260903__20260101T000001Z.json"
            bp.write_text(json.dumps(bad))
            try:
                verify(bp, repo_root=root)
                return "ADMITTED"
            except EarlyReadVerifyRefused as e:
                return str(e).split(":")[0]

        r_pair = refuses(lambda b: b["ruling"].update({"sha256": "e" * 64}),
                         "wrong pair")
        r_fam = refuses(lambda b: b["ruling"].update(
            {"path": f"{DECL_DIR}/be_race_read_declaration_v6.json"}), "family")
        ck("KNOWN-BAD -- THE WRONG PAIR IS REFUSED BY NAME, both ways: a "
           "digest the file does not have, and a path outside the params "
           "family. ***The ruling is the authority for opening four sealed "
           "days; an unverifiable citation is not authority***",
           r_pair == "EARLY_READ_RULING_NOT_THE_PAIR"
           and r_fam == "EARLY_READ_RULING_NOT_THE_PAIR",
           f"wrong digest -> {r_pair}; wrong family -> {r_fam}")

        #: v16's bar is the REAL prefix-only case, not a mutation.
        good16 = _fixture_artifact(t, v16.name, sha16, row17)
        p16 = t / f"{EARLY_FAMILY}_20260903__20260101T000002Z.json"
        p16.write_text(json.dumps(good16))
        try:
            verify(p16, repo_root=root)
            r16 = "ADMITTED"
        except EarlyReadVerifyRefused as e:
            r16 = str(e).split(":")[0]
        ck("KNOWN-BAD, AND IT IS THE REAL ONE: an artifact citing v16 -- "
           "whose bar carries only a SIXTEEN-HEX PREFIX -- is refused "
           "PREFIX_ONLY and says so. ***This reader never silently compares "
           "16 hex***: the strength of a check must not depend on which "
           "version an artifact happens to cite",
           r16 == "PREFIX_ONLY",
           f"an artifact citing v16 -> {r16}")

        r_lab = refuses(lambda b: b.pop("verdict_class"), "label missing")
        r_val = refuses(lambda b: b.update({"is_a_validation": True}), "val")
        r_day = refuses(lambda b: b.update(
            {"days_consumed": DAYS_CONSUMED[:3]}), "days")
        ck("KNOWN-BADS -- THE LABELS: a MISSING label and a DIFFERENT one "
           "refuse by different names, and `is_a_validation: true` is "
           "refused. ***A read that relabelled itself a validation would be "
           "the whole failure this batch exists to prevent***",
           r_lab == "EARLY_READ_LABEL_MISSING"
           and r_val == "EARLY_READ_LABEL_DIFFERS"
           and r_day == "EARLY_READ_LABEL_DIFFERS",
           f"missing verdict_class -> {r_lab}; is_a_validation true -> "
           f"{r_val}; three days consumed -> {r_day}")

        r_num = refuses(lambda b: b["economics_field_availability"][
            "not_computed_by_this_path"].update({"rho_adverse_over_spread":
                                                 0.42}), "number")
        r_gone = refuses(lambda b: b["economics_field_availability"][
            "not_computed_by_this_path"].pop("fills_leg"), "missing")
        ck("KNOWN-BADS -- THE FIVE UNCOMPUTED FIELDS: a status carrying a "
           "NUMBER is refused, and a status that has gone missing is refused. "
           "***A number there would be an invention wearing a status's "
           "name***",
           r_num == "EARLY_READ_STATUS_CARRIES_A_NUMBER"
           and r_gone == "EARLY_READ_STATUS_MISSING",
           f"rho = 0.42 -> {r_num}; fills_leg removed -> {r_gone}")

        r_extra = refuses(lambda b: b["day_run"]["per_day_sealed_artifacts"][0]
                          .update({"D_E_MINUS_R": -12.0}), "extra")
        r_miss = refuses(lambda b: b["day_run"]["per_day_sealed_artifacts"][0]
                         .pop("null_sd"), "missing econ")
        r_ndraw = refuses(lambda b: b["day_run"]["per_day_sealed_artifacts"][0]
                          .update({"null_draws_summary": {}}), "n")
        ck("KNOWN-BADS -- THE INVERTED CENSUS, BOTH DIRECTIONS: an EXTRA "
           "economic key (`D_E_MINUS_R`, a sealed NAME no arm-day block "
           "produces) is refused, and a MISSING one is refused, and a "
           "`null_draws_summary` with no `n` is refused. ***Here an economic "
           "name is not a leak -- it is the point -- so the census asks for "
           "exactly the computed set***",
           r_extra == "EARLY_READ_EXTRA_ECONOMIC_KEY"
           and r_miss == "EARLY_READ_ECONOMIC_FIELD_MISSING"
           and r_ndraw == "EARLY_READ_ECONOMIC_FIELD_MISSING",
           f"D_E_MINUS_R -> {r_extra}; null_sd removed -> {r_miss}; "
           f"empty draws summary -> {r_ndraw}")

        sealed = t / "p003_de_gate1_day_run_20260903_SEALED__20260906T140155Z.v2.json"
        sealed.write_text(json.dumps({"protocol": "X"}))
        try:
            verify(sealed, repo_root=root)
            r_sealed = "ADMITTED"
        except EarlyReadVerifyRefused as e:
            r_sealed = str(e).split(":")[0]
        absent = t / f"{EARLY_FAMILY}_20260909__20260101T000009Z.json"
        try:
            verify(absent, repo_root=root)
            r_abs = "ADMITTED"
        except EarlyReadVerifyRefused as e:
            r_abs = str(e).split(":")[0]
        ck("KNOWN-BAD -- A SEALED-FAMILY ARTIFACT IS REFUSED BY NAME AND IS "
           "NEVER PARSED FOR CONTENT: this reader has no business inside a "
           "sealed receipt, and one handed to it is a mistake to refuse "
           "rather than a file to read. An ABSENT artifact refuses too -- "
           "the artifact's EXISTENCE is what says the read happened",
           r_sealed == "SEALED_FAMILY_ARTIFACT_REFUSED"
           and r_abs == "EARLY_READ_ARTIFACT_ABSENT",
           f"a sealed receipt -> {r_sealed}; an absent artifact -> {r_abs}")

        r_recv = refuses(lambda b: b["preconditions"]["sealed_receipt"].update(
            {"sha256": "f" * 64}), "receipt")
        r_rname = refuses(lambda b: b["preconditions"][
            "sealed_receipt"].update({"path": "data/pm_5min/derived/"
                                      "p003_de_gate1_day_run_20260904_"
                                      "SEALED__20260906T171144Z.v3.json"}),
                          "receipt name")
        ck("KNOWN-BADS -- THE SEALED RECEIPT'S IDENTITY: a digest that is not "
           "the bar's is refused, and so is the RIGHT bar digest under "
           "ANOTHER DAY'S receipt name. A read of a day is a read of ONE "
           "receipt",
           r_recv == "EARLY_READ_RECEIPT_NOT_THE_BAR"
           and r_rname == "EARLY_READ_RECEIPT_NOT_THE_BAR",
           f"wrong digest -> {r_recv}; another day's receipt -> {r_rname}")

    #: REV 90 S A2 -- THE COUNTS' PROVENANCE IS SAID PER DAY.
    with tempfile.TemporaryDirectory() as td3:
        t3 = Path(td3)
        a03 = _fixture_artifact(t3, v17.name, sha17, row17, day="2026-09-03")
        p03 = t3 / f"{EARLY_FAMILY}_20260903__20260101T000010Z.json"
        p03.write_text(json.dumps(a03))
        r03 = verify(p03, repo_root=root)
        t03 = print_table(r03)
        row06 = next(r for r in b17["this_reads_bar"]["receipts"]
                     if r["day"] == "2026-09-06")
        a06 = _fixture_artifact(t3, v17.name, sha17, row06, day="2026-09-06")
        a06["day_run"]["day"] = "2026-09-06"
        p06 = t3 / f"{EARLY_FAMILY}_20260906__20260101T000011Z.json"
        p06.write_text(json.dumps(a06))
        r06 = verify(p06, repo_root=root)
        t06 = print_table(r06)
    ck("REV 90 S A2 -- ***THE THREE COUNTS CARRY THEIR PROVENANCE, PER DAY, "
       "IN THE PRINTED TABLE***: 09-03's were VISIBLE IN THE OPEN since "
       "2026-09-06T14:01Z under the EIGHT-name seal scope, and the other "
       "three days' are unsealed BY THIS READ under the ELEVEN-name scope. "
       "A table showing four days' counts without saying which is which "
       "would invite a comparison across two different blindness states",
       "VISIBLE IN THE OPEN" in t03 and "2026-09-06T14:01Z" in t03
       and "unsealed BY THIS READ" in t06
       and "VISIBLE IN THE OPEN" not in t06
       and r03["counts_provenance"]["says"]
       != r06["counts_provenance"]["says"],
       f"09-03 -> {r03['counts_provenance']['says'][:52]}…; 09-06 -> "
       f"{r06['counts_provenance']['says'][:52]}…")

    # ---- rule 20's clause (REV 84 S3.2 / REV 85 S3, R-726) -------------
    #: THE SHARED MODULE'S OWN FALSIFIER, AS ONE CELL OF THIS BATTERY. This
    #: module imports `declaration_chain` to REPORT the head beside the pair
    #: it resolves, so a regression in the one implementation is this
    #: battery's problem too. Spawned as a process, and driven RED against a
    #: copy with one falsifier disarmed.
    def _dc_falsify(prog):
        r = subprocess.run([sys.executable, str(prog), "--falsify"],
                           capture_output=True, text=True, timeout=300)
        ls = [x for x in (r.stdout or "").strip().splitlines() if x.strip()]
        return (r.returncode, ls[-1] if ls else "",
                [x for x in ls if x.startswith("FAIL")])

    _dc = HERE / "declaration_chain.py"
    rc, summ, bad = _dc_falsify(_dc)
    ck("REV 84 S3.2 -- ONE IMPLEMENTATION, N DETECTORS: this battery RUNS "
       "`declaration_chain.py --falsify` AS A SUBPROCESS, so a regression in "
       "the shared chain module fails every importer at once",
       rc == 0 and summ.endswith("0 failures") and not bad,
       f"rc {rc}: {summ!r} {bad or ''}")
    with tempfile.TemporaryDirectory() as td2:
        cp = Path(td2) / "declaration_chain.py"
        src = _dc.read_text()
        dis = src.replace("    if dst.exists():",
                          "    if False and dst.exists():")
        cp.write_text(dis)
        brc, bsum, bfail = _dc_falsify(cp)
    ck("KNOWN-BAD, DRIVEN: the SAME cell against a COPY with one falsifier "
       "disarmed (VERSION_PATH_EXISTS) FAILS -- so the green above is a "
       "measurement, not a cell that cannot fire",
       dis != src and brc != 0 and "1 failures" in bsum and bfail,
       f"disarmed copy -> rc {brc}: {bsum!r}; {(bfail or [''])[0][:70]}")

    print(f"\n{'SELFTEST OK' if not fails else 'SELFTEST FAILED'} -- "
          f"{len(checks)} checks, {fails} failure(s)")
    return checks, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify", metavar="ARTIFACT")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="print the coordinator's table (values)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        return 1 if selftest()[1] else 0
    if not a.verify:
        ap.error("--selftest, or --verify <artifact> [--print]")
    #: THREE OUTCOMES, THREE CODES. 2 is a REFUSAL -- the instrument declined
    #: to run on what it was given; 1 would be a check that RAN and flagged;
    #: 0 is verified. A caller that could not tell them apart would read "you
    #: handed me a sealed receipt" as "the read disagrees".
    try:
        res = verify(a.verify, data_root=a.data_root)
    except EarlyReadVerifyRefused as e:
        print(str(e))
        return 2
    if a.output:
        a.output.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    if a.do_print:
        print(print_table(res))
    else:
        print(f"{res['verdict']} -- day {res['day']}, ruling "
              f"{res['ruling']['name']} {res['ruling']['sha256'][:16]}…, "
              f"{res['census']['n_arms']} arms. {LABEL_LINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
