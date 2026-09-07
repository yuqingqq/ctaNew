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


def counts_provenance(day: str, scope: dict | None = None) -> str:
    """WHERE THIS DAY'S THREE COUNTS CAME FROM.

    READ AT THE SEALED RECEIPT when one is supplied (DA 127): the receipt's
    own `sealed_field_names` says which scope that run was sealed under,
    and whether the three counts are IN it decides whether they were
    visible before this read. The per-day constants below are kept only as
    the fallback for a caller with no receipt, and they are no longer what
    the table says when the receipt can answer -- ***the earlier days'
    scope is not assumed for a later day***.
    """
    if isinstance(scope, dict) and scope.get("read_at_the_receipt"):
        if scope.get("counts_are_sealed_names"):
            return (f"unsealed BY THIS READ -- {scope['receipt']} sealed "
                    f"them: all three are in that run's own "
                    f"`sealed_field_names` ({scope['n_sealed_names']} names) "
                    f"and none is present in its arm blocks")
        return (f"VISIBLE IN THE OPEN since {scope['receipt']} landed -- "
                f"that run's own `sealed_field_names` "
                f"({scope['n_sealed_names']} names) does NOT include them "
                f"and its arm blocks carry them")
    return COUNTS_PROVENANCE.get(str(day), COUNTS_PROVENANCE["_default"])


def counts_scope_at_the_receipt(bar_row: dict, *, data_root) -> dict:
    """The three counts' standing, read off THAT DAY'S sealed receipt."""
    name = Path(str(bar_row["path"])).name
    p = Path(data_root) / "pm_5min/derived" / name
    if not p.is_file():
        return {"read_at_the_receipt": False,
                "why": f"{name} is not under this data root"}
    rec = json.loads(p.read_bytes())
    blocks = rec.get("per_day_sealed_artifacts") or []
    if not blocks:
        return {"read_at_the_receipt": False,
                "why": f"{name} carries no arm blocks"}
    names = list(blocks[0].get("sealed_field_names") or [])
    in_scope = [c for c in COUNTS_REQUIRED if c in names]
    present = [c for c in COUNTS_REQUIRED if c in blocks[0]]
    return {"read_at_the_receipt": True, "receipt": name,
            "sealed_field_names": names, "n_sealed_names": len(names),
            "counts_are_sealed_names": len(in_scope) == len(COUNTS_REQUIRED),
            "counts_in_the_seal_scope": in_scope,
            "counts_present_in_the_receipts_arm_blocks": present,
            "why_it_is_read_here": (
                "the scope changed BETWEEN days -- 09-03 was sealed under "
                "eight names and the later days under eleven -- so a "
                "constant map would carry one day's scope onto another. "
                "The receipt is the fact")}


#: THE ONLY FIELDS THIS READER TAKES FROM A SEALED RECEIPT. The receipt is
#: opened for ONE purpose -- to learn which params the sealed run declared --
#: and the paths it may read are enumerated here so the scope is a constant
#: a reader can check, not a habit. NOTHING ECONOMIC is among them, and the
#: battery drives a receipt with a planted economic field to show none of it
#: reaches this reader's output.
SEALED_RECEIPT_READ_SCOPE = (
    "provenance.params.sha256",
    "provenance.params.path",
    "provenance.digests_at_load_and_at_emit.inputs.params",
    "provenance.status",
    "provenance.inputs.params",
)


#: THE RULING THIS READER MAY PRINT UNDER (R-764, coordinator, routine and
#: disclosed for overrule). DA 123's check refused two of the four ruled
#: days; the coordinator then DIFFED the declarations leaf by leaf and ruled
#: that what differs is the SEAL SCOPE, not the computation. The ruling does
#: NOT soften the codes -- `verify()` still refuses by name, and this mode
#: prints BESIDE the refusal, never instead of it. Any OTHER refusal refuses
#: under this mode too.
RULED_CODES = ("COMPUTATION_PARAMS_RECONSTRUCTED_NOT_STAMPED",
               "COMPUTATION_PARAMS_NOT_THE_SEALED_RUNS")
RULINGS = {
    "R-764": {
        "id": "R-764",
        "ruled_by": "the coordinator (routine, disclosed for overrule)",
        "what_was_measured": (
            "params v14 -> v15 diffed LEAF BY LEAF: 2 leaves changed, both "
            "the design pointer v21 -> v23. design v21 -> v23 leaf by leaf: "
            "42 leaves -- the R29/R30 seal-scope clauses, the R22 closure "
            "names, battery counts, output name, provenance stamps -- and "
            "the declaration's own text: 'changes_nothing_else: NO "
            "estimand, NO bar, NO pin'"),
        "measured_by": "the coordinator at R-764, not by this reader",
        "therefore": (
            "for 2026-09-03 and 2026-09-04 the early read's COMPUTATION is "
            "the sealed runs' computation; what differs is the seal scope "
            "the run was sealed under and, for 09-03, that the run stamped "
            "no pin at all"),
        "the_delta": ("design pointer v21->v23: seal scope and closure "
                      "naming only; no estimand, bar or pin -- measured by "
                      "the coordinator at R-764"),
        #: WHAT THE RULING ACTUALLY MEASURED, so its text cannot be read as
        #: covering a pair nobody diffed. R-764 diffed params v14 -> v15.
        #: A day whose sealed run stamped v14 against a read that loaded
        #: v19 is a DIFFERENT span, and this reader says so on the line
        #: rather than letting the ruling's sentence stretch over it.
        "measured_span": {"from": "v14", "to": "v15"},
        "codes_it_rides_beside": list(RULED_CODES),
        "what_it_does_NOT_do": (
            "it does not soften a refusal. `verify()` refuses by name "
            "whatever this mode prints, the code is named in the printed "
            "table, and every OTHER refusal still refuses here"),
    },
}


class EarlyReadVerifyRefused(RuntimeError):
    """A named refusal. Every message begins with its own reason code."""


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# --------------------------------------------------- the family's HEAD
#: THE FIELD SHAPE, AGREED WITH DE 138 THROUGH THE REGISTER (REV 104A S7 #2).
#: A new artifact for a day carries `supersedes: {path, sha256}` naming the
#: artifact it replaces. The digest is the FULL 64 lowercase hex -- never a
#: 16-hex prefix, which is R-754's v17 lesson: a prefix bar let a check
#: compare sixteen characters and call it a pair.
SUPERSEDES_FIELD = "supersedes"
SUPERSEDES_PAIR_KEYS = ("path", "sha256")


def _early_read_files(day: str, derived: Path) -> list:
    """The day's artifacts on disk, by the family's own naming rule."""
    key = str(day).replace("-", "")
    return sorted(derived.glob(f"{EARLY_FAMILY}_{key}__*.json"))


def resolve_early_read_head(day: str, *, data_root=None,
                            derived=None) -> dict:
    """THE ONE ARTIFACT NOTHING SUPERSEDES -- or a REFUSAL BY NAME.

    RESOLVED BY THE PAIR, never by filename or by stamp order. `supersedes`
    names {path, sha256} and BOTH halves must land on a file that is
    present: a digest that does not match the bytes is
    SUPERSESSION_PAIR_MISMATCH, because a link that cannot verify is
    half-written and picking the newer stamp would be this reader inventing
    the chain (the same discipline `declaration_chain.resolve_head` applies
    to the declaration families).

    THREE OUTCOMES, EACH NAMED. One artifact, or a verified chain, resolves
    to a head. TWO UNCHAINED artifacts for one day are AMBIGUOUS -- "the
    09-03 early read" would name nothing, and this reader will not choose
    by stamp. NONE is ABSENT.
    """
    der = Path(derived) if derived else (
        Path(data_root) / "pm_5min/derived" if data_root
        else HERE.parents[1] / "data/pm_5min/derived")
    files = _early_read_files(day, der)
    if not files:
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_HEAD_ABSENT: no artifact of {EARLY_FAMILY} for "
            f"{day} under {der}. The artifact's EXISTENCE is what says the "
            f"read has happened.")
    by_name = {f.name: {"path": f, "sha256": _sha(f)} for f in files}
    superseded, links = {}, []
    for f in files:
        try:
            doc = json.loads(f.read_bytes())
        except json.JSONDecodeError as e:
            raise EarlyReadVerifyRefused(
                f"EARLY_READ_ARTIFACT_UNREADABLE: {f.name} is not readable "
                f"JSON ({e.msg} at line {e.lineno}).")
        sup = doc.get(SUPERSEDES_FIELD)
        if sup is None:
            continue
        if not isinstance(sup, dict) or any(
                k not in sup for k in SUPERSEDES_PAIR_KEYS):
            raise EarlyReadVerifyRefused(
                f"SUPERSESSION_PAIR_MISMATCH: {f.name}'s `{SUPERSEDES_FIELD}` "
                f"is {sup!r}, which is not a {{{', '.join(SUPERSEDES_PAIR_KEYS)}}} "
                f"pair. A link is a PAIR; a path alone verifies nothing.")
        want_name = Path(str(sup["path"])).name
        want_sha = str(sup["sha256"])
        target = by_name.get(want_name)
        if target is None:
            raise EarlyReadVerifyRefused(
                f"SUPERSESSION_PAIR_MISMATCH: {f.name} supersedes "
                f"{want_name}, which is not present under {der}. A link to a "
                f"file nobody has is not a link.")
        if not DIGEST64.match(want_sha):
            raise EarlyReadVerifyRefused(
                f"SUPERSESSION_PAIR_MISMATCH: {f.name} names {want_name} at "
                f"{want_sha!r} ({len(want_sha)} chars), which is not 64 "
                f"lowercase hex. ***A SIXTEEN-HEX PREFIX IS NOT THE PAIR*** "
                f"-- R-754's v17 lesson, in this family.")
        if want_sha != target["sha256"]:
            raise EarlyReadVerifyRefused(
                f"SUPERSESSION_PAIR_MISMATCH: {f.name} names {want_name} at "
                f"{want_sha[:16]}… and that file digests "
                f"{target['sha256'][:16]}…. The bytes it claims to supersede "
                f"are not the bytes on disk.")
        superseded[want_name] = f.name
        links.append({"newer": f.name, "supersedes": want_name,
                      "sha256": want_sha})
    heads = [f for f in files if f.name not in superseded]
    if len(heads) > 1:
        raise EarlyReadVerifyRefused(
            f"EARLY_READ_HEAD_AMBIGUOUS: {day} has {len(heads)} artifacts "
            f"that nothing supersedes ({[h.name for h in heads]}). "
            f"***'the {day} early read' would name nothing***, and this "
            f"reader will not pick by stamp: a later file is not a "
            f"successor unless it SAYS so by the pair.")
    head = heads[0]
    return {"day": str(day), "head": head.name, "path": str(head),
            "sha256": by_name[head.name]["sha256"],
            "n_artifacts": len(files),
            "artifacts": [f.name for f in files],
            "superseded": sorted(superseded),
            "links": links,
            "resolved_by": ("the `supersedes` PAIR ({path, sha256}), both "
                            "halves verified against the file on disk"),
            "the_sole_artifact_is_the_head": len(files) == 1,
            "why_not_by_stamp": (
                "a filename orders by clock and says nothing about "
                "succession. Two artifacts written a minute apart, neither "
                "naming the other, are two answers to one question")}


def head_standing(path, *, data_root=None, derived=None) -> dict:
    """Is THIS artifact the day's head? A superseded one is readable as
    PROVENANCE and is labelled -- never quoted as the day's read."""
    p = Path(path)
    day = None
    try:
        doc = json.loads(p.read_bytes())
        day = (doc.get("day_run") or {}).get("day") or doc.get("day")
    except (OSError, json.JSONDecodeError):
        pass
    if day is None:
        return {"resolved": False,
                "why": "the artifact names no day; standing not resolved"}
    try:
        h = resolve_early_read_head(day, data_root=data_root,
                                    derived=derived or p.parent)
    except EarlyReadVerifyRefused as e:
        return {"resolved": False, "refusal": str(e).split(":")[0],
                "detail": str(e)[:200]}
    is_head = (p.name == h["head"])
    return {"resolved": True, "is_the_head": is_head,
            "label": "HEAD" if is_head else "SUPERSEDED",
            "the_head_is": h["head"], "n_artifacts": h["n_artifacts"],
            "what_a_superseded_artifact_is_for": (
                None if is_head else
                "PROVENANCE. It is readable, and it is never quoted as the "
                "day's read -- the head is")}


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


def _params_declared_by_the_sealed_run(receipt: dict) -> dict:
    """The params digest the SEALED RECEIPT declares -- and nothing else.

    Two places carry it and both are read: `provenance.params` (the pair the
    run recorded) and `provenance.digests_at_load_and_at_emit.inputs.params`
    (the SAME file digested at load and again at emit, which is how a file
    changing under a run is caught -- REV 75 S1.1). A receipt whose own two
    readings disagree is refused by its own name rather than averaged or
    preferred.
    """
    prov = (receipt.get("provenance") or {})
    pair = (prov.get("params") or {})
    at = (((prov.get("digests_at_load_and_at_emit") or {}).get("inputs")
           or {}).get("params") or {})
    #: THE THIRD STATE, FOUND AT THE ARTIFACTS (DA 123). 2026-09-03's
    #: receipt stamped NO params: its provenance block says of itself
    #: "RECONSTRUCTED, not stamped ... no digest here was taken at load or
    #: at emit, and none is offered as one", and the digest comes from git
    #: at the carrying commit. That is not a declaration BY THE ACT, and
    #: comparing against it would be comparing against a reconstruction.
    recon = (prov.get("inputs") or {}).get("params") or {}
    return {"declared_sha256": pair.get("sha256"),
            "declared_path": pair.get("path"),
            "at_load": at.get("sha256_at_load"),
            "at_emit": at.get("sha256_at_emit"),
            "the_receipts_own_two_readings_agree": at.get("agrees"),
            "is_RECONSTRUCTED": (str(prov.get("status")) == "RECONSTRUCTED"
                                 or str(recon.get("status"))
                                 == "RECONSTRUCTED"),
            "reconstructed_path": recon.get("path"),
            "reconstructed_sha256": recon.get(
                "sha256_AT_THE_CARRYING_COMMIT_RECONSTRUCTED"),
            "read_scope": list(SEALED_RECEIPT_READ_SCOPE)}


def check_computation_params(doc: dict, bar_row: dict, *, data_root) -> dict:
    """THE EARLY READ'S WHOLE CLAIM, MADE A PREDICATE (REV 91 S C2).

    The artifact says *"the COMPUTATION is the sealed runs' -- v15. Only the
    seal bar comes from the ruling."* That is a RECORDED FIELD, and a
    recorded field is a claim until something compares it: this compares the
    params digest the RUN ACTUALLY LOADED against the params digest THE
    DAY'S SEALED RECEIPT DECLARES. If they differ, the early read is a
    DIFFERENT computation wearing the sealed run's name, and no table drawn
    from it is comparable to the sealed days.

    ABSENCE IS NEVER A PASS. A receipt that is not there, or that declares
    no params, REFUSES by its own name -- this is the one check that cannot
    be skipped without the whole claim going unchecked.
    """
    cp = (doc.get("computation_params") or {})
    got = cp.get("sha256")
    if not (isinstance(got, str) and DIGEST64.match(got)):
        raise EarlyReadVerifyRefused(
            f"COMPUTATION_PARAMS_NOT_DECLARED: the artifact's "
            f"`computation_params.sha256` is {got!r}, not a 64-hex digest. "
            f"The early read's claim is that it computed what the sealed "
            f"runs computed; without the digest of what it loaded there is "
            f"nothing to compare that claim against.")
    name = Path(str(bar_row["path"])).name
    receipt_p = Path(data_root) / "pm_5min/derived" / name
    if not receipt_p.is_file():
        raise EarlyReadVerifyRefused(
            f"COMPUTATION_PARAMS_NOT_CHECKABLE_SEALED_RECEIPT_ABSENT: "
            f"{name} is not under {data_root}. The comparison this check "
            f"exists for needs the sealed run's own declaration, and a "
            f"check that cannot run FAILS rather than passing quietly "
            f"(R-649).")
    declared = _params_declared_by_the_sealed_run(
        json.loads(receipt_p.read_bytes()))
    if declared["is_RECONSTRUCTED"]:
        raise EarlyReadVerifyRefused(
            f"COMPUTATION_PARAMS_RECONSTRUCTED_NOT_STAMPED: {name}'s "
            f"provenance says of ITSELF that it is RECONSTRUCTED -- the run "
            f"stamped no params, and the digest "
            f"({str(declared['reconstructed_sha256'])[:16]}… for "
            f"{Path(str(declared['reconstructed_path'])).name}) comes from "
            f"git at the carrying commit, not from the act. ***A "
            f"reconstruction is not a declaration by the run***, and this "
            f"reader will not certify 'the computation is the sealed run's' "
            f"against one. THREE STATES, NOT TWO: this is neither a match "
            f"nor a mismatch, and it is said rather than resolved either "
            f"way.")
    if not (isinstance(declared["declared_sha256"], str)
            and DIGEST64.match(declared["declared_sha256"])):
        raise EarlyReadVerifyRefused(
            f"COMPUTATION_PARAMS_NOT_CHECKABLE_RECEIPT_DECLARES_NO_PARAMS: "
            f"{name} carries no `provenance.params.sha256`. The sealed run "
            f"named no params, so what it computed cannot be identified.")
    if declared["the_receipts_own_two_readings_agree"] is False:
        raise EarlyReadVerifyRefused(
            f"SEALED_RUNS_PARAMS_MOVED_UNDER_THE_RUN: {name} digested its "
            f"params at load ({str(declared['at_load'])[:16]}…) and again at "
            f"emit ({str(declared['at_emit'])[:16]}…) and they DISAGREE. "
            f"Which computation that run performed has no single answer, so "
            f"nothing may be compared to it.")
    if declared["declared_sha256"] != got:
        raise EarlyReadVerifyRefused(
            f"COMPUTATION_PARAMS_NOT_THE_SEALED_RUNS: the early read loaded "
            f"params {got[:16]}… and {name} declares "
            f"{declared['declared_sha256'][:16]}…. ***The early read's whole "
            f"claim is that the computation is the sealed runs'***; a "
            f"different params file makes it a different computation wearing "
            f"that claim, and the four days would not be comparable to the "
            f"sealed ones or to each other.")
    return {"the_artifact_loaded": {"path": cp.get("path"), "sha256": got},
            "the_sealed_receipt_declares": {
                "receipt": name, "path": declared["declared_path"],
                "sha256": declared["declared_sha256"]},
            "at_load_and_at_emit": {
                "at_load": declared["at_load"], "at_emit": declared["at_emit"],
                "agree": declared["the_receipts_own_two_readings_agree"]},
            "they_are_the_same_params": True,
            "the_receipt_was_opened_for_this_and_only_this": {
                "read_scope": list(SEALED_RECEIPT_READ_SCOPE),
                "why_opening_it_is_allowed_here": (
                    "the seal withholds the ECONOMICS. The provenance block "
                    "says which declaration the run loaded, and reading it "
                    "is what DA 117's pre-read already does. No economic "
                    "field is read, and the battery drives a receipt with a "
                    "planted economic field to show none reaches this "
                    "reader's output")},
            "why_this_is_the_load_bearing_check": (
                "every other check here is about labels, identity and shape. "
                "THIS one is the early read's own claim -- that the "
                "computation is the sealed runs' -- turned from a recorded "
                "field into a compared one")}


#: THE VALUATION, RE-IMPLEMENTED HERE FROM DE'S DECLARED RULE (R-235:
#: read as a document, never imported). `de_phase4_diag_runner`'s
#: `fill_value_cents` is the maker P&L at level-to-markout with NO fee
#: term: sgn * (mid_cents_at_markout - px_cents) * size, sgn = +1 on the
#: BUY side. The ledger's own header names the recompute entry point in
#: DE's module; this reader does not call it -- two implementations that
#: agree are evidence, one checking itself is not.
LEDGER_BUY_SIDE = "BUY_UP"
LEDGER_VALUATION = ("sgn * (mid_cents_at_markout - px_cents) * size, "
                    "sgn = +1 on BUY_UP; math.fsum over the rows")


#: THE SETTLEMENT VALUATION, RE-IMPLEMENTED FROM DE'S DECLARED RULE
#: (R-801; `settlement_legs_by_slug` / `settle_value_cents` read as
#: DOCUMENTS, never imported). Per slug:
#:     trades leg   = sum(-sgn * px * size)      a BUY pays out, a SELL takes in
#:     residual leg = net_shares * settle
#:     total        = trades + residual
#: and the identity `total == sum sgn*(settle - px)*size` holds by algebra,
#: so this reader ASSERTS it rather than trusting either form -- two
#: expressions of one quantity, computed separately and compared.
SETTLE_RECONCILE_TOL = 1e-9
WINNER_STATUS_REQUIRED = "VERIFIED_AGREE"


def recompute_settlement_from_the_ledger(rows_by_kind: dict,
                                         fills_by_arm: dict) -> dict:
    """Per arm: the settlement D and the two legs, from the FILL rows and
    the winner each SETTLEMENT_SLUG row carries -- compared against the
    SETTLEMENT_SCALARS row the file also carries."""
    import math                                               # noqa: PLC0415
    scal = {r["arm"]: r for r in rows_by_kind.get("SETTLEMENT_SCALARS", [])}
    per_slug = {}
    for r in rows_by_kind.get("SETTLEMENT_SLUG", []):
        per_slug.setdefault(r.get("arm"), {}).setdefault(
            r.get("book"), {})[r.get("slug")] = r
    if not scal and not per_slug:
        return {"status": "SETTLEMENT_ROWS_ABSENT",
                "why": ("this ledger carries neither SETTLEMENT_SCALARS nor "
                        "SETTLEMENT_SLUG rows. The day was NOT valued under "
                        "R-801's endpoint -- ***absent, not zero***, and a "
                        "reader that reported 0 would be inventing a "
                        "settlement nobody computed"),
                "per_arm": {}}
    out, flags = {}, []
    for arm, sc in sorted(scal.items()):
        books = per_slug.get(arm, {})
        #: THE WINNER PER SLUG, AND ITS STATUS. A slug whose winner is not
        #: VERIFIED_AGREE is refused: the settlement value of every fill in
        #: it rests on a winner the venue and Chainlink do not agree on.
        winners, bad_status = {}, []
        for book, per in books.items():
            for slug, row in per.items():
                st = row.get("status") or row.get("winner_status")
                if st is not None and st != WINNER_STATUS_REQUIRED:
                    bad_status.append(f"{arm}.{book}.{slug}={st}")
                if "up_won" in row:
                    winners[slug] = bool(row["up_won"])
        if bad_status:
            raise EarlyReadVerifyRefused(
                f"SETTLEMENT_WINNER_NOT_VERIFIED: {bad_status}. A slug whose "
                f"winner status is not {WINNER_STATUS_REQUIRED} carries a "
                f"settlement value that rests on a winner the venue and the "
                f"Chainlink convention do not agree on, and every fill in "
                f"that slug is valued by it.")
        legs = {}
        for book in ("ARM", "BASELINE"):
            fills = [f for f in fills_by_arm.get(arm, {}).get(book, [])]
            per = books.get(book, {})
            tr = res = 0.0
            per_fill = 0.0
            n_missing_winner = 0
            for f in fills:
                px, sz = f.get("px_cents"), float(f.get("size") or 0.0)
                slug = f.get("slug")
                if px is None or not sz:
                    continue
                if slug not in winners:
                    n_missing_winner += 1
                    continue
                sgn = 1.0 if f.get("side") == LEDGER_BUY_SIDE else -1.0
                settle = (per.get(slug) or {}).get("settle_cents")
                if settle is None:
                    settle = 100.0 if winners[slug] else 0.0
                tr += -sgn * float(px) * sz
                res += sgn * sz * float(settle)
                per_fill += sgn * (float(settle) - float(px)) * sz
            total = tr + res
            if abs(total - per_fill) > SETTLE_RECONCILE_TOL:
                raise EarlyReadVerifyRefused(
                    f"SETTLEMENT_LEGS_DO_NOT_RECONCILE: {arm}.{book} legs sum "
                    f"to {total!r} and the per-fill form gives {per_fill!r} "
                    f"(difference {total - per_fill!r}). The legs are a "
                    f"DECOMPOSITION of the ruled quantity; if they disagree "
                    f"with it, one of them is a different quantity.")
            legs[book] = {"trades_leg_cents": tr, "residual_leg_cents": res,
                          "total_cents": total,
                          "n_fills_valued": len(fills) - n_missing_winner,
                          "n_fills_without_a_winner_row": n_missing_winner}
        d_settle = legs["ARM"]["total_cents"] - legs["BASELINE"]["total_cents"]
        cmp_ = {
            "D_E_settle": {"recomputed": d_settle,
                           "in_the_row": sc.get("D_E_settle")},
            "arm_total_cents": {"recomputed": legs["ARM"]["total_cents"],
                                "in_the_row": sc.get("arm_total_cents")},
            "baseline_total_cents": {
                "recomputed": legs["BASELINE"]["total_cents"],
                "in_the_row": sc.get("baseline_total_cents")},
        }
        for k, v in cmp_.items():
            a, b = v["recomputed"], v["in_the_row"]
            v["agrees"] = (b is not None
                           and abs(float(a) - float(b))
                           <= SETTLE_RECONCILE_TOL)
            if not v["agrees"]:
                flags.append(f"{arm}.{k}")
        out[arm] = {"legs": legs, "compared": cmp_,
                    "ruling": sc.get("ruling"), "unit": sc.get("unit"),
                    "n_slugs_ARM": len(books.get("ARM", {})),
                    "n_slugs_BASELINE": len(books.get("BASELINE", {})),
                    "winner_source": sc.get("winner_source")}
    if flags:
        raise EarlyReadVerifyRefused(
            f"SETTLEMENT_SCALARS_DISAGREE: {flags} -- this reader recomputed "
            f"the settlement D and the two totals from the FILL rows and the "
            f"verified winners, and they differ from the SETTLEMENT_SCALARS "
            f"row by more than {SETTLE_RECONCILE_TOL}. Two implementations "
            f"of one quantity disagreeing is the finding, not a tolerance to "
            f"widen.")
    return {"status": "SETTLEMENT_ROWS_PRESENT", "per_arm": out,
            "valuation": ("trades = sum(-sgn*px*size); residual = "
                          "net_shares*settle; total = trades + residual; "
                          "asserted equal to sum sgn*(settle-px)*size"),
            "tolerance": SETTLE_RECONCILE_TOL,
            "winner_status_required": WINNER_STATUS_REQUIRED,
            "this_is_a_second_implementation": (
                "R-801's rule read as a DOCUMENT from "
                "`settlement_legs_by_slug`; DE's module is not imported")}


def recompute_from_the_ledger(path) -> dict:
    """PER ARM, FROM THE LEDGER'S ROWS ALONE. No artifact field is read."""
    import gzip                                               # noqa: PLC0415
    import math                                               # noqa: PLC0415
    import statistics                                         # noqa: PLC0415
    arms, draws, scal = {}, {}, {}
    raw_fills, settle_rows = {}, {}
    n_rows, kinds, fields = 0, {}, {}
    with gzip.open(str(path), "rt") as f:
        for line in f:
            r = json.loads(line)
            n_rows += 1
            k = r.get("row")
            kinds[k] = kinds.get(k, 0) + 1
            fields.setdefault(k, set()).update(r.keys())
            if k == "ARM_SCALARS":
                scal[r["arm"]] = r
            elif k == "NULL_DRAW":
                draws.setdefault(r["arm"], []).append(r["value"])
            elif k == "FILL":
                sgn = 1.0 if r.get("side") == LEDGER_BUY_SIDE else -1.0
                v = sgn * (r["mid_cents_at_markout"] - r["px_cents"]) \
                    * r["size"]
                arms.setdefault(r["arm"], {}).setdefault(
                    r.get("book"), []).append(v)
                raw_fills.setdefault(r["arm"], {}).setdefault(
                    r.get("book"), []).append(r)
            elif k in ("SETTLEMENT_SCALARS", "SETTLEMENT_SLUG"):
                #: DE 136/137's two NEW kinds, under an UNCHANGED
                #: schema_version 2. They are COLLECTED here rather than
                #: skipped -- a reader that skips them silently reports the
                #: 5-s number as the day's answer (measured, DA 131).
                settle_rows.setdefault(k, []).append(r)
    out = {}
    for arm, sc in scal.items():
        books = arms.get(arm, {})
        arm_v = math.fsum(books.get("ARM", []))
        base_v = math.fsum(books.get("BASELINE", []))
        d = draws.get(arm, [])
        mean = statistics.fmean(d) if d else None
        sd = statistics.pstdev(d) if d else None
        obs = arm_v - base_v
        ge = sum(1 for x in d if x >= obs)
        out[arm] = {
            "arm_value_cents": arm_v, "baseline_value_cents": base_v,
            "D_E0": obs,
            "Z": ((obs - mean) / sd) if (sd not in (None, 0)) else None,
            "p_location": ((1 + ge) / (1 + len(d))) if d else None,
            "null_mean": mean, "null_sd": sd, "n_draws": len(d),
            "n_fills_arm": len(books.get("ARM", [])),
            "n_fills_baseline": len(books.get("BASELINE", [])),
            "the_ledgers_own_scalars": {
                "arm_value_cents": sc.get("arm_value_cents"),
                "baseline_value_cents": sc.get("baseline_value_cents"),
                "observed_D_E0": sc.get("observed_D_E0"),
                "n_fills_arm": sc.get("n_fills_arm"),
                "n_fills_baseline": sc.get("n_fills_baseline"),
                "n_cancels_issued": sc.get("n_cancels_issued")},
        }
    #: R-795, MEASURED HERE RATHER THAN QUOTED: the day value IS the fills
    #: leg by construction, and `inventory_leg` is not a field of this
    #: ledger. The field names of every row kind are collected so that
    #: statement is a measurement over the file, not a repetition of what
    #: an artifact says about it.
    allf = sorted({f for v in fields.values() for f in v})
    return {"n_rows": n_rows, "row_kinds": kinds, "per_arm": out,
            "settlement": recompute_settlement_from_the_ledger(settle_rows,
                                                               raw_fills),
            "valuation": LEDGER_VALUATION,
            "field_names_by_row_kind": {k: sorted(v)
                                        for k, v in fields.items()},
            "has_an_inventory_leg_field": "inventory_leg" in allf,
            "inventory_inputs_present": sorted(
                f for f in allf if f.startswith("inventory_"))}


def verify_decision_ledger(doc: dict, census: dict, *, data_root) -> dict:
    """THE LEDGER THE ARTIFACT NAMES: identity, then an INDEPENDENT recompute.

    Identity first -- a ledger whose bytes are not the ones the artifact
    names is REFUSED, because everything below would then be a recompute of
    a different file. Only then are the rows read, and what comes out is
    compared to what the artifact printed.
    """
    ledger = check_decision_ledger(doc)
    if ledger["status"] != "LEDGER_PRESENT":
        return dict(ledger, recompute="NOT_ATTEMPTED_NO_LEDGER")
    blk = (doc.get("day_run") or {}).get("decision_ledger") or {}
    p = Path(str(blk.get("path")))
    if not p.is_absolute():
        p = Path(data_root) / "pm_5min/derived" / p.name
    if not p.is_file():
        raise EarlyReadVerifyRefused(
            f"LEDGER_NAMED_BUT_ABSENT_ON_DISK: the artifact names {p.name} "
            f"and it is not at {p}. A named artifact that is not there is a "
            f"pin to nothing.")
    got = _sha(p)
    if got != blk.get("sha256"):
        raise EarlyReadVerifyRefused(
            f"LEDGER_DIGEST_MISMATCH: {p.name} digests {got[:16]}… and the "
            f"artifact names {str(blk.get('sha256'))[:16]}…. Everything "
            f"recomputed from it would be a recompute of a different file.")
    rec = recompute_from_the_ledger(p)
    if rec["n_rows"] != blk.get("n_rows"):
        raise EarlyReadVerifyRefused(
            f"LEDGER_ROW_COUNT_DIFFERS: the file holds {rec['n_rows']} rows "
            f"and the artifact says {blk.get('n_rows')}.")
    #: THE COMPARISON: what the ledger says against what the artifact
    #: printed, field by field, EXACTLY -- both are finite sums of the same
    #: float64 values selected by the same rule, so a tolerance would only
    #: hide a selection difference.
    per_arm, mismatches = {}, []
    for arm, mine in rec["per_arm"].items():
        theirs = (census.get("per_arm") or {}).get(arm) or {}
        row = {}
        for k in ("D_E0", "Z", "p_location", "null_mean", "null_sd",
                  "n_draws", "n_fills_arm", "n_fills_baseline"):
            a, b = mine.get(k), theirs.get(k)
            row[k] = {"from_the_ledger": a, "in_the_artifact": b,
                      "equal": a == b}
            if a != b:
                mismatches.append(f"{arm}.{k}")
        row["baseline_value_cents"] = mine["baseline_value_cents"]
        row["arm_value_cents"] = mine["arm_value_cents"]
        row["the_ledgers_own_scalars_agree"] = (
            mine["arm_value_cents"]
            == mine["the_ledgers_own_scalars"]["arm_value_cents"]
            and mine["baseline_value_cents"]
            == mine["the_ledgers_own_scalars"]["baseline_value_cents"]
            and mine["D_E0"]
            == mine["the_ledgers_own_scalars"]["observed_D_E0"])
        per_arm[arm] = row
    return dict(
        ledger,
        recompute={
            "path": str(p), "sha256": got, "n_rows": rec["n_rows"],
            "row_kinds": rec["row_kinds"], "valuation": rec["valuation"],
            #: R-795's two measured facts, carried to the printer rather
            #: than recomputed there.
            "settlement": rec["settlement"],
            "has_an_inventory_leg_field": rec["has_an_inventory_leg_field"],
            "inventory_inputs_present": rec["inventory_inputs_present"],
            "field_names_by_row_kind": rec["field_names_by_row_kind"],
            "per_arm": per_arm,
            "n_mismatches": len(mismatches), "mismatched": mismatches,
            "verdict": "AGREES" if not mismatches else "FLAGGED",
            "what_was_recomputed": (
                "D_E0 as the difference of two fsum reductions over the "
                "ARM and BASELINE fill rows; Z from the 500 NULL_DRAW "
                "values as (observed - mean)/pstdev; p_location as "
                "(1 + #{null >= observed}) / (1 + K), one-sided; and the "
                "fill counts from the rows themselves"),
            "this_reader_did_not_call_DEs_module": (
                "the ledger's header names `de_decision_ledger.recompute`; "
                "this is a SECOND implementation from the declared "
                "valuation rule (R-235). Two implementations that agree are "
                "evidence"),
            "the_0_cancel_baseline": {
                "value_cents": {a: v["baseline_value_cents"]
                                for a, v in per_arm.items()},
                "what_it_is": (
                    "the value of the day's NO-CANCEL reference path, "
                    "summed over its own fill rows under the declared "
                    "valuation. It is the term D_E0 is an excess OVER, and "
                    "no arm-day block carries it"),
                "the_fills_leg_IS_the_total": (
                    "the ledger's own `baseline_value_cents` equals this sum "
                    "over the BASELINE fill rows exactly, so under the "
                    "declared valuation the day's value is the FILLS LEG "
                    "and carries no separate inventory term"),
                "the_inventory_leg_is_NOT_computed_here": (
                    "the five inventory fields (inventory_before, "
                    "inventory_after, inventory_unit, inventory_mark_cents, "
                    "inventory_mark_source) are on every fill row, so the "
                    "INPUTS are there -- what is missing is a DECLARED "
                    "AGGREGATION RULE: which residual position, marked at "
                    "which price, per slug or per day. No such rule is in "
                    "the ledger's header, the artifact or the design, and "
                    "this reader will not choose one. That is the missing "
                    "thing, and it is a rule rather than a field")},
        })


def check_decision_ledger(doc: dict) -> dict:
    """THE LEDGER: a REPORTED status, never a refusal and never a guess.

    `day_run.decision_ledger` is where a reader would find the 0-cancel
    baseline's own value for the day. On the early-read path no ledger was
    written (a DE fix is in flight), so the block is NULL -- and the honest
    handling is neither to refuse the day nor to reconstruct the baseline
    from the arm blocks: it is to SAY SO BY NAME beside the table, quoting
    the block as it stands. ***A number nobody computed must not appear
    because a table has a column for it*** (rule 4: exclusions are
    statuses).

    Three states, told apart: the key is missing entirely, the key is
    present and null, or a ledger is there.
    """
    dr = doc.get("day_run") or {}
    present = "decision_ledger" in dr
    val = dr.get("decision_ledger")
    if not present:
        status = "LEDGER_KEY_ABSENT"
        says = ("the artifact carries no `day_run.decision_ledger` key at "
                "all -- not even a null. That is a different fact from a "
                "null block and is named separately")
    elif val is None:
        status = "LEDGER_ABSENT"
        says = ("`day_run.decision_ledger` is present and NULL: no ledger "
                "was written on the early-read path for this day")
    else:
        status = "LEDGER_PRESENT"
        says = "a ledger block is present"
    return {
        "status": status, "says": says,
        "the_block_as_it_stands": val,
        "what_cannot_be_derived": (
            "the 0-cancel BASELINE's own value for this day. The arm blocks "
            "carry D(E0) -- a DIFFERENCE against that baseline -- and a "
            "difference does not contain either of its terms"
            if status != "LEDGER_PRESENT" else None),
        "never_approximated": (
            "this reader does not reconstruct the baseline from the arm "
            "blocks, the fill counts or anything else. An approximation "
            "printed in a table is read as a measurement"
            if status != "LEDGER_PRESENT" else None),
        "and_the_table_still_prints": True,
        "why_not_a_refusal": (
            "the day's arm-day results are what the USER ruled visible and "
            "they are all here; the ledger's absence removes ONE derivable "
            "quantity and is reported as removing it"),
    }


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
    #: TWO KEY NAMES, BOTH READ (DA 127). The block was
    #: `not_computed_by_this_path` through 09-03 and 09-04; at R-765/BE 96
    #: DE renamed it `not_computed_by_this_path_the_ARM_DAY_BLOCK` and
    #: added `where_the_five_live_now`, because four of the five ARE now
    #: computed -- in the DECISION LEDGER, not in the arm-day block. A
    #: reader of history reads both names: the earlier artifacts carry the
    #: earlier one and are not wrong for it.
    avail = doc.get("economics_field_availability") or {}
    blk = (avail.get("not_computed_by_this_path")
           or avail.get("not_computed_by_this_path_the_ARM_DAY_BLOCK"))
    where = avail.get("where_the_five_live_now")
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
    if isinstance(where, dict):
        #: NOT a refusal and NOT this reader's claim: DE's own statement of
        #: where each of the five now lives, carried so the table does not
        #: keep saying "not computed" about four quantities their own
        #: artifact says are computed elsewhere. What THIS reader verified
        #: independently is in the ledger block, not here.
        out["_where_the_five_live_now_DEs_words"] = dict(where)
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
    per_arm, extras, missing, where = {}, [], [], set()
    for blk in arms:
        arm = blk.get("arm")
        #: WHERE THE SIX LIVE, MEASURED PER ARM RATHER THAN ASSUMED. DE's
        #: UNSEALED emission nests them under `economic`; the SEALED
        #: receipts carry the same names flat at the arm's top level (they
        #: are what sealing REMOVES). ***This reader was built against the
        #: sealed shape and its fixture reproduced that assumption***, so
        #: the first real artifact refused ECONOMIC_FIELD_MISSING on an
        #: artifact that had every field. Both shapes are read now, the one
        #: in use is REPORTED, and the counts stay where DE puts them -- at
        #: the arm's top level, outside the economic block.
        econ = blk.get("economic") if isinstance(
            blk.get("economic"), dict) else {}
        where.add("economic_block" if econ else "flat_on_the_arm")
        keys = set(blk) | set(econ)

        def _g(k, _blk=blk, _e=econ):
            return _e.get(k, _blk.get(k))

        for k in ECON_REQUIRED + COUNTS_REQUIRED:
            if k not in keys:
                missing.append(f"{arm}.{k}")
        for k in sorted(keys & ECON_VOCAB - ALLOWED_ECON):
            extras.append(f"{arm}.{k}")
        nds = _g("null_draws_summary")
        n_draws = nds.get("n") if isinstance(nds, dict) else None
        per_arm[arm] = {
            "D_E0": _g("D_E0"), "Z": _g("Z"),
            "p_location": _g("p_location"),
            "null_mean": _g("null_mean"), "null_sd": _g("null_sd"),
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
            "where_the_six_were_found": sorted(where),
            "why_both_shapes_are_read": (
                "DE's UNSEALED emission nests the six under `economic`; the "
                "SEALED receipts carry the same names flat at the arm's top "
                "level. This reader was built against the sealed shape and "
                "its own fixture reproduced that assumption, so the FIRST "
                "REAL ARTIFACT refused ECONOMIC_FIELD_MISSING on an artifact "
                "that had every field -- the refusal was honest and the "
                "locator was wrong"),
            "allowed": sorted(ALLOWED_ECON),
            "vocabulary_checked_against": sorted(ECON_VOCAB - ALLOWED_ECON),
            "the_census_is_INVERTED_for_this_family": (
                "elsewhere an economic name is a LEAK; here it is the point, "
                "and the census asks for EXACTLY the computed set")}


def verify(path, *, repo_root=None, data_root=None) -> dict:
    """One artifact, end to end. EVERY failure raises BY NAME.

    This entry never softens anything: it is the one the GOs use, and the
    ruling mode below is a SECOND entry that calls the same checks.
    """
    res, err = _verify_parts(path, repo_root=repo_root, data_root=data_root)
    if err is not None:
        raise err
    return res


def _verify_parts(path, *, repo_root=None, data_root=None) -> tuple:
    """(result, the params refusal or None). EVERY OTHER refusal RAISES.

    Only the computation-params refusal is returned rather than raised, and
    only so `verify_under_ruling` can decide whether a RULING covers it.
    Nothing else is catchable here: a wrong ruling pair, a relabelled read,
    a status carrying a number or an extra economic key raises from inside
    this function exactly as before.
    """
    root = Path(repo_root) if repo_root else HERE.parents[1]
    #: THE DATA ROOT IS RESOLVED AND SAID. The params check needs the day's
    #: sealed receipt, and a reader must know WHICH ledger answered.
    if data_root is None and (root / "data").exists():
        data_root = root / "data"
    doc = load_artifact(path, repo_root=repo_root)
    day = doc.get("day_run", {}).get("day") or doc.get("day")
    #: THE STANDING IS STATED ON EVERY READ (REV 104A S7 #2). A superseded
    #: artifact is readable as PROVENANCE and says so; it is never quoted
    #: as the day's read.
    standing = head_standing(path, data_root=data_root)
    ruling = the_ruling_by_the_pair(doc, repo_root=repo_root)
    bar = the_bar_for_the_day(ruling, day)
    #: ORDER: the receipt's IDENTITY is established before a single field of
    #: it is read. Reading provenance out of a receipt whose digest has not
    #: been checked would be trusting bytes nobody pinned.
    receipt = check_receipt_against_the_bar(doc, bar, data_root=data_root)
    counts_scope = counts_scope_at_the_receipt(bar, data_root=data_root)
    #: CAUGHT, NOT SOFTENED: returned to the caller, which raises it unless a
    #: RULING covers the code. The other four checks below still raise.
    params, params_err = None, None
    try:
        params = check_computation_params(doc, bar, data_root=data_root)
    except EarlyReadVerifyRefused as e:
        params_err = e
        params = {"REFUSED": str(e).split(":")[0],
                  "measured": _params_declared_by_the_sealed_run(
                      json.loads((Path(data_root) / "pm_5min/derived"
                                  / Path(str(bar["path"])).name).read_bytes()))
                  if (Path(data_root) / "pm_5min/derived"
                      / Path(str(bar["path"])).name).is_file() else None,
                  "the_artifact_loaded": (doc.get("computation_params")
                                          or {})}
    labels = check_labels(doc, ruling)
    statuses = check_not_computed(doc)
    census = census_arm_day(doc)
    ledger = verify_decision_ledger(doc, census, data_root=data_root)
    return {
        "protocol": PROTOCOL, "artifact": str(path),
        "artifact_sha256": _sha(Path(path)), "day": day,
        "IS_A_VERIFICATION": params_err is None,
        "verdict": "VERIFIED" if params_err is None else "REFUSED",
        "ruling": {"name": ruling["name"], "sha256": ruling["sha256"],
                   "version": ruling["version"],
                   "is_the_current_head": ruling["is_the_current_head"],
                   "the_current_head": ruling["the_current_head"],
                   "resolved_by": ruling["resolved_by"]},
        "head_standing": standing,
        "bar": bar, "sealed_receipt_identity": receipt,
        "computation_params": params,
        "data_root_used": str(data_root),
        "labels": labels, "not_computed_statuses": statuses,
        "census": census, "decision_ledger": ledger,
        "label_line": LABEL_LINE,
        "counts_provenance": {
            "day": day, "says": counts_provenance(day, counts_scope),
            "scope_read_at_the_receipt": counts_scope,
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
    }, params_err


def _version_of(path_or_name) -> str:
    """`…_params_v14.json` -> `v14`. Computed from the name, never typed."""
    m = re.search(r"_v(\d+)\.json$", str(Path(str(path_or_name)).name))
    return f"v{m.group(1)}" if m else "an unversioned file"


def params_label(res: dict) -> str:
    """THE PER-DAY LABEL, from what was MEASURED on that day's receipt.

    Three readings, three sentences -- and the counts clause rides along
    where it applies, because the two facts a reader must hold about
    2026-09-03 are that it was computed under a later params version AND
    that its three counts were never sealed.
    """
    cp = res.get("computation_params") or {}
    read_v = _version_of((cp.get("the_artifact_loaded") or {}).get("path"))
    if not cp.get("REFUSED"):
        base = "params match"
    else:
        m = cp.get("measured") or {}
        if cp["REFUSED"] == "COMPUTATION_PARAMS_RECONSTRUCTED_NOT_STAMPED":
            sealed_v = _version_of(m.get("reconstructed_path"))
            base = (f"computed under {read_v}; sealed run RECONSTRUCTED "
                    f"{sealed_v}")
        else:
            sealed_v = _version_of(m.get("declared_path"))
            base = f"computed under {read_v}; sealed run stamped {sealed_v}"
    if "VISIBLE IN THE OPEN" in counts_provenance(res.get("day")):
        base += ("; counts visible since 2026-09-06T14:01Z "
                 "(eight-name scope)")
    return base


def verify_under_ruling(path, *, ruling_id="R-764", repo_root=None,
                        data_root=None) -> dict:
    """THE RULING RIDES BESIDE THE REFUSAL -- it never replaces it.

    For an artifact refused ONLY by a code the ruling covers, the table is
    printed WITH a MATERIALITY line that names the code, both params
    versions, the measured delta and the ruling's id. ***Every other
    refusal still refuses here***: a wrong ruling pair, a relabelled read,
    a status carrying a number and an extra economic key all raise out of
    `_verify_parts` before this function sees anything.
    """
    ruling = RULINGS.get(str(ruling_id))
    if ruling is None:
        raise EarlyReadVerifyRefused(
            f"UNKNOWN_RULING: {ruling_id!r} is not a ruling this reader "
            f"carries ({sorted(RULINGS)}). A mode that printed under a "
            f"ruling nobody landed would be this reader ruling.")
    res, err = _verify_parts(path, repo_root=repo_root, data_root=data_root)
    if err is None:
        return dict(res, under_ruling={
            "applies": False,
            "why": "nothing was refused; the ruling had nothing to ride "
                   "beside"})
    code = str(err).split(":")[0]
    if code not in ruling["codes_it_rides_beside"]:
        #: NOT COVERED -- and the refusal is re-raised unchanged.
        raise err
    cp = res.get("computation_params") or {}
    m = cp.get("measured") or {}
    sealed_path = (m.get("declared_path") if code
                   == "COMPUTATION_PARAMS_NOT_THE_SEALED_RUNS"
                   else m.get("reconstructed_path"))
    sealed_sha = (m.get("declared_sha256") if code
                  == "COMPUTATION_PARAMS_NOT_THE_SEALED_RUNS"
                  else m.get("reconstructed_sha256"))
    _span = ruling.get("measured_span") or {}
    _obs = {"from": _version_of(sealed_path),
            "to": _version_of((cp.get("the_artifact_loaded") or {}).get(
                "path"))}
    _within = (_obs["from"] == _span.get("from")
               and _obs["to"] == _span.get("to"))
    return dict(res, under_ruling={
        "applies": True, "ruling_id": ruling["id"],
        "refusal_code": code,
        "the_span_the_ruling_MEASURED": _span,
        "the_span_IN_FRONT_OF_IT": _obs,
        "observed_span_is_within_what_was_measured": _within,
        "if_it_is_not": (
            None if _within else
            f"R-764 diffed {_span.get('from')} -> {_span.get('to')} leaf by "
            f"leaf. THIS day's pair is {_obs['from']} -> {_obs['to']}, which "
            f"that diff does not cover. The ruling's finding is carried "
            f"here as the coordinator's, and the UNMEASURED part of the "
            f"span is named rather than absorbed into it -- reported, not "
            f"ruled"),
        "the_refusal_stands": (
            "this artifact IS refused by `verify()`, which is the entry the "
            "GOs use. This mode prints beside that refusal and names it"),
        "the_sealed_runs_params": {
            "version": _version_of(sealed_path), "path": sealed_path,
            "sha256": sealed_sha,
            "state": ("RECONSTRUCTED" if code
                      == "COMPUTATION_PARAMS_RECONSTRUCTED_NOT_STAMPED"
                      else "STAMPED")},
        "the_reads_params": {
            "version": _version_of(
                (cp.get("the_artifact_loaded") or {}).get("path")),
            "path": (cp.get("the_artifact_loaded") or {}).get("path"),
            "sha256": (cp.get("the_artifact_loaded") or {}).get("sha256")},
        "the_measured_delta": ruling["the_delta"],
        "what_was_measured": ruling["what_was_measured"],
        "measured_by": ruling["measured_by"],
        "therefore": ruling["therefore"],
        "ruled_by": ruling["ruled_by"],
        "overrulable": True,
    })


def print_table(res: dict) -> str:
    """The coordinator's table. THE LABEL LINE RIDES ON EVERY PRINT."""
    _hs = res.get("head_standing") or {}
    _lbl = (f" -- ***{_hs.get('label')}***"
            + (f" (the head is {_hs.get('the_head_is')}; this artifact is "
               f"read as PROVENANCE and is not the day's read)"
               if _hs.get("label") == "SUPERSEDED" else "")
            if _hs.get("resolved") else
            f" -- HEAD STANDING NOT RESOLVED ({_hs.get('refusal') or _hs.get('why')})")
    lines = [f"EARLY READ -- day {res['day']}{_lbl} -- {LABEL_LINE}",
             f"  ruling {res['ruling']['name']} "
             f"{res['ruling']['sha256'][:16]}… "
             f"(head today: {res['ruling']['the_current_head'].get('name')})",
             f"  sealed receipt {res['sealed_receipt_identity']['the_bar_names']['name']} "
             f"{str(res['bar']['sha256'])[:16]}…",
             ]
    #: THE COLUMNS ARE SIZED TO THE VALUES. ***A quoted number is never
    #: truncated to fit a column***: this table is the one place these
    #: numbers may be read, and a width chosen in advance would either
    #: clip a float or silently round it.
    _cols = [("arm", "arm"), ("D_E0", "D_E0"), ("Z", "Z"),
             ("p_location", "p(1-sided)"), ("null_mean", "null_mean"),
             ("null_sd", "null_sd"), ("n_draws", "n_draws"),
             ("n_fills_arm", "fills_arm"),
             ("n_fills_baseline", "fills_base"),
             ("n_cancels_issued", "cancels")]
    _rows = []
    for arm, v in sorted(res["census"]["per_arm"].items()):
        _rows.append([arm] + [repr(v[k]) if isinstance(v[k], float)
                              else str(v[k]) for k, _ in _cols[1:]])
    _w = [max(len(h), *(len(r[i]) for r in _rows)) if _rows else len(h)
          for i, (_, h) in enumerate(_cols)]
    lines.append("  " + "  ".join(
        h.ljust(_w[i]) if i == 0 else h.rjust(_w[i])
        for i, (_, h) in enumerate(_cols)))
    for r in _rows:
        lines.append("  " + "  ".join(
            c.ljust(_w[i]) if i == 0 else c.rjust(_w[i])
            for i, c in enumerate(r)))
    lines.append(f"  p is ONE-SIDED (p_location). {LABEL_LINE}.")
    lines.append(f"  the three COUNTS on this day: "
                 f"{(res.get('counts_provenance') or {}).get('says')}")
    _dl = res.get("decision_ledger") or {}
    _rc = _dl.get("recompute")
    if isinstance(_rc, dict) and _rc.get("per_arm"):
        _b = _rc["the_0_cancel_baseline"]["value_cents"]
        _one = sorted(set(_b.values()))
        lines.append(
            f"  0-cancel baseline, from the ledger, EXPLORATORY, FILLS LEG "
            f"ONLY: {_one[0]!r} cents"
            + ("" if len(_one) == 1 else f" (per arm: {_b})")
            + f" -- R-795: the day value IS the fills leg by construction. "
              f"`inventory_leg` is NOT a field of this ledger (measured: "
              f"{_rc['has_an_inventory_leg_field']}); its INPUTS are "
              f"({', '.join(_rc['inventory_inputs_present'])}), and an "
              f"inventory LEG would need an aggregation rule nobody has "
              f"declared")
        for _arm, _v in sorted(_rc["per_arm"].items()):
            lines.append(
                f"    {_arm} (fills leg only): arm "
                f"{_v['arm_value_cents']!r} - baseline "
                f"{_v['baseline_value_cents']!r} = "
                f"{_v['D_E0']['from_the_ledger']!r}  (the artifact prints "
                f"{_v['D_E0']['in_the_artifact']!r}; equal: "
                f"{_v['D_E0']['equal']})")
        _se = _rc.get("settlement") or {}
        if _se.get("status") == "SETTLEMENT_ROWS_PRESENT":
            lines.append(
                f"  SETTLEMENT (R-801, ***PRIMARY***), recomputed from the "
                f"FILL rows and the verified winners, cents:")
            for _arm, _v in sorted(_se["per_arm"].items()):
                _c = _v["compared"]
                _a, _b = _v["legs"]["ARM"], _v["legs"]["BASELINE"]
                lines.append(
                    f"    {_arm}: arm trades {_a['trades_leg_cents']!r} + "
                    f"residual {_a['residual_leg_cents']!r} = "
                    f"{_a['total_cents']!r}; 0-cancel baseline trades "
                    f"{_b['trades_leg_cents']!r} + residual "
                    f"{_b['residual_leg_cents']!r} = {_b['total_cents']!r}; "
                    f"D_settle {_c['D_E_settle']['recomputed']!r} (the row "
                    f"says {_c['D_E_settle']['in_the_row']!r}; agrees: "
                    f"{_c['D_E_settle']['agrees']})")
            lines.append(
                f"    every slug's winner is {_se['winner_status_required']}; "
                f"the legs are asserted equal to the per-fill form within "
                f"{_se['tolerance']}")
        else:
            lines.append(
                f"  SETTLEMENT (R-801, PRIMARY): {_se.get('status')} -- "
                f"{_se.get('why')}")
        lines.append(
            f"  the 5-s markout D_E0 above is the ***DIAGNOSTIC*** under "
            f"R-801, not the result.")
        lines.append(
            f"  the ledger recompute: {_rc['verdict']} -- "
            f"{_rc['n_rows']} rows, {_rc['n_mismatches']} mismatches across "
            f"D_E0, Z, p, null mean/sd, n draws and both fill counts; "
            f"Z and p re-derived from the {_rc['per_arm'][sorted(_rc['per_arm'])[0]]['n_draws']['from_the_ledger']} "
            f"NULL_DRAW rows, not read from the artifact")
    if _dl.get("status") and _dl["status"] != "LEDGER_PRESENT":
        lines.append(
            f"  {_dl['status']}: {_dl['says']} -- "
            f"`day_run.decision_ledger` = {json.dumps(_dl['the_block_as_it_stands'])}. "
            f"{_dl['what_cannot_be_derived']}. It is NOT approximated here.")
    _st = res.get("not_computed_statuses") or {}
    _wh = _st.get("_where_the_five_live_now_DEs_words")
    if isinstance(_wh, dict):
        _still = sorted(k for k, v in _wh.items()
                        if "NOT COMPUTED" in str(v).upper())
        _elsewhere = sorted(k for k in _wh if k not in _still)
        _lrc = ((res.get("decision_ledger") or {}).get("recompute")
                or {})
        _inv = _lrc.get("has_an_inventory_leg_field")
        lines.append(
            f"  NOT in the arm-day block. THE ARTIFACT SAYS "
            f"{', '.join(_elsewhere)} are COMPUTED in the decision ledger "
            f"and {', '.join(_still)} nowhere -- ***DE's words, carried, "
            f"not this reader's finding***. WHAT THIS READER MEASURED IN "
            f"THE LEDGER: the FILLS LEG is the day value itself (R-795, by "
            f"construction), and `inventory_leg` is NOT a field of the file"
            + (f" (measured over every row kind: "
               f"has_an_inventory_leg_field={_inv})"
               if _inv is not None else "")
            + ". This reader did not re-derive p_two_sided or rho.")
    else:
        lines.append("  NOT COMPUTED for these days, as named statuses: "
                     + ", ".join(NOT_COMPUTED_KEYS))
    ur = res.get("under_ruling") or {}
    if ur.get("applies"):
        lines.append(
            f"  MATERIALITY -- REFUSED {ur['refusal_code']}, PRINTED UNDER "
            f"{ur['ruling_id']}: the sealed run's params "
            f"{ur['the_sealed_runs_params']['version']} "
            f"{str(ur['the_sealed_runs_params']['sha256'])[:16]}… "
            f"({ur['the_sealed_runs_params']['state']}) against this read's "
            f"{ur['the_reads_params']['version']} "
            f"{str(ur['the_reads_params']['sha256'])[:16]}… -- "
            f"{ur['the_measured_delta']}. The refusal STANDS; "
            f"{ur['ruling_id']} is {ur['ruled_by']} and is overrulable.")
        if not ur.get("observed_span_is_within_what_was_measured"):
            lines.append(
                f"  AND THE SPAN IS NOT THE ONE THE RULING MEASURED: "
                f"{ur['ruling_id']} diffed "
                f"{ur['the_span_the_ruling_MEASURED'].get('from')} -> "
                f"{ur['the_span_the_ruling_MEASURED'].get('to')}; this day's "
                f"pair is {ur['the_span_IN_FRONT_OF_IT']['from']} -> "
                f"{ur['the_span_IN_FRONT_OF_IT']['to']}. The remainder is "
                f"NOT covered by that diff -- reported, not ruled.")
        lines.append(f"  this day: {params_label(res)}")
        return "\n".join(lines)
    lines.append(f"  this day: {params_label(res)}")
    lines.append(f"  the computation is the SEALED RUN'S: params "
                 f"{res['computation_params']['the_artifact_loaded']['sha256'][:16]}"
                 f"… equals the digest "
                 f"{res['computation_params']['the_sealed_receipt_declares']['receipt']}"
                 f" declares -- CHECKED, not recorded")
    return "\n".join(lines)


def four_day_table(paths, *, repo_root=None, data_root=None) -> str:
    """THE FOUR DAYS IN ONE BLOCK. Every number computed; no conclusion.

    Each day is read through the SAME reader and the SAME ruling mode, so a
    day that refuses for a reason the ruling does not cover appears as a
    refusal here rather than as a gap. The per-arm sign count and the
    smallest two-sided sign-test p reachable at this G are arithmetic over
    what was read -- and the floor is printed whether or not any arm
    reaches it, because ***a p that cannot go below 0.0625 is a fact about
    the DESIGN, not about the days***.
    """
    days, refused = [], []
    for p in paths:
        try:
            r = verify_under_ruling(p, repo_root=repo_root,
                                    data_root=data_root)
        except EarlyReadVerifyRefused as e:
            refused.append({"path": str(p), "code": str(e).split(":")[0]})
            continue
        cp = r.get("computation_params") or {}
        m = cp.get("measured") or {}
        dl = r.get("decision_ledger") or {}
        rc = dl.get("recompute")
        sealed_v = _version_of(m.get("declared_path")
                               or m.get("reconstructed_path"))
        days.append({
            "day": r["day"],
            "read_v": _version_of((cp.get("the_artifact_loaded")
                                   or {}).get("path")),
            "sealed_v": sealed_v,
            "sealed_state": ("RECONSTRUCTED" if cp.get("REFUSED")
                             == "COMPUTATION_PARAMS_RECONSTRUCTED_NOT_STAMPED"
                             else "STAMPED" if cp.get("REFUSED")
                             else "MATCHES"),
            "ledger": dl.get("status"),
            "absolutes": ("PRESENT (from the ledger)"
                          if isinstance(rc, dict) and rc.get("per_arm")
                          else "NOT AVAILABLE (no ledger to sum)"),
            "per_arm": r["census"]["per_arm"],
            "baseline": ({a: v["baseline_value_cents"]
                          for a, v in rc["per_arm"].items()}
                         if isinstance(rc, dict) and rc.get("per_arm")
                         else None),
        })
    out = [f"FOUR-DAY EARLY READ -- {LABEL_LINE} -- G {len(days)} of the "
           f"ruled four; every value FILLS LEG ONLY (R-795)"]
    hdr = ["day", "arm", "D_E0", "Z", "p(1-sided)", "fills_arm",
           "fills_base", "cancels"]
    rows = []
    for d in days:
        for arm, v in sorted(d["per_arm"].items()):
            rows.append([d["day"], arm, repr(v["D_E0"]), repr(v["Z"]),
                         repr(v["p_location"]), str(v["n_fills_arm"]),
                         str(v["n_fills_baseline"]),
                         str(v["n_cancels_issued"])])
    w = [max(len(hdr[i]), *(len(r[i]) for r in rows)) for i in range(len(hdr))]
    out.append("  " + "  ".join(
        h.ljust(w[i]) if i < 2 else h.rjust(w[i]) for i, h in enumerate(hdr)))
    for r in rows:
        out.append("  " + "  ".join(
            c.ljust(w[i]) if i < 2 else c.rjust(w[i])
            for i, c in enumerate(r)))
    out.append("")
    for d in days:
        b = d["baseline"]
        one = sorted(set(b.values())) if b else None
        out.append(
            f"  {d['day']}: computed under {d['read_v']}; sealed run "
            f"{d['sealed_state']} {d['sealed_v']}; absolutes "
            f"{d['absolutes']}; ledger {d['ledger']}"
            + (f"; 0-cancel baseline {one[0]!r} cents (fills leg only)"
               if one and len(one) == 1 else
               f"; 0-cancel baseline {b}" if b else ""))
    for r in refused:
        out.append(f"  {Path(r['path']).name}: REFUSED {r['code']} -- not a "
                   f"gap, a refusal")
    out.append("")
    arms = sorted({a for d in days for a in d["per_arm"]})
    G = len(days)
    for arm in arms:
        vals = [(d["day"], d["per_arm"][arm]["D_E0"]) for d in days
                if arm in d["per_arm"]]
        pos = [dy for dy, v in vals if v > 0]
        neg = [dy for dy, v in vals if v < 0]
        zero = [dy for dy, v in vals if v == 0]
        out.append(
            f"  {arm}: days ABOVE the 0-cancel baseline (D_E0 > 0): "
            f"{len(pos)} of {len(vals)} {pos or ''}; below: {len(neg)} "
            f"{neg or ''}" + (f"; exactly zero: {len(zero)}" if zero else ""))
    out.append(
        f"  the smallest TWO-SIDED sign-test p reachable at G = {G}: "
        f"2^-{G} = {2.0 ** -G!r} -- ***the floor of the design, reachable "
        f"only by a unanimous sign, and it clears no 0.05 bar on its own***. "
        f"No interval is computed here: below five complete UTC days this "
        f"programme reports a point estimate and says so.")
    out.append(f"  {LABEL_LINE}. The four days are CONSUMED (R-754). Every "
               f"number above is computed from the artifacts and their "
               f"ledgers; no conclusion is drawn here.")
    return "\n".join(out)


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
        #: v15's REAL digest: the fixture must claim what the sealed runs
        #: actually loaded, or the well-formed case would pass a check the
        #: real artifacts have to pass.
        "computation_params": {
            "path": "live/pm_research/declarations/"
                    "de_multiday_gate1_params_v15.json",
            "sha256": ("92858fc7f9493f8e8fcc721d0390843bce86bbab1"
                       "7d633a0bf3210d633fb6037"),
            "why": "the COMPUTATION is the sealed runs' -- v15"},
        "economics_field_availability": {
            "not_computed_by_this_path": {
                k: f"{k} was never computed by this path, and the reason is "
                   f"recorded rather than a number being supplied"
                for k in NOT_COMPUTED_KEYS}},
        "preconditions": {"sealed_receipt": {
            "path": f"data/pm_5min/derived/{Path(bar_row['path']).name}",
            "sha256": bar_row["sha256"]}},
        #: DE'S REAL SHAPE (measured at
        #: p003_de_early_read_day_20260903__20260907T085436Z.json,
        #: 5c8a58f501d3b61b…): the six live under `economic` and the three
        #: counts at the arm's top level. ***The first fixture put them
        #: flat, which was this reader's own assumption from the SEALED
        #: receipts, and the fixture reproduced the assumption instead of
        #: testing it*** -- so the first real artifact refused
        #: ECONOMIC_FIELD_MISSING on an artifact that had every field.
        "day_run": {"day": day, "per_day_sealed_artifacts": [
            {"arm": "CONDVALUE_X_SKEW", "day": day, "status": "OK",
             "sealed": False, "sealed_field_names": [],
             "economic": {"D_E0": -1234.5, "Z": -0.87, "p_location": 0.19,
                          "null_mean": -900.1, "null_sd": 380.4,
                          "null_draws_summary": {"n": 500}},
             "n_fills_arm": 30171, "n_fills_baseline": 46439,
             "n_cancels_issued": 5146},
            {"arm": "HAZARD_OVER_SKEWED_REF", "day": day, "status": "OK",
             "sealed": False, "sealed_field_names": [],
             "economic": {"D_E0": 210.75, "Z": 0.41, "p_location": 0.66,
                          "null_mean": 12.0, "null_sd": 480.9,
                          "null_draws_summary": {"n": 500}},
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
    _rows17 = {r["day"]: r for r in b17["this_reads_bar"]["receipts"]}
    #: THE WELL-FORMED FIXTURE USES A DAY WHOSE SEALED RECEIPT STAMPED ITS
    #: PARAMS. 2026-09-03's did not (its provenance says of itself that it
    #: is RECONSTRUCTED), and 09-04's stamped v14 -- both are DRIVEN BELOW
    #: as the two real findings this check turned up on its first run.
    row17 = _rows17["2026-09-06"]

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        good = _fixture_artifact(t, v17.name, sha17, row17,
                                 day="2026-09-06")
        good["day_run"]["day"] = "2026-09-06"
        gp = t / f"{EARLY_FAMILY}_20260906__20260101T000000Z.json"
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

        r_extra = refuses(lambda b: b["day_run"]["per_day_sealed_artifacts"]
                          [0]["economic"].update({"D_E_MINUS_R": -12.0}),
                          "extra")
        r_miss = refuses(lambda b: b["day_run"]["per_day_sealed_artifacts"]
                         [0]["economic"].pop("null_sd"), "missing econ")
        r_ndraw = refuses(lambda b: b["day_run"]["per_day_sealed_artifacts"]
                          [0]["economic"].update({"null_draws_summary": {}}),
                          "n")
        #: AND THE SAME KNOWN-BAD ON THE ARM'S TOP LEVEL, because an
        #: invented economic name must be caught wherever it is put -- the
        #: census reads BOTH shapes now, so both must refuse.
        r_extra_flat = refuses(
            lambda b: b["day_run"]["per_day_sealed_artifacts"][0].update(
                {"sd_over_abs_mean": 0.5}), "extra flat")
        ck("KNOWN-BADS -- THE INVERTED CENSUS, BOTH DIRECTIONS: an EXTRA "
           "economic key (`D_E_MINUS_R`, a sealed NAME no arm-day block "
           "produces) is refused, and a MISSING one is refused, and a "
           "`null_draws_summary` with no `n` is refused. ***Here an economic "
           "name is not a leak -- it is the point -- so the census asks for "
           "exactly the computed set***",
           r_extra == "EARLY_READ_EXTRA_ECONOMIC_KEY"
           and r_miss == "EARLY_READ_ECONOMIC_FIELD_MISSING"
           and r_ndraw == "EARLY_READ_ECONOMIC_FIELD_MISSING"
           and r_extra_flat == "EARLY_READ_EXTRA_ECONOMIC_KEY",
           f"D_E_MINUS_R -> {r_extra}; null_sd removed -> {r_miss}; "
           f"empty draws summary -> {r_ndraw}; an extra name on the arm's "
           f"top level -> {r_extra_flat}")

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

    # -- REV 91 S C2: THE EARLY READ'S OWN CLAIM, AS A PREDICATE --------
    _cp = res["computation_params"]
    ck("REV 91 S C2 -- ***THE CLAIM 'THE COMPUTATION IS THE SEALED RUNS'' IS "
       "NOW COMPARED, NOT RECORDED***: the params the artifact says the run "
       "LOADED are digested against the params THE DAY'S SEALED RECEIPT "
       "DECLARES, and the receipt's own load-and-emit pair must agree with "
       "itself first",
       _cp["they_are_the_same_params"]
       and _cp["the_artifact_loaded"]["sha256"]
       == _cp["the_sealed_receipt_declares"]["sha256"]
       and _cp["at_load_and_at_emit"]["agree"] is True,
       f"artifact loaded {_cp['the_artifact_loaded']['sha256'][:16]}…; "
       f"{_cp['the_sealed_receipt_declares']['receipt']} declares "
       f"{_cp['the_sealed_receipt_declares']['sha256'][:16]}…; the "
       f"receipt's load/emit pair agrees: "
       f"{_cp['at_load_and_at_emit']['agree']}")

    #: ITS OWN DIRECTORY: the fixture dir above has closed by now, and a
    #: cell that writes into a gone directory fails for a reason that has
    #: nothing to do with what it tests.
    _cpd_ctx = tempfile.TemporaryDirectory()
    _cpd = Path(_cpd_ctx.name)

    def _refuses_cp(mutate, dr=None, base=None, day="20260906"):
        bad = json.loads(json.dumps(base if base is not None else good))
        mutate(bad)
        bp = _cpd / f"{EARLY_FAMILY}_{day}__20260101T000020Z.json"
        bp.write_text(json.dumps(bad))
        try:
            verify(bp, repo_root=root, data_root=dr)
            return "ADMITTED"
        except EarlyReadVerifyRefused as e:
            return str(e).split(":")[0]

    _cp_diff = _refuses_cp(lambda b: b["computation_params"].update(
        {"sha256": "d" * 64}))
    _cp_gone = _refuses_cp(lambda b: b.pop("computation_params"))
    with tempfile.TemporaryDirectory() as _dr:
        (Path(_dr) / "pm_5min/derived").mkdir(parents=True)
        _cp_absent = _refuses_cp(lambda b: None, dr=Path(_dr))
    ck("KNOWN-BAD, DRIVEN -- ***A DIFFERENT PARAMS DIGEST IS REFUSED BY "
       "NAME***: an early read that loaded something other than what the "
       "sealed run declared is a DIFFERENT COMPUTATION wearing the sealed "
       "run's claim, and its four days would be comparable neither to the "
       "sealed days nor to each other. An UNDECLARED digest refuses too, "
       "and so does an ABSENT receipt -- ***this is the one check that "
       "cannot be skipped without the whole claim going unchecked***",
       _cp_diff == "COMPUTATION_PARAMS_NOT_THE_SEALED_RUNS"
       and _cp_gone == "COMPUTATION_PARAMS_NOT_DECLARED"
       and _cp_absent
       == "COMPUTATION_PARAMS_NOT_CHECKABLE_SEALED_RECEIPT_ABSENT",
       f"a different digest -> {_cp_diff}; no computation_params -> "
       f"{_cp_gone}; the receipt not under the root -> {_cp_absent}")

    #: THE TWO REAL FINDINGS THIS CHECK TURNED UP ON ITS FIRST RUN,
    #: DRIVEN AGAINST THE BAR'S OWN RECEIPTS -- not a mutation of mine.
    _a03 = _fixture_artifact(_cpd, v17.name, sha17, _rows17["2026-09-03"],
                             day="2026-09-03")
    _a03["day_run"]["day"] = "2026-09-03"
    _a04 = _fixture_artifact(_cpd, v17.name, sha17, _rows17["2026-09-04"],
                             day="2026-09-04")
    _a04["day_run"]["day"] = "2026-09-04"
    _r03 = _refuses_cp(lambda b: None, base=_a03, day="20260903")
    _r04 = _refuses_cp(lambda b: None, base=_a04, day="20260904")
    ck("***AND THE CHECK FOUND TWO REAL THINGS ON ITS FIRST RUN, ON THE "
       "BAR'S OWN RECEIPTS.*** 2026-09-03's sealed receipt is RECONSTRUCTED "
       "-- the run stamped no params and the digest comes from git at the "
       "carrying commit -- so this reader refuses to certify the claim "
       "against it; and 2026-09-04's receipt STAMPED params v14, not the "
       "v15 the early read loads, so an artifact claiming v15 for that day "
       "is refused NOT_THE_SEALED_RUNS. ***Neither is ruled on here***: the "
       "reader reports which state each day is in",
       _r03 == "COMPUTATION_PARAMS_RECONSTRUCTED_NOT_STAMPED"
       and _r04 == "COMPUTATION_PARAMS_NOT_THE_SEALED_RUNS",
       f"09-03 -> {_r03}; 09-04 -> {_r04}")
    _cpd_ctx.cleanup()

    _moved = _params_declared_by_the_sealed_run({"provenance": {
        "params": {"sha256": "a" * 64, "path": "p.json"},
        "digests_at_load_and_at_emit": {"inputs": {"params": {
            "sha256_at_load": "a" * 64, "sha256_at_emit": "b" * 64,
            "agrees": False}}}}})
    _leak_probe = _params_declared_by_the_sealed_run({"provenance": {
        "params": {"sha256": "c" * 64, "path": "p.json"},
        "D_E0": -4242.42, "null_mean": 17.5}, "per_day_sealed_artifacts": [
            {"arm": "X", "D_E0": -4242.42}]})
    ck("AND THE RECEIPT IS OPENED FOR ONE PURPOSE AND READ FOR ONE THING: a "
       "receipt whose own params digests DISAGREE between load and emit is "
       "refused by its own name, and a receipt carrying planted ECONOMIC "
       "fields yields NONE of them to this reader -- the read scope is a "
       "declared constant, not a habit",
       _moved["the_receipts_own_two_readings_agree"] is False
       and "4242.42" not in json.dumps(_leak_probe)
       and "null_mean" not in json.dumps(_leak_probe)
       and set(_leak_probe["read_scope"]) == set(SEALED_RECEIPT_READ_SCOPE),
       f"load/emit disagreement is visible: "
       f"{_moved['the_receipts_own_two_readings_agree']}; a planted D_E0 "
       f"and null_mean reach none of {sorted(_leak_probe)}")

    # -- DA 131 / REV 104B S11 #3: THE TWO NEW ROW KINDS ---------------
    import gzip as _gz                                        # noqa: PLC0415

    def _ledger(rows, where):
        p = Path(where) / "led.jsonl.gz"
        with _gz.open(p, "wt") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p

    _base_rows = [
        {"row": "HEADER", "schema_version": 2, "arms": ["A"],
         "buy_side": "BUY_UP"},
        {"row": "ARM_SCALARS", "arm": "A", "arm_value_cents": 10.0,
         "baseline_value_cents": 4.0, "observed_D_E0": 6.0,
         "n_fills_arm": 1, "n_fills_baseline": 1, "n_cancels_issued": 0},
        {"row": "NULL_DRAW", "arm": "A", "i": 0, "value": 1.0},
        {"row": "FILL", "arm": "A", "book": "ARM", "slug": "s1",
         "side": "BUY_UP", "px_cents": 40.0, "size": 1.0,
         "mid_cents_at_markout": 50.0},
        {"row": "FILL", "arm": "A", "book": "BASELINE", "slug": "s1",
         "side": "BUY_UP", "px_cents": 46.0, "size": 1.0,
         "mid_cents_at_markout": 50.0},
    ]
    #: the ruled decomposition, computed by hand for this fixture:
    #: ARM   buys 1 @ 40, up wins -> trades -40, residual +100, total  60
    #: BASE  buys 1 @ 46, up wins -> trades -46, residual +100, total  54
    #: D_settle = 60 - 54 = 6
    _sl = lambda book, px: {                                  # noqa: E731
        "row": "SETTLEMENT_SLUG", "arm": "A", "book": book, "slug": "s1",
        "n_fills": 1, "net_shares": 1.0, "trades_leg_cents": -px,
        "settle_cents": 100.0, "up_won": True,
        "residual_leg_cents": 100.0, "total_cents": 100.0 - px,
        "status": "VERIFIED_AGREE"}
    _good_rows = _base_rows + [
        {"row": "SETTLEMENT_SCALARS", "arm": "A", "ruling": "R-801",
         "unit": "cents", "D_E_settle": 6.0, "arm_total_cents": 60.0,
         "baseline_total_cents": 54.0,
         "winner_source": {"path": "w.json"}},
        _sl("ARM", 40.0), _sl("BASELINE", 46.0)]
    with tempfile.TemporaryDirectory() as _sd:
        _r_ok = recompute_from_the_ledger(_ledger(_good_rows, _sd))
        _se = _r_ok["settlement"]
        _r_absent = recompute_from_the_ledger(_ledger(_base_rows, _sd))

        def _settle_refuses(rows):
            try:
                recompute_from_the_ledger(_ledger(rows, _sd))
                return "ADMITTED"
            except EarlyReadVerifyRefused as e:
                return str(e).split(":")[0]

        _bad_scalar = _settle_refuses(
            _base_rows + [dict(_good_rows[-3], D_E_settle=99.0),
                          _sl("ARM", 40.0), _sl("BASELINE", 46.0)])
        _bad_status = _settle_refuses(
            _base_rows + [_good_rows[-3],
                          dict(_sl("ARM", 40.0), status="DISAGREE"),
                          _sl("BASELINE", 46.0)])
    ck("DA 131 -- ***THE TWO NEW ROW KINDS ARE READ AND RECOMPUTED, NOT "
       "SKIPPED***: per slug the trades leg, the residual leg and the total; "
       "per arm the settlement D from the FILL rows and the verified "
       "winners, compared against the SETTLEMENT_SCALARS row. The legs are "
       "ASSERTED equal to the per-fill form -- two expressions of one "
       "quantity, computed separately",
       _se["status"] == "SETTLEMENT_ROWS_PRESENT"
       and _se["per_arm"]["A"]["compared"]["D_E_settle"]["recomputed"] == 6.0
       and _se["per_arm"]["A"]["compared"]["D_E_settle"]["agrees"]
       and _se["per_arm"]["A"]["legs"]["ARM"]["trades_leg_cents"] == -40.0
       and _se["per_arm"]["A"]["legs"]["ARM"]["residual_leg_cents"] == 100.0
       and _se["per_arm"]["A"]["legs"]["BASELINE"]["total_cents"] == 54.0,
       f"D_settle recomputed "
       f"{_se['per_arm']['A']['compared']['D_E_settle']['recomputed']!r} "
       f"against the row's "
       f"{_se['per_arm']['A']['compared']['D_E_settle']['in_the_row']!r}; "
       f"arm legs {_se['per_arm']['A']['legs']['ARM']}")
    ck("KNOWN-BADS, DRIVEN, ALL THREE: a planted SETTLEMENT_SCALARS "
       "mismatch is SETTLEMENT_SCALARS_DISAGREE; a planted DISAGREE winner "
       "status is SETTLEMENT_WINNER_NOT_VERIFIED -- ***every fill in that "
       "slug is valued by a winner the venue and Chainlink do not agree "
       "on***; and a ledger without the rows reads SETTLEMENT_ROWS_ABSENT, "
       "***never zero***",
       _bad_scalar == "SETTLEMENT_SCALARS_DISAGREE"
       and _bad_status == "SETTLEMENT_WINNER_NOT_VERIFIED"
       and _r_absent["settlement"]["status"] == "SETTLEMENT_ROWS_ABSENT"
       and not _r_absent["settlement"]["per_arm"],
       f"planted D_E_settle=99 -> {_bad_scalar}; planted DISAGREE -> "
       f"{_bad_status}; no rows -> "
       f"{_r_absent['settlement']['status']}")

    # -- DA 130 / REV 104A S7 #2: THE FAMILY'S HEAD, BY THE PAIR -------
    with tempfile.TemporaryDirectory() as _hd:
        _h = Path(_hd)

        def _art(day, stamp, sup=None):
            b = json.loads(json.dumps(good))
            b["day_run"]["day"] = day
            if sup is not None:
                b[SUPERSEDES_FIELD] = sup
            p = _h / f"{EARLY_FAMILY}_{day.replace('-', '')}__{stamp}.json"
            p.write_text(json.dumps(b))
            return p

        _one = _art("2026-09-01", "20260101T000000Z")
        _r1 = resolve_early_read_head("2026-09-01", derived=_h)
        _two_a = _art("2026-09-02", "20260101T000000Z")
        _two_b = _art("2026-09-02", "20260101T000100Z")
        _amb = ""
        try:
            resolve_early_read_head("2026-09-02", derived=_h)
        except EarlyReadVerifyRefused as e:
            _amb = str(e).split(":")[0]
        _ch_a = _art("2026-09-08", "20260101T000000Z")
        _ch_b = _art("2026-09-08", "20260101T000100Z",
                     sup={"path": _ch_a.name, "sha256": _sha(_ch_a)})
        _r2 = resolve_early_read_head("2026-09-08", derived=_h)
        _bad_a = _art("2026-09-09", "20260101T000000Z")
        _bad_b = _art("2026-09-09", "20260101T000100Z",
                      sup={"path": _bad_a.name, "sha256": "0" * 64})
        _mm = ""
        try:
            resolve_early_read_head("2026-09-09", derived=_h)
        except EarlyReadVerifyRefused as e:
            _mm = str(e).split(":")[0]
        _pfx_a = _art("2026-09-10", "20260101T000000Z")
        _pfx_b = _art("2026-09-10", "20260101T000100Z",
                      sup={"path": _pfx_a.name, "sha256": _sha(_pfx_a)[:16]})
        _pfx = ""
        try:
            resolve_early_read_head("2026-09-10", derived=_h)
        except EarlyReadVerifyRefused as e:
            _pfx = str(e)
        _absent = ""
        try:
            resolve_early_read_head("2099-01-01", derived=_h)
        except EarlyReadVerifyRefused as e:
            _absent = str(e).split(":")[0]
        _stand_head = head_standing(_ch_b, derived=_h)
        _stand_old = head_standing(_ch_a, derived=_h)
    ck("DA 130 -- ***THE FAMILY HAS A HEAD, AND IT IS RESOLVED BY THE PAIR***: "
       "one artifact IS the head; a verified chain of two resolves to the "
       "LATER one; and the superseded one is readable as PROVENANCE, "
       "labelled SUPERSEDED, never quoted as the day's read",
       _r1["head"] == _one.name and _r1["the_sole_artifact_is_the_head"]
       and _r2["head"] == _ch_b.name and _r2["superseded"] == [_ch_a.name]
       and _stand_head["label"] == "HEAD"
       and _stand_old["label"] == "SUPERSEDED"
       and _stand_old["the_head_is"] == _ch_b.name,
       f"one -> {_r1['head']}; a chain -> {_r2['head']} (superseded "
       f"{_r2['superseded']}); the older one reads {_stand_old['label']}")
    ck("KNOWN-BADS, DRIVEN, ALL FOUR -- ***TWO UNCHAINED ARTIFACTS ARE "
       "AMBIGUOUS AND THIS READER WILL NOT PICK BY STAMP***; a chain naming "
       "a digest the file does not have is SUPERSESSION_PAIR_MISMATCH; a "
       "SIXTEEN-HEX PREFIX is refused as not the pair (R-754's v17 lesson "
       "in this family); and a day with no artifact is ABSENT",
       _amb == "EARLY_READ_HEAD_AMBIGUOUS"
       and _mm == "SUPERSESSION_PAIR_MISMATCH"
       and _pfx.startswith("SUPERSESSION_PAIR_MISMATCH")
       and "not 64" in _pfx
       and _absent == "EARLY_READ_HEAD_ABSENT",
       f"two unchained -> {_amb}; wrong digest -> {_mm}; 16-hex prefix -> "
       f"{_pfx.split(':')[0]} ('not 64 lowercase hex'); none -> {_absent}")

    # -- and THE FOUR REAL DAYS, each a single head today ---------------
    _real = {}
    for _d in ("2026-09-03", "2026-09-04", "2026-09-05", "2026-09-06"):
        try:
            _real[_d] = resolve_early_read_head(_d, data_root=root / "data")
        except EarlyReadVerifyRefused as e:
            _real[_d] = {"REFUSED": str(e).split(":")[0]}
    ck("AND THE FOUR REAL DAYS RESOLVE TODAY -- each has exactly ONE "
       "artifact, no `supersedes` field anywhere yet, and the sole artifact "
       "IS the head. ***That is the state DE 138's rule has to preserve***: "
       "the moment a second is written without the field, this day becomes "
       "AMBIGUOUS by the cell above",
       all(isinstance(v, dict) and v.get("n_artifacts") == 1
           and v.get("the_sole_artifact_is_the_head") and not v.get("links")
           for v in _real.values()),
       "; ".join(f"{d}: {v.get('head', v.get('REFUSED'))}"
                 for d, v in sorted(_real.items())))

    # -- DA 126: THE LEDGER IS A REPORTED STATUS, NEVER A GUESS --------
    _lg_null = check_decision_ledger({"day_run": {"decision_ledger": None}})
    _lg_gone = check_decision_ledger({"day_run": {}})
    _lg_here = check_decision_ledger(
        {"day_run": {"decision_ledger": {"rows": 3}}})
    _lg_res = json.loads(json.dumps(res))
    _lg_res["day_run_ledger_probe"] = None
    _lg_res["decision_ledger"] = _lg_null
    _lg_table = print_table(_lg_res)
    ck("DA 126 -- ***THE LEDGER'S ABSENCE IS A NAMED STATUS AND THE TABLE "
       "STILL PRINTS***: a NULL `day_run.decision_ledger` is LEDGER_ABSENT "
       "with the block quoted as it stands, a MISSING KEY is a different "
       "name, and a present one is LEDGER_PRESENT. ***The 0-cancel "
       "baseline's own value is NOT derived from anything else***: D(E0) is "
       "a DIFFERENCE against that baseline, and a difference does not "
       "contain either of its terms",
       _lg_null["status"] == "LEDGER_ABSENT"
       and _lg_null["the_block_as_it_stands"] is None
       and _lg_gone["status"] == "LEDGER_KEY_ABSENT"
       and _lg_here["status"] == "LEDGER_PRESENT"
       and _lg_here["what_cannot_be_derived"] is None
       and "LEDGER_ABSENT" in _lg_table
       and "`day_run.decision_ledger` = null" in _lg_table
       and "NOT approximated" in _lg_table
       and "CONDVALUE_X_SKEW" in _lg_table,
       f"null -> {_lg_null['status']}; key missing -> {_lg_gone['status']}; "
       f"present -> {_lg_here['status']}; the table carries the line AND "
       f"the arm rows")

    # -- DA 125: BOTH SHAPES, because my fixture had reproduced my own
    # assumption and the first real artifact caught it -----------------
    _flat = json.loads(json.dumps(good))
    for _b in _flat["day_run"]["per_day_sealed_artifacts"]:
        _b.update(_b.pop("economic"))
    with tempfile.TemporaryDirectory() as _fd:
        _fp2 = Path(_fd) / f"{EARLY_FAMILY}_20260906__20260101T000040Z.json"
        _fp2.write_text(json.dumps(_flat))
        _rflat = verify(_fp2, repo_root=root)
    ck("DA 125 -- ***BOTH SHAPES ARE READ, AND THE ONE IN USE IS REPORTED.*** "
       "DE's UNSEALED emission nests the six under `economic`; the SEALED "
       "receipts carry the same names FLAT at the arm's top level. This "
       "reader was built against the sealed shape and ***its own fixture "
       "reproduced that assumption***, so the first REAL artifact refused "
       "ECONOMIC_FIELD_MISSING on an artifact that had every field -- an "
       "honest refusal from a wrong locator. The fixture now carries DE's "
       "real shape and the flat one is driven beside it",
       res["census"]["where_the_six_were_found"] == ["economic_block"]
       and _rflat["census"]["where_the_six_were_found"]
       == ["flat_on_the_arm"]
       and res["census"]["per_arm"] == _rflat["census"]["per_arm"],
       f"nested -> {res['census']['where_the_six_were_found']}; flat -> "
       f"{_rflat['census']['where_the_six_were_found']}; the same values "
       f"either way: "
       f"{res['census']['per_arm'] == _rflat['census']['per_arm']}")

    # -- R-764: THE RULING RIDES BESIDE THE REFUSAL, NEVER INSTEAD ------
    _rd = tempfile.TemporaryDirectory()
    _rp = Path(_rd.name)

    def _write(base, day):
        p = _rp / f"{EARLY_FAMILY}_{day.replace('-', '')}__20260101T000030Z.json"
        p.write_text(json.dumps(base))
        return p

    _u03 = _write(_a03, "2026-09-03")
    _u04 = _write(_a04, "2026-09-04")
    _u06 = _write(good, "2026-09-06")
    _ur03 = verify_under_ruling(_u03, repo_root=root)
    _ur04 = verify_under_ruling(_u04, repo_root=root)
    _ur06 = verify_under_ruling(_u06, repo_root=root)
    _t03, _t04, _t06r = (print_table(_ur03), print_table(_ur04),
                         print_table(_ur06))
    _still03 = "ADMITTED"
    try:
        verify(_u03, repo_root=root)
    except EarlyReadVerifyRefused as e:
        _still03 = str(e).split(":")[0]
    ck("R-764 -- ***THE RULING RIDES BESIDE THE REFUSAL AND NEVER REPLACES "
       "IT***: the SAME 09-03 artifact still REFUSES under `verify()`, the "
       "entry the GOs use, while `--print-under-ruling R-764` prints the "
       "table with a MATERIALITY line that NAMES the refusal code, both "
       "params versions, the measured delta and the ruling's id",
       _still03 == "COMPUTATION_PARAMS_RECONSTRUCTED_NOT_STAMPED"
       and _ur03["under_ruling"]["applies"] is True
       and _ur03["verdict"] == "REFUSED"
       and _ur03["IS_A_VERIFICATION"] is False
       and "MATERIALITY -- REFUSED "
       "COMPUTATION_PARAMS_RECONSTRUCTED_NOT_STAMPED" in _t03
       and "R-764" in _t03 and "The refusal STANDS" in _t03
       and "seal scope and closure naming only" in _t03,
       f"verify() -> {_still03}; under the ruling -> verdict "
       f"{_ur03['verdict']}, applies "
       f"{_ur03['under_ruling']['applies']}, sealed "
       f"{_ur03['under_ruling']['the_sealed_runs_params']['version']} "
       f"({_ur03['under_ruling']['the_sealed_runs_params']['state']}) vs "
       f"read {_ur03['under_ruling']['the_reads_params']['version']}")

    ck("DA 126 -- ***THE RULING'S SENTENCE DOES NOT STRETCH OVER A SPAN IT "
       "NEVER DIFFED***: R-764 measured v14 -> v15 leaf by leaf. Where the "
       "pair in front of the reader is a DIFFERENT span, the table says so "
       "on its own line and the record carries both spans; where it is the "
       "same span, no such line appears",
       _ur03["under_ruling"]["the_span_the_ruling_MEASURED"]
       == {"from": "v14", "to": "v15"}
       and _ur03["under_ruling"][
           "observed_span_is_within_what_was_measured"] is True
       and "AND THE SPAN IS NOT THE ONE" not in _t03,
       f"09-03's pair {_ur03['under_ruling']['the_span_IN_FRONT_OF_IT']} vs "
       f"measured {_ur03['under_ruling']['the_span_the_ruling_MEASURED']} -> "
       f"within: "
       f"{_ur03['under_ruling']['observed_span_is_within_what_was_measured']}")

    ck("AND THE PER-DAY LABEL SAYS WHICH STATE THE DAY IS IN, in the words "
       "the table carries: 09-03 'computed under v15; sealed run "
       "RECONSTRUCTED v14; counts visible since 2026-09-06T14:01Z "
       "(eight-name scope)'; 09-04 'computed under v15; sealed run stamped "
       "v14'; 09-06 'params match'",
       params_label(_ur03) == ("computed under v15; sealed run "
                               "RECONSTRUCTED v14; counts visible since "
                               "2026-09-06T14:01Z (eight-name scope)")
       and params_label(_ur04) == ("computed under v15; sealed run stamped "
                                   "v14")
       and params_label(_ur06) == "params match"
       and _ur04["under_ruling"]["refusal_code"]
       == "COMPUTATION_PARAMS_NOT_THE_SEALED_RUNS"
       and _ur06["under_ruling"]["applies"] is False,
       f"09-03: {params_label(_ur03)} | 09-04: {params_label(_ur04)} | "
       f"09-06: {params_label(_ur06)} (ruling applies "
       f"{_ur06['under_ruling']['applies']})")

    def _under_ruling_refuses(mutate, base=None, day="2026-09-06"):
        bad = json.loads(json.dumps(base if base is not None else good))
        mutate(bad)
        p = _write(bad, day)
        try:
            verify_under_ruling(p, repo_root=root)
            return "ADMITTED"
        except EarlyReadVerifyRefused as e:
            return str(e).split(":")[0]

    _u_pair = _under_ruling_refuses(
        lambda b: b["ruling"].update({"sha256": "e" * 64}))
    _u_num = _under_ruling_refuses(
        lambda b: b["economics_field_availability"][
            "not_computed_by_this_path"].update(
                {"rho_adverse_over_spread": 0.42}))
    _u_lab = _under_ruling_refuses(lambda b: b.update(
        {"is_a_validation": True}))
    _u_extra = _under_ruling_refuses(
        lambda b: b["day_run"]["per_day_sealed_artifacts"][0].update(
            {"D_E_MINUS_R": -12.0}))
    _u_unknown = "ADMITTED"
    try:
        verify_under_ruling(_u06, ruling_id="R-999", repo_root=root)
    except EarlyReadVerifyRefused as e:
        _u_unknown = str(e).split(":")[0]
    _rd.cleanup()
    ck("KNOWN-BADS, DRIVEN UNDER THE MODE ITSELF -- ***EVERY OTHER REFUSAL "
       "STILL REFUSES WITH `--print-under-ruling`***: a wrong ruling pair, a "
       "status carrying a NUMBER, a relabelled read and an extra economic "
       "key all refuse under it, and an UNKNOWN ruling id refuses too "
       "(a mode that printed under a ruling nobody landed would be this "
       "reader ruling)",
       _u_pair == "EARLY_READ_RULING_NOT_THE_PAIR"
       and _u_num == "EARLY_READ_STATUS_CARRIES_A_NUMBER"
       and _u_lab == "EARLY_READ_LABEL_DIFFERS"
       and _u_extra == "EARLY_READ_EXTRA_ECONOMIC_KEY"
       and _u_unknown == "UNKNOWN_RULING",
       f"wrong pair -> {_u_pair}; a number in a status -> {_u_num}; "
       f"is_a_validation true -> {_u_lab}; an extra economic key -> "
       f"{_u_extra}; ruling R-999 -> {_u_unknown}")

    #: REV 90 S A2 -- THE COUNTS' PROVENANCE IS SAID PER DAY.
    #: 09-06 is driven END TO END through `verify` + `print_table`; 09-03
    #: is driven at the mapping, because its sealed receipt is
    #: RECONSTRUCTED and this reader refuses to certify a claim against a
    #: reconstruction (the cell above). The property here is the PER-DAY
    #: mapping, and it is asserted on both days either way.
    _t06 = print_table(res)
    _p03, _p06 = counts_provenance("2026-09-03"), counts_provenance(
        "2026-09-06")
    ck("REV 90 S A2 -- ***THE THREE COUNTS CARRY THEIR PROVENANCE, PER DAY, "
       "IN THE PRINTED TABLE***: 09-03's were VISIBLE IN THE OPEN since "
       "2026-09-06T14:01Z under the EIGHT-name seal scope, and the other "
       "three days' are unsealed BY THIS READ under the ELEVEN-name scope. "
       "A table showing four days' counts without saying which is which "
       "would invite a comparison across two different blindness states",
       "VISIBLE IN THE OPEN" in _p03 and "2026-09-06T14:01Z" in _p03
       and "unsealed BY THIS READ" in _p06
       and "VISIBLE IN THE OPEN" not in _p06
       and "unsealed BY THIS READ" in _t06
       and "VISIBLE IN THE OPEN" not in _t06,
       f"09-03 -> {_p03[:52]}…; 09-06 -> {_p06[:52]}…; the 09-06 table "
       f"carries its own line")

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
    ap.add_argument("--day", metavar="YYYY-MM-DD or YYYYMMDD",
                    help="resolve the day's HEAD by the supersedes pair and "
                         "read it, instead of naming a path")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="print the coordinator's table (values)")
    ap.add_argument("--print-under-ruling", dest="under_ruling",
                    default=None, metavar="RULING_ID",
                    help="print the table BESIDE a refusal the named ruling "
                         "covers (R-764). The refusal still stands and is "
                         "named in the table; every other refusal still "
                         "refuses under this mode")
    ap.add_argument("--four-day-table", nargs="*", metavar="ARTIFACT",
                    default=None,
                    help="print the four days in one block, each read "
                         "through the same reader and the same ruling mode")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        return 1 if selftest()[1] else 0
    if a.four_day_table:
        print(four_day_table(a.four_day_table, data_root=a.data_root))
        return 0
    if a.day and not a.verify:
        try:
            h = resolve_early_read_head(a.day, data_root=a.data_root)
        except EarlyReadVerifyRefused as e:
            print(str(e))
            return 2
        print(f"HEAD for {a.day}: {h['head']} {h['sha256'][:16]}… "
              f"({h['n_artifacts']} artifact(s); superseded "
              f"{h['superseded'] or 'none'})")
        a.verify = h["path"]
    if not a.verify:
        ap.error("--selftest, or --verify <artifact> [--print], or --day")
    #: THREE OUTCOMES, THREE CODES. 2 is a REFUSAL -- the instrument declined
    #: to run on what it was given; 1 would be a check that RAN and flagged;
    #: 0 is verified. A caller that could not tell them apart would read "you
    #: handed me a sealed receipt" as "the read disagrees".
    try:
        if a.under_ruling:
            res = verify_under_ruling(a.verify, ruling_id=a.under_ruling,
                                      data_root=a.data_root)
        else:
            res = verify(a.verify, data_root=a.data_root)
    except EarlyReadVerifyRefused as e:
        print(str(e))
        return 2
    if a.output:
        a.output.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    #: 3 -- PRINTED UNDER A RULING, AND STILL REFUSED. Never 0: a caller
    #: that could not tell this from a clean verification would read "the
    #: coordinator ruled the delta immaterial" as "there was no delta".
    if (res.get("under_ruling") or {}).get("applies"):
        print(print_table(res))
        return 3
    if a.do_print or a.under_ruling:
        print(print_table(res))
    else:
        print(f"{res['verdict']} -- day {res['day']}, ruling "
              f"{res['ruling']['name']} {res['ruling']['sha256'][:16]}…, "
              f"{res['census']['n_arms']} arms. {LABEL_LINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
