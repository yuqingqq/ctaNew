#!/usr/bin/env python3
"""BE-side trajectory export for DA's external-arm parity contract.

CONTRACT: da_replay_parity_battery.py, canon `replay_traj_canon_v1`. That module
states its external interface is a DATA contract and imports nothing from BE.

THIS MODULE DECLARES ITS OWN FIELD LIST AND DOES NOT IMPORT DA'S.

That is deliberate and it is the whole point. If BE imported DA's EVENT_FIELDS,
the two sides would agree BY CONSTRUCTION and the contract check would be
vacuous -- it would verify that one list equals itself. A data contract exists
so that two independent implementations must be SHOWN to agree, and the showing
is a falsifier that compares the two declarations (see agreement_with_contract).
Same shape as annotation_canon_v1: agreement PROVEN, never assumed.

The first real BE trajectory is the first thing that can falsify this contract,
so the export path is built and driven on synthetic data BEFORE any real run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

# --- BE's INDEPENDENT declaration of the contract, transcribed from the spec --
BE_CANON = "replay_traj_canon_v1"
BE_EVENT_FIELDS = ("t", "seq", "kind", "slug", "side", "gen", "qty", "price",
                   "note")
BE_KINDS = ("PLACE", "PLACE_WITHHELD", "CANCEL_REQUESTED", "CANCEL_EFFECTIVE",
            "CANCEL_SUPPRESSED", "FILL", "FILL_STALE")


class ExportRefused(RuntimeError):
    """Refuse at the producer rather than emit something the consumer rejects.

    A producer that emits a malformed trajectory and lets the consumer refuse it
    has moved the error one process away from its cause."""


def make_event(t, seq, kind, slug, side, gen, qty, price, note="") -> dict:
    """One event, with EXACTLY the declared fields and nothing else."""
    if kind not in BE_KINDS:
        raise ExportRefused(f"unknown kind {kind!r}; declared: {BE_KINDS}")
    return {"t": float(t), "seq": int(seq), "kind": kind, "slug": str(slug),
            "side": str(side), "gen": int(gen), "qty": float(qty),
            "price": float(price), "note": str(note)}


def export_trajectory(arm: str, events: list) -> dict:
    """A contract-shaped trajectory object.

    Refuses EMPTY, because the contract refuses it downstream and an empty
    trajectory is a producer bug rather than a population statement."""
    if not events:
        raise ExportRefused(
            f"arm {arm!r} has NO events. An empty trajectory is a producer "
            f"defect; the contract refuses it and so should the producer.")
    for i, e in enumerate(events):
        missing = [f for f in BE_EVENT_FIELDS if f not in e]
        extra = [k for k in e if k not in BE_EVENT_FIELDS]
        if missing or extra:
            raise ExportRefused(
                f"event {i}: missing={missing} undeclared={extra}. The field "
                f"set is exact in both directions.")
    return {"canon": BE_CANON, "arm": arm, "events": list(events)}


def agreement_with_contract() -> dict:
    """PROVE that BE's independent declaration matches DA's. Not assume it.

    Imported ONLY here, in the check -- never in the producer -- so the export
    path stays independent while the agreement is demonstrated."""
    import da_replay_parity_battery as DA
    disagreements = {}
    if BE_CANON != DA.CANON:
        disagreements["canon"] = (BE_CANON, DA.CANON)
    if tuple(BE_EVENT_FIELDS) != tuple(DA.EVENT_FIELDS):
        disagreements["event_fields"] = (BE_EVENT_FIELDS, DA.EVENT_FIELDS)
    if tuple(sorted(BE_KINDS)) != tuple(sorted(DA.KINDS)):
        disagreements["kinds"] = (sorted(BE_KINDS), sorted(DA.KINDS))
    if disagreements:
        raise ExportRefused(
            f"BE's declaration DISAGREES with the contract: {disagreements}. "
            f"Two implementations that have never been shown to agree are not "
            f"a contract.")
    return {"agreed": True, "canon": BE_CANON,
            "n_fields": len(BE_EVENT_FIELDS), "n_kinds": len(BE_KINDS),
            "declared_independently": True}


def selftest() -> int:
    fails = []

    def ok(c, label):
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    import da_replay_parity_battery as DA

    ok(agreement_with_contract()["agreed"],
       "0 BE's INDEPENDENT declaration is PROVEN to match DA's contract "
       "(importing DA's list would make this vacuous)")

    def ev(seq, kind, gen=1, t=None, **kw):
        return make_event(t if t is not None else seq * 1.0, seq, kind,
                          "btc-updown-5m-1787650200", "BUY_UP", gen, 5.0, 0.5,
                          **kw)

    good = export_trajectory("CONDVALUE_NEUTRAL",
                             [ev(1, "PLACE"), ev(2, "FILL")])
    tr = DA.load_external_trajectory(good)
    ok(tr.arm == "CONDVALUE_NEUTRAL" and len(tr.events) == 2,
       "1 a well-formed export LOADS through the real contract loader "
       "(the guard is not a wall)")
    ok(DA.external_lifecycle(tr)["arm"] == "CONDVALUE_NEUTRAL",
       "2 and passes the real lifecycle invariants")

    # --- the contract's refusal rules, each driven against the REAL loader ---
    for lbl, obj in (
        ("not an object", ["nope"]),
        ("wrong canon", dict(good, canon="canon_v99")),
        ("unknown arm", dict(good, arm="BE_MADE_THIS_UP")),
        ("empty events", dict(good, events=[])),
        ("event not an object", dict(good, events=["x"])),
    ):
        try:
            DA.load_external_trajectory(obj)
            ok(False, f"3 the contract REFUSES: {lbl}")
        except DA.ParityRefused:
            ok(True, f"3 the contract REFUSES: {lbl}")

    _miss = dict(good["events"][0]); _miss.pop("note")
    try:
        DA.load_external_trajectory(dict(good, events=[_miss]))
        ok(False, "4 a MISSING field is refused")
    except DA.ParityRefused as e:
        ok("MISSING" in str(e), "4 a MISSING field is refused by the contract")
    _ext = dict(good["events"][0]); _ext["be_extra"] = 1
    try:
        DA.load_external_trajectory(dict(good, events=[_ext]))
        ok(False, "5 an UNDECLARED field is refused")
    except DA.ParityRefused as e:
        ok("UNDECLARED" in str(e),
           "5 an UNDECLARED field is refused — the set is exact in BOTH "
           "directions, so a helpful extra column is a refusal")

    # --- BE refuses at the PRODUCER, not only at the consumer ---
    try:
        export_trajectory("CONDVALUE_NEUTRAL", [])
        ok(False, "6 BE refuses an EMPTY trajectory at the producer")
    except ExportRefused:
        ok(True, "6 BE refuses an EMPTY trajectory at the PRODUCER, not one "
                 "process away at the consumer")
    try:
        make_event(1.0, 1, "TELEPORT", "s", "BUY_UP", 1, 5.0, 0.5)
        ok(False, "7 BE refuses an unknown kind at the producer")
    except ExportRefused:
        ok(True, "7 BE refuses an unknown KIND at the producer")
    try:
        export_trajectory("X", [{"t": 1.0}])
        ok(False, "8 BE refuses an incomplete event at the producer")
    except ExportRefused as e:
        ok("missing=" in str(e), "8 BE refuses an INCOMPLETE event at the "
                                 "producer, naming the missing fields")

    # --- lifecycle invariants BE must satisfy, driven on synthetic arms ------
    two_req = export_trajectory("CONDVALUE_NEUTRAL", [
        ev(1, "PLACE"), ev(2, "CANCEL_REQUESTED"), ev(3, "CANCEL_REQUESTED"),
        ev(4, "CANCEL_EFFECTIVE")])
    r = DA.external_lifecycle(DA.load_external_trajectory(two_req))
    ok(any(v is False for k, v in r.items() if isinstance(v, bool)),
       "9 TWO cancel requests on one generation FAIL a lifecycle invariant")

    fill_after = export_trajectory("CONDVALUE_NEUTRAL", [
        ev(1, "PLACE"), ev(2, "CANCEL_REQUESTED"), ev(3, "CANCEL_EFFECTIVE"),
        ev(4, "FILL_STALE")])
    r2 = DA.external_lifecycle(DA.load_external_trajectory(fill_after))
    ok(any(v is False for k, v in r2.items() if isinstance(v, bool)),
       "10 a FILL_STALE AFTER effectiveness fails — STALE is DEFINED as "
       "pre-effectiveness, so that mislabel is exactly what the check catches")

    orphan = export_trajectory("CONDVALUE_NEUTRAL", [
        ev(1, "PLACE"), ev(2, "CANCEL_EFFECTIVE")])
    r3 = DA.external_lifecycle(DA.load_external_trajectory(orphan))
    ok(any(v is False for k, v in r3.items() if isinstance(v, bool)),
       "11 an EFFECTIVE cancel with no REQUEST fails — otherwise "
       "requested==effective+suppressed could be satisfied by two "
       "compensating errors")

    clean = export_trajectory("CONDVALUE_NEUTRAL", [
        ev(1, "PLACE"), ev(2, "CANCEL_REQUESTED"), ev(3, "CANCEL_EFFECTIVE")])
    r4 = DA.external_lifecycle(DA.load_external_trajectory(clean))
    ok(all(v is not False for k, v in r4.items() if isinstance(v, bool)),
       "12 a CLEAN cancel lifecycle passes every invariant")

    print(f"\n{'BE TRAJECTORY EXPORT SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(selftest())
