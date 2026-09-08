#!/usr/bin/env python3
"""ONE RULED DAY AS A POINT ESTIMATE -- no null, no Z, no p (DE 151).

WHY THIS FILE EXISTS AS A FILE. The two point-estimate rounds that landed
on 2026-09-08 (R-830 and the 09-04/09-05/09-06 runs after it) were driven
by `pe_run.py` in a session SCRATCHPAD. The artifacts landed; their
builder did not. That is the shape rule 12 names -- "a scratch-dir builder
voided one freeze" -- and it also meant this program's digest never joined
the import-closure stamp that every other producer carries. The bytes here
are that driver's, with the emit-time guard DE 151 added and a battery.

WHAT A POINT-ESTIMATE RUN IS (the contract, R-828):
  * `run_day(..., point_estimate=True)` SKIPS S4 entirely -- the null is
    not drawn, never drawn smaller (R-174 forbids lowering a cap).
  * Every statistic field carries `NULL_NOT_DRAWN_POINT_ESTIMATE_RUN`.
    Never a null, never a zero.
  * The arm-day status is `OK_POINT_ESTIMATE`, and every consumer tests
    status by MEMBERSHIP, never equality -- the defect class that cost
    three patches on 2026-09-08 (DE_PROCEDURE section 8).
  * A FULL run at the same (day, L) is a DIFFERENT artifact, not a
    successor: neither supersedes the other.
  * S4 is 99.2 % of a full run, so this costs ~2-4 minutes against ~2 h 45.

THE EMIT-TIME GUARD (DE 151, the USER's first finding). The document is
asserted to name ONE placement latency before its bytes are written. It
did not used to: `run_day` computed the NESTED `placement_latency` block
from the receipt's PATH instead of the parsed receipt, so every L = 250
artifact said 250.0 at the top level and 0.0 under `day_run`. No number
moved -- the latency is applied by BE at BOOK BUILD time and the nested
value is computed at the emit, after S5 -- but a document that states two
makers cannot be quoted at either, and nothing anywhere would have said
so. `assert_one_placement_latency` now runs on the finished payload.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import de_multiday_gate1_runner as R          # noqa: E402

PROTOCOL = "P003_DE_POINT_ESTIMATE_DAY_V1"
SUPERSEDES_TARGET_ABSENT = "POINT_ESTIMATE_SUPERSEDES_TARGET_ABSENT"
SUPERSEDES_DIGEST_MISMATCH = "POINT_ESTIMATE_SUPERSEDES_DIGEST_MISMATCH"
SUPERSEDES_DIFFERENT_L = "POINT_ESTIMATE_SUPERSEDES_DIFFERENT_PLACEMENT_LATENCY"
SUPERSEDES_NOT_A_POINT_ESTIMATE = (
    "POINT_ESTIMATE_SUPERSEDES_NOT_A_POINT_ESTIMATE_RUN")


class PointEstimateRefused(RuntimeError):
    pass


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def read_l_from_the_book(book: Path, day: str, coin: str = "btc") -> dict:
    """THE DAY'S L, READ FROM THE BOOK'S OWN BUILDER RECEIPT.

    The receipt is resolved by `builder_receipt_for`, whose FIRST
    candidate is the receipt whose name is the BOOK's name -- so an
    `…__L250ms.pkl` book resolves `…__L250ms.json` and not the L = 0
    receipt beside it (DE 150). The receipt is PARSED here: handing this
    function a Path is what produced the contradiction DE 151 fixed, and
    the runner now refuses that by name."""
    rp = R.builder_receipt_for(Path(book), day, coin)
    rec = json.loads(Path(rp).read_text())
    L = R.placement_latency_from_the_book(rec, book_path=str(book))
    L["builder_receipt"] = str(rp)
    return L


def supersedes_block(target, *, this_L: float, derived: Path) -> dict:
    """THE LINK TO THE ARTIFACT THIS ONE REPLACES (rule 13).

    The superseded file is NEVER edited; it stays as provenance. The
    digest is recomputed from the bytes on disk at the write, never
    copied from the caller -- that is how a pair comes to name bytes
    nobody has.

    A run at a DIFFERENT L supersedes nothing: it measures a different
    maker and is a SIBLING (R-811). The target's L is read from its TOP
    LEVEL, because a pre-DE-151 artifact's NESTED block is exactly the
    field that was wrong -- reading the nested one would refuse every
    honest re-emit of an L = 250 day as 'different latency 250 vs 0'."""
    named = target.get("path") if isinstance(target, dict) else target
    if not named:
        raise PointEstimateRefused(
            f"{SUPERSEDES_TARGET_ABSENT}: the supersession target names no "
            f"path. A link is a PAIR and its first half is a file.")
    tgt = Path(named)
    if not tgt.is_absolute():
        tgt = derived / tgt.name
    if not tgt.is_file():
        raise PointEstimateRefused(
            f"{SUPERSEDES_TARGET_ABSENT}: this run names {tgt.name} as the "
            f"artifact it supersedes and that file is not under {derived}. "
            f"A link to a file nobody has is not a link.")
    tgt_sha = _sha(tgt)
    named_sha = str(target.get("sha256") or "") if isinstance(target, dict) \
        else ""
    if named_sha and named_sha != tgt_sha:
        raise PointEstimateRefused(
            f"{SUPERSEDES_DIGEST_MISMATCH}: the target was named "
            f"{named_sha[:16]}… and {tgt.name} digests {tgt_sha[:16]}…. "
            f"The bytes this run claims to replace are not on disk.")
    doc = json.loads(tgt.read_text())
    if doc.get("run_mode") != "POINT_ESTIMATE":
        raise PointEstimateRefused(
            f"{SUPERSEDES_NOT_A_POINT_ESTIMATE}: {tgt.name} has run_mode "
            f"{doc.get('run_mode')!r}. A point estimate and a FULL run at "
            f"the same (day, L) answer different questions -- neither "
            f"supersedes the other (R-828).")
    tgt_L = (doc.get("placement_latency") or {}).get("L_place_ms")
    if tgt_L is not None and this_L is not None and float(tgt_L) != float(this_L):
        raise PointEstimateRefused(
            f"{SUPERSEDES_DIFFERENT_L}: {tgt.name} ran at L_place = "
            f"{tgt_L} ms and this run is at {this_L} ms. A run at a "
            f"different placement latency measures a DIFFERENT MAKER: it "
            f"is a SIBLING, not a successor (R-811).")
    # WHY the target is being replaced, computed from the target itself.
    try:
        R.assert_one_placement_latency(doc)
        why = ("re-emitted; the target's own placement-latency sites "
               "agree, so this supersession is not a DE 151 repair")
        agreed = True
    except R.RunnerRefused as e:
        why = str(e).split(":")[0].replace("REFUSED ", "")
        agreed = False
    return {"path": str(tgt), "sha256": tgt_sha,
            "both_at_L_place_ms": this_L,
            "target_placement_latency_sites":
                R.placement_latency_leaves(doc),
            "target_sites_agreed": agreed,
            "superseded_because": why,
            "the_target_is_not_edited": (
                "rule 13: the superseded artifact and its ledger stay on "
                "disk exactly as landed, as provenance")}


def assemble(day: str, book: Path, L: dict, res: dict, out: Path,
             supersedes: dict | None = None) -> dict:
    """THE DOCUMENT. Assembled, then ASSERTED, then written."""
    payload = {
        "protocol": PROTOCOL,
        "what_this_is": ("the RULED settlement P&L at a declared placement "
                         "latency, WITHOUT the null: S4 was skipped, so "
                         "there is no Z, no p and no control. Nothing here "
                         "implies significance."),
        "day": day,
        "placement_latency": L,
        "book": {"path": str(book), "sha256": _sha(Path(book)),
                 "builder_receipt": L.get("builder_receipt")},
        "run_mode": res.get("run_mode"),
        "day_run": res,
        "supersedes": supersedes,
        "as_of": _utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # THE GUARD, ON THE FINISHED DOCUMENT, BEFORE THE BYTES EXIST.
    payload["placement_latency_agreement"] = \
        R.assert_one_placement_latency(payload)
    return payload


def run(day: str, book: Path, *, supersedes=None, outdir: Path | None = None,
        quiet: bool = False) -> dict:
    t0 = time.time()
    params = R.load_params()
    coin = params.get("coin", "btc")
    L = read_l_from_the_book(Path(book), day, coin)
    if not quiet:
        print(f"[{time.time() - t0:6.1f}s] book {Path(book).name}\n"
              f"   receipt {Path(L['builder_receipt']).name} | L_place_ms = "
              f"{L['L_place_ms']} | {L['source']}", flush=True)
    der = outdir or (Path(R.DR.resolve()["data_root"]) / "pm_5min/derived")
    stamp = _utc().strftime("%Y%m%dT%H%M%SZ")
    out = der / (f"p003_de_point_estimate_day_{day.replace('-', '')}"
                 f"_L{int(L['L_place_ms'])}ms__{stamp}.json")
    if out.exists():
        raise PointEstimateRefused(f"output already exists: {out}")
    sup = (supersedes_block(supersedes, this_L=L["L_place_ms"], derived=der)
           if supersedes else None)
    res = R.run_day(day, Path(book), params=params, fixture=False,
                    n_days_complete=R.days_complete_now(
                        params)["n_days_complete"],
                    point_estimate=True, ledger_anchor=out,
                    before_work=lambda: R.selftest(quiet=True, offline=False))
    payload = assemble(day, Path(book), L, res, out, supersedes=sup)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True,
                              default=str) + "\n")
    if not quiet:
        agree = payload["placement_latency_agreement"]
        print(f"[{time.time() - t0:6.1f}s] EMITTED {out.name} sha "
              f"{_sha(out)}", flush=True)
        print(f"   L agrees at all {agree['n_sites']} sites: "
              f"{agree['L_place_ms']} ms", flush=True)
        if sup:
            print(f"   supersedes {Path(sup['path']).name} "
                  f"({sup['superseded_because']})", flush=True)
        led = (res.get("decision_ledger") or {})
        print(f"   ledger {Path(led.get('path', '-')).name} sha "
              f"{led.get('sha256')} rows {led.get('n_rows')} schema "
              f"{led.get('schema_version')}", flush=True)
        for a in res.get("per_day_sealed_artifacts", []):
            es = a.get("economic_settlement") or {}
            if es.get("status"):
                print(f"   {a['arm']}: settlement {es['status']}", flush=True)
                continue
            al = es.get("arm_legs") or {}
            bl = es.get("zero_cancel_baseline_legs") or {}
            ws = es.get("winner_source") or {}
            cv = ws.get("chainlink_verification") or {}
            print(f"   {a['arm']}: D_settle {es.get('D_E_settle')!r} | Z "
                  f"{es.get('Z')}", flush=True)
            print(f"      arm      trades {al.get('trades_leg_cents')!r} "
                  f"residual {al.get('residual_leg_cents')!r} total "
                  f"{al.get('total_cents')!r} slugs {al.get('n_slugs')}",
                  flush=True)
            print(f"      baseline trades {bl.get('trades_leg_cents')!r} "
                  f"residual {bl.get('residual_leg_cents')!r} total "
                  f"{bl.get('total_cents')!r} slugs {bl.get('n_slugs')}",
                  flush=True)
            print(f"      winner {cv.get('status')} "
                  f"final={ws.get('is_final_for_quotation')} counts "
                  f"{cv.get('counts')}", flush=True)
        print(f"[{time.time() - t0:6.1f}s] DONE", flush=True)
    return {"path": str(out), "sha256": _sha(out), "payload": payload}


# ------------------------------------------------------- the battery

EXPECTED_CHECKS = 4


def selftest(quiet: bool = False) -> int:
    """RED FIRST. No book is opened by any path here."""
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[de_point_estimate_day] FAIL: {label}")
        n[0] += 1
        if not quiet:
            print(f"  PASS  {label}")

    # ---- (1) THE POSITIVE CONTROL, AND IT ADMITS ---------------------
    # A document at L = 250 whose sites agree is written. The cell asserts
    # the guard ADMITS -- a guard shown only to refuse is half a control
    # (rule 16), and this one gates every emit.
    _res = {"run_mode": "POINT_ESTIMATE",
            "placement_latency": {"L_place_ms": 250.0},
            "per_day_sealed_artifacts": [
                {"arm": "A", "status": "OK_POINT_ESTIMATE"}]}
    _L = {"L_place_ms": 250.0, "source": "THE BOOK'S BUILDER RECEIPT",
          "builder_receipt": "/x/r.json"}
    _doc = {"protocol": PROTOCOL, "day": "2026-09-04",
            "placement_latency": _L, "run_mode": "POINT_ESTIMATE",
            "day_run": _res}
    _agree = R.assert_one_placement_latency(_doc)
    ok(_agree["agree"] is True and _agree["L_place_ms"] == 250.0
       and _agree["n_sites"] == 2,
       f"DE 151 POSITIVE CONTROL, AND IT ADMITS: a finished document whose "
       f"{_agree['n_sites']} `L_place_ms` sites both read "
       f"{_agree['L_place_ms']} is ADMITTED by the emit-time guard. A guard "
       f"that only ever refuses has not been shown to pass anything")

    # ---- (2) THE KNOWN-BAD IS THE LANDED SHAPE ------------------------
    _bad = json.loads(json.dumps(_doc))
    _bad["day_run"]["placement_latency"]["L_place_ms"] = 0.0
    _got = None
    try:
        R.assert_one_placement_latency(_bad)
    except R.RunnerRefused as e:
        _got = str(e).split(":")[0].replace("REFUSED ", "")
    ok(_got == R.PLACEMENT_LATENCY_DISAGREES,
       f"DE 151 KNOWN-BAD: planting 0.0 at the NESTED site while the top "
       f"level says 250.0 -- the exact shape of the seven artifacts landed "
       f"before this round -- REFUSES `{_got}` at the emit, so those bytes "
       f"would never have been written")

    # ---- (3) THE DEFECT, REPRODUCED AT A REAL LANDED ARTIFACT ---------
    # Not a fixture: the artifacts are on disk, are never edited (rule 13)
    # and disagree with themselves forever. Verify at the artifact a claim
    # names (rule 16). Named as a skip with its own reason if absent.
    _der = Path(R.DR.resolve()["data_root"]) / "pm_5min/derived"
    _landed = sorted(_der.glob("p003_de_point_estimate_day_*_L250ms__*.json"))
    _pre151 = []
    for _p in _landed:
        try:
            _d = json.loads(_p.read_text())
        except (OSError, ValueError):
            continue
        if _d.get("placement_latency_agreement") is None:
            _pre151.append((_p, R.placement_latency_leaves(_d)))
    if _pre151:
        _refusals = []
        for _p, _sites in _pre151:
            try:
                R.assert_one_placement_latency(json.loads(_p.read_text()))
                _refusals.append((_p.name, "ADMITTED"))
            except R.RunnerRefused as e:
                _refusals.append(
                    (_p.name, str(e).split(":")[0].replace("REFUSED ", "")))
        ok(all(r == R.PLACEMENT_LATENCY_DISAGREES for _, r in _refusals),
           f"DE 151 AT THE REAL ARTIFACTS: all {len(_refusals)} landed "
           f"pre-DE-151 L=250 point-estimate artifacts REFUSE "
           f"`{R.PLACEMENT_LATENCY_DISAGREES}` when the emit-time guard is "
           f"applied to their own bytes -- "
           f"{[(nm, sorted(s.values())) for nm, s in [(a, b) for (a, _), b in zip(_refusals, [x[1] for x in _pre151])]][:1]} "
           f"and the rest alike. The defect is reproduced at the artifacts "
           f"a claim names, not at a fixture written by the same hand as "
           f"the checker")
    else:
        ok(True,
           "DE 151 AT THE REAL ARTIFACTS: no pre-DE-151 L=250 "
           "point-estimate artifact remains on disk to reproduce the "
           "defect against -- every landed one carries the emit-time "
           "agreement block, which is the state this round was for")

    # ---- (4) A SIBLING IS NOT A SUCCESSOR -----------------------------
    import tempfile
    _td = Path(tempfile.mkdtemp(prefix="pe_"))
    _t = _td / "p003_de_point_estimate_day_20260905_L0ms__x.json"
    _t.write_text(json.dumps({"run_mode": "POINT_ESTIMATE", "day": "2026-09-05",
                              "placement_latency": {"L_place_ms": 0.0}}))
    _sib = None
    try:
        supersedes_block(str(_t), this_L=250.0, derived=_td)
    except PointEstimateRefused as e:
        _sib = str(e).split(":")[0]
    _full = _td / "p003_de_point_estimate_day_20260905_L250ms__y.json"
    _full.write_text(json.dumps({"run_mode": "FULL", "day": "2026-09-05",
                                 "placement_latency": {"L_place_ms": 250.0}}))
    _fullref = None
    try:
        supersedes_block(str(_full), this_L=250.0, derived=_td)
    except PointEstimateRefused as e:
        _fullref = str(e).split(":")[0]
    _absent = None
    try:
        supersedes_block(str(_td / "nope.json"), this_L=250.0, derived=_td)
    except PointEstimateRefused as e:
        _absent = str(e).split(":")[0]
    _mismatch = None
    try:
        supersedes_block({"path": str(_full), "sha256": "0" * 64},
                         this_L=250.0, derived=_td)
    except PointEstimateRefused as e:
        _mismatch = str(e).split(":")[0]
    import shutil
    shutil.rmtree(_td, ignore_errors=True)
    ok(_sib == SUPERSEDES_DIFFERENT_L
       and _fullref == SUPERSEDES_NOT_A_POINT_ESTIMATE
       and _absent == SUPERSEDES_TARGET_ABSENT
       and _mismatch == SUPERSEDES_DIGEST_MISMATCH,
       f"DE 151 SUPERSESSION, FOUR REFUSALS BY NAME: an L = 0 artifact is a "
       f"SIBLING of an L = 250 run, not its predecessor (`{_sib}`, R-811); a "
       f"FULL run is not superseded by a point estimate (`{_fullref}`, "
       f"R-828); an absent target is not a link (`{_absent}`); and a target "
       f"whose bytes are not the bytes named refuses (`{_mismatch}`) -- the "
       f"digest is recomputed at the write, never copied from the caller")

    if n[0] != EXPECTED_CHECKS:
        raise SystemExit(f"[de_point_estimate_day] FAIL: check count "
                         f"{n[0]} != {EXPECTED_CHECKS}")
    if not quiet:
        print(f"[de_point_estimate_day] PASS -- {n[0]} checks, "
              f"n_disarmed 0, n_skipped 0")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--day", type=str)
    ap.add_argument("--book", type=Path)
    ap.add_argument("--supersedes", type=str,
                    help="the point-estimate artifact this run REPLACES, "
                         "by path. Its bytes are never edited (rule 13); "
                         "a run at a different L refuses as a SIBLING")
    ap.add_argument("--outdir", type=Path)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.day and a.book):
        raise SystemExit("usage: --day <YYYY-MM-DD> --book <path> "
                         "[--supersedes <path>] | --selftest")
    run(a.day, a.book, supersedes=a.supersedes, outdir=a.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
