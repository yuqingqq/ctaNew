#!/usr/bin/env python3
"""BE's INERT-ARM trajectory production (R-272, unblocked by R-277).

Produces QR_SKEW_ONLY and QR_CANCEL_HOLD_X_SKEW trajectories from REAL v3.4
exposure rows with the predictor INERT, submits them through DA's real loader
and lifecycle battery, and checks the declared parity anchors.

WHY BE IMPLEMENTS ITS OWN ARM. DA ships run_stub_arm and calling it would have
been three lines. But a parity battery that checks BE's trajectory against DA's
own producer compares DA to DA, and the two-implementation contract collapses to
one implementation. The lifecycle below is written from the DECLARED spec --
  * a cancel REQUESTED is not a cancel EFFECTIVE; it binds only after
    CANCEL_EFFECTIVE_LAG_S and only if the limiter passed it;
  * a cancel SUPPRESSED by the limiter changes NOTHING: the order stays exposed;
  * a withheld quote is a STATUS (PLACE_WITHHELD), never an absence, or an arm
    that stopped emitting would be indistinguishable from one that ran out of
    opportunities
-- and AGREEMENT with run_stub_arm is a RESULT this module measures, never an
assumption it makes. Same discipline as be_trajectory_export declaring its own
field list and importing DA's only inside the agreement check.

NO ECONOMICS. Output is lifecycle only: no value, no markout, no net, no
p-value. An inert reference carrying economics would invite exactly the
comparison it is not entitled to make, and `assert_no_economics` enforces that
on the emitted object rather than trusting this paragraph.

PREDICTOR INERT. predictor="none", predictor_active=False, always. The arms
differ in their RULES (skew; cancel_hold x skew), never in an estimate.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

import be_trajectory_export as BE                     # noqa: E402
import harmful_rows_loader as RL                      # noqa: E402

SELFTEST_FLAG = "--selftest"
DEFAULT_TAPE = Path("/home/yuqing/ctaNew/data/pm_5min/derived/"
                    "harmful_exposure_rows_v3_topup.json")
OUT_DIR = Path("/home/yuqing/ctaNew/data/pm_5min/derived")

# Declared by DA and transcribed here INDEPENDENTLY, like BE_EVENT_FIELDS. The
# agreement check compares the two rather than importing one as truth.
BE_CANCEL_EFFECTIVE_LAG_S = 0.050
BE_RATE_LIMIT_WINDOW_S = 1.0
BE_INERT_ARMS = {
    "QR_SKEW_ONLY":          {"components": ("skew",), "interaction": False},
    "QR_CANCEL_HOLD_X_SKEW": {"components": ("cancel_hold", "skew"),
                              "interaction": True},
}
# Keys that must never appear in an emitted event. Economics are the point of
# the ban, but a "score" or "p_fill" leaking through would be worse.
BANNED_OUTPUT_KEYS = ("value", "cents", "markout", "net", "pnl", "score",
                      "p_fill", "ev", "harm", "sacrifice", "economics")


def _sha16_streamed(path: Path, chunk: int = 1 << 22) -> str:
    """Hash the tape WITHOUT loading it. read_bytes() on a 705MB tape would
    pull the whole file into memory just to name it, inside a run whose
    standing constraint is to use less memory, not more."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()[:16]


class InertRunRefused(RuntimeError):
    """Refuse at the producer, before an artifact exists to be believed."""


# ---------------------------------------------------------------------------
# real opportunities, from the rows' OWN fields
# ---------------------------------------------------------------------------
def stream_rows(path: Path, chunk: int = 1 << 24):
    """OK rows with the fields an OPPORTUNITY needs.

    RL.stream_ok_rows exists but projects through compact_row, which keeps
    (slug, day, t0, t_start, side, gen, latency) and DROPS `resting` and
    `level` -- the qty and price an opportunity is made of. So the header and
    decoder helpers are reused and the projection is BE's own. Importing binds
    no identity; only editing would."""
    meta, start = RL._header(path)
    if meta.get("schema") != RL.EXPECTED_SCHEMA:
        raise InertRunRefused(
            f"REFUSED: {path.name} declares schema {meta.get('schema')!r}, not "
            f"{RL.EXPECTED_SCHEMA!r}. A differently-scoped dataset is not this "
            f"dataset.")
    buf = ""
    with path.open("r", encoding="utf-8") as fh:
        fh.read(start)
        while True:
            piece = fh.read(chunk)
            if piece:
                buf += piece
            i = RL._skip_ws(buf, 0)
            while i < len(buf):
                if buf[i] == ",":
                    i = RL._skip_ws(buf, i + 1); continue
                if buf[i] == "]":
                    return
                try:
                    obj, end = RL._DEC.raw_decode(buf, i)
                except ValueError:
                    break
                if obj.get("status") == "OK":
                    yield obj
                i = RL._skip_ws(buf, end)
            buf = buf[i:]
            if not piece:
                # T2: EOF WITHOUT THE CLOSING BRACKET. The previous form simply
                # returned here, so a file truncated mid-array -- a killed
                # writer, a full disk, an interrupted copy -- yielded a SHORT
                # POPULATION AND NO ERROR. Every count downstream would then
                # describe a population nobody chose, and it would look entirely
                # normal. A partial input is not a small input.
                raise RuntimeError(
                    f"REFUSED: {path.name} ended without the closing ']' of its "
                    f"rows array. The file is TRUNCATED, and a truncated array "
                    f"read as complete silently shrinks the population.")


def opportunities(rows, limit: int | None = None) -> tuple[list, dict]:
    """One opportunity per ACTION -- distinct (slug, side, gen).

    Rule 2: rows are actions. The exposure tape carries ~1.99 rows per action,
    so an opportunity per ROW would place the same order up to 23 times and
    every downstream count would inherit the inflation. The EARLIEST row of a
    generation supplies the opportunity, because a quote is placed once, when
    the generation opens -- not re-placed at each later decision row."""
    best: dict = {}
    seen_rows = 0
    for r in rows:
        seen_rows += 1
        for f in ("slug", "side", "gen", "t_start", "resting", "level"):
            if r.get(f) is None:
                raise InertRunRefused(
                    f"REFUSED: row {seen_rows} carries no {f!r}. An opportunity "
                    f"built from a missing field is a fabricated order.")
        k = (r["slug"], r["side"], int(r["gen"]))
        t = float(r["t_start"])
        if k not in best or t < best[k]["t"]:
            best[k] = {"slug": k[0], "side": k[1], "gen": k[2], "t": t,
                       "qty": float(r["resting"]), "price": float(r["level"])}
    opps = sorted(best.values(),
                  key=lambda o: (o["t"], o["slug"], o["side"], o["gen"]))
    if limit is not None:
        opps = opps[:limit]
    return opps, {"rows_seen": seen_rows, "n_actions": len(best),
                  "n_opportunities": len(opps),
                  "rows_per_action": (round(seen_rows / len(best), 4)
                                      if best else None)}


# ---------------------------------------------------------------------------
# BE's OWN arm producer
# ---------------------------------------------------------------------------
def be_run_arm(arm: str, opps: list, *, cancel_enabled: bool = False,
               forced_cancel_keys=None, rate_limit_per_window: int | None = None,
               hold_after_first_cancel: bool = False) -> list:
    """BE's independent lifecycle. Returns contract-shaped events.

    With cancel_enabled False -- the INERT case, which is every arm in this
    run -- no decision is ever taken and the arm places exactly its
    opportunities. That is not a shortcut: it is the property the whole inert
    reference exists to establish, which is why the cancel machinery below is
    present and exercised by the selftest rather than omitted."""
    if arm not in BE_INERT_ARMS:
        raise InertRunRefused(
            f"REFUSED: {arm!r} is not an arm BE produces here; declared: "
            f"{sorted(BE_INERT_ARMS)}")
    forced = None if forced_cancel_keys is None else set(forced_cancel_keys)
    if forced is not None and not cancel_enabled:
        raise InertRunRefused(
            "REFUSED: forced cancels supplied with cancel DISABLED. Silently "
            "ignoring them would make an inert run look like it considered "
            "them.")
    events: list = []
    seq = 0

    def emit(t, kind, o, qty=0.0, price=None, note=""):
        nonlocal seq
        events.append(BE.make_event(t, seq, kind, o["slug"], o["side"],
                                    o["gen"], qty, 0.0 if price is None else price,
                                    note))
        seq += 1

    requested: set = set()
    per_window: dict = {}
    holding = False
    for o in opps:
        key = (o["slug"], o["side"], o["gen"])
        if holding:
            emit(o["t"], "PLACE_WITHHELD", o, o["qty"], o["price"],
                 "withheld after first cancel (permanent hold)")
            continue
        emit(o["t"], "PLACE", o, o["qty"], o["price"])
        if not cancel_enabled:
            continue
        if not (forced is not None and key in forced):
            continue
        if key in requested:
            raise InertRunRefused(
                f"REFUSED: generation {key} cancelled twice. One generation may "
                f"be cancelled at most once.")
        requested.add(key)
        emit(o["t"], "CANCEL_REQUESTED", o)
        w = int(o["t"] // BE_RATE_LIMIT_WINDOW_S)
        used = per_window.get(w, 0)
        if rate_limit_per_window is None or used < rate_limit_per_window:
            per_window[w] = used + 1
            emit(o["t"] + BE_CANCEL_EFFECTIVE_LAG_S, "CANCEL_EFFECTIVE", o)
        else:
            emit(o["t"], "CANCEL_SUPPRESSED", o, note=
                 f"rate limit {rate_limit_per_window}/{BE_RATE_LIMIT_WINDOW_S}s "
                 f"reached in window {w}; the order STAYS EXPOSED")
        if hold_after_first_cancel:
            holding = True
    return events


def assert_no_economics(obj: dict) -> None:
    """The emitted object carries LIFECYCLE ONLY. Checked, not promised."""
    # TOKENISED, not substring-matched. The first form scanned the JSON blob
    # for '"ev' and fired on "events" -- the banned-token list flagged the
    # trajectory's own event array. A substring ban is the silent-regex class
    # (rule 15) pointed at ourselves: it would have refused every real artifact
    # while looking strict. Keys and string values are split into words and
    # compared as whole tokens, so "events" is clean and "net_cents" is not.
    import re as _re

    def _tokens(text):
        return set(_re.split(r"[^a-z0-9]+", str(text).lower())) - {""}

    hits: set = set()

    def _walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                hits.update(_tokens(k) & set(BANNED_OUTPUT_KEYS))
                _walk(v)
        elif isinstance(o, list):
            for v in o:
                _walk(v)
        elif isinstance(o, str):
            hits.update(_tokens(o) & set(BANNED_OUTPUT_KEYS))
    _walk(obj)
    hits = sorted(hits)
    if hits:
        raise InertRunRefused(
            f"REFUSED: the trajectory carries economics-shaped fields {hits}. "
            f"An inert reference is entitled to lifecycle and nothing else; a "
            f"value in this artifact would be read as one it earned.")


def produce(arm: str, opps: list, **kw) -> dict:
    spec = BE_INERT_ARMS[arm]
    obj = BE.export_trajectory(
        arm, be_run_arm(arm, opps, **kw), predictor="none",
        predictor_active=False, components=spec["components"],
        interaction=spec["interaction"], fairprice_estimator=None)
    assert_no_economics(obj)
    return obj


def canonical(events: list) -> str:
    """Byte form parity is defined over. Excludes nothing BE controls: the
    events ARE the behaviour."""
    return hashlib.sha256(json.dumps(events, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# submission and parity anchors
# ---------------------------------------------------------------------------
def submit(objs: dict) -> dict:
    """Through DA's REAL loader and lifecycle battery.

    DA is imported HERE and nowhere above: the producer must not be able to
    borrow the checker's notion of a trajectory. A refusal on a real row is a
    FINDING and is returned as one, never swallowed."""
    import da_replay_parity_battery as DA
    out = {}
    for arm, obj in sorted(objs.items()):
        try:
            tr = DA.load_external_trajectory(obj)
            out[arm] = {"loaded": True, "n_events": len(tr.events),
                        "predictor_active": tr.predictor_active,
                        "lifecycle": DA.external_lifecycle(tr)}
        except Exception as e:                          # noqa: BLE001
            out[arm] = {"loaded": False,
                        "refusal": f"{type(e).__name__}: {e}",
                        "note": "a contract refusal on a REAL row is a FINDING"}
    return out


def agreement_with_da_stub(opps: list) -> dict:
    """Does BE's independent producer agree with DA's? A RESULT, not an input.

    Imported inside, after BE's own events exist, so nothing above can consult
    it. Disagreement is reported with the first differing event rather than
    reduced to a boolean -- 'they differ' is not a finding, 'they differ HERE'
    is."""
    import da_replay_parity_battery as DA
    mine = be_run_arm("QR_SKEW_ONLY", opps)
    theirs = [{"t": e.t, "seq": e.seq, "kind": e.kind, "slug": e.slug,
               "side": e.side, "gen": e.gen, "qty": e.qty,
               "price": 0.0 if e.price is None else e.price, "note": e.note}
              for e in DA.run_stub_arm("QR_SKEW_ONLY", opps,
                                       predictor_enabled=False).events]
    first = None
    for i in range(min(len(mine), len(theirs))):
        if mine[i] != theirs[i]:
            first = {"index": i, "be": mine[i], "da": theirs[i]}
            break
    return {"agree": mine == theirs, "n_be": len(mine), "n_da": len(theirs),
            "first_difference": first,
            "meaning": "independent implementations of the declared lifecycle "
                       "producing identical events; measured, not assumed"}


def parity_anchors(opps: list) -> dict:
    """The two declared anchors, computed as predicates (rule 10)."""
    skew = be_run_arm("QR_SKEW_ONLY", opps)
    # ANCHOR 1: the cancel-capable arm with cancel DISABLED must be
    # bit-identical to the skew-only arm. If it is not, the cancel machinery is
    # doing something even when it is switched off, and every later cancel
    # measurement would be contaminated by that residue.
    held = be_run_arm("QR_CANCEL_HOLD_X_SKEW", opps, cancel_enabled=False)
    return {
        "cancel_disabled_equals_skew_only": {
            "bit_identical": skew == held,
            "sha_skew_only": canonical(skew), "sha_cancel_held": canonical(held),
            "why": "with the predictor inert the cancel arm must place exactly "
                   "what the skew arm places; a difference is residue from "
                   "machinery that is supposed to be off"},
    }


def hashseed_anchor(arm: str, tape: Path, limit: int) -> dict:
    """Bit-identity across CROSSED PYTHONHASHSEED, in real subprocesses.

    Set/dict iteration order is the classic way a 'deterministic' pipeline
    stops being one, and it only shows under a different seed. Two seeds are
    run because one proves nothing about the other."""
    import subprocess
    shas = {}
    for seed in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--emit-canon",
             arm, str(tape), str(limit)],
            capture_output=True, text=True, env=env, timeout=1800)
        if r.returncode != 0:
            return {"identical": False,
                    "refusal": f"seed {seed} exited {r.returncode}: "
                               f"{r.stderr.strip()[-300:]}"}
        shas[seed] = r.stdout.strip()
    return {"identical": shas.get("0") == shas.get("1") and bool(shas.get("0")),
            "sha_by_seed": shas,
            "why": "a canonical trajectory that moves with PYTHONHASHSEED was "
                   "ordered by a set or a dict somewhere"}


# ---------------------------------------------------------------------------
def selftest() -> int:
    fails: list = []

    def ok(c, label):
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    def O(n=6):
        return [{"slug": "btc-updown-5m-1787650200",
                 "side": "BUY_UP" if i % 2 == 0 else "SELL_UP",
                 "gen": i, "t": 10.0 * i, "qty": 5.0, "price": 0.5}
                for i in range(n)]

    ev = be_run_arm("QR_SKEW_ONLY", O())
    ok([e["kind"] for e in ev] == ["PLACE"] * 6,
       "POSITIVE CONTROL: an INERT arm places its opportunities and does "
       "nothing else")
    ok([e["seq"] for e in ev] == list(range(6)),
       "seq is dense and emission-ordered, so (t, seq) totally orders the tape")

    ok(be_run_arm("QR_SKEW_ONLY", O()) ==
       be_run_arm("QR_CANCEL_HOLD_X_SKEW", O(), cancel_enabled=False),
       "ANCHOR: the cancel arm with cancel DISABLED is bit-identical to the "
       "skew-only arm — switched-off machinery leaves no residue")

    forced = [("btc-updown-5m-1787650200", "BUY_UP", 0)]
    ce = be_run_arm("QR_CANCEL_HOLD_X_SKEW", O(), cancel_enabled=True,
                    forced_cancel_keys=forced)
    kinds = [e["kind"] for e in ce]
    ok("CANCEL_REQUESTED" in kinds and "CANCEL_EFFECTIVE" in kinds,
       "KNOWN-GOOD: an ENABLED cancel produces a request AND an effect — the "
       "machinery is real, not decorative (a switched-off anchor that could "
       "never fire proves nothing)")
    req = next(e for e in ce if e["kind"] == "CANCEL_REQUESTED")
    eff = next(e for e in ce if e["kind"] == "CANCEL_EFFECTIVE")
    ok(abs((eff["t"] - req["t"]) - BE_CANCEL_EFFECTIVE_LAG_S) < 1e-12,
       f"a cancel REQUESTED is not a cancel EFFECTIVE: it binds exactly "
       f"{BE_CANCEL_EFFECTIVE_LAG_S}s later")

    sup = be_run_arm("QR_CANCEL_HOLD_X_SKEW", O(), cancel_enabled=True,
                     forced_cancel_keys=[("btc-updown-5m-1787650200", "BUY_UP", 0)],
                     rate_limit_per_window=0)
    sk = [e["kind"] for e in sup]
    ok("CANCEL_SUPPRESSED" in sk and "CANCEL_EFFECTIVE" not in sk,
       "a cancel SUPPRESSED by the limiter produces NO effect: the order stays "
       "exposed, which is the inflation this arm exists to make visible")

    hold = be_run_arm("QR_CANCEL_HOLD_X_SKEW", O(), cancel_enabled=True,
                      forced_cancel_keys=forced, hold_after_first_cancel=True)
    ok("PLACE_WITHHELD" in [e["kind"] for e in hold],
       "a withheld quote is a STATUS, never an absence — an arm that simply "
       "stopped emitting is indistinguishable from one out of opportunities")

    for label, fn, want in (
        ("forced cancels with cancel DISABLED",
         lambda: be_run_arm("QR_SKEW_ONLY", O(), forced_cancel_keys=forced),
         "cancel DISABLED"),
        ("an UNDECLARED arm",
         lambda: be_run_arm("NOT_AN_ARM", O()), "is not an arm"),
        ("cancelling one generation TWICE",
         lambda: be_run_arm("QR_CANCEL_HOLD_X_SKEW", O(2) + O(2),
                            cancel_enabled=True, forced_cancel_keys=forced),
         "cancelled twice"),
    ):
        try:
            fn(); got = ""
        except InertRunRefused as e:
            got = str(e)
        ok(want in got, f"KNOWN-BAD: {label} is REFUSED (got {got[:56]!r})")

    try:
        assert_no_economics({"events": [{"note": "net_cents 5"}]}); g = ""
    except InertRunRefused as e:
        g = str(e)
    ok("economics-shaped" in g,
       "KNOWN-BAD: economics leaking into the artifact is REFUSED at the "
       "producer, checked on the object rather than promised in a docstring")
    assert_no_economics(produce("QR_SKEW_ONLY", O()))
    ok(True, "POSITIVE CONTROL: a real inert trajectory PASSES the economics "
             "ban (a ban that refused everything would be useless)")

    try:
        opportunities([{"slug": "s", "side": "BUY_UP", "gen": 1,
                        "t_start": 0.0, "resting": 5.0}]); g2 = ""
    except InertRunRefused as e:
        g2 = str(e)
    ok("carries no" in g2,
       "KNOWN-BAD: a row missing a field is REFUSED — an opportunity built "
       "from a missing field is a fabricated order")

    # ---- T2: a TRUNCATED array must REFUSE, not read as complete --------
    import tempfile as _tf2
    _d2 = Path(_tf2.mkdtemp())
    _rw = [{"slug": "s", "side": "BUY_UP", "gen": i, "t_start": float(i),
            "resting": 5.0, "level": 0.5, "status": "OK", "t0": 1787897400,
            "day": "2026-08-28",
            "latency": {"50": {"preventable_value_cents": 1.0,
                               "preventable_shares": 1.0, "stale_shares": 0.0}}}
           for i in range(6)]
    _fp = _d2 / "full.json"
    _fp.write_text(json.dumps({"schema": RL.EXPECTED_SCHEMA, "rows": _rw}))
    _nfull = sum(1 for _ in stream_rows(_fp))
    ok(_nfull == 6,
       f"POSITIVE CONTROL: a COMPLETE array streams every row ({_nfull}/6) — "
       f"a reader that refused everything would pass the known-bad below")
    _txt = _fp.read_text()
    _tp = _d2 / "trunc.json"
    _tp.write_text(_txt[:_txt.rindex('{"slug"')])      # cut mid-array, no ']'
    try:
        _ntr = sum(1 for _ in stream_rows(_tp)); _terr = ""
    except RuntimeError as e:
        _ntr, _terr = -1, str(e)
    ok("TRUNCATED" in _terr,
       f"T2 KNOWN-BAD: a TRUNCATED rows array REFUSES rather than returning a "
       f"short population; before the fix it yielded {5} of {6} rows and no "
       f"error, and every downstream count would have described a population "
       f"nobody chose (got {_terr[:60]!r})")
    import shutil as _sh2; _sh2.rmtree(_d2, ignore_errors=True)

    dup = [{"slug": "s", "side": "BUY_UP", "gen": 1, "t_start": 9.0,
            "resting": 5.0, "level": 0.5},
           {"slug": "s", "side": "BUY_UP", "gen": 1, "t_start": 3.0,
            "resting": 5.0, "level": 0.5}]
    o2, st = opportunities(dup)
    ok(len(o2) == 1 and o2[0]["t"] == 3.0 and st["rows_per_action"] == 2.0,
       "rule 2: three rows of one generation are ONE action, taken at its "
       "EARLIEST row — a quote is placed once, when the generation opens")

    ag = agreement_with_da_stub(O())
    ok(ag["agree"],
       f"RESULT (not assumption): BE's independent producer agrees event-for-"
       f"event with DA's — {ag['n_be']} vs {ag['n_da']} events"
       + ("" if ag["agree"] else f"; first diff {ag['first_difference']}"))

    sub = submit({"QR_SKEW_ONLY": produce("QR_SKEW_ONLY", O())})
    ok(sub["QR_SKEW_ONLY"]["loaded"] and
       sub["QR_SKEW_ONLY"]["lifecycle"]["identity_holds"],
       f"DA's REAL loader accepts BE's trajectory and its lifecycle holds "
       f"({sub['QR_SKEW_ONLY'].get('refusal', 'no refusal')})")

    print(f"\n{'BE INERT-ARM SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if SELFTEST_FLAG in argv:
        return selftest()
    if argv and argv[0] == "--emit-canon":
        arm, tape, limit = argv[1], Path(argv[2]), int(argv[3])
        opps, _ = opportunities(stream_rows(tape), limit=limit or None)
        print(canonical(be_run_arm(arm, opps)))
        return 0
    tape = Path(argv[0]) if argv else DEFAULT_TAPE
    limit = int(argv[1]) if len(argv) > 1 else 0
    print(f"[be-inert] tape {tape.name} limit {limit or 'ALL'}", flush=True)
    opps, stats = opportunities(stream_rows(tape), limit=limit or None)
    print(f"[be-inert] {stats}", flush=True)
    objs = {a: produce(a, opps) for a in sorted(BE_INERT_ARMS)}
    res = {
        "produced_by": "be_inert_arm_run.py (BE's own producer; DA's arm code "
                       "is never imported into production)",
        "tape": {"path": str(tape), "name": tape.name,
                 "bytes": tape.stat().st_size,
                 "sha256_prefix": _sha16_streamed(tape)},
        "populations": stats,
        "predictor": {"predictor": "none", "predictor_active": False,
                      "why": "inert by construction; the arms differ in RULES"},
        "arms": {a: {"n_events": len(o["events"]),
                     "kinds": sorted({e["kind"] for e in o["events"]}),
                     "components": o["components"],
                     "interaction": o["interaction"]}
                 for a, o in sorted(objs.items())},
        "submission": submit(objs),
        "parity": parity_anchors(opps),
        "agreement_with_da_stub": agreement_with_da_stub(opps),
        "economics_in_output": False,
    }
    res["parity"]["crossed_hashseed"] = hashseed_anchor(
        "QR_SKEW_ONLY", tape, limit)
    for a, o in sorted(objs.items()):
        p = OUT_DIR / f"be_inert_trajectory_{a.lower()}.json"
        if p.exists():
            snap = p.with_suffix(".json.prev")
            snap.write_bytes(p.read_bytes())
            print(f"[be-inert] snapshotted {p.name} -> {snap.name}", flush=True)
        p.write_text(json.dumps(o, sort_keys=True, separators=(",", ":")))
        print(f"[be-inert] wrote {p}", flush=True)
    rp = OUT_DIR / "be_inert_arm_receipt.json"
    if rp.exists():
        rp.with_suffix(".json.prev").write_bytes(rp.read_bytes())
    rp.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"[be-inert] receipt {rp}", flush=True)
    print(json.dumps({k: res[k] for k in ("arms", "parity", "populations",
                                          "agreement_with_da_stub")},
                     indent=1, sort_keys=True)[:1400], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
