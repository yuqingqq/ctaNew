"""Exposure dataset v2 — REBUILT per the user's eight-issue review (2026-08-25).

SURFACE AUTHORISATION (R-126, in-file): HARMFUL_FILL_HAZARD_TOXICITY_PLAN §10
item 2, rebuilt under the user's review. v1's receipts remain on disk as
provenance; every v1 defect is named here beside its fix so it cannot recur
silently:

  #1 FILL CLOCK. v1 logged fills at the last RESYNC time (p99 error 22-125 ms,
     max ~162 ms) — invalidating every 5-250 ms latency label. v2 takes fill
     times from the ENGINE'S OWN `arm.fills` (exact trade receipt), and the
     selftest asserts recorded fills reconcile with the engine EXACTLY.
  #2 SNAPSHOT BEFORE RESYNC. v1 stamped decisions at the moment a state was
     about to END (~76% of "no-fill" rows were replaced inside the horizon).
     v2 emits GENERATION INTERVALS: the state observed in `dead(when)` is the
     post-resync state of the PREVIOUS event, live over [t_start, when). The
     decision time is t_start — when the exposure BEGAN.
  #4 DUPLICATE FILLS. v1 emitted many rows per fill (1.99 rows/unique fill,
     max 23). v2 emits ONE ROW PER (generation, side): rows ARE actions; a
     generation is cancellable once.
  #5 TRANCHE VALUATION. v1 gave all shares the first tranche's markout. v2
     values EVERY tranche at its own receipt time and level.
  #3 LATENCY. v2 publishes PER-LATENCY PREVENTABLE VALUE: at latency L only
     tranches with fill_t >= t_start + L are preventable; earlier tranches are
     STALE and their value is never claimable.

Rows come from the unchanged QR_SKEW_ONLY no-cancel reference trajectory (§3).
Exclusions remain statuses, never drops.

    python3 live/pm_research/harmful_exposure_rows.py --selftest
    python3 live/pm_research/harmful_exposure_rows.py run [--per-coin N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import policy_optimizer_queue_realistic as qr
import inventory_walk as iw

OUT = qr.base.fi.PM / "derived/harmful_exposure_rows_v2.json"
LATENCY_GRID_MS = (5, 10, 20, 30, 50, 75, 100, 150, 250)
MARKOUT_S = 5.0
SIDES = ("BUY_UP", "SELL_UP")
TRAIN_DAYS = ("2026-08-20", "2026-08-21", "2026-08-22")


class RecordingArm(qr.QueueRealisticArm):
    """Behaviour-identical QR_SKEW_ONLY arm recording generation INTERVALS.

    `dead(when)` is the loop's clock hook. The state it sees is the
    post-resync state of the PREVIOUS event, still live at `when` — so each
    change closes the previous interval at `when` and opens a new one there.
    Fill TIMES are NOT recorded here (v1's #1): they come from `arm.fills`.
    """

    _instances: list["RecordingArm"] = []

    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        self.intervals: list[dict[str, Any]] = []
        self._open: dict[str, dict[str, Any] | None] = {s: None for s in SIDES}
        RecordingArm._instances.append(self)

    def _state_key(self, ms: str):
        side = self.side(ms)
        return (self.generation[ms], side.level,
                round(side.resting, 9), round(getattr(side, "qahead", 0.0), 9))

    def dead(self, when: float) -> bool:
        for ms in SIDES:
            side = self.side(ms)
            cur = self._open[ms]
            key = self._state_key(ms)
            if cur is not None and cur["_key"] == key:
                cur["t_end"] = when          # unchanged: extend
                continue
            if cur is not None:
                cur["t_end"] = when          # close at the change
                self.intervals.append(cur)
            self._open[ms] = {
                "_key": key, "side": ms, "gen": self.generation[ms],
                "level": side.level, "resting": side.resting,
                "qahead": getattr(side, "qahead", 0.0),
                "net": (self.net() if callable(self.net) else self.net),
                "t_start": when, "t_end": when,
            }
        return super().dead(when)

    def finalize(self, t: float) -> None:
        for ms in SIDES:
            cur = self._open[ms]
            if cur is not None:
                cur["t_end"] = t
                self.intervals.append(cur)
                self._open[ms] = None


def label_interval(iv: dict[str, Any], fills: Sequence[Any], wf: Any,
                   window_s: float) -> dict[str, Any]:
    """One row per generation interval. Fills from the ENGINE's list (#1),
    each tranche valued at ITS OWN time and level (#5), preventable value per
    latency (#3). The horizon is the interval itself (#2): a generation is
    exposed exactly while it lives."""
    row = {k: iv[k] for k in ("side", "gen", "level", "resting", "qahead",
                              "net", "t_start", "t_end")}
    row["exposure_s"] = iv["t_end"] - iv["t_start"]
    row["status"] = "OK"
    if iv["t_end"] + MARKOUT_S > window_s:
        row["status"] = "TRUNCATED_HORIZON"
        return row
    if wf.touched(iv["t_start"], iv["t_end"] + MARKOUT_S):
        row["status"] = "GAP_IN_HORIZON"
        return row
    sgn = 1.0 if iv["side"] == "BUY_UP" else -1.0
    tranches = []
    for f in fills:
        if f.maker_side != iv["side"]:
            continue
        if not (iv["t_start"] <= f.t < iv["t_end"] + 1e-9):
            continue
        later = wf.mid_at(f.t + MARKOUT_S)
        if later is None:
            row["status"] = "NO_FUTURE_MID"
            return row
        mk = sgn * (later - f.level) * 100.0
        tranches.append({"t": f.t, "shares": f.size,
                         "markout_cents_per_share": mk,
                         "v_cancel_cents": -mk * f.size})
    row["any_fill"] = bool(tranches)
    row["n_tranches"] = len(tranches)
    row["fill_shares"] = sum(t["shares"] for t in tranches)
    row["v_cancel_cents_total"] = sum(t["v_cancel_cents"] for t in tranches)
    lat = {}
    for L in LATENCY_GRID_MS:
        cut = iv["t_start"] + L / 1000.0
        prev = [t for t in tranches if t["t"] >= cut]
        stale = [t for t in tranches if t["t"] < cut]
        lat[str(L)] = {
            "preventable_value_cents": sum(t["v_cancel_cents"] for t in prev),
            "preventable_shares": sum(t["shares"] for t in prev),
            "stale_shares": sum(t["shares"] for t in stale),
        }
    row["latency"] = lat
    row["tranches"] = tranches
    return row


def select_stratified(per_coin_per_day: int,
                      days: Sequence[str] = TRAIN_DAYS,
                      coins: Sequence[str] = ("btc", "eth")) -> list:
    import collections, datetime as _dt
    fi = qr.base.fi
    paths = fi._archive_paths(); tokens = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    picked: collections.Counter = collections.Counter()
    out = []
    for slug in sorted(fi.covered_slugs(fi.ERA)):
        coin = slug.split("-")[0]
        if coin not in coins or slug not in paths or slug not in tokens:
            continue
        try:
            t0 = int(slug.rsplit("-", 1)[1])
        except ValueError:
            continue
        day = _dt.datetime.fromtimestamp(t0, _dt.timezone.utc).strftime("%Y-%m-%d")
        if day not in days or picked[(coin, day)] >= per_coin_per_day:
            continue
        up, down = tokens[slug]
        out.append((slug, paths[slug], up, down, gaps.get(slug, [])))
        picked[(coin, day)] += 1
    return out


def replay_with_recorder(path, up, dn, gaps, spec):
    orig = qr.QueueRealisticArm
    qr.QueueRealisticArm = RecordingArm
    RecordingArm._instances.clear()
    try:
        cells = qr.replay_cells_queue_realistic(path, up, dn, gaps, [spec],
                                                signals={})
    finally:
        qr.QueueRealisticArm = orig
    if not cells:
        return None
    wf = cells.get(qr.QR_SKEW)
    if wf is None or len(RecordingArm._instances) != 1:
        return None
    arm = RecordingArm._instances[0]
    arm.finalize(qr.base.fi.WINDOW_S)
    return arm, wf


def reconcile_fills(rows: Sequence[dict], arm) -> dict[str, float]:
    """#1's mandated test: labelled tranches must reconcile with the ENGINE."""
    lab = sum(r.get("fill_shares", 0.0) for r in rows if r["status"] == "OK")
    eng_ok = 0.0
    ok_iv = [(r["side"], r["t_start"], r["t_end"]) for r in rows
             if r["status"] == "OK"]
    for f in arm.fills:
        if any(s == f.maker_side and a <= f.t < b + 1e-9 for s, a, b in ok_iv):
            eng_ok += f.size
    return {"labelled_shares": lab, "engine_shares_in_ok_intervals": eng_ok,
            "match": abs(lab - eng_ok) < 1e-6}


def build_rows(per_coin: int | None = None,
               coins: Sequence[str] = ("btc", "eth")) -> dict[str, Any]:
    import datetime as _dt
    spec = qr._qr_spec(qr.QR_SKEW, latency_ms=0, cancel=False)
    selected = select_stratified(per_coin or 10, coins=coins)
    rows: list[dict[str, Any]] = []
    recon_fail = 0
    n_windows = 0
    days: set[str] = set()
    for ent in selected:
        slug = ent[0]
        out = replay_with_recorder(ent[1], ent[2], ent[3], ent[4], spec)
        if out is None:
            continue
        arm, wf = out
        n_windows += 1
        t0 = int(slug.rsplit("-", 1)[1])
        day = _dt.datetime.fromtimestamp(t0, _dt.timezone.utc).strftime("%Y-%m-%d")
        days.add(day)
        wrows = []
        for iv in arm.intervals:
            if iv["level"] is None:
                continue
            r = label_interval(iv, arm.fills, wf, qr.base.fi.WINDOW_S)
            r["slug"] = slug; r["coin"] = slug.split("-")[0]
            r["day"] = day; r["t0"] = t0
            wrows.append(r)
        rec = reconcile_fills(wrows, arm)
        if not rec["match"]:
            recon_fail += 1
            for r in wrows:
                r["status"] = "RECONCILIATION_FAILED"
        rows.extend(wrows)
    return {"rows": rows, "n_windows": n_windows, "days": sorted(days),
            "reconciliation_failures": recon_fail,
            "schema": "harmful_exposure_v2_intervals"}


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    class F:                                   # engine-fill stand-in
        def __init__(self, t, side, level, size):
            self.t, self.maker_side, self.level, self.size = t, side, level, size

    class WF:
        def mid_at(self, t): return 0.60
        def touched(self, a, b): return False

    iv = {"side": "BUY_UP", "gen": 3, "level": 0.55, "resting": 5.0,
          "qahead": 0.0, "net": 0.0, "t_start": 10.0, "t_end": 10.5}
    # two tranches at DIFFERENT times/levels: each valued on its own (#5)
    fills = [F(10.1, "BUY_UP", 0.55, 2.0), F(10.4, "BUY_UP", 0.55, 3.0),
             F(11.0, "BUY_UP", 0.55, 5.0),          # after t_end: NOT ours
             F(10.2, "SELL_UP", 0.60, 5.0)]          # other side: not ours
    r = label_interval(iv, fills, WF(), 300.0)
    ok(r["n_tranches"] == 2 and abs(r["fill_shares"] - 5.0) < 1e-9,
       "only same-side tranches INSIDE the interval are attributed (#2/#4)")
    ok(abs(r["v_cancel_cents_total"] + 25.0) < 1e-9,
       "each tranche valued at its own time and level (#5)")
    # per-latency preventable value (#3): at L=150ms the 10.1s tranche is STALE
    ok(abs(r["latency"]["150"]["preventable_value_cents"] + 15.0) < 1e-9
       and abs(r["latency"]["150"]["stale_shares"] - 2.0) < 1e-9,
       "latency splits tranches: stale value is NEVER claimable (#3)")
    ok(abs(r["latency"]["5"]["preventable_value_cents"] + 25.0) < 1e-9,
       "at L=5ms both tranches are preventable")

    iv2 = dict(iv, t_start=296.5, t_end=297.0)
    ok(label_interval(iv2, [], WF(), 300.0)["status"] == "TRUNCATED_HORIZON",
       "markout past window end excludes")

    class GapWF(WF):
        def touched(self, a, b): return True
    ok(label_interval(iv, fills, GapWF(), 300.0)["status"] == "GAP_IN_HORIZON",
       "gap excludes")

    r3 = label_interval(iv, [], WF(), 300.0)
    ok(r3["status"] == "OK" and not r3["any_fill"],
       "an unfilled interval is a first-class row")

    rows = [dict(r, status="OK", slug="x")]
    class A:
        pass
    A.fills = fills
    rec = reconcile_fills(rows, A())
    ok(rec["match"],
       "#1's test: labelled shares reconcile EXACTLY with the engine's fills")
    rows2 = [dict(r, fill_shares=4.0, status="OK")]
    ok(not reconcile_fills(rows2, A())["match"],
       "and a mismatch is DETECTED, not absorbed")

    ok(qr._qr_spec(qr.QR_SKEW, 0, cancel=False)["cancel"] is False,
       "reference is the no-cancel trajectory")
    print(f"harmful_exposure_rows v2 selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd != "run":
        ap.print_help(); return 2
    built = build_rows(per_coin=a.per_coin)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(built))
    from collections import Counter
    st = Counter(r["status"] for r in built["rows"])
    print(f"rows {len(built['rows'])} windows {built['n_windows']} "
          f"days {built['days']} recon_failures {built['reconciliation_failures']}")
    print(f"statuses {dict(st)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
