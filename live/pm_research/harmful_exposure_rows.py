"""Exposure dataset v3 — chronological decision rows, generation-true labels.

SURFACE AUTHORISATION (R-126, in-file): HARMFUL_FILL_HAZARD_TOXICITY_PLAN §10
item 1, rebuilt twice under the user's audits (2026-08-25). v2's residual
defects, named beside their v3 fixes:

  STATE BEFORE RESYNC (v2 residual). v2 observed state in `dead()`, which the
     loop calls BEFORE repositioning — an order born at t=0 was recorded born
     at t=1 (user's minimal test). v3 captures state IN THE MUTATORS: every
     placement routes through `reposition()` and every fill/queue-drain through
     `consume()` (verified in `_place_from_quote`), so a segment opens at the
     EXACT event that created its state. `dead()` only advances the clock,
     extends exposure, and acts as an UNHOOKED-CHANGE DETECTOR: a state change
     it sees that no hook produced fails the window.
  ROWS-PER-GENERATION (v2: 1.37 rows/gen, max 138 — qahead drain split
     segments). v3 KEEPS chronological decision rows (they are the
     high-frequency detection mechanism) but labels each row with the value of
     cancelling ITS GENERATION FROM THIS ROW ONWARD: tranches of the SAME
     generation with fill_t >= t_row + L, to generation end. The action-level
     evaluator cancels a generation ONCE at its first score crossing.
  WEAK RECONCILIATION (v2 compared aggregate shares; missed the shift). v3
     reconciles EXACTLY: every engine fill must land in exactly one generation
     interval of its side with a MATCHING LEVEL; any orphan or level mismatch
     fails the window.
  ERA CONTINUITY. --v2-era now also verifies per-window Binance bookTicker
     continuity (era-pure events, max gap <= 1s over the window span);
     windows failing it are excluded WITH A STATUS, not silently.

    python3 live/pm_research/harmful_exposure_rows.py --selftest
    python3 live/pm_research/harmful_exposure_rows.py run [--per-coin N]
    python3 live/pm_research/harmful_exposure_rows.py run --v2-era --coins btc
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path
from typing import Any, Sequence

import policy_optimizer_queue_realistic as qr
import inventory_walk as iw

OUT = qr.base.fi.PM / "derived/harmful_exposure_rows_v3.json"
OUT_ERA = qr.base.fi.PM / "derived/harmful_exposure_rows_v3_eraB.json"
LATENCY_GRID_MS = (5, 10, 20, 30, 50, 75, 100, 150, 250)
MARKOUT_S = 5.0
SIDES = ("BUY_UP", "SELL_UP")
TRAIN_DAYS = ("2026-08-20", "2026-08-21", "2026-08-22")
BN_MAX_GAP_S = 1.0


class RecordingArm(qr.QueueRealisticArm):
    """State captured in the MUTATORS (post-change, exact time), never dead()."""

    _instances: list["RecordingArm"] = []

    def __init__(self, spec: dict[str, Any]):
        super().__init__(spec)
        self.segments: list[dict[str, Any]] = []
        self._open: dict[str, dict[str, Any] | None] = {s: None for s in SIDES}
        self._now = 0.0
        self.unhooked_changes = 0
        RecordingArm._instances.append(self)

    def _key(self, ms: str):
        side = self.side(ms)
        return (self.generation[ms], side.level,
                round(side.resting, 9), round(getattr(side, "qahead", 0.0), 9))

    def _mark(self, ms: str) -> None:
        """Close the open segment for `ms` and open one with the NEW state."""
        cur = self._open[ms]
        key = self._key(ms)
        if cur is not None:
            if cur["_key"] == key:
                return                        # nothing actually changed
            cur["t_end"] = self._now
            self.segments.append(cur)
        side = self.side(ms)
        self._open[ms] = {
            "_key": key, "side": ms, "gen": self.generation[ms],
            "level": side.level, "resting": side.resting,
            "qahead": getattr(side, "qahead", 0.0),
            "net": (self.net() if callable(self.net) else self.net),
            "t_start": self._now, "t_end": self._now,
        }

    def reposition(self, maker_side: str, level, displayed: float) -> None:
        super().reposition(maker_side, level, displayed)
        self._mark(maker_side)

    def consume(self, maker_side: str, volume: float, displayed: float) -> float:
        filled = super().consume(maker_side, volume, displayed)
        self._mark(maker_side)               # qahead drain / fill / gen bump
        return filled

    def dead(self, when: float) -> bool:
        self._now = when
        for ms in SIDES:
            cur = self._open[ms]
            if cur is not None:
                if cur["_key"] != self._key(ms):
                    self.unhooked_changes += 1   # a mutation no hook produced
                    self._mark(ms)
                else:
                    cur["t_end"] = when
        return super().dead(when)

    def finalize(self, t: float) -> None:
        for ms in SIDES:
            cur = self._open[ms]
            if cur is not None:
                cur["t_end"] = t
                self.segments.append(cur)
                self._open[ms] = None


def generation_table(segments: Sequence[dict], fills: Sequence[Any], wf: Any,
                     window_s: float) -> tuple[dict, dict]:
    """Generation intervals + EXACT fill attribution.

    Returns (gens, recon) where gens[(side, gen)] = {"t0","t1","level_by_t",
    "tranches"} and recon reports orphans / level mismatches — any nonzero
    fails the window."""
    gens: dict = {}
    for s in segments:
        if s["level"] is None:
            continue
        k = (s["side"], s["gen"])
        g = gens.setdefault(k, {"t0": s["t_start"], "t1": s["t_end"],
                                "segs": []})
        g["t0"] = min(g["t0"], s["t_start"])
        g["t1"] = max(g["t1"], s["t_end"])
        g["segs"].append((s["t_start"], s["t_end"], s["level"]))
    recon = {"orphan_fills": 0, "level_mismatch": 0, "attributed": 0}
    for k in gens:
        gens[k]["tranches"] = []
    for f in fills:
        cands = []
        for (side, gen), g in gens.items():
            if side != f.maker_side:
                continue
            if g["t0"] - 1e-9 <= f.t <= g["t1"] + 1e-9:
                lv = None
                for a, b, level in g["segs"]:
                    if a - 1e-9 <= f.t <= b + 1e-9:
                        lv = level
                        break
                cands.append(((side, gen), lv))
        hit = [(k, lv) for k, lv in cands
               if lv is not None and abs(lv - f.level) < 1e-12]
        if len(hit) != 1:
            recon["orphan_fills"] += 1
            continue
        k, _ = hit[0]
        sgn = 1.0 if f.maker_side == "BUY_UP" else -1.0
        later = wf.mid_at(f.t + MARKOUT_S)
        gens[k]["tranches"].append({
            "t": f.t, "shares": f.size, "level": f.level,
            "markout_cents_per_share": (None if later is None
                                        else sgn * (later - f.level) * 100.0),
        })
        recon["attributed"] += 1
    return gens, recon


def label_rows(segments: Sequence[dict], gens: dict, wf: Any,
               window_s: float) -> list[dict]:
    """One chronological decision row per segment; value = cancelling the
    GENERATION from this row onward, per latency."""
    rows = []
    for s in segments:
        if s["level"] is None:
            continue
        k = (s["side"], s["gen"])
        g = gens.get(k)
        row = {kk: s[kk] for kk in ("side", "gen", "level", "resting",
                                    "qahead", "net", "t_start", "t_end")}
        row["gen_t0"] = g["t0"] if g else s["t_start"]
        row["gen_t1"] = g["t1"] if g else s["t_end"]
        row["status"] = "OK"
        if (g["t1"] if g else s["t_end"]) + MARKOUT_S > window_s:
            row["status"] = "TRUNCATED_HORIZON"
            rows.append(row); continue
        if wf.touched(s["t_start"], (g["t1"] if g else s["t_end"]) + MARKOUT_S):
            row["status"] = "GAP_IN_HORIZON"
            rows.append(row); continue
        trs = (g["tranches"] if g else [])
        if any(t["markout_cents_per_share"] is None for t in trs):
            row["status"] = "NO_FUTURE_MID"
            rows.append(row); continue
        fut = [t for t in trs if t["t"] >= s["t_start"] - 1e-9]
        row["any_fill_ahead"] = bool(fut)
        lat = {}
        for L in LATENCY_GRID_MS:
            cut = s["t_start"] + L / 1000.0
            prev = [t for t in fut if t["t"] >= cut]
            lat[str(L)] = {
                "preventable_value_cents": sum(
                    -t["markout_cents_per_share"] * t["shares"] for t in prev),
                "preventable_shares": sum(t["shares"] for t in prev),
                "stale_shares": sum(t["shares"] for t in fut if t["t"] < cut),
            }
        row["latency"] = lat
        rows.append(row)
    return rows


_BN_GAPS: dict = {}


def _bn_gap_index(sym: str, lo_ns: int) -> list:
    """Era-pure gap intervals (> BN_MAX_GAP_S) for one symbol. Built ONCE per
    symbol by a single pass over its files, then every window check is an
    interval-overlap lookup — the per-window rescan would have re-read each
    2.4M-row hour file ~12 times."""
    if sym in _BN_GAPS:
        return _BN_GAPS[sym]
    import gzip, glob
    gaps = []
    prev = None
    files = sorted(glob.glob(
        f"/home/yuqing/ctaNew/data/mm_hf/raw/bookTicker/{sym}/*.csv*"))
    for f in files:
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rb") as fh:
            for line in fh:
                i = line.find(b",")
                if i < 1:
                    continue
                try:
                    r = int(line[:i])
                except ValueError:
                    continue
                if r < lo_ns:
                    continue
                if prev is not None and r - prev > BN_MAX_GAP_S * 1e9:
                    gaps.append((prev / 1e9, r / 1e9))
                prev = r
    _BN_GAPS[sym] = {"gaps": gaps, "first": None if prev is None else lo_ns/1e9,
                     "last": None if prev is None else prev / 1e9}
    return _BN_GAPS[sym]


def binance_continuity_ok(t0: int, coin: str, bounds) -> bool:
    sym = {"btc": "BTCUSDT", "eth": "ETHUSDT"}.get(coin)
    if sym is None:
        return False
    idx = _bn_gap_index(sym, int(bounds[0] * 1e9))
    if idx["last"] is None:
        return False
    a = t0 - 10.0
    b = t0 + qr.base.fi.WINDOW_S + MARKOUT_S + 1.0
    if b > idx["last"]:
        return False
    return not any(not (ge < a or gs > b) for gs, ge in idx["gaps"])


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


def v2_era_bounds() -> tuple[float, float]:
    import json as _json, time as _time
    runs = [_json.loads(l) for l in
            open('/home/yuqing/ctaNew/data/mm_hf/collector_runs.jsonl')]
    return max(r['started_at_ns'] for r in runs) / 1e9, _time.time()


def select_v2_era(coins: Sequence[str]) -> tuple[list, int]:
    fi = qr.base.fi
    bounds = v2_era_bounds()
    paths = fi._archive_paths(); tokens = fi.token_map()
    gaps = fi.gaps_by_slug(fi.ERA)
    out = []; n_gap = 0
    for slug in sorted(fi.covered_slugs(fi.ERA)):
        coin = slug.split("-")[0]
        if coin not in coins or slug not in paths or slug not in tokens:
            continue
        try:
            t0 = int(slug.rsplit("-", 1)[1])
        except ValueError:
            continue
        if t0 < bounds[0] or t0 + fi.WINDOW_S + MARKOUT_S + 5.0 > bounds[1]:
            continue
        if not binance_continuity_ok(t0, coin, bounds):
            n_gap += 1
            continue
        up, down = tokens[slug]
        out.append((slug, paths[slug], up, down, gaps.get(slug, [])))
    return out, n_gap


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


def build_rows(per_coin: int | None = None,
               coins: Sequence[str] = ("btc", "eth"),
               v2_era: bool = False) -> dict[str, Any]:
    import datetime as _dt
    spec = qr._qr_spec(qr.QR_SKEW, latency_ms=0, cancel=False)
    if v2_era:
        selected, n_bn_gap = select_v2_era(coins)
    else:
        selected, n_bn_gap = select_stratified(per_coin or 10, coins=coins), 0
    rows: list[dict[str, Any]] = []
    recon_fail = 0; unhooked = 0
    n_windows = 0; days: set[str] = set()
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
        gens, recon = generation_table(arm.segments, arm.fills, wf,
                                       qr.base.fi.WINDOW_S)
        wrows = label_rows(arm.segments, gens, wf, qr.base.fi.WINDOW_S)
        bad = (recon["orphan_fills"] or recon["level_mismatch"]
               or arm.unhooked_changes)
        if bad:
            recon_fail += 1
            unhooked += arm.unhooked_changes
            for r in wrows:
                r["status"] = "RECONCILIATION_FAILED"
        for r in wrows:
            r["slug"] = slug; r["coin"] = slug.split("-")[0]
            r["day"] = day; r["t0"] = t0
        rows.extend(wrows)
    return {"rows": rows, "n_windows": n_windows, "days": sorted(days),
            "reconciliation_failures": recon_fail,
            "unhooked_state_changes": unhooked,
            "windows_excluded_binance_gap": n_bn_gap,
            "schema": "harmful_exposure_v3_chronological"}


def selftest() -> int:
    checks = 0

    def ok(c, label):
        nonlocal checks
        if not c:
            raise AssertionError(label)
        checks += 1

    class F:
        def __init__(self, t, side, level, size):
            self.t, self.maker_side, self.level, self.size = t, side, level, size

    class WF:
        def mid_at(self, t): return 0.60
        def touched(self, a, b): return False

    segs = [
        {"side": "BUY_UP", "gen": 1, "level": 0.55, "resting": 5.0,
         "qahead": 3.0, "net": 0.0, "t_start": 0.0, "t_end": 0.4},
        {"side": "BUY_UP", "gen": 1, "level": 0.55, "resting": 5.0,
         "qahead": 0.0, "net": 0.0, "t_start": 0.4, "t_end": 1.0},
        {"side": "BUY_UP", "gen": 2, "level": 0.54, "resting": 5.0,
         "qahead": 1.0, "net": 0.0, "t_start": 1.0, "t_end": 2.0},
    ]
    fills = [F(0.45, "BUY_UP", 0.55, 2.0), F(0.9, "BUY_UP", 0.55, 3.0),
             F(1.5, "BUY_UP", 0.54, 5.0)]
    gens, recon = generation_table(segs, fills, WF(), 300.0)
    ok(recon["orphan_fills"] == 0 and recon["attributed"] == 3,
       "every fill attributes to exactly one generation, LEVEL-MATCHED")
    ok(len(gens[("BUY_UP", 1)]["tranches"]) == 2,
       "gen 1 owns its two tranches; gen 2 the later one")
    bad_fill = [F(0.45, "BUY_UP", 0.99, 2.0)]
    _, r2 = generation_table(segs, bad_fill, WF(), 300.0)
    ok(r2["orphan_fills"] == 1,
       "a level-mismatched fill is an ORPHAN, not silently attributed — this "
       "is the exact check v2 lacked and the state-shift slipped through")

    rows = label_rows(segs, gens, WF(), 300.0)
    ok(len(rows) == 3, "chronological rows retained — one per segment")
    r0, r1, _ = rows
    ok(abs(r0["latency"]["5"]["preventable_value_cents"] -
           (-(0.60 - 0.55) * 100 * 5.0)) < 1e-9,
       "row at t=0: cancelling gen 1 from here prevents BOTH its tranches")
    ok(abs(r1["latency"]["5"]["preventable_value_cents"] -
           (-(0.60 - 0.55) * 100 * 5.0)) < 1e-9
       and r1["t_start"] == 0.4,
       "a later row of the SAME generation sees the tranches still ahead of it")
    ok(abs(r1["latency"]["250"]["preventable_value_cents"] -
           (-(0.60 - 0.55) * 100 * 3.0)) < 1e-9
       and abs(r1["latency"]["250"]["stale_shares"] - 2.0) < 1e-9,
       "from t=0.4 at L=250ms (cut 0.65) the 0.45s tranche is STALE; only the "
       "0.9s tranche is preventable — latency reduces value, never adds it")
    ok(r0["gen_t0"] == 0.0 and r0["gen_t1"] == 1.0,
       "rows carry the TRUE generation interval")

    import harmful_action_eval as ae
    ok(hasattr(ae, "evaluate_policy"), "action evaluator present")
    print(f"harmful_exposure_rows v3 selftest: {checks} checks OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--per-coin", type=int, default=None)
    ap.add_argument("--v2-era", action="store_true")
    ap.add_argument("--coins", type=str, default="btc,eth")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd != "run":
        ap.print_help(); return 2
    built = build_rows(per_coin=a.per_coin,
                       coins=tuple(a.coins.split(",")), v2_era=a.v2_era)
    out = OUT_ERA if a.v2_era else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(built))
    from collections import Counter
    st = Counter(r["status"] for r in built["rows"])
    print(f"rows {len(built['rows'])} windows {built['n_windows']} "
          f"days {built['days']} recon_failures {built['reconciliation_failures']} "
          f"unhooked {built['unhooked_state_changes']} "
          f"bn_gap_excluded {built['windows_excluded_binance_gap']}")
    print(f"statuses {dict(st)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
