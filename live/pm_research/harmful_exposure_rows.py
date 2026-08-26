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
# AUDIT 4: the plan's declared target is fills in [t+L, t+H]; v3.1 had silently
# changed it to "until generation end" via a source comment. RESTORED: the
# label window is [t_start + L, min(t_start + H, generation end)] — the cap by
# generation end is not a target change, merely the fact that a cancelled
# generation cannot fill after it has been replaced.
FILL_HORIZON_S = 1.0
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
        self.fill_log: list[dict[str, Any]] = []
        self.consume_times: list[float] = []     # EVERY consume, zero fills too
        self._consume_seq = 0
        RecordingArm._instances.append(self)

    def _key(self, ms: str):
        side = self.side(ms)
        return (self.generation[ms], side.level,
                round(side.resting, 9), round(getattr(side, "qahead", 0.0), 9))

    def _mark(self, ms: str, seq: int | None = None) -> None:
        """Close the open segment for `ms` and open one with the NEW state.
        `seq` ties a consume-driven boundary to its fill for time repair."""
        cur = self._open[ms]
        key = self._key(ms)
        if cur is not None:
            if cur["_key"] == key:
                return                        # nothing actually changed
            cur["t_end"] = self._now
            cur["_end_seq"] = seq
            self.segments.append(cur)
        side = self.side(ms)
        self._open[ms] = {
            "_key": key, "side": ms, "gen": self.generation[ms],
            "level": side.level, "resting": side.resting,
            "qahead": getattr(side, "qahead", 0.0),
            "net": (self.net() if callable(self.net) else self.net),
            "t_start": self._now, "t_end": self._now,
            "_start_seq": seq,
        }

    def note_event_time(self, t: float) -> None:
        """Engine hook (audit 4): the trade receipt time arrives BEFORE any
        consume on that trade, so every consume-driven boundary — zero-fill
        queue drains included — is stamped at its true time. No repair step."""
        self._now = t

    def reposition(self, maker_side: str, level, displayed: float) -> None:
        super().reposition(maker_side, level, displayed)
        self._mark(maker_side)               # quote-driven: resync clock is exact

    def consume(self, maker_side: str, volume: float, displayed: float) -> float:
        """USER AUDIT 3 (the trade-path defect): `_now` is a RESYNC clock and
        trades arrive between resyncs (measured p99 lag 28.1 ms, max 63.7 ms) —
        and `super().consume()` bumps the generation on a full fill BEFORE any
        recording, so time-containment handed 370/458 fills to the generation
        the fill CREATED instead of the one it killed. Fix: record the
        PRE-consume generation and level explicitly, per fill, and tag every
        consume-driven segment with a sequence number so its boundary can be
        repaired to the fill's EXACT engine time after replay."""
        side = self.side(maker_side)
        pre_gen = self.generation[maker_side]
        pre_level = side.level
        filled = super().consume(maker_side, volume, displayed)
        self._consume_seq += 1
        self.consume_times.append(self._now)
        if filled > 0:
            self.fill_log.append({
                "seq": self._consume_seq, "side": maker_side,
                "pre_gen": pre_gen, "pre_level": pre_level,
                "post_gen": self.generation[maker_side], "filled": filled,
            })
        self._mark(maker_side, seq=self._consume_seq)
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


def join_fills(fill_log: Sequence[dict], engine_fills: Sequence[Any]) -> tuple[list, dict]:
    """Order-preserving join: recorder entries and `arm.fills` are produced by
    the SAME consume calls in the SAME order, so pairing k-th with k-th gives
    every fill its EXACT engine receipt time AND its explicit pre-consume
    generation. Reconciliation is per-tuple (side, level, size) — the aggregate
    check that let the 370/458 misassignment through is gone."""
    recon = {"n_engine": len(engine_fills), "n_recorded": len(fill_log),
             "count_mismatch": len(engine_fills) != len(fill_log),
             "tuple_mismatches": 0}
    joined = []
    for log, f in zip(fill_log, engine_fills):
        bad = (log["side"] != f.maker_side
               or log["pre_level"] is None
               or abs(log["pre_level"] - f.level) > 1e-12
               or abs(log["filled"] - f.size) > 1e-9)
        if bad:
            recon["tuple_mismatches"] += 1
            continue
        joined.append({"t": f.t, "side": f.maker_side, "level": f.level,
                       "shares": f.size, "gen": log["pre_gen"],
                       "post_gen": log["post_gen"], "seq": log["seq"]})
    return joined, recon


def verify_boundary_times(segments: Sequence[dict], joined: Sequence[dict]) -> int:
    """AUDIT-4 STRICT GATE. With `note_event_time` there is nothing to repair:
    every consume-driven boundary must ALREADY equal its fill's engine time.
    Returns the number of violations — part of the FAILURE condition, so a
    contaminated window can never sit in the receipt marked OK (the previous
    version counted the residual and then ignored it, which the audit called
    out as the worse half of the defect)."""
    t_by_seq = {j["seq"]: j["t"] for j in joined}
    bad = 0
    for seg in segments:
        sseq = seg.pop("_start_seq", None)
        eseq = seg.pop("_end_seq", None)
        if sseq in t_by_seq and abs(seg["t_start"] - t_by_seq[sseq]) > 1e-9:
            bad += 1
        if eseq in t_by_seq and abs(seg["t_end"] - t_by_seq[eseq]) > 1e-9:
            bad += 1
    return bad


def trade_receipt_times(path: Path, up_id: str, down_id: str) -> list[float]:
    """Independent parse of the window's trade receipt times, for the hard
    zero-fill clock gate. Reads the raw tape directly — shares NOTHING with the
    recorder's clock, so a regression in `note_event_time` cannot fool it."""
    import json as _json
    slug = path.name.split(".jsonl")[0]
    ws = int(slug.rsplit("-", 1)[1])
    out = []
    for line in qr.base.fi._gz_lines(path):
        if qr.base.fi.TRADE_MARK not in line:
            continue
        parts = line.split(b"\t", 1)
        if len(parts) != 2:
            continue
        try:
            recv = int(parts[0]) / 1e9 - ws
            payload = _json.loads(parts[1])
        except (ValueError, _json.JSONDecodeError):
            continue
        for msg in payload if isinstance(payload, list) else [payload]:
            if (isinstance(msg, dict)
                    and msg.get("event_type") == "last_trade_price"
                    and str(msg.get("asset_id")) in (up_id, down_id)):
                out.append(recv)
                break
    return sorted(out)


# Membership tolerance for the clock gate. Window-relative times are computed
# by SUBTRACTING two ~1.8e9-second values; the double ULP at that magnitude is
# ~2.4e-7 s, so two independently-computed copies of the SAME nanosecond can
# differ by that much (measured: the engine's own fill times failed a 1e-9
# membership test 262/263 while agreeing to 4dp). 1e-6 sits 4x above the ULP
# and THREE ORDERS below both the minimum inter-trade spacing (p10 gap 0.3 ms)
# and the stale-clock signature this gate exists to catch (>=ms). It cannot
# absorb a regression; it only absorbs float representation.
CLOCK_TOL_S = 1e-6

# R-153(2): the era floor is READ FROM THE MANIFEST, which owns the pin.
from harmful_candidate_manifest import ERA_BOUNDARY_NS


def verify_consume_clock(consume_times: Sequence[float],
                         trade_times: Sequence[float]) -> int:
    """AUDIT 5 BLOCKER 4: the boundary verifier covered only positive fills, so
    a future ZERO-FILL clock regression would be invisible. Every consume
    timestamp — zero fills included — must be a member of the independently
    parsed trade-receipt times: a stale resync stamp is a QUOTE-event time and
    will not coincide with a trade time at ns resolution. Violations join the
    failure condition."""
    import bisect as _b
    bad = 0
    for t in consume_times:
        i = _b.bisect_left(trade_times, t - CLOCK_TOL_S)
        if i >= len(trade_times) or abs(trade_times[i] - t) > CLOCK_TOL_S:
            bad += 1
    return bad


def generation_table(segments: Sequence[dict], joined: Sequence[dict], wf: Any,
                     window_s: float) -> tuple[dict, dict]:
    """Generation intervals + EXPLICIT attribution: each joined fill carries its
    pre-consume generation, so attribution is by construction, and the
    containment check is a VERIFIER (wrong_generation_assignments must be 0)."""
    gens: dict = {}
    for s in segments:
        if s["level"] is None:
            continue
        k = (s["side"], s["gen"])
        g = gens.setdefault(k, {"t0": s["t_start"], "t1": s["t_end"]})
        g["t0"] = min(g["t0"], s["t_start"])
        g["t1"] = max(g["t1"], s["t_end"])
    for k in gens:
        gens[k]["tranches"] = []
    recon = {"orphan_fills": 0, "wrong_generation_assignments": 0,
             "attributed": 0}
    for f in joined:
        k = (f["side"], f["gen"])
        g = gens.get(k)
        if g is None:
            recon["orphan_fills"] += 1
            continue
        if not (g["t0"] - 1e-9 <= f["t"] <= g["t1"] + 1e-9):
            recon["wrong_generation_assignments"] += 1
        sgn = 1.0 if f["side"] == "BUY_UP" else -1.0
        later = wf.mid_at(f["t"] + MARKOUT_S)
        g["tranches"].append({
            "t": f["t"], "shares": f["shares"], "level": f["level"],
            "markout_cents_per_share": (None if later is None
                                        else sgn * (later - f["level"]) * 100.0),
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
        # AUDIT 5 BLOCKER 1: observability is scoped to THE ROW'S OWN TARGET,
        # [t_start, h_end + MARKOUT_S] with h_end = min(t+H, gen end) — NOT to
        # generation end. Checking through gen_t1 + 5s marked a fully
        # observable t=100 row TRUNCATED because its generation lived to 300,
        # selectively deleting long-lived and no-fill generations from the
        # training population.
        h_end = min(s["t_start"] + FILL_HORIZON_S,
                    (g["t1"] if g else s["t_end"]) + 1e-9)
        trs = (g["tranches"] if g else [])
        fut = [t for t in trs
               if s["t_start"] - 1e-9 <= t["t"] <= h_end]
        # AUDIT 6: the observation endpoint is max(h_end, latest fill + 5s) —
        # certifying "no further fill" needs the tape only through h_end;
        # markout data is needed 5s after ACTUAL fills, not hypothetical ones.
        # Requiring h_end + 5s unconditionally excluded fully-observable
        # no-fill rows (truncation AND gap) and early-fill rows whose own
        # markouts were available.
        obs_end = max(h_end,
                      max((t["t"] + MARKOUT_S for t in fut), default=h_end))
        if obs_end > window_s:
            row["status"] = "TRUNCATED_HORIZON"
            rows.append(row); continue
        if wf.touched(s["t_start"], obs_end):
            row["status"] = "GAP_IN_HORIZON"
            rows.append(row); continue
        # only tranches INSIDE this row's horizon need markouts
        if any(t["markout_cents_per_share"] is None for t in fut):
            row["status"] = "NO_FUTURE_MID"
            rows.append(row); continue
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


# --- era bounds: PINNED LITERALS, never derived (R-153(2), Q-DA-67) --------
# THE DEFECT THIS REPLACES, and why it was not a small one:
#   floor = max(started_at_ns) over collector_runs.jsonl
#   end   = time.time()
# Both FLOAT. Every collector restart appended a ledger row, so the floor
# walked 39.6 h forward and select_v2_era admitted 0 of 926 windows -- the
# entire 08-24/25 consumed fragment fell below its own era floor. A build in
# that state produces zero windows, which reads as MODEL IRREPRODUCIBILITY
# when it is purely a selector failure. And `time.time()` made the admitted
# population depend on the wall clock, so two runs minutes apart select
# different windows: a cent-exact reproduction is IMPOSSIBLE against a moving
# population, independent of any memory or model question.
#
# THE RULE: a ledger row changes the era ONLY on a transition of
# (collector_schema_version, stamp_point). A restart on the SAME key is a
# COVERAGE GAP, not a new era -- gaps are handled per-event by the existing
# binance-continuity check, never by moving the floor.
ERA_KEY_FIELDS = ("collector_schema_version", "stamp_point")
# Verified at the ledger, not guessed: all four rows carry this exact pair.
# (BE's first draft used a TRUNCATED stamp_point copied from a 110-char
# console display; assert_no_era_transition REFUSED it immediately rather
# than admitting a wrong era. Third time this session an instrument caught
# its own author -- and the failure mode it prevented is the worst kind:
# a plausible-looking era key that silently admits the wrong tape.)
DECLARED_ERA_KEY = ("hf_ws_v2_recv_boundary",
                    "IMMEDIATELY_AFTER_WS_RECV_BEFORE_JSON_PARSE")
LEDGER_PATH = '/home/yuqing/ctaNew/data/mm_hf/collector_runs.jsonl'

# Population-scoped declared END values. Each is a LITERAL tied to a named
# population, never a clock read. v3.4: last slug t0 1787650200 (08-25
# 09:30:00Z) + WINDOW_S(300) + MARKOUT_S(5) + 5 -- the end of the last
# complete window in the consumed fragment, verified at the artifact
# (471 slugs, matching the receipt's n_windows).
DECLARED_ERA_END_S = {"v3_4_consumed_fragment": 1787650510.0}


class EraTransition(RuntimeError):
    """The ledger shows a genuine era change; a pinned literal cannot stand."""


def ledger_era_keys(path: str = LEDGER_PATH) -> list[tuple]:
    import json as _json
    out = []
    for line in open(path):
        try:
            r = _json.loads(line)
        except ValueError:
            continue
        out.append(tuple(r.get(f) for f in ERA_KEY_FIELDS))
    return out


def assert_no_era_transition(path: str = LEDGER_PATH) -> None:
    """REFUSE if the ledger contains a key other than the declared one.

    This is what makes pinning SAFE rather than merely stable: if the
    collector's schema or stamp point genuinely changes, the pinned literal no
    longer describes the admissible range, and continuing would silently mix
    eras -- exactly the failure CLAUDE.md rule 5 exists to prevent. A restart
    on the same key is fine and must NOT trip this."""
    keys = set(ledger_era_keys(path))
    unexpected = keys - {DECLARED_ERA_KEY}
    if unexpected:
        raise EraTransition(
            f"ledger carries era key(s) {sorted(unexpected)!r} besides the "
            f"declared {DECLARED_ERA_KEY!r}. The pinned boundary describes only "
            f"the declared era; re-declare the boundary before building.")


def v2_era_bounds(population: str = "v3_4_consumed_fragment",
                  era_end_s: float | None = None) -> tuple[float, float]:
    """(floor, end) as PINNED LITERALS. `time.time()` is forbidden here."""
    assert_no_era_transition()
    end = era_end_s if era_end_s is not None else DECLARED_ERA_END_S.get(population)
    if end is None:
        raise ValueError(
            f"no declared era end for population {population!r}. An end must be "
            f"DECLARED per population; defaulting to the clock is what made the "
            f"population non-reproducible.")
    return ERA_BOUNDARY_NS / 1e9, float(end)


def select_v2_era(coins: Sequence[str],
                  population: str = "v3_4_consumed_fragment") -> tuple[list, int]:
    fi = qr.base.fi
    bounds = v2_era_bounds(population)
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
               v2_era: bool = False,
               population: str = "v3_4_consumed_fragment") -> dict[str, Any]:
    import datetime as _dt
    spec = qr._qr_spec(qr.QR_SKEW, latency_ms=0, cancel=False)
    if v2_era:
        selected, n_bn_gap = select_v2_era(coins, population)
    else:
        selected, n_bn_gap = select_stratified(per_coin or 10, coins=coins), 0
    rows: list[dict[str, Any]] = []
    recon_fail = 0; unhooked = 0; wrong_gen = 0; boundary_bad_total = 0
    clock_bad_total = 0
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
        joined, jrec = join_fills(arm.fill_log, arm.fills)
        n_boundary_bad = verify_boundary_times(arm.segments, joined)
        ttimes = trade_receipt_times(ent[1], ent[2], ent[3])
        n_clock_bad = verify_consume_clock(arm.consume_times, ttimes)
        gens, recon = generation_table(arm.segments, joined, wf,
                                       qr.base.fi.WINDOW_S)
        wrows = label_rows(arm.segments, gens, wf, qr.base.fi.WINDOW_S)
        bad = (jrec["count_mismatch"] or jrec["tuple_mismatches"]
               or recon["orphan_fills"]
               or recon["wrong_generation_assignments"]
               or arm.unhooked_changes
               or n_boundary_bad
               or n_clock_bad)                 # STRICT: in the failure condition
        wrong_gen += recon["wrong_generation_assignments"]
        boundary_bad_total += n_boundary_bad
        clock_bad_total += n_clock_bad
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
            "wrong_generation_assignments": wrong_gen,
            "boundary_time_violations": boundary_bad_total,
            "consume_clock_violations": clock_bad_total,
            "windows_excluded_binance_gap": n_bn_gap,
            "schema": "harmful_exposure_v3_4_fill_scoped_markout"}


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

    # THE AUDIT-3 SCENARIO: a full fill at t=0.7 kills gen 1; the engine bumps
    # to gen 2 BEFORE any recording, and the stale resync clock says 0.5.
    # v3.0 attributed this fill to gen 2 (370/458 on real tape). v3.1 must
    # attribute it to gen 1, explicitly, and repair the boundary to 0.7.
    fill_log = [{"seq": 7, "side": "BUY_UP", "pre_gen": 1, "pre_level": 0.55,
                 "post_gen": 2, "filled": 5.0}]
    engine = [F(0.7, "BUY_UP", 0.55, 5.0)]
    joined, jrec = join_fills(fill_log, engine)
    ok(not jrec["count_mismatch"] and jrec["tuple_mismatches"] == 0
       and joined[0]["gen"] == 1,
       "the fill carries its PRE-consume generation, from the explicit record")
    segs = [
        {"side": "BUY_UP", "gen": 1, "level": 0.55, "resting": 5.0,
         "qahead": 0.0, "net": 0.0, "t_start": 0.0, "t_end": 0.7,
         "_end_seq": 7},
        {"side": "BUY_UP", "gen": 2, "level": 0.55, "resting": 5.0,
         "qahead": 2.0, "net": 0.0, "t_start": 0.7, "t_end": 2.0,
         "_start_seq": 7},
    ]
    ok(verify_boundary_times([dict(x) for x in segs], joined) == 0,
       "with the true trade clock, boundaries ALREADY equal engine times")
    stale = [dict(segs[0], t_end=0.5), dict(segs[1], t_start=0.5)]
    ok(verify_boundary_times(stale, joined) == 2,
       "a stale-stamped boundary is a VIOLATION in the failure condition — "
       "audit 4: counted-but-ignored is the worse half of the defect")
    segs2 = [dict(x) for x in segs]
    for x in segs2: x.pop("_end_seq", None); x.pop("_start_seq", None)
    gens, recon = generation_table(segs2, joined, WF(), 300.0)
    ok(recon["wrong_generation_assignments"] == 0
       and len(gens[("BUY_UP", 1)]["tranches"]) == 1
       and len(gens[("BUY_UP", 2)]["tranches"]) == 0,
       "the fill belongs to the generation it killed")

    bad_log = [dict(fill_log[0], pre_level=0.54)]
    _, jrec2 = join_fills(bad_log, engine)
    ok(jrec2["tuple_mismatches"] == 1, "level mismatch counted per tuple")
    _, jrec3 = join_fills([], engine)
    ok(jrec3["count_mismatch"], "count mismatch detected")

    # AUDIT 6 — the observation endpoint, three reproduced cases:
    # (a) no-fill row, 1s horizon observable, NEXT 5s truncated -> OK
    lgn = {("BUY_UP", 1): {"t0": 298.4, "t1": 298.9, "tranches": []}}
    segn = [{"side": "BUY_UP", "gen": 1, "level": 0.55, "resting": 5.0,
             "qahead": 0.0, "net": 0.0, "t_start": 298.4, "t_end": 298.9}]
    ok(label_rows(segn, lgn, WF(), 300.0)[0]["status"] == "OK",
       "a NO-FILL row needs the tape only through h_end — the five seconds "
       "after it are for markouts that do not exist")
    # (b) gap strictly after the 1s horizon, no fill -> OK
    class GapAfterH(WF):
        def touched(self, a, b): return b > 101.2
    lgh = {("BUY_UP", 1): {"t0": 100.0, "t1": 100.9, "tranches": []}}
    segh = [{"side": "BUY_UP", "gen": 1, "level": 0.55, "resting": 5.0,
             "qahead": 0.0, "net": 0.0, "t_start": 100.0, "t_end": 100.9}]
    ok(label_rows(segh, lgh, GapAfterH(), 300.0)[0]["status"] == "OK",
       "a gap AFTER a no-fill row's horizon does not exclude it")
    # (c) early fill with its own markout available; gap before h_end+5 -> OK
    class GapLate(WF):
        def touched(self, a, b): return b > 105.6
    lge = {("BUY_UP", 1): {"t0": 100.0, "t1": 101.5, "tranches": [
        {"t": 100.2, "shares": 5.0, "level": 0.55,
         "markout_cents_per_share": 5.0}]}}
    sege = [{"side": "BUY_UP", "gen": 1, "level": 0.55, "resting": 5.0,
             "qahead": 0.0, "net": 0.0, "t_start": 100.0, "t_end": 101.5}]
    re_ = label_rows(sege, lge, GapLate(), 300.0)[0]
    ok(re_["status"] == "OK"
       and abs(re_["latency"]["5"]["preventable_value_cents"] + 25.0) < 1e-9,
       "an early fill needs data through ITS OWN fill_t+5s (105.2), not "
       "h_end+5s (106.0) — the row labels")
    # and the guard still guards: a gap INSIDE a filled row's markout excludes
    class GapInMk(WF):
        def touched(self, a, b): return b > 104.0
    ok(label_rows(sege, lge, GapInMk(), 300.0)[0]["status"] == "GAP_IN_HORIZON",
       "a gap inside the needed markout window still excludes")

    # AUDIT 5 BLOCKER 1 — the user's three reproduced cases, as regressions:
    lg = {("BUY_UP", 1): {"t0": 100.0, "t1": 300.0, "tranches": []}}
    seg_l = [{"side": "BUY_UP", "gen": 1, "level": 0.55, "resting": 5.0,
              "qahead": 0.0, "net": 0.0, "t_start": 100.0, "t_end": 300.0}]
    rl = label_rows(seg_l, lg, WF(), 300.0)
    ok(rl[0]["status"] == "OK",
       "a t=100 row observable through 106.5 is OK even though its generation "
       "lives to 300 — long-lived generations are no longer selectively deleted")

    class LateGap(WF):
        def touched(self, a, b): return b > 150.0     # gap only AFTER the target
    ok(label_rows(seg_l, lg, LateGap(), 300.0)[0]["status"] == "OK",
       "a gap after the row's target does not exclude the row")

    lg2 = {("BUY_UP", 1): {"t0": 100.0, "t1": 300.0, "tranches": [
        {"t": 199.0, "shares": 5.0, "level": 0.55,
         "markout_cents_per_share": None}]}}
    ok(label_rows(seg_l, lg2, WF(), 300.0)[0]["status"] == "OK",
       "a missing markout for a fill OUTSIDE the 1s horizon does not poison "
       "the row — NO_FUTURE_MID applies only to the row's own tranches")

    # AUDIT 5 BLOCKER 4 — the hard clock gate:
    trades = [10.0, 10.5, 11.0]
    ok(verify_consume_clock([10.5, 11.0], trades) == 0,
       "consume times that ARE trade receipt times pass")
    ok(verify_consume_clock([10.5 + 2.4e-7], trades) == 0,
       "a ULP-level float difference is representation, not a violation")
    ok(verify_consume_clock([10.3], trades) == 1,
       "a consume stamped at a NON-trade time (a stale resync stamp) is a "
       "violation — the gate now covers zero-fill drains, which the "
       "positive-fill-only verifier could not")

    # THE PLAN'S HORIZON IS BACK: a tranche beyond t_start + H does not label,
    # even inside the generation
    gens3 = {("BUY_UP", 1): {"t0": 0.0, "t1": 5.0, "tranches": [
        {"t": 0.7, "shares": 5.0, "level": 0.55,
         "markout_cents_per_share": 5.0},
        {"t": 2.5, "shares": 5.0, "level": 0.55,
         "markout_cents_per_share": 5.0}]}}
    seg3 = [{"side": "BUY_UP", "gen": 1, "level": 0.55, "resting": 5.0,
             "qahead": 0.0, "net": 0.0, "t_start": 0.0, "t_end": 5.0}]
    rows = label_rows(seg3, gens3, WF(), 300.0)
    ok(abs(rows[0]["latency"]["5"]["preventable_value_cents"] + 25.0) < 1e-9,
       "only the tranche inside [t+L, t+H] labels (H=1.0s, the PLAN'S target); "
       "the 2.5s tranche is outside the declared horizon")

    import harmful_action_eval as ae
    ok(hasattr(ae, "evaluate_policy"), "action evaluator present")
    # ---- R-153(2) era-pin falsifiers (rule 15: both arms must fire) -------
    import tempfile as _tf, json as _js, inspect as _ins, ast as _ast

    # KNOWN-GOOD: the real ledger is one era, so bounds resolve.
    _b = v2_era_bounds()
    ok(_b[0] == ERA_BOUNDARY_NS / 1e9,
       "era floor IS the pinned literal, not a ledger max")
    ok(_b[1] == DECLARED_ERA_END_S["v3_4_consumed_fragment"],
       "era end IS the declared literal for the population")
    ok(len(set(ledger_era_keys())) == 1,
       "the real ledger carries ONE era key: the 08-26 restarts are coverage "
       "gaps, not era transitions")

    # POSITIVE CONTROL: a genuine era transition MUST be refused.
    with _tf.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(_js.dumps({"started_at_ns": 1, **dict(zip(ERA_KEY_FIELDS,
                 DECLARED_ERA_KEY))}) + "\n")
        fh.write(_js.dumps({"started_at_ns": 2,
                 "collector_schema_version": "hf_ws_v3_SOMETHING_ELSE",
                 "stamp_point": DECLARED_ERA_KEY[1]}) + "\n")
        _bad = fh.name
    try:
        assert_no_era_transition(_bad)
        ok(False, "POSITIVE CONTROL: a schema change must be REFUSED")
    except EraTransition:
        ok(True, "POSITIVE CONTROL: a real era transition is REFUSED")

    # a same-key restart must NOT trip the guard (the 08-26 case)
    with _tf.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        for n in (1, 2, 3):
            fh.write(_js.dumps({"started_at_ns": n,
                     **dict(zip(ERA_KEY_FIELDS, DECLARED_ERA_KEY))}) + "\n")
        _good = fh.name
    assert_no_era_transition(_good)
    ok(True, "three restarts on the SAME key do NOT trip the guard")

    # an undeclared population must RAISE, never fall back to a clock
    try:
        v2_era_bounds("population_that_was_never_declared")
        ok(False, "an undeclared population must raise")
    except ValueError:
        ok(True, "an undeclared population RAISES rather than defaulting to "
                 "the wall clock -- the defect that made the population "
                 "non-reproducible")

    # AST proof that no clock is consulted (a docstring may NAME time.time)
    _fn = _ast.parse(_ins.getsource(v2_era_bounds)).body[0]
    _calls = {(n.func.attr if isinstance(n.func, _ast.Attribute)
               else n.func.id if isinstance(n.func, _ast.Name) else "?")
              for n in _ast.walk(_fn) if isinstance(n, _ast.Call)}
    ok(not (_calls & {"time", "now", "today", "monotonic"}),
       "v2_era_bounds consults NO clock (checked by AST, not by grepping the "
       "source, which would match the docstring forbidding it)")

    # the consumed fragment must fall inside its own bounds
    ok(1787579400 >= _b[0] and 1787650200 + 310.0 <= _b[1],
       "the v3.4 fragment's first and last slugs fall INSIDE the pinned "
       "bounds -- the 0-of-926 selection failure cannot recur")

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
          f"wrong_gen {built['wrong_generation_assignments']} "
          f"boundary_violations {built['boundary_time_violations']} "
          f"clock_violations {built['consume_clock_violations']} "
          f"bn_gap_excluded {built['windows_excluded_binance_gap']}")
    print(f"statuses {dict(st)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
