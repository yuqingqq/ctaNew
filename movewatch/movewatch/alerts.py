"""Episode state machine: detector triggers -> a small number of messages.

One live episode per asset (either direction). Anchor = the pre-move reference
price (price at t - h for the horizon that fired). Messages:

  NEW       first trigger for an asset with no live episode
  ESCALATE  the move from the anchor crosses the next ladder level
            (up: +10 +20 +30 +50 +75 +100 +150 +200 +300 +400 +500 +750 +1000 %;
             down: -10 -15 -20 -30 -40 -50 -60 -70 -80 -90 %)
  REVERSAL  price gives back rev = clamp(0.25 * log-gain, 8%, 20%) from the extreme of
            an episode that reached >= 15%. The reversal closes the episode and opens
            the opposite one anchored at the extreme (a SPAWNED episode): later legs
            continue on its ladder, but a spawned episode never reverses again
            (no reversal chains in chop).

Noise controls (the 2026-09-28 smoke replay: 37% of messages came < 30 min after the
previous one on the same coin):
  - per-asset NEW budget: >= asset_gap_s after the coin's previous message (any kind) and
    <= asset_max_new_per_day NEW alerts per rolling 24h. ESCALATE and REVERSAL are exempt
    and do not count: they continue an episode the user has already seen and are bounded
    by the ladder and min_gap_s. Gating them hid QNT's final +33% -> +89% leg (v2), and
    counting them toward the cap delayed QNT's blow-off NEW to the top minute (v3);
    CALIBRATION.md 1c/1d.
  - a trigger in the opposite direction of a live episode is ignored unless that episode
    has fully retraced (move <= 0); direction changes are the REVERSAL rule's job.
  - an episode is not closed by retrace before min_life_s; after ttl_s without a message
    it closes; a NEW within new_cooldown_s of a close needs a move new_margin x larger
    than the closed episode's extreme.
Suppressed triggers are counted (self.suppressed) -- a status, not a silent drop.
"""
from __future__ import annotations

import itertools
import json
import math
import os
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

LADDER_UP = [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000]
LADDER_DOWN = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90]


@dataclass
class AlertCfg:
    ladder_up: list = field(default_factory=lambda: list(LADDER_UP))
    ladder_down: list = field(default_factory=lambda: list(LADDER_DOWN))
    min_gap_s: int = 180
    asset_gap_s: int = 900
    asset_max_new_per_day: int = 4
    ttl_s: int = 24 * 3600
    min_life_s: int = 2 * 3600
    new_cooldown_s: int = 3600
    new_margin: float = 1.25
    reversal_frac: float = 0.25
    reversal_min_pct: float = 8.0
    reversal_max_pct: float = 20.0
    reversal_eligible_pct: float = 15.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "AlertCfg":
        return cls(**(d or {}))


@dataclass
class Alert:
    kind: str                 # NEW | ESCALATE | REVERSAL
    asset: str
    direction: int            # +1 pump, -1 dump
    ts: float                 # unix seconds
    price: float              # reference price (same units as the anchor)
    anchor_px: float
    anchor_ts: float
    move_pct: float           # simple % from anchor
    horizon: str | None = None
    extreme_px: float | None = None
    episode_id: str = ""
    seq: int = 1
    metrics: dict = field(default_factory=dict)


@dataclass
class Episode:
    id: str
    asset: str
    direction: int
    anchor_px: float
    anchor_ts: float
    start_ts: float
    extreme_px: float
    extreme_ts: float
    last_msg_ts: float
    level_idx: int            # highest ladder index already announced (-1 = none)
    ref_key: str | None = None
    seq: int = 1
    spawned: bool = False     # opened by a REVERSAL: may escalate, never reverses

    def move(self, px: float) -> float:
        """Signed log move in the episode's direction (positive = extending)."""
        return self.direction * math.log(px / self.anchor_px)


def _level_log(direction: int, pct_level: float) -> float:
    return math.log1p(pct_level / 100.0) if direction > 0 else -math.log1p(-pct_level / 100.0)


class AlertEngine:
    def __init__(self, cfg: AlertCfg | None = None):
        self.cfg = cfg or AlertCfg()
        self.active: dict[tuple[str, int], Episode] = {}
        self.closed: dict[tuple[str, int], tuple[float, float]] = {}  # -> (close ts, extreme log move)
        self.sent: dict[str, deque] = defaultdict(deque)             # asset -> any-message timestamps (24h)
        self.new_sent: dict[str, deque] = defaultdict(deque)         # asset -> NEW timestamps (24h)
        self.suppressed: dict[str, int] = defaultdict(int)
        self._ids = itertools.count(1)
        self.lv = {+1: [_level_log(+1, p) for p in self.cfg.ladder_up],
                   -1: [_level_log(-1, p) for p in self.cfg.ladder_down]}

    # ------------------------------------------------------------ helpers
    def _levels_reached(self, direction: int, logmove: float) -> int:
        k = -1
        for i, lv in enumerate(self.lv[direction]):
            if logmove >= lv - 1e-12:
                k = i
        return k

    def _live(self, asset: str) -> Episode | None:
        return self.active.get((asset, +1)) or self.active.get((asset, -1))

    def _budget_ok(self, asset: str, ts: float) -> bool:
        """May this asset get a NEW now? Gap from its last message of ANY kind; daily cap on NEW only."""
        for q in (self.sent[asset], self.new_sent[asset]):
            while q and ts - q[0] > 86400:
                q.popleft()
        q = self.sent[asset]
        if q and ts - q[-1] < self.cfg.asset_gap_s:
            return False
        return len(self.new_sent[asset]) < self.cfg.asset_max_new_per_day

    def _record(self, asset: str, ts: float, new: bool = False) -> None:
        self.sent[asset].append(ts)
        if new:
            self.new_sent[asset].append(ts)

    def _open(self, asset: str, direction: int, ts: float, px: float, anchor_px: float,
              anchor_ts: float, ref_key: str | None, spawned: bool = False) -> Episode:
        ep = Episode(id=f"{asset}-{int(ts)}-{next(self._ids)}", asset=asset, direction=direction,
                     anchor_px=anchor_px, anchor_ts=anchor_ts, start_ts=ts, extreme_px=px,
                     extreme_ts=ts, last_msg_ts=ts, level_idx=-1, ref_key=ref_key, spawned=spawned)
        ep.level_idx = self._levels_reached(direction, ep.move(px))
        self.active[(asset, direction)] = ep
        return ep

    def _close(self, ep: Episode, ts: float) -> None:
        self.active.pop((ep.asset, ep.direction), None)
        self.closed[(ep.asset, ep.direction)] = (ts, ep.move(ep.extreme_px))

    def _alert(self, kind: str, ep: Episode, ts: float, px: float, horizon=None, metrics=None) -> Alert:
        return Alert(kind=kind, asset=ep.asset, direction=ep.direction, ts=ts, price=px,
                     anchor_px=ep.anchor_px, anchor_ts=ep.anchor_ts,
                     move_pct=(px / ep.anchor_px - 1.0) * 100.0, horizon=horizon, extreme_px=ep.extreme_px,
                     episode_id=ep.id, seq=ep.seq, metrics=metrics or {})

    # ------------------------------------------------------------ inputs
    def on_trigger(self, asset: str, ts: float, px: float, direction: int, r_log: float,
                   horizon: str, metrics: dict | None = None, ref_key: str | None = None) -> list[Alert]:
        """A horizon fired. r_log is the asset's raw log return over that horizon
        (anchor = px * exp(-r_log))."""
        out = self.on_price(asset, ts, px)
        live = self._live(asset)
        if live is not None:
            if live.direction == direction or live.move(px) > 0:
                return out                      # same leg, or opposite noise inside a live move
            self._close(live, ts)               # the live move fully retraced: a real turn
        last = self.closed.get((asset, direction))
        if last is not None and ts - last[0] < self.cfg.new_cooldown_s \
                and direction * r_log <= self.cfg.new_margin * last[1]:
            self.suppressed["cooldown"] += 1
            return out
        if not self._budget_ok(asset, ts):
            self.suppressed["asset_budget"] += 1
            return out
        h_s = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "24h": 86400}.get(horizon, 0)
        ep = self._open(asset, direction, ts, px, px * math.exp(-r_log), ts - h_s, ref_key)
        self._record(asset, ts, new=True)
        out.append(self._alert("NEW", ep, ts, px, horizon=horizon, metrics=metrics))
        return out

    def on_price(self, asset: str, ts: float, px: float, metrics: dict | None = None) -> list[Alert]:
        """Advance the asset's live episode with the latest reference price."""
        out: list[Alert] = []
        if not (px > 0):
            return out
        for d in (+1, -1):
            ep = self.active.get((asset, d))
            if ep is None:
                continue
            if d * math.log(px / ep.extreme_px) > 0:
                ep.extreme_px, ep.extreme_ts = px, ts
            mv = ep.move(px)
            # 1) reversal off the extreme (never for spawned episodes)
            gain = ep.move(ep.extreme_px)
            if not ep.spawned and gain >= math.log1p(self.cfg.reversal_eligible_pct / 100.0):
                lo = -math.log1p(-self.cfg.reversal_min_pct / 100.0)
                hi = -math.log1p(-self.cfg.reversal_max_pct / 100.0)
                need = min(max(self.cfg.reversal_frac * gain, lo), hi)
                if d * math.log(ep.extreme_px / px) >= need:
                    ep.seq += 1
                    out.append(self._alert("REVERSAL", ep, ts, px, metrics=metrics))
                    self._record(asset, ts)
                    self._close(ep, ts)
                    self._open(asset, -d, ts, px, ep.extreme_px, ep.extreme_ts, ep.ref_key, spawned=True)
                    continue
            # 2) ladder escalation
            k = self._levels_reached(d, mv)
            if k > ep.level_idx and ts - ep.last_msg_ts >= self.cfg.min_gap_s:
                ep.level_idx = k
                ep.last_msg_ts = ts
                ep.seq += 1
                out.append(self._alert("ESCALATE", ep, ts, px, metrics=metrics))
                self._record(asset, ts)
                continue
            # 3) expiry
            age = ts - ep.start_ts
            if ts - ep.last_msg_ts > self.cfg.ttl_s or (mv < self.lv[d][0] / 3.0 and age >= self.cfg.min_life_s):
                self._close(ep, ts)
        return out

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        """Live episodes, NEW cooldowns and per-asset send history, so a restart never
        re-announces a tracked move or resets a coin's message budget."""
        d = {"active": [asdict(e) for e in self.active.values()],
             "closed": [[a, dr, ts, mv] for (a, dr), (ts, mv) in self.closed.items()],
             "sent": {a: list(q) for a, q in self.sent.items() if q},
             "new_sent": {a: list(q) for a, q in self.new_sent.items() if q}}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, path)

    def load(self, path: Path, now: float) -> int:
        if not path.exists():
            return 0
        d = json.loads(path.read_text())
        n = 0
        for e in d.get("active", []):
            ep = Episode(**e)
            if now - ep.last_msg_ts <= self.cfg.ttl_s:
                self.active[(ep.asset, ep.direction)] = ep
                n += 1
        for a, dr, ts, mv in d.get("closed", []):
            if now - ts < self.cfg.new_cooldown_s:
                self.closed[(a, int(dr))] = (ts, mv)
        for a, q in d.get("sent", {}).items():
            self.sent[a] = deque(t for t in q if now - t <= 86400)
        for a, q in d.get("new_sent", {}).items():
            self.new_sent[a] = deque(t for t in q if now - t <= 86400)
        return n


class BurstGovernor:
    """Delivery-level cap for market-wide days: at most max_new NEW alerts per window_s
    go out individually; further NEW alerts are held and released as ONE digest every
    digest_every_s. ESCALATE/REVERSAL (continuations of announced episodes) always pass.
    Held alerts are still recorded (JSONL) -- the governor changes delivery, not detection.
    2026-08-22 replay: 185 messages, 105 of them within one 10-minute window."""

    def __init__(self, window_s: int = 600, max_new: int = 6, digest_every_s: int = 300):
        self.window_s, self.max_new, self.digest_every_s = window_s, max_new, digest_every_s
        self.new_ts: deque = deque()
        self.held: list[Alert] = []
        self.first_held_ts: float | None = None

    def admit(self, al: Alert) -> bool:
        if al.kind != "NEW":
            return True
        while self.new_ts and al.ts - self.new_ts[0] > self.window_s:
            self.new_ts.popleft()
        if len(self.new_ts) < self.max_new:
            self.new_ts.append(al.ts)
            return True
        if not self.held:
            self.first_held_ts = al.ts
        self.held.append(al)
        return False

    def flush(self, now: float, force: bool = False) -> list[Alert]:
        """Held alerts to publish as one digest (empty if not due)."""
        if self.held and (force or now - self.first_held_ts >= self.digest_every_s):
            out, self.held, self.first_held_ts = self.held, [], None
            return out
        return []


def govern(alerts: list[Alert], **kw) -> tuple[int, int]:
    """Replay helper: (individual messages delivered, digest messages) for a time-sorted
    alert list under BurstGovernor -- same class the live engine uses."""
    g = BurstGovernor(**kw)
    n_msg = n_dig = 0
    for a in alerts:
        if g.flush(a.ts):
            n_dig += 1
        if g.admit(a):
            n_msg += 1
    if g.flush(float("inf"), force=True):
        n_dig += 1
    return n_msg, n_dig
