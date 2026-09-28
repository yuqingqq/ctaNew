"""Per-instrument ring buffers on a global wall clock.

  sec ring  : S seconds (default 900) of last price + USD quote volume
  min ring  : M minutes (default 1500 = 25 h) of close + USD quote volume

Venue adapters call on_trade()/on_candle(); the engine calls tick(now_s) once a
second, which finalizes every whole second since the last tick (catch-up after
an event-loop stall writes carry-forward prices and zero volume, with all accrued
volume in the final second).

Outage rule: while a venue is marked down, its instruments' finalized prices are
NaN, not carried forward. Otherwise the first print after a reconnect would read
as a sudden move over every horizon that spans the outage.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .instruments import FX, Instrument
from .signals import baselines_from_minutes


class BarStore:
    def __init__(self, fx: FX, sec_len: int = 900, min_len: int = 1500, cap: int = 4096):
        self.fx = fx
        self.S, self.M = sec_len, min_len
        self.insts: list[Instrument] = []
        self.index: dict[str, int] = {}
        self._alloc(cap)
        self.t: int | None = None          # last finalized second
        self.pending_restore: dict[str, tuple] = {}   # snapshot rows for not-yet-registered keys
        self.asset_of: list[str] = []                 # asset name per instrument (price clusters)
        # baselines (recomputed by refresh_baselines)
        self.sigma1m = np.full(cap, np.nan)
        self.vol1m = np.full(cap, np.nan)
        self.vol24h = np.zeros(cap)
        self.n_hist = np.zeros(cap, dtype=np.int32)
        self.alias_bad = np.zeros(cap, dtype=bool)
        self.usd_px = np.full(cap, np.nan)

    # ------------------------------------------------------------------ registry
    def _alloc(self, cap: int) -> None:
        old = getattr(self, "cap", 0)
        def grow(name, shape, fill, dtype):
            new = np.full(shape, fill, dtype=dtype)
            if old:
                cur = getattr(self, name)
                new[: cur.shape[0]] = cur
            setattr(self, name, new)
        # float32 prices: ~1e-7 relative precision, vs a 3% smallest trigger; halves the rings
        grow("sec_px", (cap, self.S), np.nan, np.float32)
        grow("sec_qv", (cap, self.S), 0.0, np.float32)
        grow("min_px", (cap, self.M), np.nan, np.float32)
        grow("min_qv", (cap, self.M), 0.0, np.float32)
        grow("last_px", (cap,), np.nan, np.float64)
        grow("acc_qv", (cap,), 0.0, np.float64)
        grow("last_ms", (cap,), 0, np.int64)
        grow("down", (cap,), False, bool)
        for name, fill, dt in (("sigma1m", np.nan, np.float64), ("vol1m", np.nan, np.float64),
                               ("vol24h", 0.0, np.float64), ("n_hist", 0, np.int32),
                               ("alias_bad", False, bool), ("usd_px", np.nan, np.float64)):
            if old:
                grow(name, (cap,), fill, dt)
        self.cap = cap

    def register(self, inst: Instrument) -> int:
        k = inst.key
        if k in self.index:
            return self.index[k]
        n = len(self.insts)
        if n >= self.cap:
            self._alloc(max(self.cap + 512, int(self.cap * 1.25)))   # gentle growth, not doubling
        self.insts.append(inst)
        self.index[k] = n
        pend = self.pending_restore.pop(k, None)
        if pend is not None:
            vpos, px, qv = pend
            self.min_px[n, vpos] = px
            self.min_qv[n, vpos] = qv
        return n

    @property
    def n(self) -> int:
        return len(self.insts)

    # ------------------------------------------------------------------ ingest
    def on_trade(self, idx: int, price: float, quote_qty: float, ts_ms: int | None = None) -> None:
        """One trade (or an aggregated increment): native price, native quote quantity."""
        if not (price > 0):
            return
        self.last_px[idx] = price
        if quote_qty > 0:
            self.acc_qv[idx] += quote_qty * self.fx.to_usd(self.insts[idx].quote)
        self.last_ms[idx] = ts_ms or int(time.time() * 1000)

    def backfill_minute(self, idx: int, minute_open_ms: int, close: float, quote_vol: float) -> None:
        """Warm-up from a venue's candle history (e.g. Bitget subscribe snapshot or a
        persisted state file). Only fills empty slots of closed minutes in range."""
        if self.t is None:
            return
        m = minute_open_ms // 60_000
        cur_m = (self.t + 1) // 60
        if not (cur_m - self.M < m < cur_m):
            return
        pos = m % self.M
        if np.isnan(self.min_px[idx, pos]):
            self.min_px[idx, pos] = close
            self.min_qv[idx, pos] = quote_vol * self.fx.to_usd(self.insts[idx].quote)

    def set_down(self, idx, flag: bool) -> None:
        """A shard (one WS connection) lost / regained its feed. While down, its
        instruments finalize NaN prices (see module docstring)."""
        if len(idx):
            self.down[np.asarray(idx, dtype=np.int64)] = flag

    # ------------------------------------------------------------------ clock
    def tick(self, now_s: int) -> list[int]:
        """Finalize whole seconds up to now_s-1. Returns the list of minutes closed."""
        closed: list[int] = []
        if self.t is None:
            self.t = now_s - 1
            return closed
        n = self.n
        if n == 0:
            self.t = now_s - 1
            return closed
        down = self.down[:n] if self.down[:n].any() else None
        first, last = self.t + 1, now_s - 1
        if last - first > self.S:                         # stalled longer than the ring
            first = last - self.S + 1
        for s in range(first, last + 1):
            pos = s % self.S
            px = self.last_px[:n].copy()
            if down is not None:
                px[down] = np.nan
            self.sec_px[:n, pos] = px
            if s == last:
                self.sec_qv[:n, pos] = self.acc_qv[:n]
                self.acc_qv[:n] = 0.0
            else:
                self.sec_qv[:n, pos] = 0.0
            if s % 60 == 59:                              # minute (s // 60) complete
                m = s // 60
                mpos = m % self.M
                idx = [(s - k) % self.S for k in range(59, -1, -1)]
                self.min_px[:n, mpos] = self.sec_px[:n, pos]
                self.min_qv[:n, mpos] = self.sec_qv[:n, idx].sum(axis=1)
                closed.append(m)
        self.t = last
        return closed

    # ------------------------------------------------------------------ views
    def price_ago_sec(self, h: int) -> np.ndarray:
        return self.sec_px[: self.n, (self.t - h) % self.S]

    def price_ago_min(self, h_min: int) -> np.ndarray:
        """Close of the minute that ended h_min minutes before the current minute began."""
        cur_m = (self.t + 1) // 60
        return self.min_px[: self.n, (cur_m - h_min) % self.M]

    def returns(self, horizons_s: list[int]) -> dict[int, np.ndarray]:
        """Log returns for all instruments: now (last print) vs h seconds ago."""
        now = self.last_px[: self.n]
        out = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for h in horizons_s:
                past = self.price_ago_sec(h) if h < self.S else self.price_ago_min(h // 60)
                out[h] = np.log(now / past)
        return out

    def volume_usd(self, idx: np.ndarray, h: int) -> np.ndarray:
        """USD quote volume over the last h seconds for the given instruments
        (includes the not-yet-finalized current second)."""
        if h <= self.S:
            pos = [(self.t - k) % self.S for k in range(h - 1)]
            v = self.sec_qv[idx][:, pos].sum(axis=1)
        else:
            cur_m = (self.t + 1) // 60
            h_min = h // 60
            mpos = [(cur_m - k) % self.M for k in range(1, h_min)]
            v = self.min_qv[idx][:, mpos].sum(axis=1)
            sec_in_cur_min = (self.t + 1) % 60
            spos = [(self.t - k) % self.S for k in range(sec_in_cur_min)]
            if spos:
                v = v + self.sec_qv[idx][:, spos].sum(axis=1)
        return v + self.acc_qv[idx]

    # ------------------------------------------------------------------ baselines
    def refresh_baselines(self, lookback_min: int = 1440, exclude_recent_min: int = 60,
                          min_hist: int = 120, chunk: int = 1024) -> None:
        """sigma1m / vol1m from the minute ring, excluding the most recent hour so an
        ongoing move does not raise its own bar. Rows in chunks of `chunk` so the
        float64 temporaries stay ~10 MB each regardless of instrument count."""
        n = self.n
        if n == 0 or self.t is None:
            return
        cur_m = (self.t + 1) // 60
        lb = min(lookback_min, self.M - 1)
        pos = np.arange(cur_m - lb, cur_m - exclude_recent_min) % self.M
        all24 = np.arange(cur_m - min(1440, self.M - 1), cur_m) % self.M
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            C = self.min_px[a:b][:, pos].astype(np.float64)
            V = self.min_qv[a:b][:, pos].astype(np.float64)
            sig, v1m, nh = baselines_from_minutes(C, V, min_hist=min_hist)
            self.sigma1m[a:b], self.vol1m[a:b], self.n_hist[a:b] = sig, v1m, nh
            del C, V
            # 24h USD volume, normalized by how long the instrument has been observed
            # (floor 60 min): after a cold start only Bitget has a backfilled day, and raw
            # sums would make every other venue look "thin" and drop out of aggregates.
            raw = self.min_qv[a:b][:, all24].sum(axis=1, dtype=np.float64)
            seen = np.isfinite(self.min_px[a:b][:, all24]).sum(axis=1)
            self.vol24h[a:b] = raw * len(all24) / np.maximum(seen, 60)

    def refresh_consistency(self, max_dev: float = 0.40) -> list[tuple[str, str, float, float]]:
        """Split each ticker into PRICE CLUSTERS: same ticker, different asset.

        Instruments of one base are sorted by log USD price and cut wherever
        consecutive prices differ by more than max_dev (log). The cluster with the
        most 24h volume keeps the plain name ("META"); the others become
        "<BASE>@<venue of their most liquid member>" ("META@upbit") and are watched
        as separate assets -- a Korean-only coin that shares a ticker with a global
        one is a real asset, not an outlier to drop. Instruments with no price yet
        keep the plain name. Returns (key, asset, usd_px, main_cluster_px) for
        instruments whose asset assignment changed.
        """
        n = self.n
        usd = np.full(n, np.nan)
        for i, ins in enumerate(self.insts[:n]):
            px = self.last_px[i]
            if px > 0:
                usd[i] = px / ins.mult * self.fx.to_usd(ins.quote)
        self.usd_px[:n] = usd
        by_base: dict[str, list[int]] = {}
        for i, ins in enumerate(self.insts[:n]):
            by_base.setdefault(ins.base, []).append(i)
        new_asset = [ins.base for ins in self.insts[:n]]
        changed = []
        for base, ids in by_base.items():
            priced = [i for i in ids if usd[i] > 0]
            if len(priced) < 2:
                continue
            priced.sort(key=lambda i: usd[i])
            lp = np.log(usd[priced])
            cuts = np.flatnonzero(np.diff(lp) > max_dev) + 1
            if cuts.size == 0:
                continue
            clusters = np.split(np.array(priced), cuts)
            vol = [float(self.vol24h[c].sum()) + 1e-9 * len(c) for c in clusters]
            main = int(np.argmax(vol))
            main_px = float(np.exp(np.median(np.log(usd[clusters[main]]))))
            for k, c in enumerate(clusters):
                if k == main:
                    continue
                top = int(c[np.argmax(self.vol24h[c])])
                name = f"{base}@{self.insts[top].venue.split('_')[0]}"
                for i in c:
                    new_asset[i] = name
        for i in range(n):
            if i < len(self.asset_of) and self.asset_of[i] != new_asset[i]:
                changed.append((self.insts[i].key, new_asset[i], float(usd[i]),
                                float(np.nan)))
            elif i >= len(self.asset_of) and new_asset[i] != self.insts[i].base:
                changed.append((self.insts[i].key, new_asset[i], float(usd[i]), float(np.nan)))
        self.asset_of = new_asset
        self.alias_bad[:n] = False   # clusters replace exclusion; kept for API stability
        return changed

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        """Atomic snapshot of the minute ring + registry for fast restarts."""
        if self.t is None or self.n == 0:
            return
        n = self.n
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, min_px=self.min_px[:n], min_qv=self.min_qv[:n], t=np.array([self.t]),
            keys=np.array([i.key for i in self.insts[:n]]),
        )
        os.replace(tmp, path)

    def load(self, path: Path, max_age_s: int = 6 * 3600) -> int:
        """Restore minute history for instruments already registered. Returns the
        number restored. Minute positions are absolute (m % M), so a snapshot
        from < M minutes ago lines up without shifting; the gap is NaN."""
        if not path.exists() or self.t is None:
            return 0
        with np.load(path, allow_pickle=False) as z:
            t_saved = int(z["t"][0])
            if self.t - t_saved > max_age_s:
                return 0
            # read each array ONCE: every z[...] access decompresses the whole array again
            # (inside the per-key loop that was ~20 min at 100% CPU for 4.4k instruments)
            keys, min_px, min_qv = z["keys"], z["min_px"], z["min_qv"]
        restored = 0
        cur_m = (self.t + 1) // 60
        saved_m = (t_saved + 1) // 60
        valid_ms = np.arange(max(saved_m - self.M + 1, cur_m - self.M + 1), saved_m)
        vpos = valid_ms % self.M
        for j, key in enumerate(keys):
            i = self.index.get(str(key))
            if i is None:  # registered later (lazy venues): applied in register()
                self.pending_restore[str(key)] = (vpos, min_px[j, vpos], min_qv[j, vpos])
                continue
            self.min_px[i, vpos] = min_px[j, vpos]
            self.min_qv[i, vpos] = min_qv[j, vpos]
            restored += 1
        return restored

    def health(self) -> dict:
        now_ms = int(time.time() * 1000)
        by_venue: dict[str, dict] = {}
        for i, ins in enumerate(self.insts[: self.n]):
            d = by_venue.setdefault(ins.venue, {"n": 0, "fresh_60s": 0, "alias_bad": 0})
            d["n"] += 1
            if now_ms - self.last_ms[i] < 60_000:
                d["fresh_60s"] += 1
            if self.alias_bad[i]:
                d["alias_bad"] += 1
        return by_venue

    def dump_json(self) -> str:
        return json.dumps(self.health(), sort_keys=True)
