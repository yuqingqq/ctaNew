"""Live engine: feeds -> BarStore -> per-asset triggers (signals) -> episodes (alerts)
-> Notifier.

Once per second:
  1. store.tick(now)                       finalize seconds / minutes
  2. per-instrument log returns at 1m..24h (vectorized)
  3. market median per horizon over each asset's primary (most liquid) instrument
  4. candidate assets = any member instrument with |r - m| >= floor at some horizon
  5. asset aggregate (members = fresh, non-alias, not-down instruments with >= 5% of
     the top member's 24h volume):
       r      = 24h-volume-weighted mean of member returns
       sigma  = weighted mean of member sigma_1m
       vm     = sum member USD volume over h / sum member baseline (vol_1m * h/60)
       liq    = 24h USD volume summed over ALL members
     fire() per horizon; the fired horizon with the largest |r| sets the anchor
  6. AlertEngine.on_trigger / on_price (active episodes, reference instrument price)
  7. market-wide alert when |m_h| >= market_floor (one message per horizon/direction/2h)
Every refresh_min: baselines, cross-venue price consistency, asset membership.
Every 5 min: state snapshot (minute rings) for restarts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import signal
import time
from pathlib import Path

import aiohttp
import numpy as np

from .alerts import Alert, AlertCfg, AlertEngine, BurstGovernor
from .instruments import FX
from .notify import Notifier, esc, fmt_pct, fmt_px, fmt_usd, hhmm
from .signals import DEFAULT_RULES, H_NAME, HORIZONS, Rules, fire, market_median, pct
from .store import BarStore
from .venues import build_feeds
from .watchers import ListingWatcher

log = logging.getLogger("movewatch.engine")


class Engine:
    def __init__(self, cfg: dict, root: Path):
        self.cfg = cfg
        self.root = root
        self.fx = FX()
        self.store = BarStore(self.fx)
        self.rules = Rules.from_dict(cfg["rules"]) if cfg.get("rules") else DEFAULT_RULES
        self.alerts = AlertEngine(AlertCfg.from_dict(cfg.get("alerts")))
        self.governor = BurstGovernor(**(cfg.get("burst") or {}))
        self.last_m: dict = {}
        data_dir = root / cfg.get("data_dir", "data")
        self.state_path = data_dir / "state" / "store.npz"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.alert_state_path = self.state_path.parent / "alerts_state.json"
        self.notifier = Notifier(cfg.get("notify", {}), data_dir / "alerts")
        self.refresh_s = 60 * int(cfg.get("baseline_refresh_min", 15))
        self.fresh_ms = 1000 * int(cfg.get("fresh_s", 120))
        self.warmup_s = int(cfg.get("warmup_s", 90))
        self.feeds = []
        self.members: dict[str, np.ndarray] = {}
        self.primary: dict[str, int] = {}
        self._mkt_last: dict[tuple[int, int], float] = {}
        self.started = time.time()
        self.n_triggers = 0
        self.stop = asyncio.Event()
        self.venue_down_since: dict[str, float] = {}
        self.venue_down_notified: set[str] = set()
        self.venue_down_after_s = float(cfg.get("venue_down_alert_s", 600))
        self.mem_warn_mb = float(cfg.get("mem_warn_mb", 1024))
        self.mem_warned = False

    # ------------------------------------------------------------------ lifecycle
    async def run(self, duration_s: float | None = None) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self.stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        now = int(time.time())
        self.store.tick(now)
        async with aiohttp.ClientSession(headers={"User-Agent": "movewatch/0.1"}) as session:
            self.feeds = await build_feeds(self.cfg, self.store, session)
            tasks = []
            for f in self.feeds:
                try:
                    tasks += await f.start(session)
                except Exception as e:  # noqa: BLE001 -- a venue failing discovery must not stop the rest
                    log.error("%s: discovery failed: %s", f.venue, e)
                    self.notifier.emit(f"⚠️ movewatch: {esc(f.venue)} discovery failed: {esc(str(e)[:150])}",
                                       {"kind": "FEED", "venue": f.venue})
                await asyncio.sleep(0.5)
            restored = self.store.load(self.state_path)
            n_ep = self.alerts.load(self.alert_state_path, time.time())
            log.info("restored minute history for %d instruments (+%d pending), %d active episodes",
                     restored, len(self.store.pending_restore), n_ep)
            tasks.append(asyncio.create_task(self.notifier.run(session), name="notifier"))
            if self.cfg.get("watchers", {}).get("enabled", True):
                w = ListingWatcher(self.cfg.get("watchers", {}), self)
                tasks.append(asyncio.create_task(w.run(session), name="watchers"))
            self.notifier.emit(self._startup_text(), {"kind": "INFO"})
            try:
                await self._loop(duration_s)
            finally:
                self.store.save(self.state_path)
                self.alerts.save(self.alert_state_path)
                log.info("state saved; stopping")
                for t in tasks:
                    t.cancel()
                # let the notifier flush what is already queued (bounded)
                try:
                    await asyncio.wait_for(self._drain(session), timeout=10)
                except asyncio.TimeoutError:
                    pass

    async def _drain(self, session) -> None:
        while not self.notifier.q.empty():
            await self.notifier._send(session, self.notifier.q.get_nowait())

    def _startup_text(self) -> str:
        v = ", ".join(f"{f.venue}" for f in self.feeds)
        return (f"🟦 movewatch started · {self.store.n} instruments · venues: {esc(v)}\n"
                f"delivery: {esc(self.notifier.status())}")

    async def _loop(self, duration_s: float | None) -> None:
        self.started = time.time()          # warm-up and --duration count from live feeds
        last_refresh = 0.0
        last_save = time.time()
        last_health = time.time()
        while (duration_s is None or time.time() - self.started < duration_s) and not self.stop.is_set():
            await asyncio.sleep(1.0 - (time.time() % 1.0) + 0.01)
            now = time.time()
            self.store.tick(int(now))
            if now - last_refresh >= self.refresh_s or not self.members:
                self._refresh()
                last_refresh = now
            if now - self.started >= self.warmup_s:
                try:
                    self.evaluate(now)
                except Exception:  # noqa: BLE001 -- never let one bad tick stop detection
                    log.exception("evaluate failed")
            held = self.governor.flush(now)
            if held:
                self._digest(held, now)
            if now - last_save >= 300:
                self.store.save(self.state_path)
                self.alerts.save(self.alert_state_path)
                last_save = now
            if now - last_health >= 60:
                self._health(now)
                last_health = now

    # ------------------------------------------------------------------ periodic
    def _refresh(self) -> None:
        st = self.store
        st.refresh_baselines()
        for key, asset, usd, _ in st.refresh_consistency(float(self.cfg.get("alias_max_log_dev", 0.4))):
            log.info("PRICE_CLUSTER %s -> asset %s (usd=%.6g)", key, asset, usd)
        by: dict[str, list[int]] = {}
        names = st.asset_of
        for i, ins in enumerate(st.insts[: st.n]):
            by.setdefault(names[i] if i < len(names) else ins.base, []).append(i)
        self.members = {b: np.array(ids) for b, ids in by.items()}
        vol = st.vol24h[: st.n]
        self.primary = {b: int(ids[np.argmax(vol[ids])]) for b, ids in self.members.items()}

    def _health(self, now: float) -> None:
        feeds = {f.venue: f.stats() for f in self.feeds}
        rss_mb, hwm_mb = _rss_mb()
        if rss_mb > self.mem_warn_mb and not self.mem_warned:
            self.mem_warned = True
            self.notifier.emit(f"⚠️ movewatch memory {rss_mb:.0f} MB > soft limit {self.mem_warn_mb:.0f} MB "
                               f"(systemd MemoryMax is the hard cap)", {"kind": "HEALTH", "rss_mb": rss_mb})
        h = {"t": int(now), "rss_mb": round(rss_mb, 1), "peak_rss_mb": round(hwm_mb, 1),
             "instruments": self.store.n, "capacity": self.store.cap, "assets": len(self.members),
             "active_episodes": len(self.alerts.active), "krw_live": self.fx.krw_live,
             "krw_per_usd": round(self.fx.krw_per_usd, 2), "sent": self.notifier.sent,
             "send_failed": self.notifier.failed, "feeds": feeds}
        log.info("HEALTH %s", json.dumps(h, separators=(",", ":")))
        for f in self.feeds:
            s = feeds[f.venue]
            down = s["shards"] > 0 and s["up"] == 0
            if down:
                t0 = self.venue_down_since.setdefault(f.venue, now)
                if now - t0 >= self.venue_down_after_s and f.venue not in self.venue_down_notified:
                    self.venue_down_notified.add(f.venue)
                    self.notifier.emit(f"🔌 feed <b>{esc(f.venue)}</b> down since {hhmm(t0)} "
                                       f"(all {s['shards']} connections); its moves are not being watched",
                                       {"kind": "FEED", "venue": f.venue, "state": "down"})
            else:
                if f.venue in self.venue_down_notified:
                    self.notifier.emit(f"✅ feed <b>{esc(f.venue)}</b> recovered "
                                       f"(down {int((now - self.venue_down_since[f.venue]) / 60)} min)",
                                       {"kind": "FEED", "venue": f.venue, "state": "up"})
                    self.venue_down_notified.discard(f.venue)
                self.venue_down_since.pop(f.venue, None)
        hp = self.state_path.parent / "health.json"
        hp.write_text(json.dumps(h, indent=1))

    # ------------------------------------------------------------------ detection
    def evaluate(self, now: float) -> None:
        st = self.store
        n = st.n
        if n == 0 or not self.members:
            return
        R = st.returns(list(HORIZONS))
        now_ms = now * 1000
        ok = ((now_ms - st.last_ms[:n]) < self.fresh_ms) & ~st.alias_bad[:n] & ~st.down[:n]
        prim = np.array([i for i in self.primary.values() if i < n])
        liq_prim = st.vol24h[prim]
        m = {h: market_median(np.where(ok[prim], R[h][prim], np.nan), liq_prim, self.rules.market_liq_usd)
             for h in HORIZONS}
        self.last_m = m
        cand: set[str] = set()
        for h, rule in self.rules.horizons.items():
            with np.errstate(invalid="ignore"):
                hit = np.flatnonzero(ok & (np.abs(R[h] - m[h]) >= rule.floor))
            names = st.asset_of
            for i in hit:
                cand.add(names[i] if i < len(names) else st.insts[i].base)
        for asset in cand:
            a = self._asset(asset, R, m, ok)
            if a is None or not a["fired"]:
                continue
            h, r, e, z, vm = max(a["fired"], key=lambda x: abs(x[1]))
            self.n_triggers += 1
            metrics = dict(a, horizon=H_NAME[h], e=e, z=z, vm=vm, mkt=m, fired=[H_NAME[x[0]] for x in a["fired"]])
            for al in self.alerts.on_trigger(asset, now, a["ref_usd"], int(np.sign(e)), r, H_NAME[h],
                                             metrics=metrics, ref_key=a["ref_key"]):
                self._deliver(al)
        for (asset, d), ep in list(self.alerts.active.items()):
            px = self._ref_price(asset, ep.ref_key, ok)
            if px is None:
                continue
            for al in self.alerts.on_price(asset, now, px):
                if al.kind in ("ESCALATE", "REVERSAL"):
                    a = self._asset(asset, R, m, ok)
                    if a:
                        al.metrics = dict(a, mkt=m)
                self._deliver(al)
        self._market(now, m, R, ok)

    def _asset(self, asset: str, R: dict, m: dict, ok: np.ndarray) -> dict | None:
        st = self.store
        ids = self.members.get(asset)
        if ids is None:
            return None
        ids = ids[ids < st.n]
        mem = ids[ok[ids]]
        if mem.size == 0:
            return None
        vol = st.vol24h[mem]
        top = vol.max()
        if top > 0:
            mem = mem[vol >= 0.05 * top]
        w = st.vol24h[mem] + 1.0
        liq = float(st.vol24h[ids].sum())
        sig_ok = np.isfinite(st.sigma1m[mem])
        sigma = float(np.average(st.sigma1m[mem][sig_ok], weights=w[sig_ok])) if sig_ok.any() else float("nan")
        v_ok = np.isfinite(st.vol1m[mem])
        v1m_sum = float(st.vol1m[mem][v_ok].sum()) if v_ok.any() else float("nan")
        ref = int(mem[np.argmax(st.vol24h[mem])])
        ref_ins = st.insts[ref]
        ref_usd = st.last_px[ref] / ref_ins.mult * self.fx.to_usd(ref_ins.quote)
        rets, fired = {}, []
        for h in HORIZONS:
            ri = R[h][mem]
            valid = np.isfinite(ri)
            if not valid.any():
                continue
            r = float(np.average(ri[valid], weights=w[valid]))
            rets[H_NAME[h]] = r
            rule = self.rules.horizons.get(h)
            if rule is None:
                continue
            e = r - m[h]
            if abs(e) < rule.floor:
                continue
            z = e / (sigma * math.sqrt(h / 60.0)) if sigma == sigma else float("nan")
            volh = float(st.volume_usd(mem[v_ok], h).sum()) if v_ok.any() else float("nan")
            vm = volh / (v1m_sum * h / 60.0) if v1m_sum == v1m_sum and v1m_sum > 0 else float("nan")
            if fire(np.array([e]), np.array([z]), np.array([vm]), np.array([liq]), rule,
                    self.rules.liq_min_usd)[0]:
                fired.append((h, r, e, z, vm))
        # venue breakdown on the longest fired horizon (or 1h)
        hb = max((f[0] for f in fired), default=3600)
        venues = []
        for i in mem:
            ri = R[hb][i]
            if np.isfinite(ri):
                venues.append((st.insts[i].venue, float(ri)))
        venues.sort(key=lambda x: -abs(x[1]))
        vol_1h = float(st.volume_usd(mem, 3600).sum())
        funding = None
        fz = self._funding_src()
        if fz is not None:
            for i in ids:
                ins = st.insts[i]
                if ins.venue == "binance_usdm" and ins.symbol in fz.funding:
                    funding = fz.funding[ins.symbol]
                    break
        return {"asset": asset, "fired": fired, "rets": rets, "sigma": sigma, "liq": liq,
                "ref_key": ref_ins.key, "ref_usd": float(ref_usd), "venues": venues,
                "vol_1h": vol_1h, "n_members": int(mem.size), "hb": H_NAME[hb], "funding": funding}

    def _funding_src(self):
        for f in self.feeds:
            if f.venue == "binance_usdm" and hasattr(f, "funding"):
                return f
        return None

    def _ref_price(self, asset: str, ref_key: str | None, ok: np.ndarray) -> float | None:
        st = self.store
        i = st.index.get(ref_key) if ref_key else None
        if i is None or not ok[i]:
            i = self.primary.get(asset)
            if i is None or i >= st.n or not ok[i]:
                return None
        ins = st.insts[i]
        px = st.last_px[i] / ins.mult * self.fx.to_usd(ins.quote)
        return float(px) if px > 0 else None

    def _market(self, now: float, m: dict, R: dict, ok: np.ndarray) -> None:
        for h, floor_pct in self.rules.market_floor_pct.items():
            mv = pct(m[h])
            if abs(mv) < floor_pct:
                continue
            d = 1 if mv > 0 else -1
            if now - self._mkt_last.get((h, d), 0) < 7200:
                continue
            self._mkt_last[(h, d)] = now
            btc = self._asset("BTC", R, m, ok) or {}
            eth = self._asset("ETH", R, m, ok) or {}
            hn = H_NAME[h]
            text = (f"🌐 <b>MARKET</b> {hn} median {fmt_pct(mv)} across liquid coins\n"
                    f"BTC {hn} {fmt_pct(pct(btc.get('rets', {}).get(hn, float('nan'))))} · "
                    f"ETH {hn} {fmt_pct(pct(eth.get('rets', {}).get(hn, float('nan'))))}\n"
                    f"single-coin alerts are measured net of this move")
            self.notifier.emit(text, {"kind": "MARKET", "h": hn, "median_pct": mv, "ts": now})

    def describe(self, asset: str, key: str | None = None) -> str:
        """One-line market context for listing/flag alerts. `key` (venue:symbol) resolves
        the exact price cluster when a ticker is shared by several assets."""
        st = self.store
        if key is not None and key in st.index and st.index[key] < len(st.asset_of):
            asset = st.asset_of[st.index[key]]
        if st.t is None or asset not in self.members:
            return "not traded on the watched venues yet"
        R = st.returns([3600, 86400])
        n = st.n
        ok = ((time.time() * 1000 - st.last_ms[:n]) < self.fresh_ms) & ~st.alias_bad[:n] & ~st.down[:n]
        ids = self.members[asset]
        ids = ids[(ids < n)]
        live = ids[ok[ids]]
        if live.size == 0:
            return "watched venues have no fresh print"
        ref = int(live[np.argmax(st.vol24h[live])])
        ins = st.insts[ref]
        px = st.last_px[ref] / ins.mult * self.fx.to_usd(ins.quote)
        return (f"elsewhere: {fmt_px(px)} ({esc(ins.venue)}) · 1h {fmt_pct(pct(R[3600][ref]))}"
                f" · 24h {fmt_pct(pct(R[86400][ref]))} · 24h vol {fmt_usd(float(st.vol24h[ids].sum()))}"
                f" · {live.size} venues")

    # ------------------------------------------------------------------ messages
    def _deliver(self, al: Alert) -> None:
        rec = {"kind": al.kind, "alert": al}
        if self.governor.admit(al):
            self.notifier.emit(format_alert(al), rec)
        else:
            self.notifier.record_only(format_alert(al), dict(rec, held_for_digest=True))

    def _digest(self, held: list, now: float) -> None:
        """One message for NEW alerts held by the burst governor (market-wide days)."""
        m = self.last_m or {}
        top = sorted(held, key=lambda a: -abs(a.move_pct))[:10]
        lines = [f"🌊 <b>BURST</b>: {len(held)} more coins triggered since {hhmm(held[0].ts, now)}"
                 f" · mkt 15m {fmt_pct(pct(m.get(900, 0.0)))} · 1h {fmt_pct(pct(m.get(3600, 0.0)))}"]
        for a in top:
            lines.append(f"{'🟢' if a.direction > 0 else '🔴'} {esc(a.asset)} {fmt_pct(a.move_pct)}"
                         f" ({esc(a.horizon or '')}) {fmt_px(a.price)}")
        if len(held) > len(top):
            lines.append(f"… +{len(held) - len(top)} more in data/alerts/")
        self.notifier.emit("\n".join(lines), {"kind": "DIGEST", "n": len(held),
                                              "assets": [a.asset for a in held]})


def _rss_mb() -> tuple[float, float]:
    """(current RSS, peak RSS) of this process in MB, from /proc."""
    d = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS", "VmHWM")):
                    d[line.split(":")[0]] = int(line.split()[1]) / 1024.0
    except OSError:
        return float("nan"), float("nan")
    return d.get("VmRSS", float("nan")), d.get("VmHWM", float("nan"))


def format_alert(al: Alert) -> str:
    m = al.metrics or {}
    up = al.direction > 0
    rets = m.get("rets", {})
    line_r = " · ".join(f"{k} {fmt_pct(pct(v))}" for k, v in rets.items())
    venues = m.get("venues", [])
    agree = sum(1 for _, r in venues if r * al.direction > 0)
    lead = venues[0] if venues else None
    mk = m.get("mkt", {})
    mk1h = pct(mk.get(3600, 0.0)) if isinstance(mk, dict) else float("nan")
    tag = "🟢" if up else "🔴"
    if al.kind == "NEW":
        head = f"{tag} <b>{esc(al.asset)}</b> {fmt_pct(pct(m.get('rets', {}).get(al.horizon, 0.0)))} in {esc(al.horizon)}"
        sub = (f"{fmt_px(al.price)} · from {fmt_px(al.anchor_px)} ({hhmm(al.anchor_ts, al.ts)})"
               f" · z {m.get('z', float('nan')):.1f} · vol ×{m.get('vm', float('nan')):.0f}")
    elif al.kind == "ESCALATE":
        head = f"{'🚀' if up else '📉'} <b>{esc(al.asset)}</b> {fmt_pct(al.move_pct, 0)} from {fmt_px(al.anchor_px)} ({hhmm(al.anchor_ts, al.ts)})"
        sub = f"now {fmt_px(al.price)} · step {al.seq}"
    else:  # REVERSAL
        off = (al.price / al.extreme_px - 1) * 100 if al.extreme_px else float("nan")
        head = f"↩️ <b>{esc(al.asset)}</b> {fmt_pct(off)} off the {'peak' if up else 'low'} {fmt_px(al.extreme_px)}"
        size = (al.extreme_px / al.anchor_px - 1) * 100 if al.extreme_px else float("nan")
        sub = f"now {fmt_px(al.price)} · the {'pump' if up else 'dump'} was {fmt_pct(size, 0)} from {fmt_px(al.anchor_px)}"
    lines = [head, sub]
    if line_r:
        lines.append(esc(line_r))
    vparts = []
    if venues:
        vparts.append(f"{agree}/{len(venues)} venues agree")
        if lead:
            vparts.append(f"lead {esc(lead[0])} {fmt_pct(pct(lead[1]))} ({esc(m.get('hb', ''))})")
    if m.get("vol_1h") is not None:
        vparts.append(f"1h vol {fmt_usd(m.get('vol_1h'))}")
    if vparts:
        lines.append(" · ".join(vparts))
    tail = []
    if mk1h == mk1h:
        tail.append(f"mkt 1h {fmt_pct(mk1h)}")
    fr = m.get("funding")
    if fr is not None:
        tail.append(f"Binance funding {fr * 100:+.4f}%" + (" (shorts paying)" if fr < 0 else ""))
    if tail:
        lines.append(" · ".join(tail))
    return "\n".join(lines)
