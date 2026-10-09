"""Live extreme-crash trigger (movewatch/watchers.py) vs the validated rule (docs/EXTREME_CRASH.md)."""
import math
from pathlib import Path

import numpy as np
import pytest

from movewatch.watchers import CrashDetector, CrashWatcher


def _close(det, minute, prices, qv=1e7):
    for s, p in prices.items():
        det.tick(s, p, qv, minute * 60 + 30)
    return det.close_minute(minute)


def _syms(n):
    return [f"C{i:03d}USDT" for i in range(n)]


def test_fires_on_planted_crash_with_hardest_hit_basket():
    det, syms = CrashDetector(), _syms(100)
    for m in range(16):
        assert not _close(det, m, {s: 10.0 for s in syms}).get("fire")
    crash = {s: (8.9 - 0.01 * i if i < 60 else 10.0) for i, s in enumerate(syms)}
    info = _close(det, 16, crash)
    assert info["fire"] and info["n"] == 100 and info["share10"] == pytest.approx(0.6)
    assert [s for s, _ in info["basket"]] == syms[59:39:-1]          # the 20 deepest drops, worst first


def test_refractory_24h():
    det, syms = CrashDetector(), _syms(100)
    flat, down = {s: 10.0 for s in syms}, {s: 8.5 for s in syms}
    fires = []
    for m in range(16 + 1441 + 16):
        info = _close(det, m, down if m in (16, 40, 16 + 1440) else flat)
        fires += [m] if info.get("fire") else []
    assert fires == [16, 16 + 1440]


def test_thin_universe_and_no_history():
    det, syms = CrashDetector(), _syms(40)
    assert _close(det, 0, {s: 10.0 for s in syms})["status"] == "NO_HISTORY"
    for m in range(1, 16):
        info = _close(det, m, {s: 10.0 for s in syms})
    assert info["status"] == "THIN_UNIVERSE"
    assert not _close(det, 16, {s: 5.0 for s in syms}).get("fire")


def test_exclusions_hold():
    det = CrashDetector(excluded=lambda b, px: b == "ENGTRADFI")
    good = _syms(100)
    bad = ["USDCUSDT", "BTCDOMUSDT", "TSLAUSDT", "BTCUSDT_251226", "ENGTRADFIUSDT", "C999USDC"]
    det.tick("STALEUSDT", 10.0, 1e7, 0.0)                           # traded once, long ago
    for m in range(16):
        det.tick("LOWVOLUSDT", 10.0, 1e6, m * 60 + 30)                # $1M / 24 h: below liq_min
        _close(det, m, {s: 10.0 for s in good + bad})
    det.tick("LOWVOLUSDT", 5.0, 1e6, 16 * 60 + 30)
    det.px["STALEUSDT"] = 5.0
    crash = {s: (8.5 if i < 45 else 10.0) for i, s in enumerate(good)} | {s: 5.0 for s in bad}
    info = _close(det, 16, crash)
    assert info["n"] == 100 and info["share10"] == pytest.approx(0.45) and not info["fire"]


class _Notifier:
    def __init__(self):
        self.sent = []

    def emit(self, text, record=None):
        self.sent.append((text, record))


class _Engine:
    tradfi_assets = set()
    tradfi_ref: dict = {}

    def __init__(self):
        self.notifier = _Notifier()


def _msg(minute, sec, prices):
    E = int((minute * 60 + sec) * 1000)
    return {"data": [{"e": "24hrMiniTicker", "E": E, "s": s, "c": str(p), "q": "20000000"} for s, p in prices.items()]}


def test_watcher_alert_and_follow_up():
    eng = _Engine()
    w = CrashWatcher({"hold_min": 5}, eng, url="unused")
    syms = _syms(100)
    base = 29_000_000                                                # an epoch minute
    for m in range(base, base + 16):
        w.on_message(_msg(m, 10, {s: 10.0 for s in syms}))
    w.on_message(_msg(base + 16, 10, {s: (8.0 if i < 70 else 10.0) for i, s in enumerate(syms)}))
    w.on_message(_msg(base + 17, 10, {s: (8.8 if i < 70 else 10.0) for i, s in enumerate(syms)}))   # closes 16
    assert len(eng.notifier.sent) == 1 and eng.notifier.sent[0][1]["kind"] == "CRASH"
    text, rec = eng.notifier.sent[0]
    assert "EXTREME CRASH" in text and len(rec["basket"]) == 20 and "C000" in text
    for m in range(base + 18, base + 24):                            # entry = close of 17 (8.8); exit at 17+5
        w.on_message(_msg(m, 10, {s: (9.68 if i < 70 else 10.0) for i, s in enumerate(syms)}))
    kinds = [r["kind"] for _, r in eng.notifier.sent]
    assert kinds == ["CRASH", "CRASH_FOLLOWUP"]
    assert eng.notifier.sent[1][1]["gross"] == pytest.approx(0.10)  # 8.8 -> 9.68


HIST = Path(__file__).resolve().parents[1] / "data" / "replay" / "panel_2023-12-30_2024-04-03"


@pytest.mark.skipif(not HIST.exists(), reason="hist panel not on disk")
def test_live_detector_reproduces_research_on_2024_03_05():
    """Replay the research panel's minute closes through the live detector: it must fire on the same
    minute as the research EXTREME fire (2024-03-05 19:56) with the same hardest-hit basket."""
    import pandas as pd
    import research.flush_signal as fs
    segs = fs._hist_segments()
    k = next(i for i, s in enumerate(segs) if s["panel"] == ("2023-12-30", "2024-04-03"))
    syms, idx, C, V = fs._load(segs[k])
    LIQ = np.load(fs.OUT / f"liq_hist_{k}.npy")
    keep = fs.static_keep(syms)
    t = int((pd.Timestamp("2024-03-05 19:56", tz="UTC") - idx[0]) / fs.MIN)
    u, r15 = fs.universe_at(C, V, LIQ, keep, t)
    want = {syms[i] for i in u[np.argsort(r15, kind="stable")[:math.ceil(len(u) / 5)]]}
    det = CrashDetector()
    fired = []
    for j in range(t - 16, t + 1):
        em = int(idx[j].timestamp() // 60)
        for i in np.flatnonzero(keep):
            c = float(C[i, j])
            if V[i, j] > 0 and math.isfinite(c):
                det.tick(syms[i], c, float(LIQ[i, j // 15]), em * 60 + 30)
        info = det.close_minute(em)
        if info.get("fire"):
            fired.append((j, info))
    assert [j for j, _ in fired] == [t]
    info = fired[0][1]
    assert info["n"] == len(u) and {s for s, _ in info["basket"]} == want


def test_watcher_excludes_stock_perps_by_reference_price():
    eng = _Engine()
    eng.tradfi_ref = {"MRNA": [25.0]}                                 # a venue-labelled stock at $25
    eng.tradfi_assets = {"ENGSTOCK"}
    w = CrashWatcher({}, eng, url="unused")
    assert w._excluded("MRNA", 26.0) and not w._excluded("MRNA", 0.05)   # same ticker, crypto price
    assert w._excluded("ENGSTOCK", 1.0) and not w._excluded("BTC", 60000.0)
    assert not w.det.eligible("TSLAUSDT", 400.0)                      # static list
