"""Falsifier tests for the forward-test instruments (movewatch/followup.py)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movewatch.engine import format_followup  # noqa: E402
from movewatch.followup import breadth, breadth_label, hold, volume_label, volume_ratio  # noqa: E402


def test_breadth_counts_same_direction_movers_excluding_the_asset():
    n = 13
    r = np.zeros(n)
    r[[1, 2, 3]] = np.log(1.03)            # three other coins +3%
    r[0] = np.log(1.50)                    # the alerted asset itself (must be excluded)
    liq, ok = np.full(n, 1e7), np.ones(n, bool)
    excl = np.zeros(n, bool); excl[0] = True
    assert breadth(r, liq, ok, +1, excl) == pytest.approx(3 / 12)
    no_excl = np.zeros(n, bool)
    assert breadth(r, liq, ok, +1, no_excl) == pytest.approx(4 / 13)          # positive control: counting itself
    assert breadth(r, liq, ok, -1, excl) == 0.0                               # direction matters
    thin = liq.copy(); thin[[1, 2]] = 1e5
    assert breadth(r, thin, ok, +1, excl) == pytest.approx(1 / 10)            # illiquid coins ignored
    assert math.isnan(breadth(r[:8], liq[:8], ok[:8], +1, excl[:8]))          # < 10 coins: not measurable


def test_volume_ratio_uses_hours_1_to_4_vs_3h_before():
    M, n = 1500, 2
    q = np.zeros((n, M))
    m_a = 10_000
    for k in range(1, 181):                 # before: [m_a-180, m_a)
        q[:, (m_a - k) % M] = 1.0
    for k in range(60, 240):                # after: [m_a+60, m_a+240)
        q[:, (m_a + k) % M] = 3.0
    ids = np.array([0, 1])
    ratio, after, before = volume_ratio(q, M, ids, m_a)
    assert (ratio, after, before) == (pytest.approx(3.0), pytest.approx(1080.0), pytest.approx(360.0))
    q2 = q.copy()
    for k in range(0, 60):                  # hour 0-1 after the alert must NOT count
        q2[:, (m_a + k) % M] = 1000.0
    assert volume_ratio(q2, M, ids, m_a)[0] == pytest.approx(3.0)
    q3 = q.copy(); q3[:, (m_a + 239) % M] = 1000.0   # last minute of the window counts ...
    assert volume_ratio(q3, M, ids, m_a)[0] > 3.0
    q4 = q.copy(); q4[:, (m_a + 240) % M] = 1000.0   # ... the next one does not
    assert volume_ratio(q4, M, ids, m_a)[0] == pytest.approx(3.0)
    assert math.isnan(volume_ratio(np.zeros((n, M)), M, ids, m_a)[0])        # no volume: unmeasurable


def test_labels_and_hold():
    assert volume_label(0.5, 0.94, 2.26) == "quiet" and volume_label(1.5, 0.94, 2.26) == "mid"
    assert volume_label(3.0, 0.94, 2.26) == "heavy" and volume_label(float("nan"), 0.94, 2.26) == "unknown"
    assert breadth_label(0.0, 0.0051, 0.0468) == "coin-specific"
    assert breadth_label(0.2, 0.0051, 0.0468) == "market-wide"
    assert breadth_label(0.02, 0.0051, 0.0468) == "partly shared"
    assert hold(+1, 100.0, 110.0) == pytest.approx(math.log(1.1))
    assert hold(-1, 100.0, 110.0) == pytest.approx(-math.log(1.1))


def test_followup_message_says_candidate():
    fu = {"asset": "XYZ", "alert_px": 1.0}
    heavy = format_followup(fu, {"hold": math.log(1.05), "px": 1.05, "vol_ratio": 3.2, "vol_label": "heavy"})
    quiet = format_followup(fu, {"hold": 0.0, "px": 1.0, "vol_ratio": 0.5, "vol_label": "quiet"})
    assert "HEAVY" in heavy and "unvalidated" in heavy and "+5.0%" in heavy
    assert "QUIET" in quiet and "unvalidated" in quiet


def test_beta_to_recovers_a_planted_beta_and_refuses_thin_data():
    from movewatch.followup import beta_to
    rng = np.random.default_rng(7)
    rb = rng.normal(0, 0.002, 1380)
    ra = 1.8 * rb + rng.normal(0, 0.001, 1380)
    btc, coin = 80000 * np.exp(np.cumsum(rb)), 2.0 * np.exp(np.cumsum(ra))
    b, c, n = beta_to(coin, btc, step=1)
    assert b == pytest.approx(1.8, abs=0.1) and c > 0.9 and n == 1379
    coin[100:] = np.nan                                            # mostly missing -> no estimate
    b, c, n = beta_to(coin, btc, step=1)
    assert np.isnan(b) and n < 100


def test_fingerprint_ignores_watcher_and_notify_but_not_detection():
    from movewatch.followup import fingerprint
    files = [("engine.py", b"E"), ("watchers.py", b"W")]
    cfg = {"rules": {"x": 1}, "watchers": {"upbit_flags": ["A"]}, "notify": {"telegram": True}}
    f0 = fingerprint(files, cfg)
    assert fingerprint([("engine.py", b"E"), ("watchers.py", b"W2")], cfg) == f0
    assert fingerprint(files, dict(cfg, watchers={"upbit_flags": []}, notify={"telegram": False})) == f0
    assert fingerprint([("engine.py", b"E2"), ("watchers.py", b"W")], cfg) != f0
    assert fingerprint(files, dict(cfg, rules={"x": 2})) != f0
    assert fingerprint(files, dict(cfg, aliases={"A": "B"})) != f0
