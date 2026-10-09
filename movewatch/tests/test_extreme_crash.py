"""Extreme-crash strategy (docs/EXTREME_CRASH.md): configuration, plateau rule, minimum, path, guards."""
import copy

import numpy as np
import pandas as pd
import pytest

import research.extreme_crash as ec
import research.flush_signal as fs


@pytest.fixture
def cfg(monkeypatch):
    """configure() rewrites flush_signal globals; restore them after each test."""
    for name in ("DEPTHS", "SER", "TIERS", "MIN_EVENTS", "K", "CELLS", "WINDOWS"):
        monkeypatch.setattr(fs, name, copy.copy(getattr(fs, name)))
    ec.configure()


def test_configure(cfg):
    assert fs.TIERS["EXTREME"] == (0.10, 0.50) and fs.TIERS["EXTREME_D12"] == (0.12, 0.50)
    assert {"s8", "s12", "s10", "s3"} <= set(fs.SER)
    assert fs.K == 1 and fs.MIN_EVENTS == 10 and fs.CELLS == [("E1", "H")]
    segs = fs.WINDOWS["early"]
    assert segs[0]["own"][0] == pd.Timestamp("2020-01-02", tz="UTC") and segs[-1]["own"][1] == pd.Timestamp("2021-12-30", tz="UTC")
    for a, b in zip(segs, segs[1:]):
        assert a["own"][1] == b["own"][0]                  # owned ranges tile the window
    for s in segs:
        assert pd.Timestamp(s["panel"][1], tz="UTC") - s["own"][1] >= pd.Timedelta(days=2)


def test_configure_does_not_leak():
    assert fs.K == 4 and fs.MIN_EVENTS == 15 and "EXTREME_D8" not in fs.TIERS


def test_minimum_is_ten(cfg):
    t = pd.Series(pd.Timestamp("2021-01-04", tz="UTC") + pd.to_timedelta(np.arange(10) * 9, unit="D"))
    v = np.full(10, 0.05)
    assert fs.cell_stats(v, fs._weeks(t))["status"] == "OK"
    assert fs.cell_stats(v[:9], fs._weeks(t[:9]))["status"] == "INSUFFICIENT"


def test_plateau_variant_needs_three_fires():
    cc = fs.col("E1", "H")
    ev = pd.DataFrame({"tier": ["EXTREME_D8"] * 4 + ["EXTREME_S75"] * 2, cc: [0.02, 0.03, 0.01, 0.04, -0.2, -0.1]})
    allv, req = ec.plateau_split(ev, cc)
    assert set(req) == {"EXTREME_D8"} and allv["EXTREME_S75"]["fires"] == 2 and allv["EXTREME_D12"]["fires"] == 0


def test_basket_path_ends_at_recorded_gross():
    n, T = 60, 2000                               # basket_trades also computes the 24 h hold
    rng = np.random.default_rng(0)
    C = (10 * np.exp(np.cumsum(rng.normal(0, 0.002, (n, T)), axis=1))).astype(np.float32)
    C[:40, 300:] *= 0.85
    V = np.ones((n, T), np.float32)
    LIQ = np.full((n, (T + 14) // 15), 1e7, np.float32)
    keep = np.ones(n, bool)
    rec, _ = fs.basket_trades(C, V, LIQ, keep, 305, {"E1": 306, "E2": 306})
    path = ec.basket_path(C, V, LIQ, keep, 305, hold=60)
    assert abs(path[-1] - rec[fs.col("E1", "H", 60, "gross")]) <= 1e-9


def test_early_refuses_before_freeze(cfg, monkeypatch):
    monkeypatch.setattr(fs, "committed_clean", lambda p: False)
    with pytest.raises(SystemExit, match="REFUSED"):
        ec.build("early")


def test_early_is_read_once(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "committed_clean", lambda p: True)
    monkeypatch.setattr(fs, "OUT", tmp_path)
    (tmp_path / "receipt_extreme_early.json").write_text("{}")
    with pytest.raises(SystemExit, match="read once"):
        ec.validate("early")
    with pytest.raises(SystemExit, match="REFUSED"):
        ec.build("early")
