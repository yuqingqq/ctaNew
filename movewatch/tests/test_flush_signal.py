"""Flush signal instrument (docs/FLUSH_SIGNAL.md): detector, refractory, E2 rule, decision rules, guards."""
import math

import numpy as np
import pandas as pd
import pytest

import research.flush_signal as fs


def _panel(n=60, T=400, drop_at=200, n_drop=40, depth=0.06):
    C = np.full((n, T), 10.0, np.float32)
    C[:n_drop, drop_at:] *= (1 - depth)
    V = np.ones((n, T), np.float32)
    LIQ = np.full((n, (T + 14) // 15), 1e7, np.float32)
    return C, V, LIQ, np.ones(n, bool)


def _times(n, seed):
    rng = np.random.default_rng(seed)
    mins = np.sort(rng.integers(0, 600 * 1440, n))
    return pd.Series(pd.Timestamp("2023-01-02", tz="UTC") + pd.to_timedelta(mins, unit="min"))


def test_detector_matches_naive_and_fires_once():
    C, V, LIQ, keep = _panel()
    ser = fs.minute_series(C, V, LIQ, keep, 20, 380)
    for t in (150, 199, 200, 210, 214, 215, 300):
        ref = fs.naive_minute(C, V, LIQ, keep, t)
        for c in fs.SER:
            a, b = ser[c][t - 20], ref[c]
            assert (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12, (t, c, a, b)
    s5 = ser["s5"]
    assert s5[199 - 20] == 0 and s5[200 - 20] == pytest.approx(40 / 60)
    assert s5[214 - 20] == pytest.approx(40 / 60) and s5[215 - 20] == 0     # 15-minute window has passed
    times = np.datetime64("2026-01-01T00:00") + np.arange(360).astype("timedelta64[m]")
    assert fs.fire_times(times, s5 >= 0.5) == [times[180]]


def test_thin_universe_never_triggers():
    C, V, LIQ, keep = _panel(n=40, n_drop=30)
    ser = fs.minute_series(C, V, LIQ, keep, 20, 380)
    assert np.isnan(ser["s5"]).all() and not (ser["s5"] >= 0.5).any()


def test_no_volume_coin_is_not_in_universe():
    C, V, LIQ, keep = _panel(n=70, n_drop=40)
    V[:20, :] = 0                                    # 20 of the fallen coins did not trade
    ser = fs.minute_series(C, V, LIQ, keep, 20, 380)
    assert ser["n_u"][200 - 20] == 50 and ser["s5"][200 - 20] == pytest.approx(20 / 50)


def test_refractory_24h_per_tier():
    times = np.datetime64("2026-01-01T00:00") + np.arange(3000).astype("timedelta64[m]")
    cond = np.zeros(3000, bool)
    cond[[10, 11, 500, 1449, 1450, 2900]] = True
    assert fs.fire_times(times, cond) == [times[10], times[1450], times[2900]]


def test_e2_turn_and_timeout():
    m5 = np.full(200, -0.01)
    m5[30] = 0.002
    assert fs.e2_offset(m5, 10) == (21, False)       # turn at t+20 -> buy at t+21
    m5[30] = np.nan
    assert fs.e2_offset(m5, 10) == (61, True)


def test_verdict_rules():
    rng = np.random.default_rng(1)
    pool_t, pool_v = _times(5000, 2), rng.normal(-0.01, 0.02, 5000)
    ev_t, good = _times(30, 3), rng.normal(0.04, 0.02, 30)
    ok_plateau = {v: 0.03 for v in fs.PLATEAU}
    assert fs.evaluate(good, ev_t, pool_v, pool_t, ok_plateau)["verdict"] == "VALIDATED"
    assert fs.evaluate(good, ev_t, pool_v, pool_t, ok_plateau | {"FLUSH_D6": -0.001})["verdict"] == "SUGGESTIVE"
    one = np.full(30, -0.002)
    one[0] = 0.5                                     # one huge event carries the mean
    assert fs.evaluate(one, ev_t, pool_v, pool_t, ok_plateau)["verdict"] != "VALIDATED"
    assert fs.evaluate(good[:10], ev_t[:10], pool_v, pool_t, ok_plateau)["verdict"] == "INSUFFICIENT"
    bad = rng.normal(-0.01, 0.02, 30)
    assert fs.evaluate(bad, ev_t, pool_v, pool_t, {v: -0.01 for v in fs.PLATEAU})["verdict"] == "FAILED"


def test_falsifiers_fire():
    rng = np.random.default_rng(4)
    pool = pd.DataFrame({"time": _times(3000, 5), **{fs.col(e, b): rng.normal(-0.01, 0.02, 3000) for e, b in fs.CELLS}})
    out = fs.falsifiers(pool)                        # raises SystemExit if an instrument check fails
    assert all(v["planted"] == "VALIDATED" and v["empty"] == "INSUFFICIENT" for v in out.values())


def test_hist_build_refuses_before_freeze(monkeypatch):
    monkeypatch.setattr(fs, "committed_clean", lambda p: False)
    with pytest.raises(SystemExit, match="REFUSED"):
        fs.build("hist")


def test_hist_is_read_once(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "committed_clean", lambda p: True)
    monkeypatch.setattr(fs, "OUT", tmp_path)
    (tmp_path / "receipt_hist.json").write_text("{}")
    with pytest.raises(SystemExit, match="read once"):
        fs.validate("hist")
    with pytest.raises(SystemExit, match="REFUSED"):
        fs.build("hist")
