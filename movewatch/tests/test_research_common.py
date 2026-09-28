"""Falsifier tests for research/common.py (review fixes 2026-09-28)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.common import Stale, days_between, is_tradfi, metrics_row, mkt_median, panel_meta  # noqa: E402


def test_metrics_row_is_point_in_time():
    """A Vision metrics row stamped 12:00 describes [12:00, 12:05): invisible at 12:03 (the old
    `T <= X` rule would have used it: positive control), visible from 12:05."""
    m = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2026-09-01 11:55", "2026-09-01 12:00"], utc=True))
    at = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    assert m[m.index <= at("2026-09-01 12:03")]["x"].iloc[-1] == 2.0          # the old, leaky rule
    assert metrics_row(m, at("2026-09-01 12:03"), "x") == 1.0                  # fixed
    assert metrics_row(m, at("2026-09-01 12:05"), "x") == 2.0
    assert np.isnan(metrics_row(m, at("2026-09-01 11:59"), "x"))               # nothing complete yet


def test_days_between_includes_middle_days():
    days = days_between(pd.Timestamp("2026-09-03 00:30", tz="UTC"), 25)
    assert days == ["2026-09-01", "2026-09-02", "2026-09-03"]                   # old helper returned 09-01 + 09-03


def test_tradfi_list_excludes_stocks_not_crypto_collisions():
    assert is_tradfi("AAPLUSDT") and is_tradfi("CLUSDT")
    assert not any(is_tradfi(s) for s in ("QNTUSDT", "FUSDT", "TUSDT", "DIAUSDT", "SPXUSDT", "BTCUSDT"))


def _panel(tmp_path, closes: dict, vols: dict, T: int):
    idx = pd.date_range("2026-01-01", periods=T, freq="1min", tz="UTC")
    for s in closes:
        pd.DataFrame({"close": closes[s], "quote_volume": vols[s]}, index=idx).to_parquet(tmp_path / f"{s}.parquet")
    return idx


def test_stale_flags_frozen_bars_after_delisting(tmp_path):
    T = 5000
    live = np.ones(T); dead = np.ones(T)
    vd = np.ones(T); vd[3000:] = 0.0                                            # "delisted" at minute 3000: flat, no volume
    idx = _panel(tmp_path, {"AAAUSDT": live, "BBBUSDT": dead}, {"AAAUSDT": np.ones(T), "BBBUSDT": vd}, T)
    meta = panel_meta(tmp_path, tmp_path, ["AAAUSDT", "BBBUSDT"], idx)
    st = Stale(meta, 2)
    assert st.coin(1, 2500, 3500)                                               # window crosses into frozen bars
    assert st.coin(1, 3200, 3300)
    assert not st.coin(1, 100, 2900)                                            # before the delisting: fine
    assert not st.coin(0, 2500, 3500)                                           # control coin never stale
    assert list(st.mask(2500, 3500)) == [False, True]


def test_stale_treats_pre_listing_as_unavailable_not_frozen(tmp_path):
    T = 4000
    c = np.full(T, np.nan); c[2000:] = 1.0
    v = np.zeros(T); v[2000:] = 1.0
    idx = _panel(tmp_path, {"NEWUSDT": c}, {"NEWUSDT": v}, T)
    st = Stale(panel_meta(tmp_path, tmp_path, ["NEWUSDT"], idx), 1)
    assert st.coin(0, 1500, 2500)                                               # starts before listing
    assert not st.coin(0, 2100, 3900)
    assert len(st.i) == 0                                                       # pre-listing is not a frozen run


def test_mkt_median_excludes_the_coin_and_frozen_coins():
    n, T = 12, 100
    C = np.ones((n, T))
    C[:, 50] = 1.01                                                             # every coin +1% ...
    C[0, 50] = 2.0                                                              # ... coin 0 doubles
    LIQ = np.full((n, T // 15 + 1), 1e7)
    st = Stale({"stale_i": [11], "stale_a": [0], "stale_b": [99], "first_valid": [0] * n}, n)
    C[11, 50] = 0.5                                                             # coin 11 is frozen/garbage
    assert mkt_median(C, LIQ, 0, 50, st, exclude=0) == pytest.approx(np.log(1.01))
    Cx = C.copy(); Cx[1:11, 50] = [1.01, 1.01, 1.01, 1.01, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    # positive control: including coin 0 (2.0) tips the median; excluding it must not
    assert mkt_median(Cx, LIQ, 0, 50, st, exclude=None) == pytest.approx(np.log(2.0))


def test_dir_simple_short_side_is_bounded():
    """A −80% crash on a dump signal is +80% for the short — the naive exp(x) − 1 gives +400%
    (positive control for the bug), the fixed conversion must give +80%."""
    from research.common import dir_simple
    x_dump = -np.log(0.2)                      # direction-adjusted log return of a −80% crash (d = −1)
    assert np.expm1(x_dump) == pytest.approx(4.0)                         # the old, wrong conversion
    assert dir_simple(x_dump, -1) == pytest.approx(0.8)                   # fixed
    assert dir_simple(np.log(1.5), +1) == pytest.approx(0.5)              # pumps unchanged
    assert np.all(dir_simple(np.array([5.0, 50.0]), -1) <= 1.0)           # a short never makes > 100%
