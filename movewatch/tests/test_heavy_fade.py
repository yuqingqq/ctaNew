"""Heavy-volume pump short (docs/HEAVY_FADE.md): stop logic, decision rules, windows, guards."""
import numpy as np
import pandas as pd
import pytest

import research.heavy_fade as hf


def test_short_trade_stop_and_hold():
    row = np.array([10.5, 11.0, np.nan, 12.6, 9.0])
    assert hf.short_trade(row, 10.0, 0.25) == (3, 12.6)          # first close >= 12.5, gap paid in full
    assert hf.short_trade(row, 10.0, None) == (4, 9.0)           # no stop: hold to the exit
    assert hf.short_trade(row, 10.0, 0.40) == (4, 9.0)           # stop never reached


def _times(n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(pd.Timestamp("2023-01-02", tz="UTC") + pd.to_timedelta(np.sort(rng.integers(0, 500 * 1440, n)), unit="min"))


def test_decision_rules():
    rng = np.random.default_rng(1)
    t = _times(400)
    assert hf.verdict(hf.stats(rng.normal(0.05, 0.1, 400), t), 4) == "VALIDATED"
    noise = rng.normal(0, 0.1, 400)
    assert hf.verdict(hf.stats(noise - noise.mean(), t), 4) == "FAILED"
    assert hf.verdict(hf.stats(rng.normal(0.05, 0.1, 50), t[:50]), 1) == "INSUFFICIENT"
    one = np.full(400, -0.001)
    one[0] = 5.0                                                 # one huge winner carries the mean
    assert hf.verdict(hf.stats(one, t), 1) != "VALIDATED"


def test_falsifiers_run():
    assert hf.falsifiers(_times(500), 2) == {"planted": "VALIDATED", "noise": "FAILED", "empty": "INSUFFICIENT"}


def test_conf_segments_tile_without_overlap():
    segs = hf.conf_segments()
    assert segs[0]["sig"][0] == pd.Timestamp("2020-01-02 04:00", tz="UTC")
    for s in segs:
        end = pd.Timestamp(s["panel"][1], tz="UTC")
        assert end - s["sig"][1] >= pd.Timedelta(minutes=4321)       # the +72 h exit stays in the panel
    for a, b in zip(segs, segs[1:]):
        seam = a["panel_dir"] != b["panel_dir"]                 # 2021-12-31 / 2022-01-01: separate corpora, no margins
        assert a["sig"][1] <= b["sig"][0] and b["sig"][0] - a["sig"][1] <= pd.Timedelta(days=4.5 if seam else 0.25)


def test_conf_refuses_before_freeze(monkeypatch):
    monkeypatch.setattr(hf, "committed_clean", lambda p: False)
    with pytest.raises(SystemExit, match="REFUSED"):
        hf.build_rows("conf")


def test_validate_reads_once(tmp_path, monkeypatch):
    monkeypatch.setattr(hf, "RECEIPT", tmp_path / "r.json")
    (tmp_path / "r.json").write_text("{}")
    with pytest.raises(SystemExit, match="read once"):
        hf.validate()
