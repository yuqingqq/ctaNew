"""Signal search instrument: decision rules and the read-once guards (docs/SIGNAL_SEARCH.md)."""
import json

import numpy as np
import pandas as pd
import pytest

import research.signal_search as ss


def _frame(n=600, ic=0.0, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    x = ic * y + np.sqrt(1 - ic * ic) * rng.standard_normal(n)
    t = pd.date_range("2026-01-05", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"f": x, "y": y, "week": t.tz_convert(None).to_period("W").start_time, "dir": 1})


def test_cell_flags_planted_and_not_noise():
    hit = ss.cell(_frame(ic=0.3), "f", "y", draws=500)
    assert hit["status"] == "OK" and hit["ci"][0] > 0 and ss.verdict(hit, +1, 10) == "VALIDATED"
    assert ss.verdict(hit, -1, 10) == "FAILED"                 # declared sign is binding
    miss = ss.cell(_frame(ic=0.0, seed=3), "f", "y", draws=500)
    assert ss.verdict(miss, +1, 10) != "VALIDATED"


def test_insufficient_never_passes():
    d = _frame(n=60)
    c = ss.cell(d, "f", "y", draws=100)
    assert c["status"] == "INSUFFICIENT" and ss.verdict(c, +1, 1) == "INSUFFICIENT"
    assert not ss.screen_pass(c, c, c)


def test_screen_needs_same_sign_in_both_windows():
    pos = {"status": "OK", "ic": 0.1, "ci": [0.05, 0.15]}
    neg = {"status": "OK", "ic": -0.02, "ci": [-0.1, 0.06]}
    assert ss.screen_pass(pos, pos, pos)
    assert not ss.screen_pass(pos, neg, pos)
    assert not ss.screen_pass(pos, pos, {"status": "OK", "ic": 0.03, "ci": [-0.01, 0.07]})


def test_bonferroni_separates_validated_from_suggestive():
    c = {"status": "OK", "ic": 0.05, "ci": [0.01, 0.09], "p_pos": 0.01, "p_neg": 0.99}
    assert ss.verdict(c, +1, 1) == "VALIDATED"
    assert ss.verdict(c, +1, 20) == "SUGGESTIVE"


def test_validate_refuses_without_committed_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "FREEZE", tmp_path / "freeze.json")
    with pytest.raises(SystemExit, match="REFUSED"):
        ss.validate()
    (tmp_path / "freeze.json").write_text(json.dumps({"k": 1, "cells": []}))
    monkeypatch.setattr(ss, "committed_clean", lambda p: False)       # written but not committed
    with pytest.raises(SystemExit, match="REFUSED"):
        ss.validate()


def test_validate_refuses_second_read(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "FREEZE", tmp_path / "freeze.json")
    (tmp_path / "freeze.json").write_text(json.dumps({"k": 1, "cells": []}))
    monkeypatch.setattr(ss, "committed_clean", lambda p: True)
    monkeypatch.setattr(ss, "RECEIPT", tmp_path / "receipt.json")
    (tmp_path / "receipt.json").write_text("{}")
    with pytest.raises(SystemExit, match="read once"):
        ss.validate()


def test_v_build_refuses_before_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "FREEZE", tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="REFUSED"):
        ss.build_window("val_a")
