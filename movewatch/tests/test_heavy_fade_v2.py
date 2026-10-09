"""Heavy pumps with a futures discount (docs/HEAVY_FADE_V2.md): frozen cuts and the read-once guards."""
import json
from pathlib import Path

import pytest

import research.heavy_fade_v2 as v2


def test_cuts_are_the_signal_search_freeze():
    fz = json.loads((Path(v2.ROOT) / "docs" / "SIGNAL_SEARCH_FREEZE.json").read_text())
    a3 = next(c for c in fz["cells"] if c["id"] == "A3")["cuts"]
    assert v2.DISCOUNT == a3[0] and v2.PREMIUM == a3[1] and v2.DISCOUNT < 0 < v2.PREMIUM


def test_validate_refuses_uncommitted(monkeypatch, tmp_path):
    monkeypatch.setattr(v2, "RECEIPT", tmp_path / "r.json")
    monkeypatch.setattr(v2.fs, "committed_clean", lambda p: False)
    with pytest.raises(SystemExit, match="REFUSED"):
        v2.validate()


def test_validate_reads_once(monkeypatch, tmp_path):
    monkeypatch.setattr(v2, "RECEIPT", tmp_path / "r.json")
    (tmp_path / "r.json").write_text("{}")
    with pytest.raises(SystemExit, match="read once"):
        v2.validate()
