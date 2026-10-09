"""Upbit listing study (docs/UPBIT_LISTINGS.md): parser, trade arithmetic, read-once guards."""
import pandas as pd
import pytest

import research.upbit_listings as ul


def test_parser_keeps_first_krw_notice_per_ticker():
    n = [{"id": 3, "first_listed_at": "2025-01-02T10:00:05+09:00", "title": "바나(VANA) 신규 거래지원 안내 (KRW, BTC, USDT 마켓) (추가 안내)"},
         {"id": 2, "first_listed_at": "2025-01-01T10:00:05+09:00", "title": "바나(VANA) 신규 거래지원 안내 (KRW, BTC, USDT 마켓)"},
         {"id": 1, "first_listed_at": "2023-05-01T12:00:00+09:00", "title": "KRW, BTC 마켓 디지털 자산 추가 (NEAR, YGG)"},
         {"id": 0, "first_listed_at": "2019-03-01T12:00:00+09:00", "title": "원화마켓 신규 상장 (헤데라해시그래프 HBAR)"},
         {"id": 9, "first_listed_at": "2024-01-01T12:00:00+09:00", "title": "일드베이시스(YB) 신규 거래지원 안내 (BTC, USDT 마켓)"}]
    df = ul.parse(n)
    assert sorted(df["ticker"]) == ["HBAR", "NEAR", "VANA", "YGG"]
    assert df.loc[df["ticker"] == "VANA", "time"].iloc[0] == pd.Timestamp("2025-01-01T01:00:05", tz="UTC")


def test_trades_at_long_and_short():
    a = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    idx = pd.date_range(a - pd.Timedelta(minutes=5), a + pd.Timedelta(minutes=1500), freq="1min")
    close = pd.Series(10.0, index=idx)
    close.loc[a + pd.Timedelta(minutes=1):] = 11.0          # the pop
    close.loc[a + pd.Timedelta(minutes=1440):] = 9.9        # the fade
    t = ul.trades_at(close, a, cost=0.0)
    assert t["long_5"] == pytest.approx(0.0) and t["short_60_1440"] == pytest.approx(0.1)


def test_confirmation_refuses_before_freeze(monkeypatch):
    monkeypatch.setattr(ul, "committed_clean", lambda p: False)
    with pytest.raises(SystemExit, match="REFUSED"):
        ul.paths("conf")


def test_validate_reads_once(tmp_path, monkeypatch):
    monkeypatch.setattr(ul, "RECEIPT", tmp_path / "r.json")
    (tmp_path / "r.json").write_text("{}")
    with pytest.raises(SystemExit, match="read once"):
        ul.validate()
