"""Falsifiers for forward test F1 (docs/FORWARD_FADE_F1.md): the evaluator and the cost watcher (no network)."""
from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import research.forward_eval as fe  # noqa: E402
import research.forward_fade as ff  # noqa: E402
from movewatch.watchers import FadeCostWatcher, book_cost  # noqa: E402

T0 = 1_790_000_000.0          # 2026-09-21 12:13 UTC
HEAVY_RATIO = 3.0             # log10 0.477 > the A2 heavy cut 0.35259
QUIET_RATIO = 1.5


def _records(n=500, planted=True, seed=1, run="A", asset=lambda i: f"C{i}", ratio=HEAVY_RATIO):
    """Heavy Binance pumps, one asset each (no 72 h repeats). Planted: the coin falls 3% from +4 h
    to +24 h on average (a short pays); otherwise it drifts 0 with the same noise."""
    rng = np.random.default_rng(seed)
    recs = [{"kind": "FU_START", "t": T0 - 60, "code_sha": "FROZEN", "dry_run": False, "run": run}]
    for i in range(n):
        t, a = T0 + i * 6 * 3600.0, asset(i)          # 500 rows span ~18 ISO weeks
        key = f"binance_usdm:{a}USDT"
        p4 = 1.0
        p24 = p4 * math.exp((-0.03 if planted else 0.0) + rng.normal(0, 0.08))
        recs += [
            {"kind": "FU_OPEN", "id": f"a{i}", "asset": a, "dir": 1, "t": t, "alert_px": 0.9, "ref_key": key, "run": run},
            {"kind": "FU_4H", "id": f"a{i}", "asset": a, "dir": 1, "t": t + 14405, "late_s": 5.0, "px": p4, "px_key": key,
             "vol_ratio": ratio, "vol_ratio_binance_usdm": ratio, "gap_min_before": 0, "gap_min_after": 0, "run": run},
            {"kind": "FU_24H", "id": f"a{i}", "asset": a, "dir": 1, "t": t + 86400 + 3, "late_s": 3.0, "px": p24,
             "px_key": key, "run": run},
        ]
    return recs


def _rows(recs, published=lambda m: True):
    runs, start, end = fe.admission(recs, "FROZEN")
    return ff.rows(recs, runs, start, end, published=published)


def _funding(rate):
    return lambda symbol, t0, t1: rate


# ---------------------------------------------------------------- evaluator
def test_detects_a_planted_short_edge_and_not_noise():
    s1 = ff.final_stats(ff.trades(_rows(_records(planted=True)), fund=_funding(0.0)))
    s0 = ff.final_stats(ff.trades(_rows(_records(planted=False)), fund=_funding(0.0)))
    assert s1["H3_net"]["pass"] is True                     # positive control
    assert s0["H3_net"]["pass"] is False                    # known-null input
    assert s1["H3_net"]["mean"] == pytest.approx(s1["gross"]["mean"] - ff.COST, abs=1e-12)


def test_funding_is_charged_to_the_short():
    d = _rows(_records(n=50, planted=False))
    x0, x1 = ff.trades(d, fund=_funding(0.0)), ff.trades(d, fund=_funding(-0.01))
    assert np.allclose(x1["net"], x0["net"] - 0.01)          # negative funding: the short pays


def test_archive_funding_sums_only_settlements_inside_the_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, "ROOT", tmp_path)
    d = tmp_path / "data" / "forward_fade" / "funding"
    d.mkdir(parents=True)
    t0, t1 = T0, T0 + 20 * 3600                                   # 2026-09-21 12:13 -> 09-22 08:13
    ms = lambda t: int(t * 1000)  # noqa: E731
    (d / "XUSDT_2026-09.csv").write_text("calc_time,funding_interval_hours,last_funding_rate\n"
                                         f"{ms(t0)},8,0.5\n{ms(t0) + 1},8,-0.004\n{ms(t1)},8,-0.006\n{ms(t1) + 1},8,0.5\n")
    assert ff.funding_sum("XUSDT", t0, t1, get=lambda *a, **k: pytest.fail("cached")) == pytest.approx(-0.010)


def test_missing_archive_funding_is_a_status_not_a_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, "ROOT", tmp_path)

    class R:
        status_code = 404
    assert math.isnan(ff.funding_sum("XUSDT", T0, T0 + 72000, get=lambda *a, **k: R()))
    x = ff.trades(_rows(_records(n=20)), fund=_funding(float("nan")))
    assert (x["status"] == "NO_FUNDING").all() and x["net"].isna().all()


def test_unpublished_funding_month_is_pending_not_complete():
    d = _rows(_records(n=20, seed=5), published=lambda m: False)
    assert (d["status"] == "FUNDING_PENDING").all()
    assert ff.months(T0, T0 + 20 * 86400) == ["2026-09", "2026-10"]


def test_every_exclusion_is_a_status():
    recs = _records(n=40, planted=False, seed=4)
    by = {(r["kind"], r.get("id")): r for r in recs}
    by[("FU_OPEN", "a1")]["ref_key"] = "bybit_linear:C1USDT"              # not priced on Binance USDM
    for k in ("FU_4H", "FU_24H"):
        by[(k, "a1")]["px_key"] = "bybit_linear:C1USDT"
    for k in ("FU_OPEN", "FU_4H", "FU_24H"):
        by[(k, "a2")]["asset"] = "PEPE"                                   # meme list
    by[("FU_4H", "a3")]["vol_ratio_binance_usdm"] = QUIET_RATIO           # not heavy
    by[("FU_4H", "a4")]["gap_min_before"] = 6                             # > 5 gap minutes
    by[("FU_4H", "a5")]["late_s"] = 700                                   # entry check late
    by[("FU_24H", "a6")]["px_key"] = "binance_usdm:OTHERUSDT"             # exit priced elsewhere
    by[("FU_OPEN", "a7")]["run"] = "DRY"                                  # not an admitted run
    by[("FU_OPEN", "a8")]["dir"] = -1                                     # a dump: not in the population
    for k in ("FU_OPEN", "FU_4H", "FU_24H"):                              # same asset 1 h after a9: a repeat
        by[(k, "a10")]["asset"] = by[(k, "a9")]["asset"]
    by[("FU_4H", "a11")]["vol_ratio_binance_usdm"] = float("nan")          # no Binance volume ratio
    recs = [r for r in recs if not (r["kind"] == "FU_24H" and r.get("id") == "a12")]   # never written
    d = _rows(recs).set_index("id")
    want = {"a1": "NOT_BINANCE_USDM", "a2": "MEME", "a3": "NOT_HEAVY", "a4": "VOL_GAP", "a5": "4H_LATE",
            "a6": "24H_PX_SOURCE_CHANGED", "a7": "OTHER_RUN", "a9": "OK", "a10": "REPEAT_72H", "a11": "VOL_GAP",
            "a12": "24H_MISSING", "a0": "OK"}
    for k, v in want.items():
        assert d.loc[k, "status"] == v, k
    assert "a8" not in d.index
    c = ff.counts(d.reset_index(), T0, None, T0 + 86400 * 60)
    assert sum(c["status"].values()) == 39 and c["complete"] == c["status"]["OK"]


def test_a_repeat_counts_from_the_last_kept_alert_and_after_72h_is_new():
    recs = _records(n=4, asset=lambda i: "X")
    opened = {r["id"]: r["t"] for r in recs if r["kind"] == "FU_OPEN"}
    target = {"a0": 0, "a1": 71, "a2": 72, "a3": 140}           # hours: repeat, new (72 h after a0), repeat of a2
    for r in recs:
        if "id" in r:
            r["t"] += T0 + target[r["id"]] * 3600.0 - opened[r["id"]]
    d = _rows(recs).set_index("id")
    assert list(d.loc[["a0", "a1", "a2", "a3"], "status"]) == ["OK", "REPEAT_72H", "OK", "REPEAT_72H"]


def test_meme_rule_matches_the_research_base():
    import research.premium_angles as pa
    for a in ("1000PEPE", "1000000MOG", "1MBABYDOGE", "PEPE", "X1000", "1000X"):
        assert ff.base(a) == pa.base(a)


def _patch(monkeypatch, tmp_path, recs):
    monkeypatch.setattr(ff, "ROOT", tmp_path)
    monkeypatch.setattr(ff, "published_month", lambda m: True)
    monkeypatch.setattr(ff, "declaration_commit", lambda: "d3c1")
    monkeypatch.setattr(fe, "freeze_commit", lambda root=None: "c0ffee")
    monkeypatch.setattr(fe, "fingerprint_at", lambda commit, root=None: "FROZEN")
    (tmp_path / "data" / "alerts").mkdir(parents=True)
    (tmp_path / "data" / "alerts" / "followups-20260921.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_final_refuses_below_minimum(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, _records(n=30))
    with pytest.raises(SystemExit, match="refused: minimum"):
        ff.main(["--final"])


def test_final_refuses_a_second_look(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path, _records(n=30))
    monkeypatch.setattr(ff, "ready", lambda c: True)
    (tmp_path / "data" / "forward_fade").mkdir(parents=True)
    (tmp_path / "data" / "forward_fade" / "receipt_20270601T000000Z.json").write_text("{}")
    with pytest.raises(SystemExit, match="already exists"):
        ff.main(["--final"])


def test_counts_mode_reads_no_outcome(tmp_path, monkeypatch, capsys):
    _patch(monkeypatch, tmp_path, _records(n=30))
    monkeypatch.setattr(ff, "trades", lambda *a, **k: pytest.fail("counts mode must not compute trades"))
    ff.main([])
    out = capsys.readouterr().out
    assert "counts only" in out and '"complete": 30' in out


# ---------------------------------------------------------------- cost watcher
def test_book_cost_walks_the_book():
    bids, asks = [["100", "10"], ["99", "10"]], [["101", "10"], ["102", "10"]]
    c = book_cost(bids, asks, [1000, 1500, 5000])
    assert c["mid"] == 100.5 and c["spread_bps"] == pytest.approx(1e4 / 100.5)
    assert c["sell_bps"]["1000"] == pytest.approx(1e4 * 0.5 / 100.5)               # all at 100
    vwap = 1500 / (10 + 500 / 99)
    assert c["sell_bps"]["1500"] == pytest.approx(1e4 * (100.5 - vwap) / 100.5)    # 10 @ 100 + the rest @ 99
    assert c["buy_bps"]["1000"] == pytest.approx(1e4 * (1000 / (1000 / 101) - 100.5) / 100.5)
    assert c["sell_bps"]["5000"] is None and c["buy_bps"]["5000"] is None          # the book holds < $2.1k a side
    assert c["bid_usd_1.0"] == pytest.approx(1000) and c["bid_usd_2.0"] == pytest.approx(1990)
    assert c["levels"] == 2 and c["bid_usd_book"] == pytest.approx(1990) and c["ask_usd_book"] == pytest.approx(2030)


def test_book_cost_refuses_empty_or_crossed_books():
    with pytest.raises(ValueError):
        book_cost([], [["101", "1"]], [1000])
    with pytest.raises(ValueError):
        book_cost([["102", "1"]], [["101", "1"]], [1000])


class _Eng:
    def __init__(self, d):
        self.fu_dir = d


def test_watcher_starts_at_the_end_and_reads_only_complete_lines(tmp_path):
    now = T0
    day = __import__("time").strftime("%Y%m%d", __import__("time").gmtime(now))
    p = tmp_path / f"followups-{day}.jsonl"
    p.write_text(json.dumps({"kind": "FU_4H", "id": "old"}) + "\n")
    w = FadeCostWatcher({}, _Eng(tmp_path))
    assert w.new_records(now) == []                                     # history is never snapshotted
    with open(p, "a") as f:
        f.write(json.dumps({"kind": "FU_4H", "id": "new"}) + "\n" + '{"kind": "FU_24H", "id": "half')
    assert [r["id"] for r in w.new_records(now)] == ["new"]
    with open(p, "a") as f:
        f.write('"}\n')
    assert [r["id"] for r in w.new_records(now)] == ["half"]
    nxt = __import__("time").strftime("%Y%m%d", __import__("time").gmtime(now + 86400))
    (tmp_path / f"followups-{nxt}.jsonl").write_text(json.dumps({"kind": "FU_4H", "id": "nextday"}) + "\n")
    assert [r["id"] for r in w.new_records(now + 86400)] == ["nextday"]  # a new day's file is read from its start


def test_watcher_wants_binance_pump_entries_and_exits_only():
    w = FadeCostWatcher({}, _Eng(Path(".")))
    base = {"id": "a", "asset": "X", "dir": 1, "t": T0, "late_s": 5, "px": 1.0, "px_key": "binance_usdm:XUSDT"}
    e = w.wanted(dict(base, kind="FU_4H", vol_ratio_binance_usdm=3.0))
    assert e["leg"] == "entry" and e["symbol"] == "XUSDT" and e["heavy"] is True
    assert w.wanted(dict(base, kind="FU_4H", vol_ratio_binance_usdm=1.2))["heavy"] is False
    assert w.wanted(dict(base, kind="FU_24H"))["leg"] == "exit"
    assert w.wanted(dict(base, kind="FU_72H")) is None
    assert w.wanted(dict(base, kind="FU_4H", dir=-1)) is None
    assert w.wanted(dict(base, kind="FU_4H", px_key="okx_swap:X-USDT-SWAP")) is None


def test_watcher_records_an_error_instead_of_raising(tmp_path, monkeypatch):
    import movewatch.watchers as mw

    async def boom(*a, **k):
        raise RuntimeError("HTTP 451")
    monkeypatch.setattr(mw, "_first", boom)
    r = asyncio.run(FadeCostWatcher({}, _Eng(tmp_path)).snapshot(None, {"symbol": "XUSDT", "leg": "entry"}))
    assert "error" in r and "HTTP 451" in r["error"] and "spread_bps" not in r


def test_watcher_snapshot_reads_depth_and_mark_streams(tmp_path, monkeypatch):
    import movewatch.watchers as mw
    seen = []

    async def fake(url, timeout=10.0):
        seen.append(url)
        if "@depth20" in url:
            return {"e": "depthUpdate", "b": [["100", "10"], ["99", "10"]], "a": [["101", "10"], ["102", "10"]]}
        return {"e": "markPriceUpdate", "p": "100.4", "i": "100.3", "r": "-0.0021", "T": 1791619200000}
    monkeypatch.setattr(mw, "_first", fake)
    r = asyncio.run(FadeCostWatcher({"notionals_usd": [1000]}, _Eng(tmp_path)).snapshot(None, {"symbol": "XUSDT", "leg": "entry"}))
    assert seen == [mw.WS_PUBLIC + "xusdt@depth20@100ms", mw.WS_MARKET + "xusdt@markPrice@1s"]
    assert r["sell_bps"]["1000"] == pytest.approx(1e4 * 0.5 / 100.5) and r["funding_rate"] == -0.0021 and "error" not in r


def test_cost_summary_prints_no_price_and_labels_exits_by_their_entry():
    def snap(i, leg, heavy=None, spread=10.0, err=False):
        r = {"kind": "FADE_COST", "leg": leg, "id": i, "mid": 123.456, "mark": 123.4, "index": 123.3, "px_rec": 123.5}
        if leg == "entry":
            r["heavy"] = heavy
        if err:
            return dict(r, error="HTTP 400")
        side = "sell_bps" if leg == "entry" else "buy_bps"
        return dict(r, spread_bps=spread, **{side: {"1000": spread / 2, "100000": None}})
    recs = [snap("a", "entry", True, 10), snap("b", "entry", False, 30), snap("a", "exit", spread=20),
            snap("b", "exit", spread=40), snap("c", "entry", True, err=True)]
    s = ff.cost_summary(recs)
    assert s["errors"] == 1 and s["entry_heavy"]["n"] == 1 and s["entry_all"]["n"] == 2
    assert s["entry_heavy"]["spread_bps_median"] == 10 and s["exit_heavy"]["spread_bps_median"] == 20
    assert s["exit_all"]["1000_usd"]["median_bps"] == 15 and s["exit_all"]["100000_usd"]["book_too_thin"] == 2
    assert "123." not in json.dumps(s)                                   # no price of any kind leaks


def test_a_hold_across_months_with_one_archive_missing_is_nan_not_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(ff, "ROOT", tmp_path)
    d = tmp_path / "data" / "forward_fade" / "funding"
    d.mkdir(parents=True)
    t0 = 1_790_805_600.0                                          # 2026-09-30 22:00 -> 10-01 18:00
    (d / "XUSDT_2026-09.csv").write_text(f"calc_time,funding_interval_hours,last_funding_rate\n{int(t0 * 1000) + 1},8,-0.004\n")

    class R:
        status_code = 404
    assert ff.months(t0, t0 + 72000) == ["2026-09", "2026-10"]
    assert math.isnan(ff.funding_sum("XUSDT", t0, t0 + 72000, get=lambda *a, **k: R()))
