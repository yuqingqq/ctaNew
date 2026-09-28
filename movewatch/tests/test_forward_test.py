"""Falsifiers for the forward test: the evaluator and the engine's follow-up path (no network)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import research.forward_eval as fe  # noqa: E402
from movewatch.alerts import Alert  # noqa: E402
from movewatch.engine import Engine  # noqa: E402
from movewatch.followup import code_fingerprint  # noqa: E402
from movewatch.instruments import make  # noqa: E402

T0 = 1_790_000_000.0          # 2026-09-21 12:13 UTC
COINS = [f"binance_usdm:C{k}USDT" for k in range(12)]


def _snap(level: float, rng, noise=0.0) -> dict:
    return {k: float(math.exp(level + (rng.normal(0, noise) if noise else 0.0))) for k in COINS}


def _records(n=400, planted=True, seed=1, run="A"):
    """Pump rows whose market-adjusted +4h->+72h return is -0.05 * log10(vol ratio) + noise (planted)
    or pure noise, on top of a random market move the evaluator must remove."""
    rng = np.random.default_rng(seed)
    recs = [{"kind": "FU_START", "t": T0 - 60, "code_sha": "FROZEN", "dry_run": False, "run": run}]
    for i in range(n):
        t = T0 + i * 7200.0
        lvr = rng.normal(0, 0.4)
        x = (-0.05 * lvr if planted else 0.0) + rng.normal(0, 0.03)
        m0, m4, m72 = 0.0, rng.normal(0, 0.02), rng.normal(0, 0.05)
        p4 = 1.0 * math.exp(m4)
        p72 = p4 * math.exp((m72 - m4) + x)
        recs += [
            {"kind": "FU_OPEN", "id": f"a{i}", "asset": "X", "dir": 1, "t": t, "alert_px": 1.0,
             "ref_key": "binance_usdm:XUSDT", "breadth": float(rng.uniform(0, .2)), "mkt_px": _snap(m0, rng), "run": run},
            {"kind": "FU_4H", "id": f"a{i}", "t": t + 14405, "late_s": 5.0, "px": p4, "px_key": "binance_usdm:XUSDT",
             "vol_ratio": float(10 ** lvr), "vol_ratio_binance_usdm": float(10 ** lvr), "gap_min_before": 0,
             "gap_min_after": 0, "mkt_px": _snap(m4, rng), "mkt_liq": COINS, "run": run},
            {"kind": "FU_72H", "id": f"a{i}", "t": t + 72 * 3600 + 5, "late_s": 5.0, "px": p72,
             "px_key": "binance_usdm:XUSDT", "mkt_px": _snap(m72, rng), "run": run},
        ]
    return recs


def _outcomes(recs):
    runs, start, end = fe.admission(recs, "FROZEN")
    return fe.outcomes(recs, runs, start, end)


def test_evaluator_detects_a_planted_effect_and_not_noise():
    d1 = _outcomes(_records(planted=True))
    d0 = _outcomes(_records(planted=False))
    assert fe.ic_ci(d1, "lvr", "x4_72")["pass"] is True          # positive control
    assert fe.ic_ci(d0, "lvr", "x4_72")["pass"] is False         # known-null input


def test_market_move_is_removed():
    recs = _records(n=20, planted=False, seed=3)
    for r in recs:                                                 # the coin moves exactly with the market
        if r["kind"] in ("FU_4H", "FU_72H"):
            r["px"] = r["mkt_px"][COINS[0]]
    d = _outcomes(recs)
    assert np.allclose(d["x4_72"], 0.0, atol=1e-6) and np.allclose(d["x0_72"], 0.0, atol=1e-6)
    assert not np.allclose(d["raw4_72"], 0.0, atol=1e-3)        # the raw return still carries the market


def test_every_exclusion_is_a_status():
    recs = _records(n=30, seed=4)
    by = {(r["kind"], r.get("id")): r for r in recs}
    by[("FU_4H", "a1")]["late_s"] = 5000                                  # 4h check an hour+ late
    by[("FU_72H", "a2")]["px_key"] = "upbit:KRW-X"                        # priced by another instrument
    by[("FU_4H", "a3")]["gap_min_after"] = 6                              # > 5 min without data
    by[("FU_72H", "a4")]["mkt_px"] = dict(list(by[("FU_72H", "a4")]["mkt_px"].items())[:5])   # 5 coins < 10
    by[("FU_OPEN", "a5")]["run"] = "DRY"                                  # written by a non-admitted run
    by[("FU_OPEN", "a6")]["breadth"] = float("nan")
    recs = [r for r in recs if not (r["kind"] == "FU_72H" and r.get("id") == "a7")]    # never written
    d = _outcomes(recs).set_index("id")
    assert d.loc["a1", "st_H1"] == "4H_LATE" and d.loc["a1", "st_H2"] == "4H_LATE"
    assert d.loc["a2", "st_H1"] == "72H_PX_SOURCE_CHANGED" and d.loc["a2", "st_H2"] == "OK"
    assert d.loc["a3", "st_H1"] == "VOL_GAP" and d.loc["a3", "st_B1"] == "OK"
    assert d.loc["a4", "st_H1"] == "NO_MARKET" and d.loc["a4", "st_B1"] == "NO_MARKET"
    assert d.loc["a5", "st_H1"] == d.loc["a5", "st_H2"] == d.loc["a5", "st_B1"] == "OTHER_RUN"
    assert d.loc["a6", "st_H2"] == "NO_BREADTH" and d.loc["a6", "st_H1"] == "OK"
    assert d.loc["a7", "st_H1"] == "72H_MISSING" and d.loc["a7", "st_B1"] == "72H_MISSING"
    for k in ("a1", "a2", "a3", "a4", "a5", "a7"):
        assert np.isnan(d.loc[k, "x4_72"])
    c = fe.counts(d.reset_index(), T0, None, T0 + 86400 * 60)
    assert sum(c["H1_status"].values()) == 30 and c["H1_complete"] == c["H1_status"]["OK"]


def test_admission_ends_at_the_first_production_run_with_other_code():
    recs = [{"kind": "FU_START", "t": 100.0, "code_sha": "FROZEN", "dry_run": False, "run": "A"},
            {"kind": "FU_START", "t": 150.0, "code_sha": "OTHER", "dry_run": True, "run": "B"},   # dry run: ignored
            {"kind": "FU_START", "t": 200.0, "code_sha": "OTHER", "dry_run": False, "run": "C"},  # ends the test
            {"kind": "FU_START", "t": 300.0, "code_sha": "FROZEN", "dry_run": False, "run": "D"}]
    assert fe.admission(recs, "FROZEN") == ({"A"}, 100.0, 200.0)
    assert fe.admission(recs, "NEVER") == (set(), None, None)


def test_complete_weeks_counts_whole_iso_weeks_only():
    start = pd.Timestamp("2026-09-28 16:00").timestamp()                  # a Monday afternoon
    assert fe.complete_weeks(start, pd.Timestamp("2026-10-11 23:59").timestamp()) == 0
    assert fe.complete_weeks(start, pd.Timestamp("2026-10-12 00:00").timestamp()) == 1
    assert fe.complete_weeks(start, pd.Timestamp("2026-11-09 00:00").timestamp()) == 5
    assert fe.complete_weeks(pd.Timestamp("2026-10-05").timestamp(), pd.Timestamp("2026-10-12").timestamp()) == 1


def _patch_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(fe, "ROOT", tmp_path)
    monkeypatch.setattr(fe, "freeze_commit", lambda root=None: "c0ffee")
    monkeypatch.setattr(fe, "fingerprint_at", lambda commit, root=None: "FROZEN")


def test_final_refuses_below_minimum(tmp_path, monkeypatch):
    _patch_freeze(monkeypatch, tmp_path)
    (tmp_path / "data" / "alerts").mkdir(parents=True)
    (tmp_path / "data" / "alerts" / "followups-20260928.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _records(n=20)) + "\n")
    with pytest.raises(SystemExit, match="refused: minimum"):
        fe.main(["--final"])


def test_final_refuses_a_second_look(tmp_path, monkeypatch):
    _patch_freeze(monkeypatch, tmp_path)
    monkeypatch.setattr(fe, "ready", lambda c: True)
    (tmp_path / "data" / "alerts").mkdir(parents=True)
    (tmp_path / "data" / "alerts" / "followups-20260928.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _records(n=20)) + "\n")
    (tmp_path / "data" / "forward_test").mkdir(parents=True)
    (tmp_path / "data" / "forward_test" / "receipt_20261201T000000Z.json").write_text("{}")
    with pytest.raises(SystemExit, match="already exists"):
        fe.main(["--final"])


def test_fingerprint_matches_the_commit_and_catches_an_edit(tmp_path):
    pkg = tmp_path / "movewatch"
    (pkg / "venues").mkdir(parents=True)
    (pkg / "a.py").write_text("A = 1\n")
    (pkg / "venues" / "b.py").write_text("B = 2\n")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "movewatch.yaml").write_text("rules: {x: 1}\nalerts: {y: [1, 2]}\n")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(git + ["add", "."], cwd=tmp_path, check=True)
    subprocess.run(git + ["commit", "-qm", "freeze"], cwd=tmp_path, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    cfg = yaml.safe_load((tmp_path / "config" / "movewatch.yaml").read_text())
    frozen = fe.fingerprint_at(commit, root=tmp_path)
    assert code_fingerprint(cfg, pkg=pkg) == frozen                        # same code + config: equal
    assert code_fingerprint(dict(cfg, rules={"x": 2}), pkg=pkg) != frozen  # a config change: caught
    (pkg / "venues" / "b.py").write_text("B = 3\n")
    assert code_fingerprint(cfg, pkg=pkg) != frozen                        # a code change: caught


# ---------------------------------------------------------------- engine follow-up path
def _engine(tmp_path) -> Engine:
    cfg = {"data_dir": str(tmp_path / "data"), "notify": {"dry_run": True, "telegram": False},
           "followup": {"enabled": True, "message_pumps": True, "record_horizons_h": [4, 24, 72],
                        "vol_quiet_max": 0.94, "vol_heavy_min": 2.26}}
    return Engine(cfg, tmp_path)


def _market(eng, alert_minute, now_minute):
    """XYZ on two venues + 12 liquid coins; prices in every minute of the ring window."""
    st = eng.store
    a = st.register(make("binance_usdm", "XYZUSDT", "XYZ", "USDT", "perp"))
    b = st.register(make("bybit_linear", "XYZUSDT", "XYZ", "USDT", "perp"))
    coins = [st.register(make("binance_usdm", f"C{k}USDT", f"C{k}", "USDT", "perp")) for k in range(12)]
    st.t = now_minute * 60
    for m in range(alert_minute - 200, now_minute):
        st.min_px[:, m % st.M] = 1.0
    for k in range(1, 181):
        st.min_qv[[a, b], (alert_minute - k) % st.M] = 10.0       # before: 2 x 180 x 10 = 3,600
    for k in range(60, 240):
        st.min_qv[[a, b], (alert_minute + k) % st.M] = 50.0       # after:  2 x 180 x 50 = 18,000 -> 5.0x
    st.last_px[[a, b]] = 1.2
    st.last_px[coins] = 2.0
    st.last_ms[: st.n] = int(time.time() * 1000)
    st.vol24h[[a, b]] = 2e7
    st.vol24h[coins] = 1e7
    eng.members = {"XYZ": np.array([a, b]), **{f"C{k}": np.array([c]) for k, c in enumerate(coins)}}
    eng.primary = {"XYZ": a, **{f"C{k}": c for k, c in enumerate(coins)}}
    return a, b, coins


def _alert(asset, alert_minute, direction=1):
    return Alert(kind="NEW", asset=asset, direction=direction, ts=alert_minute * 60 + 30.0, price=1.0, anchor_px=0.9,
                 anchor_ts=alert_minute * 60.0, move_pct=11.0, horizon="1h", episode_id=f"{asset}-1",
                 metrics={"ref_key": f"binance_usdm:{asset}USDT", "breadth": 0.0, "breadth_label": "coin-specific"})


def _fu_records(tmp_path):
    return [json.loads(x) for f in (tmp_path / "data" / "alerts").glob("followups-*.jsonl") for x in f.read_text().splitlines()]


def test_engine_followup_records_snapshots_volume_and_messages_once(tmp_path):
    eng = _engine(tmp_path)
    alert_minute = int(time.time() // 60) - 300
    a, b, coins = _market(eng, alert_minute, alert_minute + 300)
    eng._schedule_followup(_alert("XYZ", alert_minute), messaged=True)
    due = (alert_minute + 240) * 60 + 5
    eng._process_followups(due + 1)
    recs = _fu_records(tmp_path)
    op = next(r for r in recs if r["kind"] == "FU_OPEN")
    assert len(op["mkt_px"]) == 12 and "binance_usdm:XYZUSDT" not in op["mkt_px"]      # alerted asset excluded
    assert op["member_keys"] == ["binance_usdm:XYZUSDT", "bybit_linear:XYZUSDT"] and op["run"] == eng.run_id
    fu4 = [r for r in recs if r["kind"] == "FU_4H"]
    assert len(fu4) == 1 and fu4[0]["vol_ratio"] == pytest.approx(5.0) and fu4[0]["vol_label"] == "heavy"
    assert fu4[0]["vol_ratio_binance_usdm"] == pytest.approx(5.0) and fu4[0]["n_inst_used"] == 2
    assert fu4[0]["gap_min_before"] == 0 and fu4[0]["gap_min_after"] == 0
    assert fu4[0]["px_key"] == "binance_usdm:XYZUSDT" and fu4[0]["hold"] == pytest.approx(math.log(1.2))
    assert len(fu4[0]["mkt_liq"]) == 12 and len(fu4[0]["mkt_px"]) == 12
    assert eng.notifier.q.qsize() == 1                           # one +4h message for the pump
    eng._process_followups(due + 50)
    assert eng.notifier.q.qsize() == 1                           # never twice


def test_engine_followup_messages_only_sent_pumps(tmp_path):
    eng = _engine(tmp_path)
    alert_minute = int(time.time() // 60) - 300
    _market(eng, alert_minute, alert_minute + 300)
    eng._schedule_followup(_alert("XYZ", alert_minute), messaged=False)                 # held for a digest
    eng._schedule_followup(_alert("C1", alert_minute, direction=-1), messaged=True)     # a dump
    eng._process_followups((alert_minute + 240) * 60 + 6)
    assert eng.notifier.q.qsize() == 0
    assert len([r for r in _fu_records(tmp_path) if r["kind"] == "FU_4H"]) == 2           # both recorded


def test_engine_followup_volume_gap_and_new_venue(tmp_path):
    eng = _engine(tmp_path)
    alert_minute = int(time.time() // 60) - 300
    a, b, _ = _market(eng, alert_minute, alert_minute + 300)
    st = eng.store
    for k in range(1, 181):                         # b listed mid-window: no prices or volume before the alert
        st.min_px[b, (alert_minute - k) % st.M] = np.nan
        st.min_qv[b, (alert_minute - k) % st.M] = 0.0
    eng._schedule_followup(_alert("XYZ", alert_minute), messaged=True)
    eng._process_followups((alert_minute + 240) * 60 + 6)
    fu4 = next(r for r in _fu_records(tmp_path) if r["kind"] == "FU_4H")
    assert fu4["n_inst_used"] == 1 and fu4["vol_ratio"] == pytest.approx(5.0)          # b excluded (else 10x)
    eng3 = _engine(tmp_path / "h")
    a, b, _ = _market(eng3, alert_minute, alert_minute + 300)
    for k in range(60, 240):                        # b went dark after the alert: no prices or volume
        eng3.store.min_px[b, (alert_minute + k) % eng3.store.M] = np.nan
        eng3.store.min_qv[b, (alert_minute + k) % eng3.store.M] = 0.0
    eng3._schedule_followup(_alert("XYZ", alert_minute), messaged=True)
    eng3._process_followups((alert_minute + 240) * 60 + 6)
    fu4 = next(r for r in _fu_records(tmp_path / "h") if r["kind"] == "FU_4H")
    assert fu4["n_inst_used"] == 1 and fu4["vol_ratio"] == pytest.approx(5.0)          # b excluded (else 2.5x)
    eng2 = _engine(tmp_path / "g")
    a, b, _ = _market(eng2, alert_minute, alert_minute + 300)
    for k in range(100, 110):                       # the service was down 10 min in the after window
        eng2.store.min_px[[a, b], (alert_minute + k) % eng2.store.M] = np.nan
    eng2._schedule_followup(_alert("XYZ", alert_minute), messaged=True)
    eng2._process_followups((alert_minute + 240) * 60 + 6)
    fu4 = next(r for r in _fu_records(tmp_path / "g") if r["kind"] == "FU_4H")
    assert fu4["gap_min_after"] == 10 and fu4["vol_label"] == "incomplete"
    assert "unreliable" in eng2.notifier.q.get_nowait()


def test_engine_followup_late_check_records_without_volume_or_message(tmp_path):
    eng = _engine(tmp_path)
    alert_minute = int(time.time() // 60) - 600
    _market(eng, alert_minute, alert_minute + 600)
    eng._schedule_followup(_alert("XYZ", alert_minute), messaged=True)
    eng._process_followups((alert_minute + 240) * 60 + 5 + 3600)  # an hour late
    fu4 = next(r for r in _fu_records(tmp_path) if r["kind"] == "FU_4H")
    assert fu4["vol_label"] == "late" and fu4["vol_ratio"] is None
    assert eng.notifier.q.qsize() == 0


def test_followup_state_roundtrip(tmp_path):
    eng = _engine(tmp_path)
    now = time.time()
    eng.followups = [{"id": "k", "alert_ts": now, "done": [4], "mkt_keys0": ["x"]},
                     {"id": "old", "alert_ts": now - 79 * 3600, "done": [], "mkt_keys0": ["x"]},
                     {"id": "legacy", "alert_ts": now, "done": []}]            # pre-snapshot format
    eng._save_followups()
    eng2 = _engine(tmp_path)
    assert eng2._load_followups(now) == 1 and eng2.followups[0]["done"] == [4]
