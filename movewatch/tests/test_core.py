"""Unit tests. Every checker ships a falsifier (CLAUDE.md rule 15): each test pairs
a positive control the instrument must flag with a known-bad input it must refuse.

Run: python3 -m pytest -q tests/
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movewatch.alerts import Alert, AlertCfg, AlertEngine, BurstGovernor, govern  # noqa: E402
from movewatch.instruments import FX, make, split_multiplier  # noqa: E402
from movewatch.signals import DEFAULT_RULES, baselines_from_minutes, fire  # noqa: E402
from movewatch.store import BarStore  # noqa: E402

T0 = 1_790_000_000 // 60 * 60  # minute-aligned epoch second


# ----------------------------------------------------------------------------- identity
def test_multiplier_prefixes():
    assert split_multiplier("1000PEPE") == ("PEPE", 1000.0)
    assert split_multiplier("SHIB1000") == ("SHIB", 1000.0)
    assert split_multiplier("kPEPE") == ("PEPE", 1000.0)
    assert split_multiplier("1000000MOG") == ("MOG", 1e6)
    # known-bad: tickers that legitimately start with digits are NOT stripped
    assert split_multiplier("1INCH") == ("1INCH", 1.0)


def test_stables_and_quotes_refused():
    assert make("binance_spot", "USDCUSDT", "USDC", "USDT", "spot") is None
    assert make("binance_spot", "BTCTRY", "BTC", "TRY", "spot") is None
    ins = make("upbit", "KRW-XRP", "XRP", "KRW", "spot")
    assert ins is not None and ins.quote == "KRW"


# ----------------------------------------------------------------------------- store
def _store_with(n_insts=1):
    st = BarStore(FX(), sec_len=900, min_len=300)
    ids = [st.register(make("binance_usdm", f"C{i}USDT", f"C{i}", "USDT", "perp")) for i in range(n_insts)]
    st.tick(T0)
    return st, ids


def _run(st, idx, t_from, t_to, px_fn, qv=0.0):
    for s in range(t_from, t_to):
        st.on_trade(idx, px_fn(s), qv)
        st.tick(s + 1)


def test_outage_does_not_read_as_a_jump():
    """Positive control: WITHOUT outage marking a reconnect at a new price reads as a
    +20% 1m move. With set_down during the gap, the 1m return spanning it is NaN."""
    for marked in (False, True):
        st, (i,) = _store_with()
        _run(st, i, T0, T0 + 300, lambda s: 100.0)
        if marked:
            st.set_down([i], True)
        for s in range(T0 + 300, T0 + 420):   # 2 min outage: no prints
            st.tick(s + 1)
        if marked:
            st.set_down([i], False)
        st.on_trade(i, 120.0, 1.0)            # first print after reconnect
        st.tick(T0 + 421)
        r1m = st.returns([60])[60][i]
        if marked:
            assert math.isnan(r1m)
        else:
            assert r1m == pytest.approx(math.log(1.2))


def test_catchup_writes_minutes():
    st, (i,) = _store_with()
    _run(st, i, T0, T0 + 60, lambda s: 10.0 + (s - T0) * 0.01, qv=5.0)
    st.tick(T0 + 200)                          # a 140 s stall
    m = (T0 + 59) // 60
    assert st.min_px[i, m % st.M] == pytest.approx(10.59)
    assert st.min_qv[i, m % st.M] == pytest.approx(300.0)
    assert st.min_px[i, (m + 1) % st.M] == pytest.approx(10.59)   # carried forward, zero volume
    assert st.min_qv[i, (m + 1) % st.M] == 0.0


# ----------------------------------------------------------------------------- triggers
def _panel(n=40, L=1440, seed=0):
    rng = np.random.default_rng(seed)
    lr = rng.normal(0, 0.002, size=(n, L))
    C = 10 * np.exp(np.cumsum(lr, axis=1))
    V = rng.lognormal(mean=np.log(20_000), sigma=0.5, size=(n, L))
    return C, V


def test_trigger_needs_price_volume_and_liquidity():
    C, V = _panel()
    sig, v1m, nh = baselines_from_minutes(C[:, :-60], V[:, :-60])
    rule = DEFAULT_RULES.horizons[900]                  # 15m: 7%, z5, vol x3
    liq = V.sum(axis=1)
    e = np.full(C.shape[0], math.log(1.12))             # +12% net of market
    z = e / (sig * math.sqrt(15))
    vm_hot = np.full(C.shape[0], 8.0)
    vm_cold = np.full(C.shape[0], 1.0)
    assert fire(e, z, vm_hot, liq, rule, 2e6).all()                    # positive control
    assert not fire(e, z, vm_cold, liq, rule, 2e6).any()               # no volume -> refuse
    assert not fire(e, z, vm_hot, liq * 0 + 1e5, rule, 2e6).any()      # illiquid -> refuse
    assert not fire(e * 0.3, z * 0.3, vm_hot, liq, rule, 2e6).any()    # +3.5% -> below floor
    assert not fire(np.full_like(e, np.nan), z, vm_hot, liq, rule, 2e6).any()  # NaN never fires


def test_baseline_ignores_single_wick_and_needs_history():
    C, V = _panel(n=2)
    C2 = C.copy()
    C2[0, 700] *= 1.5                                   # one 50% wick
    s_clean, _, _ = baselines_from_minutes(C, V)
    s_wick, _, _ = baselines_from_minutes(C2, V)
    assert s_wick[0] < 2.0 * s_clean[0]                 # p99 clip keeps the scale
    s_short, v_short, _ = baselines_from_minutes(C[:, :60], V[:, :60])
    assert np.isnan(s_short).all() and np.isnan(v_short).all()   # < min_hist -> not eligible


# ----------------------------------------------------------------------------- episodes
def _qnt_like():
    """Anchor 72 -> 373 over ~4 days with pauses, then a dump to 229."""
    pts = [(0, 72), (3600, 83), (7200, 86), (86400, 95), (2 * 86400, 104), (2 * 86400 + 7200, 123),
           (2 * 86400 + 30000, 150), (3 * 86400, 182), (3 * 86400 + 70000, 228), (3 * 86400 + 73000, 373),
           (3 * 86400 + 76000, 286), (3 * 86400 + 90000, 229)]
    ts = np.arange(0, pts[-1][0] + 1, 60.0)
    px = np.interp(ts, [p[0] for p in pts], [p[1] for p in pts])
    return ts + T0, px


def test_ladder_escalates_then_reverses():
    eng = AlertEngine(AlertCfg(ttl_s=10 * 86400))
    ts, px = _qnt_like()
    out = []
    k = int(np.argmax(px >= 72 * 1.15))
    out += eng.on_trigger("QNT", ts[k], px[k], +1, math.log(px[k] / 72.0), "4h")
    for t, p in zip(ts[k + 1:], px[k + 1:]):
        out += eng.on_price("QNT", t, p)
    kinds = [a.kind for a in out]
    assert kinds[0] == "NEW"
    esc_moves = [round(a.move_pct) for a in out if a.kind == "ESCALATE" and a.direction > 0]
    assert esc_moves == sorted(esc_moves) and esc_moves[-1] >= 300    # monotone ladder, reaches +300%
    rev = [a for a in out if a.kind == "REVERSAL"]
    peak = px.max()                                   # sampled peak (373 falls between samples)
    assert len(rev) == 1 and rev[0].extreme_px == pytest.approx(peak)
    dump = [a for a in out if a.direction < 0 and a.kind == "ESCALATE"]
    assert dump and all(a.anchor_px == pytest.approx(peak) for a in dump)   # anchored at the peak
    gaps = np.diff([a.ts for a in out if a.direction > 0])
    assert (gaps >= 180).all()                                                      # min_gap respected


def test_no_duplicate_new_within_cooldown():
    eng = AlertEngine(AlertCfg())
    a = eng.on_trigger("X", T0, 1.10, +1, math.log(1.10), "1h")
    assert [x.kind for x in a] == ["NEW"]
    b = eng.on_trigger("X", T0 + 5, 1.11, +1, math.log(1.11), "1h")
    assert b == []                                         # same episode: no second NEW
    eng.on_price("X", T0 + 4000, 1.0)                      # retrace before min_life: stays open
    assert ("X", 1) in eng.active
    eng.on_price("X", T0 + 8000, 1.0)                      # retrace after min_life: closes
    assert ("X", 1) not in eng.active
    c = eng.on_trigger("X", T0 + 8100, 1.09, +1, math.log(1.09), "1h")
    assert c == [] and eng.suppressed["cooldown"] == 1     # cooldown: smaller move refused, counted
    d = eng.on_trigger("X", T0 + 8200, 1.3, +1, math.log(1.3), "1h")
    assert [x.kind for x in d] == ["NEW"]                  # 25%+ bigger than the closed extreme: allowed


def test_asset_budget_gap_and_daily_cap():
    """Positive control: an unbudgeted engine would send every trigger. With the budget,
    <= 4 NEW / 24h and >= 15 min apart; suppressions are counted."""
    eng = AlertEngine(AlertCfg(asset_max_new_per_day=4, asset_gap_s=900, min_life_s=0, new_cooldown_s=0))
    sent = []
    for k in range(48):                                    # a pump+full-retrace every 30 min for 24h
        t = T0 + k * 1800
        sent += eng.on_trigger("Z", t, 1.12, +1, math.log(1.12), "15m")
        eng.on_price("Z", t + 900, 1.0)                    # retrace -> closes (min_life 0)
    ts = [a.ts for a in sent]
    assert 1 <= len(sent) <= 4
    assert all(b - a >= 900 for a, b in zip(ts, ts[1:]))
    assert eng.suppressed["asset_budget"] >= 40
    free = AlertEngine(AlertCfg(asset_max_new_per_day=10_000, asset_gap_s=0, min_life_s=0, new_cooldown_s=0))
    n_free = sum(len(free.on_trigger("Z", T0 + k * 1800, 1.12, +1, math.log(1.12), "15m"))
                 + 0 * len(free.on_price("Z", T0 + k * 1800 + 900, 1.0)) for k in range(48))
    assert n_free == 48                                    # the control really would have spammed


def test_no_reversal_chain_in_chop():
    eng = AlertEngine(AlertCfg())
    out = eng.on_trigger("C", T0, 1.30, +1, math.log(1.30), "4h")      # +30% pump
    out += eng.on_price("C", T0 + 3600, 1.40)                         # extreme 1.40
    out += eng.on_price("C", T0 + 7200, 1.12)                         # -20% off the top -> REVERSAL
    out += eng.on_price("C", T0 + 10800, 1.35)                        # +20% bounce off 1.12
    out += eng.on_price("C", T0 + 14400, 1.10)                        # and down again
    kinds = [a.kind for a in out]
    assert kinds.count("REVERSAL") == 1                                # the spawned dump never reverses


def test_opposite_trigger_inside_live_move_is_ignored():
    eng = AlertEngine(AlertCfg())
    eng.on_trigger("O", T0, 1.12, +1, math.log(1.12), "1h")            # +12% pump (< 15%: no reversal rule)
    a = eng.on_trigger("O", T0 + 600, 1.08, -1, math.log(1.08 / 1.12), "5m")
    assert a == [] and ("O", 1) in eng.active                          # still +8% vs anchor: noise
    b = eng.on_trigger("O", T0 + 2400, 0.95, -1, math.log(0.95 / 1.08), "15m")
    assert [x.kind for x in b] == ["NEW"] and b[0].direction == -1     # below the anchor: a real turn
    assert ("O", 1) not in eng.active


def test_alert_state_roundtrip(tmp_path):
    eng = AlertEngine(AlertCfg())
    eng.on_trigger("Y", T0, 2.0, +1, math.log(1.2), "15m")
    p = tmp_path / "a.json"
    eng.save(p)
    eng2 = AlertEngine(AlertCfg())
    assert eng2.load(p, T0 + 60) == 1
    assert eng2.on_trigger("Y", T0 + 61, 2.0, +1, math.log(1.2), "15m") == []   # no re-announce
    assert len(eng2.sent["Y"]) == 1                                             # budget history kept


# ----------------------------------------------------------------------------- clusters
def test_same_ticker_price_clusters():
    fx = FX()
    st = BarStore(fx, sec_len=120, min_len=120)
    rows = [("binance_usdm", "METAUSDT", "USDT", 729.0, 5e8), ("bitget_fut", "METAUSDT", "USDT", 728.5, 1e8),
            ("upbit", "KRW-META", "KRW", 11.7, 3e6), ("bithumb", "KRW-META", "KRW", 11.6, 1e6),
            ("okx_spot", "META-USDT", "USDT", 2.5, 2e5), ("binance_usdm", "BTCUSDT", "USDT", 80000, 1e10)]
    for v, s, q, px, vol in rows:
        i = st.register(make(v, s, s.replace("KRW-", "").replace("-USDT", "").replace("USDT", ""), q, "perp"))
        st.last_px[i] = px
        st.vol24h[i] = vol
    fx.krw_per_usd = 1360.0
    st.refresh_consistency(0.4)
    names = dict(zip([i.key for i in st.insts], st.asset_of))
    assert names["binance_usdm:METAUSDT"] == "META" and names["bitget_fut:METAUSDT"] == "META"
    assert names["upbit:KRW-META"] == names["bithumb:KRW-META"] == "META@upbit"
    assert names["okx_spot:META-USDT"] == "META@okx"
    assert names["binance_usdm:BTCUSDT"] == "BTC"          # negative control: single cluster untouched


def test_vol24h_normalized_by_observed_minutes():
    """Cold start: one venue has a full backfilled day, another 2 hours of the same
    per-minute volume. Raw sums differ 12x (positive control); the estimate must not."""
    st = BarStore(FX(), sec_len=120, min_len=1500)
    a = st.register(make("bitget_fut", "XUSDT", "X", "USDT", "perp"))
    b = st.register(make("binance_usdm", "XUSDT", "X", "USDT", "perp"))
    st.tick(T0)
    st.t = T0 + 1440 * 60 - 1                                  # jump the clock to a full day later
    cur_m = (st.t + 1) // 60
    for m in range(cur_m - 1440, cur_m):
        st.min_px[a, m % st.M], st.min_qv[a, m % st.M] = 1.0, 1000.0
    for m in range(cur_m - 120, cur_m):
        st.min_px[b, m % st.M], st.min_qv[b, m % st.M] = 1.0, 1000.0
    raw_a = 1440 * 1000.0
    raw_b = 120 * 1000.0
    assert raw_a / raw_b == pytest.approx(12.0)
    st.refresh_baselines()
    assert st.vol24h[a] == pytest.approx(st.vol24h[b], rel=0.02)
    assert st.vol24h[a] == pytest.approx(1.44e6, rel=0.02)


def test_blowoff_escalations_not_blocked_by_asset_budget():
    """QNT 09-27: +20% NEW at 21:30, then +33% -> +89% within ~12 minutes. Every ladder level
    >= 3 min apart must be announced (the v2 budget hid +50%/+75%). Falsifier: a NEW trigger
    for a DIFFERENT episode inside the same 15-min gap is still refused."""
    eng = AlertEngine(AlertCfg())
    anchor = 197.81
    out = eng.on_trigger("QNT", T0, anchor * 1.20, +1, math.log(1.20), "15m")
    path = [(T0 + 60 * k, anchor * f) for k, f in ((4, 1.33), (8, 1.52), (12, 1.77), (15, 1.89))]
    for t, px in path:
        out += eng.on_price("QNT", t, px)
    esc = [round(a.move_pct) for a in out if a.kind == "ESCALATE"]
    assert esc == [33, 52, 77], esc                 # +30, +50, +75 levels all delivered
    assert all(b.ts - a.ts >= 180 for a, b in zip(out, out[1:]))
    other = eng.on_trigger("ZZZ", T0, 1.2, +1, math.log(1.2), "15m")
    again = eng.on_trigger("ZZZ", T0 + 300, 1.0 * 1.2 * 1.5, +1, math.log(1.5), "5m")
    assert [a.kind for a in other] == ["NEW"]
    assert [a.kind for a in again] == ["ESCALATE"]  # a live episode extends; never a duplicate NEW
    eng2 = AlertEngine(AlertCfg(min_life_s=0, new_cooldown_s=0))
    eng2.on_trigger("Y", T0, 1.2, +1, math.log(1.2), "15m")
    eng2.on_price("Y", T0 + 60, 1.0)                # closes (retrace, min_life 0)
    blocked = eng2.on_trigger("Y", T0 + 120, 1.3, +1, math.log(1.3), "15m")
    assert blocked == [] and eng2.suppressed["asset_budget"] == 1   # NEW still budgeted


def _alert(kind, asset, ts, move=10.0):
    return Alert(kind=kind, asset=asset, direction=1, ts=ts, price=1.0, anchor_px=1.0, anchor_ts=ts,
                 move_pct=move, horizon="5m")


def test_burst_governor_digests_market_wide_days():
    g = BurstGovernor(window_s=600, max_new=6, digest_every_s=300)
    admitted = [g.admit(_alert("NEW", f"C{i}", T0 + 15 * i)) for i in range(20)]   # 20 NEW in 5 min
    assert sum(admitted) == 6 and len(g.held) == 14
    assert g.admit(_alert("ESCALATE", "C0", T0 + 200))                           # continuations pass
    assert g.flush(T0 + 100) == []                                               # not due yet
    held = g.flush(T0 + 90 + 300)
    assert len(held) == 14 and g.held == []
    n_msg, n_dig = govern([_alert("NEW", f"D{i}", T0 + i) for i in range(20)])
    assert (n_msg, n_dig) == (6, 1)                                              # replay helper agrees
    assert govern([_alert("NEW", f"E{i}", T0 + 3600 * i) for i in range(20)]) == (20, 0)  # control


def test_continuations_do_not_consume_the_new_cap():
    """QNT 09-26/27 under v3: escalations + reversals used up a 6-message cap, delaying the
    blow-off NEW to the top. Now: 3 NEW episodes with many continuations in between -> the
    third NEW still goes out."""
    eng = AlertEngine(AlertCfg(asset_max_new_per_day=4))
    out = eng.on_trigger("Q", T0, 1.20, +1, math.log(1.20), "1h")           # NEW 1
    for k, px in enumerate((1.32, 1.52, 1.80), 1):                            # 3 escalations
        out += eng.on_price("Q", T0 + 600 * k, px)
    out += eng.on_price("Q", T0 + 2400, 1.50)                                 # reversal (+ spawned dump)
    out += eng.on_price("Q", T0 + 9000, 1.95)                                 # dump fully retraced
    out += eng.on_trigger("Q", T0 + 9060, 2.30, +1, math.log(2.30 / 1.95), "15m")   # NEW 2 (blow-off)
    kinds = [a.kind for a in out]
    assert kinds.count("NEW") == 2 and kinds.count("ESCALATE") >= 3 and kinds.count("REVERSAL") == 1


def test_cli_status_runs_without_config(capsys):
    """Regression: `status` crashed on args.config (it has no --config option)."""
    from movewatch.__main__ import main
    assert main(["status", "--n", "1"]) == 0
    assert "alerts" in capsys.readouterr().out


def test_store_restore_reads_each_array_once_and_restores_pending(tmp_path, monkeypatch):
    """Regression: z[...] inside the per-key loop decompressed the whole array per instrument
    (~20 min at 100% CPU for 4.4k instruments; found by the 2026-09-28 restart rehearsal)."""
    import movewatch.store as store_mod
    from movewatch.instruments import FX, make
    from movewatch.store import BarStore
    t = 1_790_000_000
    s1 = BarStore(FX(), min_len=100)
    for k in range(40):
        s1.register(make("binance_usdm", f"C{k}USDT", f"C{k}", "USDT", "perp"))
    s1.t = t
    s1.min_px[:40, :] = np.arange(40, dtype=np.float32)[:, None] + 1.0
    s1.min_qv[:40, :] = 7.0
    s1.save(tmp_path / "store.npz")
    s2 = BarStore(FX(), min_len=100)
    for k in range(30):                               # C30..C39 register later (lazy venue)
        s2.register(make("binance_usdm", f"C{k}USDT", f"C{k}", "USDT", "perp"))
    s2.t = t + 120
    reads, real = [], np.load

    class Counting:
        def __init__(self, z):
            self.z = z

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.z.close()

        def __getitem__(self, k):
            reads.append(k)
            return self.z[k]
    monkeypatch.setattr(store_mod.np, "load", lambda *a, **kw: Counting(real(*a, **kw)))
    assert s2.load(tmp_path / "store.npz") == 30
    assert len(reads) <= 4                            # t, keys, min_px, min_qv: once each
    m = (t + 1) // 60 - 1                             # the last whole minute before the save
    assert s2.min_px[5, m % 100] == 6.0 and s2.min_qv[5, m % 100] == 7.0
    j = s2.register(make("binance_usdm", "C35USDT", "C35", "USDT", "perp"))
    assert s2.min_px[j, m % 100] == 36.0              # pending restore applied at registration
    assert np.isnan(s2.min_px[5, ((s2.t + 1) // 60 - 1) % 100])   # minutes after the save stay empty


def test_aliases_merge_tickers_of_one_token():
    from movewatch.instruments import set_aliases
    set_aliases({"LONGXIA": "龙虾", "PUMPFUN": "PUMP"})
    try:
        assert make("bybit_linear", "LONGXIAUSDT", "LONGXIA", "USDT", "perp").base == "龙虾"
        assert make("binance_usdm", "龙虾USDT", "龙虾", "USDT", "perp").base == "龙虾"
        assert make("bybit_linear", "1000PUMPFUNUSDT", "1000PUMPFUN", "USDT", "perp").base == "PUMP"   # after the multiplier
        assert make("bybit_linear", "PUMPUSDT", "PUMP", "USDT", "perp").base == "PUMP"
    finally:
        set_aliases({})
    assert make("bybit_linear", "LONGXIAUSDT", "LONGXIA", "USDT", "perp").base == "LONGXIA"


def test_binance_usdm_universe_keeps_symbols_missing_from_the_ticker_window(monkeypatch):
    import asyncio
    import movewatch.venues.binance as bn
    live = {"p": "1.01", "i": "1.00", "r": "0.0001", "T": 1790956800000}
    mark = {s: dict(live, s=s) for s in ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDC", "EEE_251226")}
    mark["OLDUSDT"] = {"s": "OLDUSDT", "p": "0.5", "i": "0.5", "r": "0.00000000", "T": 1790956800000}  # delisted
    mark["NOFUNDUSDT"] = dict(live, s="NOFUNDUSDT", T=0)
    tick = {"AAAUSDT": {"q": "5000000"}, "BBBUSDT": {"q": "50000"},       # CCC/DDD did not tick in the window
            "TRADEDUSDT": {"q": "200000"}}                                  # ticked: kept even though mark == index
    mark["TRADEDUSDT"] = {"s": "TRADEDUSDT", "p": "2", "i": "2", "r": "0", "T": 1790956800000}

    async def fake(url, secs):
        return mark if "markPrice" in url else tick
    monkeypatch.setattr(bn, "_collect", fake)
    f = bn.BinanceUsdm.__new__(bn.BinanceUsdm)
    f.cfg, f.venue = {"min_quote_volume_24h": 100_000}, "binance_usdm"
    got = sorted(i.symbol for i in asyncio.run(f.discover(None)))
    assert got == ["AAAUSDT", "CCCUSDT", "DDDUSDC", "TRADEDUSDT"]          # BBB low; OLD/NOFUND settled; dated skipped
