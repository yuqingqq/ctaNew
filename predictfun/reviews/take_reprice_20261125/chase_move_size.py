"""How big are the Binance moves the chase line enters on, against the moves the Tokyo taker probe's after-move BUYs were sent at?
(USER 2026-11-28: "Ok, can you commit your request/result to pmmm Tokyo repo": the next Tokyo request times BUYs at the line's kind of
move, and its trigger is set from this.) P-2026-005, public data, no orders. Declared before the run; nothing is chosen from it but the
request's trigger, which is a measurement setting, not a strategy parameter.
Per action of the frozen forward selections (11-20 .. 11-26, read by this line; EARLY scope as the forward result), from the day's Binance
futures mid changes (chase_takes.mid_series, Binance's event time E): the move in the action's direction over the last 100 and 1,000 ms
(USD), the mid changes in the last 100 ms, and the mid changes and the move in the 200 ms after (the probe's slow acceptances came with
the move continuing after the send). The same for the probe's 20 BUYs, on Tokyo's own Binance stream (pmmmrust data 1185f65, read from
git through tokyo_taker_check), with each BUY's measured acceptance delay.
Stages: selftest | run (write-once chase_move_size.json).
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/chase_move_size.py run
"""
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import chase_forward as CF  # noqa: E402
import chase_forward_tokyo as CFT  # noqa: E402
import tokyo_taker_check as T  # noqa: E402

C, TR, CT, EF = CF.C, CF.TR, CF.CT, CF.EF
OUT = H / 'chase_move_size.json'
QS = (.1, .25, .5, .75, .9)
SELS = ('MOVE_top10', 'CONTROL_onset')


def move_stats(E, M, t, d):
    """Per decision time t (Binance E, ms) and direction d (+1 = the move up): the mid at or before a time is the last change at or before
    it. -> dict of arrays: up100 / up1000 = d x (mid(t) - mid(t - w)), n100 = changes in (t - 100, t], n_post200 = changes in (t, t + 200],
    up_post200 = d x (mid(t + 200) - mid(t))."""
    E, M, t, d = np.asarray(E, float), np.asarray(M, float), np.asarray(t, float), np.asarray(d, float)
    C.require(len(E) > 1 and np.all(np.diff(E) > 0), 'the mid series must be strictly increasing in E')

    def at(x):
        i = np.searchsorted(E, x, 'right') - 1
        return np.where(i >= 0, M[np.maximum(i, 0)], np.nan)
    cnt = lambda a, b: np.searchsorted(E, b, 'right') - np.searchsorted(E, a, 'right')  # noqa: E731
    m0 = at(t)
    return dict(up100=d * (m0 - at(t - 100.)), up1000=d * (m0 - at(t - 1000.)), n100=cnt(t - 100., t).astype(float),
                n_post200=cnt(t, t + 200.).astype(float), up_post200=d * (at(t + 200.) - m0))


def quant(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return dict(n=int(len(x)), **{f'p{int(q * 100)}': float(np.quantile(x, q)) for q in QS}, mean=float(x.mean()))


def tokyo_series(files):
    """Tokyo's Binance futures bookTicker stream -> mid changes (the last state per E ms), as chase_takes.mid_series collapses ours."""
    rows = []
    for recv, _mono, txt in T.gzl(files['run1/binance.jsonl.gz']):
        x = json.loads(txt).get('data', {})
        if x.get('e') == 'bookTicker':
            rows.append((float(x['E']), (float(x['b']) + float(x['a'])) / 2., recv))
    df = pd.DataFrame(rows, columns=['E', 'mid', 'recv']).sort_values(['E', 'recv'], kind='stable').drop_duplicates('E', keep='last')
    E, M = df.E.to_numpy(), df.mid.to_numpy()
    k = np.r_[True, np.abs(np.diff(M)) > 1e-9]
    return E[k], M[k]


def probe_buys(files, chk):
    """The probe's 20 BUYs: the Binance E they were decided on, their direction on Binance (+1 = Up bought), kind, acceptance delay."""
    acc = {o['hash']: o for o in chk['orders'] if o['tag'] == 'run1' and o['side'] == 'BUY'}
    out = []
    for r in T.jl(files['run1/probe.jsonl']):
        if r['kind'] == 'taker_buy_plan':
            v = r['v']
            o = acc[v['hash']]
            out.append(dict(n=v['n'], kind=v['kind'], E=float(v['binance']['E']), d=1. if v['outcome'] == 'Up' else -1.,
                            accept_ms=o['accept_wallet_after_send_ms'], filled=o.get('filled', 0.) > 0))
    C.require(len(out) == 20 and len({b['n'] for b in out}) == 20, 'expected the 20 run BUYs')
    return sorted(out, key=lambda b: b['n'])


def spearman(a, b):
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    return float(a.rank().corr(b.rank()))


def run():
    C.require(not OUT.exists(), f'{OUT.name} exists (write-once)')
    CF.frozen()
    CF.forward_mode(CF.FWD)
    stats = {s: {k: [] for k in ('up100', 'up1000', 'n100', 'n_post200', 'up_post200')} for s in SELS}
    for day in CF.FWD:
        X = pd.read_parquet(CFT.OUT_T / f'{day}.parquet', columns=['day', 't', 'd', 'cut_minus_act_ms', *SELS])
        E, M = CT.mid_series(day)
        early = X.cut_minus_act_ms.to_numpy(float) >= EF.EARLY_MS
        for s in SELS:
            m = X[s].to_numpy(bool) & early
            st = move_stats(E, M, X.t.to_numpy(float)[m], X.d.to_numpy(float)[m])
            for k, v in st.items():
                stats[s][k].append(v)
        print(day, int((X.MOVE_top10.to_numpy(bool) & early).sum()), flush=True)
    line = {s: {k: quant(np.concatenate(v)) for k, v in d_.items()} for s, d_ in stats.items()}
    files, _, _ = T.load_files()
    chk = json.loads(CFT.CHECK.read_text())
    E, M = tokyo_series(files)
    buys = probe_buys(files, chk)
    st = move_stats(E, M, [b['E'] for b in buys], [b['d'] for b in buys])
    for i, b in enumerate(buys):
        b.update({k: float(v[i]) for k, v in st.items()})
    mv = [b for b in buys if b['kind'] == 'Move']
    probe = dict(buys=buys, after_move={k: quant([b[k] for b in mv]) for k in st},
                 spearman_accept_after_move={k: spearman([b['accept_ms'] for b in mv], [b[k] for b in mv]) for k in st})
    share = {k: float(np.mean(np.concatenate(stats['MOVE_top10'][k]) >= probe['after_move'][k]['p50'])) for k in ('up100', 'n_post200')}
    TR.dump(OUT, dict(asof=dt.datetime.now(dt.timezone.utc).isoformat(), days=list(CF.FWD), line=line, probe=probe,
                      line_share_at_or_above_probe_median=share, tokyo_data_commit=chk['data_commit']))
    print(json.dumps(TR.clean(dict(line=line['MOVE_top10'], probe=probe['after_move'], share=share))), flush=True)


def selftest():
    E = np.array([0., 50., 120., 180., 260., 900., 1000.])
    M = np.array([100., 101., 103., 102., 105., 104., 110.])
    s = move_stats(E, M, [180., 180.], [1., -1.])
    C.require(np.allclose(s['up100'], [102. - 101., -(102. - 101.)]) and np.allclose(s['n100'], [2., 2.]), f'past window {s}')
    C.require(np.allclose(s['n_post200'], [1., 1.]) and np.allclose(s['up_post200'], [3., -3.]), f'post window {s}')
    C.require(np.isnan(move_stats(E, M, [180.], [1.])['up1000'][0]), 'no mid before the series: NaN, not 0')
    try:
        move_stats(E[::-1], M, [180.], [1.])
        raise AssertionError('accepted an unsorted series')
    except ValueError as e_:
        assert 'strictly increasing' in str(e_), e_
    q = quant([1., 2., np.nan, 3.])
    C.require(q['n'] == 3 and q['p50'] == 2., f'quant {q}')
    C.require(abs(spearman([1, 2, 3], [10, 30, 20]) - .5) < 1e-12, 'spearman')
    keep = globals()['OUT']
    with tempfile.NamedTemporaryFile(suffix='.json') as tf:
        globals()['OUT'] = Path(tf.name)
        try:
            run()
            raise AssertionError('run overwrote its result')
        except ValueError as e_:
            assert 'write-once' in str(e_), e_
        finally:
            globals()['OUT'] = keep
    print('chase_move_size selftest PASS')


if __name__ == '__main__':
    dict(selftest=selftest, run=run)[sys.argv[1]]()
