"""STOP_EV PRESENCE (2026-11-16, session aea5bfb9; USER: "why it just has 3.4% on both sides? Does it mean it pulls still quite often"). POST HOC,
descriptive: no readings, no verdicts.

WHAT: the strategy scorecard's STOP_EV / v3 / recorded58 case re-run window for window with its own setup (pf_strategy_scorecard: frozen
sources verified; pf_entry_arrival.load's windows and imputed tape; the v3 teacher heads; pf_stop_ev_latency.Policy with the fresh-EV cache;
clocks 58 / 37 ms; pf_stop_ev_latency.simulate at M50). The policy's entry gate and resting decisions are OBSERVED, never changed, to split
the time without quotes into (a) the MODEL declines - no rung of the side has a fresh EV >= 0 at the decision step - and (b) the LANE - the
model would quote but no order is live yet (placement latency, a pull still landing, a refused or held placement); and to measure the churn:
orders a minute, guard pulls, how each order ends (guard pull / drift / cross / no book / end / filled / refused), how long it is live, and
how soon a pulled rung is re-posted. The fresh EV is split into its two heads - P(fill within 1 s) and E[value | fill] - by rung.
CONTROLS (any failure -> no output): (1) every window's terminal P&L, shares, fills, orders, guard pulls and quote availability (any / both /
bid / ask) equal the scorecard's windows ledger (|diff| <= 1e-6); (2) P(fill) x E[value | fill] equals pf_ev_lag_gap.fresh (the fresh EV the
scorecard's policy reads) on the first window of every day; (3) planted errors on the first window with an order - the legacy 40 / 22 clock,
an entry gate that never closes - must each break (1); (4) the order-end classes sum to the orders and the guard-pull cancels equal the lane's
pull count; (5) synthetic checks of the step-state and order-end helpers.
LIMITS: the scorecard's assumptions (simulated: M50, queue M, unknown depth 0, young fills B1a); consumed days; first minute only.
Usage (run_light, PF_SLOT=claude): pf_stop_ev_presence.py selftest | day <day> | report; driver run_stop_ev_presence.sh
"""
import collections
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_strategy_scorecard as S                         # noqa: E402

A, Q, L, P, V, G, M, K = S.A, S.Q, S.L, S.P, S.V, S.G, S.M, S.K
pd = S.pd
DAYS = ('20261027', '20261031', '20261101', '20261102', '20261103')
CASE = 'STOP_EV__recorded58'
OUTD = Path(K.IO) / 'stop_ev_presence_20261116'
SC = Path(K.IO) / 'strategy_scorecard_20261116_v1'
PLACE, CANCEL = 58.0, 37.0
PCOLS = ['terminal_pnl_usd', 'shares', 'fills', 'orders', 'pulls', 'quote_any_ms', 'quote_both_ms', 'quote_bid_ms', 'quote_ask_ms']
ENDS = ('trig', 'drift', 'cross', 'nobook', 'end', 'filled', 'refused')
PF_BINS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0000001)
DUR_EDGES = (100.0, 250.0, 500.0, 1000.0, 5000.0)
MUT_CLOCK40 = MUT_GATE_OPEN = 0                           # planted errors (the first-window check only)


def step_live(tg, ivs):
    """Steps t in tg with at least one interval [a, b) holding t."""
    out = np.zeros(len(tg), bool)
    for a, b in ivs:
        i, j = np.searchsorted(tg, a, side='left'), np.searchsorted(tg, b, side='left')
        out[i:j] = True
    return out


def end_class(o, why):
    """How an order ended: refused at activation, the reason of its (first) cancel - even when the rest filled while the cancel was
    landing - or filled in full with no cancel."""
    if o['ref']:
        return 'refused'
    if o['tc'] < math.inf:
        return why.get((o['side'], o['rung'], o['tc']), 'end')
    return 'filled' if o['rem'] <= 1e-12 else 'end'


def selftest():
    bad = []
    if step_live(np.array([0., 50., 100., 150.]), [(40., 120.)]).tolist() != [False, True, True, False]:
        bad.append('step_live')
    if step_live(np.array([0., 50.]), []).tolist() != [False, False]:
        bad.append('step_live empty')
    why = {(0, 1, 75.0): 'trig'}
    cases = [(dict(ref=True, rem=30., side=0, rung=1, tc=75.0), 'refused'), (dict(ref=False, rem=0., side=0, rung=1, tc=math.inf), 'filled'),
             (dict(ref=False, rem=30., side=0, rung=1, tc=75.0), 'trig'), (dict(ref=False, rem=30., side=1, rung=1, tc=75.0), 'end'),
             (dict(ref=False, rem=0., side=0, rung=1, tc=75.0), 'trig')]
    for o, want in cases:
        if end_class(o, why) != want:
            bad.append(f'end_class {o} -> {end_class(o, why)}')
    if bad:
        sys.exit(f'STOP-EV-PRESENCE SELFTEST FAILED -> no output: {bad}')
    print('stop-ev-presence selftests passed', flush=True)


def heads_split(heads, W, X):
    """P(fill) and E[value | fill] of a fresh order at every (side, rung, step): the vectors of pf_ev_lag_gap.fresh."""
    pol = L.Policy(heads, W, X, 'EV', (0., 1.), collections.Counter()); keys = []; rows = []
    for k, t in enumerate(W[0]['tg']):
        for s in (0, 1):
            for d in range(P.NR):
                v = pol._vec(s, d, k, float(t), None)
                if v is not None:
                    keys.append((s, d, k)); rows.append(v)
    pf = np.full((2, P.NR, W[0]['n']), np.nan); vf = pf.copy()
    if rows:
        x = np.vstack(rows); a = heads[0].predict(x, num_threads=1); b = heads[1].predict(x, num_threads=1)
        for key, pa, vb in zip(keys, a, b):
            pf[key] = pa; vf[key] = vb
    return pf, vf


def replay(W, X, heads, choice, ev, rec_g=None, rec_d=None):
    pol = L.Policy(heads, W, X, 'EV', choice, collections.Counter(), fresh=ev)

    def wg(s, d, g0):
        def g(k):
            x = False if MUT_GATE_OPEN else g0(k)
            if rec_g is not None:
                rec_g[(s, d, bool(x))] += 1
            return x
        return g

    def wd(s, d, f0):
        def f(k, t, cur):
            x = f0(k, t, cur)
            if rec_d is not None:
                rec_d[(s, d, bool(x))] += 1
            return x
        return f
    dec = {(s, d): wd(s, d, pol.dec(s, d)) for s in (0, 1) for d in range(P.NR)}
    gate = {(s, d): wg(s, d, pol.gate(s, d)) for s in (0, 1) for d in range(P.NR)}
    with G.clocks(40., 22.) if MUT_CLOCK40 else G.clocks(PLACE, CANCEL):
        return L.simulate(W, dec, gate, 50.)


def differs(v, ref):
    return [c for c in PCOLS if abs(float(v[c]) - float(ref[c])) > 1e-6]


def day_run(day):
    if day not in DAYS:
        sys.exit(f'REFUSED: {day} is not a scorecard date')
    selftest(); S.verify(); T0 = time.time()
    ref = pd.read_parquet(SC / day / 'windows.parquet'); ref = ref[ref.case == CASE].set_index('m')
    heads, _ = A.teacher(day); choice = json.loads((G.PREVIOUS / f'receipt_{day}_v3.json').read_text())['qr_choice']
    D, windows, _ = A.load(day)
    st = collections.Counter(); mut = {}; fresh_checked = False
    acc = collections.defaultdict(float)                  # step states, orders, ends, heads: flat sums keyed by strings
    durs = []; pull_after = []; repost = []; flips = collections.Counter(); nsteps_rung = collections.Counter()
    for W, X, src, im in windows():
        m = int(W[0]['m']); st['windows'] += 1
        pf, vf = heads_split(heads, W, X); ev = pf * vf
        if not fresh_checked:
            if not np.array_equal(ev, G.fresh(heads, W, X), equal_nan=True):
                sys.exit(f'STOP-EV-PRESENCE CONTROL FAILED on {day}: P(fill) x E[value | fill] != the fresh EV -> no output')
            fresh_checked = True
        rec_g = collections.Counter(); rec_d = collections.Counter()
        r = replay(W, X, heads, choice, ev, rec_g, rec_d)
        v, _ = Q.account(W, r, src, im, 50., day, CASE)
        bad = differs(v, ref.loc[m])
        if bad:
            sys.exit(f'STOP-EV-PRESENCE CONTROL FAILED on {day} window {m}: {bad} differ from the scorecard -> no output')
        if not mut and v['orders'] > 0:
            for nm in ('MUT_CLOCK40', 'MUT_GATE_OPEN'):
                globals()[nm] = 1
                try:
                    v2, _ = Q.account(W, replay(W, X, heads, choice, ev), src, im, 50., day, CASE)
                    mut[nm] = bool(differs(v2, ref.loc[m]))
                finally:
                    globals()[nm] = 0
        for c in ('quote_any_ms', 'quote_both_ms', 'quote_bid_ms', 'quote_ask_ms', 'quote_observed_ms', 'orders', 'pulls', 'fills', 'shares'):
            acc[c] += float(v[c])
        tg = np.asarray(W[0]['tg'], float); n = len(tg)
        why = {(s_, d_, t_): w_.split(':', 1)[1] for (t_, s_, d_, w_, _) in r['log'] if w_.startswith('cancel:')}
        live = {}
        for s in (0, 1):
            ivs = []
            for d in range(P.NR):
                os_ = r['order_states'][(s, d)]
                for j, o in enumerate(os_):
                    e = end_class(o, why); acc[f'end_{e}'] += 1; acc[f'end_{e}_r{d}'] += 1
                    if o['ref']:
                        continue
                    b = min(o['tr'], o['tfill'], float(r['terminal_time'])); ivs.append((o['ta'], b))
                    durs.append(b - o['ta'])
                    if e == 'trig':
                        pull_after.append(o['tc'] - o['ta']); acc['trig_before_live'] += int(o['tc'] < o['ta'])
                        if j + 1 < len(os_):
                            repost.append(os_[j + 1]['ts'] - o['tc'])
                        else:
                            acc['trig_never_reposted'] += 1
                    acc['orders_filled_any'] += int(len(o['fl']) > 0)
                    acc['filled_after_cancel'] += int(o['tc'] < math.inf and o['rem'] <= 1e-12)
            live[s] = step_live(tg, ivs)
        acc['trig_log'] += sum(1 for x in r['log'] if x[3] == 'cancel:trig')
        fin = np.isfinite(ev)
        opn = {s: (fin[s] & (ev[s] >= 0)).any(axis=0) for s in (0, 1)}
        nob = {s: ~fin[s].any(axis=0) for s in (0, 1)}
        acc['steps'] += n
        for s in (0, 1):
            acc[f'live_{s}'] += live[s].sum(); acc[f'open_{s}'] += opn[s].sum(); acc[f'novec_{s}'] += nob[s].sum()
            acc[f'live_open_{s}'] += (live[s] & opn[s]).sum(); acc[f'live_closed_{s}'] += (live[s] & ~opn[s]).sum()
        acc['live_both'] += (live[0] & live[1]).sum(); acc['open_both'] += (opn[0] & opn[1]).sum()
        acc['live_both_open_both'] += (live[0] & live[1] & opn[0] & opn[1]).sum()
        acc['live_any'] += (live[0] | live[1]).sum(); acc['open_any'] += (opn[0] | opn[1]).sum()
        for s in (0, 1):
            for d in range(P.NR):
                e_ = ev[s, d]; f_ = np.isfinite(e_); p_ = pf[s, d]; w_ = vf[s, d]
                acc[f'h_n_{s}{d}'] += f_.sum(); acc[f'h_evpos_{s}{d}'] += (f_ & (e_ >= 0)).sum(); acc[f'h_vpos_{s}{d}'] += (f_ & (w_ >= 0)).sum()
                acc[f'h_pf_{s}{d}'] += p_[f_].sum(); acc[f'h_vf_{s}{d}'] += w_[f_].sum()
                sg = np.sign(e_[f_]); flips[(s, d)] += int((sg[1:] != sg[:-1]).sum()); nsteps_rung[(s, d)] += int(f_.sum())
                for g, (lo, hi) in enumerate(zip(PF_BINS[:-1], PF_BINS[1:])):
                    b_ = f_ & (p_ >= lo) & (p_ < hi)
                    acc[f'b_n_{g}'] += b_.sum(); acc[f'b_vpos_{g}'] += (b_ & (w_ >= 0)).sum(); acc[f'b_vf_{g}'] += w_[b_].sum()
        for (s, d, x), c in rec_g.items():
            acc[f'gate_{int(x)}_r{d}'] += c
        for (s, d, x), c in rec_d.items():
            acc[f'dec_{int(x)}_r{d}'] += c
        if st['windows'] % 24 == 0:
            print(json.dumps(dict(day=day, windows=st['windows'], seconds=round(time.time() - T0), rss_mib=K.rss_mib())), flush=True)
            A.E.mem_guard('stop-ev presence')
    fails = []
    if st['windows'] != len(ref):
        fails.append(f"windows {st['windows']} vs the scorecard's {len(ref)}")
    if not mut or not all(mut.values()):
        fails.append(f'planted errors not caught: {mut}')
    if sum(acc[f'end_{e}'] for e in ENDS) != acc['orders']:
        fails.append('order-end classes do not sum to the orders')
    if acc['end_trig'] != acc['pulls'] or acc['trig_log'] != acc['pulls']:
        fails.append(f"guard-pull cancels {acc['end_trig']} / log {acc['trig_log']} vs lane pulls {acc['pulls']}")
    if fails:
        sys.exit(f'STOP-EV-PRESENCE CONTROL FAILED on {day} -> no output: {fails}')
    durs = np.asarray(durs); pull_after = np.asarray(pull_after); repost = np.asarray(repost)
    out = dict(day=day, stat=dict(st), mutations=mut, acc={k_: float(v_) for k_, v_ in acc.items()},
               dur_q=[float(x) for x in np.quantile(durs, [0.1, 0.25, 0.5, 0.75, 0.9])] if len(durs) else None,
               dur_le={str(e): float((durs < e).mean()) for e in DUR_EDGES} if len(durs) else None, n_live_orders=len(durs),
               pull_after_q=[float(x) for x in np.quantile(pull_after, [0.1, 0.25, 0.5, 0.75, 0.9])] if len(pull_after) else None,
               repost_q=[float(x) for x in np.quantile(repost, [0.1, 0.25, 0.5, 0.75, 0.9])] if len(repost) else None, n_repost=len(repost),
               flips={f'{s}{d}': [flips[(s, d)], nsteps_rung[(s, d)]] for s in (0, 1) for d in range(P.NR)}, seconds=time.time() - T0)
    OUTD.mkdir(parents=True, exist_ok=True)
    np.save(OUTD / f'durs_{day}.npy', durs); np.save(OUTD / f'pull_after_{day}.npy', pull_after); np.save(OUTD / f'repost_{day}.npy', repost)
    (OUTD / f'day_{day}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(day=day, windows=st['windows'], mutations=mut, seconds=round(out['seconds']))), flush=True)


def report():
    selftest()
    J = {d: json.loads((OUTD / f'day_{d}.json').read_text()) for d in DAYS}
    a = collections.Counter()
    for d in DAYS:
        a.update(J[d]['acc'])
    durs = np.concatenate([np.load(OUTD / f'durs_{d}.npy') for d in DAYS]); pa = np.concatenate([np.load(OUTD / f'pull_after_{d}.npy') for d in DAYS])
    rp = np.concatenate([np.load(OUTD / f'repost_{d}.npy') for d in DAYS])
    win = sum(J[d]['stat']['windows'] for d in DAYS); obs = a['quote_observed_ms']; st_ = a['steps']
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    pct = lambda x, y: 100.0 * x / y if y else float('nan')    # noqa: E731
    Ls = [f'STOP_EV PRESENCE (POST HOC, descriptive): the scorecard\'s STOP_EV / v3 / recorded58 re-run on its {win} first-minute windows (5 '
          f'consumed dates), policy observed not changed; as of {asof}',
          f'controls: every window equals the scorecard (P&L, shares, fills, orders, pulls, quote availability); fresh EV = the policy\'s cache; '
          f'planted errors caught on every date ({", ".join(sorted(J[DAYS[0]]["mutations"]))}); order ends sum to the orders, guard-pull cancels = '
          f'the lane\'s pulls', '',
          'PART 1: quote availability (share of the quoting time, continuous; the scorecard\'s definition)',
          f'  any side {pct(a["quote_any_ms"], obs):5.2f}% | both sides {pct(a["quote_both_ms"], obs):5.2f}% | bid {pct(a["quote_bid_ms"], obs):5.2f}% | '
          f'ask {pct(a["quote_ask_ms"], obs):5.2f}%', '',
          'PART 2: at the 50 ms decision steps - the MODEL would quote (some rung of the side has fresh EV >= 0) vs an order IS live',
          f'  steps {st_:,.0f} ({win} windows)']
    for s, nm in ((0, 'bid'), (1, 'ask')):
        Ls.append(f'  {nm}: model would quote {pct(a[f"open_{s}"], st_):5.2f}% of steps | live {pct(a[f"live_{s}"], st_):5.2f}% | live when the model '
                  f'would quote {pct(a[f"live_open_{s}"], a[f"open_{s}"]):5.1f}% | live when it would not {pct(a[f"live_closed_{s}"], st_ - a[f"open_{s}"]):5.2f}%'
                  f' | no fresh vector on any rung {pct(a[f"novec_{s}"], st_):4.1f}%')
    pb = a['open_0'] / st_; pa_ = a['open_1'] / st_
    Ls.append(f'  both: model would quote both {pct(a["open_both"], st_):5.2f}% (if the sides were independent: {100 * pb * pa_:5.2f}%) | both live '
              f'{pct(a["live_both"], st_):5.2f}% | both live when the model would quote both {pct(a["live_both_open_both"], a["open_both"]):5.1f}%')
    Ls.append(f'  any: model would quote {pct(a["open_any"], st_):5.2f}% | live {pct(a["live_any"], st_):5.2f}%')
    Ls += ['', 'PART 3: churn - orders and how they end (5 dates)',
           f'  orders {a["orders"]:,.0f} ({a["orders"] / win:.1f} a window-minute), guard pulls {a["pulls"]:,.0f} ({a["pulls"] / win:.1f}), '
           f'orders with a fill {a["orders_filled_any"]:,.0f} ({pct(a["orders_filled_any"], a["orders"]):.2f}%; filled in full while a cancel was '
           f'landing {a["filled_after_cancel"]:,.0f}), fills {a["fills"]:,.0f}',
           '  end: ' + ' | '.join(f'{e} {pct(a[f"end_{e}"], a["orders"]):5.1f}%' for e in ENDS),
           '  guard pulls by rung: ' + ' | '.join(f'r{d} {a[f"end_trig_r{d}"]:,.0f}' for d in range(P.NR)),
           f'  live time of an order (activation -> removal, ms): p10 / p25 / p50 / p75 / p90 = ' + ' / '.join(f'{x:,.0f}' for x in np.quantile(durs, [.1, .25, .5, .75, .9])),
           '  live less than: ' + ' | '.join(f'{e:,.0f} ms {100 * (durs < e).mean():5.1f}%' for e in DUR_EDGES),
           f'  guard pull decided after activation (ms): p10 / p25 / p50 / p75 / p90 = ' + ' / '.join(f'{x:,.0f}' for x in np.quantile(pa, [.1, .25, .5, .75, .9])) +
           f'; decided before the order was live {pct(a["trig_before_live"], a["end_trig"]):.1f}%',
           f'  pulled rung re-posted after (ms, decision to decision): p10 / p25 / p50 / p75 / p90 = ' + ' / '.join(f'{x:,.0f}' for x in np.quantile(rp, [.1, .25, .5, .75, .9])) +
           f'; never re-posted in the window {a["trig_never_reposted"]:,.0f}',
           '  entry gate closed (fresh EV < 0) / evaluations by rung: ' + ' | '.join(
               f'r{d} {pct(a[f"gate_1_r{d}"], a[f"gate_1_r{d}"] + a[f"gate_0_r{d}"]):.1f}%' for d in range(P.NR)),
           '  resting decisions that pull (EV < 0) by rung: ' + ' | '.join(
               f'r{d} {pct(a[f"dec_1_r{d}"], a[f"dec_1_r{d}"] + a[f"dec_0_r{d}"]):.1f}% of {a[f"dec_1_r{d}"] + a[f"dec_0_r{d}"]:,.0f}' for d in range(P.NR)), '',
           'PART 4: why the model declines - a fresh order\'s two heads by rung (bid | ask): EV >= 0 | mean P(fill 1 s) | E[value | fill] >= 0 | mean E[value | fill] c']
    for d in range(P.NR):
        parts = []
        for s in (0, 1):
            nn = a[f'h_n_{s}{d}']
            parts.append(f'{pct(a[f"h_evpos_{s}{d}"], nn):5.1f}% {a[f"h_pf_{s}{d}"] / nn:6.3f} {pct(a[f"h_vpos_{s}{d}"], nn):5.1f}% {a[f"h_vf_{s}{d}"] / nn:+7.2f}')
        fl = sum(J[x]['flips'][f'0{d}'][0] + J[x]['flips'][f'1{d}'][0] for x in DAYS)
        Ls.append(f'  r{d}: ' + ' | '.join(parts) + f' | EV sign changes {fl / win:.1f} a window-minute')
    Ls.append('  by P(fill): ' + ' | '.join(f'[{lo:g}, {min(hi, 1):g}) {pct(a[f"b_n_{g}"], sum(a[f"b_n_{h}"] for h in range(len(PF_BINS) - 1))):4.1f}% of rows, '
                                            f'E[v|fill] >= 0 {pct(a[f"b_vpos_{g}"], a[f"b_n_{g}"]):4.1f}%, mean {a[f"b_vf_{g}"] / max(a[f"b_n_{g}"], 1):+6.2f} c'
                                            for g, (lo, hi) in enumerate(zip(PF_BINS[:-1], PF_BINS[1:]))))
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / 'stop_ev_presence.txt').write_text('\n'.join(Ls) + '\n')
    print('\n'.join(Ls), flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        selftest()
    elif cmd == 'day':
        day_run(sys.argv[2])
    elif cmd == 'report':
        report()
    else:
        sys.exit('usage: pf_stop_ev_presence.py selftest | day <day> | report')
