"""take_logic_v2_selftest.py - synthetic proof of take_logic_v2.py (P-2026-005 MAKER-TRACE, TAKE-LOGIC v2; take_logic_v2_contract.md fixtures 1-5).
The REAL chain_context (position / pending / btc_ret / book_tob / time_left; synthetic markets through chain_context.build_market + join_windows,
built by the v1 selftest's SM class, reused unchanged), the REAL take_logic v1 functions, and chain_take's ACTUAL classify / combine: every
synthetic take's recorded (chain_context.position 'recorded' at its settle) / certain(0) ('certain' at its x_lo_hard) / simultaneous classes and
its take_class come from them (take_core's pos_fn path). Fixtures:
 (1) bias control: a random-side, random-time maker (hazard 0.15 / s, side +-1 at random) on a square-wave inventory |q| in {20, 300} sh (25 s
     halves, per-market phase) with 40-sh maker fills (exponential gaps, mean 2 s) that flip the sign only at |q| = 20, so the recorded /
     certain variants disagree mostly at small |q| (checked: fine-class and grouped disagreement shares by |q| bin, coverage table = direct
     count). The v1 outcome (classified reduce, ambiguous dropped; take_logic.analyse) must show the spurious RR_size > 1 and O1 RR_size must
     not be supported while testable (>= 5 supported markets, >= 30 events); the decision reads withdrawn.
 (2) planted size policy: opposing-take hazard 0.03 at |q(t)| <= 50, 0.15 at > 200 (same-side noise 0.03): O1 RR_size supported, rank 1 and
     pooled; the decision is not withdrawn.
 (3) shift checks: the raw-schedule shift keeps every pair's raw count in every replicate, stays in its pair, re-derives the direction at the
     destination (the tool's null event sets equal a brute-force rebuild for 5 replicates x 3 clocks; null strata counts equal), sign changes
     between source and destination exist (the check can fire), collisions are invariant, offsets deterministic; a hand-made schedule (3
     takes with mixed sides in one second, 2 in another, a q = 0 take, a side_mixed record) counts each outcome once per second.
 (4) mutants MT_TL2_MUT shift_labels / drop_ambiguous / sum_types, flipped in-process: each must be caught by its named check evaluating
     False without raising (the suite fails if any mutant passes).
 (5) clean ledger: a planted excluded leg removes exactly its pair's seconds from its window start (k >= 98: 202 maker-seconds), the ratio
     codes are -1 there and unchanged elsewhere, a take there leaves the clean panel only.
Also: O4 grouping (hand table), the decision reading (hand verdict tables), the census fix (synthetic rows + falsifier), the variant
reproduction (positive control + refusals), the overwrite refusal (asserting its message, also through the production entry), production
refusals, and every writer end to end (two-market subset).
Usage (run_light, MT_RUN=v2): take_logic_v2_selftest.py"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import take_logic_v2 as V2  # noqa: E402
import take_logic_selftest as TLS  # noqa: E402   (the v1 selftest's synthetic market builder SM, reused unchanged)

TL, CT, CX, C, X = V2.TL, V2.CT, V2.CX, V2.C, V2.X
NAN = float('nan')
RES = []
FIX = {}
A1, A2, A3 = TL.TOP3
DAYS = ('20261024', '20261025')
SMALL, JIT, BIG, TAKE_Q = 20., 40., 280., 0.25
HALF_MS, JIT_MEAN_MS = 25_000, 2_000.
KIND_ID = dict(random=1, size=2)
H_RANDOM = 0.15
H_SIZE = dict(small=0.03, mid=0.06, large=0.15, same=0.03)
REPS = (0, 1, 17, 250, 499)
EXPECT_CATCH = dict(shift_labels=('F3_DIR',), drop_ambiguous=('COUNT',), sum_types=('DEDUPE',))
VERDICTS = ('supported', 'inconsistent', 'insufficient_support', 'not_supported')
SELFTEST_CPU_CAP = 8.0
O1_ONLY = ('O1',)


def check(name, ok, info=''):
    RES.append((name, bool(ok), str(info)[:700]))


def expect(name, fn, needle):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        check(name, needle in str(e), f'{type(e).__name__}: {e}')
        return
    check(name, False, 'did not raise')


def section(name, fn, *a):
    if time.process_time() > SELFTEST_CPU_CAP:
        check(f'{name}: skipped - selftest CPU {time.process_time():.1f} s > {SELFTEST_CPU_CAP} s (budget guard)', False)
        return None
    try:
        return fn(*a)
    except Exception as e:  # noqa: BLE001
        check(f'{name}: section crashed', False, f'{type(e).__name__}: {e} | {traceback.format_exc()[-1500:]}')
        return None


def _same(a, b):
    a, b = float(a), float(b)
    return (np.isnan(a) and np.isnan(b)) or a == b


def _o1(RK, pn='full', v='D0'):
    return {sc: RK[(pn, v, sc, 'O1', 'RR_size')] for sc in V2.SCOPES2}


def _brief(r):
    return dict(RR_MH=round(float(r['RR_MH']), 4), p_hi=r['p_hi'], markets=f"{r['n_markets_rr_gt1']}/{r['n_markets_supported']}",
                events=r['n_events_informative'], verdict=r['verdict_rule'])


# ------------------------------------------------------------------ synthetic worlds (classes through chain_take.classify / combine) ------------------------------------------------------------------
def classify_takes(M, day, mid, tk, sids, mk):
    """chain_take's take variants on the built market (take_core with pos_fn = chain_context.position): recorded = 'recorded' at the take's
    settle (legs with settle < it), certain(0) = 'certain' at its x_lo_hard (legs with hi < it), simultaneous = recorded + the other own
    actions at its settle ms (present only then); classes by chain_take.classify, take_class by chain_take.combine."""
    if not tk:
        return []
    xlo = np.array([x[0] for x in tk], np.int64)
    hi = np.array([x[1] for x in tk], np.int64)
    s = np.array([x[2] for x in tk], np.int64)
    rec = np.asarray(CX.position(M, A1, hi)['recorded'], float)
    cert = np.asarray(CX.position(M, A1, xlo)['certain'], float)
    dq = s * TAKE_Q
    a_set = np.concatenate([np.array([x[0] for x in mk], np.int64), hi])
    a_dq = np.concatenate([np.array([x[1] * x[2] for x in mk], float), dq])
    a_id = np.concatenate([np.full(len(mk), -1, np.int64), np.arange(len(tk), dtype=np.int64)])
    tie, n_tie = np.zeros(len(tk)), np.zeros(len(tk), np.int64)
    for i in range(len(tk)):
        m = (a_set == hi[i]) & (a_id != i)
        tie[i], n_tie[i] = float(a_dq[m].sum()), int(m.sum())
    c_rec = CT.classify(rec, dq)
    c_cert = CT.classify(cert, dq).astype(np.int16)
    c_sim = np.where(n_tie > 0, CT.classify(rec + tie, dq), -1).astype(np.int16)
    cls = CT.combine(c_rec, c_cert, c_sim)
    nm = np.array(CT.CLS, dtype=object)
    return [dict(day=day, mid=int(mid), rank=1, addr=A1, sid=sids[i], s=int(s[i]), dq=float(dq[i]), p_avg=0.50 + 0.01 * int(s[i]), x_lo_hard_ms=int(xlo[i]),
                 hi_ms=int(hi[i]), settle_ms=int(hi[i]), status='ok', take_class=str(nm[cls[i]]), tk_hash=f'0xt{mid}{i:04d}', q_all=TAKE_Q,
                 cls_rec=str(nm[c_rec[i]]), cls_sim=str(nm[c_sim[i]]) if c_sim[i] >= 0 else '', cls_cert_0=str(nm[c_cert[i]])) for i in range(len(tk))]


class World:
    """n_mk synthetic markets over two days (fixture 1 'random', fixture 2 'size'; module docstring). Rank 1 = the maker; rank 2 enters at
    0 and is filled 30 sh at ws + 50 s (never takes); rank 3 has no leg. Takes 0.25 sh, x_lo_hard in (t, t + 1000], hi = settle =
    x_lo_hard + 2000 (settle-based windows); the generator's q(t) = legs with settle <= t = the tool's D0 q."""

    def __init__(self, kind, seed, n_mk=8):
        rng = np.random.default_rng(seed)
        self.kind, self.markets, rows = kind, [], []
        for j in range(n_mk):
            day = DAYS[j % 2]
            ws = CX.day_ms(day) + 3_600_000 * (2 + j)
            sm = TLS.SM(day, 960_000 + 100 * KIND_ID[kind] + j, ws, rng)
            sg = 1 if j % 2 == 0 else -1
            mk = [(ws - 60_000, sg, SMALL)]
            b0 = ws - 50_000 + (j * 7_000) % 50_000
            m = 0
            while b0 + HALF_MS * m < sm.we:
                mk.append((b0 + HALF_MS * m, sg if m % 2 == 0 else -sg, BIG))
                m += 1
            tj, jd = ws - 10_000, 0
            while True:
                tj += 1 + int(rng.exponential(JIT_MEAN_MS))
                if tj >= sm.we:
                    break
                mk.append((tj, -sg if jd == 0 else sg, JIT))
                jd ^= 1
            mk.sort()
            ms_ = np.array([x[0] for x in mk], np.int64)
            msq = np.array([x[1] * x[2] for x in mk], float)
            tk = []
            for k in range(300):
                t = ws + 1_000 * k
                q = float(msq[ms_ <= t].sum()) + sum(s_ * TAKE_Q for _, h_, s_ in tk if h_ <= t)
                sgq = 1 if q > 1e-6 else (-1 if q < -1e-6 else 0)
                u = rng.random()
                if kind == 'random':
                    if u >= H_RANDOM:
                        continue
                    s = 1 if rng.random() < 0.5 else -1
                else:
                    if sgq == 0:
                        continue
                    h = H_SIZE['large'] if abs(q) > 200 else (H_SIZE['small'] if abs(q) <= 50 else H_SIZE['mid'])
                    if u < h:
                        s = -sgq
                    elif u < h + H_SIZE['same']:
                        s = sgq
                    else:
                        continue
                xlo = t + 1 + int(rng.integers(0, 1_000))
                tk.append((xlo, xlo + 2_000, s))
            for st_, s_, q_ in mk:
                sm.fill(A1, st_, s_, q_, 0.50)
            sm.fill(A2, ws + 50_000, 1, 30., 0.50)
            sids = [sm.fill(A1, hi_, s_, TAKE_Q, 0.50 + 0.01 * s_, role='taker') for _, hi_, s_ in tk]
            M = sm.build()
            rows += classify_takes(M, day, sm.mid, tk, sids, mk)
            self.markets.append((day, sm.mid, M))
        self.takes = pd.DataFrame(rows, columns=list(V2.TAKE_READ))


def base(w, k=None):
    """The mutant-independent part: panel (take_logic.market_part / assemble), events (take_logic.events), O4 status, state codes."""
    mk = w.markets if k is None else w.markets[:k]
    parts = [TL.market_part(d, m, M) for d, m, M in mk]
    P = TL.assemble(parts)
    T = TL.prepare_takes(w.takes[w.takes['mid'].isin([m for _, m, _ in mk]).to_numpy()])
    T2, EV1 = TL.events(P, T)
    o4, nav = V2.o4_status(T2)
    return dict(parts=parts, P=P, T=T, T2=T2, EV1=EV1, o4=o4, nav=nav, codes={v: TL.variant_codes(P, v) for v in V2.VARIANTS})


def derived(B):
    """The mutant-dependent part (MT_TL2_MUT read at call time): raw schedule, outcomes, directions."""
    RT = V2.raw_takes(B['P'], B['T2'], B['o4'], B['nav'])
    E, D = V2.outcomes(B['P'], RT, B['codes'])
    return RT, E, D


def ratios(B, views=V2.ALL_VIEWS, outs=V2.OUTCOMES):
    RT, E, D = derived(B)
    return V2.ratio_tables(B['P'], B['codes'], E, RT, TL.shifted(B['P'], RT['gi']), views=views, outs=outs)[0]


# ------------------------------------------------------------------ fixtures 1 / 2 ------------------------------------------------------------------
def f1_pre(B):
    """Fine-class (v1 timing_sensitive) and grouped (O4 reduces_some) disagreement concentrate at |q| (0, 50] (coverage table = direct count)."""
    RT, E, D = derived(B)
    cov = V2.coverage_table(B['P'], B['codes'], E, RT, D, B['EV1'])
    c = cov[(cov['panel'] == 'full') & (cov['variant'] == 'D0') & (cov['scope'] == 'rank1')]
    sm_, lg_ = c[c['qbin'] == '(0,50]'].iloc[0], c[c['qbin'] == '(200,500]'].iloc[0]
    asg = np.asarray(B['T2']['pop_status'], dtype=object) == 'assigned'
    gi = B['T2']['gi'].to_numpy(np.int64)[asg]
    qb = B['codes']['D0']['qb'][gi]
    ts = V2._str_arr(B['T2']['take_class'])[asg] == 'timing_sensitive'
    some = B['o4'][asg] == 1
    direct = dict(fine_small=float(ts[qb == 1].mean()), fine_large=float(ts[qb == 3].mean()), grouped_small=float(some[qb == 1].mean()),
                  grouped_large=float(some[qb == 3].mean()))
    tab = dict(fine_small=float(sm_['v1_fine_disagreement_share_takes']), fine_large=float(lg_['v1_fine_disagreement_share_takes']),
               grouped_small=float(sm_['o4_disagreement_share_takes']), grouped_large=float(lg_['o4_disagreement_share_takes']))
    same = all(abs(direct[k] - tab[k]) < 1e-12 for k in direct)
    conc = all(tab[f'{x}_small'] >= 0.25 and tab[f'{x}_small'] - tab[f'{x}_large'] >= 0.2 and tab[f'{x}_small'] >= 3 * tab[f'{x}_large']
               for x in ('fine', 'grouped'))
    FIX['F1_disagreement_shares_rank1_D0'] = dict(tab, takes_small=int(sm_['raw_takes']), takes_large=int(lg_['raw_takes']))
    return same and conc, dict(table=tab, direct=direct)


def f1_v1(B):
    """The v1 outcome on the same panel (take_logic.analyse: classified reduce, ambiguous dropped)."""
    R = TL.analyse(B['P'], B['T2'], B['EV1'], full=False)
    V = {(m['mechanism'], m['scope']): m for m in R['mechanisms']}
    rr = {sc: float(V[('inventory_size_driven', sc)]['D0']['RR_MH']) for sc in V2.SCOPES2}
    vd = {sc: (V[('inventory_size_driven', sc)]['D0']['verdict_rule'], V[('inventory_size_driven', sc)]['D0']['p_hi']) for sc in V2.SCOPES2}
    FIX['F1_v1_reduce_RR_size_D0'] = {sc: dict(RR_MH=round(rr[sc], 4), verdict=vd[sc][0], p_hi=vd[sc][1]) for sc in V2.SCOPES2}
    return all(rr[sc] > 1 for sc in V2.SCOPES2), FIX['F1_v1_reduce_RR_size_D0']


def f1_v2(B, views=V2.ALL_VIEWS, outs=O1_ONLY):
    RK = ratios(B, views, outs)
    o1 = _o1(RK)
    not_sup = all(o1[sc]['verdict_rule'] != 'supported' for sc in V2.SCOPES2)
    live = all(o1[sc]['n_markets_supported'] >= TL.MIN_MARKETS and o1[sc]['n_events_informative'] >= TL.MIN_EVENTS for sc in V2.SCOPES2)
    return not_sup and live, {sc: _brief(o1[sc]) for sc in V2.SCOPES2}, RK


def f2(B):
    RK = ratios(B, V2.ALL_VIEWS, O1_ONLY)
    o1 = _o1(RK)
    dec = {sc: V2.decision(RK, sc) for sc in V2.SCOPES2}
    FIX['F2_O1_RR_size_full_D0'] = {sc: _brief(o1[sc]) for sc in V2.SCOPES2}
    FIX['F2_reading'] = {sc: dec[sc]['reading'] for sc in V2.SCOPES2}
    return all(o1[sc]['verdict_rule'] == 'supported' for sc in V2.SCOPES2), FIX['F2_O1_RR_size_full_D0'], dec


# ------------------------------------------------------------------ fixture 3: the raw-schedule shift ------------------------------------------------------------------
def f3_dir(B):
    """The tool's O1 / O2 null event sets (raw_null) equal a brute-force rebuild: every raw take moved by its pair's offset, its direction
    taken at the DESTINATION second's sign(q) of the clock, de-duplicated per second; the size-definition null strata counts equal too."""
    RT, E, D = derived(B)
    P, codes = B['P'], B['codes']
    DEST = TL.shifted(P, RT['gi'])
    bad, fire, n_cmp = [], 0, 0
    for v in V2.VARIANTS:
        sg = np.asarray(codes[v]['sg'], np.int64)
        RN = V2.raw_null(P, RT, sg, DEST)
        code, grp, S, G = V2.panel_def(P['rank'], codes[v], 'size', 'full', P['lg'])
        nul = V2.pair_counts(RN['O1'][0], RN['O1'][1], code, S, G)
        for r in REPS:
            ref = {'O1': set(), 'O2': set()}
            for j in range(len(RT['gi'])):
                if not RT['side_ok'][j]:
                    continue
                dst = int(DEST[r, j])
                d = int(RT['s'][j]) * int(sg[dst])
                if d < 0:
                    ref['O1'].add(dst)
                elif d > 0:
                    ref['O2'].add(dst)
                fire += int(int(sg[dst]) != int(sg[RT['gi'][j]]))
            for o in ('O1', 'O2'):
                ri, ci = RN[o]
                n_cmp += 1
                got = set(ci[ri == r].tolist())
                if got != ref[o]:
                    bad.append((v, r, o, len(got), len(ref[o])))
            ev = np.array(sorted(ref['O1']), np.int64)
            cc = np.asarray(code, np.int64)[ev] if len(ev) else np.zeros(0, np.int64)
            bf = np.bincount(cc[cc >= 0], minlength=S * G).reshape(S, G).astype(float)
            if not np.array_equal(bf, nul[r]):
                bad.append((v, r, 'O1 size strata counts'))
    return (not bad) and fire > 0, dict(bad=bad[:6], comparisons=n_cmp, source_destination_sign_changes=fire)


def f3_counts(B):
    RT, E, D = derived(B)
    P = B['P']
    DEST = TL.shifted(P, RT['gi'])
    cs = V2.collision_stats(P, RT, DEST)
    npair = len(P['pairs'])
    b_ = np.bincount(P['pair'][RT['gi']], minlength=npair)
    kept = all(np.array_equal(np.bincount(P['pair'][DEST[r]], minlength=npair), b_) for r in range(TL.N_SHIFT))
    inside = bool(np.all(P['pair'][DEST] == P['pair'][RT['gi']][None, :]))
    det = bool(np.array_equal(P['U'], TL.assemble(B['parts'])['U']))
    SH = TL.shifted(P, np.flatnonzero(E['O4']))
    nb = TL.null_counts(np.zeros(int(P['N']), np.int64), 1, 1, SH).reshape(TL.N_SHIFT)
    bench = bool(np.all(nb == int(E['O4'].sum())) and int(E['O4'].sum()) > 0)
    ok = (kept and inside and det and bench and cs['raw_counts_per_pair_preserved'] and cs['shifted_takes_stay_in_pair']
          and cs['collisions_invariant_under_shift'] and cs['raw_takes'] == RT['n_raw'])
    return ok, dict(kept=kept, inside=inside, deterministic=det, benchmark_counts=bench, collisions=cs)


def count_check(B):
    """Every assigned take record is in the raw schedule (ambiguous included, per pair); O3 = the distinct take seconds."""
    RT, E, D = derived(B)
    T2, P = B['T2'], B['P']
    asg = np.asarray(T2['pop_status'], dtype=object) == 'assigned'
    gi_a = T2['gi'].to_numpy(np.int64)[asg]
    npair = len(P['pairs'])
    amb = int((V2._str_arr(T2['take_class'])[asg] == 'timing_sensitive').sum())
    per_pair = np.array_equal(np.bincount(P['pair'][RT['gi']], minlength=npair), np.bincount(P['pair'][gi_a], minlength=npair))
    ok = (RT['n_raw'] == int(asg.sum()) == RT['n_assigned'] and per_pair and amb > 0 and int(E['O3'].sum()) == int(len(np.unique(gi_a))))
    return ok, dict(n_raw=RT['n_raw'], n_assigned=int(asg.sum()), ambiguous_assigned=amb, per_pair=per_pair, O3=int(E['O3'].sum()),
                    distinct_seconds=int(len(np.unique(gi_a))))


def dedupe_case(B0):
    """Hand-made schedule on one market: rank 1 at k 40 buy / buy / sell, at k 41 sell / sell; rank 3 (q = 0) a buy at k 50; rank 1 a
    side_mixed record at k 60. Each outcome counts a second once; q = 0 and side_unknown are statuses; collisions 3, mixed seconds 1."""
    P, codes = B0['P'], B0['codes']
    pm = {p['rank']: p for p in P['pairs']}
    d, mid, ws = pm[1]['day'], pm[1]['mid'], pm[1]['ws']
    spec = [(1, 40, 1, 'ok'), (1, 40, 1, 'ok'), (1, 40, -1, 'ok'), (1, 41, -1, 'ok'), (1, 41, -1, 'ok'), (3, 50, 1, 'ok'), (1, 60, 1, 'side_mixed')]
    rows = []
    for i, (rk, k, s, st) in enumerate(spec):
        xlo = ws + 1_000 * k + 100 + i
        rows.append(dict(day=d, mid=mid, rank=rk, addr=TL.TOP3[rk - 1], sid=f'dd-{i}', s=s, dq=TAKE_Q * s, p_avg=0.5, x_lo_hard_ms=xlo, hi_ms=xlo + 2_000,
                         settle_ms=xlo + 2_000, status=st, take_class='add', tk_hash=f'0xdd{i}', q_all=TAKE_Q, cls_rec='add', cls_sim='', cls_cert_0='add'))
    T2, _ = TL.events(P, TL.prepare_takes(pd.DataFrame(rows, columns=list(V2.TAKE_READ))))
    o4, nav = V2.o4_status(T2)
    RT = V2.raw_takes(P, T2, o4, nav)
    E, D = V2.outcomes(P, RT, codes)
    g = {k: pm[1]['base'] + k for k in (40, 41, 60)}
    g50 = pm[3]['base'] + 50
    sg = np.asarray(codes['D0']['sg'], np.int64)
    s40, s41, s60, s50 = int(sg[g[40]]), int(sg[g[41]]), int(sg[g[60]]), int(sg[g50])
    pre = s40 != 0 and s41 != 0 and s60 != 0 and s50 == 0
    o1, o2, o3 = E[('O1', 'D0')], E[('O2', 'D0')], E['O3']
    binary = all(int(np.max(e, initial=0)) <= 1 for e in E.values())
    ok_o3 = int(o3.sum()) == 4 and all(int(o3[x]) == 1 for x in (g[40], g[41], g[60], g50))
    ok_dir = (int(o1[g[40]]) == 1 and int(o2[g[40]]) == 1 and int(o1[g[41]]) == int(s41 > 0) and int(o2[g[41]]) == int(s41 < 0)
              and int(o1[g50]) == 0 and int(o2[g50]) == 0 and int(o1[g[60]]) == 0 and int(o2[g[60]]) == 0)
    dirs = Counter(V2.DIR_LAB[i] for i in D['D0'])
    ok_st = dirs['q0'] == 1 and dirs['side_unknown'] == 1 and dirs['opposing'] + dirs['same'] == 5
    cs = V2.collision_stats(P, RT, TL.shifted(P, RT['gi']))
    ok_coll = (cs['collisions_extra_takes'] == 3 and cs['seconds_with_ge2_takes'] == 2 and cs['mixed_side_seconds'] == 1 and cs['distinct_seconds'] == 4
               and cs['collisions_invariant_under_shift'] and cs['raw_counts_per_pair_preserved'] and cs['shifted_takes_stay_in_pair'])
    return pre and binary and ok_o3 and ok_dir and ok_st and ok_coll, dict(signs=(s40, s41, s50, s60), binary=binary, o3=ok_o3, direction=ok_dir,
                                                                          statuses=dict(dirs), collisions=cs)


# ------------------------------------------------------------------ fixture 5: the clean-ledger filter ------------------------------------------------------------------
def f5():
    rng = np.random.default_rng(55)
    day = DAYS[0]
    ws = CX.day_ms(day) + 3_600_000 * 15
    sm = TLS.SM(day, 979_501, ws, rng)
    sm.fill(A1, ws - 100_000, 1, 400., 0.50)
    sm.fill(A1, ws + 100_000, 1, 5., 0.50, status='record_leg_excluded')      # window [ws + 98 s, ws + 100 s]: ledger-conditional from k = 98
    sm.fill(A2, ws + 20_000, -1, 7., 0.50)
    M = sm.build()
    P = TL.assemble([TL.market_part(day, sm.mid, M)])
    pm = {p['rank']: p for p in P['pairs']}
    keep = V2.clean_mask(P)
    b1, n1 = pm[1]['base'], pm[1]['n']
    ok_r1 = bool(np.array_equal(keep[b1:b1 + n1], np.arange(n1) < 98))
    ok_oth = all(bool(keep[pm[r]['base']:pm[r]['base'] + pm[r]['n']].all()) for r in (2, 3))
    n_rem = int((~keep).sum())
    C0 = TL.variant_codes(P, 'D0')
    ok_codes, nontriv = True, 0
    for dn in V2.DEFS:
        cf, gf, S, G = V2.panel_def(P['rank'], C0, dn, 'full', P['lg'])
        cc, gc, _, _ = V2.panel_def(P['rank'], C0, dn, 'clean', P['lg'])
        ok_codes &= bool(np.all(cc[~keep] == -1) and np.all(gc[~keep] == -1) and np.array_equal(cc[keep], cf[keep]) and np.array_equal(gc[keep], gf[keep]))
        nontriv += int((cf[~keep] >= 0).sum())
    rows = [dict(day=day, mid=sm.mid, rank=1, addr=A1, sid=f'c-{i}', s=-1, dq=-TAKE_Q, p_avg=0.5, x_lo_hard_ms=ws + 1_000 * k + 200,
                 hi_ms=ws + 1_000 * k + 2_200, settle_ms=ws + 1_000 * k + 2_200, status='ok', take_class='reduce_partial', tk_hash=f'0xc{i}', q_all=TAKE_Q,
                 cls_rec='reduce_partial', cls_sim='', cls_cert_0='reduce_partial') for i, k in enumerate((50, 150))]
    T2, _ = TL.events(P, TL.prepare_takes(pd.DataFrame(rows, columns=list(V2.TAKE_READ))))
    o4, nav = V2.o4_status(T2)
    RT = V2.raw_takes(P, T2, o4, nav)
    E, D = V2.outcomes(P, RT, {v: TL.variant_codes(P, v) for v in V2.VARIANTS})
    n_ev = {}
    for pn in V2.PANELS:
        code, grp, S, G = V2.panel_def(P['rank'], C0, 'size', pn, P['lg'])
        pt, ob = V2.obs_counts(code, S, G, E[('O1', 'D0')])
        n_ev[pn] = (float(ob.sum()), float(pt.sum()))
    ok_ev = n_ev['full'][0] == 2. and n_ev['clean'][0] == 1. and n_ev['full'][1] - n_ev['clean'][1] == 202.
    ok = ok_r1 and ok_oth and n_rem == 202 and ok_codes and nontriv > 0 and ok_ev
    return ok, dict(rank1_exact=ok_r1, others_untouched=ok_oth, removed=n_rem, codes=ok_codes, nontrivial_codes_removed=nontriv, events_persontime=n_ev)


# ------------------------------------------------------------------ unit tests ------------------------------------------------------------------
def test_o4():
    cases = [('reduce_partial', 'reduce_partial', '', 'reduces_all', 2), ('reduce_partial', 'reversal_reducing', '', 'reduces_all', 2),
             ('full_close', 'reduce_partial', 'reversal_reducing', 'reduces_all', 3), ('reduce_partial', 'add', '', 'reduces_some', 2),
             ('add', 'add', 'reduce_partial', 'reduces_some', 3), ('overshoot', 'reduce_partial', '', 'reduces_some', 2),
             ('add', 'from_flat', 'overshoot', 'reduces_none', 3), ('no_change', 'no_change', '', 'reduces_none', 2), ('reduce_partial', '', '', 'reduces_all', 1),
             ('add', '', '', 'reduces_none', 1), ('unclassified', 'reduce_partial', '', 'unavailable', 2), ('reduce_partial', 'unclassified', '', 'unavailable', 2),
             ('', '', '', 'unavailable', 0)]
    df = pd.DataFrame(dict(cls_rec=[c[0] for c in cases], cls_cert_0=[c[1] for c in cases], cls_sim=[c[2] for c in cases]))
    st, nav = V2.o4_status(df)
    got = [V2.O4_ST[i] for i in st]
    check('O4: fine subtypes grouped BEFORE agreement (reduce_partial / full_close / reversal_reducing reduce; add / from_flat / overshoot / no_change '
          'not); absent variants ignored; an unclassified variant or none -> unavailable (hand table)',
          got == [c[3] for c in cases] and nav.tolist() == [c[4] for c in cases], (got, nav.tolist()))
    check('O4: the grouping is chain_take.REDUCING vs the other inventory classes', set(V2.GROUP_RED) == set(CT.REDUCING) and not set(V2.GROUP_NOT) & set(CT.REDUCING)
          and set(V2.GROUP_RED) | set(V2.GROUP_NOT) == set(CT.CLS) - {'unclassified', 'timing_sensitive'})
    expect('refuse: a variant class outside chain_take.CLS (O4)', lambda: V2.o4_status(pd.DataFrame(dict(cls_rec=['bogus'], cls_cert_0=[''], cls_sim=['']))),
           'outside chain_take.CLS')


def test_decision():
    def rk(vv, sc='pooled'):
        return {(pn, v, sc, 'O1', 'RR_size'): dict(verdict_rule=vv.get((pn, v), 'supported'), RR_MH=2.0, RR_crude=2.1, p_hi=0.01, n_events_informative=40,
                                                  n_markets_supported=6, n_markets_rr_gt1=5) for pn, v in V2.ALL_VIEWS}
    cases = [({}, 'qualified_candidate_association'), ({('full', 'D0'): 'not_supported'}, 'withdrawn'), ({('full', 'D0'): 'insufficient_support'}, 'withdrawn'),
             ({('full', 'D0'): 'inconsistent'}, 'withdrawn'), ({('clean', 'D0'): 'not_supported'}, 'unidentified'),
             ({('full', 'D1000'): 'inconsistent'}, 'unidentified'), ({('full', 'pend'): 'insufficient_support'}, 'unidentified'),
             ({('clean', 'pend'): 'not_supported'}, 'unidentified'), ({('full', 'D0'): 'not_supported', ('clean', 'D0'): 'supported'}, 'withdrawn')]
    got = [V2.decision(rk(vv), 'pooled')['reading'] for vv, _ in cases]
    d = V2.decision(rk({('clean', 'pend'): 'not_supported'}), 'pooled')
    check('decision: primary not supported -> withdrawn; all six views supported -> qualified; otherwise unidentified (clean x pend alone: '
          'the conservative six-view reading, four named views reported True)',
          got == [c[1] for c in cases] and d['predicates']['four_named_views_supported'] and not d['predicates']['all_six_views_supported'], got)
    d1 = V2.decision(rk({}, 'rank1'), 'rank1')
    check('decision: rank 1 reads the same rule, reported as secondary', d1['role'] == 'secondary' and d1['reading'] == 'qualified_candidate_association'
          and V2.decision(rk({}), 'pooled')['role'] == 'primary')


def census_frame(extra=()):
    rows = [('maker_fill', 'pooled', 'all', 10, 'observed_execution'), ('take_add', 'pooled', 'all', 4, 'observed_execution'),
            ('take_reduce_partial', 'pooled', 'all', 3, 'observed_execution'), ('take_timing_sensitive', 'pooled', 'all', 2, 'observed_execution'),
            ('take_kind_mint', 'pooled', 'all', 5, 'observed_execution'), ('take_multi_record_order', 'pooled', 'all', 4, 'observed_execution'),
            ('take_kind_merge', 'pooled', 'all', 1, 'observed_execution'), ('take_self_trade', 'pooled', 'all', 1, 'observed_execution'),
            ('keep_after_fill', 'pooled', 'all', 7, 'continued_execution'), ('maker_fill', 'rank1', '20261024', 6, 'observed_execution'),
            ('take_add', 'rank1', '20261024', 2, 'observed_execution')] + list(extra)
    return pd.DataFrame([dict(source='chains/take/actions.csv', action_type=a, scope=s, day=d, subset='all', n=n, evidence_grade=g) for a, s, d, n, g in rows])


def test_census():
    CF = V2.census_fix(census_frame())
    r = CF[(CF['scope'] == 'pooled') & (CF['day'] == 'all')].iloc[0]
    ok = (int(r['maker_fill_rows']) == 10 and int(r['take_records']) == 9 and int(r['attr_mint']) == 5 and int(r['attr_multi_record']) == 4
          and int(r['attr_merge']) == 1 and int(r['attr_self_trade']) == 1 and int(r['attr_sweep']) == 0 and int(r['attribute_incidences']) == 11
          and int(r['old_observed_execution_sum']) == 30 and bool(r['old_sum_is_base_plus_attributes']) and len(CF) == 2)
    check('census: maker-fill rows and take records counted separately, attributes kept as attributes, old observed sum = base + attributes', ok,
          CF.to_dict('records'))
    cc = V2.census_claims(CF, 9)
    check('census claims: computed predicates (take records = the takes table rows; contract numbers compared, here False)',
          cc['predicates']['take_records_equal_takes_csv_rows'] and not cc['predicates']['maker_fill_rows_is_9026']
          and cc['predicates']['every_row_old_sum_is_base_plus_attributes'], cc['predicates'])
    CFb = V2.census_fix(census_frame([('take_mystery', 'pooled', 'all', 3, 'observed_execution')]))
    rb = CFb[(CFb['scope'] == 'pooled') & (CFb['day'] == 'all')].iloc[0]
    check('census falsifier: an observed_execution row that is neither base nor attribute breaks the decomposition predicate',
          (not bool(rb['old_sum_is_base_plus_attributes'])) and int(rb['old_minus_base_minus_attributes']) == 3
          and not V2.census_claims(CFb, 9)['predicates']['every_row_old_sum_is_base_plus_attributes'])
    expect('refuse: census rows without their n column', lambda: V2.census_fix(census_frame().drop(columns=['n'])), 'required column n')


def test_reproduction(w):
    r = V2.check_variant_reproduction(w.takes)
    check('variants: the fixture classes (chain_take.classify / combine on chain_context positions) reproduce take_class row by row (positive control)',
          r['reproduces'] and r['rows'] == len(w.takes))
    tc, cr, cc = (V2._str_arr(w.takes[c]) for c in ('take_class', 'cls_rec', 'cls_cert_0'))
    i = int(np.flatnonzero((tc == 'add') & (cr == 'add') & (cc == 'add'))[0])
    bad = w.takes.copy()
    bad.loc[i, 'cls_cert_0'] = 'reduce_partial'
    expect('refuse: saved variant classes that do not reproduce take_class through chain_take.combine', lambda: V2.check_variant_reproduction(bad),
           'do not reproduce')
    bad2 = w.takes.copy()
    bad2.loc[0, 'cls_sim'] = 'bogus_class'
    expect('refuse: a variant class outside chain_take.CLS (reproduction)', lambda: V2.check_variant_reproduction(bad2), 'outside chain_take.CLS')


# ------------------------------------------------------------------ fixture 4: mutants ------------------------------------------------------------------
def mutant_suite(B1, B0):
    out = {}
    tests = (('F1', lambda: f1_v2(B1, (('full', 'D0'),), O1_ONLY)[:2]), ('F3_DIR', lambda: f3_dir(B1)), ('COUNT', lambda: count_check(B1)),
             ('DEDUPE', lambda: dedupe_case(B0)))
    for nm, fn in tests:
        try:
            ok, info = fn()
            out[nm] = (bool(ok), False, info)
        except Exception as e:  # noqa: BLE001
            out[nm] = (False, True, f'{type(e).__name__}: {e}')
    return out


def test_mutants(B1, B0):
    """The unmutated baseline of F1 / F3_DIR / COUNT / DEDUPE is the main section's checks (each must pass there)."""
    for m in V2.MUTANTS:
        os.environ['MT_TL2_MUT'] = m
        try:
            res = mutant_suite(B1, B0)
        finally:
            os.environ.pop('MT_TL2_MUT', None)
        failed = [k for k, (ok, _, _) in res.items() if not ok]
        caught = [k for k in EXPECT_CATCH[m] if k in failed and not res[k][1]]
        FIX[f'mutant_{m}_failed_checks'] = failed
        check(f'mutant {m}: caught by {"/".join(EXPECT_CATCH[m])} evaluating False without raising (failed: {failed})', bool(caught),
              {k: res[k][2] for k in failed})
    check('mutants: MT_TL2_MUT unset after the suite', 'MT_TL2_MUT' not in os.environ)


# ------------------------------------------------------------------ refusals, writers ------------------------------------------------------------------
def test_refusals(tmp):
    d = tmp / 'exists'
    d.mkdir()
    expect('refuse: write-once on an existing output directory (write_out)', lambda: V2.write_out(d, {}, {}, {}, lambda o: {}), 'write-once')
    expect('refuse: the production pilot into an existing output directory, before any input is read', lambda: V2.execute('pilot', (V2.PILOT_DAY,), d),
           'write-once')

    def boom(o):
        raise RuntimeError('boom in the receipt writer')
    try:
        V2.write_out(tmp / 'fails', {'a.parquet': pd.DataFrame({'x': [1]})}, {'a.csv': pd.DataFrame({'x': [1]})}, {}, boom)
    except RuntimeError:
        pass
    check('write-once: a failed write leaves no output and no partial directory', not (tmp / 'fails').exists()
          and not any(p.name.startswith('.fails.partial') for p in tmp.iterdir()))
    for var, m, needle in (('MT_TL2_MUT', 'drop_ambiguous', 'MT_TL2_MUT'), ('MT_TL2_MUT', 'bogus', 'not a declared mutant'), ('MT_TL_MUT', 'flip_align', 'MT_TL_MUT')):
        os.environ[var] = m
        try:
            expect(f'refuse: production with {var}={m} before any input is read', lambda: V2.execute('pilot', (V2.PILOT_DAY,), tmp / 'never'), needle)
        finally:
            os.environ.pop(var, None)
    check('refuse: nothing written by a refused production call', not (tmp / 'never').exists())
    expect('refuse: a pilot day outside the measurement days', lambda: V2.execute('pilot', (C.ID_DAY,), tmp / 'never'), 'measurement days')
    g = X._git([f for f in V2.TOOL_FILES if Path(f).exists()])
    needle = 'freeze' if not g.get('unmodified') else 'no recorded pilot projection'
    expect(f'refuse: the full run before any input ({needle})', lambda: V2.execute('run', C.MEASURE_DAYS, tmp / 'run_out', pilot_receipt=tmp / 'absent.json'),
           needle)
    expect('refuse: the full run without a recorded pilot projection', lambda: V2.preflight_projection(tmp / 'absent.json', {}), 'no recorded pilot projection')
    for nm, J, needle in (('p41.json', dict(kind='pilot', version=V2.VERSION, projection=dict(full_run_cpu_s=41.0)), '> 40 s (refused before launch)'),
                          ('pv1.json', dict(kind='pilot', version=TL.VERSION, projection=dict(full_run_cpu_s=10.0)), f'not a {V2.VERSION} pilot receipt')):
        (tmp / nm).write_text(json.dumps(J))
        expect(f'refuse: preflight {nm} ({needle})', lambda: V2.preflight_projection(tmp / nm, {}), needle)
    (tmp / 'p20.json').write_text(json.dumps(dict(kind='pilot', version=V2.VERSION, projection=dict(full_run_cpu_s=20.0))))
    h = {}
    r = V2.preflight_projection(tmp / 'p20.json', h)
    check('preflight: a 20 s v2 projection passes and the pilot receipt is hashed as an input', r['full_run_cpu_s'] == 20.0 and len(h) == 1)


def test_e2e(tmp, Be):
    """run_core on a two-market subset of fixture 1: every writer, hashes, shapes, the written ratios = the in-process path."""
    out = tmp / 'e2e_out'
    rec, claims, R = V2.run_core('selftest', DAYS, Be['parts'], Be['T'], out, {}, Counter(), {}, lambda nm: None, {}, 'asof', time.time(),
                                 3 * len(Be['parts']), census_frame())
    names = sorted(p.name for p in out.iterdir())
    check('e2e: run_core writes exactly the eight contract outputs', names == sorted(V2.OUT_FILES), names)
    Rj = json.loads((out / 'receipt.json').read_text())
    check('e2e: receipt.json hashes every other output (sha256 = the file) and the sources', all(Rj['outputs'][f]['sha256'] == X._sha(out / f)
          for f in V2.OUT_FILES if f != 'receipt.json') and len(Rj['sources']) >= 10 and Rj['mutant'] == '')
    pn, ev = pd.read_parquet(out / 'panel.parquet'), pd.read_parquet(out / 'events.parquet')
    check('e2e: panel.parquet = one row per maker-second (outcomes beside the state); events.parquet = one row per take record',
          len(pn) == Be['P']['N'] and len(ev) == len(Be['T']) and set(ev['o4_status']) <= set(V2.O4_ST) and set(ev['dir_D0']) <= set(V2.DIR_LAB)
          and int(pn['O3'].sum()) == claims['events']['full']['O3']['event_seconds'] and int(ev['in_raw_schedule'].sum()) == claims['takes']['raw_schedule']['n_raw'],
          (len(pn), len(ev)))
    rt = pd.read_csv(out / 'ratios.csv')
    n_exp = len(V2.PANELS) * len(V2.VARIANTS) * len(V2.SCOPES2) * len(V2.OUTCOMES) * len(V2.RATIOS)
    check(f'e2e: ratios.csv = 2 panels x 3 clocks x 2 scopes x 4 outcomes x 6 ratios = {n_exp} rows, each with a v1 verdict',
          len(rt) == n_exp and bool(rt['verdict_rule'].isin(VERDICTS).all()), len(rt))
    RKd = ratios(Be, V2.ALL_VIEWS, O1_ONLY)
    check('e2e: the run_core ratios equal the direct ratio path (O1, six views, both scopes)', all(_same(R['RK'][k]['RR_MH'], RKd[k]['RR_MH'])
          and R['RK'][k]['verdict_rule'] == RKd[k]['verdict_rule'] and R['RK'][k]['c_hi'] == RKd[k]['c_hi'] for k in RKd))
    J = json.loads((out / 'claims.json').read_text())
    check('e2e: claims.json carries the decision for pooled (primary) and rank 1 (secondary), each a declared reading; multiplicity 96',
          set(J['decision']) == {'pooled', 'rank1'} and all(J['decision'][s]['reading'] in V2.READINGS for s in J['decision'])
          and J['decision']['pooled']['role'] == 'primary' and J['multiplicity']['tests'] == 96
          and J['decision']['pooled']['reading'] == V2.decision(R['RK'], 'pooled')['reading'])
    cv = pd.read_csv(out / 'coverage.csv')
    c = cv[(cv['panel'] == 'full') & (cv['variant'] == 'D0') & (cv['scope'] == 'pooled')]
    check('e2e: coverage by |q| bin adds up to the maker-seconds and the raw takes', int(c['maker_seconds'].sum()) == Be['P']['N']
          and int(c['raw_takes'].sum()) == claims['takes']['raw_schedule']['n_raw'])
    cf, cs = pd.read_csv(out / 'census_fix.csv'), pd.read_csv(out / 'consistency.csv')
    check('e2e: census_fix.csv rows and its claims predicates; consistency.csv = a market row per market and an all-days summary per ratio row',
          len(cf) == 2 and 'predicates' in J['census_fix'] and int((cs['row'] == 'market').sum()) == n_exp * len(Be['P']['markets'])
          and int(((cs['row'] == 'summary') & (cs['day'].astype(str) == 'all')).sum()) == n_exp)
    dg = V2.digest(out, claims, R, rec)
    s_ = json.dumps(X._clean(dg), indent=1, sort_keys=True, default=str)
    check('e2e: the pilot digest (the printed key numbers) builds and serialises', len(s_) > 100 and set(dg['RR_size_full_D0']) ==
          {f'{sc}:{o}' for sc in V2.SCOPES2 for o in V2.OUTCOMES} and len(dg['coverage_full_D0']) == 2 * len(TL.Q_LAB), len(dg['coverage_full_D0']))
    expect('e2e: a second write into the written directory is refused', lambda: V2.write_out(out, {}, {}, {}, lambda o: {}), 'write-once')


def main():
    t0 = time.process_time()
    check('selftest environment: MT_RUN=v2, no MT_TL2_MUT / MT_TL_MUT', C.RUN == 'v2' and 'MT_TL2_MUT' not in os.environ and 'MT_TL_MUT' not in os.environ, C.RUN)
    tmp = Path(tempfile.mkdtemp(prefix='take_logic_v2_selftest_'))
    try:
        section('O4', test_o4)
        section('decision', test_decision)
        section('census', test_census)
        w1 = section('world random', World, 'random', 2031)
        w2 = section('world size', World, 'size', 2032)
        B1 = section('base 1', base, w1) if w1 else None
        B0 = section('base 0', base, w1, 1) if w1 else None
        Be = section('base e2e', base, w1, 2) if w1 else None
        B2 = section('base 2', base, w2) if w2 else None
        if w1 and w2:
            check('fixtures: planted takes (random, size)', min(len(w1.takes), len(w2.takes)) >= 150, (len(w1.takes), len(w2.takes)))
        if B1:
            r = section('F1 precondition', f1_pre, B1)
            if r:
                check('F1 precondition: fine-class (v1 timing_sensitive) and grouped (O4 reduces_some) disagreement concentrate at |q| (0,50] '
                      '(coverage table = direct count)', r[0], r[1])
            r = section('F1 v1', f1_v1, B1)
            if r:
                check('F1 bias control: the v1 outcome (classified reduce, ambiguous dropped) shows the spurious RR_size > 1 (rank 1 and pooled)', r[0], r[1])
            r = section('F1 v2', f1_v2, B1)
            if r:
                FIX['F1_O1_RR_size_full_D0'] = r[1]
                check('F1 bias control: O1 RR_size is NOT supported (rank 1 and pooled) and is testable (>= 5 supported markets, >= 30 events)', r[0], r[1])
                dec = V2.decision(r[2], 'pooled')
                FIX['F1_reading'] = dec['reading']
                check('F1: the declared decision reads withdrawn for the random-side maker (pooled)', dec['reading'] == 'withdrawn', dec['views'])
        if B2:
            r = section('F2', f2, B2)
            if r:
                check('F2: a planted size policy (opposing-take hazard rising with |q|) reads O1 RR_size supported (rank 1 and pooled)', r[0], r[1])
                check('F2: the decision is not withdrawn (pooled)', r[2]['pooled']['reading'] != 'withdrawn', {sc: r[2][sc]['views'] for sc in r[2]})
        if B1:
            r = section('F3 direction', f3_dir, B1)
            if r:
                check('F3: the raw-schedule null re-derives the direction at the destination second (tool = brute force, 3 clocks x 5 replicates, '
                      'strata counts equal); source / destination sign changes exist', r[0], r[1])
            r = section('F3 counts', f3_counts, B1)
            if r:
                check('F3: raw counts per pair preserved in every replicate, shifted takes stay in their pair, collisions invariant, offsets '
                      'deterministic; the O3 / O4 benchmark shift keeps its count', r[0], r[1])
            r = section('COUNT', count_check, B1)
            if r:
                check('raw schedule: every assigned take record kept (ambiguous included, per pair); O3 = the distinct take seconds', r[0], r[1])
        if B0:
            r = section('DEDUPE', dedupe_case, B0)
            if r:
                check('dedupe: one indicator per second and outcome (3 takes / mixed sides -> O1 and O2 once each; q = 0 and side_mixed are '
                      'statuses); collisions 3, mixed seconds 1', r[0], r[1])
        r = section('F5', f5)
        if r:
            check('F5 clean ledger: a planted excluded leg removes exactly its pair\'s seconds from its window start (k >= 98, 202 maker-seconds); '
                  'codes -1 there, unchanged elsewhere; a take there leaves the clean panel only', r[0], r[1])
        if w1:
            section('reproduction', test_reproduction, w1)
        if B1 and B0:
            section('mutants', test_mutants, B1, B0)
        section('refusals', test_refusals, tmp)
        if Be:
            section('e2e', test_e2e, tmp, Be)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop('MT_TL2_MUT', None)
    fails = [r for r in RES if not r[1]]
    for name, ok, info in RES:
        print(('PASS ' if ok else 'FAIL ') + name + ('' if ok else f'  | {info}'))
    print('FIXTURES ' + json.dumps(X._clean(FIX), sort_keys=True, default=str))
    print(json.dumps(dict(checks=len(RES), failed=len(fails), cpu_s=round(time.process_time() - t0, 3), process_cpu_s=round(time.process_time(), 3),
                          vmhwm_mib=round(CT.hwm(), 1))))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
