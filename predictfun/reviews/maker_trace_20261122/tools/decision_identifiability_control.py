"""IDENTIFIABILITY CONTROL of the frozen top-3 decision module (decision_contract.md Amendment 2 item 2; review 567f741d9; P-2026-005,
session aea5bfb9, 2026-11-23). SYNTHETIC only. pf_mt_decision.py (frozen afb589d46), pf_mt_lifecycle.py and both selftests are imported
unchanged. Two hidden command histories A / B on one BOOK market produce the SAME public tape (match records / legs, every book frame and
level). Each world then goes SEPARATELY through the same public pipeline - the frozen pf_mt_lifecycle.run_market (life orders / links),
legs -> fills_valued rows, frames + levels -> tob, one shared BTC grid - and the frozen pf_mt_decision.run. Asserted: (1) identical public
tapes, (2) identical derived inputs, (3) identical decision outputs (every table, the timelines, claims_decision.json, and the receipt
without its clock / path / input-hash fields). Every hidden posting-decision quantity that differs between A and B is therefore
UNIDENTIFIED by the instrument; each is printed with the value the module reports in both worlds. FALSIFIER: world C = A plus one order shown
in three frames must change the public tape AND at least one decision output (else the comparison could not fire). The frozen positive
controls run once (pf_mt_decision_selftest.main: every check must pass).
Hidden histories (ms after T0 = ws = 2026-10-24 01:00Z; frames every 100 ms; rank2 = the maker under study):
 bid 50, the plan's two-history case - A: hM (rank2) 5 sh posted at 20 050, exhausted by R1 at 60 060; hN (another maker) 45 sh queued
   behind it from 20 070. B: hM 50 sh posted at 20 050, R1 fills 5, the remaining 45 cancelled at 60 070; hN posts its 45 sh only at 60 080.
   Every frame shows 50 before R1 and 45 after it in both.
 ask 55, posting time and an unseen order - A: hP (rank2) 8 sh posted at 150 030. B: hQ (rank2) 8 sh at ask 57 posted at 150 010 and cancelled
   at 150 040 (no frame in between), then hP posted at 150 070. R3 at 200 060 exhausts hP in both.
 R2 at 120 060: rank1 takes 10 sh of hN (sells YES); R3's taker is rank3 (buys YES): the take paths exist in both worlds.
Usage (run_light, PF_SLOT=claude2): decision_identifiability_control.py"""
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RS = Path('/home/yuqing/ctaNew/predictfun/research')
sys.path.insert(0, str(RS))
import pf_mt_common as C  # noqa: E402
import pf_mt_decision as M  # noqa: E402
import pf_mt_lifecycle as LC  # noqa: E402
import pf_mt_lifecycle_selftest as LS  # noqa: E402
import pf_mt_decision_selftest as ST  # noqa: E402

A1, A2, A3 = M.TOP3
NN = '0x00000000000000000000000000000000000000Bb'      # another maker (not a top-3 address)
TA = '0x00000000000000000000000000000000000000Cc'      # another taker
T0, D, MID = LS.T0, '20261024', 101
WS_S = T0 // 1000
FR_T = list(range(-299_600, 300_900, 100))            # frame times (ms after ws): ws - 299.6 s .. we + 0.8 s
RES = []

HIST = dict(
    A={('bid', 50): [[(20_050, 5.), (60_060, 0.)], [(20_070, 45.), (120_060, 35.)]],
       ('ask', 55): [[(150_030, 8.), (200_060, 0.)]]},
    B={('bid', 50): [[(20_050, 50.), (60_060, 45.), (60_070, 0.)], [(60_080, 45.), (120_060, 35.)]],
       ('ask', 57): [[(150_010, 8.), (150_040, 0.)]],
       ('ask', 55): [[(150_070, 8.), (200_060, 0.)]]})
HIST['C'] = dict(HIST['A'])
HIST['C'][('ask', 54)] = [[(150_010, 8.), (150_350, 0.)]]   # the falsifier: an order the frames 150 100 .. 150 300 show
# the hidden posting-decision quantities (truth per history) and the module field that would have to report them
TRUTH = (('hM original size', '5 sh', '50 sh', 'life S_FIFO size_lb / size_exact / size_class'),
         ('hM end of life', 'exhausted by R1 at 60.060 s', '5 sh filled by R1, 45 sh cancelled at 60.070 s',
          'life removal_set / removal window / removal_class; removal.csv'),
         ('hN posting time', '20.070 s (queued behind hM)', '60.080 s (after the cancel)', 'life hN post_lo / post_hi / post_class'),
         ('hN queued behind hM before R1', 'yes (45 sh behind 5 sh)', 'no (absent until 60.080 s)', 'life hN ahead_lo / ahead_hi / queue_class'),
         ('hP posting time', '150.030 s', '150.070 s', 'life hP post_lo / post_hi; post_posting.csv rank2 ask'),
         ('hQ (rank2, ask 57, 8 sh, 150.010 -> 150.040 s)', 'never existed', 'posted and cancelled between frames',
          'no field (no public trace)'))


def check(name, cond, detail=''):
    RES.append((name, bool(cond)))
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  [{str(detail)[:700]}]'), flush=True)


def btc_table():
    rng = np.random.default_rng(424242)                    # one BTC grid for every world (public, identical)
    g = np.arange(T0 - 420_000, T0 + 360_000, 100, dtype=np.int64)
    mid = np.exp(np.log(110_000.) + np.cumsum(rng.normal(0., np.sqrt(.1), len(g))) / 1e4)
    return pd.DataFrame(dict(g_ms=g, bid=mid - .05, ask=mid + .05, mid=mid, n_upd=np.full(len(g), 3, np.int64),
                             age_ms=rng.integers(0, 100, len(g)).astype(np.int64), status=['ok'] * len(g)))


def tob_table(frames, levels):
    """Top of book per frame from the full-depth frames / levels (the extractor's tob fields)."""
    fid = frames['fid'].to_numpy(np.int64)
    lv = levels[levels['status'] == 'ok']
    out = {}
    for side, pick in (('bid', 'last'), ('ask', 'first')):
        s = lv[lv['side'] == side].sort_values(['fid', 'ptick'], kind='mergesort')
        best = getattr(s.groupby('fid'), pick)()
        out[side] = (best['ptick'].reindex(fid).to_numpy(float) / 100., best['shares'].reindex(fid).to_numpy(float),
                     s.groupby('fid').size().reindex(fid).fillna(0).to_numpy(np.int64))
    usd = lambda col: np.array([sum(u for _, u in json.loads(x)) for x in frames[col]], float)  # noqa: E731
    return pd.DataFrame(dict(day=[D] * len(fid), mid=frames['mid'].to_numpy(np.int64), upd_ms=frames['upd_ms'].to_numpy(np.int64),
                             recv_ns=frames['recv_ns'].to_numpy(np.int64), conn=frames['conn'].astype(str).tolist(),
                             order_count=np.zeros(len(fid), np.int64), n_bid=out['bid'][2], n_ask=out['ask'][2], best_bid=out['bid'][0],
                             best_bid_sh=np.nan_to_num(out['bid'][1]), best_ask=out['ask'][0], best_ask_sh=np.nan_to_num(out['ask'][1]),
                             pend_bid_usd=usd('pend_bids'), pend_ask_usd=usd('pend_asks'), status=frames['status'].astype(str).tolist()))


def fv_table(legs):
    """fills_valued rows (the decision module's columns): one maker and one taker row per leg, YES space."""
    rows = []
    for r in legs.itertuples(index=False):
        s_mk = 1 if r.y_side == 'bid' else -1
        for role, addr, s in (('maker', r.mk_signer, s_mk), ('taker', r.tk_signer, -s_mk)):
            rows.append(dict(day=r.day, mid=int(r.mid), sid=r.sid, role=role, addr=addr, s=int(s), q=float(r.shares), p=float(r.y_price),
                             settle_ms=int(r.settle_ms), self_trade=bool(r.mk_signer == r.tk_signer), status='ok'))
    return pd.DataFrame(rows).astype(dict(mid='int64', s='int64', q='float64', p='float64', settle_ms='int64', self_trade='bool'))


def world(root, orders):
    """Hidden commands -> public tape (the venue aggregate per frame, the match records) -> the public pipeline -> the frozen decision run."""
    m = LS.Mk(MID)
    m.ws, m.we = WS_S, WS_S + 300
    wb = LS.win((60_100, 60_500, .50, 2.5), (120_100, 120_500, .50, 5.))
    wa = LS.win((200_100, 200_500, .55, 4.4))
    m.frames(FR_T, LS.sim(orders), pend=lambda tr: (wb(tr), wa(tr)))
    m.rec(60_060, [('hM', A2, 'bid', 50, 5.)], tk=TA)
    m.rec(120_060, [('hN', NN, 'bid', 50, 10.)], tk=A1)
    m.rec(200_060, [('hP', A2, 'ask', 55, 8.)], tk=A3)
    pub = m.dfs()
    fills, ords, links, summ = LC.run_market(*pub, m.market())
    (root / 'econ').mkdir(parents=True)
    (root / 'btc').mkdir()
    (root / 'life').mkdir()
    inputs = dict(fills_valued=fv_table(pub[0]), tob=tob_table(pub[2], pub[3]), btc=btc_table(),
                  markets=pd.DataFrame(dict(day=[D], mid=[MID], slug=['btc-updown-5m'], ws=[WS_S], we=[WS_S + 300], outcome=['Up'], status=['ok'])),
                  life_orders=ords, life_links=links)
    inputs['fills_valued'].to_parquet(root / 'econ' / 'fills_valued.parquet', index=False)
    (root / 'econ' / 'identities.json').write_text(json.dumps(dict(ex_ante=dict(top10_shares=[dict(rank=i + 1, addr=a) for i, a in enumerate(M.TOP3)]))))
    inputs['markets'].to_parquet(root / f'markets_{D}.parquet', index=False)
    inputs['tob'].to_parquet(root / f'tob_{D}.parquet', index=False)
    inputs['btc'].to_parquet(root / 'btc' / f'btc_{D}.parquet', index=False)
    ords.to_parquet(root / 'life' / f'{D}_{MID}_orders.parquet', index=False)
    lt = dict(mid='int64', mk_signer='str', y_side='str', kind='str', prev_hash='str', next_hash='str', prev_ptick='int64', next_ptick='int64',
              prev_last_hi_ms='int64', next_first_hi_ms='int64', gap_ms='int64', x_latest='bool', label='str', status='str')   # pf_mt_lifecycle._links
    arrow = {'str': pa.string(), 'int64': pa.int64(), 'bool': pa.bool_()}
    pq.write_table(pa.Table.from_pandas(links[list(lt)], schema=pa.schema([(c, arrow[t]) for c, t in lt.items()]), preserve_index=False),
                   root / 'life' / f'{D}_{MID}_x_links.parquet')       # explicit types: an empty table must not be written as null columns
    rec = M.run(inp=root, out=root / 'decision', days=(D,), book={D: [MID]}, btc_dir=root / 'btc', log=lambda *a, **k: None)
    return dict(pub=pub, life=(fills, ords, links), summ=summ, inputs=inputs, out=root / 'decision', rec=rec)


VOLATILE = ('asof_start', 'asof_end', 'wall_s', 'cpu_s', 'memory', 'argv', 'slot', 'inputs', 'btc_files')


def receipt(p):
    j = json.loads(p.read_text())
    return {k: v for k, v in j.items() if k not in VOLATILE}


def compare_outputs(oA, oB):
    names = sorted({p.name for p in oA.iterdir()} | {p.name for p in oB.iterdir()})
    same = {}
    for n in names:
        a, b = oA / n, oB / n
        if not (a.is_file() and b.is_file()):
            same[n] = False
        elif n == 'decision.json':
            same[n] = receipt(a) == receipt(b)
        else:
            same[n] = a.read_bytes() == b.read_bytes()
    return same


def read_csv(p):
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:                  # a table without rows or columns (e.g. no links in this world)
        return pd.DataFrame()


def life_row(ords, h):
    r = ords[(ords['scenario'] == 'S_FIFO') & (ords['mk_hash'] == h)]
    if len(r) != 1:
        return None
    r = r.iloc[0]
    rel = lambda v: None if v < 0 else round((v - T0) / 1000., 3)  # noqa: E731
    return dict(size_lb=r['size_lb'], size_exact=r['size_exact'], size_class=r['size_class'], post_lo_s=rel(r['post_lo_ms']), post_hi_s=rel(r['post_hi_ms']),
                post_class=r['post_class'], ahead_lo=r['ahead_lo'], ahead_hi=r['ahead_hi'], queue_class=r['queue_class'], removal_set=r['removal_set'],
                removal_lo_s=rel(r['removal_lo_ms']), removal_hi_s=rel(r['removal_hi_ms']), removal_class=r['removal_class'])


def main():
    c0 = time.process_time()
    with tempfile.TemporaryDirectory(prefix='mt_dec_identifiability_') as tmp:
        W = {k: world(Path(tmp) / k, HIST[k]) for k in ('A', 'B', 'C')}
        A, B, Cw = W['A'], W['B'], W['C']
        # ---- (1) the public tapes ----
        names = ('legs', 'records', 'frames', 'levels')
        eq_pub = {n: A['pub'][i].equals(B['pub'][i]) for i, n in enumerate(names)}
        sizes = ', '.join(f'{n} {len(A["pub"][i])} rows' for i, n in enumerate(names))
        check(f'(1) the two hidden histories produce the SAME public tape ({sizes})',
              all(eq_pub.values()), eq_pub)
        # ---- (2) the derived inputs (each world's own pipeline run) ----
        eq_in = {k: A['inputs'][k].equals(B['inputs'][k]) for k in A['inputs']}
        eq_in['life_fills'] = A['life'][0].equals(B['life'][0])
        eq_in['life_summary'] = json.dumps(A['summ'], sort_keys=True, default=str) == json.dumps(B['summ'], sort_keys=True, default=str)
        check('(2) identical derived inputs: fills_valued, tob, BTC, markets, life orders / links / fills / summary (frozen pf_mt_lifecycle)',
              all(eq_in.values()), eq_in)
        sha = {k: A['rec']['inputs'][k]['sha256'] == B['rec']['inputs'][k]['sha256'] for k in A['rec']['inputs']}
        print(f'INFO input file sha256 equal across A / B (byte level, not asserted): {sha}', flush=True)
        # ---- (3) the decision outputs ----
        same = compare_outputs(A['out'], B['out'])
        check(f'(3) identical decision outputs: {len(same)} files (every table, timeline, claims_decision.json; the receipt without '
              'clock / path / input-hash fields)', all(same.values()) and len(same) >= 15, {k: v for k, v in same.items() if not v})
        print('IDENTICAL OUTPUT FIELDS (A == B):', flush=True)
        for n in sorted(same):
            p = A['out'] / n
            if n.endswith('.csv'):
                df = read_csv(p)
                print(f'  {n}: {len(df)} rows; fields: {", ".join(df.columns)}', flush=True)
            elif n == 'claims_decision.json':
                cl = json.loads(p.read_text())['claims']
                print(f'  {n}: {len(cl)} claims; ids: {", ".join(sorted({c["claim"] for c in cl}))}', flush=True)
            else:
                print(f'  {n}: fields {", ".join(sorted(receipt(p)))}', flush=True)
        # ---- the hidden differences the instrument cannot see ----
        oA = A['life'][1]
        rep = {h: life_row(oA, h) for h in ('hM', 'hN', 'hP')}
        PP = read_csv(A['out'] / 'post_posting.csv')
        pp = PP[(PP['scope'] == 'rank2') & (PP['day'].astype(str) == 'all') & (PP['L'] == 0)][['clock', 'scenario', 'side', 'n', 'offset_mean', 'tau_med']] \
            if len(PP.columns) else PP
        RM = read_csv(A['out'] / 'removal.csv')
        rm = RM[(RM['scope'] == 'rank2') & (RM['L'] == 0) & (RM['metric'] == 'adverse') & (RM['group'] == 'cancel_possible')][
            ['scenario', 'phase', 'n_orders', 'obs_mean', 'null_mean', 'p_one_sided']] if len(RM.columns) else RM
        print('HIDDEN-COMMAND DIFFERENCES THAT ARE INVISIBLE (truth differs, the module reports the same in A and B -> UNIDENTIFIED):', flush=True)
        for q, ta, tb, field in TRUTH:
            print(f'  UNIDENTIFIED {q}: A = {ta}; B = {tb}; reported field: {field}', flush=True)
        for h, r in rep.items():
            print(f'    module (A == B) {h}: {json.dumps(r, default=str)}', flush=True)
        print(f'    module (A == B) post_posting rank2 L0: {json.dumps(pp.round(4).to_dict("records"), default=str)}', flush=True)
        print(f'    module (A == B) removal rank2 L0 adverse cancel_possible: {json.dumps(rm.round(4).to_dict("records"), default=str)}', flush=True)
        check('every listed hidden quantity differs between the histories (truth A != B) while every output is identical',
              all(ta != tb for _, ta, tb, _ in TRUTH) and all(same.values()))
        check('the module states the ambiguity instead of resolving it: hM keeps exhausted AND cancelled_or_expired (S_FIFO), size_lb 5 = the fill',
              rep['hM'] is not None and {'exhausted', 'cancelled_or_expired'} <= set(str(rep['hM']['removal_set']).split('|')) and
              rep['hM']['size_lb'] == 5., rep['hM'])
        blob = ''.join(p.read_text() for p in sorted(B['out'].iterdir()))
        check('hQ / ask 57 leaves no trace: absent from the B tape, the life tables and every output', (B['pub'][3]['ptick'] != 57).all() and
              not (B['life'][1]['y_ptick'] == 57).any() and 'hQ' not in blob, '')
        # ---- falsifier: a VISIBLE difference must change the tape and a decision output ----
        eq_c = {n: A['pub'][i].equals(Cw['pub'][i]) for i, n in enumerate(names)}
        same_c = compare_outputs(A['out'], Cw['out'])
        diff_c = sorted(k for k, v in same_c.items() if not v)
        print(f'INFO falsifier world C (one order shown in 3 frames): tape equal {eq_c}; decision files that differ: {diff_c}', flush=True)
        check('FALSIFIER: a visible order changes the public tape and at least one decision output (the comparison can fire)',
              not all(eq_c.values()) and len(diff_c) > 0, (eq_c, diff_c))
    cpu_ctl = time.process_time() - c0
    # ---- the frozen positive controls, once ----
    rc = 0
    try:
        ST.main()
    except SystemExit as e:
        rc = int(e.code or 0)
    n_ok = sum(ok for _, ok in ST.RES)
    ctl = [n for n, ok in ST.RES if n.startswith('CONTROL') and ok]
    check(f'frozen positive controls (pf_mt_decision_selftest.main): {n_ok}/{len(ST.RES)} checks, {len(ctl)} CONTROL checks pass', rc == 0 and
          n_ok == len(ST.RES) and len(ctl) >= 12, [n for n, ok in ST.RES if not ok])
    n_ok = sum(ok for _, ok in RES)
    print(f'IDENTIFIABILITY CONTROL {n_ok}/{len(RES)} checks passed; control cpu {cpu_ctl:.1f} s, total cpu {time.process_time() - c0:.1f} s', flush=True)
    sys.exit(0 if n_ok == len(RES) else 1)


if __name__ == '__main__':
    main()
