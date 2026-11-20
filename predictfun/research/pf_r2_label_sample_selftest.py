"""SELFTEST of pf_r2_label_sample (synthetic only; no market data). run(S) checks the module object S. The FIRST failed check exits with
'R2 LABEL SAMPLE SELFTEST FAILED: <check>' (fail fast; the mutation check needs the first catch only). Every block is a computed predicate:
 (1) the UTC-hour rule; the world sample on enumerated tiny populations (a sparse and a dense stratum, the pilot = each stratum's first-ranked
     world, canonical order, determinism, ties broken by the canonical key);
 (2) the picks and weights on enumerated pools:
     - shapes: mandatory + extra; no mandatory; singleton; same-time rows of two sources; empty and absent heads;
     - the extra is the lowest-ranked other row; duplicate keys and two mandatory rows are refused;
     - the exact inverse-probability identity, enumerating every equally likely world subset and extra draw of a two-stratum population:
       E[sum w x] = the sum over originals and heads of the eligible pool's mean;
 (3) end to end on the synthetic world (stub forecasts, injected inputs / files / hours; one UTC hour, so the full-sample path with 4 worlds
     in partitions of 2; its dates are not its day name, so STRICT_DAY is off here and block 1 tests it). Covered:
     - plans: two equal plans; an existing plan is never rewritten; the plan checkpoint refuses and leaves a stop that refuses the directory;
     - runs: every part runs; a planted interruption promotes nothing; the rerun's part and pool bytes equal the uninterrupted ones; a done
       part is skipped;
 (4) the validator passes the intact part and refuses each corruption. Picks / pairs:
     Y + 1, another fork, a false reconciliation, an arm binding failure, a causal mismatch, a fork-input hash, hidden-copy or rebuild
     inequality, an extra row (a world of another partition), a duplicate row, a missing pick, a wrong weight, a self-consistent N, a changed
     row key. Ledgers: R != cash + position x mark, a self-consistent ledger change against the fills, an added fill, a consistent
     pre-fork fill in one arm, an overfilled generation. Context: another side, original or decision. Pools: an added key, a dropped line,
     inconsistent counts. Records: a part byte, a pool byte, other sources / inputs / plan, a breach, wrong counts;
 (5) identity, each refused by its own check:
     - before a done part is skipped: a stale source, changed loader files (no day input is read), changed tracked hours, a changed teacher,
       a changed feature schema, a changed tape offset; a changed plan (its records bind the old plan);
     - after loading: real_inputs' own tracked hashes; a changed keep / pull output;
 (6) stops:
     - a hard-RSS status holds every part until a disposition of THIS plan names it;
     - two successive stops at one status file both survive, and the first one's disposition leaves the second holding (review B3);
     - a partition process reads its one day's inputs once, and every earlier world is released before the next is labelled (a lifetime
       sentinel; its positive control, worlds deliberately retained, must report them; review B4);
     - the stop rule latches STOP_RESOURCE with nothing promoted: for the high-water (at the loading), a pair over its limit (at the
       pair) and the process bound (at the loading);
 (7) promotion refuses an invalid part; a resume refuses a corrupted promoted part; every completion record carries its timing (the loading,
     each world's census and pairs);
 (8) finalize refuses an unpromoted part, an undisposed stop, a changed input, a changed teacher and a high-water over the checkpoint; its
     accepted manifest equals the parts (rows, statuses, weight totals).
PF_TEACHER_SYN: the cached synthetic world (pf_r2_teacher_selftest.build_world)."""
import contextlib
import gc
import hashlib
import io
import itertools
import json
import os
import shutil
import sys
import tempfile
import types
import weakref
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_r2_labels_selftest as LS                        # noqa: E402
import pf_r2_shadow as R2                                 # noqa: E402
import pf_r2_teacher_selftest as TS                       # noqa: E402

C, pd, I, KP, X = R2.C, R2.pd, R2.I, R2.KP, R2.X
FAIL = 'R2 LABEL SAMPLE SELFTEST FAILED'


def must(ok, what):
    if not ok:
        raise SystemExit(f'{FAIL}: {what}')


def refuses(fn, *a, msg=None, **k):
    """True iff fn raises ValueError (whose text contains msg, if given)."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn(*a, **k)
    except ValueError as e:
        return msg is None or msg in str(e)
    except Exception:                                                                       # a crash is not the declared refusal
        return False
    return False


def vbad(S, *a):
    """The validator's failures, or None if it crashed (a crash is not a refusal)."""
    try:
        return S.validate(*a)
    except Exception:
        return None


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def fake_worlds(spec):
    """spec: {hour: count} -> fake worlds of day d (two originals each)."""
    out, oid = [], 0
    for h, n in spec.items():
        for i in range(n):
            out.append(dict(day='d', m=100 + i % 3, t_ms=h * 3600000 + 5000 * (i + 1), hour=h, o={0: types.SimpleNamespace(oid=oid), 1: types.SimpleNamespace(oid=oid + 1)}))
            oid += 2
    return out


def cand(source, k, kind='post', side=0, offset=None, status='ok', shadow=None, oid=7):
    return dict(day='d', instrument_id=1, original_episode_id=oid, side=side, source=source, kind=kind, offset_ms=offset if offset is not None else float('nan'),
                generation=1., decision_ms=50. * k, status=status, shadow_source=shadow or ('entry' if source in ('neutral', 'delayed') else source))


def run(S):  # noqa: C901
    K = S.row_key
    # ---- (1) the UTC-hour rule and the world sample
    d0 = int(pd.Timestamp('2026-10-24', tz='UTC').value // 10 ** 6)
    must(S.utc_hour('20261024', d0) == 0 and S.utc_hour('20261024', d0 + 5 * S.HOUR_MS + 1) == 5 and S.utc_hour('20261024', d0 + S.DAY_MS - 1) == 23, 'the UTC hour of t0')
    must(refuses(S.utc_hour, '20261024', d0 - 1) and refuses(S.utc_hour, '20261024', d0 + S.DAY_MS), 'a t0 outside its UTC day was accepted')
    W = fake_worlds({3: 1, 7: 5}); st, sel = S.sample_worlds(W, per_hour=2); by = {(x['day'], x['hour']): x for x in st}
    must(by[('d', 3)]['n'] == 1 and by[('d', 3)]['pi'] == 1. and by[('d', 7)]['N'] == 5 and by[('d', 7)]['n'] == 2 and abs(by[('d', 7)]['pi'] - .4) < 1e-12,
         f'sample strata {st}')
    must(len(sel) == 3 and sum(x['pilot'] for x in sel) == 2 and sel == sorted(sel, key=lambda x: (x['day'], x['m'], x['t_ms'])) and S.sample_worlds(W, per_hour=2) == (st, sel),
         'sample selection / canonical order / determinism')
    h7 = sorted([w for w in W if w['hour'] == 7], key=lambda w: (S.rank('world', S.wkey(w)), S.ckey(S.wkey(w))))
    must({(x['m'], x['t_ms']) for x in sel if x['hour'] == 7} == {(w['m'], w['t_ms']) for w in h7[:2]}, 'sample: not the lowest-ranked worlds')
    must([x for x in sel if x['hour'] == 7 and x['pilot']][0]['t_ms'] == h7[0]['t_ms'], 'sample: the pilot is not the first-ranked world')
    r0 = S.rank
    try:
        S.rank = lambda kind, key: '0' * 64
        sel_t = S.sample_worlds(W, per_hour=2)[1]
    finally:
        S.rank = r0
    must(sorted((x['m'], x['t_ms']) for x in sel_t if x['hour'] == 7) == sorted((w['m'], w['t_ms']) for w in sorted(h7, key=lambda w: S.ckey(S.wkey(w)))[:2]),
         'sample: ties are not broken by the canonical key')
    must(S.ckey(S.wkey(W[0])) == '{"day":"d","m":%d,"t_ms":%d}' % (W[0]['m'], W[0]['t_ms']), 'the canonical world key')
    must(S.ckey(dict(t_ms=1, day='d', m=2, x=None)) == '{"day":"d","m":2,"t_ms":1,"x":null}', 'the canonical key: sorted keys, explicit nulls')
    cl = S.closure(); lazy_only = {'pf_gap_top_makers.py', 'pf_move_info.py', 'pf_qr2_v3_early.py'}                  # reached only by function-level imports (2026-11-20)
    must(set(S.CORE) <= set(cl) and not any(n.endswith(S.CHECKS) for n in cl) and lazy_only <= set(cl), 'the source closure (core modules, function-level imports, no checks)')
    # ---- (2) the picks and weights
    pi = .4
    pools = dict(full=[cand('neutral', 0, offset=0.)] + [cand(s_, k) for s_, k in (('delayed', 5), ('pi0', 10), ('joint_chase', 15))]
                 + [cand('pi0', 20, 'keep'), cand('neutral', 25, 'keep', offset=0., shadow='retain')],
                 nomand=[cand('delayed', 5), cand('pi0', 10), cand('joint_post_keep', 15)], single=[cand('neutral', 0, offset=0.)],
                 dup=[cand('neutral', 0, offset=0.), cand('pi0', 0), cand('pi0', 30, 'keep')],
                 empty=[cand('delayed', 5, status='unsampled'), cand('pi0', 10, 'keep', status='cap_blocked')])
    P = lambda n, h: S.picks_of([K(c) for c in pools[n] if c['kind'] == h and c['status'] == 'ok'], h, pi)
    get = lambda n, h, p: next((x for x in P(n, h) if x[0] == p), (p, None, None, None, None))   # an absent pick fails its check, never crashes
    tot = lambda n, h: sum(x[4] for x in P(n, h) if x[4] is not None)
    must(get('full', 'post', 'mandatory')[3] == 1. and get('full', 'post', 'extra')[3] == 1 / 3 and get('full', 'post', 'extra')[2] == 4 and abs(tot('full', 'post') - 1 / pi) < 1e-12,
         'picks: mandatory + extra (q 1, 1 / (N - 1); total 1 / pi)')
    must(get('full', 'keep', 'keep')[3] == .5 and abs(tot('full', 'keep') - 1 / pi) < 1e-12, 'picks: keep (q 1 / N)')
    must(get('nomand', 'post', 'only')[3] == 1 / 3 and abs(tot('nomand', 'post') - 1 / pi) < 1e-12 and get('nomand', 'keep', 'keep')[1] is None,
         'picks: no mandatory row (q 1 / N) / an absent head')
    try:
        sp = P('single', 'post')
    except Exception as e:                                                                  # a failure to pick is this check's failure
        sp = None; must(False, f'picks: a singleton pool raised {e!r}')
    must([x[0] for x in sp] == ['mandatory', 'extra'] and sp[0][4] == 1 / pi and sp[1][1] is None, 'picks: a singleton pool')
    must(S.is_mandatory(K(cand('neutral', 0, offset=0.))) and not S.is_mandatory(K(cand('neutral', 5, offset=0.))) and not S.is_mandatory(K(cand('delayed', 0, offset=50.)))
         and not S.is_mandatory(K(cand('neutral', 0, offset=0., shadow='retain'))), 'the mandatory rule (neutral, entry, decision 0)')
    must(get('dup', 'post', 'mandatory')[2] == 2 and get('dup', 'post', 'extra')[1]['source'] == 'pi0' and abs(tot('dup', 'post') - 1 / pi) < 1e-12,
         'picks: same-time rows of two sources')
    must(get('empty', 'post', 'only')[1] is None and get('empty', 'keep', 'keep')[1] is None and get('empty', 'post', 'only')[2] == 0, 'picks: empty pools')
    oth = sorted((K(r) for r in pools['full'] if r['kind'] == 'post' and r['source'] != 'neutral'), key=lambda k: (S.rank('row', k), S.ckey(k)))
    must(get('full', 'post', 'extra')[1] == oth[0] and not S.is_mandatory(get('full', 'post', 'extra')[1]), 'picks: the extra is not the lowest-ranked other row')
    must(refuses(S.picks_of, [K(pools['full'][1]), K(pools['full'][1])], 'post', pi), 'picks: two rows with one key were accepted')
    must(refuses(S.picks_of, [K(cand('neutral', 0, offset=0.)), K(dict(cand('neutral', 0, offset=0.), generation=2.))], 'post', pi), 'picks: two mandatory rows were accepted')
    rng = np.random.default_rng(11); pops = {}
    for h, nw in ((0, 3), (1, 1)):                                                          # two strata: N 3 (n 2) and N 1 (n 1)
        for i in range(nw):
            pl_ = [cand('neutral', 0, offset=0., oid=10 * h + i)] * (i % 2 == 0) + [cand(s_, 5 * (j + 1), oid=10 * h + i) for j, s_ in enumerate(('delayed', 'pi0', 'joint_chase')[:i + 1])]
            pl_ += [cand('pi0', 5 * (j + 1), 'keep', oid=10 * h + i) for j in range(i % 2 + 1)] if h == 0 else []
            pops[(h, i)] = [(K(c), float(rng.integers(-50, 50))) for c in pl_]
    n_h = {0: 2, 1: 1}; N_h = {0: 3, 1: 1}; truth = 0.; per_world = {}
    for (h, i), xs in pops.items():
        e_w = 0.
        for head in ('post', 'keep'):
            ks = [k for k, _ in xs if k['kind'] == head]; x = {S.ckey(k): v for k, v in xs if k['kind'] == head}
            if not ks:
                continue
            truth += np.mean(list(x.values())); ps = S.picks_of(ks, head, n_h[h] / N_h[h]); m = [p for p in ps if p[0] == 'mandatory']
            if m:                                                                           # every equally likely extra draw
                others = [k for k in ks if not S.is_mandatory(k)]; ex = [p for p in ps if p[0] == 'extra'][0]
                e_w += np.mean([m[0][4] * x[S.ckey(m[0][1])] + (ex[4] * x[S.ckey(k)] if others else 0.) for k in (others or [None])])
            else:                                                                           # every eligible row equally likely
                e_w += np.mean([ps[0][4] * x[S.ckey(k)] for k in ks])
        per_world[(h, i)] = e_w
    est = [sum(per_world[(0, i)] for i in sub) + per_world[(1, 0)] for sub in itertools.combinations(range(3), 2)]   # every equally likely world subset
    must(abs(np.mean(est) - truth) < 1e-9, f'weights: E[sum w x] {np.mean(est)} != the population total of pool means {truth}')
    # ---- (3) end to end on the synthetic world
    wroot = Path(os.environ['PF_TEACHER_SYN']) if os.environ.get('PF_TEACHER_SYN') else None
    sd0 = S.STRICT_DAY; S.STRICT_DAY = False
    try:
        with tempfile.TemporaryDirectory(dir=str(I.K.IO)) as tmp:
            tmp = Path(tmp)
            if wroot is None:
                wroot = TS.build_world(tmp / 'world')
            wj = json.loads((wroot / 'world.json').read_text()); kp = Path(wj['kp']); day = '20261024'
            M, sup = LS.stubs(-12.), LS.OpenSupport(); files = {'loader': 'v1', 'clock': 'c1', 'metadata': 'r1'}
            ht = lambda d: (X.syn_inputs(d, [])[2], {})
            kw = dict(src=KP.Syn(), inputs=X.syn_inputs, feat_fn=X.syn_features(), files_fn=lambda d: dict(files), ht_fn=ht, assets=lambda: (M, sup, None), cfg=None)
            pkw = dict(days=(day,), pilot=False, per_hour=4, kp=kp, src=KP.Syn(), files_fn=lambda d: dict(files), ht_fn=ht, cfg=None, freeze_sha='synthetic',
                       assets=lambda: (M, sup, None), part_size=2)
            try:
                pa = quiet(S.make_plan, tmp / 'A', **pkw); quiet(S.make_plan, tmp / 'B', **pkw); np_ = len(pa['partitions'])
                pp_ = quiet(S.make_plan, tmp / 'Pi', **dict(pkw, pilot=True))
            except Exception as e:                                                          # a failed plan is this check's failure
                pa = None; must(False, f'plan: making a plan raised {e!r}')
            ks = [(x['m'], x['t_ms']) for x in pa['sample'][day]['selected'][:2]]; wk_ = S.worlds_all(day, kp, keys=ks)
            must(sorted((x['m'], x['t_ms']) for x in wk_) == sorted(ks) and all(set(x['o']) == {0, 1} for x in wk_), 'worlds: a keyed build is not exactly the keyed worlds')
            first = sorted((x for x in pa['sample'][day]['selected']), key=lambda x: x['rank'])[0]
            must(pp_['run_set'] == 'pilot' and [p_['members'] for p_ in pp_['partitions']] == [[S.wkey(first)]] and pp_['sample'] == pa['sample'],
                 'plan: the pilot run set is not the first-ranked world of each stratum of the same sample')
            must(np_ >= 2 and (tmp / 'A' / 'plan.json').read_bytes() == (tmp / 'B' / 'plan.json').read_bytes(), f'plan: {np_} partitions (>= 2 needed) or two equal plans differ')
            must(refuses(S.make_plan, tmp / 'A', **pkw, msg='immutable'), 'plan: an existing plan was rewritten')
            hw0 = S.hwm
            try:
                S.hwm = lambda: S.PEAK + 1.
                r_pk = refuses(S.make_plan, tmp / 'Pk', **pkw, msg='high-water')
            finally:
                S.hwm = hw0
            stp_pk = json.loads((tmp / 'Pk' / 'plan_status.json').read_text()) if (tmp / 'Pk' / 'plan_status.json').exists() else {}
            must(r_pk and not (tmp / 'Pk' / 'plan.json').exists() and stp_pk.get('state') == 'STOP_RESOURCE', 'plan: the high-water checkpoint did not refuse and latch')
            must(refuses(S.make_plan, tmp / 'Pk', **pkw, msg='stop history'), 'plan: a directory with a stop history was reused')
            try:
                sa = [quiet(S.run, tmp / 'A', k, **kw) for k in range(np_)]
                sb0 = quiet(S.run, tmp / 'B', 0, stop_after=1, **kw); prom0 = S.part_paths(tmp / 'B', 0)[0].exists()
                sb = [quiet(S.run, tmp / 'B', k, **kw) for k in range(np_)]; sa2 = quiet(S.run, tmp / 'A', 0, **kw)
            except Exception as e:                                                          # a failed run is this check's failure
                sa = None; must(False, f'runs: a run raised {e!r}')
            same = all(S.part_paths(tmp / 'A', k)[j].read_bytes() == S.part_paths(tmp / 'B', k)[j].read_bytes() for k in range(np_) for j in (1, 2))
            must(sa == ['PART_DONE'] * np_ and sb0 == 'STOPPED' and not prom0 and sb == ['PART_DONE'] * np_ and same and sa2 == 'DONE',
                 f'runs / resume: {sa}, stopped {sb0} (promoted {prom0}), {sb}, equal {same}, rerun {sa2}')
            # ---- (4) the validator
            plan, psha = S.load_plan(tmp / 'A'); part = plan['partitions'][0]; rp, fp, pp = S.part_paths(tmp / 'A', 0)
            rec = json.loads(rp.read_text()); b0, pb0 = fp.read_bytes(), pp.read_bytes(); rows, pools_ = S.gz_rows(b0), S.gz_rows(pb0)
            must(not S.validate(plan, psha, part, rows, pools_, rec, (b0, pb0)) and not S.validate(plan, psha, part, rows, pools_),
                 f'validator: the intact part is refused ({S.validate(plan, psha, part, rows, pools_, rec, (b0, pb0))[:2]})')
            tm = rec.get('timing') or {}; picked0 = sum(1 for r in rows if r['status'] in S.PICKED)
            must(isinstance(tm.get('load_s'), float) and tm['load_s'] > 0 and len(tm.get('worlds', [])) == len(part['members']) and tm.get('pairs') == picked0
                 and all(w_['world_s'] >= 0 for w_ in tm['worlds']), f'records: the completion record lacks its timing ({str(tm)[:120]})')
            okr = [i for i, r in enumerate(rows) if r['status'] == 'ok']; mx = [i for i in okr if rows[i]['pick'] == 'extra']
            must(bool(okr) and bool(mx), 'validator: part 0 has no ok row or no ok extra pick for the corruptions')
            i = okr[0]; j = mx[0]; arm0, arm1 = S.NAMES[rows[i]['head']]; ow = plan['partitions'][1]['members'][0]
            def mod(fn, which='rows'):
                rr, qq = json.loads(json.dumps(rows)), json.loads(json.dumps(pools_)); fn(rr if which == 'rows' else qq); return rr, qq
            def mod2(fn):
                rr, qq = json.loads(json.dumps(rows)), json.loads(json.dumps(pools_)); fn(rr, qq); return rr, qq
            def relabel(r, A):                                                              # keep R = cash + position x mark, the fills hash / count and Y consistent
                A['fills'] = len(A['fill_list']); A['fills_sha256'] = S.L._h(A['fill_list']); sg = lambda s: 1. if int(s) == 0 else -1.
                A['cash'] = -sum(sg(f[2]) * float(f[3]) * float(f[4]) for f in A['fill_list']); A['position'] = sum(sg(f[2]) * float(f[3]) for f in A['fill_list'])
                A['value'] = A['cash'] + A['position'] * A['mark']; a0, a1 = S.NAMES[r['head']]; r['result']['Y'] = r['result']['arms'][a0]['value'] - r['result']['arms'][a1]['value']
            def add_fill(r, arm, before, qty_over=False):
                A = r['result']['arms'][arm]; g = float(r['result']['k']) * S.L.STEP; filled = {}
                for f in A['fill_list']:
                    filled[int(f[1])] = filled.get(int(f[1]), 0.) + float(f[3])
                gen = [x for x in range(len(A['sizes'])) if A['sizes'][x] - filled.get(x, 0.) >= 1.][0]
                q = (A['sizes'][gen] - filled.get(gen, 0.) + 1.) if qty_over else 1.
                A['fill_list'] = sorted(A['fill_list'] + [[g - 1. if before else g + 1., gen, 0, q, 40.]], key=lambda f: float(f[0])); relabel(r, A)
            def ledger_shift(r):
                A = r['result']['arms'][arm0]; A['position'] += 1; A['cash'] -= A['mark']
            def keyshift(r):
                r['row_key'] = dict(r['row_key'], generation=(r['row_key']['generation'] or 0) + 7); r['row_rank'] = S.rank('row', r['row_key'])   # only the identity check sees it
            def bump_n(r):
                r['N'] += 1; r['q'] = 1. / (r['N'] - 1); r['w'] = 1. / (r['pi'] * r['N'] * r['q'])
            pk_i = [n for n, p in enumerate(pools_) if p['keys']][0]
            corrupt = {
                'Y + 1': mod(lambda rr: rr[i]['result'].__setitem__('Y', rr[i]['result']['Y'] + 1.)),
                'another fork': mod(lambda rr: rr[i]['result'].__setitem__('fork_fingerprint', '0' * 64)),
                'reconciled false': mod(lambda rr: rr[i]['result'].__setitem__('reconciled', False)),
                'binding failure': mod(lambda rr: rr[i]['result']['arms'][arm0]['teacher'].__setitem__('binding_bad', 1)),
                'causal mismatch': mod(lambda rr: rr[i]['causal'][0].__setitem__('mismatch', 1)),
                'fork input hash': mod(lambda rr: rr[i]['fork'].__setitem__('inputs_sha256', '0' * 64)),
                'hidden copy unequal': mod(lambda rr: rr[i]['fork'].__setitem__('hidden_equal', False)),
                'rebuild unequal': mod(lambda rr: rr[i]['fork'].__setitem__('rebuild_equal', False)),
                'extra row (another partition)': mod(lambda rr: rr.append(dict(rr[i], m=ow['m'], t_ms=ow['t_ms']))),
                'duplicate row': mod(lambda rr: rr.append(dict(rr[i]))),
                'missing pick': mod(lambda rr: rr.pop(j)),
                'wrong weight': mod(lambda rr: rr[i].__setitem__('w', rr[i]['w'] * 1.01)),
                'a self-consistent N': mod(lambda rr: bump_n(rr[j])),
                'a changed row key': mod(lambda rr: keyshift(rr[i])),
                'R != cash + position x mark': mod(lambda rr: (rr[i]['result']['arms'][arm0].__setitem__('value', rr[i]['result']['arms'][arm0]['value'] + 1.),
                                                               rr[i]['result'].__setitem__('Y', rr[i]['result']['Y'] + 1.))),
                'a self-consistent ledger change': mod(lambda rr: ledger_shift(rr[i])),
                'an added (zero) fill': mod(lambda rr: rr[i]['result']['arms'][arm0]['fill_list'].append([1e9, 0, 0, 0., 1.])),
                'a consistent pre-fork fill in one arm': mod(lambda rr: add_fill(rr[i], arm1, True)),
                'an overfilled generation': mod(lambda rr: add_fill(rr[i], arm0, False, True)),
                'another side': mod(lambda rr: rr[i]['result'].__setitem__('side', 1 - rr[i]['result']['side'])),
                'another original': mod(lambda rr: rr[i]['result']['ident'].__setitem__('original_episode_id', rr[i]['result']['ident']['original_episode_id'] + 1)),
                'another decision': mod(lambda rr: rr[i]['result'].__setitem__('k', rr[i]['result']['k'] + 1)),
                'a pool key added': mod(lambda qq: qq[pk_i]['keys'].append(dict(qq[pk_i]['keys'][0], decision_ms=987650)), 'pools'),
                'a pool line dropped': mod(lambda qq: qq.pop(0), 'pools'),
                'inconsistent pool counts': mod(lambda qq: qq[pk_i]['counts'].__setitem__('ok', qq[pk_i]['counts']['ok'] + 1), 'pools'),
                'a duplicated pool line': mod(lambda qq: qq.append(dict(qq[0])), 'pools'),
                'a duplicated pool key': mod(lambda qq: (qq[pk_i]['keys'].append(dict(qq[pk_i]['keys'][0])), qq[pk_i]['counts'].__setitem__('ok', qq[pk_i]['counts']['ok'] + 1),
                                                         qq[pk_i]['kinds'].__setitem__(qq[pk_i]['head'], qq[pk_i]['kinds'][qq[pk_i]['head']] + 1)), 'pools'),
                'a NaN mark': mod(lambda rr: rr[i]['result']['arms'][arm0].__setitem__('mark', float('nan')))}
            w0 = part['members'][0]; wk0 = (w0['m'], w0['t_ms']); sel0 = [x for x in plan['sample'][day]['selected'] if (x['m'], x['t_ms']) == wk0][0]
            def as_world_status(rr, qq, n=4, keep_pool=False):                              # a world without an episode: four status picks, no pool lines
                base = dict(day=day, m=w0['m'], t_ms=w0['t_ms'], hour=sel0['hour'], world_rank=sel0['rank'], pi=sel0['pi'])
                rr[:] = [r for r in rr if (r['m'], r['t_ms']) != wk0] + [dict(base, side=s_, oid=sel0[f'oid{s_}'], head=h_, pick=p_, status='world:no_episode', N=None, q=None,
                                                                              w=None) for s_ in (0, 1) for h_, p_ in (('post', 'only'), ('keep', 'keep'))][:n]
                if not keep_pool:
                    qq[:] = [x for x in qq if (x['m'], x['t_ms']) != wk0]
            def to_single(rr, qq, status='no_additional_post'):                             # a POST pool reduced to its mandatory row
                ln = [x for x in qq if x['head'] == 'post' and any(S.is_mandatory(k) for k in x['keys']) and len(x['keys']) >= 2][0]
                mk_ = [k for k in ln['keys'] if S.is_mandatory(k)][0]; n_rm = len(ln['keys']) - 1
                ln['keys'] = [mk_]; ln['counts']['ok'] -= n_rm; ln['counts']['unsampled'] = ln['counts'].get('unsampled', 0) + n_rm
                at = lambda r: (r['m'], r['t_ms'], r['side'], r['head']) == (ln['m'], ln['t_ms'], ln['side'], 'post')
                for r in rr:
                    if at(r) and r['pick'] == 'mandatory':
                        r['N'] = 1; r['w'] = 1. / (r['pi'] * 1 * 1.)
                ex = [r for r in rr if at(r) and r['pick'] == 'extra'][0]
                rr[rr.index(ex)] = dict({k_: ex[k_] for k_ in ('day', 'm', 't_ms', 'hour', 'world_rank', 'pi', 'side', 'oid', 'head', 'pick')}, N=1, q=None, w=None, status=status)
            must(not S.validate(plan, psha, part, *mod2(as_world_status)), f'validator: a valid world status was refused {S.validate(plan, psha, part, *mod2(as_world_status))[:2]}')
            must(not S.validate(plan, psha, part, *mod2(to_single)), f'validator: a valid no_additional_post pick was refused {S.validate(plan, psha, part, *mod2(to_single))[:2]}')
            corrupt['a world status with a missing pick'] = mod2(lambda rr, qq: as_world_status(rr, qq, n=3))
            corrupt['a world status with pool lines'] = mod2(lambda rr, qq: as_world_status(rr, qq, keep_pool=True))
            corrupt['an empty pick with the wrong status'] = mod2(lambda rr, qq: to_single(rr, qq, 'empty_pool'))
            missed = [n for n, (rr, qq) in corrupt.items() if not vbad(S, plan, psha, part, rr, qq)]   # no record: each check alone
            rec_bad = {'a part byte': (rec, (b0[:-1] + bytes([b0[-1] ^ 1]), pb0)), 'a pool byte': (rec, (b0, pb0[:-1] + bytes([pb0[-1] ^ 1]))),
                       'other sources': (dict(rec, sources=dict(rec['sources'], **{'pf_r2_labels.py': '0' * 64})), (b0, pb0)),
                       'other inputs': (dict(rec, inputs=dict(rec['inputs'], offset=999.)), (b0, pb0)), 'another plan': (dict(rec, plan_sha256='0' * 64), (b0, pb0)),
                       'a breach': (dict(rec, hwm_mib=S.PEAK + 1.), (b0, pb0)), 'wrong counts': (dict(rec, rows=rec['rows'] + 1), (b0, pb0))}
            missed += [n for n, (rc, raw) in rec_bad.items() if not vbad(S, plan, psha, part, rows, pools_, rc, raw)]
            must(not missed, f'validator: corruptions not refused {missed}')
            plan_s = json.loads(json.dumps(plan)); plan_s['sources']['pf_r2_labels.py'] = '0' * 64
            must(any('executing sources' in x for x in (vbad(S, plan_s, psha, part, rows, pools_) or [])), 'validator: other executing sources were not refused')
            must(any('not a selected world' in x for x in (vbad(S, plan, psha, dict(part, members=part['members'] + [dict(day=day, m=999999, t_ms=1)]), rows, pools_) or [])),
                 'validator: a member outside the sample was accepted')
            n_del = sum(1 for k in range(np_) for r in S.gz_rows(S.part_paths(tmp / 'A', k)[1].read_bytes()) if r['status'] == 'ok' and r['fork'].get('delivered_changed'))
            must(n_del > 0, 'fork inputs: the delivered positive control never changed the inputs')
            # ---- (5) identity
            def copy_of(name, keep_parts=True):
                d = tmp / name; shutil.copytree(tmp / 'A', d)
                if not keep_parts:
                    shutil.rmtree(d / 'parts')
                return d
            def no_read(dd, mk):
                raise RuntimeError('a day input was read before the identity check')
            d = copy_of('Rs'); pl = json.loads((d / 'plan.json').read_text()); pl['sources']['pf_r2_labels.py'] = '0' * 64; (d / 'plan.json').write_text(json.dumps(pl))
            must(refuses(S.run, d, 0, **kw, msg='stale source'), 'identity: a stale source was not refused before the skip')
            must(refuses(S.run, copy_of('Rf'), 0, **dict(kw, files_fn=lambda dd: dict(files, clock='c2'), inputs=no_read), msg='changed input'),
                 'identity: changed loader files were not refused before the skip')
            must(refuses(S.run, copy_of('Rh'), 0, **dict(kw, ht_fn=lambda dd: (ht(dd)[0], {'hour': 'changed'}), inputs=no_read), msg='changed input'),
                 'identity: changed tracked hours were not refused before the skip')
            def teacher_changed():
                raise ValueError('not the frozen pi0_50_v1 (freeze sha)')
            must(refuses(S.run, copy_of('Rt'), 0, **dict(kw, assets=teacher_changed), msg='frozen'), 'identity: a changed teacher was not refused')
            class Shifted(KP.Syn):
                def offset(self, d):
                    return super().offset(d) + 1.
            must(refuses(S.run, copy_of('Ro'), 0, **dict(kw, src=Shifted(), inputs=no_read), msg='changed input'), 'identity: a changed tape offset was not refused before the skip')
            d = copy_of('Rx'); pl = json.loads((d / 'plan.json').read_text()); pl['feature_schema']['n'] = 106; (d / 'plan.json').write_text(json.dumps(pl))
            must(refuses(S.run, d, 0, **kw, msg='feature schema'), 'identity: a changed feature schema was not refused')
            d = copy_of('Rpl'); pl = json.loads((d / 'plan.json').read_text()); pl['limits']['load_s'] = 601.; (d / 'plan.json').write_text(json.dumps(pl))
            must(refuses(S.run, d, 0, **kw, msg='fails validation ('), 'identity: a changed plan did not refuse the records of the old one')
            must(refuses(S.run, copy_of('Rp', False), 0, **dict(kw, inputs=lambda dd, mk: ({}, None, ht(dd)[0], {'hour': 'changed'})), msg='real_inputs'),
                 'identity: real_inputs\' own tracked hashes were not compared')
            d = copy_of('Rm', False); pl = json.loads((d / 'plan.json').read_text()); fake = dict(pl['sample'][day]['selected'][0], m=999999, t_ms=1)
            pl['sample'][day]['selected'].append(fake); pl['partitions'][0]['members'].append(S.wkey(fake)); (d / 'plan.json').write_text(json.dumps(pl))
            must(refuses(S.run, d, 0, **kw, msg="not one of the day's worlds"), 'identity: a member world missing from the day was not refused')
            d = copy_of('Rk', False); kp2 = tmp / 'kp2'; shutil.copytree(kp, kp2); f_ = kp2 / day / 'orders.parquet'; O_ = pd.read_parquet(f_)
            O_.loc[O_.index[0], 'p'] = O_.p.iloc[0] + 1.; O_.to_parquet(f_); pl = json.loads((d / 'plan.json').read_text()); pl['kp_dir'] = str(kp2); (d / 'plan.json').write_text(json.dumps(pl))
            must(refuses(S.run, d, 0, **kw, msg='keep-pull outputs changed'), 'identity: a changed keep / pull output was not refused')
            # ---- (6) stops
            d = copy_of('H', False); (d / 'part_000_status.json').write_text(json.dumps(dict(state='STOP_HARD_RSS', rss_mib=2100.)))
            h1, h1b = quiet(S.run, d, 0, **kw), quiet(S.run, d, 1, **kw)
            ev = [json.loads(x) for x in (d / 'stops.jsonl').read_text().splitlines()] if (d / 'stops.jsonl').exists() else []
            (d / 'disposition.json').write_text(json.dumps(dict(plan_sha256='0' * 64, entries=[dict(id=e['id'], name='selftest', reason='selftest') for e in ev])))
            h2 = quiet(S.run, d, 0, **kw)
            (d / 'disposition.json').write_text(json.dumps(dict(plan_sha256=C.sha(d / 'plan.json'), entries=[dict(id=e['id'], name='selftest', reason='selftest') for e in ev])))
            h3 = quiet(S.run, d, 0, **kw)
            must(h1 == 'HELD' and h1b == 'HELD' and len(ev) == 1 and h2 == 'HELD' and h3 == 'PART_DONE', f'stops: a hard stop not held until its disposition ({h1}, {h1b}, {len(ev)}, {h2}, {h3})')
            d = copy_of('H2', False); stp2 = d / 'part_000_status.json'
            stp2.write_text(json.dumps(dict(state='STOP_RESOURCE', note='first'))); g1 = quiet(S.run, d, 0, **kw)
            ev1 = [json.loads(x) for x in (d / 'stops.jsonl').read_text().splitlines()]
            (d / 'disposition.json').write_text(json.dumps(dict(plan_sha256=C.sha(d / 'plan.json'), entries=[dict(id=e['id'], name='selftest', reason='selftest') for e in ev1])))
            stp2.write_text(json.dumps(dict(state='STOP_RESOURCE', note='second'))); g2 = quiet(S.run, d, 0, **kw)
            ev2 = [json.loads(x) for x in (d / 'stops.jsonl').read_text().splitlines()]
            (d / 'disposition.json').write_text(json.dumps(dict(plan_sha256=C.sha(d / 'plan.json'), entries=[dict(id=e['id'], name='selftest', reason='selftest') for e in ev2])))
            g3 = quiet(S.run, d, 0, **kw)
            must(g1 == 'HELD' and len(ev1) == 1 and g2 == 'HELD' and len(ev2) == 2 and ev2[0]['id'] == ev1[0]['id'] and g3 == 'PART_DONE',
                 f'stops: two successive stops at one status file did not both hold ({g1}, {len(ev1)}, {g2}, {len(ev2)}, {g3})')
            class _Life:
                pass
            reg, viol, keep, calls = [], [], [], []; wb0, wr0 = S.L.world_bind, S.world_rows
            def wb_s(*a_, **k_):
                world_, st_ = wb0(*a_, **k_)
                if world_ is not None:
                    s_ = _Life(); world_['_lifetime'] = s_; reg.append(weakref.ref(s_)); keep.extend([world_] if retain else [])
                return world_, st_
            def wr_s(day_, w_, world_, *a_, **k_):                                          # at every world: no earlier world alive
                gc.collect(); cur = world_.get('_lifetime') if world_ else None; viol.extend(r_ for r_ in reg if r_() is not None and r_() is not cur)
                return wr0(day_, w_, world_, *a_, **k_)
            def inputs_count(dd, mk):
                calls.append((dd, tuple(mk))); return X.syn_inputs(dd, mk)
            life = {}
            for retain, name in ((False, 'Lf0'), (True, 'Lf1')):
                reg.clear(); viol.clear(); keep.clear(); calls.clear(); d = copy_of(name, False); S.L.world_bind, S.world_rows = wb_s, wr_s
                try:
                    st_l = quiet(S.run, d, 0, **dict(kw, inputs=inputs_count))
                except Exception as e:
                    st_l = f'raised {e!r}'
                finally:
                    S.L.world_bind, S.world_rows = wb0, wr0
                life[retain] = (st_l, len(viol), list(calls))
            keep.clear(); mk0 = tuple(sorted({m_['m'] for m_ in part['members']}))
            must(life[False][0] == 'PART_DONE' and life[False][1] == 0 and life[False][2] == [(day, mk0)] and life[True][1] > 0,
                 f'lifetime: earlier worlds alive {life[False][1]} (must be 0), inputs read {life[False][2]} (once, for its day), retained control {life[True][1]} (must be > 0)')
            def run_state(d, **k_):                                                         # a refusal or crash is a state for the must below
                try:
                    return quiet(S.run, d, 0, **dict(kw, **k_))
                except Exception as e:
                    return f'raised {e!r}'
            def latched(d, stage, field):
                s_ = json.loads((d / 'part_000_status.json').read_text()) if (d / 'part_000_status.json').exists() else {}
                la = s_.get('latch') or {}
                return s_.get('state') == 'STOP_RESOURCE' and la.get('stage') == stage and field(la) and not S.part_paths(d, 0)[0].exists()
            d = copy_of('Lp', False); hw0 = S.hwm
            try:
                S.hwm = lambda: S.PEAK + 1.
                sp = run_state(d)
            finally:
                S.hwm = hw0
            must(sp == 'STOP_RESOURCE' and latched(d, 'load', lambda la: la['hwm_mib'] > la['peak_mib']), 'stops: a high-water over the checkpoint did not latch at the loading')
            quiet(S.make_plan, tmp / 'Lq', **dict(pkw, limits=dict(load_s=600., world_s=120., pair_s=1e-9)))
            must(run_state(tmp / 'Lq') == 'STOP_RESOURCE' and latched(tmp / 'Lq', 'pair', lambda la: la['max_pair_s'] > la['pair_limit_s']),
                 'stops: a pair over its limit did not latch at the pair')
            quiet(S.make_plan, tmp / 'Lb', **dict(pkw, limits=dict(load_s=0., world_s=0., pair_s=120.)))
            must(run_state(tmp / 'Lb') == 'STOP_RESOURCE' and latched(tmp / 'Lb', 'load', lambda la: la['elapsed_s'] > la['bound_s']),
                 'stops: the process bound did not latch at the loading')
            must(run_state(tmp / 'Lb') == 'HELD', 'stops: a latched stop did not hold the rerun')
            def seq_hwm(n_low):
                calls = [0]
                def f():
                    calls[0] += 1
                    return 100. if calls[0] <= n_low else S.PEAK + 1.
                return f
            n_pairs0 = sum(1 for r in rows if (r['m'], r['t_ms']) == wk0 and r['status'] in S.PICKED)
            for name, n_low, stage in (('Lc', 1, 'census'), ('Lw', 2 + n_pairs0, 'world')):
                d = copy_of(name, False); hw0 = S.hwm
                try:
                    S.hwm = seq_hwm(n_low); sp = run_state(d)
                finally:
                    S.hwm = hw0
                must(sp == 'STOP_RESOURCE' and latched(d, stage, lambda la: la['hwm_mib'] > la['peak_mib']), f'stops: the high-water was not checked at the {stage}')
            # ---- (7) promotion and resume
            d = copy_of('V', False); tv0 = S.TOL_V
            try:
                S.TOL_V = -1.
                r_v = refuses(S.run, d, 0, **kw, msg='refused at promotion')
            finally:
                S.TOL_V = tv0
            must(r_v and not S.part_paths(d, 0)[0].exists() and not S.part_paths(d, 0)[1].exists(), 'promotion: an invalid part was promoted')
            d = copy_of('Cr'); f_ = S.part_paths(d, 0)[2]; f_.write_bytes(S.gz_bytes(S.gz_rows(f_.read_bytes())[1:]))
            must(refuses(S.run, d, 0, **kw, msg='do not hash to the record'), 'resume: a changed promoted part was skipped as done')
            d = copy_of('Ch'); rp_, fp_, _ = S.part_paths(d, 0); rr = S.gz_rows(fp_.read_bytes()); rr[i]['result']['Y'] += 1.; nb = S.gz_bytes(rr); fp_.write_bytes(nb)
            rc_ = json.loads(rp_.read_text()); rc_['sha256'] = hashlib.sha256(nb).hexdigest(); rp_.write_text(json.dumps(rc_))
            must(refuses(S.run, d, 0, **kw, msg='fails validation ('), 'resume: a re-hashed invalid part was skipped as done')
            # ---- (8) finalize
            fkw = dict(src=KP.Syn(), files_fn=lambda dd: dict(files), ht_fn=ht, assets=lambda: (M, sup, None), cfg=None)
            d = copy_of('F1'); [S.part_paths(d, 1)[x].unlink() for x in range(3)]
            must(refuses(S.finalize, d, **fkw, msg='not promoted'), 'finalize: an unpromoted part was accepted')
            d = copy_of('F2'); (d / 'part_001_status.json').write_text(json.dumps(dict(state='STOP_RESOURCE', note='selftest')))
            must(refuses(S.finalize, d, **fkw, msg='undisposed'), 'finalize: an undisposed stop was accepted')
            must(refuses(S.finalize, copy_of('F3'), **dict(fkw, files_fn=lambda dd: dict(files, metadata='r2')), msg='changed input'), 'finalize: a changed input was accepted')
            must(refuses(S.finalize, copy_of('F4'), **dict(fkw, assets=teacher_changed), msg='frozen'), 'finalize: a changed teacher was accepted')
            d = copy_of('F6'); pl = json.loads((d / 'plan.json').read_text()); pl['partitions'][1]['members'] = pl['partitions'][1]['members'] + pl['partitions'][0]['members'][:1]
            (d / 'plan.json').write_text(json.dumps(pl))
            must(refuses(S.finalize, d, **fkw, msg='exactly once'), 'finalize: overlapping partitions were accepted')
            hw0 = S.hwm
            try:
                S.hwm = lambda: S.PEAK + 1.
                r_f = refuses(S.finalize, copy_of('F5'), **fkw, msg='high-water')
            finally:
                S.hwm = hw0
            must(r_f and not (tmp / 'F5' / 'accepted.json').exists(), 'finalize: a high-water over the checkpoint was accepted')
            acc = quiet(S.finalize, tmp / 'A', **fkw); allrows = [r for k in range(np_) for r in S.gz_rows(S.part_paths(tmp / 'A', k)[1].read_bytes())]
            wt, st_ = {}, {}
            for r in allrows:
                st_[f"{r['day']} {r['head']} {r['status']}"] = st_.get(f"{r['day']} {r['head']} {r['status']}", 0) + 1
                if r['w'] is not None:
                    wt[f"{r['day']} {r['head']}"] = wt.get(f"{r['day']} {r['head']}", 0.) + r['w']
            must(acc['accepted'] and acc['rows'] == len(allrows) and acc['statuses'] == st_ and set(acc['weight_totals']) == set(wt)
                 and all(abs(acc['weight_totals'][k_] - v_) < 1e-9 for k_, v_ in wt.items()) and (tmp / 'A' / 'accepted.json').exists(),
                 'finalize: the accepted manifest does not match the parts')
    finally:
        S.STRICT_DAY = sd0
    print(f'r2 label sample selftests passed (synthetic only): UTC hour / world sample, picks / weights / enumerated IPW identity, end to end ({np_} parts, '
          f'resume identical), validator refuses {len(corrupt) + len(rec_bad)} corruptions, identity refusals (10), stops (held, two successive stops, high-water at the loading / census / world, pair, bound), lifetime sentinel, '
          f'promotion / resume, finalize (6 refusals, manifest)')


if __name__ == '__main__':
    import pf_r2_label_sample as S_
    run(S_)
