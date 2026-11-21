"""R2-F PRODUCTION CHECKS of the full r2_label_sample_v2 build (review 38febc0b's next priority). Declared before any production output is
read (2026-11-20, session aea5bfb9); consumed Oct 24-25. Every claim is a computed predicate in checks.json; the readings are reported, never
gated. Label-only: target support and readiness, no model, no prediction skill.
CHECKS
 (A) plan: the frozen sample equals attempt 4's (the same 3,072 selected worlds, strata and probabilities). The run set is the full sample,
     in partitions of <= 32 that cover it exactly once; audit_every 35; the declared limits; strict UTC days; 107 features.
 (B) runs: every partition job exits 0 and is promoted.
 (C) parity: r2f_parity.py reproduced all 288 accepted pilot pairs byte for byte (only the declared audit fields whitelisted).
 (D) acceptance: finalize accepted the production dataset (the shared validator over every record).
 (E) resources: every job's measured peak and every record's high-water <= 1,536 MiB; every pair <= 120 s; every partition within its
     bound; no stop archived.
 (F) labels: every part revalidated now, with the executing sources. Over every ok pair's arms: 0 causal, parity and binding failures,
     with audit checks > 0. The delivered positive control changed the fork inputs on >= 1 ok row a day.
 (G) controls (in memory, every partition): each refusal below on its first ok row; the intact part valid with its record and bytes:
     - the review 4393ad19 four: Y + 1, another fork, reconciled false, binding_bad 1;
     - review 38febc0b's three: feature_unavailable with a false reconciliation, an intact pair relabelled pair_not_ok, END 4,000.
 (H) suites: the producing files equal this round's freeze (logs/build_sources.txt); the v2 per-mutant summary passed on them with every
     planted error caught; v1 (35 / 35) and runner v5 (30 / 30) are unchanged since their passing runs; the summary selftest passes.
READINGS (review 38febc0b step 5; descriptive; no intervals; the full sample's own populations)
 - population: planned versus realized worlds, markets, UTC hours, originals, picks, pairs; statuses by day / side / head / source;
 - Y by day and head: zero / positive / negative counts; the distinct originals and worlds behind them, with the UNION of nonzero
   originals; mean, median, tails (1 / 5 / 95 / 99 %), min, max; the top-5 and top-1 % shares of the total |Y|;
 - fill paths: per ok pair, identical arm fill lists, path-only differences (lists differ, Y = 0, equal cash and inventory), or economic
   differences (Y != 0);
 - both arms by day and head: value / cash / position sums, nonzero end inventory, fills, filled quantity, generations, open exposure, the
   teacher's calls / actions / statuses;
 - coverage of the ok rows by day / head: source; decision_ms (elapsed since the original); remaining horizon (5,000 - decision_ms);
   generation; partial queue; support; UTC hour. Absent categories are explicit zeros;
 - weights by day and head: raw totals (picked, ok), weighted and raw mean Y, effective sample size, largest and top-10 % weight shares;
 - cost: actual partition CPU core-hours (and the parity run's), pairs (mean / max seconds), loading, census, peaks, every stop.
Usage: r2f_checks.py -> checks.json, population.csv, coverage.csv, y_summary.csv, paths.csv, arms.csv, weights.csv, cost.csv (here)"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_label_sample as S                            # noqa: E402

HERE = Path(__file__).resolve().parent; LOGS = Path('/home/yuqing/ctaNew/predictfun/reviews/r2f_production_20261120/logs'); RES = Path('/home/yuqing/ctaNew/predictfun/research')
D = Path('/home/yuqing/ctaNew/data/predict_fun/research/r2_label_sample_v2_20261120'); P, PILOT = D / 'parity_pilot_e35', D / 'attempt4_pilot_A'
BUILD = dict(ln.split() for ln in (LOGS / 'build_sources.txt').read_text().splitlines() if ln.strip())
PROD = ('pf_r2_label_sample.py', 'pf_r2_labels.py', 'pf_r2_runner.py')
RL = re.compile(r'tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')


def run_light(tag):
    p = LOGS / f'{tag}.log'
    m = [RL.search(x) for x in (p.read_text().splitlines() if p.exists() else []) if 'tag=' in x]
    m = [x for x in m if x and x.group(1) == tag]
    return dict(rc=int(m[-1].group(2)), peak=int(m[-1].group(3)), wall=float(m[-1].group(4)), cpu=float(m[-1].group(5))) if m else None


def dbin(ms):
    return '0' if ms == 0 else '50-500' if ms <= 500 else '550-1000' if ms <= 1000 else '1050-2000' if ms <= 2000 else '2050-4950'


def rbin(ms):
    r = 5000 - ms
    return '<=500' if r <= 500 else '550-1000' if r <= 1000 else '1050-2000' if r <= 2000 else '2050-4950' if r < 5000 else '5000'


def wcsv(name, head, rows_):
    with open(HERE / name, 'w', newline='') as f:
        w_ = csv.writer(f); w_.writerow(head); [w_.writerow(x) for x in rows_]


def main():  # noqa: C901
    ck, rd = {}, {}
    pa, psha = S.load_plan(P); po, _ = S.load_plan(PILOT); days = pa['days']
    sel = {d: pa['sample'][d]['selected'] for d in days}
    ck['A_plan'] = bool(pa['sample'] == po['sample'] and pa['run_set'] == 'full' and all(len(p['members']) <= 32 for p in pa['partitions'])
                        and sorted(S.ckey(m) for p in pa['partitions'] for m in p['members']) == sorted(S.ckey(S.wkey(w)) for d in days for w in sel[d])
                        and pa['audit_every'] == 35 and pa['strict_day'] is True and pa['feature_schema']['n'] == 107
                        and pa['limits'] == dict(load_s=600., world_s=120., pair_s=120., peak_mib=1536., guard_mib=2048.))
    jobs = {p['part']: run_light(f"r2f_run_{p['part']}") for p in pa['partitions']}
    ck['B_runs'] = bool(all(j is not None and j['rc'] == 0 for j in jobs.values()) and all(S.part_paths(P, p['part'])[0].exists() for p in pa['partitions']))
    par = json.loads((HERE / 'parity.json').read_text()); ck['C_parity'] = bool(par['parity'])
    acc = json.loads((P / 'accepted.json').read_text()); ck['D_acceptance'] = bool(acc['accepted'] and acc['plan_sha256'] == psha)
    # v1.2 (logs/declaration_v1_2.txt): one partition in memory at a time. Per partition: the revalidation, the controls, then the rows reduced
    # to the scalar fields the readings use (slim) and the per-row accumulations (causal, teacher, delivered, paths, arms, coverage), in v1.1's
    # order. The predicates and readings are v1.1's.
    recs, reval, ctl, slim = [], [], [], []
    cz, tch, dl, pth, cov = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    arm = collections.defaultdict(collections.Counter)

    def corrupt(part_, rows_, pools_, i, fn):
        rr = json.loads(json.dumps(rows_)); fn(rr[i]); return bool(S.validate(pa, psha, part_, rr, pools_))
    for part_ in pa['partitions']:
        rp, fp, pp = S.part_paths(P, part_['part']); rec = json.loads(rp.read_text()); b, q = fp.read_bytes(), pp.read_bytes(); recs.append(rec)
        reval.append(S.validate(pa, psha, part_, S.gz_rows(b), S.gz_rows(q), rec, (b, q)))
        rows_, pools_ = S.gz_rows(b), S.gz_rows(q); i = [n for n, r in enumerate(rows_) if r['status'] == 'ok'][0]; a0 = S.NAMES[rows_[i]['head']][0]
        c_ = {'Y + 1': corrupt(part_, rows_, pools_, i, lambda r: r['result'].__setitem__('Y', r['result']['Y'] + 1.)),
              'another fork': corrupt(part_, rows_, pools_, i, lambda r: r['result'].__setitem__('fork_fingerprint', '0' * 64)),
              'reconciled false': corrupt(part_, rows_, pools_, i, lambda r: r['result'].__setitem__('reconciled', False)),
              'binding_bad 1': corrupt(part_, rows_, pools_, i, lambda r: r['result']['arms'][a0]['teacher'].__setitem__('binding_bad', 1)),
              'feature_unavailable, reconciled false': corrupt(part_, rows_, pools_, i, lambda r: (r.__setitem__('status', 'feature_unavailable'),
                                                                                                r['fork'].__setitem__('status', 'feature_unavailable'),
                                                                                                r['result'].__setitem__('reconciled', False))),
              'relabelled pair_not_ok': corrupt(part_, rows_, pools_, i, lambda r: r.__setitem__('status', 'pair_not_ok')),
              'END 4,000': corrupt(part_, rows_, pools_, i, lambda r: r['result'].__setitem__('end', 4000.))}
        ctl.append(dict(part=part_['part'], refused=c_, intact=not S.validate(pa, psha, part_, rows_, pools_, rec, (b, q))))
        rows_ = pools_ = None
        for r in S.gz_rows(b):
            ok_r = r['status'] == 'ok'
            slim.append(dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], side=r['side'], head=r['head'], hour=r['hour'], source=r.get('source'), status=r['status'],
                             w=r['w'], Y=r['result']['Y'] if ok_r else None))
            if not ok_r:
                continue
            for c in r['causal']:
                for k_ in ('checked', 'mismatch', 'parity_checked', 'parity_mismatch'):
                    cz[k_] += c[k_]
            for n_ in S.NAMES[r['head']]:
                t = r['result']['arms'][n_]['teacher']
                tch['binding_checked'] += t['binding_checked']; tch['binding_bad'] += t['binding_bad']; tch['binding_bad_arms'] += int(t['binding_bad'] != 0)
                tch['zero_callback_arms'] += int(t['calls'] == 0)
            if r['fork']['delivered_changed']:
                dl[r['day']] += 1
            a_, b_ = (r['result']['arms'][n] for n in S.NAMES[r['head']])
            same = a_['fill_list'] == b_['fill_list']
            cls = 'identical fills' if same else ('path-only (Y = 0, equal cash and inventory)' if r['result']['Y'] == 0 and a_['cash'] == b_['cash'] and a_['position'] == b_['position']
                                                  else 'economic (Y != 0)' if r['result']['Y'] != 0 else 'other (Y = 0, cash or inventory differ)')
            pth[(r['day'], r['head'], cls)] += 1
            for n_ in S.NAMES[r['head']]:
                A_ = r['result']['arms'][n_]; t_ = A_['teacher'] or {}; c_ = arm[(r['day'], r['head'], n_)]
                c_['rows'] += 1; c_['value_sum'] += A_['value']; c_['cash_sum'] += A_['cash']; c_['position_sum'] += A_['position']; c_['inventory_nonzero_rows'] += int(A_['position'] != 0)
                c_['fills'] += A_['fills']; c_['filled_qty'] += sum(float(f[3]) for f in A_['fill_list']); c_['generations'] += A_['generations']
                c_['open_at_end_rows'] += int(any(float(v) > 0 for v in A_['open_at_end'].values())); c_['teacher_calls'] += t_.get('calls', 0)
                for k_, v_ in (t_.get('by_action') or {}).items():
                    c_[f'action {k_}'] += v_
                for k_, v_ in (t_.get('by_status') or {}).items():
                    c_[f'status {k_}'] += v_
            k = r['row_key']; g_ = k['generation']
            for dim, v in (('source', r['source']), ('decision_ms', dbin(k['decision_ms'])), ('remaining_ms', rbin(k['decision_ms'])),
                           ('generation', 'none' if g_ is None else ('3+' if g_ >= 3 else str(g_))), ('partial_queue', 'known_filled>0' if r['result']['fork_known_filled'] > 0 else 'none'),
                           ('support', str(r['fork']['support'])), ('utc_hour', f"{r['hour']:02d}"), ('side', str(r['side']))):
                cov[(r['day'], r['head'], dim, v)] += 1
        b = q = None
    rows = slim; ok = [r for r in rows if r['status'] == 'ok']; picked = [r for r in rows if r['status'] in S.PICKED]
    peaks = [j['peak'] for j in jobs.values() if j] + [x['peak'] for x in (run_light('r2f_plan_production'), run_light('r2f_finalize')) if x]
    bounds = {p['part']: pa['limits']['load_s'] + pa['limits']['world_s'] * len(p['members']) for p in pa['partitions']}
    ck['E_resources'] = bool(max(peaks) <= 1536 and all(r['hwm_mib'] <= 1536 and r['max_pair_s'] <= 120 and r['elapsed_s'] <= bounds[r['part']] for r in recs)
                             and not ((P / 'stops.jsonl').exists() and (P / 'stops.jsonl').read_text().strip()))
    ck['F_labels'] = bool(not any(reval) and cz['checked'] > 0 and cz['mismatch'] == 0 and cz['parity_mismatch'] == 0
                          and tch['binding_bad_arms'] == 0 and all(dl[d] >= 1 for d in days))
    ck['G_controls'] = bool(all(all(c['refused'].values()) and c['intact'] for c in ctl))
    now = {f: S.C.sha(RES / f)[:16] for f in PROD}
    summ = (LOGS / 'mutants_v2_summary.log').read_text(); cnt = re.search(r'mutation check passed: the control passes its selftest, (\d+) / (\d+) planted', summ)
    # v1.1 (logs/declaration_v1_1.txt): 113, not 114 - the status-against-result mutant is equivalent and was removed from the harness
    ck['H_suites'] = bool(now == {f: BUILD[f] for f in PROD} and S.sources_now() == pa['sources'] and cnt and cnt.group(1) == cnt.group(2) == '113'
                          and 'selftests passed' in (LOGS / 'v2_selftest.log').read_text() and 'mutant summary selftest passed' in (LOGS / 'summary_selftest.log').read_text())
    # ---- readings
    pop = []
    for d in days:
        mem = [m for p in pa['partitions'] if p['day'] == d for m in p['members']]; rr = [r for r in rows if r['day'] == d]
        for k_, v_ in (('worlds_planned', len(mem)), ('worlds_realized', len({(r['m'], r['t_ms']) for r in rr})), ('markets', len({m['m'] for m in mem})),
                       ('utc_hours', len({r['hour'] for r in rr})), ('originals', len({(r['m'], r['t_ms'], r['side']) for r in rr})), ('picks', len(rr)),
                       ('pairs_run', sum(1 for r in rr if r['status'] in S.PICKED)), ('ok', sum(1 for r in rr if r['status'] == 'ok'))):
            pop.append([d, k_, v_])
        for (sd, hd, src_, st_), n in sorted(collections.Counter((r['side'], r['head'], r.get('source') or '-', r['status']) for r in rr).items()):
            pop.append([d, f'side {sd} {hd} {src_} {st_}', n])
    wcsv('population.csv', ['day', 'item', 'n'], pop)
    ys = []; orig = lambda r: (r['day'], r['m'], r['t_ms'], r['side']); world = lambda r: (r['day'], r['m'], r['t_ms'])
    for d in days + ['both']:
        for h in ('post', 'keep', 'both'):
            s_ = [r for r in ok if (d == 'both' or r['day'] == d) and (h == 'both' or r['head'] == h)]; y = np.array([r['Y'] for r in s_], float)
            nz = [r for r in s_ if r['Y'] != 0]; ab = np.sort(np.abs(y))[::-1]
            ys.append([d, h, len(y), int((y == 0).sum()), int((y > 0).sum()), int((y < 0).sum()), len({orig(r) for r in s_ if r['Y'] > 0}),
                       len({orig(r) for r in s_ if r['Y'] < 0}), len({orig(r) for r in nz}), len({world(r) for r in nz}), len({orig(r) for r in s_}),
                       *([float(np.mean(y)), float(np.median(y)), *[float(np.percentile(y, q_)) for q_ in (1, 5, 95, 99)], float(y.min()), float(y.max()),
                          float(ab[:5].sum() / ab.sum()) if ab.sum() > 0 else 0., float(ab[:max(1, len(ab) // 100)].sum() / ab.sum()) if ab.sum() > 0 else 0.] if len(y) else [None] * 10)])
    YH = ['day', 'head', 'n_ok', 'zero', 'positive', 'negative', 'originals_positive', 'originals_negative', 'originals_nonzero_union', 'worlds_nonzero', 'originals',
          'mean_c', 'median_c', 'p01_c', 'p05_c', 'p95_c', 'p99_c', 'min_c', 'max_c', 'top5_abs_share', 'top1pct_abs_share']
    wcsv('y_summary.csv', YH, ys)
    wcsv('paths.csv', ['day', 'head', 'class', 'ok_pairs'], [[*k, v] for k, v in sorted(pth.items())])
    keys = sorted({k for c_ in arm.values() for k in c_})
    wcsv('arms.csv', ['day', 'head', 'arm'] + keys, [[*k, *[arm[k].get(x, 0) for x in keys]] for k in sorted(arm)])
    cats = dict(source=list(S.L.SOURCES), decision_ms=['0', '50-500', '550-1000', '1050-2000', '2050-4950'],
                remaining_ms=['5000', '2050-4950', '1050-2000', '550-1000', '<=500'], generation=['1', '2', '3+', 'none'], partial_queue=['known_filled>0', 'none'])
    covrows = []
    for d in days:
        for h in ('post', 'keep'):
            for dim, vals in list(cats.items()) + [('support', sorted({v for (d_, h_, dm, v) in cov if dm == 'support'})), ('utc_hour', [f'{x:02d}' for x in range(24)]), ('side', ['0', '1'])]:
                for v in vals:
                    covrows.append([d, h, dim, v, cov.get((d, h, dim, v), 0)])
    wcsv('coverage.csv', ['day', 'head', 'dimension', 'value', 'ok_rows'], covrows)
    wrows = []
    for d in days + ['both']:
        for h in ('post', 'keep'):
            pk = [r for r in picked if (d == 'both' or r['day'] == d) and r['head'] == h]; ok_ = [r for r in pk if r['status'] == 'ok']
            w_all = np.array([r['w'] for r in pk], float); w = np.array([r['w'] for r in ok_], float); y = np.array([r['Y'] for r in ok_], float)
            wrows.append([d, h, len(pk), float(w_all.sum()), len(ok_), float(w.sum()), float((w * y).sum() / w.sum()) if w.sum() > 0 else None,
                          float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else None, float(w.max() / w.sum()) if len(w) else None,
                          float(np.sort(w)[::-1][:max(1, len(w) // 10)].sum() / w.sum()) if len(w) else None, float(np.mean(y)) if len(y) else None])
    wcsv('weights.csv', ['day', 'head', 'picked', 'raw_weight_total', 'ok', 'ok_weight_total', 'weighted_mean_Y_c', 'ess', 'max_weight_share', 'top10pct_weight_share', 'raw_mean_Y_c'], wrows)
    ps_all = [x for r in recs for w in r['timing']['worlds'] for x in w['pair_s']]
    cost = [['partition_jobs', len(jobs)], ['partition_cpu_core_hours', sum(j['cpu'] for j in jobs.values() if j) / 3600.], ['partition_wall_hours_sum', sum(j['wall'] for j in jobs.values() if j) / 3600.],
            ['parity_cpu_core_hours', sum(x['cpu'] for x in (run_light(f'r2f_parity_run_{k}') for k in range(6)) if x) / 3600.], ['pairs', len(ps_all)],
            ['pair_s_mean', float(np.mean(ps_all))], ['pair_s_max', float(max(ps_all))], ['load_s_mean', float(np.mean([r['timing']['load_s'] for r in recs]))],
            ['census_s_mean', float(np.mean([w['census_s'] for r in recs for w in r['timing']['worlds'] if w['census_s'] is not None]))], ['peak_mib_jobs', max(peaks)],
            ['peak_mib_records', max(r['hwm_mib'] for r in recs)], ['ceiling_core_hours', 40.]]
    wcsv('cost.csv', ['item', 'value'], cost)
    rd.update(y_summary=[dict(zip(YH, x)) for x in ys], paths={' | '.join(k): v for k, v in sorted(pth.items())}, cost=dict(cost),
              statuses={' '.join(map(str, k)): v for k, v in sorted(collections.Counter((r['day'], r['head'], r['status']) for r in rows).items())},
              weights=[dict(zip(['day', 'head', 'picked', 'raw_weight_total', 'ok', 'ok_weight_total', 'weighted_mean_Y_c', 'ess'], x[:8])) for x in wrows],
              causal=dict(checked=cz['checked'], mismatch=cz['mismatch'], parity_checked=cz['parity_checked'], parity_mismatch=cz['parity_mismatch'],
                          binding_checked=tch['binding_checked'], binding_bad=tch['binding_bad'], zero_callback_arms=tch['zero_callback_arms'], fork_delivered_changed=dict(dl)),
              controls=ctl, parity={k: v for k, v in par.items() if k != 'rows'} | dict(rows=par['rows']), asof_records=max(r['asof'] for r in recs),
              stops=[json.loads(x)['content'] for x in (P / 'stops.jsonl').read_text().splitlines()] if (P / 'stops.jsonl').exists() else [])
    out = dict(checks=ck, all_true=all(ck.values()), readings=rd, plan_sha256=psha, accepted={x: acc[x] for x in ('dataset_sha256', 'rows', 'pairs', 'statuses', 'audit_every')}, asof=S.C.stamp())
    (HERE / 'checks.json').write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(json.dumps(dict(checks=ck, all_true=out['all_true'], cost=dict(cost))))


if __name__ == '__main__':
    main()
