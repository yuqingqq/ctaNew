"""R2-E LAUNCH-GATE CHECKS of the r2_label_sample_v2 pilot (review 4393ad19). Declared before any pilot output is read (2026-11-20, session
aea5bfb9); consumed Oct 24-25. Every claim below is a computed predicate in checks.json; the readings are reported, never gated.
CHECKS
 (A) plan: A and B plans byte-identical; N_worlds 15,352 / 14,998 (the review's day metadata); 24 strata a day with n_h = min(64, N_h);
     1,536 worlds a day. The pilot is one world per stratum, its first-ranked, in partitions of <= 8; limits (600, 120, 120, 1536, 2048);
     strict UTC days; 107 features.
 (B) runs: every A and B partition job rc 0 and promoted; the planted interruption (B part 0 after 3 worlds) printed STOPPED with rc 0 before its
     rerun.
 (C) resume identity: every part's label and pool bytes A = B.
 (D) acceptance: finalize accepted A and B; equal dataset hashes, rows, pairs and statuses; the partitions cover the 48 pilot worlds once.
 (E) resources: every job's measured peak (run_light) and every record's high-water <= 1,536 MiB; every pair <= 120 s; every partition <= its
     bound; no stop archived in A or B.
 (F) labels: every part revalidated now (the shared validator with the executing sources); 0 causal, parity and binding failures summed over
     every ok pair's arms with checks > 0; the delivered positive control changed the fork inputs on >= 1 ok row a day; at most 288 pairs.
 (G) controls (r2e_controls.py, every A partition): reversal equal on every ok pair; the stale binding caught (binding_bad > 0) on each
     partition's first ok pair; the four corruptions refused; the intact parts valid; controls peaks <= 1,536 MiB.
 (H) suites: every build file's hash equals the freeze's (logs/build_sources.txt). The v1 and v2 selftests passed on them. Both per-mutant
     summaries passed: their controls passed and every planted error was caught (35 / 35 v1, 100 / 100 v2). The runner v5 suite passed
     (30 / 30) on its unchanged sources.
READINGS (review 0ca01d1f's list; descriptive, the pilot's own populations, no intervals)
 - population: planned versus realized worlds, markets, UTC hours, originals, picks, pairs; statuses by day / side / head / source;
 - Y by day and head: zero / positive / negative counts, the distinct originals behind each, mean, median, tails (1 / 5 / 95 / 99 %), min,
   max, the top-5 |Y| share of the total |Y|;
 - both arms by day and head: value / cash / position sums, fills, filled quantity, generations (bid / ask), open exposure at END, the
   teacher's calls, actions and statuses (its fallbacks);
 - weights by day and head: raw weight totals (picked and ok), the weighted and raw mean Y, the effective sample size, the largest and the
   top-10 % weight shares;
 - timing from the records: load, census, pairs (mean / max), worlds; every attempt's archived stops;
 - statuses by day / head / status;
 - coverage of the ok rows: source x head, horizon bin, generation, partial queue (fork_known_filled > 0), support status, UTC hour;
 - Y by head (diagnostic: n, zeros, positives, negatives, sum; the pilot is not a representative sample);
 - throughput: per partition job wall time, worlds, pairs. A least-squares fit wall = a + b x pairs + c x worlds over the 13 partition runs
   (the interrupted one with its 3 worlds' pairs) projects the full sample: 96 partitions of 32 worlds and 3,072 worlds, at the pilot's
   pairs a world for each day. The projection is compared with the review's 40 core-hour planning ceiling (a reading: above it, the
   review asks for a pause at a partition boundary and a proposal). Sensitivities: the fit without a per-world term (wall = a + b x pairs),
   and a conservative bound (the slowest pilot partition's seconds a world x 3,072 worlds).
Usage: r2e_checks.py -> checks.json, population.csv, coverage.csv, y_summary.csv, arms.csv, weights.csv, throughput.csv (in this directory)"""
import collections
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_label_sample as S                            # noqa: E402

HERE = Path(__file__).resolve().parent; LOGS = HERE / 'logs'
D = Path('/home/yuqing/ctaNew/data/predict_fun/research/r2_label_sample_v2_20261120'); A, B = D / 'attempt4_pilot_A', D / 'attempt4_pilot_B'                  # attempts 1-3 kept (logs/attempt*)
BUILD = dict(ln.split() for ln in (HERE / 'logs' / 'build_sources.txt').read_text().splitlines() if ln.strip())   # file -> sha256[:16] at the freeze (v.1: was parsed hash -> file)
RL = re.compile(r'tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')


def run_light(tag):
    m = [RL.search(x) for x in (LOGS / f'{tag}.log').read_text().splitlines() if 'tag=' in x]
    m = [x for x in m if x and x.group(1) == tag]
    return dict(rc=int(m[-1].group(2)), peak=int(m[-1].group(3)), wall=float(m[-1].group(4)), cpu=float(m[-1].group(5))) if m else None


def parts_of(out, plan):
    return [(p, json.loads(S.part_paths(out, p['part'])[0].read_text()), S.part_paths(out, p['part'])[1].read_bytes(), S.part_paths(out, p['part'])[2].read_bytes())
            for p in plan['partitions']]


def hbin(ms):
    return '0' if ms == 0 else '50-500' if ms <= 500 else '550-1000' if ms <= 1000 else '1050-2000' if ms <= 2000 else '2050-5000'


def main():  # noqa: C901
    ck, rd = {}, {}
    pa, psha = S.load_plan(A); pb, _ = S.load_plan(B)
    sel = {d: pa['sample'][d]['selected'] for d in pa['days']}
    first = {d: {h: sorted((w for w in sel[d] if w['hour'] == h), key=lambda w: w['rank'])[0] for h in range(24)} for d in pa['days']}
    ck['A_plan'] = bool((A / 'plan.json').read_bytes() == (B / 'plan.json').read_bytes()
                        and {d: pa['sample'][d]['N_worlds'] for d in pa['days']} == {'20261024': 15352, '20261025': 14998}
                        and all(len(pa['sample'][d]['strata']) == 24 and all(x['n'] == min(64, x['N']) for x in pa['sample'][d]['strata']) and len(sel[d]) == 1536 for d in pa['days'])
                        and sorted(S.ckey(m) for p in pa['partitions'] for m in p['members']) == sorted(S.ckey(S.wkey(first[d][h])) for d in pa['days'] for h in range(24))
                        and all(len(p['members']) <= 8 for p in pa['partitions']) and pa['run_set'] == 'pilot' and pa['strict_day'] is True
                        and pa['limits'] == dict(load_s=600., world_s=120., pair_s=120., peak_mib=1536., guard_mib=2048.) and pa['feature_schema']['n'] == 107)
    jobs = {f'r2e_run_{x}{k}': run_light(f'r2e_run_{x}{k}') for x in 'AB' for k in range(6)}
    stop = run_light('r2e_run_B0s3'); stop_txt = (LOGS / 'r2e_run_B0s3.log').read_text()
    PA, PB = parts_of(A, pa), parts_of(B, pb)
    ck['B_runs'] = bool(all(j is not None and j['rc'] == 0 for j in jobs.values()) and all(S.part_paths(o, p['part'])[0].exists() for o in (A, B) for p in pa['partitions'])
                        and stop is not None and stop['rc'] == 0 and 'STOPPED' in stop_txt.split())
    ck['C_resume_identity'] = bool(all(a[2] == b[2] and a[3] == b[3] for a, b in zip(PA, PB)))
    acc = {x: json.loads((o / 'accepted.json').read_text()) for x, o in (('A', A), ('B', B))}
    ck['D_acceptance'] = bool(all(v['accepted'] for v in acc.values()) and all(acc['A'][k] == acc['B'][k] for k in ('dataset_sha256', 'rows', 'pairs', 'statuses')))
    recs = [r for _, r, _, _ in PA + PB]; peaks = [j['peak'] for j in list(jobs.values()) + [stop] if j]
    for t_ in ('r2e_plan_A', 'r2e_plan_B', 'r2e_finalize_A', 'r2e_finalize_B'):
        peaks.append(run_light(t_)['peak'])
    bounds = {p['part']: pa['limits']['load_s'] + pa['limits']['world_s'] * len(p['members']) for p in pa['partitions']}
    ck['E_resources'] = bool(max(peaks) <= 1536 and all(r['hwm_mib'] <= 1536 and r['max_pair_s'] <= 120 and r['elapsed_s'] <= bounds[r['part']] for r in recs)
                             and not any((o / 'stops.jsonl').exists() and (o / 'stops.jsonl').read_text().strip() for o in (A, B)))
    rows = [r for _, _, b, _ in PA for r in S.gz_rows(b)]; ok = [r for r in rows if r['status'] == 'ok']
    reval = [S.validate(pa, psha, p, S.gz_rows(b), S.gz_rows(pb_), rec, (b, pb_)) for p, rec, b, pb_ in PA]
    cz = [c for r in ok for c in r['causal']]; tch = [r['result']['arms'][n]['teacher'] for r in ok for n in S.NAMES[r['head']]]
    dl = collections.Counter(r['day'] for r in ok if r['fork']['delivered_changed'])
    n_pairs = sum(1 for r in rows if r['status'] in S.PICKED)
    ck['F_labels'] = bool(not any(reval) and sum(c['checked'] for c in cz) > 0 and sum(c['mismatch'] for c in cz) == 0 and sum(c['parity_checked'] for c in cz) > 0
                          and sum(c['parity_mismatch'] for c in cz) == 0 and all(t['binding_checked'] > 0 and t['binding_bad'] == 0 for t in tch)
                          and all(dl[d] >= 1 for d in pa['days']) and n_pairs <= 288)
    ctl = [json.loads((A / 'controls' / f"part_{p['part']:03d}.json").read_text()) for p in pa['partitions']]
    ctl_peaks = [run_light(f"r2e_controls_{p['part']}")['peak'] for p in pa['partitions']]
    ck['G_controls'] = bool(all(c['reversal']['n'] > 0 and c['reversal']['equal'] == c['reversal']['n'] and c['stale_binding'] and c['stale_binding']['binding_bad'] > 0
                                and all(c['corruptions_refused'].values()) and c['intact_valid'] and c['plan_sha256'] == psha for c in ctl) and max(ctl_peaks) <= 1536)
    RES = Path('/home/yuqing/ctaNew/predictfun/research')
    now = {f: S.C.sha(RES / f)[:16] for f in BUILD}
    summ = {k: (LOGS / f'mutants_{k}_summary.log').read_text() for k in ('v1', 'v2')}
    cnt = {k: re.search(r'mutation check passed: the control passes its selftest, (\d+) / (\d+) planted', v) for k, v in summ.items()}
    st_ok = all('selftests passed' in (LOGS / f'{k}_selftest.log').read_text() for k in ('v1', 'v2'))
    rv5 = (LOGS / 'runner_v5' / 'runner_v5_mutants.log').read_text()
    ck['H_suites'] = bool(now == BUILD and S.sources_now() == pa['sources'] and st_ok and all(cnt.values()) and int(cnt['v1'].group(1)) == int(cnt['v1'].group(2)) == 35
                          and int(cnt['v2'].group(1)) == int(cnt['v2'].group(2)) == 100 and '30 / 30 planted code errors are caught by a check' in rv5)
    # ---- readings (review 0ca01d1f): descriptive, the pilot's own populations; not a training set, no intervals
    def wcsv(name, head, rows_):
        with open(HERE / name, 'w', newline='') as f:
            w_ = csv.writer(f); w_.writerow(head); [w_.writerow(x) for x in rows_]
    picked = [r for r in rows if r['status'] in S.PICKED]; days = pa['days']
    pop = []
    for d in days:
        mem = [m for p in pa['partitions'] if p['day'] == d for m in p['members']]; rr = [r for r in rows if r['day'] == d]
        pop.append([d, 'worlds_planned', len(mem)]); pop.append([d, 'worlds_realized', len({(r['m'], r['t_ms']) for r in rr})])
        pop.append([d, 'markets', len({m['m'] for m in mem})]); pop.append([d, 'utc_hours', len({r['hour'] for r in rr})])
        pop.append([d, 'originals', len({(r['m'], r['t_ms'], r['side']) for r in rr})]); pop.append([d, 'picks', len(rr)]); pop.append([d, 'pairs_run', sum(1 for r in rr if r['status'] in S.PICKED)])
        for (sd, hd, src_, st_), n in sorted(collections.Counter((r['side'], r['head'], r.get('source') or '-', r['status']) for r in rr).items()):
            pop.append([d, f'side {sd} {hd} {src_} {st_}', n])
    wcsv('population.csv', ['day', 'item', 'n'], pop)
    cov = collections.Counter(); cats = dict(source=list(S.L.SOURCES), horizon_ms=['0', '50-500', '550-1000', '1050-2000', '2050-5000'], generation=['1', '2', '3+', 'none'],
                                             partial_queue=['known_filled>0', 'none'])
    for r in ok:
        k = r['row_key']; g_ = k['generation']
        for dim, v in (('source', r['source']), ('horizon_ms', hbin(k['decision_ms'])), ('generation', 'none' if g_ is None else ('3+' if g_ >= 3 else str(g_))),
                       ('partial_queue', 'known_filled>0' if r['result']['fork_known_filled'] > 0 else 'none'), ('support', str(r['fork']['support'])), ('utc_hour', f"{r['hour']:02d}")):
            cov[(r['day'], r['head'], dim, v)] += 1
    covrows = []
    for d in days:
        for h in ('post', 'keep'):
            for dim, vals in list(cats.items()) + [('support', sorted({v for (d_, h_, dm, v) in cov if dm == 'support'})), ('utc_hour', [f'{x:02d}' for x in range(24)])]:
                for v in vals:
                    covrows.append([d, h, dim, v, cov.get((d, h, dim, v), 0)])           # absent states as explicit zeros
    wcsv('coverage.csv', ['day', 'head', 'dimension', 'value', 'ok_rows'], covrows)
    ys = []
    for d in days + ['both']:
        for h in ('post', 'keep', 'both'):
            sel_ = [r for r in ok if (d == 'both' or r['day'] == d) and (h == 'both' or r['head'] == h)]; y = np.array([r['result']['Y'] for r in sel_], float)
            orig = lambda f: len({(r['day'], r['m'], r['t_ms'], r['side']) for r in sel_ if f(r['result']['Y'])})
            ab = np.sort(np.abs(y))[::-1]
            ys.append([d, h, len(y), int((y == 0).sum()), int((y > 0).sum()), int((y < 0).sum()), orig(lambda v: v == 0), orig(lambda v: v > 0), orig(lambda v: v < 0),
                       *([float(np.mean(y)), float(np.median(y)), *[float(np.percentile(y, q_)) for q_ in (1, 5, 95, 99)], float(y.min()), float(y.max()),
                          float(ab[:5].sum() / ab.sum()) if ab.sum() > 0 else 0.] if len(y) else [None] * 9)])
    wcsv('y_summary.csv', ['day', 'head', 'n_ok', 'zero', 'positive', 'negative', 'originals_zero', 'originals_positive', 'originals_negative', 'mean_c', 'median_c', 'p01_c', 'p05_c',
                           'p95_c', 'p99_c', 'min_c', 'max_c', 'top5_abs_share'], ys)
    arm = collections.defaultdict(lambda: collections.Counter())
    for r in ok:
        for n_ in S.NAMES[r['head']]:
            A_ = r['result']['arms'][n_]; t_ = A_['teacher'] or {}; c_ = arm[(r['day'], r['head'], n_)]
            c_['rows'] += 1; c_['value_sum'] += A_['value']; c_['cash_sum'] += A_['cash']; c_['position_sum'] += A_['position']; c_['fills'] += A_['fills']
            c_['filled_qty'] += sum(float(f[3]) for f in A_['fill_list']); c_['generations'] += A_['generations']; c_['gens_bid'] += A_['gens_bid']; c_['gens_ask'] += A_['gens_ask']
            c_['open_at_end_rows'] += int(any(float(v) > 0 for v in A_['open_at_end'].values())); c_['teacher_calls'] += t_.get('calls', 0)
            for k_, v_ in (t_.get('by_action') or {}).items():
                c_[f'action {k_}'] += v_
            for k_, v_ in (t_.get('by_status') or {}).items():
                c_[f'status {k_}'] += v_
    keys = sorted({k for c_ in arm.values() for k in c_})
    wcsv('arms.csv', ['day', 'head', 'arm'] + keys, [[*k, *[arm[k].get(x, 0) for x in keys]] for k in sorted(arm)])
    wrows = []
    for d in days + ['both']:
        for h in ('post', 'keep'):
            pk = [r for r in picked if (d == 'both' or r['day'] == d) and r['head'] == h]; ok_ = [r for r in pk if r['status'] == 'ok']
            w_all = np.array([r['w'] for r in pk], float); w = np.array([r['w'] for r in ok_], float); y = np.array([r['result']['Y'] for r in ok_], float)
            wrows.append([d, h, len(pk), float(w_all.sum()), len(ok_), float(w.sum()), float((w * y).sum() / w.sum()) if w.sum() > 0 else None,
                          float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else None, float(w.max() / w.sum()) if len(w) else None,
                          float(np.sort(w)[::-1][:max(1, len(w) // 10)].sum() / w.sum()) if len(w) else None, float(np.mean(y)) if len(y) else None])
    wcsv('weights.csv', ['day', 'head', 'picked', 'raw_weight_total', 'ok', 'ok_weight_total', 'weighted_mean_Y_c', 'ess', 'max_weight_share', 'top10pct_weight_share', 'raw_mean_Y_c'], wrows)
    rd['Y_summary'] = [dict(zip(['day', 'head', 'n_ok', 'zero', 'positive', 'negative', 'originals_zero', 'originals_positive', 'originals_negative', 'mean_c', 'median_c'], x[:11])) for x in ys]
    rd['weights'] = [dict(zip(['day', 'head', 'picked', 'raw_weight_total', 'ok', 'ok_weight_total', 'weighted_mean_Y_c', 'ess'], x[:8])) for x in wrows]
    rd['statuses'] = {' '.join(map(str, k)): v for k, v in sorted(collections.Counter((r['day'], r['head'], r['status']) for r in rows).items())}
    rd['pairs'] = n_pairs; rd['ok_pairs'] = len(ok); rd['asof_records'] = max(r['asof'] for r in recs)
    rd['causal'] = dict(checked=sum(c['checked'] for c in cz), mismatch=sum(c['mismatch'] for c in cz), parity_checked=sum(c['parity_checked'] for c in cz),
                        parity_mismatch=sum(c['parity_mismatch'] for c in cz), probe_delivered_changed=sum(c['delivered_changed'] for c in cz),
                        binding_checked=sum(t['binding_checked'] for t in tch), binding_bad=sum(t['binding_bad'] for t in tch), fork_delivered_changed=dict(dl))
    tm = [(p['day'], r['timing']) for p, r, _, _ in PA]; ps_all = [x for _, t_ in tm for w_ in t_['worlds'] for x in w_['pair_s']]
    rd['timing'] = dict(load_s=dict(mean=float(np.mean([t_['load_s'] for _, t_ in tm])), max=float(max(t_['load_s'] for _, t_ in tm))),
                        census_s=dict(mean=float(np.mean([w_['census_s'] for _, t_ in tm for w_ in t_['worlds'] if w_['census_s'] is not None])),
                                      max=float(max(w_['census_s'] for _, t_ in tm for w_ in t_['worlds'] if w_['census_s'] is not None))),
                        pair_s=dict(n=len(ps_all), mean=float(np.mean(ps_all)), max=float(max(ps_all))), world_s_mean=float(np.mean([w_['world_s'] for _, t_ in tm for w_ in t_['worlds']])))
    pw = {p['part']: sum(1 for r in S.gz_rows(b) if r['status'] in S.PICKED) for p, _, b, _ in PA}
    first3 = [S.ckey(m) for m in pa['partitions'][0]['members'][:3]]
    pw_stop = sum(1 for r in S.gz_rows(PA[0][2]) if r['status'] in S.PICKED and S.ckey(dict(day=r['day'], m=r['m'], t_ms=r['t_ms'])) in first3)
    tp = [dict(job=f'r2e_run_{x}{p["part"]}', day=p['day'], worlds=len(p['members']), pairs=pw[p['part']], **jobs[f'r2e_run_{x}{p["part"]}']) for x in 'AB' for p in pa['partitions']]
    tp.append(dict(job='r2e_run_B0s3', day=pa['partitions'][0]['day'], worlds=3, pairs=pw_stop, **stop))
    with open(HERE / 'throughput.csv', 'w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=list(tp[0])); w_.writeheader(); [w_.writerow(x) for x in tp]
    Xm = np.array([[1., x['pairs'], x['worlds']] for x in tp]); yv = np.array([x['wall'] for x in tp]); coef = np.linalg.lstsq(Xm, yv, rcond=None)[0]
    ppw = {d: sum(pw[p['part']] for p in pa['partitions'] if p['day'] == d) / sum(len(p['members']) for p in pa['partitions'] if p['day'] == d) for d in days}
    n_full = {d: len(sel[d]) for d in days}; parts_full = sum(-(-n_full[d] // 32) for d in days)
    total_s = coef[0] * parts_full + sum(coef[1] * ppw[d] * n_full[d] + coef[2] * n_full[d] for d in days)
    c2 = np.linalg.lstsq(Xm[:, :2], yv, rcond=None)[0]
    rd['throughput'] = dict(fit_wall_s=dict(per_partition=float(coef[0]), per_pair=float(coef[1]), per_world=float(coef[2])), pairs_per_world=ppw,
                            residual_max_s=float(np.max(np.abs(Xm @ coef - yv))), full_partitions=parts_full, full_worlds=sum(n_full.values()),
                            full_pairs_expected=float(sum(ppw[d] * n_full[d] for d in days)), projected_core_hours=float(total_s / 3600.), ceiling_core_hours=40.,
                            above_ceiling=bool(total_s / 3600. > 40.), pilot_core_hours=float(sum(x['cpu'] for x in tp) / 3600.),
                            sensitivity=dict(no_world_term_core_hours=float((c2[0] * parts_full + sum(c2[1] * ppw[d] * n_full[d] for d in days)) / 3600.),
                                             slowest_partition_core_hours=float(max(x['wall'] / x['worlds'] for x in tp) * sum(n_full.values()) / 3600.)))
    rd['peaks_mib'] = dict(jobs=max(peaks), controls=max(ctl_peaks), records=max(r['hwm_mib'] for r in recs))
    rd['stops'] = {str(o.name): [json.loads(x)['content'] for x in (o / 'stops.jsonl').read_text().splitlines()] if (o / 'stops.jsonl').exists() else []
                   for o in sorted(D.glob('attempt*_pilot_*'))}
    out = dict(checks=ck, all_true=all(ck.values()), readings=rd, plan_sha256=psha, build=BUILD, accepted={k: {x: v[x] for x in ('dataset_sha256', 'rows', 'pairs')} for k, v in acc.items()},
               asof=S.C.stamp())
    (HERE / 'checks.json').write_text(json.dumps(out, indent=1, sort_keys=True, default=str))
    print(json.dumps(dict(checks=ck, all_true=out['all_true'], projected_core_hours=rd['throughput']['projected_core_hours'], above_ceiling=rd['throughput']['above_ceiling'])))


if __name__ == '__main__':
    main()
