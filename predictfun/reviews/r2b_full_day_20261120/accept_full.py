"""The full Oct 23 teacher-data acceptance (review e6b0efa4's six output checks), declared before the full-day outputs exist. Every part is read
alone, checked and released; the full day is never concatenated. Each check is a computed predicate and the verdict is ALL TRUE only when every
one holds.
(1) Population and identity: the keep / pull orders (receipt-verified) hold 30,466 originals, 15,233 per side. The plan selects all of them
    with their statuses. Its 477 partitions of at most 64 equal an independent recomputation (the (market, original) order, chunks of 64),
    disjoint and complete. Every record is accepted against the plan hash, its members and the actual parquet hash. records.sha256 is written
    and re-verified. The finalizer's preflight is equal. Sources, settings, schema, costs and the audit version are bound in the plan.
(2) Streaming reconciliation, per part, recomputed from its rows: row, status, sampled-ok and refusal counts; canonical identities unique;
    one status row OR episode rows per original, never both; the represented originals = the record's members; weight groups summing to
    one over sampled ok rows, zero elsewhere; the coarse support cells recomputed independently = the record's contribution; the record's
    full contribution = support_part of its rows. Totals = the census readings; the merged contributions = the census support JSON.
(3) Labels and clocks: per-fill reconciliation counted on every sampled ok row with zero failures; q >= 0 and finite targets; qm = value -
    s (ref - p) q for q > 0 and qm = value otherwise; severe = max(-value - SEVERE, 0); decision_abs = t0 + decision; end_abs = t0 + END;
    t_to_end = END - decision; o_horizon = (END - decision) / 1000 (float32); effective = activation (POST) or decision + 37 ms (retain).
    Statuses by source / head / side are reported.
(4) Inputs and audit: the 107 ordered inputs, float32, in every part. An independent r2_full_causal_v1 selection on each part's rows = the
    record's audit keys. Every check carries its seeds, draws and firing flag. Zero failures in every component and zero not-evaluable.
    Counts and retries reconcile to the census.
(5) Support assessment (reported; data acceptance does not require a cell to pass): the 28 coarse cells under the unchanged 200 / 20 floors,
    the categories (partial fills, queue states, pending and other-side states), cross-cells, per side / head ranges, delayed entry and later
    generations; the supported teacher states and those that need the absent-WAIT / live-KEEP fallback.
(6) Run disposition: every record PASS with its elapsed time and high-water; no breach flag, and independently elapsed <= 900 s and
    high-water <= 1,536 MiB. Every shard status PARTS_DONE with no latch. Every declared run log of the build present, and every log in logs/
    except the acceptance's own (accept*.log) rc 0 with its peak <= 1,536 MiB, by accept_logs.disposition, whose falsifier cases must all
    hold in the same run. The shadow mutation suite of c8d8fc8d archived as passed.
v2, 2026-11-20 10:03 UTC (v1 = accept_full_v1.py, kept): only check (6)'s log predicate changed. v1 globbed every log in logs/, including
its own: the production run redirected into logs/, so it read its own unfinished log, which had no footer, and (6) failed on the instrument,
not the build. The dry run had written its log elsewhere.
Usage: accept_full.py [tag]   (default _full); writes accept<tag>.json and the CSV copies beside this script."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2                                  # noqa: E402
import accept_logs as AL                                   # noqa: E402  (beside this script)

TAG = sys.argv[1] if len(sys.argv) > 1 else '_full'; DAY = '20261023'; HERE = Path(__file__).resolve().parent; OUT = Path(R2.OUT)
d = OUT / f'census_{DAY}{TAG}'; plan = json.loads((d / 'plan.json').read_text()); psha = R2.C.sha(d / 'plan.json')
J = json.loads((OUT / f'census_{DAY}{TAG}.json').read_text()); Rd = J['readings']; SUP = json.loads((OUT / f'support_{DAY}{TAG}.json').read_text())
KP, END, SEV = R2.KP, float(R2.END), float(R2.KP.SEVERE); res = {}; notes = {}
# ---- (1) population and identity
kp = Path(plan['inputs']['kp_dir']); r_ = json.loads((kp / DAY / 'receipt.json').read_text())
kp_ok = all(R2.C.sha(kp / DAY / x) == h for x, h in r_['outputs'].items()) and R2.C.sha(kp / DAY / 'receipt.json') == plan['inputs']['kp_receipt']
O = pd.read_parquet(kp / DAY / 'orders.parquet'); n_side = O.side.value_counts().to_dict()
ids = sorted([int(a), int(b)] for a, b in zip(O.side, O.oid))
S_ = O.sort_values(['m', 'oid'], kind='stable').reset_index(drop=True)
parts_ind = [sorted([int(a), int(b)] for a, b in zip(S_.side.iloc[i:i + 64], S_.oid.iloc[i:i + 64])) for i in range(0, len(S_), 64)]
pm = [p['members'] for p in plan['partitions']]; flat = [tuple(x) for p in pm for x in p]
pop_ok = (len(O) == 30466 and n_side == {0: 15233, 1: 15233} and plan['population']['kind'] == 'all' and plan['population']['selected_ids'] == ids
          and len(pm) == 477 and max(len(p) for p in pm) <= 64 and pm == parts_ind and len(flat) == len(set(flat)) == 30466)
recs, rhash, acc = [], [], True
for i in range(len(pm)):
    rp = d / 'parts' / f'part_{i:04d}.json'; rec = json.loads(rp.read_text()); f = d / 'parts' / rec['file']
    acc &= rec['plan_sha256'] == psha and rec['part'] == i and rec['members'] == pm[i] and R2.C.sha(f) == rec['sha256']
    recs.append(rec); rhash.append(f'{R2.C.sha(rp)}  census_{DAY}{TAG}/parts/part_{i:04d}.json')
(HERE / f'records{TAG}.sha256').write_text('\n'.join(rhash) + '\n')
re_ok = all(R2.C.sha(d / l.split('  ')[1].split('/', 1)[1]) == l.split('  ')[0] for l in (HERE / f'records{TAG}.sha256').read_text().splitlines())
pre = json.loads((HERE / f'preflight{TAG}.json').read_text())
bind = dict(sample=plan['causal']['sample'], on=plan['causal']['on'], costs=plan['costs'], schema=len(plan['schema']['inputs']) + len(plan['schema']['ctx']),
            dtype=plan['schema']['dtype'], settings=plan['engine_settings'] == R2.BE.SETTINGS)
res['1_population_identity'] = bool(kp_ok and pop_ok and acc and re_ok and pre['equal'] and pre['plan_sha256'] == psha and bind['sample'] == 'r2_full_causal_v1'
                                    and bind['on'] and bind['schema'] == 107 and bind['dtype'] == 'float32' and bind['settings']
                                    and plan['costs']['fee_bps'] == 0. and plan['costs']['rebate_bps'] == 0.)
print(f"(1) population: {len(O)} originals {n_side}, 477 partitions = independent: {pm == parts_ind}; records accepted: {acc}; records.sha256 re-verified: "
      f"{re_ok}; preflight equal: {pre['equal']}; bound {bind['sample']}, 107 float32, zero costs: {res['1_population_identity']}")
# ---- (2)-(4) one part at a time
SK = R2.SORT_KEY; COLS = R2.INPUTS + R2.CTX; tot = dict(rows=0, ok=0, groups=0, groups1=0); stc, refs, side_tab = {}, {}, {}
p2 = p3 = p4 = True; why = {}
def key(r):
    off = None if not np.isfinite(float(r.offset_ms)) else int(round(float(r.offset_ms)))
    return json.dumps(['r2_full_causal_v1', str(r.day), int(r.instrument_id), int(r.original_episode_id), int(r.side), str(r.source), str(r.head), off,
                       int(round(float(r.generation))), int(round(float(r.decision_ms)))], separators=(',', ':'))
hb = pd.IntervalIndex.from_breaks(R2.HORIZON_BINS, closed='right')
cells_tot = {}; aud_n = ret_n = 0; comp_tot = {}
for i, rec in enumerate(recs):
    D = pd.read_parquet(d / 'parts' / rec['file']); ok = (D.status == 'ok') & D.sampled.astype(bool); K = D[ok]
    # (2)
    st = {f'{a} {b} {c}': int(v) for (a, b, c), v in D.groupby(['source', 'head', 'status']).size().items()}
    ks = list(zip(D.side.astype(int), D.original_episode_id.astype(int))); isst = (D.source == 'original').values
    n_st, n_ep = {}, {}
    for k_, x_ in zip(ks, isst):
        (n_st if x_ else n_ep)[k_] = (n_st if x_ else n_ep).get(k_, 0) + 1
    mem = set(map(tuple, rec['members']))
    mult = set(n_st) | set(n_ep) == mem and all((n_st.get(k, 0) == 1 and k not in n_ep) or (k not in n_st and n_ep.get(k, 0) > 0) for k in mem)
    gw = K.groupby(['original_episode_id', 'side', 'head']).weight.sum(); w_ok = bool(np.allclose(gw.values, 1., rtol=0, atol=1e-12) and (D.weight[~ok] == 0).all())
    rf = {s_: int((g_.refused == True).sum()) for s_, g_ in K[K['head'] == 'post'].groupby('source')}  # noqa: E712
    ind = K.assign(hbin=pd.cut(END - K.decision_ms.astype(float), R2.HORIZON_BINS, right=True).astype(str),
                   sh=np.where(K.side == 0, 'bid', 'ask') + ' ' + K['head'].astype(str), nz=K.q > 1e-12)
    cells_i = {(r.sh, r.hbin): (int(r.rows), int(r.originals), int(r.nonzero_originals)) for r in ind.groupby(['sh', 'hbin']).apply(
        lambda g_: pd.Series(dict(rows=len(g_), originals=g_.original_episode_id.nunique(), nonzero_originals=g_[g_.nz].original_episode_id.nunique()))).reset_index().itertuples()}
    cells_r = {(c['sh'], c['hbin']): (c['rows'], c['originals'], c['nonzero_originals']) for c in (rec['support'] or {}).get('cells', [])}
    sup_same = R2.support_part(D) == rec['support']
    a2 = (len(D) == rec['rows'] and st == rec['status_counts'] and int(ok.sum()) == rec['sampled_ok'] and not D.duplicated(subset=[c for c in SK if c in D]).any()
          and mult and w_ok and len(gw) == rec['weight_groups'] == rec['weight_groups_one'] and rf == {k_: v for k_, v in rec['refusals'].items() if k_ in rf}
          and sum(rec['refusals'].values()) == sum(rf.values()) and cells_i == cells_r and sup_same)
    # (3)
    sg = np.where(K.side == 0, 1., -1.); q, v = K.q.values.astype(float), K.value.values.astype(float)
    qm_want = np.where(q > 0, v - sg * (K.pf_mid_ref.values - K.price.values) * q, v)
    post = (K['head'] == 'post').values
    a3 = (rec['recon']['checked'] == rec['sampled_ok'] and rec['recon']['failed'] == 0 and (q >= 0).all() and np.isfinite(np.c_[q, v, K.qm.values, K.severe.values]).all()
          and np.allclose(K.qm.values, qm_want, rtol=0, atol=1e-9) and np.allclose(K.severe.values, np.maximum(-v - SEV, 0.), rtol=0, atol=1e-9)
          and np.allclose(K.decision_abs_ms - K.t0 - K.decision_ms, 0., atol=1e-6) and np.allclose(K.end_abs_ms - K.t0 - END, 0., atol=1e-6)
          and np.allclose(K.t_to_end.astype(float), END - K.decision_ms, atol=1e-3) and np.allclose(K.o_horizon.astype(float) * 1000., END - K.decision_ms, atol=1.)
          and np.allclose(K.effective_ms.values[post], K.activation_ms.values[post], atol=1e-9)
          and np.allclose(K.effective_ms.values[~post], K.decision_ms.values[~post] + float(KP.CANCEL), atol=1e-9))
    # (4)
    sel = []
    for _, g in K.groupby(['original_episode_id', 'side', 'source', 'head']):
        kk = sorted((hashlib.sha256(x.encode('utf-8')).hexdigest(), x) for x in (key(r) for r in g.itertuples()))
        sel.extend(x for _, x in kk[:min(len(kk), max(1, math.ceil(len(kk) / 64)))])
    au = rec['causal'].get('audit', []); cz = rec['causal']
    a4 = (all(str(D[c].dtype) == 'float32' for c in COLS) and sorted(sel) == sorted(x['key'] for x in au) and len(au) == cz.get('checked')
          and all(x.get('draws') and 'seed_market' in x and 'seed_hidden' in x and x.get('fires') is True and x.get('rebuild') and x.get('perturbed') and x.get('visibility')
                  for x in au) and cz.get('failed') == 0 and all(int(cz.get(k, 0)) == 0 for k in ('stored_vs_rebuild', 'stored_vs_perturbed', 'stored_vs_visibility', 'firing', 'not_evaluable')))
    p2 &= a2; p3 &= a3; p4 &= a4
    if not (a2 and a3 and a4):
        why.setdefault(i, dict(reconciliation=bool(a2), labels_clocks=bool(a3), audit=bool(a4)))
    tot['rows'] += len(D); tot['ok'] += int(ok.sum()); tot['groups'] += len(gw); tot['groups1'] += int(np.isclose(gw.values, 1., rtol=0, atol=1e-12).sum())
    for k_, v_ in st.items():
        stc[k_] = stc.get(k_, 0) + v_
    for k_, v_ in rec['refusals'].items():
        refs[k_] = refs.get(k_, 0) + int(v_)
    for (a_, b_, c_, s_), v_ in D.groupby(['source', 'head', 'status', 'side']).size().items():
        side_tab[(a_, b_, c_, int(s_))] = side_tab.get((a_, b_, c_, int(s_)), 0) + int(v_)
    for k_, v_ in cells_r.items():
        cells_tot[k_] = tuple(x + y for x, y in zip(cells_tot.get(k_, (0, 0, 0)), v_))
    aud_n += len(au); ret_n += int(cz.get('retried', 0))
    for k_ in ('stored_vs_rebuild', 'stored_vs_perturbed', 'stored_vs_visibility', 'firing', 'not_evaluable'):
        comp_tot[k_] = comp_tot.get(k_, 0) + int(cz.get(k_, 0))
    D = K = ind = None
sup_cells = {(k.rsplit(' (', 1)[0], '(' + k.rsplit(' (', 1)[1]): (v['rows'], v['episodes'], v['nonzero_episodes']) for k, v in SUP['cells'].items()}
merged_ok = all(sup_cells.get(k_, (0, 0, 0)) == v_ for k_, v_ in cells_tot.items()) and R2.support_merge([r['support'] for r in recs]) == SUP
tot_ok = (tot['rows'] == Rd['census_rows'] and tot['ok'] == Rd['sampled_ok_rows'] and stc == Rd['by_source_head_status'] and refs == Rd['refusals_by_source']
          and Rd['weights'] == dict(groups=tot['groups'], groups_summing_to_one=tot['groups1']) and tot['groups'] == tot['groups1'])
res['2_streaming_reconciliation'] = bool(p2 and tot_ok and merged_ok)
print(f"(2) per part: {p2}; totals = census ({tot['rows']} rows, {tot['ok']} sampled ok, {tot['groups']} weight groups): {tot_ok}; merged support = census: {merged_ok}: "
      f"{res['2_streaming_reconciliation']}")
res['3_labels_clocks'] = bool(p3 and Rd['labels']['verdict'] == 'PASS' and Rd['labels']['checked'] == tot['ok'] and Rd['labels']['failed'] == 0)
print(f"(3) labels / clocks per part: {p3}; census labels {Rd['labels']}: {res['3_labels_clocks']}")
cz_ = Rd['causality']
res['4_inputs_audit'] = bool(p4 and cz_['verdict'] == 'PASS' and cz_['checked'] == aud_n == cz_['audited'] and cz_['retried'] == ret_n
                            and all(v == 0 for v in cz_['components'].values()) and comp_tot == {k: 0 for k in comp_tot})
print(f"(4) inputs / audit per part: {p4}; {aud_n} audited checks (census {cz_['checked']}), {ret_n} retried, components {cz_['components']}: {res['4_inputs_audit']}")
if why:
    print(f'    failing parts (first 10): {dict(list(why.items())[:10])}')
pd.DataFrame([dict(source=a, head=b, status=c, side=s, rows=n) for (a, b, c, s), n in sorted(side_tab.items())]).to_csv(HERE / f'statuses{TAG}.csv', index=False)
# ---- (5) support assessment (reported)
cl = SUP['cells']; nsup = sum(v['support'] == 'SUPPORTED' for v in cl.values())
pd.DataFrame([dict(cell=k, **v) for k, v in cl.items()]).to_csv(HERE / f'support_cells{TAG}.csv', index=False)
for nm in ('categories', 'cross_cells', 'coverage'):
    pd.DataFrame(SUP[nm]).to_csv(HERE / f'support_{nm}{TAG}.csv', index=False)
pd.DataFrame([dict(side_head=h, field=c, lo=v[0], hi=v[1]) for h, r in SUP['ranges'].items() for c, v in r.items()]).to_csv(HERE / f'support_ranges{TAG}.csv', index=False)
pd.DataFrame([dict(side_head=h, dim=dm, value=x, originals=n) for h, r in SUP['presence'].items() for dm, xs in r.items() for x, n in xs.items()]).to_csv(
    HERE / f'support_presence{TAG}.csv', index=False)
res['5_support_reported'] = bool(len(cl) == 28 and all((v['fallback'] is None) == (v['support'] == 'SUPPORTED') for v in cl.values())
                                 and all(v['support'] == ('SUPPORTED' if (v['episodes'] >= 200 and v['nonzero_episodes'] >= 20) else 'INSUFFICIENT_TRAINING_SUPPORT')
                                         for v in cl.values()))
notes['supported_cells'] = sorted(k for k, v in cl.items() if v['support'] == 'SUPPORTED')
notes['fallback_cells'] = {k: v['fallback'] for k, v in cl.items() if v['support'] != 'SUPPORTED'}; notes['unsupported_state_category'] = SUP['unsupported_state_category']
print(f"(5) support: {nsup} of 28 cells SUPPORTED under 200 / 20; fallbacks {notes['fallback_cells']}; floors applied as declared: {res['5_support_reported']}")
# ---- (6) run disposition
el = [r['elapsed_s'] for r in recs]; hw = [r['hwm_mib'] for r in recs]
stt = {p.name: json.loads(p.read_text()) for p in sorted(d.glob('run_status*.json'))}
BUILD = {'_full': 'r2full', '_shard4': 'shard4'}[TAG]
DECL = [f'{BUILD}_{x}.log' for x in ('plan', 'run_0', 'run_1', 'run_2', 'run_3', 'preflight', 'finalize')]
if TAG == '_full':                                         # the launch chain's gates before the full day
    DECL += ['runner_selftest.log', 'runner_mutants.log', 'shard4_check.log'] + [f'shard4_{x}.log' for x in ('plan', 'run_0', 'run_1', 'run_2', 'run_3', 'preflight', 'finalize')]
fals = AL.cases(); fals_ok = all(w == g for _, w, g in fals); LD = AL.disposition(HERE / 'logs', DECL); logs = LD['logs']
logs_ok = bool(fals_ok and LD['ok'])
print(f"    logs: {len(logs)} checked, {len(DECL)} declared, missing {LD['missing']}, failing {LD['bad']}, excluded (own) {LD['excluded']}; "
      f"falsifier cases {sum(w == g for _, w, g in fals)} / {len(fals)} as expected")
sh_ok = all(v['state'] == 'PARTS_DONE' and v.get('latch') is None for k, v in stt.items() if k.startswith('run_status_')) and sum(k.startswith('run_status_') for k in stt) == 4
suite = (HERE.parent / 'r2a_launch_gate_20261120' / 'logs' / 'shadow_mutants_final.log').read_text()
res['6_run_disposition'] = bool(all(r['status'] == 'PASS' for r in recs) and not any(any(r['breach'].values()) for r in recs) and max(el) <= 900. and max(hw) <= 1536.
                                and sh_ok and logs_ok and 'r2 shadow mutation check passed' in suite and J['state'] == 'COMPLETE')
print(f"(6) records PASS, no breach, elapsed max {max(el):.0f} s, high-water max {max(hw):.0f} MiB; shards {[(k, v['state']) for k, v in stt.items()]}; run logs "
      f"rc 0 / peak <= 1,536: {logs_ok}; the shadow suite archived as passed: {'r2 shadow mutation check passed' in suite}; state {J['state']}: {res['6_run_disposition']}")
pd.DataFrame([dict(part=r['part'], rows=r['rows'], sampled_ok=r['sampled_ok'], elapsed_s=r['elapsed_s'], hwm_mib=r['hwm_mib'], part_peak_rss_mib=r['part_peak_rss_mib'],
                   audited=len(r['causal'].get('audit', [])), retried=r['causal'].get('retried', 0), status=r['status']) for r in recs]).to_csv(HERE / f'parts{TAG}.csv', index=False)
(HERE / f'accept{TAG}.json').write_text(json.dumps(dict(asof=R2.C.stamp(), census=str(d), plan_sha256=psha, results=res, notes=notes, failing_parts=why,
                                                       logs=logs, logs_declared=DECL, logs_missing=LD['missing'], logs_failing=LD['bad'],
                                                       logs_excluded=LD['excluded'], log_falsifiers=fals), indent=1, default=str))
print(f'FULL-DAY ACCEPTANCE: {res}; ALL TRUE: {all(res.values())}')
sys.exit(0 if all(res.values()) else 1)
