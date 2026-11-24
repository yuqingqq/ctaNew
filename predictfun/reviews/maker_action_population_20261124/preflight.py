import importlib.util,json
import common as B
paths=list(B.H.glob('*.py'))+[B.H/'contract.md',B.R/'predictfun/research/pf_mt_signed_evidence_v3.py',B.OLD/'outputs/chains.csv',B.C/'post_fill/orders.csv',B.T/'outputs/post_estimates.csv.gz',B.C/'take_logic_v2/panel.parquet',B.C/'posting_opportunity/rows.parquet',B.R/'data/predict_fun/research/maker_trace_v1_20261122/econ/identities.json']
for root in ('maker_terminal_chain_20261124','maker_signed_six_20261124'):paths+=list((B.R/'data/predict_fun/research'/root).glob('batch_*.json'))
paths += [B.R/f'data/predict_fun/research/maker_trace_v1_20261122/legs_{d}.parquet' for d in ('20261024','20261025','20261026','20261027')]
posts=B.read(B.C/'post_fill/orders.csv');pairs=sorted({(r['day'],r['mid']) for r in posts})
paths += [B.V/'life'/f'{d}_{m}_fills.parquet' for d,m in pairs]
assert len(pairs)==19 and not (B.H/'manifest.json').exists()
spec=importlib.util.spec_from_file_location('cpu',B.R/'predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);task=sum(x['cpu_s'] for x in m.ledger())/3600+m.EXTRA_H
assert task+.004+.05<=2 and m.LINE_BEFORE+task+.004+.05<=5.3,'resource projection'
src=[str(p.relative_to(B.R)) for p in paths if p.suffix=='.py' or p.name=='contract.md']
B.dump(B.H/'manifest.json',dict(files={str(p.relative_to(B.R)):B.sha(p) for p in sorted(set(paths))},sources=src,task_before=task,packet_cap_coreh=.004,pairs=pairs,selection='all_cached_top_three_signed_hashes',candidate_count=0))
print(json.dumps(dict(pairs=len(pairs),source_bytes=sum(p.stat().st_size for p in paths),task_before=task,gate='input_identity_only')))
