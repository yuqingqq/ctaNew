"""The sharded runner on real data, declared before its run (USER 2026-11-20: "Yea 4 cores to make it faster"): the frozen 256 originals of
consumed Oct 23 (the gate2 population) run as 4 parallel shards (one per research slot, each loading only its own markets) and finalized once
must reproduce gate2A exactly. Each a computed predicate:
(1) the sharded run COMPLETE, every shard PARTS_DONE with exactly its parts (i % 4 == k);
(2) the plan equal to gate2A's in every field but the runner's own source hash (the sharded runner is a new source; the plan binds content
    and code, never the output location);
(3) every part byte-identical to gate2A's (sha256), every completion record identical except elapsed / memory / time stamps and the plan hash;
(4) the same support JSON and the same census readings except timings and memory;
(5) every shard's measured peak (run_light) <= 1,536 MiB.
Prints one line per predicate and ALL TRUE."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2                                  # noqa: E402

D, HERE, DAY = Path(R2.OUT), Path(__file__).resolve().parent, '20261023'
G, H = D / f'census_{DAY}_gate2A', D / f'census_{DAY}_shard4'
JG, JH = json.loads((D / f'census_{DAY}_gate2A.json').read_text()), json.loads((D / f'census_{DAY}_shard4.json').read_text())
T = ('elapsed_s', 'part_peak_rss_mib', 'hwm_mib', 'asof', 'breach')
recs = lambda d: [json.loads(p.read_text()) for p in sorted((d / 'parts').glob('part_*.json'))]
RG, RH = recs(G), recs(H); res = {}
st = {k: json.loads((H / f'run_status_{k}of4.json').read_text()) for k in range(4)}
res['states'] = JH['state'] == 'COMPLETE' and all(st[k]['state'] == 'PARTS_DONE' and st[k]['done'] == [i for i in range(4) if i % 4 == k] for k in range(4))
print(f"(1) sharded COMPLETE: {JH['state']}; shards {[(k, st[k]['state'], st[k]['done']) for k in range(4)]}: {res['states']}")
PG, PH = json.loads((G / 'plan.json').read_text()), json.loads((H / 'plan.json').read_text())
diff = sorted(k for k in set(PG) | set(PH) if PG.get(k) != PH.get(k)); sdiff = sorted(k for k in set(PG['sources']) | set(PH['sources']) if PG['sources'].get(k) != PH['sources'].get(k))
res['plan'] = diff == ['sources'] and sdiff == ['pf_r2_runner.py']
print(f"(2) the plans differ only in {diff} / {sdiff}: {res['plan']}")
strip = lambda r: {k: v for k, v in r.items() if k not in T + ('plan_sha256',)}
res['parts_records'] = len(RH) == len(RG) == 4 and [r['sha256'] for r in RH] == [r['sha256'] for r in RG] and [strip(r) for r in RH] == [strip(r) for r in RG]
print(f"(3) parts byte-identical and records identical except timing: {res['parts_records']}")
sr = lambda R: {k: v for k, v in R.items() if k not in ('partitions', 'peak_rss_mib', 'finalize_hwm_mib')}
res['support_readings'] = (json.loads((D / f'support_{DAY}_shard4.json').read_text()) == json.loads((D / f'support_{DAY}_gate2A.json').read_text())
                           and sr(JH['readings']) == sr(JG['readings']))
print(f"(4) the same support and readings: {res['support_readings']}")
pk = {k: [int(x) for x in re.findall(r'peak_rss_mib=(\d+)', (HERE / 'logs' / f'shard4_run_{k}.log').read_text())] for k in range(4)}
res['memory'] = all(len(v) == 1 and v[0] <= 1536 for v in pk.values())
print(f'(5) shard peaks {pk} MiB (<= 1,536): {res["memory"]}')
print(f'SHARD CHECK: {res}; ALL TRUE: {all(res.values())}')
sys.exit(0 if all(res.values()) else 1)
