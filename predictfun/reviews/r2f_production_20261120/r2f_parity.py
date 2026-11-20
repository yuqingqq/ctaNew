"""R2-F PILOT PARITY (review 38febc0b step 3; declared before the parity run; consumed Oct 24-25). The 288 accepted pilot pairs (attempt 4,
causal audit every 7th call) are reproduced by the production code (audit every 35th) under a new plan namespace with the same pilot run set.
Every row and pool line must be byte-identical in canonical JSON (sorted keys). Excluded from the comparison, and only these:
 - each row's 'causal' (the per-arm audit counters: the declared rate change);
 - each arm's 'decision_list' / 'fallback_list' (the new, hash-bound logs that the old rows do not carry; their decisions_sha256 IS
   compared).
Compared, with everything else: selected keys / ranks / statuses / N / q / w; the 107 float32 inputs, their hash and price; the support
status; the hidden / delivered / rebuild flags; fork fingerprint, tape id and identity; every arm's fills, fill lists, order sizes, fill /
decision / request hashes, cash / position / mark / value, open exposure, generations and teacher calls / actions / statuses / binding
counts; Y and the reconciliation flags; the pools (eligible keys, census counts, kinds). Also reported: the new audit denominators.
Usage: r2f_parity.py <accepted pilot dir> <parity dir> -> parity.json (this directory)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_label_sample as S                            # noqa: E402

HERE = Path(__file__).resolve().parent


def canon_row(r):
    r = json.loads(json.dumps(r)); r.pop('causal', None)
    for a in ((r.get('result') or {}).get('arms') or {}).values():
        a.pop('decision_list', None); a.pop('fallback_list', None)
    return S.ckey(r)


def key(r):
    return (r['day'], r['m'], r['t_ms'], r['side'], r['head'], r['pick'])


def main(old, new):
    old, new = Path(old), Path(new); po, _ = S.load_plan(old); pn, _ = S.load_plan(new)
    C = S.C; out = dict(plans_same_sample=po['sample'] == pn['sample'], plans_same_partitions=po['partitions'] == pn['partitions'],
                        audit_every=dict(old=po.get('audit_every', 7), new=pn['audit_every']))
    rows_o, rows_n, pools_o, pools_n, recs = {}, {}, {}, {}, []
    for p in po['partitions']:
        for d_, rr, pp in ((old, rows_o, pools_o), (new, rows_n, pools_n)):
            rp, fp, qp = S.part_paths(d_, p['part'])
            for r in S.gz_rows(fp.read_bytes()):
                rr[key(r)] = r
            for q in S.gz_rows(qp.read_bytes()):
                pp[(q['day'], q['m'], q['t_ms'], q['side'], q['head'])] = S.ckey(q)
        recs.append(json.loads(S.part_paths(new, p['part'])[0].read_text()))
    diff = sorted(str(k) for k in set(rows_o) | set(rows_n) if k not in rows_o or k not in rows_n or canon_row(rows_o[k]) != canon_row(rows_n[k]))
    pdiff = sorted(str(k) for k in set(pools_o) | set(pools_n) if pools_o.get(k) != pools_n.get(k))
    au = [(sum(c['checked'] for c in r['causal']), sum(c['checked'] for c in rows_o[k]['causal'])) for k, r in rows_n.items() if r['status'] == 'ok' and k in rows_o]
    out.update(rows=dict(old=len(rows_o), new=len(rows_n), differing=len(diff), first_differing=diff[:5]), pools=dict(old=len(pools_o), new=len(pools_n), differing=len(pdiff)),
               audit_checks=dict(new=sum(a for a, _ in au), old=sum(b for _, b in au)),
               new_cpu_s=sum(r['elapsed_s'] for r in recs), new_pair_s_mean=(sum(sum(w['pair_s']) for r in recs for w in r['timing']['worlds']) /
                                                                              max(1, sum(r['timing']['pairs'] for r in recs))),
               new_hwm_mib=max(r['hwm_mib'] for r in recs), asof=C.stamp())
    out['parity'] = bool(out['plans_same_sample'] and out['plans_same_partitions'] and len(rows_o) == len(rows_n) == 288 and not diff and not pdiff and len(pools_o) == len(pools_n))
    (HERE / 'parity.json').write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'} | dict(differing_rows=out['rows']['differing'])))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
