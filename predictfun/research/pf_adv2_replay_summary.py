"""THE R3 REPLAY SUMMARY v2: SUPERSEDES pf_adv2_replay.summarize (frozen in freeze 2, c8655f75) FOR replay.json (P-2026-005 R3; 2026-11-21,
session aea5bfb9). The frozen summarize wrote episodes / reached / churn / exposure.csv and then failed writing replay.json (2026-11-21
05:41Z, logs/r3_replay_summary.log): its reached status counts come from Series.size().astype(int).reset_index(), whose count column is
named 0, and json.dumps(sort_keys=True) cannot order that int key among the str keys. Its selftest never ran summarize.
v2 is the frozen function with that one column named 'n'; nothing else changes. It reads the same parts (each part's bytes against its
record's sha256) and rebuilds the same frames by the same code. It refuses unless each of the four CSVs it would write is byte-identical to
the one the frozen function wrote, and only then writes replay.json: the episodes, the reached pairs and their records stay the frozen
run_part's, and the frozen CSVs stay as written.
Usage (run_light): pf_adv2_replay_summary.py selftest | summarize"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2_replay as AR                              # noqa: E402  frozen (freeze 2): population, paths, the policies

C, pd, LS, A = AR.C, AR.pd, AR.LS, AR.A
VERSION = 'adv2_replay_summary_v2'
CSVS = ('episodes', 'reached', 'churn', 'exposure')


def frames(out, n_parts):
    """The frozen summarize's frames from the parts (each part's bytes against its record) -> (E, R, statuses, churn, exposure)."""
    out = Path(out); ep, rc, st, chn, exq = [], [], {}, [], []
    for k in range(n_parts):
        rec = json.loads((out / 'parts' / f'part_{k:03d}.json').read_text()); b = (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'], f'part {k}: bytes differ from its record')
        for r in LS.gz_rows(b):
            st[r['status']] = st.get(r['status'], 0) + 1
            if r['status'] != 'ok':
                continue
            for name, e in r['episodes'].items():
                ep.append(dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], hour=r['hour'], policy=name, **{k_: v_ for k_, v_ in e.items() if not isinstance(v_, (dict, list))}))
                chn += [dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], policy=name, **c_) for c_ in e.get('churn', [])]
                exq += [dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], policy=name, key=k_, n=v_) for k_, v_ in (e.get('exposure') or {}).items()]
            for x in r['reached']:
                rc.append(dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], side=x['key']['side'] if 'key' in x else None, head=x['key']['head'] if 'key' in x else None,
                               status=x['status'], eligible=x.get('eligible'), Y=x.get('Y'), A=x.get('A'), action=x.get('action'),
                               **{f'pred_{c}': (x.get('pred') or {}).get(c) for c in A.CANDS}))
    return pd.DataFrame(ep), pd.DataFrame(rc), st, pd.DataFrame(chn), pd.DataFrame(exq)


def summary(E, R, st):
    """The frozen summarize's replay.json content (per day / policy totals, the paired differences against PI0 and ENTRY_PA, the reached
    status counts), the count column named 'n'."""
    tot = E.groupby(['day', 'policy']).agg(worlds=('value', 'size'), value=('value', 'sum'), value_bid=('value_bid', 'sum'), value_ask=('value_ask', 'sum'),
                                          fills=('fills', 'sum'), filled_shares=('filled_shares', 'sum'), generations=('generations', 'sum'),
                                          loss=('loss', 'sum'), severe=('severe', 'sum'), ledger_ok=('ledger_ok', 'all'), binding_bad=('binding_bad', 'sum')).reset_index()
    piv = E.pivot_table(index=['day', 'm', 't_ms'], columns='policy', values='value')
    diffs = [dict(day=d, policy=p, versus=b, n=int(g[[p, b]].dropna().shape[0]), diff_sum=float((g[p] - g[b]).sum()), positive=int(((g[p] - g[b]) > 0).sum()),
                  negative=int(((g[p] - g[b]) < 0).sum())) for d, g in piv.groupby(level=0) for p in A.POLICIES for b in ('PI0', 'ENTRY_PA') if p != b and p in g and b in g]
    reached = R.groupby(['day', 'head', 'status']).size().astype(int).rename('n').reset_index().to_dict('records') if len(R) else []
    return dict(version=VERSION, statuses=st, totals=tot.to_dict('records'), diffs=diffs, reached=reached, worlds=int(len(piv)))


def summarize(out=AR.OUT, n_parts=None):
    """The v2 summary (header) -> replay.json, after the four CSVs are proved byte-identical to the frozen summarize's."""
    out = Path(out)
    if n_parts is None:
        n_parts = len(AR.population(json.loads((AR.AE.EVAL_DIR / 'plan.json').read_text())))
    E, R, st, Ch, Ex = frames(out, n_parts)
    for name, df in zip(CSVS, (E, R, Ch, Ex)):
        C.require(df.to_csv(index=False).encode() == (out / f'{name}.csv').read_bytes(), f"{name}.csv differs from the frozen summarize's")
    js = dict(summary(E, R, st), supersedes='pf_adv2_replay.summarize (freeze 2 c8655f75): replay.json only', csv_sha256={n: C.sha(out / f'{n}.csv') for n in CSVS},
              source_sha256=C.sha(Path(__file__)), asof=C.stamp())
    (out / 'replay.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(statuses=st, worlds=js['worlds'], diffs=len(js['diffs']), reached=len(js['reached'])), indent=1))


def selftest():
    import pf_adv2_replay_summary_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['summarize']:
        summarize()
    else:
        sys.exit(__doc__)
