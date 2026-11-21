"""THE R4E REPLAY SUMMARY v2: SUPERSEDES pf_adv2_entry_replay.summarize (frozen in 602fdfc5) FOR replay.json (P-2026-005 R4E; 2026-11-21,
session aea5bfb9). The frozen summarize wrote episodes / reached / churn / exposure.csv and then failed (logs/r4e_replay_summary.log): its
totals aggregate a top-level binding_bad that this round's episodes keep inside their traces (trace.teacher.binding_bad), so the column does
not exist. Its selftest never ran summarize (the same class of defect as R3's, feedback 2026-11-21).
v2 builds the same frames with the frozen pf_adv2_entry_replay.frames (identity, membership, sources and bytes checked per part). It refuses
unless each of the four CSVs it would write is byte-identical to the one the frozen function wrote. It then writes replay.json with the
frozen totals less that one column, and adds the per-day / policy binding failures summed from the traces. The episodes, reached pairs and
part records stay the frozen run_part's.
Usage (run_light): pf_adv2_entry_replay_summary.py selftest | summarize"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2_entry_replay as NR                        # noqa: E402  frozen (602fdfc5): frames, the paths, the policies

C, pd, LS, AR, AE, N = NR.C, NR.pd, NR.LS, NR.AR, NR.AE, NR.N
VERSION = 'adv2_entry_smooth_replay_summary_v2'
CSVS = ('episodes', 'reached', 'churn', 'exposure')


def bindings(out, parts):
    """Per (day, policy) the binding failures summed from every episode trace (bytes checked against each part's record)."""
    out = Path(out); acc = {}
    for k, _ in enumerate(parts):
        rec = json.loads((out / 'parts' / f'part_{k:03d}.json').read_text()); b = (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'], f'part {k}: bytes differ from its record')
        for r in LS.gz_rows(b):
            if r['status'] == 'ok':
                for name, e in r['episodes'].items():
                    acc[(r['day'], name)] = acc.get((r['day'], name), 0) + int(((e.get('trace') or {}).get('teacher') or {}).get('binding_bad', 1))
    return acc


def summary(E, R, st, bind):
    tot = E.groupby(['day', 'policy']).agg(worlds=('value', 'size'), value=('value', 'sum'), value_bid=('value_bid', 'sum'), value_ask=('value_ask', 'sum'),
                                          fills=('fills', 'sum'), filled_shares=('filled_shares', 'sum'), generations=('generations', 'sum'),
                                          loss=('loss', 'sum'), severe=('severe', 'sum'), ledger_ok=('ledger_ok', 'all')).reset_index()
    tot['binding_bad'] = [bind.get((d, p), None) for d, p in zip(tot.day, tot.policy)]
    piv = E.pivot_table(index=['day', 'm', 't_ms'], columns='policy', values='value')
    diffs = [dict(day=d, policy=p, versus=b, n=int(g[[p, b]].dropna().shape[0]), diff_sum=float((g[p] - g[b]).sum()), positive=int(((g[p] - g[b]) > 0).sum()),
                  negative=int(((g[p] - g[b]) < 0).sum())) for d, g in piv.groupby(level=0) for p in N.POLICIES for b in ('ENTRY_ONLY', 'PI0') if p != b and p in g and b in g]
    reached = R.groupby(['day', 'status']).size().astype(int).rename('n').reset_index().to_dict('records') if len(R) else []
    return dict(version=VERSION, statuses=st, totals=tot.to_dict('records'), diffs=diffs, reached=reached, worlds=int(len(piv)))


def summarize(out=NR.OUT, parts=None):
    """The v2 summary (header) -> replay.json, after the four CSVs are proved byte-identical to the frozen summarize's."""
    out = Path(out); parts = parts if parts is not None else AR.population(json.loads((AE.EVAL_DIR / 'plan.json').read_text()))
    E, R, Ch, Ex, st = NR.frames(out, parts)
    for name, df in zip(CSVS, (E, R, Ch, Ex)):
        C.require(df.to_csv(index=False).encode() == (out / f'{name}.csv').read_bytes(), f"{name}.csv differs from the frozen summarize's")
    js = dict(summary(E, R, st, bindings(out, parts)), supersedes='pf_adv2_entry_replay.summarize (602fdfc5): replay.json only', sources=NR.sources(),
              csv_sha256={n: C.sha(out / f'{n}.csv') for n in CSVS}, source_sha256=C.sha(Path(__file__)), asof=C.stamp())
    (out / 'replay.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(statuses=st, worlds=js['worlds'], diffs=len(js['diffs']), reached=js['reached']), indent=1, default=str))


def selftest():
    import pf_adv2_entry_replay_summary_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['summarize']:
        summarize()
    else:
        sys.exit(__doc__)
