"""ENTRY-EXPOSURE-v1 SUMMARY v2 (P-2026-005; 2026-11-21, session aea5bfb9): supersedes the frozen summarize's override readings, in band.
The declared check C (exposure_checks.py) found the frozen KEEP wrapper's `overrides` counter above its decision log's overrides on 424 of
756 forks. The counter is incremented in pf_entry_exposure.KeepInitial.decide, which the causal probe also calls on its hidden / delivered
copies; the log holds only the branch's own callbacks. tools/probe_counter_check.py reran a failing fork: with the probe off the counter
equals the log, with it on the saved counter is reproduced, and the fills, decisions and value are identical both ways. Values, fills,
decisions, components, the predicates and the diagnosis never read the counter. This module changes only the readings built on it, from
the saved partitions:
 - overrides := the logged override decisions (the shadow log); the frozen counter is kept as overrides_counter_v1;
 - keep_baseline_equal := true with a logged override (the prefix rule covers it); without one, the KEEP arm's saved fills / decisions
   hashes and value must equal POST_PI0's; a false flag on an ok fork is the status keep_baseline_mismatch;
 - everything else is the frozen pf_entry_exposure.flat / summaries / reasons / predicates, unchanged.
Outputs under OUT/v2/ with the frozen file names (forks.csv, summary.csv, reasons.csv, exposure.json naming what it supersedes), the
manifest's byte copy and the partitions directory linked, so the frozen checks run there (exposure_checks_v2.py). Immutable: an existing
v2 exposure.json is refused. Usage (run_light): pf_entry_exposure_summary.py selftest | summarize"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_entry_exposure as M                            # noqa: E402  frozen

C, pd, LS = M.C, M.pd, M.LS
VERSION = 'entry_exposure_v1_summary_v2'
OUT2 = M.OUT / 'v2'
SUPERSEDES = ('exposure.json / forks.csv / summary.csv of the frozen summarize: overrides, overrides_forks and keep_baseline_equal counted the '
              'causal probe\'s decide calls on its copies (check C); every other field is unchanged')


def sources_v2():
    return {n: C.sha(HERE / n) for n in ('pf_entry_exposure_summary.py', 'pf_entry_exposure_summary_selftest.py')}


def logged_overrides(r):
    return len(r.get('shadow') or [])


def flat_v2(r):
    """One partition row -> the frozen flat row with the override readings from the log (header)."""
    out = M.flat(r); out['overrides_counter_v1'] = r.get('overrides')
    if r.get('arms') and 'shadow' in r:
        n = logged_overrides(r); a, b = r['arms']['POST_KEEP_INITIAL'], r['arms']['POST_PI0']
        eq = bool(n > 0 or (a['fills_sha256'] == b['fills_sha256'] and a['decisions_sha256'] == b['decisions_sha256'] and a['value'] == b['value']))
        out.update(overrides=n, keep_baseline_equal=eq)
        if r.get('status') == 'ok' and not eq:
            out['status'] = 'keep_baseline_mismatch'
    return out


def summarize_v2(out=M.OUT, out2=OUT2):
    """The v2 readings (header) -> out2/{forks.csv, summary.csv, reasons.csv, exposure.json}, out2/manifest.json, out2/parts -> out/parts."""
    out, out2 = Path(out), Path(out2); C.require(not (out2 / 'exposure.json').exists(), 'the v2 summary exists (immutable)')
    man = json.loads((out / 'manifest.json').read_text()); msha = C.sha(out / 'manifest.json'); rows = []; changed = 0
    for k, pt in enumerate(man['parts']):
        rec = json.loads((out / 'parts' / f'part_{k:03d}.json').read_text()); b = (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'] and rec['manifest_sha256'] == msha and rec['sources'] == M.sources(), f'part {k}: identity')
        rr = LS.gz_rows(b); C.require(len(rr) == 2 * len(pt['worlds']), f'part {k}: not two forks per world')
        for r in rr:
            f = flat_v2(r); changed += int(f.get('overrides') != f.get('overrides_counter_v1') and r.get('arms') is not None); rows.append(f)
    F = pd.DataFrame(rows); S = M.summaries(F); W = M.reasons(F); P = M.predicates(F)
    out2.mkdir(parents=True, exist_ok=True); F.to_csv(out2 / 'forks.csv', index=False); pd.DataFrame(S).to_csv(out2 / 'summary.csv', index=False)
    pd.DataFrame(W).to_csv(out2 / 'reasons.csv', index=False); shutil.copyfile(out / 'manifest.json', out2 / 'manifest.json')
    if not (out2 / 'parts').exists():
        (out2 / 'parts').symlink_to(out / 'parts', target_is_directory=True)
    st = F.groupby(['day', 'status']).size().rename('n').reset_index().to_dict('records')
    v1 = out / 'exposure.json'
    js = dict(version=VERSION, supersedes=SUPERSEDES, superseded_sha256=C.sha(v1) if v1.exists() else None, forks=int(len(F)), statuses=st, summary=S, reasons=W, predicates=P,
              reproduced_all=bool(F[F.status.isin(['ok', 'reproduction_failed', 'keep_baseline_mismatch'])].reproduced.all()), overrides_changed_forks=int(changed),
              manifest_sha256=msha, sources=M.sources(), sources_v2=sources_v2(), asof=C.stamp())
    (out2 / 'exposure.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(forks=js['forks'], statuses=st, overrides_changed_forks=changed, diagnosis=P['diagnosis'], reproduced_all=js['reproduced_all']), indent=1, default=str))


def selftest():
    import pf_entry_exposure_summary_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a == ['summarize']:
        summarize_v2()
    else:
        sys.exit(__doc__)
