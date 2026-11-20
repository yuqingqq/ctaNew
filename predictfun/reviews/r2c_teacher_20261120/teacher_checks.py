"""The pi0-50 receipt's computed checks, declared 2026-11-20 before any fit, freeze, diagnostic or rollout output existed (review 3d7eae6a's next
priority). Reads the outputs in data/predict_fun/research/r2_teacher_20261120 and writes checks.json beside this script.

CONTRACT PREDICATES (must all hold; each is computed):
 (A) the fits: 16 heads (4 groups x p / q / qm / severe), each 200 iterations, each saved model reloading to identical predictions; every
     head's training rows = the extraction manifest (p and severe: the group's rows; q and qm: its filled rows);
 (B) the freeze: freeze.json's 16 model hashes equal the fit records and the files on disk. It names the plan, the records manifest, the
     training manifest, the support JSON, the 107 ordered inputs and the resolved parameters. Multiplicity: 1 teacher, 0 candidates;
 (C) the adapter: categories and cells recomputed on every training row equal the census support JSON;
 (D) the rollout: all 128 worlds with an episode (or a counted status); every ledger invariant true; every policy call logged; causal checks
     > 0 with zero mismatches; the delivered control fired; inputs = the independent rebuild on every sampled decision (parity zero).
DECLARED READINGS (reported, never gated, never tuned on):
 - per side / head / horizon: the in-sample diagnostics (diag_horizon.csv);
 - rollout: decisions by status and action, engine fallbacks and constraints, entries (offset 0 / delayed / none), later generations,
   PULLs, fills, refusals;
 - delayed_entry_present: at least one original (per side) with a delayed first entry;
 - fallback_dominant: the support / forecast fallbacks (feature_unavailable, unsupported_horizon, insufficient_training_support,
   unsupported_state_category, outside_training_range, incoherent_forecast) plus the engine's no-book fallbacks, as a share of all
   policy-relevant decisions (policy calls + engine fallbacks), is > 0.5. The structural statuses cap_blocked / near_end are not
   fallbacks.
 If delayed_entry_present is false or fallback_dominant is true, the receipt reports it: no genuine adaptive-timing claim.
Usage: teacher_checks.py"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
O = Path('/home/yuqing/ctaNew/data/predict_fun/research/r2_teacher_20261120')
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_teacher as T                                  # noqa: E402

C = T.C; res = {}; read = {}
man = json.loads((O / 'train' / 'manifest.json').read_text()); fits = [json.loads((O / f'fit_{k}.json').read_text()) for k in range(4)]
want_rows = {f'{s}_{h}': dict(p=man['groups'][f'{s}_{h}']['rows'], severe=man['groups'][f'{s}_{h}']['rows'], q=man['groups'][f'{s}_{h}']['filled'],
                              qm=man['groups'][f'{s}_{h}']['filled']) for s, h in T.GROUPS}
res['A_fits'] = bool(len(fits) == 4 and all(len(f['heads']) == 4 and all(f['heads'][hd]['iterations'] == 200 and f['heads'][hd]['reload_equal']
                                                                        and f['heads'][hd]['rows'] == want_rows[f['group']][hd] for hd in T.HEADS) for f in fits))
fz = json.loads((O / 'freeze.json').read_text())
disk = {f'{s}_{h}_{hd}': C.sha(O / 'models' / f'{s}_{h}_{hd}.txt') for s, h in T.GROUPS for hd in T.HEADS}
res['B_freeze'] = bool(fz['models'] == disk == {f"{f['group']}_{hd}": f['heads'][hd]['sha256'] for f in fits for hd in T.HEADS} and fz['plan_sha256'] == T.PLAN_SHA
                       and fz['records_sha256'] == C.sha(T.REAL['records']) and fz['train_manifest_sha256'] == C.sha(O / 'train' / 'manifest.json')
                       and fz['support_sha256'] == C.sha(T.REAL['support']) and fz['features'] == T.FEATS and fz['params'] == T.PARAMS
                       and fz['multiplicity'] == dict(teacher_versions=1, candidates=0))
dg = json.loads((O / 'diag' / 'diag.json').read_text())
res['C_adapter'] = bool(dg['adapter']['categories_equal'] and dg['adapter']['cells_equal'])
ro = json.loads((O / 'rollout' / 'rollout.json').read_text()); sm = ro['summary']; cz = sm['causal']
W = pd.read_csv(O / 'rollout' / 'worlds.csv'); D = pd.read_csv(O / 'rollout' / 'decisions.csv'); Pq = pd.read_csv(O / 'rollout' / 'per_side.csv')
ok_w = W[W.status == 'ok']
res['D_rollout'] = bool(len(W) == 128 and len(ok_w) > 0 and ok_w.ledger_ok.astype(bool).all() and (ok_w.logged == ok_w.engine_asked - ok_w.engine_fallbacks).all()
                        and cz['checked'] > 0 and cz['mismatch'] == 0 and cz['delivered_changed'] > 0 and cz['parity_checked'] > 0 and cz['parity_mismatch'] == 0)
FB = ('feature_unavailable', 'unsupported_horizon', 'insufficient_training_support', 'unsupported_state_category', 'outside_training_range', 'incoherent_forecast')
n_fb = int(D.status.isin(FB).sum()) + int(ok_w.engine_fallbacks.sum()); n_rel = int(len(D)) + int(ok_w.engine_fallbacks.sum())
read.update(worlds=int(len(W)), worlds_ok=int(len(ok_w)), world_statuses=W.status.value_counts().to_dict(), decisions=int(len(D)),
            by_status={f'{s} {k}': int(v) for (s, k), v in D.groupby(['side', 'status']).size().items()},
            by_action={f'{s} {k}': int(v) for (s, k), v in D.groupby(['side', 'action']).size().items()},
            engine_fallbacks=int(ok_w.engine_fallbacks.sum()), entries={f'{s} {k}': int(v) for (s, k), v in Pq.groupby(['side', 'entry']).size().items()},
            later_generations=int(Pq.later_generations.sum()), originals_with_later_generation=int((Pq.later_generations > 0).sum()),
            pulls=int(Pq.pulls.sum()), posts=int(Pq.posts.sum()), fills=int(Pq.fills.sum()), refused=int(Pq.refused.sum()),
            fallback_share=n_fb / n_rel if n_rel else None, causal=cz)
read['delayed_entry_present'] = {str(s): bool(((Pq.side == s) & (Pq.entry == 'delayed')).any()) for s in (0, 1)}
read['fallback_dominant'] = bool(n_rel and n_fb / n_rel > .5)
read['limitation'] = (not all(read['delayed_entry_present'].values())) or read['fallback_dominant']
out = dict(asof=C.stamp(), predicates=res, all_true=all(res.values()), readings=read)
(HERE / 'checks.json').write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(dict(predicates=res, all_true=all(res.values())), default=str)); print(json.dumps(read, default=str)[:3000])
sys.exit(0 if all(res.values()) else 1)
