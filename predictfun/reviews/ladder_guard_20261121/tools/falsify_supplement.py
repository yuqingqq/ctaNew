"""POST-RUN FALSIFIER SUPPLEMENT for guard_checks.py (ASK-RESERVATION-GUARD-v1; session aea5bfb9, 2026-11-21). Added after the run; it changes
no reading and no frozen file. The frozen checker's own falsifier covers 7 of its 21 checks. This supplement adds:
 - the review f8c5e579 probe's two corruptions, applied exactly (reviews/automatic/20261121_ladder_audit_review/checker_probe.py):
   (a) the eight live / ctx_bid_reserved_shares q5 means set to their q1 mean + 1;
   (b) the audit's generation censor counter set to -999,999;
 - one in-memory corruption for every other check.
(b) targets the frozen audit field cross.generation_tails.mo1s_censored (-3,195, the known-wrong formula of finding 3). No reading uses that
field, and the frozen checker never reads it. The supplement therefore adds one check, X_frozen_censor_field: the stored field must equal the
frozen formula recomputed from generations_v2.csv (isna - no-fill), while H_censor_count carries the corrected count (13). Every case must flip
exactly its named check on an in-memory copy while the clean evidence passes all 21 + 1. Usage: falsify_supplement.py"""
import copy
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

R = Path('/home/yuqing/ctaNew/predictfun/reviews/ladder_guard_20261121'); D = Path('/home/yuqing/ctaNew/data/predict_fun/research/ladder_guard_v1_20261121')
spec = importlib.util.spec_from_file_location('guard_checks', R / 'guard_checks.py'); CK = importlib.util.module_from_spec(spec); spec.loader.exec_module(CK)


def frozen_field(E):
    g = E['gens']; return int(E['aaudit']['cross']['generation_tails']['mo1s_censored']) == int(g.lifetime_mo1s.isna().sum() - (g.n_fills == 0).sum())


def checks(E):
    r = CK.run_checks(E); r['checks']['X_frozen_censor_field'] = frozen_field(E); r['all_ok'] = bool(all(r['checks'].values())); return r


def probe_means(E):
    b = E['bins']; mask = (b.kind == 'live') & (b.feature == 'ctx_bid_reserved_shares') & (b.bin == 'q5')
    for i in b.index[mask]:
        z = b.loc[i]; q1 = b[(b.kind == z.kind) & (b.feature == z.feature) & (b.day == z.day) & (b.side == z.side) & (b.bin == 'q1')].iloc[0]
        b.loc[i, 'mo1s_mean'] = float(q1.mo1s_mean) + 1.
    E['_altered'] = int(mask.sum())


def lead_row(E):
    return E['leads'].index[(E['leads'].kind == E['aaudit']['candidate']['kind']) & (E['leads'].feature == E['aaudit']['candidate']['feature'])][0]


def logged(E):
    return next(r for r in E['recs'] if r['guard']['guard_log'] and r['guard']['requests'])


def fake_logs(E):
    E['_R'] = Path(tempfile.mkdtemp(dir=str(D))); (E['_R'] / 'logs').mkdir()
    (E['_R'] / 'logs' / 'fake.log').write_text('2026-11-21T00:00:00Z tag=fake rc=0 peak_rss_mib=2000 wall_s=3000.0 cpu_user_s=3000.0\n')


day0 = lambda E: sorted(E['days'])[0]
CASES = [
    ('probe_q5_means_q1_plus_1', 'G_bin_means', probe_means),
    ('probe_censor_counter_-999999', 'X_frozen_censor_field', lambda E: E['aaudit']['cross']['generation_tails'].__setitem__('mo1s_censored', -999999)),
    ('corrected_censor_counter', 'H_censor_count', lambda E: E['selection'].__setitem__('generation_censor_count', -999999)),
    ('lead_flag_flipped', 'G_lead_flags', lambda E: E['leads'].__setitem__('lead', E['leads'].lead.where(E['leads'].index != lead_row(E), ~E['leads'].lead.astype(bool)))),
    ('pooled_contrast_negated', 'G_lead_flags', lambda E: E['leads'].__setitem__('pooled_q5_minus_q1', E['leads'].pooled_q5_minus_q1.where(E['leads'].index != lead_row(E), -E['leads'].pooled_q5_minus_q1))),
    ('candidate_sign', 'G_candidate', lambda E: E['aaudit']['candidate'].__setitem__('sign', -int(E['aaudit']['candidate']['sign']))),
    ('candidate_feature', 'G_candidate', lambda E: E['aaudit']['candidate'].__setitem__('feature', 'ctx_ask_reserved_shares')),
    ('candidate_lead_count', 'G_candidate', lambda E: E['aaudit']['candidate'].__setitem__('n_leads', 2)),
    ('END_reading', 'F_END_reading', lambda E: E['maudit']['predicates']['END_paired'].__setitem__('answer', 'IMPROVES_EVERY_DAY')),
    ('per_share_reading', 'F_per_share_reading', lambda E: E['maudit']['predicates']['mo1s_per_share']['all'].__setitem__('answer', 'MIXED')),
    ('selection_answer', 'F_selection', lambda E: E['maudit']['predicates'].__setitem__('selection', 'SELECTION_BEYOND_EXPOSURE')),
    ('member_dropped', 'B_membership', lambda E: E['recs'].pop()),
    ('pilot_reordered', 'B_pilot', lambda E: E['gm']['pilot'].__setitem__(day0(E), list(reversed(E['gm']['pilot'][day0(E)])))),
    ('disabled_fill_dropped', 'C_disabled_equals_baseline', lambda E: E['recs'][0]['disabled'].__setitem__('fills', E['recs'][0]['disabled']['fills'][:-1])),
    ('repeated_request', 'D_rule', lambda E: logged(E)['guard']['requests'].append(list(logged(E)['guard']['requests'][0]))),
    ('cash_changed', 'E_accounting', lambda E: E['recs'][0]['guard'].__setitem__('cash', E['recs'][0]['guard']['cash'] + 1.)),
    ('edge_changed', 'A_edge', lambda E: E['gm'].__setitem__('edge', 150.)),
    ('rule_sha_changed', 'A_rule', lambda E: E['days'][day0(E)].__setitem__('rule_sha256', '0' * 64)),
    ('sources_changed', 'A_sources', lambda E: E['days'][day0(E)].__setitem__('sources', {})),
    ('output_hash_changed', 'A_outputs', lambda E: E['days'][day0(E)]['sha256'].__setitem__('gz', '0' * 64)),
    ('uncommitted_file', 'A_committed', None),
    ('resources_over_cap', 'I_resources', fake_logs),
]


def main():
    E = CK.load(); base = checks(E); out = dict(clean_all_ok=base['all_ok'], clean_checks=base['checks'], cases={})
    for name, target, fn in CASES:
        X_ = copy.deepcopy(E); gb, rr = CK.git_blob, CK.R
        try:
            if fn is None:
                CK.git_blob = lambda rel: None
            else:
                fn(X_)
            if '_R' in X_:
                CK.R = X_['_R']
            r = checks(X_)
        finally:
            CK.git_blob, CK.R = gb, rr
            if '_R' in X_:
                shutil.rmtree(X_['_R'])
        flipped = sorted(k for k, v in r['checks'].items() if base['checks'].get(k) and not v)
        out['cases'][name] = dict(target=target, target_refused=target in flipped, flipped=flipped, altered_rows=X_.get('_altered'))
    out['covered_checks'] = sorted({v['target'] for v in out['cases'].values() if v['target_refused']})
    out['uncovered_checks'] = sorted(set(base['checks']) - set(out['covered_checks']))
    out['falsifier_ok'] = bool(base['all_ok'] and all(v['target_refused'] for v in out['cases'].values()))
    (R / 'checks_falsifier_supplement.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(clean_all_ok=out['clean_all_ok'], falsifier_ok=out['falsifier_ok'], uncovered=out['uncovered_checks'],
                          cases={k: (v['target_refused'], v['flipped']) for k, v in out['cases'].items()}), indent=1))


if __name__ == '__main__':
    main()
