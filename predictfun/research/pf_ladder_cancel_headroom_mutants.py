"""MUTATION CHECK of pf_ladder_cancel_headroom (FEASIBLE-CANCEL-HEADROOM-v1; 2026-11-22, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_cancel_headroom_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once.
Not planted (equivalent on every fixture and on the saved paths): the fills' (t, trade index) sort key (one generation's same-time fills share
their sign and price, so their order never changes a value) and the run's panel / fill-count guards (exercised on the saved data only)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_cancel_headroom.py'
SELFTEST = 'pf_ladder_cancel_headroom_selftest'
FAIL = 'CANCEL HEADROOM SELFTEST FAILED'
MUTANTS = [
    ('a fill at the effect retained', "return F['t'] >= cut + effect if cut is not None", "return F['t'] > cut + effect if cut is not None"),
    ('the effect at the request tick', "LIVE, KEEP, FALLBACK, EFFECT_MS, NOTICE_MS, TOL, REPL_MS = 0, 2, 11, 10., 100., 1e-6, 1000.",
     "LIVE, KEEP, FALLBACK, EFFECT_MS, NOTICE_MS, TOL, REPL_MS = 0, 2, 11, 0., 100., 1e-6, 1000."),
    ('KEEP loses exact ties', "    if not vals.max() > tot + TOL:", "    if not vals.max() > tot - TOL:"),
    ('the latest of equal cuts', "(vals > tot + TOL) & (vals >= vals.max() - TOL))[0]])", "(vals > tot + TOL) & (vals >= vals.max() - TOL))[-1]])"),
    ('unvalued fills counted as losses', "np.cumsum(np.nan_to_num(F['v1']))", "np.cumsum(np.nan_to_num(F['v1'], nan=-1.))"),
    ('one unvalued fill not incomplete', "    if p['n_unv'] > 0:", "    if p['n_unv'] > 1:"),
    ('pure without the tolerance', "    if L > TOL and G <= TOL:", "    if L > TOL and G <= 0.:"),
    ('compensated and uncompensated swapped', "return 'compensated_mixed' if G - L > TOL else ('uncompensated_mixed' if L - G > TOL else 'mixed_tie')",
     "return 'uncompensated_mixed' if G - L > TOL else ('compensated_mixed' if L - G > TOL else 'mixed_tie')"),
    ('ties without the tolerance', "else ('uncompensated_mixed' if L - G > TOL else 'mixed_tie')", "else ('uncompensated_mixed' if L - G > 0. else 'mixed_tie')"),
    ('the panel without the incomplete precedence', "out[U > 0] = 'incomplete_valuation'; return out", "return out"),
    ('the panel suffix from the right', "k = np.searchsorted(F['t'], keeps + effect, side='left'); cq", "k = np.searchsorted(F['t'], keeps + effect, side='right'); cq"),
    ('the panel known remaining without the notice', "kn = np.searchsorted(F['t'] + notice, keeps, side='right')", "kn = np.searchsorted(F['t'], keeps, side='right')"),
    ('the anchor known remaining without the notice', "size - float(F['q'][F['t'] + notice <= cut].sum())", "size - float(F['q'][F['t'] <= cut].sum())"),
    ('the venue remaining from the suffix', "rem_venue=np.nan if cut is None else size - float(F['q'][~sm].sum())", "rem_venue=np.nan if cut is None else size - float(F['q'][sm].sum())"),
    ('a filled-out order never filled', "('filled_out' if float(o['rem']) <= 1e-9 else", "('filled_out' if float(o['rem']) < 0. else"),
    ('the unacknowledged flag inverted', "return cause, bool(cause == 'recorded_cancellation' and cl is None)", "return cause, bool(cause == 'recorded_cancellation' and cl is not None)"),
    ('the replacement strictly after the request', "and float(o['req']) <= float(x['submit']) <= float(o['req']) + REPL_MS)", "and float(o['req']) < float(x['submit']) <= float(o['req']) + REPL_MS)"),
    ('the earliest cut before the bad fill', "c = keeps[keeps + effect <= tb]; r.update(seq_bad_then_good=True, seq_btg_t_bad=float(tb), seq_btg_cut=float(c.max())",
     "c = keeps[keeps + effect <= tb]; r.update(seq_bad_then_good=True, seq_btg_t_bad=float(tb), seq_btg_cut=float(c.min())"),
    ('an effect at the good fill as intervening', "c = keeps[(keeps + effect > tg) & (keeps + effect <= tb)]", "c = keeps[(keeps + effect >= tg) & (keeps + effect <= tb)]"),
    ('one sign change as interleaved', "seq_interleaved=bool(changes >= 2)", "seq_interleaved=bool(changes >= 1)"),
    ('simultaneous good / bad read as good', "sg.append(2 if (Gj > TOL and Lj > TOL) else", "sg.append(1 if (Gj > TOL and Lj > TOL) else"),
    ('the whole-generation class of nothing', "r['gen_class'] = suffix_class(part(F, np.ones(len(F['t']), bool)))", "r['gen_class'] = suffix_class(part(F, np.zeros(len(F['t']), bool)))"),
    ('the last second valued', "mv[0] if t <= end - 1000. else np.nan", "mv[0] if t <= end else np.nan"),
    ('the END value without the side', "mv[2], sg * q * (mark - p)))", "mv[2], q * (mark - p)))"),
    ('fallback flags inverted', "y[o, 8].astype(int) == FALLBACK)", "y[o, 8].astype(int) != FALLBACK)"),
    ('PULL callbacks as cuts', "x = cm[(cm[:, 2] == LIVE) & (cm[:, 7] == KEEP)]", "x = cm[(cm[:, 2] == LIVE) & (cm[:, 7] >= KEEP)]"),
    ('the reproduction dropped', "    need(not bad and len(J) == len(arms_ref), f'the unchanged path does not reproduce the accepted totals: {bad[:4]}'); return int(len(J))",
     "    return int(len(J))"),
    ('the oracle class check dropped', "    out['b_classes'] = bool(set(G_.b_cls) <= {", "    out['b_classes'] = True or bool(set(G_.b_cls) <= {"),
    ('the reading on the recorded path', "d['reading'] = 'no_local_cancel_rescue' if d['b_v'] < -TOL else", "d['reading'] = 'no_local_cancel_rescue' if d['v_keep'] < -TOL else"),
    ('the loss share over the suffixes only', "d[f'{X}_{c}_Lshare'] = d[f'{X}_{c}_L'] / d['all_L'] if d['all_L'] > 0 else np.nan",
     "d[f'{X}_{c}_Lshare'] = d[f'{X}_{c}_L'] / max(d['a_suf_L'], 1e-12) if d['all_L'] > 0 else np.nan"),
    ('the top market only', "d['b_removed_top3_market_share'] = float(byM.head(3).sum() / byM.sum())", "d['b_removed_top3_market_share'] = float(byM.head(1).sum() / byM.sum())"),
    ('compensated and uncompensated counts swapped', "compensated=int((net > TOL).sum()), uncompensated=int((net < -TOL).sum()))",
     "compensated=int((net < -TOL).sum()), uncompensated=int((net > TOL).sum()))"),
    ('the class table by the first anchor', "            for k, g in G_.groupby(list(ks) + [f'{X}_cls']):", "            for k, g in G_.groupby(list(ks) + ['a_cls']):"),
    ('the casebook by gid', "for r in g.sort_values('key').head(3).to_dict('records'):", "for r in g.sort_values('gid').head(3).to_dict('records'):"),
    ('the projection gate dropped', "need(pp.exists() and json.loads(pp.read_text()).get('fits') is True, 'the projection is missing or does not fit')",
     "need(True, 'the projection is missing or does not fit')"),
    ('run without write-once', "need(not (od / 'audit.json').exists(), 'the audit exists (write-once)')", "need(True, 'the audit exists (write-once)')"),
    ('build_manifest without write-once', "need(not path.exists(), 'the manifest exists (write-once)')", "need(True, 'the manifest exists (write-once)')"),
    ('project without write-once', "need(not path.exists(), 'the projection exists (write-once)')", "need(True, 'the projection exists (write-once)')"),
]
