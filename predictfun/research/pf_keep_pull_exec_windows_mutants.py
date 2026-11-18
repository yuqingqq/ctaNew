"""MUTATION CHECK of the execution-window ledger's selftest (2026-11-18, session aea5bfb9; v3 + the case ledger: the review of 1fd77004): a checker that cannot fail proves nothing. Each
entry plants one code error in a COPY of pf_keep_pull_exec_windows.py (text replacement, executed in a subprocess; the repo file is never
touched) and runs the copy's selftest (synthetic only). Required: the unchanged copy passes, every mutant fails, and every replacement
target occurs exactly as often as declared (a stale target is a failure, never a silent pass). No market data is read.
Usage (run_light, PF_SLOT=claude): pf_keep_pull_exec_windows_mutants.py [index | targets]
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'pf_keep_pull_exec_windows.py'
# (name, text in the source, replacement[, occurrences]) - entry 0 is the unchanged copy
MUT = [
    ('none (must pass)', None, None),
    # --- the ordering classifier
    ('touching endpoints read as separated (the replay convention taken as evidence)', "cf = c_hi < m_lo; mf = m_hi < c_lo;", "cf = c_hi <= m_lo; mf = m_hi <= c_lo;"),
    ('an open endpoint read as a bounded overlap', "np.where(opn, 4, 2)", "np.where(opn, 2, 2)"),
    ('an empty match window not contradictory', "return np.where(m_lo > m_hi, 5,", "return np.where(False, 5,"),
    # --- typed observations
    ('REST executedAt accepted as a point', "if kind in ('rest_confirmed_s', 'matched_status_s') and point:", "if kind in ('matched_status_s',) and point:"),
    ('REST executedAt second substituted for the match (no longer an upper bound only)', "x = np.asarray(x, float); return np.full(x.shape, -np.inf), (x + 1.) * 1000.",
     "x = np.asarray(x, float); return x * 1000., x * 1000. + 999.999"),
    ('a match instant accepted without an established source', "        raise ValueError(f'REFUSED: no source has established match-instant semantics",
     "        return x, x\n        raise ValueError(f'REFUSED: no source has established match-instant semantics"),
    # --- windows and events
    ('WS window centred on the pending time', "S1=(np.where(cls < 2, x - 250., x - 500.), np.where(ws, x, np.where(e2, x + 250., x + 500.)))",
     "S1=(np.where(cls < 2, x - 250., x - 500.), np.where(ws, x + 250., np.where(e2, x + 250., x + 500.)))"),
    ('event legs as a hull, not an intersection', "S1_lo=('S1_lo', 'max')", "S1_lo=('S1_lo', 'min')"),
    ('evidence cap dropped', "E[f'{k}_hi'] = np.minimum(E[f'{k}_hi'], E.EV_hi)", "E[f'{k}_hi'] = E[f'{k}_hi']"),
    ('an evidence-contradicted event not marked', "np.where(E[f'{k}_lo'].values > E[f'{k}_hi'].values, 2, np.where(cap, 1, 0))", "np.where(False, 2, np.where(cap, 1, 0))"),
    ('legs that disagree not marked', "np.where(legs_empty, 3,", "np.where(False, 3,"),
    ('a contradictory event keeps its empty window in the bounds', "bad = EVT[f'{k}_st'].values >= 2; anc", "bad = EVT[f'{k}_st'].values >= 9; anc"),
    # --- link and identity
    ('a leg that cannot fill names an event', "    y = y[np.where(y.side.values == 0, y.ptick.values <= y.p.values, y.ptick.values >= y.p.values)]\n    pairs", "    pairs"),
    ('anchor tolerance 1 s (a wrong identity)', "direction='nearest', tolerance=0.01)", "direction='nearest', tolerance=1000.)"),
    ('event legs cut to one per event', "return pairs, LT[LT.group.isin(pairs.group.unique())]", "return pairs, LT[LT.group.isin(pairs.group.unique())].drop_duplicates('group')"),
    ('leg fields not reconciled with legs_timed2', "bad += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())", "bad += 0"),
    ('event id not checked against the taker-order rule', "bad += int((LG.group.values.astype(np.int64) != np.where(tk >= 0, tk, -1 - LG.row_id.values.astype(np.int64))).sum())", "bad += 0"),
    ('disagreeing candidates take the first class', "return np.where(hi < 0, -1, np.where(lo == hi, lo, 2))", "return np.where(hi < 0, -1, lo)"),
    # --- contributions and the sweep
    ('score landing ignores the cancel latency', "(tau >= KP.landing(np.clip(F['T'], 1, KP.NSTEP), lat))", "(tau >= KP.landing(np.clip(F['T'], 1, KP.NSTEP), KP.CANCEL))"),
    ('null landings ignore the cancel latency', "La = KP.landing(np.arange(1, KP.NSTEP + 1), lat); top", "La = KP.landing(np.arange(1, KP.NSTEP + 1), KP.CANCEL); top"),
    ('a breakpoint repeated for one fill counted twice', "dup[:, 1:] = B[:, 1:] == B[:, :-1]", "dup[:, 1:] = False"),
    ('equal instants of an event not aggregated', "st = np.flatnonzero(np.r_[True, (np.diff(ee) != 0) | (np.diff(bb) != 0)])", "st = np.arange(len(ee))"),
    ('the window start value omitted', "np.add.at(g0, ev, np.where(fin, contrib(np.where(fin, s0, 0.), F, CW, lat), 0.))", "np.add.at(g0, ev, np.zeros(len(ev)))"),
    ('a breakpoint at the window end excluded', "(Bev <= whi[e_, None])", "(Bev < whi[e_, None])"),
    # --- reporting population and decisions
    ('cross-tab over all fills (v1\'s population)', "xtab(cls[k, c], removed, val, pl)", "xtab(cls[k, c], removed, val, np.ones(n, bool))"),
    ('conservation check on fills disabled', "if f != t['fills'] or abs(", "if False and f != t['fills'] or abs("),
    ('an ambiguous fill counted as identified in a decision', "~np.isin(cls[k, 'CP'], (0, 1))", "~np.isin(cls[k, 'CP'], (0, 1, 2))", 2),
    # --- the run's controls
    ('V0a collapsed control ignored', "ok_a = abs(e0 - float(pf['score'].sum() - pf['null'].sum())) < 1e-6 and abs(e0 - expect[0]) <= tol and abs(s0 - expect[1]) <= tol", "ok_a = True"),
    ('V0b v1 regression ignored', "ok_b = all(abs(reg[k][x] - v1[k][x]) <= tol for k in reg for x in ('E_min', 'E_max')) and all(v1[k]['order'] is None or reg[k]['order'] == v1[k]['order'] for k in reg)",
     "ok_b = True"),
    ('V0d identity control ignored', "ok_d = sum(recon.values()) == 0", "ok_d = True"),
    # --- the review of c23e6f29: contradictions over all candidates, exact joint bounds, the K4 reading
    ('contradictions counted on unique-event fills only', "cst[k] = cs = candidate_status(pf_, flag, n)", "cst[k] = cs = np.where(link_st == 1, candidate_status(pf_, flag, n), 1)"),
    ('some candidates counted as all', "np.where(h < k, 2, 3)", "np.where(h < k, 3, 3)"),
    ('contradiction accounting check always passes', "    return all(sum(v['fills'].values()) == n and abs(", "    return True or all(sum(v['fills'].values()) == n and abs("),
    ('joint solver takes the first identity assignment', ".groupby(level=0).agg(mn=('mn', 'min'), mx=('mx', 'max'))", ".groupby(level=0).agg(mn=('mn', 'first'), mx=('mx', 'first'))"),
    ('enumeration cap ignored (never relaxed)', "        if na > cap:\n", "        if False:\n"),
    ('relaxed bounds labelled exact', "bounds='outer' if rel.any() else 'exact'", "bounds='exact'"),
    ('ambiguous fills moved independently even when enumerable (no joint solve)', "        if na > cap:\n", "        if na > -1:\n"),
    ('K4 always consistent', "'CONSISTENT' if all(v['missed'] == 0 and v['cancel_first'] == 0 for v in k4.values()) else 'CONTRADICTED'", "'CONSISTENT'"),
    ('known verdict ignores K4', "'; REPLAY-ANCHOR-' + read['K4 REPLAY-ANCHOR-VS-LIFECYCLE']", "'; REPLAY-ANCHOR-CONSISTENT'"),
    # --- explicit bound types, bases, representative time, the column dictionary
    ('an evidence-capped upper bound labelled an assumption', "np.where(st == 1, 'evidence_bound', 'assumption_range')", "np.where(st == 1, 'assumption_range', 'assumption_range')"),
    ('contradictory endpoints labelled assumptions', "E[f'{k}_lo_type'] = np.where(st >= 2, 'contradictory', 'assumption_range')", "E[f'{k}_lo_type'] = np.where(st >= 9, 'contradictory', 'assumption_range')"),
    ('the evidence cap basis dropped', "E[f'{k}_hi_basis'] = np.where(st == 1, CAP_BASIS,", "E[f'{k}_hi_basis'] = np.where(st == 99, CAP_BASIS,"),
    ('representative time without its rule shift', "E['rep_time'], E['rep_rule'] = E.anc - SHIFT, REP_RULE", "E['rep_time'], E['rep_rule'] = E.anc, REP_RULE"),
    ('undocumented columns accepted', "if c not in SCHEMA.get(t, {}) and not any(", "if False and c not in SCHEMA.get(t, {}) and not any("),
    ('the structural landing kept at 37 ms in the latency bounds', "    return dict(F, lend=struct_landing(gs, lat))", "    return dict(F)"),
    ('zero-latency structural cancel stops the trade at its own instant', "return np.where(np.isnan(g), np.inf, np.where(x > g, x, np.nextafter(g, np.inf)))",
     "return np.where(np.isnan(g), np.inf, x)"),
    ('K4 offset over legs without an anchor', "ok = np.isfinite(K.Z.values)  ", "ok = np.ones(len(K), bool)  "),
    # --- the known-lifecycle checks
    ('K1 cancel window from the ack alone', "c_lo = np.where(np.isfinite(K.req.values), K.req.values, K.place.values); c_hi", "c_lo = K.ack.values; c_hi"),
    ('K2 evidence use never tested', "evidence_contradictions=int(miss(*rest_ev, K).sum())", "evidence_contradictions=0"),
    ('K3 permutation not applied', "Kp.loc[Kp.index[ix], ['place', 'upper']] = K[['place', 'upper']].values[pm]", "Kp = Kp"),
    # --- the review of 1fd77004: S3 semantics windows
    ('S3 candidates without the +250 ms frame delay', "np.where(has, iq * span + (hi0[q] + WS_DELAY - base), -np.inf)", "np.where(has, iq * span + (hi0[q] - base), -np.inf)"),
    ('S3 REST range centred on the Confirmed second again', "S = np.asarray(exec_s, float) * 1000.; return S - REST_LAG, S + 1000.", "S = np.asarray(exec_s, float) * 1000.; return S - 500., S + 500."),
    ('S3 chunked candidate pairs lose their chunk offset', "qi = a_ + np.repeat(np.arange(b_ - a_), c_);", "qi = np.repeat(np.arange(b_ - a_), c_);"),
    ('S3 value tolerance ignored', "ok = np.abs(inc[pi] - vq) <= np.maximum(VAL_TOL[0] * vq, VAL_TOL[1])", "ok = np.ones(len(pi), bool)"),
    ('S3 cell-linked legs valued by their own shares', "v[c2] = g.groupby(['m', 's', 'd', 'p']).q.transform('sum').values * pt[c2] / 100.", "v[c2] = v[c2]"),
    ('S3 E2 legs keep their drop point', "lo, hi = lo0.copy(), hi0.copy(); nc = np.zeros(n, np.int64)",
     "e2_ = R.t_e2.values.astype(float); lo, hi = np.where(np.isfinite(e2_), e2_ - 250., lo0), np.where(np.isfinite(e2_), e2_ + 250., hi0); nc = np.zeros(n, np.int64)"),
    ('S3 hull from the latest candidate only', "lo[j] = np.maximum(mn[g_] - WS_DELAY, lo0[j])", "lo[j] = np.maximum(mx[g_] - WS_DELAY, lo0[j])"),
    ('S3 window not clipped to the REST range', "hi[j] = np.minimum(mx[g_], hi0[j])", "hi[j] = mx[g_]"),
    ('S3 late links not marked', "late = (how > 0) & (R.t_ms.values.astype(float) > hi0 + WS_DELAY)", "late = np.zeros(n, bool)"),
    ('S3 events built from the S1 leg windows', "S3_lo=W['S3'][0], S3_hi=W['S3'][1]))", "S3_lo=W['S1'][0], S3_hi=W['S1'][1]))"),
    ('the ledger legs take another row\'s S3 window', "lg = pd.concat([lg, sem.iloc[lg.row_id.values.astype(np.int64)].reset_index(drop=True)], axis=1)",
     "lg = pd.concat([lg, sem.iloc[np.zeros(len(lg), np.int64)].reset_index(drop=True)], axis=1)"),
    # --- contradictory timing assumptions by cause
    ('evidence-cap and leg-intersection causes merged', "flags[k + '_cap'] = st == 2; flags[k + '_legs'] = st == 3", "flags[k + '_cap'] = st >= 2; flags[k + '_legs'] = st == 3"),
    ('mixed causes not counted', "mixed = (cst[k] == 3) & (cst[k + '_cap'] != 3) & (cst[k + '_legs'] != 3)", "mixed = (cst[k] == 3) & False"),
    ('the cause reconciliation always passes', "    return all(V3[k]['fills']['all'] == V3[k + '_cap']['fills']['all']", "    return True or all(V3[k]['fills']['all'] == V3[k + '_cap']['fills']['all']"),
    ('binding pair from the earliest start', "a = D.src.values[g.lo.idxmax().values]", "a = D.src.values[g.lo.idxmin().values]"),
    ('leg disagreements over every event', "bad = (EVT[f'{k}_st'].values == 3)[e]", "bad = (EVT[f'{k}_st'].values >= 0)[e]"),
    # --- K5 per order, both endpoints
    ('K5 ignores the range\'s lower end (the review\'s finding)', "return np.where((hi < R[0]) | (lo > R[1]), 'CONTRADICTED',", "return np.where((lo > R[1]), 'CONTRADICTED',"),
    ('K5 certified and compatible not distinguished', "np.where((lo >= R[0]) & (hi <= R[1]), 'CERTIFIED', 'COMPATIBLE-UNIDENTIFIED')", "np.where(hi <= R[1], 'CERTIFIED', 'COMPATIBLE-UNIDENTIFIED')"),
    ('K5 per leg, not per order', "U = K.drop_duplicates(['run', 'cid']).reset_index(drop=True); pl = U.place.values.astype(float)", "U = K.reset_index(drop=True); pl = U.place.values.astype(float)"),
    ('a late confirmation read as a slow cancel effect', "interval_status(np.zeros(len(U)), c_hi, CANCEL_RANGE)", "interval_status(c_hi, c_hi, CANCEL_RANGE)"),
    ('K5 reading ignores contradictions', "read['K5 ORDER-PATH INTERVALS vs DECLARED RANGES'] = ('CONTRADICTED' if k5", "read['K5 ORDER-PATH INTERVALS vs DECLARED RANGES'] = ('CONTRADICTED' if False and k5"),
    ('the confirmation source not kept apart', "np.where(~np.isfinite(uc) | (ack < uc - 1e-6), 'http_answer', 'user_stream')", "np.where(True, 'http_answer', 'user_stream')"),
    # --- K7 clock offset
    ('K7 without the receipt bounds of the Matched notices', "a = np.r_[K.M.values.astype(float) * 1000. - K.rM.values.astype(float), K.Cs.values.astype(float) * 1000. - K.rC.values.astype(float)]",
     "a = np.r_[K.Cs.values.astype(float) * 1000. - K.rC.values.astype(float)]"),
    ('K7 without the placement bounds', "b = (K.M.values.astype(float) + 1.) * 1000. - K.place.values.astype(float); a, b = a[np.isfinite(a)], b[np.isfinite(b)]",
     "b = np.array([np.inf]); a = a[np.isfinite(a)]"),
    # --- K6 the ledger's own windows and the gate
    ('K6 worst case at the latest offset', "cf0, cfw, cfr = ok & (c_hi < lo), ok & (c_hi < lo - dl), ok & (c_hi < lo - dh)", "cf0, cfw, cfr = ok & (c_hi < lo), ok & (c_hi < lo - dh), ok & (c_hi < lo - dh)"),
    ('K6 a leg certified avoided when one candidate is', "return has & (np.bincount(i, weights=np.asarray(b, float), minlength=n) == nk)", "return has & (np.bincount(i, weights=np.asarray(b, float), minlength=n) > 0)"),
    ('K6 disagreeing candidates take the first verdict', "return np.where(~has, -9, np.where(lo_ == hi_, lo_, fill))", "return np.where(~has, -9, lo_)"),
    ('K6 every miss blamed on the own source', "np.where(own, 0, 1 + KE.src.values.astype(int))", "np.where(False, 0, 1 + KE.src.values.astype(int))"),
    ('K6 gate always passes', "read[f'K6 GATE {k}'] = 'PASS' if k6[k]['false_cancel_first']['some_offset'] == 0 else 'FAIL'", "read[f'K6 GATE {k}'] = 'PASS'"),
    ('K6 every-offset verdict read at offset 0', "vr = np.where(st >= 2, v0, np.where(hi - dl < place, 1, np.where(lo - dh > upper, 2, 0)))", "vr = v0"),
    ('K6 a contradictory event classified as a window', "ok = can & (st < 2); cf0", "ok = can & (st < 9); cf0"),
    # --- the case linkage and the case ledger
    ('the anchor counted as the own row\'s whenever it is a pending time', "a_own = np.where(origin == 'pending_increment', (how > 0) & (t_ms == wmin),",
     "a_own = np.where(origin == 'pending_increment', how > 0,"),
    ('a pinned anchor the rule does not reproduce not marked', "anchor_rule_ok=rule == Zr,", "anchor_rule_ok=np.ones(len(r), bool),"),
    ('candidate rows not checked against the audit\'s counts', "C.require(np.array_equal(np.bincount(M_.i.values, minlength=len(K)), K.candidate_legs.values.astype(int)),",
     "C.require(True or np.array_equal(np.bincount(M_.i.values, minlength=len(K)), K.candidate_legs.values.astype(int)),"),
    ('own-row windows not capped by the row\'s REST second', "KE[f'own_{k}_lo'], KE[f'own_{k}_hi'] = W[k][0], np.minimum(W[k][1], cap)", "KE[f'own_{k}_lo'], KE[f'own_{k}_hi'] = W[k][0], W[k][1]"),
    ('the case rows carry the own row\'s window instead of the event\'s',
     "KE[f'{k}_lo'], KE[f'{k}_hi'], KE[f'{k}_st'] = EVT[f'{k}_lo'].values[e], EVT[f'{k}_hi'].values[e], EVT[f'{k}_st'].values[e]",
     "KE[f'{k}_lo'], KE[f'{k}_hi'], KE[f'{k}_st'] = W[k][0], W[k][1], 0"),
    ('K4 per-leg misses not written to the case ledger', "k4c.update({f'K4_{nm}_missed': mi, f'K4_{nm}_cancel_first': cf_})",
     "k4c.update({f'K4_{nm}_missed': np.zeros(len(K), bool), f'K4_{nm}_cancel_first': cf_})"),
    ('the K4 failures not traced in the report', "T_ = KC[KC.K4_S1_missed.values.astype(bool)].to_dict('records')", "T_ = KC[KC.K4_S1_missed.values.astype(bool)].to_dict('records')[:0]"),
    ('the case ledger accepts undocumented columns', "if c not in CASE_SCHEMA.get(t, {}) and not any(", "if False and c not in CASE_SCHEMA.get(t, {}) and not any("),
]
STALE = 97                                                # exit code of a replacement target that does not occur as declared


def source(i):
    name, a, b = MUT[i][:3]; n = (MUT[i] + (1,))[3]; s = SRC.read_text()
    if a is not None:
        if s.count(a) != n:
            print(f'mutation target occurs {s.count(a)} times, declared {n}: {name}', file=sys.stderr); sys.exit(STALE)
        s = s.replace(a, b)
    return s


def one(i):
    g = dict(__name__='pf_keep_pull_exec_windows_mutant', __file__=str(SRC)); exec(compile(source(i), str(SRC), 'exec'), g); g['selftest']()


def main():
    bad = 0
    for i, (name, *_) in enumerate(MUT):
        r = subprocess.run([sys.executable, __file__, str(i)], capture_output=True, text=True)
        tail = r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ['']
        ok = (r.returncode == 0) if i == 0 else (r.returncode not in (0, STALE))
        bad += not ok
        print(f'{i:2d} {name}: rc {r.returncode} -> {"passed" if r.returncode == 0 else " / ".join(x.strip()[:200] for x in tail[-2:])}{"" if ok else "   <<< UNEXPECTED"}', flush=True)
    if bad:
        sys.exit(f'EXEC-WINDOW MUTATION CHECK FAILED: {bad} unexpected outcomes of {len(MUT)}')
    print(f'exec-window mutation check passed: the unchanged copy passes its selftest, {len(MUT) - 1} / {len(MUT) - 1} planted code errors fail it', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['targets']:
        [source(i) for i in range(len(MUT))]; print(f'all {len(MUT) - 1} mutation targets occur as declared')
    elif len(sys.argv) > 1:
        one(int(sys.argv[1]))
    else:
        main()
