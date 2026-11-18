"""MUTATION CHECK of the execution-window ledger's selftest (2026-11-18, session aea5bfb9): a checker that cannot fail proves nothing. Each
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
    ('V0b v1 regression ignored', "ok_b = all(abs(reg[k][x] - v1[k][x]) <= tol for k in reg for x in ('E_min', 'E_max')) and all(reg[k]['order'] == v1[k]['order'] for k in reg)",
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
    ('known verdict ignores K4', "'; RESEARCH-WINDOWS-' + read['K4 RESEARCH-WINDOWS-VS-LIFECYCLE']", "'; RESEARCH-WINDOWS-CONSISTENT'"),
    ('K5 always covered', "('COVERED' if len(ab) and len(cb) and k5['activation_upper_ms']['p95'] <= ACT_RANGE[1]", "('COVERED' if True or len(ab) and len(cb) and k5['activation_upper_ms']['p95'] <= ACT_RANGE[1]"),
    ('K5 cancel bound not measured from the request', "cb = (K.ack.values - K.req.values)", "cb = (K.ack.values - K.ack.values)"),
    # --- the known-lifecycle checks
    ('K1 cancel window from the ack alone', "c_lo = np.where(np.isfinite(K.req.values), K.req.values, K.place.values); c_hi", "c_lo = K.ack.values; c_hi"),
    ('K2 evidence use never tested', "evidence_contradictions=int(miss(*rest_ev, K).sum())", "evidence_contradictions=0"),
    ('K3 permutation not applied', "Kp.loc[Kp.index[ix], ['place', 'upper']] = K[['place', 'upper']].values[pm]", "Kp = Kp"),
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
