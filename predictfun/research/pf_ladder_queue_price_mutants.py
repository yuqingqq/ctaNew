"""MUTATION CHECK of pf_ladder_queue_price (QUEUE-PRICE-STATE-AUDIT-v1; 2026-11-22, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_queue_price_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_queue_price.py'
SELFTEST = 'pf_ladder_queue_price_selftest'
FAIL = 'QUEUE PRICE SELFTEST FAILED'
MUTANTS = [
    ('at the touch counted inside', "    return 'inside_best' if depth < 0 else", "    return 'inside_best' if depth <= 0 else"),
    ('a near-touch depth at the touch', "('at_touch' if depth == 0 else 'behind_best')", "('at_touch' if abs(depth) < 0.5 else 'behind_best')"),
    ('a negative display counted zero', "    return 'shown_zero' if shown == 0 else", "    return 'shown_zero' if shown <= 0 else"),
    ('a small display not positive', "('shown_positive' if shown > 0 else 'shown_invalid')", "('shown_positive' if shown >= 1 else 'shown_invalid')"),
    ('a negative queue estimate not unknown', "    if not fin(Q) or Q < 0:\n        return 'unknown'", "    if not fin(Q):\n        return 'unknown'"),
    ('the duplicated-key check dropped', "    need(not F.duplicated(KEY).any(), 'a duplicated feature key')\n", ""),
    ('the join without the decision time', "KEY = ['day', 'm', 'gid', 't', 'side']", "KEY = ['day', 'm', 'gid', 'side']"),
    ('the queue agreement dropped', "need(same(J.Q, J.fp_queue) and same(J.R, J.fp_remaining) and same(J.o_depth, J.o_depth_f),", "need(same(J.R, J.fp_remaining) and same(J.o_depth, J.o_depth_f),"),
    ('a missing value never equal', "same = lambda a, b: bool(((a == b) | (a.isna() & b.isna())).all())", "same = lambda a, b: bool((a == b).all())"),
    ('the saved-band check dropped', "need(qb == list(S_.queue_band), 'a queue band differs from the saved one')", "need(True, 'a queue band differs from the saved one')"),
    ('the principal as either condition', "pr = (S_.price_state == PRINCIPAL[0]) & (S_.display == PRINCIPAL[1])", "pr = (S_.price_state == PRINCIPAL[0]) | (S_.display == PRINCIPAL[1])"),
    ('the exact-depth view without the display', "bh = (S_.price_state == 'behind_best') & (S_.display == 'shown_positive')", "bh = (S_.price_state == 'behind_best')"),
    ('the union status ignoring the at-risk test', "U['follow_up'] = np.where(U._r, U.cause.map(CAUSE_STATUS), 'not_at_risk')", "U['follow_up'] = U.cause.map(CAUSE_STATUS)"),
    ('the valid any-fill rate over all at-risk states', "D_[f'h{h}_any_fill_rate'] = np.where(vr > 0, vf / vr.where(vr > 0, 1), NAN)",
     "D_[f'h{h}_any_fill_rate'] = np.where(ar > 0, vf / ar.where(ar > 0, 1), NAN)"),
    ('sparse on the all-state filled states', "D_[f'h{h}_sparse'] = vf < SPARSE_MIN", "D_[f'h{h}_sparse'] = G.all_filled < SPARSE_MIN"),
    ('per share over all shares', "D_[f'h{h}_per_share'] = np.where(q > 0, G[k + 'net'] / q.where(q > 0, 1.), NAN)",
     "D_[f'h{h}_per_share'] = np.where(q > 0, G[k + 'net'] / G.all_shares.where(G.all_shares > 0, 1.), NAN)"),
    ('censoring from the native cohort', "'censored_from_common': G.all_opportunities - G['common__0__valid']", "'censored_from_common': G.all_opportunities - G['native__0__valid']"),
    ('the time-to-END count at 60 s', "F['lt30'] = F.t_to_end < T30", "F['lt30'] = F.t_to_end < 2 * T30"),
    ('valid filled counting invalid fills', "k + 'valid_filled': (ok & F.filled).astype(int)", "k + 'valid_filled': F.filled.astype(int)"),
    ('the common cohort replaced by the native one', "ok = (F[f'ok{h}'] if coh == 'native' else F['common']).astype(bool)", "ok = F[f'ok{h}'].astype(bool)"),
    ('run without the write-once check', "need(not (od / 'audit.json').exists(), 'the audit exists (write-once)');", ""),
    ('the interval-row count not checked', "    need(len(I_) == len(INTERVALS) * len(S_) and", "    need(True or len(I_) == len(INTERVALS) * len(S_) and"),
    ('build_manifest without the write-once check', "need(not path.exists(), 'the manifest exists (write-once)');", ""),
]
