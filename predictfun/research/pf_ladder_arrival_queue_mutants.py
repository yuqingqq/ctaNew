"""MUTATION CHECK of pf_ladder_arrival_queue (ARRIVAL-QUEUE-DECOMPOSITION-v1; 2026-11-22, session aea5bfb9), run by pf_mutant_one.py with
pf_ladder_arrival_queue_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs exactly once."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_arrival_queue.py'
SELFTEST = 'pf_ladder_arrival_queue_selftest'
FAIL = 'ARRIVAL QUEUE SELFTEST FAILED'
MUTANTS = [
    ('interval boundaries closed on the right', "        if e + a <= t_f < e + b:", "        if e + a < t_f <= e + b:"),
    ('a completing fill at an interval start not at risk', "        if (G < lo) if cause == 'full_fill' else (G <= lo):", "        if G <= lo:"),
    ('a removal at an interval start at risk', "        if (G < lo) if cause == 'full_fill' else (G <= lo):", "        if G < lo:"),
    ('END censoring not distinguished', "end='end_censored')", "end='removed_in_interval')"),
    ('the removal first in the tie order', "    return G, ('full_fill' if full == G else 'end' if end == G else 'removal')",
     "    return G, ('removal' if float(o['remove']) == G else 'full_fill' if full == G else 'end')"),
    ('a negative queue estimate not unknown', "    if not fin(Q) or Q < 0:\n        return 'unknown'", "    if not fin(Q):\n        return 'unknown'"),
    ('Q = R beyond R', "return 'within_R' if Q <= R else 'beyond_R'", "return 'within_R' if Q < R else 'beyond_R'"),
    ('R = 0 valid', "    if not fin(R) or R <= 0:", "    if not fin(R) or R < 0:"),
    ('q5 strictly above the edge', "('q5' if x >= last else 'positive_below_q5')", "('q5' if x > last else 'positive_below_q5')"),
    ('a zero move adverse', "return 'non_adverse' if x <= 0 else", "return 'non_adverse' if x < 0 else"),
    ('the comparison without the float32 cast', "    x = float(np.float32(x))", "    x = float(x)"),
    ('a horizon past END valid', "or t + h > end) else SG[s]", ") else SG[s]"),
    ('a missing mark counted as zero', "            if not r[f'ok{h}']:\n                r[f'v{h}'] = NAN", "            if False:\n                r[f'v{h}'] = NAN"),
    ('known remaining without the notice', "if float(f[0]) + NOTICE <= t))", "if float(f[0]) <= t))"),
    ('a later state used as history', "    if tp > d:\n        return ('future_state_refused', NAN, NAN, NAN)\n", ""),
    ('the same-tick first review as progress', "    if tp == d:\n        return (f'{kind}_is_decision', Qp, NAN, 0.)\n", ""),
    ('the order match without the price', "and float(o_a['size']) == float(o_b['size']) and float(o_a['price']) == float(o_b['price'])", "and float(o_a['size']) == float(o_b['size'])"),
    ('not-yet-born decisions not distinguished', "    if d < float(o['act']):\n        return ('not_yet_born', NAN, NAN, NAN)\n", ""),
    ('the venue record at a nearby tick', "    if float(venue[0]) != tf or not same_order(o, ident_of(first, o)):", "    if abs(float(venue[0]) - tf) > 50. or not same_order(o, ident_of(first, o)):"),
    ('the notice gap with a closed lower end', "if tf - NOTICE < float(f[0]) <= tf)", "if tf - NOTICE <= float(f[0]) <= tf)"),
    ('recent depletion at a nearby state', "if first is not None and float(first['t']) == d - RECENT and", "if first is not None and abs(float(first['t']) - (d - RECENT)) <= 250. and"),
    ('the window reconciliation dropped', "    C.require(not bad, f'the interval union does not reproduce the audit window: {bad[:5]}')\n", ""),
    ('the exactly-once check dropped', "keys = list(zip(S_.day, S_.m, S_.gid)); C.require(len(keys) == len(set(keys)) == len(states), 'the live snapshots are not each present exactly once')",
     "keys = list(zip(S_.day, S_.m, S_.gid))"),
    ('the pre-effect count not reconciled', "int(r.fills) == int(r.win_n) and int(r.pre_effect_fills) == int(r.pre_n) and", "int(r.fills) == int(r.win_n) and"),
    ('the interval-row count not checked', "    C.require(len(I_) == len(INTERVALS) * len(S_) and", "    C.require(True or len(I_) == len(INTERVALS) * len(S_) and"),
    ('run without the write-once check', "C.require(not (od / 'audit.json').exists(), 'the decomposition exists (write-once)');", ""),
    ('the union status of a boundary end as at-risk', "U['follow_up'] = np.where(U._r, U.cause.map(CAUSE_STATUS), 'not_at_risk')", "U['follow_up'] = np.where(U._r, 'at_risk_full', 'not_at_risk')"),
    ('the union status ignoring the at-risk test', "U['follow_up'] = np.where(U._r, U.cause.map(CAUSE_STATUS), 'not_at_risk')", "U['follow_up'] = U.cause.map(CAUSE_STATUS)"),
    ('progress without the unchanged band', "('advanced' if pr_f[2] < 0 else 'unchanged' if pr_f[2] == 0 else 'increased')", "('advanced' if pr_f[2] < 0 else 'increased')"),
    ('partial counting filled-out states', "F['partial'] = F.any_fill & ~F.filled_out", "F['partial'] = F.any_fill"),
    ('the common cohort replaced by the native one', "ok = F[f'ok{h}'].astype(bool) if coh == 'native' else F['common'].astype(bool)", "ok = F[f'ok{h}'].astype(bool)"),
    ('the state-level positive part from fills', "k + 'state_positive': v.clip(lower=0.)", "k + 'state_positive': F[f'fpos{h}'].where(ok, 0.)"),
    ('the any-fill rate over all opportunities', "G['any_fill_rate'] = np.where(G.at_risk > 0, G.any_fill / G.at_risk.where(G.at_risk > 0, 1), NAN)",
     "G['any_fill_rate'] = np.where(G.at_risk > 0, G.any_fill / G.opportunities, NAN)"),
    ('the snapshot side not checked against its order', "    C.require(int(z['side']) == int(o['side']) and int(z['rung']) == round(float(o['rung']))",
     "    C.require(int(z['rung']) == round(float(o['rung']))"),
    ('a window fill after the generation end accepted', "        C.require(t_f < G or (cause == 'full_fill' and t_f == G), 'a window fill after the generation end')\n", ""),
    ('pre-effect counting the fills before the decision', "pre += int(d <= t_f < e); before += int(t_f < d)", "pre += int(t_f < e); before += int(t_f < d)"),
    ('build_manifest without the write-once check', "C.require(not path.exists(), 'the manifest exists (write-once)');", ""),
]
