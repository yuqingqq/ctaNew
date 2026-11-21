"""MUTATION CHECK of pf_adv2_replay (the bounded own-state R4 diagnostic; review 315979d2's round; 2026-11-21, session aea5bfb9). Each planted
error is a textual replacement in a copy of pf_adv2_replay.py, executed under its real path in a fresh process by pf_mutant_one.py, which runs
pf_adv2_replay_selftest.run on it (the cached synthetic world: PF_TEACHER_SYN). Required: the control passes; every mutant is caught BY A
CHECK; every target occurs exactly once. Per-mutant jobs: pf_mutant_queue.sh / pf_mutant_one.py / pf_mutant_summary.py."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_adv2_replay.py'
SELFTEST = 'pf_adv2_replay_selftest'
FAIL = 'ADV2 REPLAY SELFTEST FAILED'
MUTANTS = [
    ('nine worlds an hour', "PER_HOUR, PART, AUDIT_EVERY, TAIL = 8, 32, 35, 30.", "PER_HOUR, PART, AUDIT_EVERY, TAIL = 9, 32, 35, 30."),
    ('the highest ranks', "            sel += sorted(by[h], key=lambda w: (w['rank'], LS.ckey(LS.wkey(w))))[:PER_HOUR]", "            sel += sorted(by[h], key=lambda w: (w['rank'], LS.ckey(LS.wkey(w))))[-PER_HOUR:]"),
    ('fallback decisions eligible', "        if r['status'] != 'model' or r['state'] not in ('absent', 'live')", "        if r['state'] not in ('absent', 'live')"),
    ('the maximum rank', "        if (r['side'], head) not in best or rk < best[(r['side'], head)][0]:", "        if (r['side'], head) not in best or rk > best[(r['side'], head)][0]:"),
    ('the prediction in the key', "generation=int(r['generation']),\n                   decision_ms=int(round(r['t'])))", "generation=int(r['generation']),\n                   decision_ms=int(round(r['t'])), A=round(float(r['A']), 6))"),
    ('the union keeps gaps', "        if out and a <= out[-1][1]:", "        if out and a < out[-1][1]:"),
    ('the intersection advances both', "        if p[i][1] < q[j][1]:\n            i += 1\n        else:\n            j += 1\n", "        i += 1; j += 1\n"),
    ('refused orders quote', "        if o['refused']:\n            continue\n        cum, full", "        cum, full"),
    ('full fills ignored', "a, b = float(o['act']), min(float(o['remove']), full if full is not None else np.inf, end)", "a, b = float(o['act']), min(float(o['remove']), end)"),
    ('at best on any frame', "            if j >= 0 and f[j, col] == o['price']:", "            if j >= 0:"),
    ('the ask value with the bid sign', "side_val = {s: float(sum(SG[s] * (mark - p) * q", "side_val = {s: float(sum(SG[0] * (mark - p) * q"),
    ('the END markout at the fill price', "        mk['mend'] += SG[s] * (mark - p) * q", "        mk['mend'] += SG[s] * (p - p) * q"),
    ('every pull a first-review pull', "first_review_pulls=sum(1 for a in live_first.values() if a == 'PULL')", "first_review_pulls=sum(1 for e in req if e[1] == 'policy_request')"),
    # Run 1's 'the fingerprint after the decision' (fp after inner(v)) escaped as an EQUIVALENT mutant: inside the callback the branch cannot
    # change (pf_branch_engine.decide applies the action after the policy returns; the causal probe works on copies). Replaced at the same site:
    ('the fingerprint under the other side\'s identity', "fp = L.focal_fingerprint(B, day, self.o[s]);", "fp = L.focal_fingerprint(B, day, self.o[1 - s]);"),
    ('the fork check skipped', "        if ctx['fingerprint'] != d['fingerprint']:\n            out.append(dict(key=x['key'], status='fork_mismatch', eligible=x['eligible'])); continue\n", ""),
    ('PI0 as the roll-in', "        d = x['dec']; fac = A.factory('RICH', heads, csup, M, sup, world)", "        d = x['dec']; fac = A.factory('PI0', heads, csup, M, sup, world)"),
    ('the replay without the committed fit', "    C.require(AE.committed(str(rec_dir / 'fit' / 'fit.json'), A.OUT / 'fit.json'), 'the fit manifest is not committed in HEAD: no outcome before the model hashes')\n", ""),
    ('another domain', "DOMAIN = 'adv2_rich_v1_reached_20261121'", "DOMAIN = 'adv2_rich_v1_reached'"),
    ('churn: the last review, not the first', "        if d.get('state') == 'live' and d.get('gid') is not None and d['gid'] not in rv:\n", "        if d.get('state') == 'live' and d.get('gid') is not None:\n"),
    ('churn: the change reversed', "x[f'd_{c}'] = (a1 - a0) if", "x[f'd_{c}'] = (a0 - a1) if"),
    ('churn: the post time is the review time', "tp = made.get(gid, (None, None))[0];", "tp = float(r['t']);"),
    ('churn: a candidate value read as the teacher\'s', "    return None if not d else d.get('A', d.get('V'))", "    return None if not d else d.get('V')"),
    ('exposure: the dimension keys dropped', "keys = [f\"{d['state']}|{d['status']}|hbin={hb[i]}\" for i, d in enumerate(rows)] + [f\"{d['state']}|{d['status']}|{k}={cats[k][i]}\" for k in R2.DIMS for i, d in enumerate(rows)]", "keys = [f\"{d['state']}|{d['status']}|hbin={hb[i]}\" for i, d in enumerate(rows)]"),
    ('cap_ok on the wrong bound', "cap_ok=bool(pk <= BE.CAP + 1e-9)", "cap_ok=bool(pk < 0.)"),
    ('reached: no status for a side / head without a selection', "    out = [dict(key=dict(side=int(s), head=h), status='no_eligible_decision', eligible=0) for s, h in A.HEADS if (s, h) not in sel]", "    out = []"),
    ('reached: the inclusion count of the first decision', "head = 'post' if r['state'] == 'absent' else 'keep'; n[(r['side'], head)] = n.get((r['side'], head), 0) + 1", "head = 'post' if r['state'] == 'absent' else 'keep'; n[(r['side'], head)] = 1"),
]
