"""ADV2-RICH v1: THE BOUNDED OWN-STATE R4 DIAGNOSTIC ON CONSUMED OCT 26-27 (plan v2 R4, diagnostic only; P-2026-005; 2026-11-21, session
aea5bfb9). Review 315979d2's next priority, items 11-14. Declared and frozen (freeze 2) before any fit and before any Oct 26-27 outcome is
read; it refuses to run unless the fit manifest is committed (as pf_adv2_eval). This is NOT the R4 gate: same-history NEU2 and the full
economic / quoting qualification stay deferred; PI0 is not NEU2.

POPULATION: the nested first eight selected worlds of every (day, UTC hour) stratum of the frozen evaluation plan (8619eec5...), by the
 plan's world rank (ties by the canonical key): at most 8 x 24 x 2 = 384 worlds; partitions of at most 32 worlds of one day in canonical
 (m, t0) order. A world without an episode tape is a status. The partition's day inputs must equal the evaluation plan's recorded identity
 (pf_adv2_sample.guarded_files and the runner's tracked hours) before any is loaded: the replay reads exactly the labels' inputs.
POLICIES (pf_adv2.POLICIES): N0, PI0, RICH, FAIR, MECH, ENTRY_ONLY, RETAIN_ONLY, ENTRY_PA. Each world: one joint two-sided 5 s episode
 per policy from t0 on the 50 ms grid, from the same empty state to the same END, each policy on its OWN branch (inputs rebuilt on it; the
 binding check at every callback; the causal probe every 35th call: hidden copies leave the decision unchanged, the delivered copy is the
 positive control, parity with an independent rebuild). Each episode is valued once (cash + position x the END mark).
REACHED STATES (RICH; item 13): every RICH callback in an absent or live state records the focal fingerprint of the state the decision
 sees (pf_r2_labels.focal_fingerprint). Eligible = a model decision (supported, finite prediction) in an absent (POST candidate) or live
 (KEEP candidate) state. Per world / side / head the eligible decision of minimum sha256('adv2_rich_v1_reached_20261121|' + key), key =
 the canonical (day, instrument, original, side, head, generation, decision ms), ties by the key, selected after the episode and before
 any pair runs: no score, fill or label enters the choice. Its fork context is rebuilt by pf_r2_labels.bound_fork with RICH as the roll-in
 (its fingerprint must equal the recorded one, else the pair is refused), then pf_r2_labels.pair_bound with the frozen pi0 continuation
 gives the reached-state label Y, valued once and never added to an episode's P&L. At most 4 a world, 1,536 in all. RICH / FAIR / MECH
 predictions at the reached state come from the decision's own 107 inputs. Every world carries exactly one entry per side / head: the pair
 with its inclusion count (the eligible decisions it was chosen from), or the status no_eligible_decision (count 0).
EPISODE READINGS (item 14; metrics()): value, cash, position; each side's value at the mark; generations, refused posts, fills, filled
 shares; policy pulls, structural requests, effective cancels; first-review pulls over the generations live at a review; cap_ok (the peak
 absolute engine position within the 30-share cap; the engine's own reservation and position checks ran at every notice and command, and a
 violation stops the partition); engine-active
 executable quote time per side, either and both (activation to the removal effect, the full fill or END; cancel-pending stays exposed,
 pending posts do not), and the same at the side's best (the latest venue frame at or before each moment); integrated and peak absolute
 engine position; terminal position and open exposure; the tail costs of V (loss, V < 0, V < -30 c, severe excess); fill markouts
 sign x (mid - price) x shares at the fill, +100 ms and +1 s (venue frames; an endpoint after END has no frame: missing, never filled from
 another hour) and at the END mark.
CHURN (item 14; churn()): every policy, per generation reviewed live: the POST decision that created it (the branch's decision log) and its
 first live review - their times, the deciding policy's value (a candidate's A, the teacher's V), the review's status and action, and the
 values at the post / changes to the review of the recorded inputs CHURN. Recorded state only; no cause is inferred from it.
OWN-STATE EXPOSURE (item 1; exposure(), RICH): its callbacks by state and status; the decisions with inputs by state, status and each
 support dimension's category (pf_r2_teacher.categories) / horizon bin, beside the training rows' coverage (pf_adv2_eval).
Usage (run_light): pf_adv2_replay.py selftest | run <part> | summarize
"""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_adv2 as A                                      # noqa: E402
import pf_adv2_eval as AE                                # noqa: E402
import pf_adv2_sample as AS                              # noqa: E402

C, pd, T, LS, L, R2, KP = A.C, A.pd, A.T, A.LS, A.L, A.R2, A.KP
BE, X = L.BE, LS.X
VERSION = 'adv2_replay_v1'
DOMAIN = 'adv2_rich_v1_reached_20261121'
PER_HOUR, PART, AUDIT_EVERY, TAIL = 8, 32, 35, 30.
OUT = A.OUT / 'replay_v1'
END, STEP, NT = A.END, A.STEP, L.NT
SG = {0: 1., 1: -1.}
CHURN = ('fair_edge', 'o_edge', 'fp_queue', 'fp_remaining', 'pf_mid', 'pf_spread', 'adv_pf_imb1', 't_to_end')


def population(plan):
    """The nested first PER_HOUR selected worlds of every (day, hour) stratum by world rank -> partitions [(day, [worlds])]."""
    parts = []
    for day in plan['days']:
        by = {}
        for w in plan['sample'][day]['selected']:
            by.setdefault(int(w['hour']), []).append(w)
        sel = []
        for h in sorted(by):
            sel += sorted(by[h], key=lambda w: (w['rank'], LS.ckey(LS.wkey(w))))[:PER_HOUR]
        sel.sort(key=lambda w: (w['m'], w['t_ms']))
        parts += [(day, sel[i:i + PART]) for i in range(0, len(sel), PART)]
    return parts


def count(xs):
    out = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items()))


def rank_of(key):
    return hashlib.sha256((DOMAIN + '|' + A.ck(key)).encode()).hexdigest()


class Recorder:
    """Wraps a policy object's fn: before each absent / live callback, the focal fingerprint of the state and the side's generation ordinal."""

    def __init__(self, pol, day, world):
        self.pol, self.day, self.o, self.rec = pol, day, world['o'], []
        inner = pol.fn

        def fn(v):
            s, st = int(v['side']), v['state']; B = pol.B
            ordinal = sum(1 for x in B.E.o if x['side'] == s) + (1 if st == 'absent' else 0)
            fp = L.focal_fingerprint(B, day, self.o[s]); k = int(round(float(v['t']) / STEP)); n0 = len(pol.log)
            a = inner(v); d = pol.log[n0] if len(pol.log) > n0 else {}
            self.rec.append(dict(k=k, t=float(v['t']), side=s, state=st, generation=int(ordinal), fingerprint=fp, status=d.get('status'), action=a,
                                 A=d.get('A'), row=d.get('row'), oid=int(self.o[s].oid)))
            return a
        pol.fn = fn


def select_reached(rec, day, m):
    """The eligible RICH decisions (header) -> per (side, head) the minimum-rank one (no outcome enters)."""
    best, n = {}, {}
    for r in rec:
        if r['status'] != 'model' or r['state'] not in ('absent', 'live') or r['A'] is None or not np.isfinite(r['A']):
            continue
        head = 'post' if r['state'] == 'absent' else 'keep'; n[(r['side'], head)] = n.get((r['side'], head), 0) + 1
        key = dict(day=str(day), instrument_id=int(m), original_episode_id=int(r['oid']), side=int(r['side']), head=head, generation=int(r['generation']),
                   decision_ms=int(round(r['t'])))
        rk = (rank_of(key), A.ck(key))
        if (r['side'], head) not in best or rk < best[(r['side'], head)][0]:
            best[(r['side'], head)] = (rk, key, r)
    return {k: dict(key=v[1], rank=v[0][0], dec=v[2], eligible=n[k]) for k, v in sorted(best.items())}


def episode(name, heads, csup, M, sup, day, w, world, every=AUDIT_EVERY, record=False):
    """One joint own-state episode of a policy -> (branch, ledger, policy, causal stats, recorder or None)."""
    probe, cst = T.causal_probe(every)
    pol = A.make_policy(name, heads, csup, M, sup, causal=probe); B = BE.Branch(world['path'], 0, ident=R2.episode_ident(day, w['o'][0])); pol.bind(B, world)
    rcd = Recorder(pol, day, world) if record else None
    for k in range(NT):
        BE.step(B, k, pol)
    return B, BE.finish(B), pol, cst, rcd


def _union(iv):
    out = []
    for a, b in sorted(x for x in iv if x[1] > x[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _len(iv):
    return float(sum(b - a for a, b in iv))


def _inter(p, q):
    out, i, j = [], 0, 0
    while i < len(p) and j < len(q):
        a, b = max(p[i][0], q[j][0]), min(p[i][1], q[j][1])
        if b > a:
            out.append([a, b])
        if p[i][1] < q[j][1]:
            i += 1
        else:
            j += 1
    return out


def active_intervals(E, end=END):
    """Per side: the engine-active intervals of every non-refused generation (activation to min(removal effect, full fill, END)) and the
    sub-intervals at the side's best (the latest venue frame at or before each moment shows the order's price as the best)."""
    f = E.f; vt = f[:, 0]; any_, best = {0: [], 1: []}, {0: [], 1: []}
    for o in E.o:
        if o['refused']:
            continue
        cum, full = 0., None
        for t, g, s, q, p, _ in E.fills:
            if g == o['gid']:
                cum += q
                if full is None and cum >= o['size'] - 1e-9:
                    full = float(t)
        a, b = float(o['act']), min(float(o['remove']), full if full is not None else np.inf, end)
        if b <= a:
            continue
        any_[o['side']].append([a, b]); col = 2 + o['side'] * 10
        j = int(np.searchsorted(vt, a, side='right')) - 1; t = a                   # the latest venue frame at or before a (-1: none yet)
        while t < b:
            nxt = float(vt[j + 1]) if j + 1 < len(vt) else np.inf; seg_end = min(b, nxt)
            if j >= 0 and f[j, col] == o['price']:
                best[o['side']].append([t, seg_end])
            t = seg_end; j += 1
    return {s: _union(v) for s, v in any_.items()}, {s: _union(v) for s, v in best.items()}


def mid_at(f, t):
    """The venue mid at t (the latest frame with venue time <= t; both bests present), or None."""
    j = int(np.searchsorted(f[:, 0], t, side='right')) - 1
    if j < 0:
        return None
    b, a = f[j, 2], f[j, 12]
    return float((b + a) / 2.) if b >= 0 and a >= 0 else None


def metrics(B, Lx, dec, end=END):
    """The episode readings of the header (one policy, one world)."""
    if Lx['status'] != 'ok':
        return dict(status=Lx['status'])
    E = B.E; mark = B.mark; V = float(Lx['value'])
    side_val = {s: float(sum(SG[s] * (mark - p) * q for t, g, s_, q, p, _ in Lx['fills'] if s_ == s)) for s in (0, 1)}
    gens = [o for o in E.o]; req = [e for e in E.events if e[1].endswith('_request')]
    eff = sum(1 for o in gens if o['req'] is not None and o['remove'] <= end and o['rem'] > 1e-12)
    live_first = {}
    for d in dec:
        if d.get('state') == 'live' and d.get('gid') is not None and d['gid'] not in live_first:
            live_first[d['gid']] = d.get('action')
    any_, best = active_intervals(E, end)
    either, both = _union(any_[0] + any_[1]), _inter(any_[0], any_[1]); eb, bb = _union(best[0] + best[1]), _inter(best[0], best[1])
    fills = Lx['fills']; filled = float(sum(f[3] for f in fills))
    pos_t, ip, pk, cur, last = [], 0., 0., 0., 0.
    for t, g, s, q, p, _ in sorted(fills):
        ip += abs(cur) * (min(t, end) - last); cur += SG[s] * q; last = min(t, end); pk = max(pk, abs(cur))
    ip += abs(cur) * (end - last)
    mk = dict(m0=0., m100=0., m1000=0., mend=0., missing100=0, missing1000=0, missing0=0)
    for t, g, s, q, p, _ in fills:
        for lab, h in (('0', 0.), ('100', 100.), ('1000', 1000.)):
            tt = t + h; mm = mid_at(E.f, tt) if tt <= end else None
            if mm is None:
                mk[f'missing{lab}'] += 1
            else:
                mk[f'm{lab}'] += SG[s] * (mm - p) * q
        mk['mend'] += SG[s] * (mark - p) * q
    return dict(status='ok', value=V, cash=float(Lx['cash']), position=float(Lx['position']), value_bid=side_val[0], value_ask=side_val[1],
                generations=len(gens), generations_bid=sum(1 for o in gens if o['side'] == 0), generations_ask=sum(1 for o in gens if o['side'] == 1),
                refused=sum(1 for o in gens if o['refused']), fills=len(fills), filled_shares=filled,
                policy_pulls=sum(1 for e in req if e[1] == 'policy_request'), structural_requests=sum(1 for e in req if e[1] == 'structural_request'),
                effective_cancels=int(eff), reviewed_generations=len(live_first), first_review_pulls=sum(1 for a in live_first.values() if a == 'PULL'),
                quote_ms_bid=_len(any_[0]), quote_ms_ask=_len(any_[1]), quote_ms_either=_len(either), quote_ms_both=_len(both),
                best_ms_bid=_len(best[0]), best_ms_ask=_len(best[1]), best_ms_either=_len(eb), best_ms_both=_len(bb),
                inv_integral=float(ip), inv_peak=float(pk), cap_ok=bool(pk <= BE.CAP + 1e-9), open_bid=float(Lx['open_at_end'][0]), open_ask=float(Lx['open_at_end'][1]),
                loss=max(-V, 0.), neg=int(V < 0), severe=int(V < -TAIL), severe_excess=max(-V - TAIL, 0.), **mk)


def _val(d):
    return None if not d else d.get('A', d.get('V'))


def churn(log, B):
    """Per generation reviewed live (header's CHURN): its POST decision and first live review, with the recorded input values / changes."""
    made = {gid: (float(g), int(s)) for g, s, st, a, gid in B.decisions if st == 'absent' and a == 'POST' and gid is not None}
    at = {(float(d['t']), int(d['side'])): d for d in log if d.get('state') == 'absent'}; rv = {}
    for d in log:
        if d.get('state') == 'live' and d.get('gid') is not None and d['gid'] not in rv:
            rv[d['gid']] = d
    out = []
    for gid, r in sorted(rv.items()):
        tp = made.get(gid, (None, None))[0]; p = at.get((tp, int(r['side']))) if tp is not None else None; rp, rr = (p or {}).get('row'), r.get('row')
        x = dict(gid=int(gid), side=int(r['side']), t_post=tp, t_review=float(r['t']), dt=(float(r['t']) - tp) if tp is not None else None, by_post=(p or {}).get('policy'),
                 by_review=r.get('policy'), value_post=_val(p), value_review=_val(r), status_review=r.get('status'), action=r.get('action'))
        for c in CHURN:
            a0 = float(rp[c]) if rp is not None else None; a1 = float(rr[c]) if rr is not None else None
            x[f'{c}_post'] = a0; x[f'd_{c}'] = (a1 - a0) if a0 is not None and a1 is not None else None
        out.append(x)
    return out


def exposure(log):
    """RICH's own-state exposure (header): 'state|status' counts and 'state|status|dim=value' counts of the decisions with inputs."""
    out = count([f"{d['state']}|{d['status']}" for d in log]); rows = [d for d in log if d.get('row') is not None]
    if rows:
        x = pd.DataFrame([{c: d['row'][c] for c in T.FEATS} for d in rows]); x['price'] = [d['row']['price'] for d in rows]; x['generation'] = [d['row']['generation'] for d in rows]
        cats = T.categories(x); hb = T.hbin(END - np.array([d['row']['decision_ms'] for d in rows], float))
        keys = [f"{d['state']}|{d['status']}|hbin={hb[i]}" for i, d in enumerate(rows)] + [f"{d['state']}|{d['status']}|{k}={cats[k][i]}" for k in R2.DIMS for i, d in enumerate(rows)]
        out.update(count(keys))
    return dict(sorted(out.items()))


def reached_pairs(sel, day, w, world, heads, csup, M, sup, fz_heads):
    """The selected reached states -> their reached-state labels, and a status entry for every side / head without one (header)."""
    out = [dict(key=dict(side=int(s), head=h), status='no_eligible_decision', eligible=0) for s, h in A.HEADS if (s, h) not in sel]
    for (s, head), x in sel.items():
        d = x['dec']; fac = A.factory('RICH', heads, csup, M, sup, world)
        ctx = L.bound_fork(world['path'], fac, d['k'], s, R2.episode_ident(day, w['o'][s]))
        if ctx['fingerprint'] != d['fingerprint']:
            out.append(dict(key=x['key'], status='fork_mismatch', eligible=x['eligible'])); continue
        tf = L.teacher_factory(M, sup, world, every=AUDIT_EVERY); t0 = time.time(); res = L.pair_bound(ctx, tf, head)
        row = d['row']; xr = np.asarray([[row[c] for c in T.FEATS]], np.float32)
        preds = {c: float(A.predict(fz_heads[c][(s, head)], c, xr)[0]) for c in A.CANDS}
        tst = [t_.stats() for t_, _ in tf.made]
        out.append(dict(key=x['key'], rank=x['rank'], status=res.get('status'), Y=res.get('Y'), reconciled=res.get('reconciled'), pred=preds, A=d['A'],
                        action=d['action'], fork_fingerprint=ctx['fingerprint'], seconds=time.time() - t0, eligible=x['eligible'],
                        binding_bad=sum(t_['binding_bad'] for t_ in tst), binding_checked=sum(t_['binding_checked'] for t_ in tst)))
    return out


def run_part(k, out=OUT, rec_dir=AE.RECEIPT):
    """One replay partition (header) -> out/parts/part_<k>.jsonl.gz + its record."""
    out = Path(out); (out / 'parts').mkdir(parents=True, exist_ok=True); rp = out / 'parts' / f'part_{k:03d}.json'
    if rp.exists():
        return 'DONE'
    C.require(AE.committed(str(rec_dir / 'fit' / 'fit.json'), A.OUT / 'fit.json'), 'the fit manifest is not committed in HEAD: no outcome before the model hashes')
    plan = json.loads((AE.EVAL_DIR / 'plan.json').read_text()); C.require(C.sha(AE.EVAL_DIR / 'plan.json') == AE.EVAL_PLAN_SHA, 'not the frozen evaluation plan')
    parts = population(plan); day, members = parts[int(k)]
    M, sup, _ = L.teacher_assets(); fz, heads, csup = A.frozen_fit()
    kp = Path(plan['kp_dir']); src = KP.Tape(); G = LS.Guard(600. + 120. * len(members), 120.); mon = LS.RN.Sampler(out / 'parts' / f'part_{k:03d}_status.json', LS.GUARD); mon.start()
    try:
        C.require(LS.identity_now(day, kp, src, AS.guarded_files, LS.hours_tracked) == plan['sample'][day]['inputs'], f'{day}: the inputs differ from the evaluation plan\'s')
        allw = {LS.ckey(LS.wkey(x)): x for x in LS.worlds_all(day, kp, keys=[(x['m'], x['t_ms']) for x in members])}
        mk = sorted({x['m'] for x in members}); fair, depth, hrs, trk = X.real_inputs(day, mk)
        C.require(LS.day_identity(day, kp, src, hrs, trk, AS.guarded_files) == plan['sample'][day]['inputs'], 'real_inputs\' hours / tracked differ from the plan')
        data = src.load(day, mk, src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None; G.check('load')
        rows = []
        for x in members:
            aw = allw[LS.ckey(LS.wkey(x))]; w = dict(m=x['m'], t=aw['t'], o=aw['o']); world, wst = L.world_bind(day, w, prs, fair, depth, X.real_features)
            base = dict(day=day, m=int(x['m']), t_ms=int(x['t_ms']), hour=int(x['hour']), rank=x['rank'], pi=x['pi'], oid0=int(aw['o'][0].oid), oid1=int(aw['o'][1].oid))
            if world is None:
                rows.append(dict(base, status=f'world:{wst}')); G.check('world'); continue
            eps, sel = {}, {}
            for name in A.POLICIES:
                B, Lx, pol, cst, rcd = episode(name, heads, csup, M, sup, day, w, world, record=(name == 'RICH'))
                dec = [{k_: v_ for k_, v_ in d_.items() if k_ != 'row'} for d_ in pol.log]
                eps[name] = dict(metrics(B, Lx, dec), ledger_ok=bool(Lx['status'] == 'ok' and L.reconcile_arm(Lx, B)), binding_bad=pol.bind_bad,
                                 binding_checked=pol.bind_checked, causal={k_: v_ for k_, v_ in cst.items() if k_ not in ('details', 'n')},
                                 calls=len(pol.log), statuses=count([d_['status'] for d_ in pol.log]), actions=count([d_['action'] for d_ in pol.log]),
                                 decisions_sha256=L._h(Lx.get('decisions')), fills_sha256=L._h(Lx.get('fills')), churn=churn(pol.log, B))
                if name == 'RICH':
                    sel = select_reached(rcd.rec, day, x['m']); eps[name]['exposure'] = exposure(pol.log)
                G.check('episode')
            pr = reached_pairs(sel, day, w, world, heads, csup, M, sup, heads)
            for r_ in pr:
                G.check('pair', r_.get('seconds', 0.))
            rows.append(dict(base, status='ok', episodes=eps, reached=pr)); world = None; G.check('world')
        b = LS.gz_bytes(rows); fp = out / 'parts' / f'part_{k:03d}.jsonl.gz'; fp.write_bytes(b)
        rec = dict(version=VERSION, part=int(k), day=day, members=[LS.wkey(x) for x in members], rows=len(rows), sha256=hashlib.sha256(b).hexdigest(),
                   eval_plan_sha256=AE.EVAL_PLAN_SHA, fit_sha256=C.sha(A.OUT / 'fit.json'), sources={n: C.sha(HERE / n) for n in ('pf_adv2.py', 'pf_adv2_replay.py', 'pf_adv2_eval.py')},
                   elapsed_s=time.time() - G.t0, hwm_mib=LS.hwm(), timing=LS.timing(G.log), asof=C.stamp())
        LS.RN.atomic_json(rp, rec); return 'PART_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(out / 'parts' / f'part_{k:03d}_status.json', dict(state='STOP_RESOURCE', part=int(k), latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


def summarize(out=OUT):
    """Every partition -> episodes.csv, reached.csv and replay.json (per policy / day totals and the paired differences against PI0 and
    ENTRY_PA; reached-state prediction errors)."""
    out = Path(out); plan = json.loads((AE.EVAL_DIR / 'plan.json').read_text()); parts = population(plan); ep, rc, st, chn, exq = [], [], {}, [], []
    for k in range(len(parts)):
        rec = json.loads((out / 'parts' / f'part_{k:03d}.json').read_text()); b = (out / 'parts' / f'part_{k:03d}.jsonl.gz').read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256'], f'part {k}: bytes differ from its record')
        for r in LS.gz_rows(b):
            st[r['status']] = st.get(r['status'], 0) + 1
            if r['status'] != 'ok':
                continue
            for name, e in r['episodes'].items():
                ep.append(dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], hour=r['hour'], policy=name, **{k_: v_ for k_, v_ in e.items() if not isinstance(v_, (dict, list))}))
                chn += [dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], policy=name, **c_) for c_ in e.get('churn', [])]
                exq += [dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], policy=name, key=k_, n=v_) for k_, v_ in (e.get('exposure') or {}).items()]
            for x in r['reached']:
                rc.append(dict(day=r['day'], m=r['m'], t_ms=r['t_ms'], side=x['key']['side'] if 'key' in x else None, head=x['key']['head'] if 'key' in x else None,
                               status=x['status'], eligible=x.get('eligible'), Y=x.get('Y'), A=x.get('A'), action=x.get('action'),
                               **{f'pred_{c}': (x.get('pred') or {}).get(c) for c in A.CANDS}))
    E = pd.DataFrame(ep); R = pd.DataFrame(rc); E.to_csv(out / 'episodes.csv', index=False); R.to_csv(out / 'reached.csv', index=False)
    pd.DataFrame(chn).to_csv(out / 'churn.csv', index=False); pd.DataFrame(exq).to_csv(out / 'exposure.csv', index=False)
    tot = E.groupby(['day', 'policy']).agg(worlds=('value', 'size'), value=('value', 'sum'), value_bid=('value_bid', 'sum'), value_ask=('value_ask', 'sum'),
                                          fills=('fills', 'sum'), filled_shares=('filled_shares', 'sum'), generations=('generations', 'sum'),
                                          loss=('loss', 'sum'), severe=('severe', 'sum'), ledger_ok=('ledger_ok', 'all'), binding_bad=('binding_bad', 'sum')).reset_index()
    piv = E.pivot_table(index=['day', 'm', 't_ms'], columns='policy', values='value')
    diffs = [dict(day=d, policy=p, versus=b, n=int(g[[p, b]].dropna().shape[0]), diff_sum=float((g[p] - g[b]).sum()), positive=int(((g[p] - g[b]) > 0).sum()),
                  negative=int(((g[p] - g[b]) < 0).sum())) for d, g in piv.groupby(level=0) for p in A.POLICIES for b in ('PI0', 'ENTRY_PA') if p != b and p in g and b in g]
    js = dict(version=VERSION, statuses=st, totals=tot.to_dict('records'), diffs=diffs, reached=R.groupby(['day', 'head', 'status']).size().astype(int).reset_index().to_dict('records') if len(R) else [],
              asof=C.stamp())
    (out / 'replay.json').write_text(json.dumps(js, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(dict(statuses=st, worlds=int(len(piv))), indent=1))


def selftest():
    import pf_adv2_replay_selftest as ST
    ST.run(sys.modules[__name__])


if __name__ == '__main__':
    a = sys.argv[1:]
    if a == ['selftest']:
        selftest()
    elif a[:1] == ['run'] and len(a) == 2:
        print(run_part(int(a[1])))
    elif a == ['summarize']:
        summarize()
    else:
        sys.exit(__doc__)
