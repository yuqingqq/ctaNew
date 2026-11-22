"""PAIRED-TURNOVER-ATTRIBUTION-v1 (P-2026-005; 2026-11-22, session aea5bfb9). Review d799cb77's ONE next priority: a saved-evidence completion of
LONG-CONTINUATION-PAIRED-VALUE-v1 (frozen 7407bdf1, results 6b806c19) on the same 64 pairs - no replay, label day, model or threshold. Inside
the approved pilot allocation: this follow-up <= 0.04 core-hours plus a 0.02 review reserve (batch <= 0.40, line <= 2.85; no USER decision).
INPUTS: the pilot's day records and per-pair ledgers (hash-bound by the day records; their copies committed in the pilot receipt) and its
 measure pairs / replacements (= its audit). Primary Y, rule, sampling and END unchanged; every pair and status kept.
FIFO LOTS (a declared allocation convention, not causal necessity): every arm's fills (pre- and post-fork) in (time, trade index) order; a
 fill closes opposite-sign open lots first-in-first-out, the rest opens a lot (side, price, quantity, actual acquisition time, arm-local
 gid). The lots open at the fork are CARRIED: their incremental basis is the fork mark M_fork (pre-fork gains / losses are never charged
 again) while their acquisition times are kept; post-fork lots have their fill price as basis.
DECOMPOSITION: Q = matched closing profit (post-fork closes: q (close - basis) for a long lot, q (basis - close) for a short) + residual
 (lots open at END marked at M_END against their basis) - exact (1e-6 c; refused otherwise); Y = dMatched + dResidual. Reported per arm:
 matched shares, profit, profit per share, carried vs new (matched and remaining), quantity-weighted holding time from the ACTUAL acquisition,
 lineage-involved vs other matched profit (a lot's opening or closing fill in the focal lineage).
FOCAL ROUND TRIPS: each post-fork focal-lineage fill's own END value s q (M_END - p) kept apart from its round-trip outcome: closing earlier
 lots (profit), opening lots closed later by any order (profit), or open at END (residual). Per arm: own_end_{negative,positive,zero} x
 round_trip_{profitable,loss,none}; an own-END-negative fill with a profitable opposite close is never an uncompensated loss.
EVENTS: retain vs alternative post-fork fills keyed by execution event (time, trade index, side, price, quantity), never shared gids;
 Y = sum of retain-only event values - sum of alternative-only event values (identity, refused otherwise); counts per pair.
CORRECTIONS (review d799cb77 findings 2-3, from the saved rows): the first replacement delay per arm (REVIEW: the first lineage
 submission; ADMISSION POST arm: the first after the forked POST; ADMISSION WAIT arm: the first at or after the fork whatever its numeric
 gid), attempts and accepted (non-refused) apart; the day signs per kind computed (POSITIVE_EVERY_DAY / NEGATIVE_EVERY_DAY / MIXED).
OUTPUTS: arms / pairs / lots / focal / delays / summary .csv and audit.json (write-once). Descriptive: no winner, no risk rule; pairs overlap
 (repeated counterfactuals of 8 markets), so sums are never portfolio P&L; queue mechanism stays 'unidentified'.
Usage (run_light): pf_ladder_turnover_attribution.py selftest | manifest | run [out]"""
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = 'paired_turnover_attribution_v1'
TOL = 1e-6
LC_RECEIPT = 'predictfun/reviews/ladder_long_continuation_20261122'
LC_OUT = REPO / 'data/predict_fun/research/ladder_long_continuation_v1_20261122'
RECEIPT = 'predictfun/reviews/ladder_turnover_attribution_20261122'
OUT = REPO / 'data/predict_fun/research/ladder_turnover_attribution_v1_20261122'
CAP_FOLLOWUP, REVIEW_RESERVE = 0.04, 0.02
SOURCES = ('pf_ladder_turnover_attribution.py', 'pf_ladder_turnover_attribution_selftest.py')
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'ta_checks.py')
SG = {0: 1., 1: -1.}


def need(cond, msg):
    if not cond:
        raise ValueError(msg)


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def committed(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def sources():
    return {n: sha_b((HERE / n).read_bytes()) for n in SOURCES}


def frozen():
    return bool(all(committed(f'predictfun/research/{n}') == (HERE / n).read_bytes() for n in SOURCES)
                and all(committed(f'{RECEIPT}/{n}') == (REPO / RECEIPT / n).read_bytes() for n in FROZEN_RECEIPT))


# ---------------------------------------------------------------------------------------------------------------------- the accounting
def lots_fifo(pre, post, M0, M1):
    """FIFO over the arm's whole fill history (header) -> (closes, open lots at END). A lot: [sign, qty, price, t_acq, gid, carried, basis];
    a close: dict(q, sign of the lot, basis, open price / time / gid, carried, close price / time / gid, profit)."""
    order = lambda fl: sorted(fl, key=lambda f: (f[0], f[5]))
    lots = []
    for f in order(pre):
        s, q = SG[int(f[2])], float(f[3])
        while q > 1e-12 and lots and lots[0][0] == -s:
            x = min(q, lots[0][1]); lots[0][1] -= x; q -= x
            if lots[0][1] <= 1e-12:
                lots.pop(0)
        if q > 1e-12:
            lots.append([s, q, float(f[4]), float(f[0]), int(f[1]), False, float(f[4])])
    for lot in lots:
        lot[5], lot[6] = True, float(M0)
    closes = []
    for f in order(post):
        s, q, p = SG[int(f[2])], float(f[3]), float(f[4])
        while q > 1e-12 and lots and lots[0][0] == -s:
            lot = lots[0]; x = min(q, lot[1])
            closes.append(dict(q=x, sign=lot[0], basis=lot[6], open_price=lot[2], open_t=lot[3], open_gid=lot[4], carried=lot[5], close_price=p, close_t=float(f[0]),
                               close_gid=int(f[1]), profit=x * (p - lot[6]) * lot[0]))
            lot[1] -= x; q -= x
            if lot[1] <= 1e-12:
                lots.pop(0)
        if q > 1e-12:
            lots.append([s, q, p, float(f[0]), int(f[1]), False, p])
    return closes, lots


def decompose(a):
    """One arm's FIFO decomposition of Q (header); refuses when matched + residual != Q."""
    closes, lots = lots_fifo(a['pre_fills'], a['post_raw'], a['M_fork'], a['mark']); M1 = float(a['mark']); lin = set(int(g) for g in a['lineage'])
    matched = float(sum(c['profit'] for c in closes)); resid = float(sum(l[1] * (M1 - l[6]) * l[0] for l in lots))
    need(abs(matched + resid - float(a['Q'])) <= TOL * max(1., abs(float(a['Q']))), f'the FIFO decomposition does not reconcile Q ({matched} + {resid} vs {a["Q"]})')
    mq = float(sum(c['q'] for c in closes))
    hold = float(sum(c['q'] * (c['close_t'] - c['open_t']) for c in closes) / mq) if mq > 0 else np.nan
    inl = [c for c in closes if c['open_gid'] in lin or c['close_gid'] in lin]
    return dict(matched=matched, residual=resid, matched_q=mq, profit_per_share=matched / mq if mq > 0 else np.nan, hold_ms=hold,
                matched_carried=float(sum(c['profit'] for c in closes if c['carried'])), matched_new=float(sum(c['profit'] for c in closes if not c['carried'])),
                matched_q_carried=float(sum(c['q'] for c in closes if c['carried'])), carried_q=float(sum(c['q'] for c in closes if c['carried'])) + float(sum(l[1] for l in lots if l[5])),
                residual_carried=float(sum(l[1] * (M1 - l[6]) * l[0] for l in lots if l[5])), residual_new=float(sum(l[1] * (M1 - l[6]) * l[0] for l in lots if not l[5])),
                remaining_q=float(sum(l[1] * l[0] for l in lots)), remaining_q_carried=float(sum(l[1] * l[0] for l in lots if l[5])),
                matched_lineage=float(sum(c['profit'] for c in inl)), matched_other=matched - float(sum(c['profit'] for c in inl))), closes, lots


def focal_classes(a, closes, lots):
    """Per post-fork focal-lineage fill group (one gid at one time: indivisible): its own END value vs its round trip (header) -> counts per class and sums."""
    M1 = float(a['mark']); lin = set(int(g) for g in a['lineage']); out = {}; grp = {}
    for f in a['post_raw']:
        if int(f[1]) in lin:
            grp.setdefault((float(f[0]), int(f[1])), []).append(f)
    for (t, g), fs in sorted(grp.items()):
        own = sum(SG[int(f[2])] * float(f[3]) * (M1 - float(f[4])) for f in fs)
        rt = sum(c['profit'] for c in closes if (c['close_t'] == t and c['close_gid'] == g) or (c['open_t'] == t and c['open_gid'] == g and not c['carried']))
        has = any((c['close_t'] == t and c['close_gid'] == g) or (c['open_t'] == t and c['open_gid'] == g and not c['carried']) for c in closes)
        k = ('own_end_negative' if own < -TOL else ('own_end_positive' if own > TOL else 'own_end_zero')) + '_' + \
            (('round_trip_profitable' if rt > TOL else ('round_trip_loss' if rt < -TOL else 'round_trip_flat')) if has else 'round_trip_none')
        d = out.setdefault(k, dict(n=0, own=0., round_trip=0.)); d['n'] += 1; d['own'] += own; d['round_trip'] += rt
    return out


def event_key(f):
    return (round(float(f[0]), 6), int(f[5]), int(f[2]), round(float(f[4]), 9), round(float(f[3]), 9))


def event_diff(R, A):
    """Retain-only and alternative-only post-fork execution events (multisets by event key) and their END values (header)."""
    from collections import Counter
    v = lambda f, M1: SG[int(f[2])] * float(f[3]) * (M1 - float(f[4]))
    cr, ca = Counter(event_key(f) for f in R['post_raw']), Counter(event_key(f) for f in A['post_raw']); only_r, only_a = cr - ca, ca - cr
    byk = lambda fl: {event_key(f): f for f in fl}
    kr, ka = byk(R['post_raw']), byk(A['post_raw'])
    vr = float(sum(n * v(kr[k], R['mark']) for k, n in only_r.items())); va = float(sum(n * v(ka[k], A['mark']) for k, n in only_a.items()))
    return dict(events_common=int(sum((cr & ca).values())), events_retain_only=int(sum(only_r.values())), events_alternative_only=int(sum(only_a.values())),
                value_retain_only=vr, value_alternative_only=va, Y_events=vr - va)


def first_delays(rows, kind, arm, t):
    """(first attempt, first accepted) replacement delays of one arm from its saved replacement rows (header)."""
    rows = sorted(rows, key=lambda r: (float(r['submit']), int(r['gid'])))
    if kind == 'ADMISSION' and arm == 'retain':
        rows = [r for r in rows if float(r['submit']) > t]
    else:
        rows = [r for r in rows if float(r['submit']) >= t]
    att = next((float(r['submit']) - t for r in rows), np.nan); acc = next((float(r['submit']) - t for r in rows if not bool(r['refused'])), np.nan)
    return att, acc


def day_signs(T_, col='Y'):
    """Per kind: POSITIVE_EVERY_DAY / NEGATIVE_EVERY_DAY / MIXED of the per-day sums of col (computed, never printed by hand)."""
    out = {}
    for k, g in T_.groupby('kind'):
        s = g.groupby('day')[col].sum()
        out[k] = dict(by_day={d: float(v) for d, v in s.items()}, sign='POSITIVE_EVERY_DAY' if (s > TOL).all() else ('NEGATIVE_EVERY_DAY' if (s < -TOL).all() else 'MIXED'))
    return out


# ---------------------------------------------------------------------------------------------------------------------- inputs, manifest, run
def lc_inputs(gm_lc=None):
    """The pilot's day records (sha256 of each, their ledgers' bytes checked) and its measure copies' audit."""
    gm_lc = gm_lc or json.loads((REPO / LC_RECEIPT / 'manifest.json').read_text()); recs = {}
    for part, days in gm_lc['parts'].items():
        for day in days:
            p_ = LC_OUT / part / f'{day}.json'; r = json.loads(p_.read_text()); need(sha_b((LC_OUT / part / f'{day}_ledgers.jsonl.gz').read_bytes()) == r['sha256']['ledgers'],
                                                                                   f'{day}: ledger bytes differ from the record')
            recs[f'{part}/{day}'] = sha_b(p_.read_bytes())
    au = json.loads((LC_OUT / 'measure' / 'audit.json').read_text())
    need(all(au['outputs'][n] == sha_b((LC_OUT / 'measure' / n).read_bytes()) for n in ('pairs.csv', 'replacements.csv', 'arms.csv')), 'the pilot measure copies differ from its audit')
    return recs, sha_b((LC_OUT / 'measure' / 'audit.json').read_bytes())


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); need(not path.exists(), 'the manifest exists (write-once)')
    gm_lc = json.loads((REPO / LC_RECEIPT / 'manifest.json').read_text()); recs, ash = lc_inputs(gm_lc)
    man = dict(version=VERSION, review='d799cb77', pilot_freeze='7407bdf1', pilot_results='6b806c19', lc_records=recs, lc_audit_sha256=ash, sources=sources(),
               caps=dict(followup_core_hours=CAP_FOLLOWUP, review_reserve=REVIEW_RESERVE, batch_core_hours=0.40, line_ceiling=2.85),
               line_receipts=gm_lc['line_receipts'] + [LC_RECEIPT.split('/')[-1]], line_reviews=gm_lc['line_reviews'] + ['20261122_long_continuation_review'],
               prior_unlogged_s=gm_lc['caps']['prior_unlogged_s'], asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def ledgers(gm_lc):
    out = []
    for part, days in gm_lc['parts'].items():
        for day in days:
            out += [json.loads(x) for x in gzip.decompress((LC_OUT / part / f'{day}_ledgers.jsonl.gz').read_bytes()).decode().splitlines()]
    return [x for x in out if x['kind'] != 'BASE']


def levels(T_, fn):
    rows = []
    for ks in (('kind', 'day', 'side'), ('kind', 'day'), ('kind', 'side'), ('kind',), ()):
        for k, g in (T_.groupby(list(ks)) if ks else [((), T_)]):
            k = k if isinstance(k, tuple) else (k,); d = dict(zip(ks, k)); d.setdefault('kind', 'all'); d.setdefault('day', 'all'); d.setdefault('side', 'all'); d['side'] = str(d['side'])
            rows.append(fn(g, d))
    return pd.DataFrame(rows)


def srow(g, d):
    d.update(pairs=int(len(g)), retain_better=int((g.Y > TOL).sum()), alternative_better=int((g.Y < -TOL).sum()), tie=int((g.Y.abs() <= TOL).sum()))
    for c in ('Y', 'dMatched', 'dResidual', 'dMatched_lineage', 'dMatched_other', 'Y_events'):
        d[f'{c}_sum'] = float(g[c].sum()); d[f'{c}_abs_sum'] = float(g[c].abs().sum())
    a = g.Y.abs().sort_values(ascending=False); d['top5_share_abs_Y'] = float(a.head(5).sum() / a.sum()) if a.sum() > TOL else np.nan
    d.update({f'Y_q{int(q * 100)}': float(g.Y.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
    d['turnover_share_abs'] = float(g.dMatched.abs().sum() / (g.dMatched.abs().sum() + g.dResidual.abs().sum())) if (g.dMatched.abs().sum() + g.dResidual.abs().sum()) > TOL else np.nan
    return d


def run(out=None, require_frozen=True, man_path=None):
    """The attribution (header) -> OUTPUTS and audit.json in out, write-once."""
    if require_frozen:
        need(frozen(), 'the attribution builder / selftest / declaration / manifest / checker are not the committed ones')
    mp = Path(man_path or REPO / RECEIPT / 'manifest.json'); gm = json.loads(mp.read_text()); need(gm['sources'] == sources(), 'a source differs from the frozen one')
    od = Path(out or OUT); need(not (od / 'audit.json').exists(), 'the audit exists (write-once)'); od.mkdir(parents=True, exist_ok=True); t0 = time.time()
    gm_lc = json.loads((REPO / LC_RECEIPT / 'manifest.json').read_text()); recs, ash = lc_inputs(gm_lc); need(recs == gm['lc_records'] and ash == gm['lc_audit_sha256'], 'a pilot input differs from the manifest')
    Pp = pd.read_csv(LC_OUT / 'measure' / 'pairs.csv', dtype={'day': str}); Rp = pd.read_csv(LC_OUT / 'measure' / 'replacements.csv', dtype={'day': str}); st = {r['key']: r for r in Pp.to_dict('records')}
    arms, pairs, lot_rows, foc, dl = [], [], [], [], []
    for x in ledgers(gm_lc):
        p_ = st[x['key']]; res = {}
        for name in ('retain', 'alternative'):
            a = x[name]; d, closes, lots = decompose(a); fc = focal_classes(a, closes, lots); res[name] = d
            rows = Rp[(Rp.key == x['key']) & (Rp.arm == name)].to_dict('records'); att, acc = first_delays(rows, x['kind'], name, float(p_['t']))
            arms.append(dict(key=x['key'], day=x['day'], m=x['m'], kind=x['kind'], side=p_['side'], arm=name, Q=float(a['Q']), **d, first_attempt_delay=att, first_accepted_delay=acc))
            lot_rows += [dict(key=x['key'], arm=name, **c) for c in closes]; foc += [dict(key=x['key'], kind=x['kind'], arm=name, cls=k, **v) for k, v in fc.items()]
            dl.append(dict(key=x['key'], day=x['day'], m=x['m'], kind=x['kind'], arm=name, first_attempt_delay=att, first_accepted_delay=acc))
        ev = event_diff(x['retain'], x['alternative']); Y = float(x['retain']['Q'] - x['alternative']['Q'])
        need(abs(Y - float(p_['Y'])) <= TOL * max(1., abs(Y)), f'{x["key"]}: Y differs from the pilot'); need(abs(ev['Y_events'] - Y) <= 1e-6 * max(1., abs(Y)), f'{x["key"]}: the event identity fails')
        pairs.append(dict(key=x['key'], day=x['day'], m=x['m'], kind=x['kind'], side=p_['side'], Y=Y, dMatched=res['retain']['matched'] - res['alternative']['matched'],
                          dResidual=res['retain']['residual'] - res['alternative']['residual'], dMatched_lineage=res['retain']['matched_lineage'] - res['alternative']['matched_lineage'],
                          dMatched_other=res['retain']['matched_other'] - res['alternative']['matched_other'], **ev))
    A_, P_, L_, F_, D_ = (pd.DataFrame(v) for v in (arms, pairs, lot_rows, foc, dl)); need(len(P_) == len(Pp) == 64, 'the pairs are not the 64 pilot pairs')
    S_ = levels(P_, srow); FS = F_.groupby(['kind', 'arm', 'cls'])[['n', 'own', 'round_trip']].sum().reset_index()
    tabs = dict(arms=A_, pairs=P_, lots=L_, focal=FS, delays=D_, summary=S_)
    for n, df in tabs.items():
        df.to_csv(od / f'{n}.csv', index=False)
    Ap = pd.read_csv(LC_OUT / 'measure' / 'arms.csv', dtype={'day': str})[['key', 'arm', 'first_rep_delay']]; J = A_.merge(Ap, on=['key', 'arm'], how='left')
    diff = ~(((J.first_attempt_delay - J.first_rep_delay).abs() <= 1e-9) | (J.first_attempt_delay.isna() & J.first_rep_delay.isna()))
    alt = A_[(A_.arm == 'alternative') & A_.first_attempt_delay.notna()]
    pr = dict(day_signs=day_signs(P_), Y_unchanged=True, decomposition_reconciles=True, event_identity=True, delays_differ_from_pilot=int(diff.sum()),
              delays_differ_by=J[diff].groupby(['kind', 'arm']).size().rename('n').reset_index().to_dict('records'), alternative_mean_attempt_delay=float(alt.first_attempt_delay.mean()),
              alternative_arms_with_replacement=int(len(alt)))
    (od / 'audit.json').write_text(json.dumps(dict(version=VERSION, asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), predicates=pr, pairs=int(len(P_)), arms=int(len(A_)),
                                                   outputs={f'{n}.csv': sha_b((od / f'{n}.csv').read_bytes()) for n in tabs}, sources=sources(), manifest_sha256=sha_b(mp.read_bytes()),
                                                   elapsed_s=time.time() - t0, note='descriptive FIFO attribution; overlapping pairs, never portfolio P&L; queue mechanism unidentified'),
                                              indent=1, default=str) + '\n')
    print(json.dumps(dict(pairs=len(P_), day_signs={k: v['sign'] for k, v in pr['day_signs'].items()}), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_turnover_attribution_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1500])
    elif cmd == 'run':
        run(*(sys.argv[2:3] or [None]))
    else:
        sys.exit('usage: pf_ladder_turnover_attribution.py selftest | manifest | run [out]')
