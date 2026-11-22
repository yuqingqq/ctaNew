"""QUEUE-PRICE-STATE-AUDIT-v1 (P-2026-005; 2026-11-22, session aea5bfb9). Review 72a36de9 (target 75e9531d) accepted ARRIVAL-QUEUE-DECOMPOSITION-v1
with four nonblocking corrections and named ONE next priority: separate the queue estimate from the current price-level occupancy and distance on
the same neutral states, saved data only (<= 0.020 core-hours; no new threshold, model, day, venue decision input or replay; the stopped
POSITIVE-PRESSURE-GUARD-v1 stays stopped). Consumed Oct 24-27 development days; exploratory description, no skill or promotion claim.
INPUTS (hash-bound): the ARRIVAL-QUEUE receipt's committed states / intervals copies (raw sha256 = its committed audit.json); the four original
 LADDER_REST feature parquets (sha256 = the committed day records' sha256['feat']; the data-dir records = the committed copies).
JOIN (join): every state to exactly one feature row of kind live at the same (day, m, gid, t, side) (refused otherwise, never nearest); the
 row's fp_queue / fp_remaining / o_depth must equal the state's Q / R / o_depth exactly (refused otherwise).
GROUPS (declared before any economics; state_table): price_state from o_depth (ticks behind the received best): inside_best (< 0), at_touch
 (= 0), behind_best (> 0), unknown (missing); display from fp_shown (the external size displayed at the order's price in the last received
 frame): shown_zero (= 0), shown_positive (> 0), shown_invalid (< 0), shown_unknown (missing); the queue band (front / within_R / beyond_R /
 unknown / invalid_R) re-derived with ARRIVAL-QUEUE's rule and required to equal its saved band. Q <= shown is NOT imposed (estimated
 progress and later displayed arrivals differ). The exact depth, rung and price are exported.
VIEWS: price_display (price_state | display); price_display_queue (| queue band; the PRINCIPAL cells are at_touch|shown_positive|<band>,
 compared by side / day); behind_depth_queue (behind_best & shown_positive: depth=<exact o_depth>|<band>; every other state 'outside', kept);
 principal_pressure (the principal states: <band>|<the frozen Binance band ARRIVAL-QUEUE saved>; others 'outside') - a support census.
INTERVALS: ARRIVAL-QUEUE's I0..I3 rows unchanged (arrival after the cancel effect; its fixed common cohort = every saved horizon valid) and
 their union per state (values missing when any interval's is; status = the end cause when any interval is at risk, else not_at_risk).
CELLS (cells; one row per view x band x interval (and the union) x day (and all) x side (and all) x cohort (native / common)): all-state counts
 named all_* (opportunities, at-risk, filled states, fills, shares, partial, filled-out, every follow-up status, the median time to END at
 the decision and the states with < 30 s to END) and the any-fill rate all_filled / all_at_risk; censored_from_common = all opportunities -
 common-valid states; per horizon h in H: valid states, valid at-risk, valid filled, valid shares, the valid any-fill rate (valid filled /
 valid at-risk), net, state positive / negative, fill positive / negative, per opportunity (valid states), per share (valid shares) and
 sparse (valid filled < SPARSE_MIN, i.e. the declared minimum on the filled states that supply the value).
Usage (run_light): pf_ladder_queue_price.py selftest | manifest | run [out]"""
import gzip
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = 'queue_price_state_audit_v1'
AQ_OUT = 'predictfun/reviews/ladder_arrival_queue_20261122/outputs'
REST_DAYS = 'predictfun/reviews/ladder_rest_20261121/outputs/days'
FEAT_DIR = REPO / 'data/predict_fun/research/ladder_rest_v1_20261121/days'
DAYS = ('20261024', '20261025', '20261026', '20261027')
H = (0, 1000, 5000, 30000)
INTERVALS = ('I0', 'I1', 'I2', 'I3')
SPARSE_MIN, T30 = 20, 30000.
FOLLOW = ('not_at_risk', 'at_risk_full', 'removed_in_interval', 'filled_out_in_interval', 'end_censored')
CAUSE_STATUS = dict(removal='removed_in_interval', full_fill='filled_out_in_interval', end='end_censored')
VIEWS = ('price_display', 'price_display_queue', 'behind_depth_queue', 'principal_pressure')
PRINCIPAL = ('at_touch', 'shown_positive')
KEY = ['day', 'm', 'gid', 't', 'side']
FEAT_COLS = ['day', 'm', 'gid', 'kind', 't', 'side', 'fp_queue', 'fp_remaining', 'fp_shown', 'o_depth', 't_to_end', 'rung', 'price', 'fp_size']
RECEIPT = 'predictfun/reviews/ladder_queue_price_20261122'
OUT = REPO / 'data/predict_fun/research/ladder_queue_price_v1_20261122'
SOURCES = ('pf_ladder_queue_price.py', 'pf_ladder_queue_price_selftest.py')
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_queue_price.sh')
NAN = float('nan')


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


def fin(x):
    return x is not None and bool(np.isfinite(float(x)))


# ---------------------------------------------------------------------------------------------------------------------- labels
def price_state(depth):
    if not fin(depth):
        return 'unknown'
    return 'inside_best' if depth < 0 else ('at_touch' if depth == 0 else 'behind_best')


def display(shown):
    if not fin(shown):
        return 'shown_unknown'
    return 'shown_zero' if shown == 0 else ('shown_positive' if shown > 0 else 'shown_invalid')


def queue_band(Q, R):
    """ARRIVAL-QUEUE's bins; a missing or negative Q is never the front."""
    if not fin(R) or R <= 0:
        return 'invalid_R'
    if not fin(Q) or Q < 0:
        return 'unknown'
    if Q == 0:
        return 'front'
    return 'within_R' if Q <= R else 'beyond_R'


# ---------------------------------------------------------------------------------------------------------------------- inputs
def load_inputs():
    """The hash-bound inputs -> (input hashes, states, intervals, live feature rows)."""
    ab = (REPO / AQ_OUT / 'audit.json').read_bytes(); need(committed(f'{AQ_OUT}/audit.json') == ab, 'the ARRIVAL-QUEUE audit.json is not the committed one')
    audit = json.loads(ab); tabs, hashes = {}, dict(aq_audit=sha_b(ab))
    for n in ('states.csv', 'intervals.csv'):
        gzb = (REPO / AQ_OUT / f'{n}.gz').read_bytes(); need(committed(f'{AQ_OUT}/{n}.gz') == gzb, f'{n}.gz is not the committed copy')
        raw = gzip.decompress(gzb); need(sha_b(raw) == audit['outputs'][n], f'{n}: the copy differs from its audit.json hash'); hashes[n] = sha_b(raw)
        tabs[n] = pd.read_csv(io.BytesIO(raw), dtype={'day': str}, float_precision='round_trip')
    feats = []
    for d in DAYS:
        rb = committed(f'{REST_DAYS}/{d}.json'); need(rb is not None and (FEAT_DIR / f'{d}.json').read_bytes() == rb, f'{d}: the day record is not the committed copy')
        fb = (FEAT_DIR / f'{d}_features.parquet').read_bytes(); need(sha_b(fb) == json.loads(rb)['sha256']['feat'], f'{d}: the feature parquet differs from its record')
        f = pd.read_parquet(io.BytesIO(fb), columns=FEAT_COLS); f['day'] = f.day.astype(str); feats.append(f[f.kind == 'live']); hashes[f'feat_{d}'] = sha_b(fb)
    return hashes, tabs['states.csv'], tabs['intervals.csv'], pd.concat(feats, ignore_index=True)


def join(S, F):
    """Each state to its one live feature row at the exact key; the shared fields must agree exactly."""
    need(not F.duplicated(KEY).any(), 'a duplicated feature key')
    J = S.merge(F[KEY + ['fp_queue', 'fp_remaining', 'fp_shown', 'o_depth', 't_to_end', 'fp_size']].rename(columns={'o_depth': 'o_depth_f'}), on=KEY, how='left', indicator=True)
    need(len(J) == len(S) and bool((J._merge == 'both').all()), 'a state without its feature row')
    same = lambda a, b: bool(((a == b) | (a.isna() & b.isna())).all())
    need(same(J.Q, J.fp_queue) and same(J.R, J.fp_remaining) and same(J.o_depth, J.o_depth_f), 'the feature row disagrees with the state')
    return J.drop(columns=['_merge', 'fp_queue', 'fp_remaining', 'o_depth_f'])


def state_table(J):
    """The declared groups and the four view labels per state."""
    S_ = J.copy(); S_['price_state'] = S_.o_depth.map(price_state); S_['display'] = S_.fp_shown.map(display)
    qb = [queue_band(q, r) for q, r in zip(S_.Q, S_.R)]; need(qb == list(S_.queue_band), 'a queue band differs from the saved one')
    pr = (S_.price_state == PRINCIPAL[0]) & (S_.display == PRINCIPAL[1]); bh = (S_.price_state == 'behind_best') & (S_.display == 'shown_positive')
    S_['principal'] = pr; S_['price_display'] = S_.price_state + '|' + S_.display; S_['price_display_queue'] = S_.price_display + '|' + S_.queue_band
    S_['behind_depth_queue'] = np.where(bh, 'depth=' + S_.o_depth.astype(str) + '|' + S_.queue_band, 'outside')
    S_['principal_pressure'] = np.where(pr, S_.queue_band + '|' + S_.bn_band, 'outside'); return S_


# ---------------------------------------------------------------------------------------------------------------------- tables
def union(X_):
    """Per state the union of its interval rows (values missing when any interval's is; status = the end cause when any interval is at risk)."""
    nan_sum = lambda x: x.sum() if x.notna().all() else NAN
    keep = [c for c in X_.columns if c not in ('interval', 'follow_up', 'at_risk_ms', 'fills', 'shares', 'filled_out', 'vEND', 'common', 'cause')
            and not any(c == f'{p}{h}' for p in ('v', 'ok', 'fpos', 'fneg') for h in (0, 50, 100, 1000, 5000, 30000))]
    agg = {c: (c, 'first') for c in keep if c not in ('day', 'm', 'gid')}
    agg.update(cause=('cause', 'first'), fills=('fills', 'sum'), shares=('shares', 'sum'), filled_out=('filled_out', 'any'), vEND=('vEND', 'sum'), common=('common', 'all'),
               _r=('_r', 'any'))
    for h in H:
        agg.update({f'v{h}': (f'v{h}', nan_sum), f'ok{h}': (f'ok{h}', 'all'), f'fpos{h}': (f'fpos{h}', 'sum'), f'fneg{h}': (f'fneg{h}', 'sum')})
    U = X_.assign(_r=X_.follow_up != 'not_at_risk').groupby(['day', 'm', 'gid'], as_index=False, sort=True).agg(**agg)
    U['follow_up'] = np.where(U._r, U.cause.map(CAUSE_STATUS), 'not_at_risk'); U['interval'] = 'union'; return U.drop(columns='_r')


def cells(S_, I_):
    """The cells (header), one row per view x band x interval x day x side x cohort."""
    need(len(I_) == len(INTERVALS) * len(S_) and bool((I_.groupby(['day', 'm', 'gid']).interval.nunique() == len(INTERVALS)).all()), 'an interval row is missing or extra')
    X_ = I_.merge(S_[['day', 'm', 'gid', 'side', 't_to_end'] + list(VIEWS)], on=['day', 'm', 'gid'], how='left', validate='many_to_one'); need(X_.side.notna().all(), 'an interval row without its state')
    F = pd.concat([X_, union(X_)], ignore_index=True); F['side'] = F.side.astype(int).astype(str); F['risk'] = F.follow_up != 'not_at_risk'; F['filled'] = F.fills > 0
    F['filled_out'] = F.filled_out.astype(bool); F['partial'] = F.filled & ~F.filled_out; F['lt30'] = F.t_to_end < T30
    new = {f'all_n_{st}': F.follow_up == st for st in FOLLOW}; mcols = []
    for coh in ('native', 'common'):
        for h in H:
            ok = (F[f'ok{h}'] if coh == 'native' else F['common']).astype(bool); v = F[f'v{h}'].where(ok, 0.).fillna(0.); k = f'{coh}__{h}__'
            cols = {k + 'valid': ok.astype(int), k + 'valid_risk': (ok & F.risk).astype(int), k + 'valid_filled': (ok & F.filled).astype(int), k + 'valid_shares': F.shares.where(ok, 0.),
                    k + 'net': v, k + 'state_positive': v.clip(lower=0.), k + 'state_negative': v.clip(upper=0.), k + 'fill_positive': F[f'fpos{h}'].where(ok, 0.),
                    k + 'fill_negative': F[f'fneg{h}'].where(ok, 0.)}
            new.update(cols); mcols += list(cols)
    F = pd.concat([F, pd.DataFrame(new, index=F.index)], axis=1)
    FA = pd.concat([F, F.assign(day='all'), F.assign(side='all'), F.assign(day='all', side='all')], ignore_index=True)
    base = dict(all_opportunities=('gid', 'size'), all_at_risk=('risk', 'sum'), all_filled=('filled', 'sum'), all_fills=('fills', 'sum'), all_shares=('shares', 'sum'),
                all_partial=('partial', 'sum'), all_filled_out=('filled_out', 'sum'), all_t_to_end_median=('t_to_end', 'median'), all_t_to_end_lt30s=('lt30', 'sum'),
                **{f'all_n_{st}': (f'all_n_{st}', 'sum') for st in FOLLOW})
    out = []
    for view in VIEWS:
        G = FA.groupby([view, 'interval', 'day', 'side']).agg(**base, **{c: (c, 'sum') for c in mcols}).reset_index().rename(columns={view: 'band'})
        ar = G.all_at_risk
        for coh in ('native', 'common'):
            D_ = {'view': view, **{c: G[c] for c in ['band', 'interval', 'day', 'side'] + list(base)}, 'all_any_fill_rate': np.where(ar > 0, G.all_filled / ar.where(ar > 0, 1), NAN),
                  'censored_from_common': G.all_opportunities - G['common__0__valid'], 'cohort': coh}
            for h in H:
                k = f'{coh}__{h}__'; n, q, vr, vf = G[k + 'valid'], G[k + 'valid_shares'], G[k + 'valid_risk'], G[k + 'valid_filled']
                for x in ('valid', 'valid_risk', 'valid_filled', 'valid_shares', 'net', 'state_positive', 'state_negative', 'fill_positive', 'fill_negative'):
                    D_[f'h{h}_{x}'] = G[k + x]
                D_[f'h{h}_any_fill_rate'] = np.where(vr > 0, vf / vr.where(vr > 0, 1), NAN); D_[f'h{h}_per_opportunity'] = np.where(n > 0, G[k + 'net'] / n.where(n > 0, 1), NAN)
                D_[f'h{h}_per_share'] = np.where(q > 0, G[k + 'net'] / q.where(q > 0, 1.), NAN); D_[f'h{h}_sparse'] = vf < SPARSE_MIN
            out.append(pd.DataFrame(D_))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------------- manifest and the run
STATE_COLS = ['day', 'm', 'gid', 't', 'side', 'rung', 'price', 'size', 'o_depth', 'fp_shown', 'Q', 'R', 'Q_over_R', 'o_q_rel', 't_to_end', 'queue_band', 'bn_band', 'pf_band',
              'price_state', 'display', 'principal'] + list(VIEWS) + ['first_status', 'venue_status', 'client_minus_venue']


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); need(not path.exists(), 'the manifest exists (write-once)'); hashes, S, _, F = load_inputs()
    man = dict(version=VERSION, review='72a36de984fa2ace2b626bea6e8b6db5682fcb16', target_reviewed='75e9531d21610a3b16f5597153299bc92c210a3a', inputs=hashes, days=list(DAYS),
               horizons_ms=list(H), intervals=list(INTERVALS), sparse_min_filled=SPARSE_MIN, views=list(VIEWS), principal=list(PRINCIPAL), follow_up=list(FOLLOW),
               states=int(len(S)), live_feature_rows=int(len(F)), sources=sources(), cap_core_hours=0.020, line_cap_core_hours=2.0, asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def run(out=None, require_frozen=True, man_path=None):
    """The audit (header) -> states.csv, cells.csv, audit.json, write-once."""
    if require_frozen:
        need(frozen(), 'the builder / selftest / declaration / manifest / launcher are not the committed ones')
    mp = Path(man_path or REPO / RECEIPT / 'manifest.json'); gm = json.loads(mp.read_text()); need(gm['sources'] == sources(), 'a source differs from the frozen one')
    od = Path(out or OUT); need(not (od / 'audit.json').exists(), 'the audit exists (write-once)'); od.mkdir(parents=True, exist_ok=True); t0 = time.time()
    hashes, S, I_, F = load_inputs(); need(hashes == gm['inputs'], 'an input differs from the manifest')
    S_ = state_table(join(S, F)); T = cells(S_, I_)
    S_[STATE_COLS].to_csv(od / 'states.csv', index=False); T.to_csv(od / 'cells.csv', index=False)
    summ = dict(states=int(len(S_)), price_display=S_.price_display.value_counts().to_dict(), price_display_queue=S_.price_display_queue.value_counts().to_dict(),
                principal=int(S_.principal.sum()), behind_depth_queue=S_.behind_depth_queue.value_counts().to_dict(), principal_pressure=S_.principal_pressure.value_counts().to_dict(),
                q_above_shown=int((S_.Q > S_.fp_shown).sum()), cell_rows=int(len(T)))
    out_h = {n: sha_b((od / n).read_bytes()) for n in ('states.csv', 'cells.csv')}
    (od / 'audit.json').write_text(json.dumps(dict(version=VERSION, asof=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), summary=summ, outputs=out_h, inputs=hashes, sources=sources(),
                                                   manifest_sha256=sha_b(mp.read_bytes()), elapsed_s=time.time() - t0,
                                                   note='exploratory saved-data description; no rule, threshold or skill claim'), indent=1, default=str) + '\n')
    print(json.dumps(dict(states=summ['states'], principal=summ['principal'], cell_rows=summ['cell_rows'], elapsed_s=round(time.time() - t0, 2)), indent=1))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_queue_price_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1500])
    elif cmd == 'run':
        run(*(sys.argv[2:3] or [None]))
    else:
        sys.exit('usage: pf_ladder_queue_price.py selftest | manifest | run [out]')
