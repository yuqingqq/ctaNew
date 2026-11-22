"""INDEPENDENT CHECKS of FEASIBLE-CANCEL-HEADROOM-v1 (declaration.md; session aea5bfb9, 2026-11-22). Every generation is recomputed from the ASYNC
receipt's hash-bound records with this checker's own code (time-ordered fill lists, a pointer over the sorted callbacks, explicit suffix sums);
the module is not imported. Families:
 A identity: the frozen files committed; the ASYNC records = the manifest (gzip / npz = their records); the committed ASYNC arms copy = its audit;
   the outputs and the projection = the audit's hashes; generations_core = its columns of generations;
 B membership: generations.csv = every order of both arms on the 19 markets once; fills once; panel rows = live KEEP callbacks;
 E end causes and queue fields from the closures, open orders and queue records (unknown = NaN);
 O anchors: status, both cuts, classes, prefix / suffix G / L, retained +1 s and END, unvalued counts, the whole-generation class;
 Q sequences: the flags, their cuts, G / L and classes;  H horizons: the oracle suffix's 5 s / 30 s / END parts and common cohort;
 R reproduction: the unchanged path = the ASYNC receipt's per-market eligible +1 s and END;
 S tables: summary.csv (the cited totals, class counts / losses / shares, concentration, horizon signs, readings), classes.csv, sequences.csv;
 K panel: panel_summary.csv per class (rows, generations, G, L) from this checker's own per-callback suffixes;
 C casebook: the 3 smallest sha256(arm|day|m|gid) per category and arm;  P readings per arm and day and pooled;
 I resources: this round (logs + the declared unlogged allowance) <= 0.05 core-hours, peak < 1,536 MiB, line + round + 0.02 reserve <= 2.50;
 X the Codex review's independent first-anchor classification (c72f0d37; its committed generations, sha256 = its SHA256SUMS): per generation
   the class (its GOOD_ONLY / NEAR_ZERO / NO_SUFFIX_FILLS split of good_only_or_no_fill), suffix G / L, prefix G / L, whole +1 s and END, the
   end cause and the anchor.
falsify(): nineteen corruptions (a known-bad classification among them), each refused by its family while the clean evidence passes.
Usage: ch_checks.py [out_dir]"""
import glob
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('/home/yuqing/ctaNew'); R = REPO / 'predictfun/reviews/ladder_cancel_headroom_20261122'; D = REPO / 'data/predict_fun/research/ladder_cancel_headroom_v1_20261122'
AF = REPO / 'data/predict_fun/research/ladder_async_fast_v1_20261122'; AFR = 'predictfun/reviews/ladder_async_fast_20261122'; ARMS = ('CURRENT_ASYNC_FAST10', 'FRONT_ASYNC_FAST10')
EFF, TOL, EXPECT_MARKETS = 10., 1e-6, 19
CLS = ('incomplete_valuation', 'pure_observed_loss', 'compensated_mixed', 'uncompensated_mixed', 'mixed_tie', 'good_only_or_no_fill')
COMMITTED = ['predictfun/research/pf_ladder_cancel_headroom.py', 'predictfun/research/pf_ladder_cancel_headroom_selftest.py', 'predictfun/research/pf_ladder_cancel_headroom_mutants.py'] + \
    [f'predictfun/reviews/ladder_cancel_headroom_20261122/{n}' for n in ('declaration.md', 'manifest.json', 'launch_cancel_headroom.sh', 'ch_checks.py', 'tools/async_identity.py',
                                                                    'tools/read_headroom.py', 'tools/rehearse_checks.py', 'logs/counted_codex_queue_fill_classes.log')]
CODEX = 'predictfun/reviews/automatic/20261122_queue_fill_classes'
CMAP = dict(NO_OPPORTUNITY='no_feasible_callback', INCOMPLETE='incomplete_valuation', PURE_BAD='pure_observed_loss', MIXED_COMPENSATED='compensated_mixed',
            MIXED_UNCOMPENSATED='uncompensated_mixed', MIXED_TIE='mixed_tie', NO_SUFFIX_FILLS='good_only_or_no_fill', GOOD_ONLY='good_only_or_no_fill', NEAR_ZERO='good_only_or_no_fill')
OUTS = ('generations.csv', 'generations_core.csv', 'summary.csv', 'classes.csv', 'markets.csv', 'sequences.csv', 'panel_summary.csv', 'cases.csv', 'panel.csv.gz')
RL = re.compile(r'^(\S+) tag=(\S+) rc=(\d+) peak_rss_mib=(\d+) wall_s=([\d.]+) cpu_user_s=([\d.]+)')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True); return r.stdout if r.returncode == 0 else None


def close(a, b, tol=1e-6):
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or (not math.isnan(a) and not math.isnan(b) and abs(a - b) <= tol * max(1., abs(a), abs(b)))


def same(a, b):
    """Two columns equal: text / booleans exactly (NaN / None as ''), numbers within 1e-6 relative (NaN = NaN)."""
    a, b = pd.Series(list(a)), pd.Series(list(b))
    if len(a) != len(b):
        return False
    num = lambda s: pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if not (num(a) and num(b)):
        txt = lambda s: ['' if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v) for v in s.tolist()]
        return txt(a) == txt(b)
    x, y = a.astype(float).values, b.astype(float).values; nn = np.isnan(x) & np.isnan(y)
    with np.errstate(invalid='ignore'):
        return bool(np.all(nn | (np.abs(x - y) <= 1e-6 * np.maximum(1., np.maximum(np.abs(x), np.abs(y))))))


def klass(G, L, unv):
    """The declared suffix class, written here independently."""
    if unv:
        return 'incomplete_valuation'
    if G > TOL and L > TOL:
        return 'compensated_mixed' if G - L > TOL else ('uncompensated_mixed' if L - G > TOL else 'mixed_tie')
    return 'pure_observed_loss' if L > TOL else 'good_only_or_no_fill'


def load():
    gm = json.loads((R / 'manifest.json').read_text()); E = dict(gm=gm, src={}, markets={})
    for key, want in gm['af_records'].items():
        part, day = key.split('/'); rb = (AF / part / f'{day}.json').read_bytes(); rec = json.loads(rb); gb = (AF / part / f'{day}.jsonl.gz').read_bytes(); nb = (AF / part / f'{day}_arrays.npz').read_bytes()
        E['src'][key] = bool(sha(rb) == want and sha(gb) == rec['sha256']['gz'] and sha(nb) == rec['sha256']['npz']); z = np.load(io.BytesIO(nb))
        for ln in gzip.decompress(gb).decode().splitlines():
            r = json.loads(ln); E['markets'][(day, int(r['m']))] = (r, {a: z[f'{int(r["m"])}__{a}__commands'] for a in ARMS})
    ab = blob(f'{AFR}/outputs/audit.json'); raw = gzip.decompress(blob(f'{AFR}/outputs/arms.csv.gz')); E['src']['af_arms'] = sha(raw) == json.loads(ab)['outputs']['arms.csv']
    E['arms'] = pd.read_csv(io.BytesIO(raw), dtype={'day': str}); E['audit'] = json.loads((D / 'audit.json').read_text())
    rd = lambda n: pd.read_csv(D / n, dtype={'day': str, 'side': str, 'why': str})
    E.update(gen=rd('generations.csv'), core=rd('generations_core.csv'), sum=rd('summary.csv'), cls=rd('classes.csv'), seq=rd('sequences.csv'), pan=rd('panel_summary.csv'), cases=rd('cases.csv'))
    E['gen']['side'] = E['gen'].side.astype(int); E['core']['side'] = E['core'].side.astype(int)
    cb, sums = blob(f'{CODEX}/outputs/generations.csv.gz'), blob(f'{CODEX}/SHA256SUMS')
    E['src']['codex'] = bool(cb is not None and sums is not None and f'{sha(cb)}  outputs/generations.csv.gz' in sums.decode())
    E['codex'] = pd.read_csv(io.BytesIO(gzip.decompress(cb)), dtype={'day': str}) if cb is not None else pd.DataFrame(); return E


def mine(E):
    """This checker's generations and its panel totals {(arm, day, side, cls): [rows, G, L, generations]}."""
    out, pan = [], {}
    for (day, m), (r, cms) in sorted(E['markets'].items()):
        end = float(r['path_info']['end_rel'])
        for arm in ARMS:
            rec, cm = r[arm], cms[arm]; mark = float(rec['mark']); per = {}
            for f, mm, c in zip(rec['fills'], rec['fill_mids'], rec['fill_class']):
                t, s, q, p = float(f[0]), int(f[2]), float(f[3]), float(f[4]); sg = 1. if s == 0 else -1.
                hv = [None if mm[j] is None else sg * q * (float(mm[j]) - p) for j in (6, 8, 10)]
                per.setdefault(int(f[1]), []).append((t, int(f[5]), q, c[0] == 'through', hv[0] if t <= end - 1000. else None, hv[1], hv[2], sg * q * (mark - p)))
            kk = {}
            for row in cm[(cm[:, 2] == 0) & (cm[:, 7] == 2)]:
                kk.setdefault(int(row[3]), []).append(float(row[0]))
            closed, opn, qv, qk = rec['closed'], rec['open_at_end_gid'], rec['queue_venue'], rec['queue_known']
            for o in rec['orders']:
                g = int(o['gid']); fl = sorted(per.get(g, []), key=lambda x: (x[0], x[1])); ks = sorted(kk.get(g, [])); n = len(fl)
                v = [x[4] if x[4] is not None else 0. for x in fl]; un = [x[4] is None for x in fl]
                pre = [0.] * (n + 1); Gs = [0.] * (n + 1); Ls = [0.] * (n + 1); Us = [0] * (n + 1)
                for i in range(n):
                    pre[i + 1] = pre[i] + v[i]
                for i in range(n - 1, -1, -1):
                    Gs[i] = Gs[i + 1] + max(v[i], 0.); Ls[i] = Ls[i + 1] - min(v[i], 0.); Us[i] = Us[i + 1] + int(un[i])
                idx, i = [], 0
                for d in ks:
                    while i < n and fl[i][0] < d + EFF:
                        i += 1
                    idx.append(i)
                keep = pre[n]; vals = [pre[i] for i in idx]; best = max(vals) if vals else None; cut, ci = None, n
                if best is not None and best > keep + TOL:
                    for d, i, val in zip(ks, idx, vals):
                        if val > keep + TOL and val >= best - TOL:
                            cut, ci = d, i; break
                cl = closed.get(str(g), [None])[0]
                cause = 'refused' if o['refused'] else ('filled_out' if cl == 'filled' else ('recorded_cancellation' if cl == 'removed' else ('end_open' if str(g) in opn else 'recorded_cancellation')))
                a0, f1, k1 = qv['ahead0'].get(str(g)), qv['first'].get(str(g)), qk.get(str(g)) or {}; ai = idx[0] if ks else n
                x = dict(day=day, m=m, arm=arm, gid=g, side=int(o['side']), n=n, n_keeps=len(ks), status='feasible' if ks else 'no_feasible_callback', end_cause=cause,
                         unacked_at_end=bool(cause == 'recorded_cancellation' and cl is None), q_ahead_act=a0[1] if a0 else np.nan, q_first_venue_ahead=f1[1] if f1 else np.nan,
                         q_first_known_ahead=np.nan if k1.get('first') is None else k1['first'], v_keep=keep, end_keep=sum(y[7] for y in fl),
                         a_cut=ks[0] if ks else np.nan, a_cls=klass(Gs[ai], Ls[ai], Us[ai]) if ks else 'no_feasible_callback', a_suf_G=Gs[ai], a_suf_L=Ls[ai], a_suf_n_unv=Us[ai],
                         a_pre_G=sum(max(y, 0.) for y in v[:ai]), a_pre_L=-sum(min(y, 0.) for y in v[:ai]), a_v=pre[ai], a_end=sum(y[7] for y in fl[:ai]),
                         b_cut=np.nan if cut is None else cut, b_cls=('no_feasible_callback' if not ks else ('oracle_keep' if cut is None else klass(Gs[ci], Ls[ci], Us[ci]))),
                         b_suf_G=Gs[ci], b_suf_L=Ls[ci], b_suf_n_unv=Us[ci], b_v=pre[ci], b_end=sum(y[7] for y in fl[:ci]), all_L=Ls[0], all_G=Gs[0], gen_class=klass(Gs[0], Ls[0], Us[0]))
                sf = fl[ci:]
                for hh, j in (('5', 5), ('30', 6)):
                    x[f'b_suf_G{hh}'] = sum(y[j] for y in sf if y[j] is not None and y[j] > 0); x[f'b_suf_L{hh}'] = -sum(y[j] for y in sf if y[j] is not None and y[j] < 0)
                x['b_suf_Gend'] = sum(y[7] for y in sf if y[7] > 0); x['b_suf_Lend'] = -sum(y[7] for y in sf if y[7] < 0); cc = [y for y in sf if y[4] is not None and y[5] is not None and y[6] is not None]
                x.update(b_suf_cc_n=len(cc), b_suf_cc_v1=sum(y[4] for y in cc), b_suf_cc_v5=sum(y[5] for y in cc), b_suf_cc_v30=sum(y[6] for y in cc), b_suf_cc_vend=sum(y[7] for y in cc))
                ts = sorted({y[0] for y in fl}); sgn = []
                for tj in ts:
                    w = [y[4] for y in fl if y[0] == tj and y[4] is not None]; Gj = sum(z for z in w if z > 0); Lj = -sum(z for z in w if z < 0)
                    sgn.append('both' if (Gj > TOL and Lj > TOL) else ('good' if Gj > TOL else ('bad' if Lj > TOL else 'none')))
                nz = [z for z in sgn if z != 'none']; ch = sum(1 for a_, b_ in zip(nz, nz[1:]) if a_ != b_) + 2 * nz.count('both')
                isb = [z in ('bad', 'both') for z in sgn]; isg = [z in ('good', 'both') for z in sgn]
                fb = next((j for j in range(len(ts)) if isb[j]), None); btg = fb is not None and any(isg[j] for j in range(fb + 1, len(ts)))
                jb = next((j for j in range(len(ts)) if isb[j] and any(isg[:j])), None); bc = max([d for d in ks if d + EFF <= ts[fb]], default=None) if btg else None; gc = None
                if jb is not None:
                    tg = ts[max(j for j in range(jb) if isg[j])]; gc = max([d for d in ks if tg < d + EFF <= ts[jb]], default=None)
                for nm, flag, c_ in (('btg', btg, bc), ('gtb', jb is not None, gc)):
                    k_ = next((i_ for i_ in range(n) if fl[i_][0] >= c_ + EFF), n) if c_ is not None else None
                    x.update({f'seq_{nm}': bool(flag), f'seq_{nm}_cut': np.nan if c_ is None else c_, f'seq_{nm}_G': Gs[k_] if c_ is not None else np.nan,
                              f'seq_{nm}_L': Ls[k_] if c_ is not None else np.nan, f'seq_{nm}_class': klass(Gs[k_], Ls[k_], Us[k_]) if c_ is not None else ''})
                x.update(seq_interleaved=bool(ch >= 2), seq_simultaneous_good_bad='both' in sgn); out.append(x)
                for i in idx:
                    c_ = klass(Gs[i], Ls[i], Us[i])
                    for key in ((arm, day, str(int(o['side'])), c_), (arm, day, 'all', c_), (arm, 'all', 'all', c_)):
                        a_ = pan.setdefault(key, [0, 0., 0., set()]); a_[0] += 1; a_[1] += Gs[i]; a_[2] += Ls[i]; a_[3].add((day, m, g))
    return pd.DataFrame(out), pan


GEN_COLS = ('status', 'n_keeps', 'v_keep', 'end_keep', 'a_cut', 'a_cls', 'a_suf_G', 'a_suf_L', 'a_suf_n_unv', 'a_pre_G', 'a_pre_L', 'a_v', 'a_end', 'b_cut', 'b_cls', 'b_suf_G',
            'b_suf_L', 'b_suf_n_unv', 'b_v', 'b_end', 'all_L', 'all_G', 'gen_class')
EQ_COLS = ('end_cause', 'unacked_at_end', 'q_ahead_act', 'q_first_venue_ahead', 'q_first_known_ahead')
SEQ_COLS = ('seq_bad_then_good', 'seq_btg_cut', 'seq_btg_G', 'seq_btg_L', 'seq_btg_class', 'seq_good_then_bad', 'seq_gtb_cut', 'seq_gtb_G', 'seq_gtb_L', 'seq_gtb_class',
            'seq_interleaved', 'seq_simultaneous_good_bad')
HOR_COLS = ('b_suf_G5', 'b_suf_L5', 'b_suf_G30', 'b_suf_L30', 'b_suf_Gend', 'b_suf_Lend', 'b_suf_cc_n', 'b_suf_cc_v1', 'b_suf_cc_v5', 'b_suf_cc_v30', 'b_suf_cc_vend')
RENAME = dict(seq_btg='seq_bad_then_good', seq_gtb='seq_good_then_bad')


def compare(E, cols):
    J = E['mine'].rename(columns=RENAME).merge(E['gen'], on=['day', 'm', 'arm', 'gid'], suffixes=('', '_s'), how='left')
    return [c for c in cols if not same(J[c], J[f'{c}_s'])], len(J)


def fam_A(E):
    c = dict(A_committed=all(blob(p) == (REPO / p).read_bytes() for p in COMMITTED), A_inputs=bool(E['src'] and all(E['src'].values())))
    c['A_outputs'] = all(E['audit']['outputs'].get(n) == sha((D / n).read_bytes()) for n in OUTS) and E['audit']['projection_sha256'] == sha((R / 'logs' / 'projection.json').read_bytes())
    core = E['core']; full = E['gen'][list(core.columns)]; c['A_core'] = bool(len(core) == len(full) and all(same(full[k], core[k]) for k in core.columns)); return c, {}


def fam_B(E):
    want = sorted((d, m, a, int(o['gid'])) for (d, m), (r, _) in E['markets'].items() for a in ARMS for o in r[a]['orders']); G = E['gen']; gm = E['gm']
    got = sorted(zip(G.day, G.m.astype(int), G.arm, G.gid.astype(int))); mk = sorted(E['markets']); nf = sum(len(r[a]['fills']) for (r, _) in E['markets'].values() for a in ARMS)
    ok = bool(got == want and len(got) == len(set(got)) and mk == sorted((d, int(m)) for d in gm['days'] for m in gm['complete'][d]) and len(mk) == EXPECT_MARKETS
              and int(G.all_n.sum()) == nf == int(E['mine'].n.sum()) and int(G.n_keeps.sum()) == int(E['audit']['panel_rows']) == int(E['mine'].n_keeps.sum()))
    return dict(B_membership=ok), dict(B=dict(generations=len(got), fills=nf, panel_rows=int(E['mine'].n_keeps.sum())))


def fam_E(E):
    bad, _ = compare(E, EQ_COLS); return dict(E_causes_queue=not bad), dict(E=dict(mismatched=bad))


def fam_O(E):
    bad, n = compare(E, GEN_COLS); return dict(O_anchors=not bad and n == len(E['gen'])), dict(O=dict(mismatched=bad))


def fam_Q(E):
    bad, _ = compare(E, SEQ_COLS); return dict(Q_sequences=not bad), dict(Q=dict(mismatched=bad))


def fam_H(E):
    bad, _ = compare(E, HOR_COLS); return dict(H_horizons=not bad), dict(H=dict(mismatched=bad))


def fam_R(E):
    T = E['mine'].groupby(['day', 'm', 'arm']).agg(v=('v_keep', 'sum'), e=('end_keep', 'sum')).reset_index().merge(E['arms'], on=['day', 'm', 'arm'], how='left')
    bad = [(r.day, r.m, r.arm) for r in T.itertuples() if not (close(r.v, r.all_id_mo1s) and close(r.e, r.value))]
    return dict(R_reproduction=not bad and len(T) == 2 * EXPECT_MARKETS), dict(R=dict(failed=bad[:5]))


def my_levels(M_):
    out = {}
    for ks in (('arm', 'day', 'side'), ('arm', 'day'), ('arm', 'side'), ('arm',)):
        for k, g in M_.groupby(list(ks)):
            k = k if isinstance(k, tuple) else (k,); d = dict(zip(ks, k)); out[(d['arm'], d.get('day', 'all'), str(d.get('side', 'all')))] = g
    return out


def fam_S(E):
    M_ = E['mine']; lv = my_levels(M_); S_ = E['sum']; bad, n = [], 0
    for r in S_.to_dict('records'):
        g = lv.get((r['arm'], r['day'], str(r['side'])))
        if g is None:
            bad.append(('extra', r['arm'], r['day'], r['side'])); continue
        n += 1; aL = g.all_L.sum()
        x = dict(generations=len(g), feasible=int((g.status == 'feasible').sum()), oracle_cuts=int(g.b_cut.notna().sum()), v_keep=g.v_keep.sum(), a_v=g.a_v.sum(), b_v=g.b_v.sum(),
                 end_keep=g.end_keep.sum(), a_end=g.a_end.sum(), b_end=g.b_end.sum(), all_L=aL, all_G=g.all_G.sum(), a_suf_L=g.a_suf_L.sum(), b_suf_L=g.b_suf_L.sum(),
                 b_suf_G=g.b_suf_G.sum(), headroom_oracle=g.b_v.sum() - g.v_keep.sum(), b_suf_cc_v1=g.b_suf_cc_v1.sum(), b_suf_cc_v5=g.b_suf_cc_v5.sum(),
                 b_suf_cc_v30=g.b_suf_cc_v30.sum(), b_suf_cc_vend=g.b_suf_cc_vend.sum(), reading='no_local_cancel_rescue' if g.b_v.sum() < -TOL else 'local_cancel_headroom')
        for X, cl in (('a', CLS + ('no_feasible_callback',)), ('b', CLS + ('oracle_keep', 'no_feasible_callback'))):
            for c in cl:
                h = g[g[f'{X}_cls'] == c]; x[f'{X}_{c}_n'] = len(h); x[f'{X}_{c}_L'] = h[f'{X}_suf_L'].sum(); x[f'{X}_{c}_G'] = h[f'{X}_suf_G'].sum()
                x[f'{X}_{c}_Lshare'] = h[f'{X}_suf_L'].sum() / aL if aL > 0 else np.nan
        for hh in ('5', '30', 'end'):
            cv = g[f'b_suf_cc_v{hh}'].sum(); x[f'b_cc_sign_{hh}'] = int(cv > TOL) - int(cv < -TOL); x[f'b_loss_recovers_by_{hh}'] = bool(g.b_suf_cc_v1.sum() < -TOL and cv > TOL)
        bm = g.groupby(['day', 'm']).b_suf_L.sum().sort_values(ascending=False); x['b_removed_top3_market_share'] = bm.head(3).sum() / bm.sum() if bm.sum() > TOL else np.nan
        miss = [k for k, v in x.items() if not ((str(r[k]) == str(v)) if isinstance(v, (str, bool, np.bool_)) else close(r[k], v))]
        if miss:
            bad.append((r['arm'], r['day'], r['side'], miss[:4]))
    C_ = E['cls']; cb = []
    for r in C_.to_dict('records'):
        g = lv[(r['arm'], r['day'], str(r['side']))]; h = g[g[f"{r['anchor']}_cls"] == r['cls']]
        if not (r['generations'] == len(h) and close(r['G'], h[f"{r['anchor']}_suf_G"].sum()) and close(r['L'], h[f"{r['anchor']}_suf_L"].sum())):
            cb.append((r['arm'], r['day'], r['side'], r['anchor'], r['cls']))
    n_cls = sum(g[f'{X}_cls'].nunique() for g in lv.values() for X in ('a', 'b'))
    Q_ = E['seq']; sb = []
    for r in Q_.to_dict('records'):
        g = M_[M_.arm == r['arm']] if r['day'] == 'all' else M_[(M_.arm == r['arm']) & (M_.day == r['day'])]; f = r['flag']
        if f.startswith('gen_'):
            h = g[g.gen_class == f[4:]]; okr = r['generations'] == len(h) and close(r['G'], h.all_G.sum()) and close(r['L'], h.all_L.sum())
        else:
            col = {'seq_bad_then_good': 'seq_btg', 'seq_good_then_bad': 'seq_gtb'}.get(f, f); h = g[g[col]]; okr = r['generations'] == len(h)
            if f in ('seq_bad_then_good', 'seq_good_then_bad'):
                fe = h[h[f'{col}_cut'].notna()]; net = fe[f'{col}_G'] - fe[f'{col}_L']
                okr = okr and r['feasible'] == len(fe) and close(r['G'], fe[f'{col}_G'].sum()) and close(r['L'], fe[f'{col}_L'].sum()) \
                    and r['compensated'] == int((net > TOL).sum()) and r['uncompensated'] == int((net < -TOL).sum())
        if not okr:
            sb.append((r['arm'], r['day'], f))
    return dict(S_summary=not bad and n == len(lv), S_classes=not cb and len(C_) == n_cls, S_sequences=not sb and len(Q_) == 2 * (len(E['gm']['days']) + 1) * 10), \
        dict(S=dict(rows=n, bad=bad[:3], cls_bad=cb[:3], seq_bad=sb[:3]))


def fam_K(E):
    my = E['pan_mine']; P_ = E['pan']; bad = []
    for r in P_.to_dict('records'):
        a_ = my.get((r['arm'], r['day'], str(r['side']), r['cls']))
        if a_ is None or not (r['rows'] == a_[0] and close(r['G'], a_[1]) and close(r['L'], a_[2]) and r['generations'] == len(a_[3])):
            bad.append((r['arm'], r['day'], r['side'], r['cls']))
    return dict(K_panel=not bad and len(P_) == len(my)), dict(K=dict(rows=len(P_), bad=bad[:4]))


CATS = (('a_pure_observed_loss', lambda g: g.a_cls == 'pure_observed_loss'), ('a_compensated_mixed', lambda g: g.a_cls == 'compensated_mixed'),
        ('a_uncompensated_mixed', lambda g: g.a_cls == 'uncompensated_mixed'), ('a_mixed_tie', lambda g: g.a_cls == 'mixed_tie'),
        ('a_incomplete_valuation', lambda g: g.a_cls == 'incomplete_valuation'), ('bad_then_good_feasible', lambda g: g.seq_btg & g.seq_btg_cut.notna()),
        ('bad_then_good_infeasible', lambda g: g.seq_btg & g.seq_btg_cut.isna()), ('good_then_bad_feasible', lambda g: g.seq_gtb & g.seq_gtb_cut.notna()),
        ('good_then_bad_infeasible', lambda g: g.seq_gtb & g.seq_gtb_cut.isna()), ('interleaved', lambda g: g.seq_interleaved))


def fam_C(E):
    M_ = E['mine']; want = []
    for name, sel in CATS:
        for arm in ARMS:
            g = M_[(M_.arm == arm) & sel(M_)]; ks = sorted((sha(f'{arm}|{d}|{int(m)}|{int(gg)}'.encode()), d, int(m), int(gg)) for d, m, gg in zip(g.day, g.m, g.gid))
            want += [(name, arm, d, m, gg) for _, d, m, gg in ks[:3]]
    got = [(r['category'], r['arm'], r['day'], int(r['m']), int(r['gid'])) for r in E['cases'].to_dict('records')]
    return dict(C_casebook=got == want), dict(C=dict(cases=len(got)))


def fam_P(E):
    M_ = E['mine']; rd = E['audit']['reading']; bad = []
    for arm in ARMS:
        g = M_[M_.arm == arm]; by = {d: ('no_local_cancel_rescue' if h.b_v.sum() < -TOL else 'local_cancel_headroom') for d, h in g.groupby('day')}
        pooled = 'no_local_cancel_rescue' if g.b_v.sum() < -TOL else 'local_cancel_headroom'
        bad += [] if (rd[arm]['by_day'] == by and rd[arm]['pooled'] == pooled and rd[arm]['days_no_rescue'] == sum(v == 'no_local_cancel_rescue' for v in by.values())) else [arm]
    return dict(P_readings=not bad), dict(P=dict(failed=bad))


def cpu(paths):
    tot, peak = 0., 0
    for p in sorted(set(paths)):
        for ln in Path(p).read_text(errors='replace').splitlines():
            mm = RL.match(ln)
            if mm:
                tot += float(mm.group(6)); peak = max(peak, int(mm.group(4)))
    return tot / 3600., peak


def fam_I(E, root=None):
    gm = E['gm']; cp = gm['caps']; own, peak = cpu(glob.glob(str((root or R) / 'logs' / '**' / '*.log'), recursive=True)); own += cp['unlogged_inspection_s'] / 3600.
    line, _ = cpu([p for n in gm['line_receipts'] for p in glob.glob(str(REPO / 'predictfun/reviews' / n / 'logs' / '**' / '*.log'), recursive=True)])
    rev, _ = cpu([p for n in gm['line_reviews'] for p in glob.glob(str(REPO / 'predictfun/reviews/automatic' / n / '*.log'))])
    return dict(I_resources=bool(own <= cp['round_core_hours'] and peak < 1536 and line + rev + own + cp['review_reserve'] <= cp['line_ceiling'])), \
        dict(I=dict(round_core_hours=own, line_before=line + rev, line_with_round=line + rev + own, with_reserve=line + rev + own + cp['review_reserve'], peak_mib=peak))


def fam_X(E):
    C_ = E['codex']
    if C_.empty:
        return dict(X_codex_classification=False), dict(X=dict(error='the Codex generations are not committed'))
    J = C_.merge(E['gen'], on=['day', 'm', 'arm', 'gid'], how='outer', indicator=True); both = bool((J._merge == 'both').all() and len(J) == len(E['gen']))
    sub = np.where(J.a_suf_n == 0, 'NO_SUFFIX_FILLS', np.where((J.a_suf_G > TOL) & (J.a_suf_L <= TOL), 'GOOD_ONLY', 'NEAR_ZERO')); opp = J.category != 'NO_OPPORTUNITY'
    cls_ok = bool(((J.category.map(CMAP) == J.a_cls) & ((J.a_cls != 'good_only_or_no_fill') | (J.category == sub))).all())
    end_ = np.where(J.end_cause.eq('end_open') | J.unacked_at_end.astype(bool), 'open_or_unnotified_at_END',
                    J.end_cause.map(dict(recorded_cancellation='removed', filled_out='filled', refused='refused')))
    val = dict(good=same(J.good, J.a_suf_G), loss=same(J.loss, J.a_suf_L), prefix_good=same(J.prefix_good[opp], J.a_pre_G[opp]), prefix_loss=same(J.prefix_loss[opp], J.a_pre_L[opp]),
               whole_1s=same(J.whole_1s, J.v_keep), whole_end=same(J.whole_end, J.end_keep), anchor=same(J.anchor, J.a_cut), end_cause=bool((J.known_end_cause == end_).all()))
    ok = bool(E['src'].get('codex') and both and cls_ok and all(val.values()))
    return dict(X_codex_classification=ok), dict(X=dict(rows=int(len(J)), both=both, classes=cls_ok, **val))


FAMS = dict(A=fam_A, B=fam_B, E=fam_E, O=fam_O, Q=fam_Q, H=fam_H, R=fam_R, S=fam_S, K=fam_K, C=fam_C, P=fam_P, I=fam_I, X=fam_X)


def run_checks(E):
    chk, det = {}, {}
    for k in FAMS:
        c, d = FAMS[k](E); chk.update(c); det.update(d)
    return dict(checks=chk, details=det, all_ok=bool(all(chk.values())))


def _row(E, sel, col, fn, key='gen'):
    X_ = E[key].copy(); i = X_.index[sel(X_)][0]; X_.loc[i, col] = fn(X_.loc[i, col]); E[key] = X_


def _fake(E):
    E['_root'] = Path(tempfile.mkdtemp(dir=str(D))); (E['_root'] / 'logs').mkdir()
    (E['_root'] / 'logs' / 'fake.log').write_text('2026-11-22T00:00:00Z tag=fake rc=0 peak_rss_mib=100 wall_s=999.0 cpu_user_s=999.0\n')


CASES = [
    ('an_output_hash_changed', 'A', 'A_outputs', lambda E: E.__setitem__('audit', dict(E['audit'], outputs=dict(E['audit']['outputs'], **{'summary.csv': '0' * 64})))),
    ('a_core_value_changed', 'A', 'A_core', lambda E: _row(E, lambda x: x.b_cut.notna(), 'b_v', lambda v: v + 1., key='core')),
    ('a_generation_dropped', 'B', 'B_membership', lambda E: E.__setitem__('gen', E['gen'].iloc[1:])),
    ('an_end_cause_changed', 'E', 'E_causes_queue', lambda E: _row(E, lambda x: x.end_cause == 'filled_out', 'end_cause', lambda v: 'end_open')),
    ('a_queue_field_changed', 'E', 'E_causes_queue', lambda E: _row(E, lambda x: x.q_ahead_act.notna(), 'q_ahead_act', lambda v: v + 1.)),
    ('a_cut_moved', 'O', 'O_anchors', lambda E: _row(E, lambda x: x.b_cut.notna(), 'b_cut', lambda v: v + 50.)),
    ('a_removed_value_changed', 'O', 'O_anchors', lambda E: _row(E, lambda x: x.b_cut.notna(), 'b_suf_L', lambda v: v - 1.)),
    ('a_known_bad_classification', 'O', 'O_anchors', lambda E: _row(E, lambda x: x.a_cls == 'pure_observed_loss', 'a_cls', lambda v: 'compensated_mixed')),
    ('a_status_changed', 'O', 'O_anchors', lambda E: _row(E, lambda x: x.status == 'no_feasible_callback', 'status', lambda v: 'feasible')),
    ('a_sequence_flag_flipped', 'Q', 'Q_sequences', lambda E: _row(E, lambda x: x.seq_good_then_bad.astype(bool), 'seq_good_then_bad', lambda v: False)),
    ('a_horizon_value_changed', 'H', 'H_horizons', lambda E: _row(E, lambda x: x.b_cut.notna(), 'b_suf_cc_v30', lambda v: v + 1.)),
    ('the_reproduction_broken', 'R', 'R_reproduction', lambda E: E.__setitem__('arms', E['arms'].assign(value=E['arms'].value + 1.))),
    ('a_summary_total_changed', 'S', 'S_summary', lambda E: _row(E, lambda x: x.index == 0, 'b_v', lambda v: v + 1., key='sum')),
    ('a_class_row_changed', 'S', 'S_classes', lambda E: _row(E, lambda x: x.index == 0, 'L', lambda v: v + 1., key='cls')),
    ('a_panel_total_changed', 'K', 'K_panel', lambda E: _row(E, lambda x: x.index == 0, 'L', lambda v: v + 1., key='pan')),
    ('a_case_replaced', 'C', 'C_casebook', lambda E: _row(E, lambda x: x.index == 0, 'gid', lambda v: v + 100000, key='cases')),
    ('a_reading_flipped', 'P', 'P_readings', lambda E: E.__setitem__('audit', dict(E['audit'], reading={a: dict(v, pooled='local_cancel_headroom' if v['pooled'] != 'local_cancel_headroom'
                                                                                                             else 'no_local_cancel_rescue') for a, v in E['audit']['reading'].items()}))),
    ('resources_over_the_cap', 'I', 'I_resources', _fake),
    ('a_codex_class_disagrees', 'X', 'X_codex_classification', lambda E: _row(E, lambda x: x.category == 'PURE_BAD', 'category', lambda v: 'GOOD_ONLY', key='codex')),
]


def falsify(E, base):
    out = {}
    for name, fam, target, fn in CASES:
        X_ = dict(E); refused = False
        try:
            fn(X_); c, _ = FAMS[fam](X_, X_['_root']) if fam == 'I' else FAMS[fam](X_); refused = not c[target]
        except (IndexError, KeyError) as e_:
            print(f'falsifier case {name}: corruption not applied ({e_!r})')
        finally:
            if '_root' in X_:
                shutil.rmtree(X_['_root'], ignore_errors=True)
        out[name] = dict(family=fam, target=target, clean=bool(base['checks'][target]), refused=bool(refused))
    return dict(cases=out, falsifier_ok=bool(base['all_ok'] and all(v['clean'] and v['refused'] for v in out.values())))


def main(argv):
    E = load(); E['mine'], E['pan_mine'] = mine(E); r_ = run_checks(E); fz = falsify(E, r_); rd = Path(argv[0]) if argv else R
    for n, obj in (('checks.json', r_), ('checks_falsifier.json', fz)):
        p = rd / n; assert not p.exists(), f'{p} exists (write-once)'; p.write_text(json.dumps(obj, indent=1, default=str) + '\n')
    print(json.dumps(dict(all_ok=r_['all_ok'], failed=[k for k, v in r_['checks'].items() if not v], falsifier_ok=fz['falsifier_ok'],
                          cases={k: (v['clean'], v['refused']) for k, v in fz['cases'].items()}, details=r_['details']), indent=1, default=str)[:4000])


if __name__ == '__main__':
    main(sys.argv[1:])
