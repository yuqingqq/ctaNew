"""MODEL AUDIT (2026-11-14, session ce68ebe0), POST HOC and descriptive. USER: "Can you check if the model is good for our task, or check if
the model is reasonable at all? We shouldn't have triggered so many pulls, is our training target aligned with our goal, or do we train it
properly".

Two models decide our pulls (read from the code and the freeze receipt before this script):
  GUARD (pm_ladder_v1's frozen guard, preds_t60k; pf_badfill_info M1 -> pf_step5_guard_preds PF_GUARD_KS=60 -> pf_v1_apply): per side a
        LightGBM regression of v = s (mid(t_sw + 1 s) - p) if a market leg trades at a WORSE price than the known best p within
        (tau + 22, tau + 522] ms (a sweep through the best), else 0; neutral grid rows every 500 ms; v_side = prediction - c60, c60 = the
        60th percentile of its in-sample predictions; the guard ladder pulls EVERY rung of a side when v_side <= -0.10 (GUARD_PULL).
  KEEP / CANCEL (ORDER-HARM v3): A_hat = keep now with the option to cancel later minus pull now (re-entry at the first 50 ms step after the
        guard clears), fills valued at 1 s, horizon the clear + 2 s, decided only inside guard episodes, threshold 0.
READINGS (computed; descriptive; the 9 consumed replay days, each window's first 60 s):
 G1 ALARMS: per side, the share of 50 ms steps (start price known) with v_side <= -0.10, alarm onsets per minute, on / off spell lengths.
 G2 TARGET SIGN: the guard's training targets (its grid, both sides): share < 0 / = 0 / > 0 and mean; its predictions (v_side + c60) on the
    replay days: share > 0. COST-ONLY if share(target > 0) < 0.1 x share(target < 0) and share(prediction > 0) < 1%: the target prices
    the cost of a sweep and credits nothing for staying (no good fills, no queue place).
 K1 FREE PULLS (v3 labels): rung-episodes whose keep-all and pull branches have the SAME value within the label's horizon (KAV = Pe; at the
    first decision), by rung, 2 s and 5 s tails: a pull there costs nothing in training. FREE-PULLS if the share is >= 50% pooled over
    rungs 2-5 (2 s tail).
 K2 FLICKER (5 test days, final model): sign flips of A_hat between consecutive decisions of one rung-episode; the share of pull decisions
    with -0.01 < A_hat <= 0; orders kept at the onset that meet a pull decision later in the same episode, by episode length. NOISY if
    >= 10% of consecutive decisions flip.
 K3 WHERE IT PULLS (test days): among pull decisions (A_hat <= 0) and keep decisions, the share with realized G = 0 (the choice changes
    nothing within the horizon), G > 0 (keeping was better), G < 0.
Controls: the guard values are the frozen pm_ladder_v1 guard's (preds_t60k, written by pf_v1_apply / the freeze builder; c60 from the freeze
receipt); the 50 ms grid spacing is checked per window-side (count printed); v3 predictions aligned with their rows (refused if not).
Usage (run_light, PF_SLOT=claude): pf_model_audit.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pf_kp_s4 as K                                      # noqa: E402
import pf_order_harm_v3 as V3                             # noqa: E402

OUTD = f'{K.IO}/model_audit_20261114'
DAYS9 = K.DAYS9; TESTS = tuple(V3.FOLDS)
GUARD_PULL = K.GUARD_PULL; MASK_V = -1.0
FZ = os.path.join(os.path.dirname(HERE), 'freeze', 'pm_ladder_v1', 'params.json')
GRID = f'{K.IO}/step4/badfill_info_tx_tk/grid.parquet'


def windows(day):
    reg = pd.read_parquet(f'{K.IO}/step1/{day}/registry.parquet', columns=['mid', 'ws', 'outcome'])
    d0 = K._ms(f'{day[:4]}-{day[4:6]}-{day[6:]}'); d1 = d0 + 86400000
    W0 = {}
    for m, ws, o in zip(reg.mid, reg.ws, reg.outcome):
        w = float(ws) * 1000.0
        if d0 <= w < d1 and not (K.SEAL_MS[0] <= w < K.SEAL_MS[1] or w >= K.FWD_MS) and o in ('Up', 'Down'):
            W0[int(m)] = w
    return W0


def runs(on):
    """Lengths of consecutive True and False runs."""
    if not len(on):
        return [], []
    edges = np.flatnonzero(np.diff(on.astype(np.int8))) + 1
    starts = np.r_[0, edges]; ends = np.r_[edges, len(on)]
    ln = ends - starts; val = on[starts]
    return ln[val].tolist(), ln[~val].tolist()


def main():
    T0 = time.time(); L = []; out = {}
    P = json.load(open(FZ)); c60 = P['guard']['threshold_c60']
    # G1 / G2 predictions
    steps = on_steps = onsets = 0; mins = 0.0; on_runs = []; off_runs = []; pred_pos = pred_n = 0; spacing_bad = 0
    for day in DAYS9:
        W0 = windows(day)
        p = pd.read_parquet(f'{K.IO}/step5/{day}/preds_t60k.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
        p = p[p.mid.isin(list(W0))]
        for m, g in p.groupby('mid'):
            ws = W0[int(m)]; g = g[(g.tau >= ws) & (g.tau < ws + 60000.0)].sort_values('tau')
            if len(g) < 2:
                continue
            spacing_bad += int(np.abs(np.diff(g.tau.values) - 50.0).max() > 1.0)
            for col, side in (('v_bid', 'bid'), ('v_ask', 'ask')):
                v = g[col].values.astype(float)
                ok = np.isfinite(v) & (v != MASK_V)
                v = v[ok]
                if len(v) < 2:
                    continue
                on = v <= GUARD_PULL
                steps += len(v); on_steps += int(on.sum()); onsets += int((on[1:] & ~on[:-1]).sum()); mins += len(v) * 50.0 / 60000.0
                a, b = runs(on); on_runs += a; off_runs += b
                pred = v + c60[side]; pred_pos += int((pred > 0).sum()); pred_n += len(pred)
        del p
    on_r = np.array(on_runs) * 50.0; off_r = np.array(off_runs) * 50.0
    G1 = dict(on_share=on_steps / steps, onsets_per_min=onsets / mins, on_med=float(np.median(on_r)), on_p75=float(np.percentile(on_r, 75)),
              off_med=float(np.median(off_r)), off_p75=float(np.percentile(off_r, 75)), side_minutes=mins, grid_spacing_bad=spacing_bad)
    out['G1'] = G1
    L += [f'MODEL AUDIT (POST HOC, descriptive), 9 consumed days, first 60 s; as of {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}', '',
          f'GUARD: frozen pm_ladder_v1 guard; c60 bid {c60["bid"]:+.4f} / ask {c60["ask"]:+.4f} c/sh; the ladder pulls every rung of a side when '
          f'v_side = prediction - c60 <= {GUARD_PULL:+.2f}, i.e. prediction <= {c60["bid"] + GUARD_PULL:+.3f} (bid) / {c60["ask"] + GUARD_PULL:+.3f} (ask)',
          f'G1 ALARMS: on in {G1["on_share"]:.1%} of 50 ms steps; {G1["onsets_per_min"]:.1f} alarm onsets per side-minute; on spells median '
          f'{G1["on_med"]:.0f} ms (p75 {G1["on_p75"]:.0f}); off spells median {G1["off_med"]:.0f} ms (p75 {G1["off_p75"]:.0f}); '
          f'{mins:.0f} side-minutes (window-sides with a non-50 ms grid gap: {spacing_bad})']
    # G2 target sign
    gr = pq.read_table(GRID, columns=['day', 'v_bid', 'v_ask']).to_pandas()
    gv = np.concatenate([gr.v_bid.values, gr.v_ask.values]).astype(float); gv = gv[np.isfinite(gv)]
    neg, zer, pos = float((gv < 0).mean()), float((gv == 0).mean()), float((gv > 0).mean())
    lab = 'COST-ONLY' if (pos < 0.1 * neg and pred_pos / pred_n < 0.01) else 'NOT-COST-ONLY'
    out['G2'] = dict(neg=neg, zero=zer, pos=pos, mean=float(gv.mean()), pred_pos=pred_pos / pred_n, label=lab, grid_rows=int(len(gr)))
    L.append(f'G2 TARGET SIGN: guard training targets ({len(gr)} grid rows x 2 sides): < 0 {neg:.1%}, = 0 {zer:.1%}, > 0 {pos:.1%}, mean '
             f'{gv.mean():+.3f} c/sh; predictions > 0 on the replay days {pred_pos / pred_n:.2%} -> {lab}')
    del gr, gv
    # K1 free pulls (v3 labels, 9 days)
    tie = {}; tie5 = {}
    for day in DAYS9:
        R = pq.read_table(f'{V3.OUTD}/rows_{day}.parquet', columns=['rung_i', 'j', 'KAV', 'Pe', 'KAV5', 'Pe5']).to_pandas()
        f = R[R.j == 0]
        for r_ in range(V3.NR):
            x = f[f.rung_i == r_]
            e = tie.setdefault(r_, [0, 0]); e[0] += int((x.KAV.values == x.Pe.values).sum()); e[1] += len(x)
            y = x[np.isfinite(x.KAV5.values) & np.isfinite(x.Pe5.values)]
            e5 = tie5.setdefault(r_, [0, 0]); e5[0] += int((y.KAV5.values == y.Pe5.values).sum()); e5[1] += len(y)
        del R, f
    deep = sum(tie[r_][0] for r_ in range(2, 6)) / sum(tie[r_][1] for r_ in range(2, 6))
    lab = 'FREE-PULLS' if deep >= 0.5 else 'NOT-FREE'
    out['K1'] = dict(tie={r_: tie[r_][0] / tie[r_][1] for r_ in tie}, tie5={r_: tie5[r_][0] / max(tie5[r_][1], 1) for r_ in tie5}, deep=deep, label=lab)
    L += ['', 'KEEP / CANCEL (ORDER-HARM v3)',
          'K1 FREE PULLS: rung-episodes with keep-all value = pull value within the horizon (no fill on either branch, the usual case), by rung 0-5: '
          + ' | '.join(f'{tie[r_][0] / tie[r_][1]:.1%}' for r_ in range(V3.NR)) + '  (5 s tail: '
          + ' | '.join(f'{tie5[r_][0] / max(tie5[r_][1], 1):.1%}' for r_ in range(V3.NR)) + f') -> rungs 2-5 {deep:.1%}: {lab}']
    # K2 / K3 (test days)
    flips = pairs = 0; pulls = marg = 0; k3 = {'pull': [0, 0, 0, 0], 'keep': [0, 0, 0, 0]}
    kept_on = {}; kept_then = {}
    BK = ((2, 3, '2-3 decisions (100-150 ms)'), (4, 6, '4-6 (200-300 ms)'), (7, 20, '7-20 (0.35-1 s)'), (21, 10 ** 9, '21+ (> 1 s)'))
    for day in TESTS:
        R = pq.read_table(f'{V3.OUTD}/rows_{day}.parquet', columns=['m', 'side', 'k', 'rung_i', 'ep', 'j', 'J']).to_pandas()
        Pr = pq.read_table(f'{V3.OUTD}/pred_{day}.parquet', columns=['m', 'side', 'k', 'rung_i', 'a_final', 'target']).to_pandas()
        for c in ('m', 'side', 'k', 'rung_i'):
            if not np.array_equal(Pr[c].values, R[c].values):
                sys.exit(f'REFUSED: v3 predictions not aligned with their rows on {c} ({day})')
        D = pd.DataFrame(dict(m=R.m.values, side=R.side.values, ep=R.ep.values, r=R.rung_i.values, j=R.j.values, J=R.J.values,
                              a=Pr.a_final.values, g=Pr.target.values))
        del R, Pr
        D.sort_values(['m', 'side', 'ep', 'r', 'j'], inplace=True, kind='stable')
        a = D.a.values; kp = a > 0
        same = (D.m.values[1:] == D.m.values[:-1]) & (D.side.values[1:] == D.side.values[:-1]) & (D.ep.values[1:] == D.ep.values[:-1]) & \
               (D.r.values[1:] == D.r.values[:-1])
        pairs += int(same.sum()); flips += int((same & (kp[1:] != kp[:-1])).sum())
        pl = ~kp; pulls += int(pl.sum()); marg += int((pl & (a > -0.01)).sum())
        g = D.g.values
        for nm, msk in (('pull', pl), ('keep', kp)):
            k3[nm][0] += int((msk & (g == 0)).sum()); k3[nm][1] += int((msk & (g > 0)).sum()); k3[nm][2] += int((msk & (g < 0)).sum())
            k3[nm][3] += int(msk.sum())
        # onset-kept orders pulled later in the same episode
        grp = (~np.r_[False, same]).cumsum()
        first_keep = pd.Series(kp).groupby(grp).first().values
        later_pull = pd.Series(pl & (D.j.values > 0)).groupby(grp).any().values
        Jg = pd.Series(D.J.values).groupby(grp).first().values
        for lo, hi, nm in BK:
            msk = first_keep & (Jg >= lo) & (Jg <= hi)
            kept_on[nm] = kept_on.get(nm, 0) + int(msk.sum()); kept_then[nm] = kept_then.get(nm, 0) + int((msk & later_pull).sum())
        del D
    fr = flips / pairs
    lab = 'NOISY' if fr >= 0.10 else 'STABLE'
    out['K2'] = dict(flip_rate=fr, marginal_pull_share=marg / pulls, kept_then_pulled={nm: kept_then[nm] / max(kept_on[nm], 1) for nm in kept_on},
                     label=lab)
    L.append(f'K2 FLICKER: {fr:.1%} of consecutive decisions in a rung-episode flip keep <-> pull ({pairs} pairs); {marg / pulls:.1%} of pull '
             f'decisions have -0.01 < A_hat <= 0; orders kept at the onset that meet a pull decision later in the episode: '
             + ' | '.join(f'{nm} {kept_then[nm] / max(kept_on[nm], 1):.0%}' for _, _, nm in BK) + f' -> {lab}')
    out['K3'] = {nm: dict(zero=v[0] / v[3], better_keep=v[1] / v[3], better_pull=v[2] / v[3], n=v[3]) for nm, v in k3.items()}
    L.append('K3 WHERE IT PULLS (realized G under the final policy): ' + ' | '.join(
        f'{nm} decisions: G = 0 {v[0] / v[3]:.1%}, G > 0 (keep better) {v[1] / v[3]:.1%}, G < 0 (pull better) {v[2] / v[3]:.1%} (n {v[3]})'
        for nm, v in k3.items()))
    os.makedirs(OUTD, exist_ok=True)
    txt = '\n'.join(L)
    open(f'{OUTD}/model_audit.txt', 'w').write(txt + '\n')
    json.dump(out, open(f'{OUTD}/model_audit.json', 'w'), indent=1, default=float)
    print(txt); print(f'done in {time.time() - T0:.0f} s')


if __name__ == '__main__':
    main()
