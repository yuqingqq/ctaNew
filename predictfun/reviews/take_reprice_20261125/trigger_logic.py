"""What is the winners' trigger logic: a big Binance imbalance, the traded amount, and the three inputs proposed for the predictor (the
opposite level's shrink split into traded and withdrawn, the near level's growth, a tick just printed)? (USER 2026-11-26: "you mean the
logic is when it shows a big imbalance, they fire. is it related to trades, like trading amount"; "add these inputs as well").
Readout of winners_trigger_points.parquet and depletion_cause_points.parquet (rows aligned and checked); four consumed days; point
estimates; no strategy change.
Inputs at the send s, signed their way: imbalance (bt_imb, at s); traded amount (fl300: BTC, aggressive their way net of the other side,
300 ms); traded and withdrawn fractions of the opposite level at s - 300 (V / Q, W / Q) and near growth (near best size at s minus at
s - 300, BTC), all three on same / gone points only; tick (the opposite level gone in those 300 ms: 1, still there: 0).
Buckets fixed here before the result: imbalance < 0 / 0-0.5 / 0.5-0.8 / >= 0.8; traded amount against (< -0.01) / none (|.| <= 0.01) /
0.01-0.5 / 0.5-2 / >= 2; traded fraction none (V <= 0.5 mBTC) / < 0.2 / 0.2-0.5 / >= 0.5; withdrawn fraction < 0 (net re-added) / 0-0.2 /
0.2-0.5 / >= 0.5; near growth < -1 / -1..1 / 1-3 / >= 3 BTC; tick yes / no.
Per group and cell: sends, random times (the declared 12-shift null), lift = share of sends / share of random times, the Binance futures
mid moving their way within 150 ms (bn_f150 > 0, NaN = no) after the sends / the random times in the cell, the sends' 1 s value (cents per
share, per order). State rules (list below) read at random times: how often the state holds and how often the tick follows.
Models of the sends, winners_trigger's protocol (median fill from the null, standardized, balanced logistic, C = 1, leave-one-day-out,
150,000-row cap): LOGIC (the six inputs + opposite size), BASE (winners_trigger's 30 features; the top taker's folds must reproduce the
stored AUCs), BASE+ (BASE + the four new inputs). A depth-3 tree on the LOGIC inputs (top taker; leaves >= 1 % of rows) read as rules,
with its leave-one-day-out AUC. Top taker's size per decision (its orders sharing market and first-trade time, a burst; winners_legs with
analyze's filters, matched to its sends): orders and shares by imbalance and amount bucket, Spearman rho.
Output (write-once): trigger_logic.json.
Usage: PF_SLOT=<slot> predictfun/research/run_light.sh <tag> -- python3 predictfun/reviews/take_reprice_20261125/trigger_logic.py [dir]
"""
import json
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import depletion_cause as DC  # noqa: E402

C, TR, WT = DC.C, DC.TR, DC.WT
TOP1 = ('0x7cab', 'c0AD')
NEW = ['tfrac', 'wfrac', 'ngrow', 'tick']
LOGIC = ['bt_imb', 'fl300', 'tfrac', 'wfrac', 'ngrow', 'tick', 'bt_opp_log']
NAMES = dict(bt_imb='imbalance', fl300='traded_amount_btc', tfrac='traded_frac', wfrac='withdrawn_frac', ngrow='near_growth_btc', tick='tick', bt_opp_log='opp_size_log')
BUCKETS = {
    'bt_imb': (('<0', '0-0.5', '0.5-0.8', '>=0.8'), lambda D: [D.bt_imb < 0., D.bt_imb < .5, D.bt_imb < .8, D.bt_imb >= .8]),
    'fl300': (('against', 'none', '0.01-0.5', '0.5-2', '>=2'), lambda D: [D.fl300 < -.01, D.fl300 <= .01, D.fl300 < .5, D.fl300 < 2., D.fl300 >= 2.]),
    'tfrac': (('none', '<0.2', '0.2-0.5', '>=0.5'), lambda D: [D.pre_V <= DC.TOL, D.tfrac < .2, D.tfrac < .5, D.tfrac >= .5]),
    'wfrac': (('<0', '0-0.2', '0.2-0.5', '>=0.5'), lambda D: [D.wfrac < 0., D.wfrac < .2, D.wfrac < .5, D.wfrac >= .5]),
    'ngrow': (('<-1', '-1..1', '1-3', '>=3'), lambda D: [D.ngrow < -1., D.ngrow <= 1., D.ngrow < 3., D.ngrow >= 3.]),
    'tick': (('no', 'yes'), lambda D: [D.tick == 0, D.tick == 1]),
}
RULES = {
    'imb>=0.5': lambda D: D.bt_imb >= .5,
    'imb>=0.8': lambda D: D.bt_imb >= .8,
    'imb>=0.5&amount>=0.5': lambda D: (D.bt_imb >= .5) & (D.fl300 >= .5),
    'imb>=0.8&amount>=0.5': lambda D: (D.bt_imb >= .8) & (D.fl300 >= .5),
    'tick': lambda D: D.tick == 1,
    'tick&imb>=0.5': lambda D: (D.tick == 1) & (D.bt_imb >= .5),
    'withdrawn>=0.5': lambda D: D.wfrac >= .5,
    'withdrawn>=0.5&imb>=0.5': lambda D: (D.wfrac >= .5) & (D.bt_imb >= .5),
    'traded>=0.5': lambda D: D.tfrac >= .5,
    'near_growth>=1&imb>=0.5': lambda D: (D.ngrow >= 1.) & (D.bt_imb >= .5),
}


def bucket(D, col):
    names, conds = BUCKETS[col]
    return np.select([np.asarray(c, bool) for c in conds(D)], list(range(len(names))), -1)


def mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def cell(S, N, ms, mn):
    ms, mn = np.asarray(ms, bool), np.asarray(mn, bool)
    ns, nn = int(ms.sum()), int(mn.sum())
    return dict(sends=ns, random=nn, share_sends=ns / len(S), share_random=nn / len(N), lift=(ns / len(S)) / (nn / len(N)) if nn else None,
                hit_send=mean(S.bn_f150.to_numpy()[ms] > 0) if ns else None, hit_random=mean(N.bn_f150.to_numpy()[mn] > 0) if nn else None,
                v1_send=mean(S.v1.to_numpy()[ms]) if ns else None)


def design(G, cols):
    Z = G[cols].copy()                                  # winners_trigger.analyze verbatim (float32 columns stay float32)
    med = Z[G.y == 0].median()
    Z = Z.fillna(med)
    mu, sd = Z.mean(), Z.std().replace(0, 1.)
    return ((Z - mu) / sd).clip(-6, 6)


def lodo(G, cols):
    """winners_trigger.analyze's model protocol on `cols`: held-out AUC per day and the all-days coefficients."""
    rng = np.random.default_rng(WT.SEED)
    Z, yv = design(G, cols), G.y.to_numpy()

    def cap(mask):
        ix = np.flatnonzero(mask)
        return np.sort(rng.choice(ix, WT.LR_ROWS, replace=False)) if len(ix) > WT.LR_ROWS else ix
    cv = {}
    for dy in WT.DAYS:
        tr_, te_ = cap((G.day != dy).to_numpy()), cap((G.day == dy).to_numpy())
        if len(set(yv[te_])) < 2 or len(set(yv[tr_])) < 2:
            continue
        lr = LogisticRegression(max_iter=1000, C=1., class_weight='balanced').fit(Z.iloc[tr_], yv[tr_])
        cv[dy] = float(roc_auc_score(yv[te_], lr.predict_proba(Z.iloc[te_])[:, 1]))
    ia = cap(np.ones(len(G), bool))
    lr = LogisticRegression(max_iter=1000, C=1., class_weight='balanced').fit(Z.iloc[ia], yv[ia])
    return dict(cv_auc_by_day=cv, coef=dict(sorted(((c, float(w)) for c, w in zip(cols, lr.coef_[0])), key=lambda kv: -abs(kv[1]))))


def tree_rules(G):
    """Depth-3 tree on the LOGIC inputs: leave-one-day-out AUC, then the all-days tree's leaves as rules with their send / random-time stats."""
    Zf = G[LOGIC].astype(float)
    Zf = Zf.fillna(Zf[G.y == 0].median())
    yv, leaf_min = G.y.to_numpy(), max(200, int(.01 * len(G)))
    mk = lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=leaf_min, class_weight='balanced', random_state=0)
    cv = {}
    for dy in WT.DAYS:
        tr_, te_ = (G.day != dy).to_numpy(), (G.day == dy).to_numpy()
        if len(set(yv[te_])) < 2 or len(set(yv[tr_])) < 2:
            continue
        cv[dy] = float(roc_auc_score(yv[te_], mk().fit(Zf[tr_], yv[tr_]).predict_proba(Zf[te_])[:, 1]))
    t = mk().fit(Zf, yv)
    T, paths = t.tree_, {}

    def rec(node, conds):
        if T.children_left[node] == -1:
            paths[node] = conds
            return
        f, thr = NAMES[LOGIC[T.feature[node]]], float(T.threshold[node])
        rec(T.children_left[node], conds + [f'{f} <= {thr:.3g}'])
        rec(T.children_right[node], conds + [f'{f} > {thr:.3g}'])
    rec(0, [])
    lf = t.apply(Zf)
    S, N = G[G.y == 1], G[G.y == 0]
    rules = [dict(rule=' & '.join(c), **cell(S, N, lf[yv == 1] == node, lf[yv == 0] == node)) for node, c in paths.items()]
    return dict(cv_auc_by_day=cv, leaf_min_rows=leaf_min, leaves=sorted(rules, key=lambda r: -(r['lift'] or 0.)))


def top1_orders(X):
    """The top taker's orders exactly as winners_trigger.analyze builds them, matched to its send points (same market, same send time)."""
    vs = json.loads((WT.VR.OUT / 'summary.json').read_text())
    who, out = None, []
    for day in WT.DAYS:
        comp = {int(r['mid']) for r in vs['rows'] if r['day'] == day and r['status'] == 'complete'}
        J = pd.read_parquet(WT.OUT / f'winners_legs_{day}.parquet', columns=['mid', 'side', 'shares', 't_ms2', 'how2', 'taker', 'taker_hash', 'ambiguous'])
        if who is None:
            t = [x for x in J.taker.dropna().unique() if x.startswith(TOP1[0]) and x.endswith(TOP1[1])]
            C.require(len(t) <= 1, 'the top taker prefix / suffix is ambiguous')
            who = t[0] if t else None
        J = J[(J.taker == who) & ~J.ambiguous & J.mid.isin(comp) & J.how2.isin([1, 3])].copy()
        J['t_v'] = J.t_ms2.astype(float) - WT.M50
        J['d'] = np.where(J.side == 1, 1, -1)
        O = J.groupby('taker_hash').agg(mid=('mid', 'first'), t=('t_v', 'min'), d=('d', 'first'), dn=('d', 'nunique'), shares=('shares', 'sum')).reset_index()
        O = O[O.dn == 1]
        O['day'], O['s'] = day, O.t - WT.HOLD
        out.append(O[['day', 'mid', 's', 'd', 'shares']])
    C.require(who is not None, 'the top taker was not found')
    O = pd.concat(out, ignore_index=True)
    key = ['day', 'mid', 's']
    B = O.groupby(key).agg(orders=('shares', 'size'), shares=('shares', 'sum'), d=('d', 'first'), dn=('d', 'nunique')).reset_index()
    S = X[(X.grp == 'TOP1') & (X.y == 1)][key + ['d', 'bt_imb', 'fl300']]
    U = S.drop_duplicates(key)                             # one decision = the orders sharing market and first-trade time (a burst)
    M = U.merge(B, on=key, how='left', suffixes=('', '_o'), validate='one_to_one')
    C.require(len(M) == len(U) and M.shares.notna().all() and int(M.orders.sum()) == len(S), 'the top taker bursts do not match its sends')
    mixed = (M.dn > 1).to_numpy()                          # both directions in one burst: excluded, counted
    C.require((M.d[~mixed] == M.d_o[~mixed]).all(), 'a burst direction differs from its sends')
    return M[~mixed].copy(), int(mixed.sum())


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (1200, 1201))
    fn = (Path(sys.argv[1]) if len(sys.argv) > 1 else H) / 'trigger_logic.json'     # an argument: rehearsal dir
    C.require(not fn.exists(), 'trigger_logic.json exists (write-once)')
    WJ = json.loads((H / 'winners_trigger.json').read_text())
    stored, BASE = WJ['groups'], WJ['features']
    X = pd.read_parquet(WT.OUT / 'winners_trigger_points.parquet')
    Z = pd.read_parquet(DC.OUT / 'depletion_cause_points.parquet', columns=['day', 'k', 'y', 'd', 's', 'pre_st', 'pre_Q', 'pre_V', 'pre_W', 'near_a', 'near_b'])
    C.require(len(X) == len(Z) and all((X[c].to_numpy() == Z[c].to_numpy()).all() for c in ('day', 'k', 'y', 'd', 's')), 'the two point files are not row-aligned')
    sp = np.isin(Z.pre_st.to_numpy(), [DC.SAME, DC.GONE])
    Q, V, W = (Z[c].to_numpy(float) for c in ('pre_Q', 'pre_V', 'pre_W'))
    X['pre_V'] = np.where(sp, V, np.nan)
    X['tfrac'], X['wfrac'] = np.where(sp, V / Q, np.nan), np.where(sp, W / Q, np.nan)
    X['ngrow'] = np.where(sp, Z.near_b.to_numpy(float) - Z.near_a.to_numpy(float), np.nan)
    X['tick'] = np.where(sp, (Z.pre_st.to_numpy() == DC.GONE).astype(float), np.nan)
    del Z
    out = dict(asof='2026-11-26', points=int(len(X)), buckets={NAMES[c]: b[0] for c, b in BUCKETS.items()}, rules=list(RULES), groups={})
    for g in DC.GROUPS:
        G = X[X.grp == g].reset_index(drop=True)
        S, N = G[G.y == 1], G[G.y == 0]
        r = dict(sends=int(len(S)), random=int(len(N)), hit_send_all=mean(S.bn_f150 > 0), hit_random_all=mean(N.bn_f150 > 0), v1_send_all=mean(S.v1))
        b = stored[g]['bet']
        C.require(abs(r['hit_send_all'] - b['move_their_way_150']) < 1e-9 and abs(r['hit_random_all'] - b['null_150']) < 1e-9, f'{g}: the stored hit rates are not reproduced')
        r['univariate'] = {}
        for c in LOGIC:
            ok = G[c].notna().to_numpy()
            a = float(roc_auc_score(G.y[ok], G[c][ok]))
            if c in stored[g]['uni']:
                C.require(abs(a - stored[g]['uni'][c]['auc']) < 1e-9, f'{g} {c}: the stored AUC is not reproduced')
            r['univariate'][NAMES[c]] = dict(auc=a, send_median=float(S[c].median()), random_median=float(N[c].median()), send_mean=mean(S[c]), random_mean=mean(N[c]))
        r['by_input'] = {}
        for c in BUCKETS:
            bs, bn = bucket(S, c), bucket(N, c)
            r['by_input'][NAMES[c]] = {nm: cell(S, N, bs == i, bn == i) for i, nm in enumerate(BUCKETS[c][0])}
        bsi, bsf, bni, bnf = bucket(S, 'bt_imb'), bucket(S, 'fl300'), bucket(N, 'bt_imb'), bucket(N, 'fl300')
        r['grid_imbalance_x_amount'] = {f'{a}|{f}': cell(S, N, (bsi == i) & (bsf == j), (bni == i) & (bnf == j))
                                        for i, a in enumerate(BUCKETS['bt_imb'][0]) for j, f in enumerate(BUCKETS['fl300'][0])}
        r['grid_tick_x_imbalance'] = {f'tick={t}|imb{">=" if h else "<"}0.5': cell(S, N, (S.tick == t) & ((S.bt_imb >= .5) == h), (N.tick == t) & ((N.bt_imb >= .5) == h))
                                      for t in (1, 0) for h in (True, False)}
        r['rules_at_random_times'] = {k: dict(share_random=mean(f(N)), hit_random=mean(N.bn_f150[np.asarray(f(N), bool)] > 0), share_sends=mean(f(S)),
                                              hit_send=mean(S.bn_f150[np.asarray(f(S), bool)] > 0)) for k, f in RULES.items()}
        best = max((v['hit_random'] for v in r['rules_at_random_times'].values() if v['hit_random'] is not None), default=None)
        r['best_rule_hit_random'] = best
        r['best_rule_below_sends_hit'] = best is not None and best < r['hit_send_all']
        r['models'] = dict(LOGIC=lodo(G, LOGIC), BASE=lodo(G, BASE), BASE_PLUS=lodo(G, BASE + NEW))
        if g == 'TOP1':
            sc = stored[g]['multi']['cv_auc_by_day']
            C.require(all(abs(r['models']['BASE']['cv_auc_by_day'][d] - v) < 1e-9 for d, v in sc.items()), 'TOP1: BASE does not reproduce the stored held-out AUCs')
            r['tree'] = tree_rules(G)
        for m in r['models'].values():
            m['cv_auc_mean_days_with_50_sends'] = mean([v for d, v in m['cv_auc_by_day'].items() if int(((G.day == d) & (G.y == 1)).sum()) >= 50])
        out['groups'][g] = r
        C.require(DC.F.rss_mib() < DC.F.RSS_GUARD_MIB, 'RSS over the self-guard')
        print(g, json.dumps({k: v['cv_auc_by_day'] for k, v in r['models'].items()}), flush=True)
    M, n_mixed = top1_orders(X)
    ok = M.bt_imb.notna()
    bi, bf = bucket(M, 'bt_imb'), bucket(M, 'fl300')
    sz = lambda m: dict(bursts=int(m.sum()), median_orders=float(M.orders[m].median()), mean_orders=float(M.orders[m].mean()), median_shares=float(M.shares[m].median()),
                        mean_shares=float(M.shares[m].mean())) if m.any() else None
    out['top1_size_per_decision'] = dict(bursts=int(len(M)), excluded_mixed_direction=n_mixed, **sz(np.ones(len(M), bool)),
                                         orders_per_burst={k: int((M.orders == k).sum()) for k in (1, 2, 3)} | {'4+': int((M.orders >= 4).sum())},
                                         by_imbalance={nm: sz(bi == i) for i, nm in enumerate(BUCKETS['bt_imb'][0])},
                                         by_amount={nm: sz(bf == i) for i, nm in enumerate(BUCKETS['fl300'][0])},
                                         spearman_shares_imbalance=float(spearmanr(M.shares[ok], M.bt_imb[ok])[0]), spearman_shares_amount=float(spearmanr(M.shares, M.fl300)[0]))
    TR.dump(fn, out)
    print(json.dumps(dict(top1_size_per_decision=out['top1_size_per_decision']), indent=1))


if __name__ == '__main__':
    main()
