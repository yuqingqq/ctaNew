"""Apply after the p30 rows rebuild: v6 options in pf_stop_model.py (volatility-normalized Binance returns z_*, pending flows relative to
W30, v6 fit / report, T5 over 10 permutation draws)."""
p = '/home/yuqing/ctaNew/predictfun/research/pf_stop_model.py'; s = open(p).read()
R = [
("""X5_F = X5_FAM['PF_OFI'] + X5_FAM['BN_DEPTH']""",
"""X5_F = X5_FAM['PF_OFI'] + X5_FAM['BN_DEPTH']
VOLZ = ['z_bn_r250', 'z_bn_r1000', 'z_bn_r3000']            # 2026-11-15: adverse Binance return / (1e4 sigma sqrt(h / 1 s)) (Codex 'condition' idea;
PEND_REL = {'own_pend': 'r_own_pend', 'opp_pend': 'r_opp_pend', 'pf_pend_tot5000': 'r_pf_pend_tot5000'}   # the low-vol coverage flag)
X6_JSON = f'{OUTD}/x6test_20261115.json'; T5_DRAWS = 10; T5_MAX = 0.01"""),
("""    def col(c):
        if c in X5_F:""",
"""    def col(c):
        if c.startswith('z_'):                             # volatility-normalized adverse Binance return
            h_ = float(c.split('_r')[-1]); return col('adv_' + c[2:]) / (1e4 * col('sigma') * np.sqrt(h_ / 1000.0))
        if c in X5_F:"""),
("""    elif cmd == 'fit5':
        fit_fold_v4(sys.argv[2], vtag='_v5')""",
"""    elif cmd == 'fit5':
        fit_fold_v4(sys.argv[2], vtag='_v5')
    elif cmd == 'fit6':
        fit_fold_v4(sys.argv[2], vtag='_v6')"""),
("""    elif cmd == 'report5':
        report('5')""",
"""    elif cmd == 'report5':
        report('5')
    elif cmd == 'report6':
        report('6')"""),
("""    x5add = []
    if vtag == '_v5':""",
"""    x5add = []
    if vtag in ('_v5', '_v6'):"""),
("""        F = F + x5add""",
"""        F = F + x5add
    if vtag == '_v6':                                      # v6 = v5 + the x6 test's adopted representation changes (on the rebuilt rows)
        if not os.path.exists(X6_JSON):
            sys.exit('REFUSED: no x6 test (run pf_stop_model_review.py x6test first)')
        X6 = json.load(open(X6_JSON))
        if 'PEND_REL' in X6['adopt']:
            F = [PEND_REL.get(c, c) for c in F]
        if 'VOLZ' in X6['adopt']:
            F = F + VOLZ"""),
("""        bp = fit_head(X[ii], yp, wc, F, valid=vv, params=lgbp); bpr = float(np.average(yp, weights=wc)); del vv""",
"""        bp = [fit_head(X[ii], yp, wc, F, valid=vv, params=lgbp)]; bpr = float(np.average(yp, weights=wc))
        if vtag == '_v6':                                  # v6: T5 over T5_DRAWS permutations (v5's single draw refused at +0.00115; P31 / rule 6)
            for sd_ in range(1, T5_DRAWS):
                bp.append(fit_head(X[ii], np.random.default_rng(99 + sd_).permutation(yp), wc, F, valid=vv, params=lgbp))
        del vv"""),
("""            pq_ = bp[0].predict(Xd[r], num_iteration=bp[1]); a_p, n_p = auc_nan(yt.astype(bool), pq_)
            bss = 1.0 - float(((pq_ - yt) ** 2).mean()) / float(((bpr - yt) ** 2).mean())
            T5 = dict(auc=a_p, bss=bss, rows=n_p, rounds=int(bp[1]), ok=bool(bss <= T5_BSS), auc_within_declared_v3=bool(abs(a_p - 0.5) <= T5_AUC))""",
"""            dr = []
            for (bb, nn) in bp:
                pq_ = bb.predict(Xd[r], num_iteration=nn); a_p, n_p = auc_nan(yt.astype(bool), pq_)
                dr.append((a_p, 1.0 - float(((pq_ - yt) ** 2).mean()) / float(((bpr - yt) ** 2).mean())))
            bss = float(np.mean([x[1] for x in dr])); mx = float(max(x[1] for x in dr)); a_p = float(np.mean([x[0] for x in dr]))
            ok5 = bool(bss <= T5_BSS and mx <= T5_MAX) if len(dr) > 1 else bool(dr[0][1] <= T5_BSS)
            T5 = dict(auc=a_p, bss=bss, max_bss=mx, draws=dr, rows=n_p, rounds=int(bp[0][1]), ok=ok5, auc_within_declared_v3=bool(abs(a_p - 0.5) <= T5_AUC))"""),
]
for a, b in R:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s); print('patched')
