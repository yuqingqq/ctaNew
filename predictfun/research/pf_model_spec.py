"""Frozen model spec for P-2026-005 step 5 (2026-11-01): feature sets, LightGBM params, rounds."""
BN = ['bn_r25', 'bn_r50', 'bn_r100', 'bn_r250', 'bn_r500', 'bn_r1000', 'bn_r3000', 'bn_imb', 'bn_age', 'bn_flow250', 'bn_flow1000', 'bn_flow5000', 'bn_ntr1000']
PF = ['pf_spread', 'pf_mid', 'pf_micro_dev', 'pf_imb1', 'pf_imb3', 'pf_lbs', 'pf_las', 'pf_age', 'pf_nfr1000', 'pf_dm250', 'pf_dm1000', 'pf_dm5000',
      'pf_pend_bid1000', 'pf_pend_ask1000', 'pf_pend_tot5000', 't_rel', 'sigma', 'basis_bps', 'gap']
SETS = {'BN': BN, 'PF': PF, 'ALL': BN + PF}
PARAMS = dict(objective='regression', learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=1, verbose=-1, seed=7)
ROUNDS = 300
