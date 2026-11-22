"""MUTATION CHECK of pf_ladder_retention_discrimination (LONG-VALUE-RETENTION-DISCRIMINATION-v1; 2026-11-22, session aea5bfb9), run by pf_mutant_one.py
with pf_ladder_retention_discrimination_selftest. Required: the control passes; every mutant is caught BY A CHECK; every target occurs once.
Not planted (equivalent on the fixtures, or exercised on saved data only and recomputed by rd_checks.py): the census, the per-day draw count
(any count leaves the constructed p small), the FeatureGuard's not-yet-applied condition (one callback per gid per tick) and the day
runner's write-once and memory guards (its earlier checks refuse first)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'pf_ladder_retention_discrimination.py'
SELFTEST = 'pf_ladder_retention_discrimination_selftest'
FAIL = 'RETENTION DISCRIMINATION SELFTEST FAILED'
MUTANTS = [
    ('the distance without the side sign', "dist_ticks=sg * (best - price)", "dist_ticks=(best - price)"),
    ('the level size from the other side', "fk = B.E.f[B.E.known]; p0 = 2 + 10 * s;", "fk = B.E.f[B.E.known]; p0 = 2 + 10 * (1 - s);"),
    ('inventory without the side sign', "position_toward=sg * float(v['position'])", "position_toward=float(v['position'])"),
    ('the reservations swapped', "reservation_side=float(v['B'] if s == 0 else v['A']), reservation_opp=float(v['A'] if s == 0 else v['B'])",
     "reservation_side=float(v['A'] if s == 0 else v['B']), reservation_opp=float(v['B'] if s == 0 else v['A'])"),
    ('a stale pressure value used', "pressure_x=float(x) if (st == 'ok' and np.isfinite(x)) else 0.", "pressure_x=float(x)"),
    ('the mid change over 100 ms', "jp = int(np.searchsorted(B.E.f[:, 1], t - 1000., side='right')) - 1", "jp = int(np.searchsorted(B.E.f[:, 1], t - 100., side='right')) - 1"),
    ('the mid change on venue time', "jp = int(np.searchsorted(B.E.f[:, 1], t - 1000., side='right')) - 1", "jp = int(np.searchsorted(B.E.f[:, 0], t - 1000., side='right')) - 1"),
    ('the mode inverted', "restricted=float(v['mode'] != 'UNRESTRICTED')", "restricted=float(v['mode'] == 'UNRESTRICTED')"),
    ('features in the KEEP arm only', "        if f is not None and f['kind'] == 'REVIEW' and float(v['t']) == f['t']", "        if f is not None and f['action'] == 'KEEP' and float(v['t']) == f['t']"),
    ('the expected callback with PULL', "R_['applied'][0][:4] == ('live', st['t'], st['gid'], 'KEEP')", "R_['applied'][0][:4] == ('live', st['t'], st['gid'], 'PULL')"),
    ('no status on a failed gate', "            pr['status'] = 'gate_failed' if pr['status'] == 'ok' else pr['status']\n", "            pass\n"),
    ('the unused-exemption gate dropped', "pr['gate_unused_exemption'] = bool(all(ss.values()))", "pr['gate_unused_exemption'] = True"),
    ('the winsor ceiling at the floor', "hi = np.quantile(X, WINSOR_Q[0], axis=0), np.quantile(X, WINSOR_Q[1], axis=0)", "hi = np.quantile(X, WINSOR_Q[0], axis=0), np.quantile(X, WINSOR_Q[0], axis=0)"),
    ('lambda not scaled by n', "np.linalg.solve(Z.T @ Z + LAMBDA_PER_N * len(y) * np.eye(Z.shape[1])", "np.linalg.solve(Z.T @ Z + LAMBDA_PER_N * np.eye(Z.shape[1])"),
    ('the target unclipped', "yc = np.clip(y, yl, yh)", "yc = y"),
    ('a zero sd kept', "sd = np.where(sd > 0, sd, 1.)", "sd = sd"),
    ('the prediction without the intercept', "    return m['ym'] + Z @ np.asarray(m['w'])", "    return Z @ np.asarray(m['w'])"),
    ('retain the negative predictions', "    retain = Ev.pred > 0; V = float(Ev.Y[retain].sum())", "    retain = Ev.pred < 0; V = float(Ev.Y[retain].sum())"),
    ('the constant control inverted', "const_retain = bool(Tr.Y.mean() > 0)", "const_retain = bool(Tr.Y.mean() < 0)"),
    ('p from the lower tail', "p_draw=float((1 + (D_ >= V).sum()) / (1 + len(D_)))", "p_draw=float((1 + (D_ <= V).sum()) / (1 + len(D_)))"),
    ('no evaluation support floor', "res['eval_nontie'] < MIN_NONTIE or", "res['eval_nontie'] < 0 or"),
    ('no daily condition', "    daily = all(x['V_rule'] > 0 and x['V_rule'] > max(x['always_retain'], 0., x['constant']) for x in per.values())",
     "    daily = True or all(x['V_rule'] > 0 and x['V_rule'] > max(x['always_retain'], 0., x['constant']) for x in per.values())"),
    ('the rest without the projection', "need(part == 'prefix' or (pp.exists() and json.loads(pp.read_text()).get('run_rest') is True), 'the rest needs the write-once projection with run_rest true')",
     "need(True, 'the rest needs the write-once projection with run_rest true')"),
    ('measure without the freeze', "        need(frozen(), 'the discrimination builder / selftest / declaration / manifest / census / launcher / checker are not the committed ones')\n    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); od = Path(out or OUT)",
     "        pass\n    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); od = Path(out or OUT)"),
    ('build_manifest without write-once', "need(not path.exists() and not cpath.exists(), 'the manifest / census exists (write-once)')", "need(True, 'the manifest / census exists (write-once)')"),
    ('project without write-once', "path = Path(path or REPO / RECEIPT / 'logs' / 'projection.json'); need(not path.exists(), 'the projection exists (write-once)')",
     "path = Path(path or REPO / RECEIPT / 'logs' / 'projection.json'); need(True, 'the projection exists (write-once)')"),
]
