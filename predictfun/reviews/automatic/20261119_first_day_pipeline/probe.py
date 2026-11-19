"""Independent bounded falsifiers against the frozen work-in-progress snapshot."""
from pathlib import Path
from datetime import datetime, timezone
from types import SimpleNamespace
import contextlib
import hashlib
import io
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/yuqing/ctaNew')
snapshot = json.loads((HERE/'snapshot.json').read_text())
TREE = Path(snapshot['isolated_checkout'])
for path, expected in snapshot['reviewed_working_sources'].items():
    assert hashlib.sha256((TREE/path).read_bytes()).hexdigest() == expected
sys.path.insert(0, str(TREE/'predictfun/research'))
import numpy as np
import pandas as pd
import pf_pipeline_export as EX
import pf_pipeline_bench as B
import pf_pipeline_policy as P


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def book(future_receipt):
    return dict(kf=np.array([-10., future_receipt]), tg=np.arange(0., 201., 50.),
                P5=np.array([[50., 49., 48., 47., 46.]]*2),
                S5=np.array([[12., 10., 10., 10., 10.]]*2))


def block(w):
    empty = (np.array([]), np.array([]), np.array([], int))
    f = {clock: {} for clock, _ in EX.CLOCKS}
    o = SimpleNamespace(side=0, t=0., p=50., oid=0, m=1, phase=1, mark=51., status='eligible',
                        g_struct=np.nan, q=0., v=0., t_full=np.nan,
                        q_m0=0., v_m0=0., t_full_m0=np.nan,
                        q_m250=0., v_m250=0., t_full_m250=np.nan, f00=51., f02=51.)
    x = np.zeros(len(EX.FC.MARKET), np.float32); x[EX.FC.MARKET.index('pf_mid')] = 51.
    features = {0: {0: (x, 30.), 2: (x, 30.)}}
    fails = []
    out = EX.order_block('20261023', o, np.array([2]), np.array([0., 100.]), np.array([0, 2]),
                         np.array([True, True]), True, w, features, f, empty,
                         {p: {} for p in EX.KP.POLICIES}, {p: [0., 0] for p in EX.KP.POLICIES}, fails)
    assert not fails
    return out


def posting_leak():
    # Same known history through decision 0; the next update arrives at +30 or
    # +80. Both are future; activation is +58. Only its future receipt changes.
    a, b = block(book(30.)), block(book(80.))
    x, y = a.iloc[0], b.iloc[0]
    changed = [c for c in EX.INPUTS if not ((pd.isna(x[c]) and pd.isna(y[c])) or x[c] == y[c])]
    assert 'o_qunk' in changed and 'fp_queue' in changed
    assert x.o_qunk == 1. and y.o_qunk == 0.
    # Positive and negative controls on the time-local shown quantity.
    assert x.fp_shown == y.fp_shown == 12.
    w = book(30.); w['S5'][0, 0] = 18.
    assert block(w).iloc[0].fp_shown == 18.
    assert np.array_equal(a.iloc[1][EX.INPUTS].astype(float), b.iloc[1][EX.INPUTS].astype(float), equal_nan=True)
    return dict(reproduced=True, decision_ms=0., activation_ms=58., future_receipts_ms=[30., 80.],
                changed_post_inputs=changed, unknown_flag=[float(x.o_qunk), float(y.o_qunk)],
                shown_unchanged=True, same_time_positive_control=True, later_live_state_agrees=True)


def contract_bypass():
    common = dict(day='20261027', kind='retain', j=2, side=0, t=900., decision_ms=1000.,
                  pf_mid=50., p=50., fp_remaining=30., J_P=.5)
    good = dict(common, oid=0, J_Eq=5., J_Eqm=-5., J_V=-5.)
    bad = dict(common, oid=1, J_Eq=1., J_Eqm=-200., J_V=-200.)
    pred = pd.DataFrame([good, bad])
    parity = P.contract_parity(pred, trained=0., n=2)
    hits = P.triggers_of(pred)
    assert parity == (1, 1, 1) and len(hits) == 2
    # This is the exact P0f predicate at controls(), not a substitute criterion.
    comp, refused, agree = parity
    p0f = comp > 0 and agree == comp
    assert p0f
    assert P.contract_parity(pd.DataFrame([good]), trained=0.) == (1, 0, 1)
    assert P.contract_parity(pd.DataFrame([bad]), trained=0.) == (0, 1, 0)
    return dict(reproduced=True, compared=comp, refused=refused, agree=agree,
                p0f_accepts=p0f, pulls_including_invalid_order=len(hits),
                invalid_implied_endpoint_mid=-150., valid_only_positive_control=True,
                invalid_only_contract_refusal=True)


def gate_bias():
    n = 20; actual = np.linspace(-10., -1., n)
    df = pd.DataFrame(dict(q_m50=np.ones(n), qm_m50=actual, value_m50=actual, severe_m50=np.zeros(n)))
    pr = dict(P=np.ones(n), Eq=np.ones(n), Eqm=actual+20., V=actual+20.)
    r = B.read_one(df, pr, -actual)
    R = {k: {s: {d: {'J': r} for d in B.EVAL} for s in (0, 1)} for k in B.KINDS}
    gates = B.gate_of(R)
    assert gates == dict(post='PASS', retain='PASS')
    assert r['Q2 MOVE TERM'] == 'MOVE-TERM-NOT-SHOWN'
    assert (pr['V'] > 0).all() and (actual < 0).all()
    bad_quantity = dict(pr, Eq=2*np.ones(n))
    rbad = B.read_one(df, bad_quantity, -actual)
    Rbad = {k: {s: {d: {'J': rbad} for d in B.EVAL} for s in (0, 1)} for k in B.KINDS}
    assert B.gate_of(Rbad) == dict(post='FAIL', retain='FAIL')
    return dict(reproduced=True, samples=n, mapping_gate=gates,
                amount_ratio=r['amount_ratio'], value_rank=r['value_rank'], fair_rank=r['fair_edge_rank'],
                move_gate=r['Q2 MOVE TERM'], mean_predicted=float(np.mean(pr['V'])), mean_actual=float(np.mean(actual)),
                all_zero_threshold_choices_have_wrong_value_sign=True, quantity_failure_control=True)


def scenario_state():
    w = book(80.); d = np.array([300.]); amounts = np.array([5.])
    a = EX.known_steps(w, 0, 58., 50., (np.array([280.]), amounts), d, 5000.)
    b = EX.known_steps(w, 0, 58., 50., (np.array([80.]), amounts), d, 5000.)
    qa, qb = float(a['known_remaining'][0]), float(b['known_remaining'][0])
    assert (qa, qb) == (30., 25.)
    # A synthetic causal predictor demonstrates why unchanged market receipt
    # clocks do not license reusing forecasts whose OWN state changed.
    assert (qa-27.) > 0 and (qb-27.) < 0
    return dict(reproduced=True, decision_ms=300., fill_notice_delay_ms=100.,
                fill_times_ms=[280., 80.], known_remaining=[qa, qb],
                example_retained_predictions=[qa-27., qb-27.], frozen_trigger_can_differ_from_recomputed_policy=True)


def posting_disconnected():
    rows = pd.DataFrame([dict(day='20261027', oid=0, kind='post', j=0, J_V=-100.),
                         dict(day='20261027', oid=0, kind='retain', j=2, J_V=1.)])
    a = P.triggers_of(rows); rows.loc[rows.kind == 'post', 'J_V'] = 100.
    b = P.triggers_of(rows)
    assert a.equals(b) and len(a) == 0
    return dict(post_forecast_sign_does_not_change_rule=True, whole_entry_controller_connected=False)


def exported_scope():
    # Saved already-consumed Oct27 columns only, in bounded batches. No raw data.
    day = ROOT/'data/predict_fun/research/pipeline_export_20261119/20261027'
    receipt = json.loads((day/'receipt.json').read_text())
    assert receipt['source_sha256'] == snapshot['reviewed_working_sources']['predictfun/research/pf_pipeline_export.py']
    statuses = {}; qflags = {}; posts = 0; retain = 0; identities = set(); inputs = {}
    for name, h in receipt['outputs'].items():
        assert sha(day/name) == h; inputs[name] = h
        g = pd.read_parquet(day/name, columns=['oid', 'kind', 'status', 'o_qunk'], filters=[('kind', '==', 'post')])
        assert not set(g.oid) & identities and not g.oid.duplicated().any()
        identities.update(g.oid); posts += len(g)
        for k, v in g.status.value_counts().items(): statuses[k] = statuses.get(k, 0)+int(v)
        for (s, v), nn in g.groupby(['status', 'o_qunk'], dropna=False).size().items():
            key = f'{s}:{v}'; qflags[key] = qflags.get(key, 0)+int(nn)
    assert posts == receipt['stat']['post_rows']
    return dict(day='20261027', consumed=True, asof=receipt['asof'], post_rows=posts,
                statuses=statuses, post_queue_unknown_counts=qflags,
                retain_rows=receipt['stat']['retain_rows'], source_matches_snapshot=True,
                receipt_sha256=sha(day/'receipt.json'), verified_part_hashes=inputs,
                fresh_raw_tape_read=False, standalone_bench_result_present=(ROOT/'data/predict_fun/research/pipeline_bench_20261119/bench.json').exists())


def main():
    out = dict(asof=datetime.now(timezone.utc).isoformat(), base_commit=snapshot['base_commit'],
               snapshot_sha256=sha(HERE/'snapshot.json'), posting_future_information=posting_leak(),
               rejected_forecast_used=contract_bypass(), mapping_gate_false_pass=gate_bias(),
               scenario_own_state=scenario_state(), scope=posting_disconnected())
    logs = io.StringIO()
    with contextlib.redirect_stdout(logs):
        EX.selftest()
        P.selftest()
    (HERE/'selftests.txt').write_text(logs.getvalue())
    out['snapshot_selftests'] = dict(export='pass', policy='pass', benchmark='not_run_no_training_in_review')
    out['saved_export'] = exported_scope()
    out['completed_asof'] = datetime.now(timezone.utc).isoformat()
    with (HERE/'receipt.json').open('x') as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(receipt=str(HERE/'receipt.json'), reproduced=5, snapshot_selftests_passed=2)))


if __name__ == '__main__':
    main()
