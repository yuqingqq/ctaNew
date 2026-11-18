"""Bounded pinned-source review: saved receipts, synthetic controls, no tape replay."""
import contextlib
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent
TARGET = '541aeca9ad9cefddd6b6346146072ea05692e1ed'
assert subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip() == TARGET
sys.path.insert(0, str(ROOT / 'predictfun/research'))
import pf_keep_pull_exec_windows as X
import pf_timing_case_audit as A
import pf_known_clock_census as N
np, pd = X.np, X.pd
DATA = Path('/home/yuqing/ctaNew/data/predict_fun/research')
SAVED = ROOT / 'predictfun/reviews/known_clock_gate_20261118'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
load = lambda p: json.loads(Path(p).read_text())
R = dict(target=TARGET, asof=datetime.now(timezone.utc).isoformat(), consumed_days=['20261108', '20261023'],
         untouched_validation_read=False, full_tape_replayed=False, training_run=False)
hashes = {}
for line in (SAVED / 'sources.sha256').read_text().splitlines():
    h, rel = line.split()
    p = Path('/home/yuqing/ctaNew') / rel if rel.startswith('data/') else ROOT / rel
    assert sha(p) == h, p
    hashes[str(p)] = h
for nm, path in [('census_v3.json', DATA/'known_clock_census_v3_20261118/census.json'),
                 ('known_v6.json', DATA/'keep_pull_exec_windows_known_v6_20261118/known.json')]:
    assert sha(path) == sha(SAVED/nm)
cj, kj = load(SAVED/'census_v3.json'), load(SAVED/'known_v6.json')
aj = load(ROOT/'predictfun/reviews/timing_case_audit_20261118/audit_v2.json')
assert aj == load(DATA/'timing_case_audit_v2_20261118/audit.json')
R['verified_hashes'] = hashes
with (OUT/'selftests.txt').open('w') as f, contextlib.redirect_stdout(f):
    assert A.selftest()
    assert N.selftest()
    X.selftest()
R['synthetic_suites'] = 'three pinned selftests passed'

# Saved small tables, no full journal or tape scan.
L = pd.read_parquet(DATA/'known_clock_census_v3_20261118/legs.parquet')
O = pd.read_parquet(DATA/'known_clock_census_v3_20261118/orders.parquet')
actual_o = N.order_census(L)
pd.testing.assert_frame_equal(actual_o, O)
read, vals = N.readings(L, actual_o, aj['legs'])
assert read == cj['readings'] and vals == cj['R']
K = pd.read_parquet(DATA/'keep_pull_exec_windows_known_v6_20261118/known_cases.parquet')
KE = pd.read_parquet(DATA/'keep_pull_exec_windows_known_v6_20261118/known_case_rows.parquet')
b, _ = X.book_bounds(K)
known, kr = X.known_checks(K, kj['R']['K0_ref'], KE, book=b)
known = {k: v for k, v in known.items() if not k.startswith('_')}
assert known == kj['R'] and kr == kj['readings']
R['saved_reproduction'] = dict(census=True, known=True, census_asof=cj['asof'], known_asof=kj['asof'],
                              legs=len(L), orders=len(O), candidate_rows=len(KE), census_readings=read,
                              per_domain=known['K6_per_domain'])

# N3's purported temporal finding is invariant to notice/answer ordering.
def order_at(notice, answer=100., cls='CANCELED'):
    return pd.DataFrame([dict(run='synthetic', cid=1, A1=cls, A1_at=answer, acknowledged=cls=='CANCELED',
                             size=5., filled=5., snapshots='[]', alerts='[]', first_notice=notice)])
def n3(rows):
    return N.readings(L, N.order_census(rows), aj['legs'])[0]['N3 CANCEL-ANSWER-ORDERS-FILLS']
before = n3(order_at(50.))
after = n3(order_at(150.))
control = n3(order_at(50., cls='NO-EFFECT'))
assert before == after and before.startswith('NO: 1 ') and control == 'NOT-CONTRADICTED'
contradictory_bounds = [dict(leg=g['leg'], premise=g['A1_premise'], bound=q)
                        for g in aj['legs'] for q in g['bounds']
                        if q['name']=='match <= cancel answer receipt (c_hi)' and q['type']=='EVIDENCE'
                        and g['A1_premise']=='NOT-ORDERED-BY-THE-ANSWER']
assert len(contradictory_bounds) == 3
cp = L.groupby(['run','cid'], sort=True).agg(cls=('A1','first'), full=('filled','first'), size=('size','first'),
                                            first_notice=('first_notice','min'), first_answer=('A1_at','min'))
mask = (cp.cls=='CANCELED') & (cp.full >= cp['size']-1e-9)
R['cancel_semantics'] = dict(synthetic_notice_before_answer=before, synthetic_notice_after_answer=after,
                             no_cancel_control=control, receipt_bound_contradictions=contradictory_bounds,
                             actual_canceled_full_orders=int(mask.sum()),
                             actual_notice_before_first_answer=int((mask & (cp.first_notice < cp.first_answer)).sum()),
                             actual_notice_after_first_answer=int((mask & (cp.first_notice > cp.first_answer)).sum()))

# A3 uses relative tolerance on epoch timestamps; corrupt only its recomputed endpoint.
plant = list(A._planted_leg())
clean = A.leg_eval(0, *plant)[0]
corrupt = copy.deepcopy(plant)
corrupt[0]['lo'] += 5000.
bad = A.leg_eval(0, *corrupt)[0]
refused = copy.deepcopy(plant)
refused[0]['lo'] += 100_000_000.
far = A.leg_eval(0, *refused)[0]
assert clean['A3_reproduces'] and bad['A3_reproduces'] and bad['A5'] == A.VERDICT_OK
assert not far['A3_reproduces'] and far['A5'] != A.VERDICT_OK
R['endpoint_tolerance'] = dict(clean_verdict=clean['A5'], shift_ms=5000., corrupted_reproduces=bad['A3_reproduces'],
                               corrupted_verdict=bad['A5'], larger_shift_control_refused=True,
                               default_relative_tolerance_ms=abs(float(plant[0]['row'].S3_lo))*1e-5)

# The domain split must retain both constituent lower bounds before transformation.
# A single WS leg has raw book lower 90050 and REST lower 90000. On the bot clock
# book offset 0 and status offset -200 make REST=90200 bind. Current split loses it.
leg = pd.DataFrame(dict(mid=[7], exec_s=[100.], first_recv_ns=[1], side=[0], ptick=[50], shares=[10.],
                        price_executed=[.5], taker_index=[1], taker_quote=[0], how=[1], t_ms=[90300.],
                        t_e2=[np.nan], tk=[5], Zr=[90300.]))
pk = pd.DataFrame(dict(mid=[7], t_ms=[90300.], ptick=[50], inc=[5.]))
key = leg[X.CKEY].assign(candidate_legs=1)
e = X.case_linkage(key, leg, pk, dict(h=600., q01=-900., q99=1050.), day='20990101')
assert e.S3_lo.iloc[0] == 90050. and np.isneginf(e.S3_lo_status.iloc[0])
short = pd.DataFrame(dict(ack=[90100.]))
got, _ = X.window_checks_dom(short, e, [0.], -200., scen=('S3',))
proper = max(90300.-X.WS_DELAY-0., 100_000.-X.REST_LAG-(-200.))
assert proper == 90200. and short.ack.iloc[0] < proper
assert got['S3']['false_cancel_first']['some_offset_per_domain'] == 0
control_e = e.copy(); control_e['S3_lo_status'] = 90000.
control_got, _ = X.window_checks_dom(short, control_e, [0.], -200., scen=('S3',))
assert control_got['S3']['false_cancel_first']['some_offset_per_domain'] == 1
R['domain_split'] = dict(raw_book_lower=90050., raw_status_lower=90000., book_offset=0., status_offset=-200.,
                          cancel_ack=90100., expected_mapped_lower=proper,
                          current_false_avoidance_count=got['S3']['false_cancel_first']['some_offset_per_domain'],
                          retained_bounds_control_count=control_got['S3']['false_cancel_first']['some_offset_per_domain'])

# Reconcile three archived raw lifecycles without scanning the live journal.
raw = [json.loads(line) for line in (ROOT/'predictfun/reviews/timing_case_audit_20261118/raw_records_v2.jsonl').read_text().splitlines()]
repro = []
for case in aj['legs']:
    ix = case['leg']; row=K.iloc[ix]
    rec = [(z['at_ms'],z['kind'],z['line']) for z in raw if z['leg']==ix and z['source']=='kernel']
    br = [z['line'] for z in raw if z['leg']==ix and z['source']=='pfmm_inputs']
    lr=A.lifecycle(row,rec,br)
    assert lr['a0'] and lr['a1']==case['A1'] and lr['a1_at']==case['A1_answer_at']
    repro.append(dict(leg=ix, records=len(rec), input_records=len(br), A0=lr['a0'], answer=lr['a1'],
                      answer_ms=lr['a1_at'], first_fill_notice_ms=min(lr['pos'])))
R['archived_raw_lifecycles']=repro
(OUT/'receipt.json').write_text(json.dumps(R, indent=2, allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in R.items() if k not in ['verified_hashes','saved_reproduction']},indent=2))
print('Saved census and all known-v6 readings reproduce; three review counterexamples reproduced.')
