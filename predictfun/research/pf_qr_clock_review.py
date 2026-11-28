"""Synthetic clock review of QR fast-birth entries, declared before running.

No raw market data, model fit or historical P&L recomputation. Reuse QR's exact
sim3, birth detector and schedule expression. A birth at t=100 with live delay
30 and unchanged L_PLACE=40 schedules its decision at t=90. Compare histories
identical through t=95, changing only that future birth's price. A causal policy
must have identical action prefixes. Controls: the 150 ms schedule is invariant
before the birth; a deliberately future-reading schedule must be refused by the
clock checker; unsorted schedules must be refused. Preserve all original QR
artifacts. Also run QR's existing synthetic controls to distinguish its existing
coverage from the new event-availability requirement.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_qr_queue_race as Q


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_available(schedule, birth, known):
    previous = float('-inf')
    for send, _, _ in schedule:
        if send < previous:
            raise ValueError('unsorted schedule')
        if send < birth or send < known:
            raise ValueError('decision precedes event availability')
        previous = send


def rejected(fn):
    try:
        fn()
    except ValueError:
        return True
    raise AssertionError('known-bad input accepted')


def run_case(future_price, delay):
    frames = [(-100., [(50, 10)], 60),
              (100., [(future_price, 5)], 60),
              (300., [(future_price, 8)], 60)]
    x = Q.syn(frames, [(5., 49, 1)], kdelay=60.)
    b0, valid, _, _ = Q.runs(x['tf'], x['P5'], x['opp'], 0)
    births = Q.fb_births(x['tf'], b0, valid, 0., x['t_end'], 0)
    schedule = [(float(x['tf'][i] + delay - Q.L_PLACE),
                 int(b0[i]), float(x['kf'][i])) for i in births]
    _, orders, _, _ = Q.run3(x, 'fbirth', 'M', sched=schedule)
    events = [dict(send_ms=float(o['ts']), live_ms=float(o['ta']),
                   price=int(o['pt'])) for o in orders]
    prefix = [o for o in events if o['send_ms'] <= 95.]
    return schedule, events, prefix


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: pf_qr_clock_review.py OUTPUT_JSON')
    out = Path(sys.argv[1])
    if out.exists():
        raise ValueError('refuse to overwrite review receipt')
    assert Q.L_PLACE == 40., 'review premise changed'
    assert not Q.selftest(), 'existing QR synthetic control failed'
    existing_mutations = Q.mutation_check()
    assert all(existing_mutations.values()), existing_mutations
    fast_a, events_a, prefix_a = run_case(51, 30.)
    fast_b, events_b, prefix_b = run_case(52, 30.)
    slow_a, _, slow_prefix_a = run_case(51, 150.)
    _, _, slow_prefix_b = run_case(52, 150.)
    assert prefix_a and prefix_b and prefix_a != prefix_b
    assert prefix_a[0]['send_ms'] == 90. and prefix_a[0]['live_ms'] == 130.
    assert slow_prefix_a == slow_prefix_b == []
    assert_available(slow_a, 100., 160.)
    refused_fast = rejected(lambda: assert_available(fast_a, 100., 160.))
    refused_unsorted = rejected(lambda: assert_available(
        [(220., 51, 160.), (210., 51, 160.)], 100., 160.))
    result = dict(
        as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='SYNTHETIC_CLOCK_FINDING_CONFIRMED',
        source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
        source_sha=sha(__file__), qr_source_sha=sha(HERE/'pf_qr_queue_race.py'),
        historical_results_recomputed=False, raw_payloads_read=False,
        assumptions=dict(birth_ms=100., current_feed_known_ms=160.,
                         common_history_until_ms=95., unchanged_place_ms=Q.L_PLACE),
        fast_30=dict(schedule_a=fast_a, schedule_b=fast_b,
                     events_a=events_a, events_b=events_b,
                     prefix_a=prefix_a, prefix_b=prefix_b,
                     prefix_invariant=prefix_a == prefix_b),
        controls=dict(existing_qr_selftest=True, existing_mutations=existing_mutations,
                      causal_150_prefix_invariant=slow_prefix_a == slow_prefix_b,
                      causal_schedule_accepted=True, future_schedule_refused=refused_fast,
                      unsorted_schedule_refused=refused_unsorted),
        interpretation='The exact QR schedule mutates a pre-birth action when only the future birth price changes. With unchanged40ms placement, the30ms arm submits10ms before the birth; it is not a causal implementation of a30ms total reaction. Historical bias size/direction is not estimated. PartA/C decomposition and all other results remain preserved. A new source+receipt is needed for a causal speed study.')
    body=json.dumps(result, indent=2, allow_nan=False)+'\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        f.write(body)
    print(json.dumps(dict(status=result['status'], source_commit=result['source_commit'],
                          prefix_a=prefix_a, prefix_b=prefix_b, controls=result['controls'])))


if __name__ == '__main__':
    main()
