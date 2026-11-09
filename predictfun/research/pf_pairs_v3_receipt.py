"""Verify the exact-arm supplement and supersede the component-only wording."""
import copy
import hashlib
import json
import subprocess

from . import pf_pairs_v3_arm as study

s = study.s
KINDS = ('no_trade', 'reduce_fill', 'cap_fill')
NAMES = [f'v3/{side}/{kind}' for side in (0, 1) for kind in KINDS]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def reconcile(cases, rows):
    if [c['name'] for c in cases] != NAMES or [r['name'] for r in rows] != NAMES:
        raise ValueError('exact-arm population/identity differs')
    totals = dict.fromkeys(('events', 'requests', 'decisions', 'fills', 'accepts', 'rejects', 'retry_events'), 0)
    for case, row in zip(cases, rows):
        events = case['events']
        kinds = [e['kind'] for e in events]
        expected = dict(events=len(events), requests=sum(k in ('place', 'cancel') for k in kinds),
                        decisions=kinds.count('D'), fills=kinds.count('fill'), accepts=kinds.count('accepted'),
                        rejects=kinds.count('rejected'), retry_events=kinds.count('retry'))
        if any(row[k] != v for k, v in expected.items()) or row['requests_and_pair_state_match'] is not True:
            raise ValueError('native counts/state differs')
        orders = {e['id']: e for e in events if e['kind'] == 'place'}
        net = cash = 0.
        for event in events:
            if event['kind'] == 'fill':
                order = orders[event['id']]
                signed = event['qty'] * (1 if order['lane'] % 2 == 0 else -1)
                net += signed
                cash -= signed * order['tick'] / 100.
        if len(row['known_position']) != 2 or any(abs(a-b) > 1e-8 for a, b in zip(row['known_position'], (net, cash))):
            raise ValueError('known-fill ledger differs')
        for key in totals:
            totals[key] += row[key]
    return totals


def regenerate():
    from .tests.test_ladder_pairs_v3_arm import exact_case
    rows, cases = [], []
    for side in (0, 1):
        for kind in KINDS:
            paired, trace = exact_case(side, kind)
            generated, _ = exact_case(side, kind, paired=False)
            frozen, _ = exact_case(side, kind, paired=False, frozen=True)
            rows.append(dict(side=side, kind=kind, generated_sha256=digest(generated), frozen_sha256=digest(frozen),
                             paired_sha256=digest(paired), no_token_byte_identical=digest(generated) == digest(frozen),
                             resize_positive=paired['live_ctr'].get('lz_resize', 0) > 0 and paired['requests'] != generated['requests'],
                             resize_count=paired['live_ctr']['lz_resize']))
            cases.append(dict(name=f'v3/{side}/{kind}', events=trace, minimum=1., rpm=500.))
        for size in (5., 25.):
            paired, _ = exact_case(side, size=size)
            single, _ = exact_case(side, paired=False, size=size)
            rows.append(dict(side=side, size=size, kind='fixed_no_trade', paired_sha256=digest(paired),
                             single_sha256=digest(single), byte_identical=digest(paired) == digest(single)))
    return rows, cases


def check_parity(saved, regenerated):
    if saved != regenerated:
        raise ValueError('saved output hashes differ from regenerated outputs')
    if len(saved) != 10:
        raise ValueError('wrong parity population')
    for row in saved:
        keys = ('byte_identical',) if row['kind'] == 'fixed_no_trade' else ('no_token_byte_identical', 'resize_positive')
        if not all(row[k] is True for k in keys):
            raise ValueError('exact-arm parity/positive failed')


def main():
    declaration = study.check()
    dest, review = study.DEST, study.REVIEW
    log = (dest/'logs/checks.log').read_text()
    if '16 passed' not in log or 'rc=0 ' not in log:
        raise ValueError('exact-arm checks did not complete')
    cases = [json.loads(line) for line in (dest/'synthetic.jsonl').open()]
    rows = [json.loads(line) for line in (dest/'native.jsonl').open()]
    parity = json.loads((dest/'parity.json').read_text())
    totals = reconcile(cases, rows)
    regenerated, traces = regenerate()
    check_parity(parity, regenerated)
    # JSON round trips encode tuples as lists. Compare the same canonical
    # serialization used for output hashes, not Python container types.
    if digest(cases) != digest(traces):
        raise ValueError('saved traces differ from regenerated engine events')
    controls = {}
    for field, value in [('name', 'wrong-arm'), ('requests', rows[0]['requests'] + 1), ('known_position', [1., 1.])]:
        bad = copy.deepcopy(rows)
        bad[0][field] = value
        try:
            reconcile(cases, bad)
        except ValueError:
            controls[f'corrupt_{field}_refused'] = True
        else:
            raise ValueError(f'checker accepted corrupt {field}')
    bad = copy.deepcopy(parity)
    bad[0]['generated_sha256'] = '0' * 64
    try:
        check_parity(bad, regenerated)
    except ValueError:
        controls['corrupt_output_hash_refused'] = True
    else:
        raise ValueError('checker accepted corrupt output hash')
    binary = s.TARGET / f'release/examples/{study.parent.NAME}'
    subprocess.run([str(binary), str(dest/'synthetic.jsonl'), str(dest/'native_recheck.jsonl')], check=True)
    native_exact = s.sha(dest/'native.jsonl') == s.sha(dest/'native_recheck.jsonl')
    if not native_exact:
        raise ValueError('native exact-arm replay changed')
    parent = json.loads((study.parent.DEST/'receipt.json').read_text())
    combined_totals = {key: value + parent['totals'][key] for key, value in totals.items()}
    paths = [dest/name for name in ('declaration.json', 'synthetic.jsonl', 'native.jsonl', 'native_recheck.jsonl', 'parity.json', 'logs/checks.log')]
    receipt = dict(as_of=s.now(), source_commit=declaration['source_commit'], sources=declaration['sources'],
                   checker_source_commit=s.command(['git', 'rev-parse', 'HEAD'], cwd=s.a.HERE),
                   checker_sha256=s.sha(s.a.HERE/'pf_pairs_v3_receipt.py'),
                   declaration_sha256=s.sha(dest/'declaration.json'), parent_receipt_sha256=s.sha(study.parent.DEST/'receipt.json'),
                   correction=declaration['correction'], scope=declaration['scope'],
                   population=dict(synthetic_traces=len(cases), excluded_traces=0, parity_cases=len(parity), python_checks=16),
                   totals=totals, combined_native_totals=combined_totals, combined_synthetic_traces=232+len(cases),
                   combined_python_checks=35+16, parity=parity, controls=controls,
                   regenerated_outputs_and_traces_exact=True, native_replay_byte_identical=native_exact,
                   limitations=declaration['limitations'] + [
                       'Native oracle consumes Python desired quotes; it does not independently recompute the model or all policy guards.',
                       'No native kernel, wallet, venue cross-market limits, deployment scheduling, or economic benefit is verified.'],
                   no_market_data_read=True, no_protected_dates_read=True,
                   verifier_attempt_superseded=dict(log_sha256=s.sha(dest/'logs/receipt.log'),
                       reason='First verifier compared reloaded JSON lists with in-memory tuple fields. Canonical JSON comparison fixes only verification; no engine, trace or parity output changed.'),
                   artifacts={str(p): s.sha(p) for p in paths})
    s.write(dest/'receipt.json', receipt)
    s.write(review/'receipt.json', receipt)
    verification = dict(as_of=s.now(), receipt_sha256=s.sha(dest/'receipt.json'),
                        regenerated_outputs_and_traces_exact=True, native_replay_byte_identical=native_exact, controls=controls)
    s.write(dest/'verification.json', verification)
    s.write(review/'verification.json', verification)
    previous = review.parent/'findings_v5.json'
    findings = dict(as_of=receipt['as_of'], supersedes='findings_v5.json', supersedes_sha256=s.sha(previous),
                    reason=declaration['correction'], paired_engine_receipt='paired_engine_v2/receipt.json',
                    paired_engine_receipt_sha256=s.sha(study.parent.DEST/'receipt.json'), exact_v3=receipt,
                    verification_sha256=s.sha(dest/'verification.json'),
                    next=json.loads(previous.read_text())['next'])
    s.write(review.parent/'findings_v6.json', findings)
    s.write(dest.parent/'findings_v6.json', findings)
    plan_path = review.parent/'plan.json'
    plan = json.loads(plan_path.read_text())
    plan.update(as_of=receipt['as_of'], findings='findings_v6.json')
    item = next(row for row in plan['loop'] if row['round'] == 'paired_engine')
    item.update(exact_arm_receipt='paired_v3_arm/receipt.json', exact_arm_verification='paired_v3_arm/verification.json')
    plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(population=receipt['population'], totals=totals, combined=combined_totals, controls=controls), indent=2))


if __name__ == '__main__':
    main()
