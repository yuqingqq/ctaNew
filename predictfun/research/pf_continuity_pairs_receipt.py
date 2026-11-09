"""Verify saved LC50 lifecycle evidence; report mechanics, never simulated profit."""
from collections import Counter
import copy
import json
import subprocess
from . import pf_continuity_pairs_validation as study

s = study.s


def check_lifecycle(case, lifecycle):
    if case['name'] != lifecycle['name']: raise ValueError('lifecycle identity differs')
    active = {}
    reasons = Counter()
    admitted = 0
    events = case['events']
    places = {e['id']:e for e in events if e['kind'] == 'place'}
    for event in lifecycle['holds']:
        lane = event['lane']
        if event['kind'] == 'refused': continue
        if event['kind'] == 'open':
            if lane in active: raise ValueError('overlapping hold')
            if event['deadline'] != event['start']+50. or event['start'] != event['t']:
                raise ValueError('deadline extended')
            order = places[event['id']]
            notified = sum(e['qty'] for e in events if e['kind'] == 'fill' and e['id'] == event['id'] and e['t'] <= event['t'])
            if order['size'] > 5. or abs(order['size']-notified-event['remaining']) > 1e-8:
                raise ValueError('retained known quantity differs')
            if not any(e['kind'] == 'accepted' and e['id'] == event['id'] and e['t'] <= event['t'] for e in events):
                raise ValueError('retained unaccepted order')
            active[lane] = event; admitted += 1
        elif event['kind'] == 'close':
            opened = active.pop(lane, None)
            if opened is None or any(event[key] != opened[key] for key in ('id','start','deadline')):
                raise ValueError('hold close identity differs')
            if not opened['t'] <= event['t'] <= opened['deadline']:
                raise ValueError('hold exceeded deadline')
            if event['reason'] == 'deadline' and event['t'] != event['deadline']:
                raise ValueError('wrong deadline clock')
            if any(e['kind'] == 'place' and e['lane'] % 4 == lane and opened['t'] <= e['t'] < event['t'] for e in events):
                raise ValueError('new order while guard off')
            if any(e['kind'] == 'cancel' and e['id'] == opened['id'] and opened['t'] <= e['t'] < event['t'] for e in events):
                raise ValueError('retained order cancelled before hold closes')
            if event['reason'] not in ('same_price_reopen','other_price_reopen','order_unavailable'):
                if not any(e['kind'] == 'cancel' and e['id'] == opened['id'] and e['t'] == event['t'] for e in events):
                    raise ValueError('missing exact-time cancellation')
            reasons[event['reason']] += 1
        else: raise ValueError('unknown retention event')
    if active: raise ValueError('unclosed hold')
    return admitted, reasons


def check_rows(cases, rows, lifecycle, identities):
    if len(identities) != 228 or len(set(identities)) != 228:
        raise ValueError('wrong population')
    if any([x['name'] for x in data] != identities for data in (cases,rows,lifecycle)):
        raise ValueError('population identity differs')
    totals = Counter(); closures = Counter()
    for case, row, life in zip(cases,rows,lifecycle):
        events = case['events']; kinds = Counter(e['kind'] for e in events)
        expected = dict(events=len(events), requests=kinds['place']+kinds['cancel'], decisions=kinds['D'],
                        fills=kinds['fill'], accepts=kinds['accepted'], rejects=kinds['rejected'], retry_events=kinds['retry'])
        if any(row[k] != v for k,v in expected.items()) or row['requests_and_pair_state_match'] is not True:
            raise ValueError('native counts/state differs')
        orders = {e['id']:e for e in events if e['kind']=='place'}
        net = cash = 0.
        for event in events:
            if event['kind']=='fill':
                o=orders[event['id']]; q=event['qty']*(1 if o['lane']%2==0 else -1)
                net+=q; cash-=q*o['tick']/100.
        if len(row['known_position']) != 2 or any(abs(a-b)>1e-8 for a,b in zip(row['known_position'],(net,cash))):
            raise ValueError('known-fill ledger differs')
        admitted, reasons = check_lifecycle(case,life)
        totals.update(expected); totals['cancellation_only_probes'] += kinds['pull']; totals['holds'] += admitted
        closures.update(reasons)
    if not totals['holds'] or not totals['cancellation_only_probes'] or not totals['fills']:
        raise ValueError('positive lifecycle branches did not fire')
    return dict(totals),dict(closures)


def main():
    d=study.check(); dest=study.DEST; review=study.REVIEW
    log=(dest/'logs/matrix.log').read_text()
    if '33 passed' not in log or 'verified 228 synthetic traces' not in log or 'rc=0 ' not in log:
        raise ValueError('formal matrix incomplete')
    manifest=json.loads((dest/'manifest.json').read_text())
    if s.sha(dest/'synthetic.jsonl') != manifest['trace_sha256'] or s.sha(dest/'lifecycle.jsonl') != manifest['lifecycle_sha256']:
        raise ValueError('saved input identity changed')
    load=lambda name:[json.loads(line) for line in (dest/name).open()]
    cases,rows,lifecycle=load('synthetic.jsonl'),load('native.jsonl'),load('lifecycle.jsonl')
    totals,closures=check_rows(cases,rows,lifecycle,manifest['identities'])
    controls={}
    for key,value in [('name','wrong'),('requests',rows[0]['requests']+1)]:
        bad=copy.deepcopy(rows);bad[0][key]=value
        try:check_rows(cases,bad,lifecycle,manifest['identities'])
        except ValueError:controls[f'corrupt_{key}_refused']=True
        else:raise ValueError('corrupt artifact accepted')
    bad=copy.deepcopy(lifecycle[0]); next(e for e in bad['holds'] if e['kind']=='open')['deadline']+=1
    try:check_lifecycle(cases[0],bad)
    except ValueError:controls['extended_deadline_refused']=True
    else:raise ValueError('extended deadline accepted')
    bad=copy.deepcopy(cases[0]);opened=next(e for e in lifecycle[0]['holds'] if e['kind']=='open')
    bad['events'].append(dict(kind='place',id=999999,lane=opened['lane'],t=opened['t']+1,size=2.,tick=50))
    try:check_lifecycle(bad,lifecycle[0])
    except ValueError:controls['off_guard_replenishment_refused']=True
    else:raise ValueError('off-guard replenishment accepted')
    binary=s.TARGET/f'release/examples/{study.NAME}'
    subprocess.run([str(binary),str(dest/'synthetic.jsonl'),str(dest/'native_recheck.jsonl')],check=True)
    exact=s.sha(dest/'native.jsonl')==s.sha(dest/'native_recheck.jsonl')
    if not exact:raise ValueError('native artifact reproduction differs')
    corrupt=copy.deepcopy(cases[0]);corrupt['name']='known_bad_request'
    next(e for e in corrupt['events'] if e['kind']=='place')['size']+=1
    with (dest/'known_bad.jsonl').open('x') as f:f.write(json.dumps(corrupt,separators=(',',':'))+'\n')
    with (dest/'logs/known_bad.log').open('x') as log:
        result=subprocess.run([str(binary),str(dest/'known_bad.jsonl'),str(dest/'known_bad_native.jsonl')],stdout=log,stderr=subprocess.STDOUT)
    log=(dest/'logs/known_bad.log').read_text()
    if result.returncode==0 or 'known_bad_request' not in log or 'request mismatch' not in log:
        raise ValueError('native negative control failed for wrong reason')
    controls['native_wrong_request_refused']=True
    paths=[dest/k for k in ('declaration.json','manifest.json','synthetic.jsonl','lifecycle.jsonl','native.jsonl','native_recheck.jsonl',
                           'logs/build.log','logs/matrix.log','known_bad.jsonl','known_bad_native.jsonl','logs/known_bad.log')]
    receipt=dict(as_of=s.now(),declaration_sha256=s.sha(dest/'declaration.json'),source_commit=d['source_commit'],sources=d['sources'],
        checker_source_commit=s.command(['git','rev-parse','HEAD'],cwd=s.a.HERE),checker_sha256=s.sha(s.a.HERE/'pf_continuity_pairs_receipt.py'),
        design=d['design'],population=d['population'],build=d['build'],scope=d['scope'],limitations=d['limitations'],
        totals=totals,closures=closures,controls=controls,native_replay_byte_identical=exact,
        no_off_guard_replenishment=True,no_hold_deadline_exceeded=True,no_market_data_read=True,no_protected_dates_read=True,
        artifacts={str(p):s.sha(p) for p in paths})
    s.write(dest/'receipt.json',receipt);s.write(review/'receipt.json',receipt)
    verification=dict(as_of=s.now(),receipt_sha256=s.sha(dest/'receipt.json'),native_replay_byte_identical=exact,
                      counts_and_fill_ledger_reconciled=True,hold_lifecycle_verified=True,controls=controls)
    s.write(dest/'verification.json',verification);s.write(review/'verification.json',verification)
    previous=review.parent/'findings_v6.json'
    combined=dict(as_of=receipt['as_of'],supersedes='findings_v6.json',supersedes_sha256=s.sha(previous),
                  reason='Add bounded retention plus paired-fill lifecycle verification; no new economic evidence or strategy selection.',
                  paired_continuity=receipt,verification_sha256=s.sha(dest/'verification.json'),
                  next=['Audit remaining optimization and promotion gates against the completed bounded loop; prioritize evidence that can change the economic decision.',
                        'Historical fill calibration and actual deployed timer behavior remain required before profit or inventory-variance claims.'])
    s.write(review.parent/'findings_v7.json',combined);s.write(dest.parent/'findings_v7.json',combined)
    path=review.parent/'plan.json';plan=json.loads(path.read_text())
    plan.update(as_of=receipt['as_of'],findings='findings_v7.json',status='active_between_passes')
    plan['loop'].append(dict(round='paired_continuity',status='synthetic_lifecycle_verified',receipt='paired_continuity/receipt.json',
                            verification='paired_continuity/verification.json',finding='LC50 obeys bounded retention, fill-budget and hard-preemption controls; native effective-want/cancel lifecycle matches. No economics or deployed-timer claim.'))
    plan['next_evidence']=combined['next'];plan['next_pass'].update(status='planned',focus=combined['next'][0])
    path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(totals=totals,closures=closures,controls=controls),indent=2))


if __name__=='__main__':main()
