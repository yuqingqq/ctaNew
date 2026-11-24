"""Reconcile the small saved census, record limitations and publish owned state."""
import copy,csv,hashlib,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
e=json.loads((H/'evidence.json').read_text())
rows=list(csv.DictReader((H/'markets.csv').open()))
def verify(rows,e):
    assert len(rows)==1152 and len({(r['day'],r['mid']) for r in rows})==1152
    for d in e['days']:
        for scope in ['all','coverage_ok']:
            z=[r for r in rows if r['day']==d['day'] and (scope=='all' or r['coverage_ok']=='True')]
            s=d[scope];assert len(z)==s['markets']
            for k in ['accepted_maker','accepted_taker','accepted_any','recorded','excluded_only']:
                assert sum(r[k]=='True' for r in z)==s[k]
    assert all(int(r['n_records'])>0 for r in rows)
verify(rows,e)
bad=copy.deepcopy(e);bad['days'][0]['all']['accepted_maker']-=1
try:verify(rows,bad)
except AssertionError:pass
else:raise AssertionError('count corruption escaped')
badrows=copy.deepcopy(rows);badrows[-1]=badrows[0]
try:verify(badrows,e)
except AssertionError:pass
else:raise AssertionError('duplicate market escaped')
sources={}
D=Path(e['source_root'])
for p in [D/'econ/maker_market.parquet',D/'econ/identities.json']+[D/f'{kind}_{day}.parquet' for day in ['20261024','20261025','20261026','20261027'] for kind in ['markets']]:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    sources[str(p)]=h.hexdigest()
(H/'checks.json').write_text(json.dumps(dict(census_reconciles=True,all_markets_have_other_trade_records=True,corruptions_refused=2,sources=sources),indent=2)+'\n')
(H/'resources.json').write_text(json.dumps(dict(previous_packet_upper_cpu_s=163.99,probe_wall_upper_s=1.33,terminated_census_wall_upper_s=12.16,compact_first_wall_upper_s=.51,compact_v2_wall_upper_s=.51,publication_reserve_s=1.0,packet_upper_cpu_s=179.50,packet_cap_cpu_s=180),indent=2)+'\n')
report='''# Maker #3 market participation — 2026-11-24

Maker #3 traded almost every market while active. These records do not support routine selection of a small market subset as the main explanation for its better fills.

Fixed identity: Oct 23 maker-volume rank 3, `0x57FbC891E89184DA071AE9D97E745531cd7F718a`. Existing Oct 24–27 BTC five-minute registry, 1,152 markets, as of this analysis. No new data or replay; no economic or predictive test.

| UTC opening day | Registry markets | Markets with accepted maker fills | Markets with accepted taker fills |
|---|---:|---:|---:|
| Oct 24 | 288 | 288 | 278 |
| Oct 25 | 288 | 288 | 273 |
| Oct 26 | 288 | 287 | 267 |
| Oct 27 | 288 | 71 | 67 |

The first three days have maker fills in 863/864 markets (99.88%). The only absent market is 3798063, opening Oct 26 05:30 UTC; its market source has 2,423 records and no incomplete-poll flag. On Oct 27 the 71 active markets are consecutive, opening 00:00 through 05:50 UTC. From the 05:55 opening onward this address has no recorded fill. The final Oct 27 market is boundary-censored because the next-hour envelope was deliberately clipped: 216 covered no-record markets plus that one censored market. The observed pattern is an activity stop, not evidence of 217 independent market-selection decisions. Whether it stopped quoting, went offline or changed address is unidentified.

The saved checked `econ/maker_market.parquet` contains 934 accepted maker-active markets for this address across the four days. The older decision-path table's 876 was a narrower population (another 58 excluded there); it was not the complete participation numerator. Counts above include the per-market captured fill envelope, not only the first minute or a requirement to quote immediately at opening.

Coverage: every registry market has recorded trades from some participant. REST-envelope missing hours are empty; Oct 27's next-day hour is clipped. Incomplete-poll flags occur in 16/10/20/29 markets respectively and are retained, so nonparticipation is an observation of the captured records, not proof of no orders. A maker fill positively establishes passive participation even if some polling is incomplete. No accepted-only exclusions exist among this address's aggregate rows in these four days. Posting that never fills and other wallets remain unobservable.

Verification: `compact.py` reads the existing canonical aggregate and registry tables, verifies the fixed address and unique 288-market day populations. `publish.py` reconciles every reported count with `markets.csv`, refuses altered counts and duplicate market membership, and hashes the source artifacts. This does not independently rebuild the underlying fill extractor. `probe.py` read schemas; the raw-leg `census.py` attempt hit its CPU limit and produced no result. The first compact attempt refused the Oct 27 clipped envelope; v2 retains that status rather than asserting complete coverage. All attempts remain recorded. Conservative packet upper including publication: 179.50/180 CPU seconds.

Practical implication: focus comparison on when and at what price/side it starts quoting within a market, and how it manages those orders. Market-level avoidance is not an evidenced explanation for the Oct 24–26 gap. This does not establish any entry rule or profitability.

Next priority remains the already authorized saved-ledger comparison of opening-only versus default losing cycles; do not add a market-selection filter based on this census. Any new compute beyond the remaining packet allowance needs the applicable resource authorization.
'''
(H/'report.md').write_text(report)
subprocess.run(['git','diff','--cached','--quiet'],cwd=R,check=True)
block='''<!-- MAKER3_COVERAGE_20261124_CODEX -->
## 2026-11-24 — Maker #3 participates broadly, with an Oct 27 activity stop

[Census](reviews/maker3_market_coverage_20261124/report.md): fixed Oct23 rank3 address has accepted maker fills in 288/288, 288/288, 287/288 and 71/288 BTC5m markets on Oct24–27. First three days 863/864; Oct27 activity covers every opening through05:50UTC, then no recorded fills from05:55. Last market is boundary-censored; no-fill does not prove no quotes/offline/selection. Prior876 decision-path markets excluded58 and was not the full participation count934. Thus routine skipping of many markets does not explain the first-three-day comparison. Existing fill aggregate only, no replay/new data/skill claim; hashes and two count/membership refusals retained. Packet conservative upper179.50/180 CPU seconds, including the terminated raw-leg attempt. Existing opening-only scope and next priority unchanged.
<!-- /MAKER3_COVERAGE_20261124_CODEX -->

'''
state='''maker3_coverage_20261124_codex:
  report: predictfun/reviews/maker3_market_coverage_20261124/report.md
  accepted_maker_markets_by_day: [288, 288, 287, 71]
  registry_markets_per_day: 288
  observed_activity_gap_start_utc: "2026-10-27T05:55:00Z"
  quotes_without_fills_identified: false
  packet_upper_cpu_s: 179.50
  packet_cap_cpu_s: 180
'''
for rel,text in [('predictfun/HANDOFF.md',block),('predictfun/FINDINGS.md',block),('predictfun/STATUS.yml',state)]:
    p=R/rel;working=p.read_text();base=subprocess.check_output(['git','show','HEAD:'+rel],cwd=R).decode()
    assert 'MAKER3_COVERAGE_20261124_CODEX' not in working and 'maker3_coverage_20261124_codex:' not in working
    p.write_text(text+working)
    oid=subprocess.check_output(['git','hash-object','-w','--stdin'],cwd=R,input=(text+base).encode()).decode().strip()
    subprocess.run(['git','update-index','--cacheinfo','100644',oid,rel],cwd=R,check=True)
print('Census verified; receipt and owned state staged; unrelated changes preserved.')
