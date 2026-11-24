"""Post-run presentation of every chain; no hypothesis/threshold selection."""
import csv,gzip,json,collections,io,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];O=H/'outputs'
def rows(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def sha(b):return hashlib.sha256(b).hexdigest()
a=rows(O/'chains.csv');cx=rows(O/'contexts.csv.gz');ops=rows(O/'competing_events.csv.gz');lookup={(x['order_hash'],x['stage'],x['lag_ms']):x for x in cx};res=json.loads((O/'result.json').read_text());checks=json.loads((H/'checks.json').read_text());assert checks['pass_all']
def nice(a,b,ws):return '?' if a in ('',None) or b in ('',None) else f'{(int(a)-int(ws))/1000:.3f}–{(int(b)-int(ws))/1000:.3f}s'
def mdtable(header,rs):return '| '+' | '.join(header)+' |\n|'+'|'.join('---' for _ in header)+'|\n'+''.join('| '+' | '.join(str(x).replace('|',', ') for x in r)+' |\n' for r in rs)
seq=[x for x in a if x['successor_relation']=='sequential_after_under_bounds'];assert len(seq)==7
allrows=[]
for x in sorted(seq,key=lambda z:(z['day'],int(z['rank']),int(z['mid']),z['order_hash'])):
 ec=lookup[(x['order_hash'],'ending','0')];nc=lookup[(x['order_hash'],'successor_post','0')];nlag=lookup[(x['order_hash'],'successor_post','1000')]
 allrows.append([x['day']+' / '+x['mid']+' / #'+x['rank'],x['order_hash'][:12],f"{float(x['original_shares']):g} / {float(x['observed_shares']):.2f}",x['terminal'],nice(x['end_lo_ms'],x['end_hi_ms'],x['ws_ms']),x['successor_kind']+' '+x['successor_direction'],nice(x['successor_post_lo_ms'],x['successor_post_hi_ms'],x['ws_ms']),nc['btc_side']+' / '+nlag['btc_side'],nc['inventory_side']])
phase={p:dict(records=sum(x['phase']==p for x in ops),orders=len({x['order_hash'] for x in ops if x['phase']==p}),statuses=dict(collections.Counter(x['witness_status'] for x in ops if x['phase']==p))) for p in sorted({x['phase'] for x in ops})}
repeated=[x for x in a if int(x['maker_fill_records'])>1];between={x['order_hash'] for x in ops if x['phase']=='between_fills'}
context=[]
for stage,group in [('post','all'),('ending','conditional_withdrawal'),('successor_post','sequential_after')]:
 hs={x['order_hash'] for x in a if group=='all' or (group=='conditional_withdrawal' and x['group']==group) or (group=='sequential_after' and x['successor_relation']=='sequential_after_under_bounds')}
 for lag in ('0','1000'):
  part=[x for x in cx if x['order_hash'] in hs and x['stage']==stage and x['lag_ms']==lag];context.append(dict(stage=stage,group=group,lag_ms=int(lag),orders=len(part),coverage=dict(collections.Counter(x['coverage'] for x in part)),btc=dict(collections.Counter(x['btc_side'] for x in part)),inventory=dict(collections.Counter(x['inventory_side'] for x in part)),inventory_ledger=dict(collections.Counter(x['inventory_ledger_status'] for x in part))))
changed=[dict(order_hash=x['order_hash'],day=x['day'],mid=x['mid'],old_hi_ms=int(x['old_end_hi_ms']),new_hi_ms=int(x['end_hi_ms']),tightened_ms=int(x['old_end_hi_ms'])-int(x['end_hi_ms']),witness=x['end_witness']) for x in a if x['end_witness']]
readout=dict(phases=phase,repeated_fill_orders=len(repeated),repeated_with_between_record=len({x['order_hash'] for x in repeated}&between),context=context,ending_refinements=changed,distinct_successors=len({x['successor_hash'] for x in a if x['successor_hash']}))
(H/'readout.json').write_text(json.dumps(readout,indent=2)+'\n')
maptext=f'''# Maker action and logic chains — current evidence

Built across **all 88 fixed signed-order chains**, not selected anecdotes. Consumed Oct24–27; as of {res['as_of_utc']}. We can reconstruct portions of the sequence and some repeated market-condition associations. We have not identified a complete private maker controller. Counts below are this selected chain sample, not unconditional action rates.

```mermaid
flowchart TD
 M[Spread, depth, BTC, inventory, time remaining] --> A[Visible quote creation / response]
 A -. maker identity and command-time gap .-> P[Conditional posting interval]
 P --> F[Observed partial fills; same signed budget]
 F --> K[Repeat fills / possible queue retention]
 F --> E[Exact budget exhaustion]
 K --> U[Conditional unavailability or unresolved ending]
 E --> S[Next order: refill or reprice]
 U --> S
 F --> I[Reconstructed inventory]
 I --> T[Later observed take]
 T -. exit attribution requires fill matching .-> V[Completed turnover and residual exposure]
```

## What the new population reconstruction adds

| Chain component | Result | Meaning |
|---|---:|---|
| Multiple fills on one signed order | {len(repeated)}/88 | The original size persists and successive fills consume its remaining budget. This is compatible with retaining priority, not proof of uninterrupted rest. |
| Observed budget exhaustion | 41/88 | No terminal pull is needed to explain these endings. |
| Conditional unavailability | 40/88 | Positive budget remains; unavailability timing requires the declared completeness/priority/reservation premises. Seven further orders remain unresolved. |
| Successor definitely after old ending under windows | 7/88 | 3 exhaustion→same-price refill, 2 exhaustion→reprice, 2 conditional ending→reprice. These are sequence estimates, not identified intent. |
| Successor timing overlaps | 35/88 | Sequential replacement and already-resting/overlapping quotes remain compatible. A later first fill cannot decide between them. |
| Existing successor but unusable window | 7/88 | No ordered-chain claim. |
| No recorded successor | 39/88 | Unfilled successors are not visible; this is not evidence no further order was posted. |
| Later take after the last maker fill | 81 references to 68 distinct takes | An observed later action of the same maker. It is not automatically this order's exit or an inventory reduction. |

Two additional ending upper bounds tighten. Posting bounds do not improve. There are zero new private-pull identifications and zero exhausted-order false pulls. No continuously-available-priority contradiction is found **between** the repeated fills; the exact between-fill coverage is in readout.json and must accompany that zero. It does not establish continuous rest.

## Condition → action logic we can currently support

These repeated associations come from the prior independently checked **neutral populations** (17,100 maker-seconds;11,400 market-side seconds), not a regression over the 88 filled orders. All64 comparisons remain available in [the full evidence map](../maker_condition_action_20261124/outputs/overview.csv); these rows explain the sequence without upgrading the original verdicts.

| Condition | Observed action pattern | Strength and actor gap |
|---|---|---|
| Wider spread | More fresh quote levels appear: matched ratio1.36,1.10 with1s state lag | Repeatable anonymous market quoting association; not attributed to a particular top maker. |
| Thin own touch depth | More fresh levels overall (ratio1.12 at both clocks), but fewer same-side restorations after a retreat | Different opportunity definitions produce different behaviors; thin depth is not a universal refill trigger. |
| Own maker fills accumulate | More opposing takes in the original comparison; the more tightly matched sequential result weakens with1s lag | Plausible passive-fill→inventory→take link, timing-sensitive. |
| Large absolute reconstructed inventory | Fewer same-direction takes (ratio0.51); no demonstrated increase in opposing takes (0.91) | Better evidence for restraining further exposure than automatically liquidating a large position. |
| Last30s | More deep fresh levels (ratio7.60), fewer takes (any-take0.29) | Placement is anonymous; takes are maker-attributed. No general aggressive flatten-at-expiry rule is established. |
| BTC reversal / adverse BTC | Some conditional endings are compatible, but prior post-before-reversal test was not demonstrated | No reliable “BTC reverses → this maker cancels” rule yet. The current physical reconstruction does not change that conclusion. |

The emerging working chain is: **quote in response to an opportunity; allow the same order budget to execute repeatedly; after depletion, sometimes replenish or reprice; manage resulting inventory using both passive and aggressive actions.** The quote-maker identity, continuous queue position and cause of withdrawal remain estimated or missing. This is a hypothesis structure supported to different degrees at each link, not an executable profitable strategy.

## Every ordered successor chain (all seven, no outcome selection)

Times below are seconds from that market's start. Size shows original / recorded filled shares. BTC and inventory are ranges over the entire successor posting window on the saved1s as-of grid; mixed categories mean timing uncertainty. D0 / lag1s are both shown. Inventory is reconstructed venue exposure, not private client state.

{mdtable(['Day / market / maker rank','Old hash','Size / filled','Ending','Ending interval','Next action candidate','Next posting interval','BTC side D0 / lag1s','Inventory side D0'],allrows)}

These rows answer “what sequence and what conditions are compatible?” They do not establish why an individual decided to act. Original size of a successor is unknown unless that successor itself has signed evidence. Repricing direction is geometric (toward/away from spread at the old price), not a demonstrated response to BTC or a measured improvement in economic value.

## Where the inference still breaks

- **Posting identity/time:** anonymous additions cannot reliably identify which maker posted, and first fill is still the posting upper bound.
- **Queue continuity:** repeated hashes and positive remainders do not rule out off-chain removal/reentry or transient reservations.
- **Pull vs reservation:** insufficient saved pending quantity is evidence under a scenario; hidden/failed/coalesced reservations remain possible.
- **Replacement vs ladder:** overlapping posting/ending windows cannot identify intent.
- **Take vs exit:** subsequent takes need exact inventory/lot attribution before calling them an unwind of these fills.

The next useful test is one **declared falsifiable chain hypothesis**, using the new action classes and all relevant controls, rather than another broad indicator search. Start with **depletion → same-side refill/reprice**, because exhaustion is observed. Treat “withdraw on reversal” as a separate weaker hypothesis until timing is identifiable. A same-format expansion is warranted only if it adds independent signed orders with observable successor timing; enlarging anonymous book samples alone does not close the identity gap.
'''
(H/'logic_chains.md').write_text(maptext)
print(json.dumps(readout,indent=2))
# Split the large receipt into two lossless tracked archives; retain the original.
p=O/'competing_events.csv.gz';raw=gzip.decompress(p.read_bytes());lines=raw.splitlines(keepends=True);mid=1+(len(lines)-1)//2;parts=[]
for j,body in enumerate((lines[1:mid],lines[mid:])):
 blob=lines[0]+b''.join(body);dst=O/f'competing_events_part{j:02}.csv.gz'
 with dst.open('wb') as f:
  with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0,compresslevel=1) as z:z.write(blob)
 assert dst.stat().st_size<1024*1024;parts.append(dict(file=dst.name,sha256=sha(dst.read_bytes()),rows=len(body)))
joined=gzip.decompress((O/parts[0]['file']).read_bytes())+b''.join(gzip.decompress((O/parts[1]['file']).read_bytes()).splitlines(keepends=True)[1:]);assert joined==raw
buf=io.BytesIO()
with gzip.GzipFile(filename='',fileobj=buf,mode='wb',mtime=0,compresslevel=1) as z:z.write(joined)
assert buf.getvalue()==p.read_bytes()
(H/'archive.json').write_text(json.dumps(dict(original=p.name,original_sha256=sha(p.read_bytes()),uncompressed_sha256=sha(raw),parts=parts,byte_identical_reconstruction=True),indent=2)+'\n')
