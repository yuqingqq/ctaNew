"""Assemble all six old examples with signed terms and order-specific economics.

Pure saved-evidence presentation; no new economic calculation/selection or fit.
Additional timeline lookup is the already-consumed Oct24 recorded fill clock.
"""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT=Path('/home/yuqing/ctaNew');HERE=Path(__file__).resolve().parent
OLD=ROOT/'predictfun/reviews/maker_trace_20261122/v2/chains';PREV=ROOT/'predictfun/reviews/maker_signed_evidence_20261124'
def need(x,msg):
    if not x:raise ValueError(msg)
def read(p):return list(csv.DictReader(p.open()))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def manifest():
    files=[Path(__file__),HERE/'check.py',HERE/'contract.md',PREV/'six_chains.csv',OLD/'post_fill/orders.csv',
           OLD/'correction_v1/post_regrade_orders.csv',OLD/'continuation_econ/orders_econ.csv',OLD/'chains_v4/chain_links.csv',
           ROOT/'data/predict_fun/research/maker_trace_v1_20261122/legs_20261024.parquet']
    with (HERE/'presentation_manifest.json').open('x') as f:json.dump({str(p.relative_to(ROOT)):sha(p) for p in files},f,indent=2);f.write('\n')


def main():
    for rel,h in json.loads((HERE/'presentation_manifest.json').read_text()).items():need(sha(ROOT/rel)==h,'source/input identity')
    for p in (Path(__file__),HERE/'presentation_manifest.json'):
        need(subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+str(p.relative_to(ROOT))])==p.read_bytes(),'presentation freeze')
    signed=read(PREV/'six_chains.csv');hashes={r['order_hash'] for r in signed};need(len(hashes)==6,'six fixed examples')
    posts={r['mk_hash']:r for r in read(OLD/'post_fill/orders.csv') if r['mk_hash'] in hashes}
    grades={r['mk_hash']:r for r in read(OLD/'correction_v1/post_regrade_orders.csv') if r['mk_hash'] in hashes}
    econ={r['order_hash']:r for r in read(OLD/'continuation_econ/orders_econ.csv') if r['order_hash'] in hashes}
    links=read(OLD/'chains_v4/chain_links.csv');need(set(posts)==set(grades)==set(econ)==hashes,'source membership')
    timelines={h:[] for h in hashes}
    for day in sorted({r['day'] for r in signed}):
        p=ROOT/f'data/predict_fun/research/maker_trace_v1_20261122/legs_{day}.parquet'
        cols=['mid','sid','tx','mk_hash','mk_idx','mk_signer','shares','y_side','y_price','settle_ms','status']
        for b in pq.ParquetFile(p).iter_batches(columns=cols,batch_size=65536):
            for r in b.filter(pc.is_in(b.column('mk_hash'),value_set=pa.array(sorted(hashes)))).to_pylist():
                timelines[r['mk_hash']].append(r)
    doc=['# Whole maker action chains — signed terms plus graded context','',
         'All six previously selected examples, consumed Oct24. Relative times below are settlementId/recorded bounds from market opening, not submission or cancellation times. New signed terms resolve final budget, not continuous resting or private motives. Maker/taker FIFO closing roles are accounting assignments; they do not prove intent.','']
    export=[]
    for r in signed:
        h=r['order_hash'];p=posts[h];g=grades[h];e=econ[h];ls=[x for x in links if x['order_hash']==h]
        ts=sorted(timelines[h],key=lambda x:(x['settle_ms'],x['sid'],x['mk_idx']))
        need(len(ts)==int(r['n_events']) and abs(sum(x['shares'] for x in ts)-float(r['observed_shares']))<1e-6,'timeline reconciliation')
        need(all(x['mk_signer'].lower()==r['maker'].lower() and str(x['mid'])==r['market'] for x in ts),'timeline owner')
        need(e['exact_link']=='available' and e['attribution_ok']=='True','exact FIFO unavailable')
        ws=float(p['ws_ms']);relative=lambda x:(float(x)-ws)/1000
        row=dict(chain=r['chain'],rank=r['rank'],order_hash=h,side=p['y_side'],price_cents=p['tick'],signed_shares=r['signed_quote_shares'],
                 observed_shares=r['observed_shares'],recorded_first_fill_s=relative(p['first_hi_ms']),recorded_last_fill_s=relative(p['last_hi_ms']),
                 scenario_post_lo_s=relative(g['post_lo_union_ms']),post_hi_s=relative(g['post_hi_ms']),post_grade=g['grade'],
                 budget_status=r['budget_status'],old_next_kind=p['next_kind'],old_next_hash=p['next_hash'],old_next_link_status=p['next_link_status'],
                 posting_inventory_min=p['pos_min'],posting_inventory_max=p['pos_max'],posting_btc_5s_set=p['btc_5s_set'],
                 shares_closing_earlier_inventory=e['earlier_sh'],earlier_inventory_close_net_usd=e['earlier_net'],
                 new_entry_close_shares=e['close_leEND_sh'],new_entry_close_net_usd=e['close_leEND_net'],
                 entry_closed_by_taker_shares=e['close_leEND_taker_sh'],entry_closed_by_maker_shares=e['close_leEND_maker_sh'],residual_shares=e['resid_sh'])
        export.append(row)
        timing='Rejected inferred interval; G1_conflict is not a valid posting-time bound.' if g['grade'].startswith('G1') else 'Scenario/bound grade must be retained; this is not a timestamped posting command.'
        doc += [f"## Chain {r['chain']} — rank {r['rank']}, market {r['market']}",'',
                f"Order `{h}`. **{p['y_side'].upper()} {p['tick']}c; signed {float(r['signed_quote_shares']):g}shares.**",'',
                f"1. **Candidate market context — causal link unresolved.** Saved inferred interval {relative(g['post_lo_union_ms']):.3f}–{relative(g['post_hi_ms']):.3f}s; grade `{g['grade']}` ({g['window_basis']}). {timing} Nearby saved state inside that interval: recorded inventory range {p['pos_min']}..{p['pos_max']}shares; aligned BTC5s set `{p['btc_5s_set']}`; spread {p['spread_min']}..{p['spread_max']}ticks. These are alternative/nearby contexts, not proven conditions before the posting decision.",
                f"2. **Order/queue — terms known, queue unknown.** Signed size {float(r['signed_quote_shares']):g}shares; private posting time, initial ahead quantity and uninterrupted resting remain unidentified.",
                f"3. **Executions — observed.** {len(ts)}recorded fills, {sum(x['shares'] for x in ts):.6f}shares, first/last settlement records at {relative(p['first_hi_ms']):.3f}/{relative(p['last_hi_ms']):.3f}s. The span is not pre-fill resting time.",
                f"4. **Ending/next order — graded.** `{r['budget_status']}`. Old next-order link: `{p['next_kind']}` / `{p['next_link_status']}`, hash `{p['next_hash']}`. Exhaustion removes the larger-cancelled-remainder explanation at final completion; successor submission timing, overlap and replacement intent remain separate.",
                f"5. **Inventory turnover — exact FIFO accounting.** {float(e['earlier_sh']):.6f}shares closed earlier inventory for ${float(e['earlier_net']):+.4f}net. New-entry completed closes: {float(e['close_leEND_sh']):.6f}shares, ${float(e['close_leEND_net']):+.4f}; maker closes {float(e['close_leEND_maker_sh']):.6f}, taker closes {float(e['close_leEND_taker_sh']):.6f}. Residual {float(e['resid_sh']):.6f}shares. These components cannot be replaced by the whole account's P&L.",
                '6. **Taking — observed executions, motive not proved.** '+next((x['summary'] for x in ls if x['link']=='takes'),'no saved take summary')+' The old episode window differs from the continuation-to-END window; a later take is an exit of this order only where the exact FIFO allocation above says so.','',
                '| Recorded settlement time after opening (s) | Shares | YES price |', '|---:|---:|---:|']
        doc += [f"| {relative(x['settle_ms']):.3f} | {x['shares']:.6f} | {100*x['y_price']:.2f}c |" for x in ts]
        doc += ['', 'The identifiable chain is **signed quote → recorded executions → inventory allocation → exhausted/unresolved budget → observed subsequent activity**. The market-condition reason for the original quote/pull and the queue path remain hypotheses.', '']
    with (HERE/'six_chain_summary.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(export[0]),lineterminator='\n');w.writeheader();w.writerows(export)
    with (HERE/'whole_chain.md').open('x') as f:f.write('\n'.join(doc)+'\n')
    with (HERE/'six_fill_timelines.json').open('x') as f:json.dump(timelines,f,indent=2);f.write('\n')
    print(json.dumps({'six_order_timelines_and_exact_economics_reconciled':True,'orders':len(export),'fills':sum(map(len,timelines.values()))}))


if __name__=='__main__':manifest() if len(sys.argv)>1 and sys.argv[1]=='manifest' else main()
