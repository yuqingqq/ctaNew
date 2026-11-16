"""Full-contract causal features and joint public-flow / price labels on neutral actions."""
import argparse
from collections import Counter
import copy
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_flow_price_common as C
import pf_ev_lag_gap as G
import pf_io as IO
np, pd, I, AC = C.np, C.I.pd, C.I, C.AC


def feature_rows(W, fair, depth, chosen):
    scale, _ = I.M.pair_scales(W); po = I.X5.pair_ofi(W); out = {}
    for s, w in W.items():
        ks = np.array(sorted(set(chosen[s])), int)
        if not len(ks):
            out[s] = {}; continue
        _, valid, rs, _ = I.M.QR.runs(w['tf'], w['P5'], w['opp'], s)
        rso = I.M.run_starts(np.asarray(w['opp'], 'int64'), valid)
        raw = [I.M.all_feats(w, int(k), s, rs, rso, np.full(w['n'], np.nan), fair,
                            bool(w['vkl'][k] == I.M.MASK_V)) for k in ks]
        frame = pd.DataFrame(np.asarray(raw, np.float32), columns=I.M.FEAT)
        frame['scale_w30'] = scale[ks].astype(np.float32)
        more = I.X5.side_inputs(po, I.X5.depth_feats(depth, w['tg'][ks]), scale, s, ks)
        for name, value in more.items(): frame[name] = value.astype(np.float32)
        x = I.M.model_matrix(frame, 'W30', C.MARKET)
        out[s] = {int(k): (x[j], float(scale[k])) for j, k in enumerate(ks)}
    return out


def public_target(w, t, price, current_mid, shift=C.SHIFT):
    tt = np.asarray(w['anc']) - shift
    pr = np.asarray(w['ptl']); q = np.asarray(w['shl'])
    reach = (pr <= price) if w['side'] == 0 else (pr >= price)
    mask = (tt >= t + C.SETTINGS.cancel_ms) & (tt <= t + 1000.) & reach
    tf, mid = AC.valid_marks(w, int(w['side']))
    j = np.searchsorted(tf, t + 1000., side='right') - 1
    C.require(j >= 0 and t + 1000. - tf[j] <= 1000., 'missing end mark')
    move = (1 if w['side'] == 0 else -1) * (mid[j] - current_mid)
    result = dict(flow_quantity=float(q[mask].sum()), move_cents=float(move),
                  through_quantity=float(q[mask & (pr != price)].sum()))
    for horizon in (50, 100, 250, 500):
        result[f'flow_q_{horizon}ms'] = float(q[mask & (tt <= t + horizon)].sum())
    return result


def mark_support(w, t):
    """Uniform coverage of potential marks, independent of whether this action fills."""
    expiry = float(w['ws']) + 300000.
    if t + 2000. > expiry: return 'expiry_censored'
    tf, _ = AC.valid_marks(w, int(w['side']))
    if len(tf) == 0: return 'missing_future_marks'
    lo = np.searchsorted(tf, t + 1000., side='right') - 1
    hi = np.searchsorted(tf, t + 2000., side='right') - 1
    if lo < 0 or hi < lo or float(w['tf'][-1]) < t + 2000.: return 'missing_future_marks'
    gaps = np.diff(np.r_[tf[lo:hi+1], t + 2000.])
    if (gaps > 1000.).any(): return 'stale_future_marks'
    return 'ok'


def provenance(D, W):
    m = int(W[0]['m']); info = {}; source = {}
    for s in (0, 1):
        g = D['LT'].iloc[D['LI'].get((m, s), np.zeros(0, int))]
        g = g.iloc[np.lexsort(((-g.ptick.to_numpy() if s == 0 else g.ptick.to_numpy()), g.anc.to_numpy()))]
        C.require(np.array_equal(g.anc.to_numpy(), W[s]['anc']), 'raw leg alignment')
        for i, row in enumerate(g.itertuples()):
            info[s, i] = AC.LegInfo(str(int(row.row_id)), 'ws' if row.ws_timed else 'e2',
                                    0. if row.ws_timed else (2000. if row.imputed else 250.))
        source[s] = g
    return info, source


def build_window(day, D, W, fair, depth):
    with G.clocks(58., 37.):
        r = AC.replay_neutral(W, size=C.SIZE, shift_ms=C.SHIFT, queue='M', nr=6)
    info, src = provenance(D, W)
    tf = np.asarray(W[0]['tf']); ws = W[0]['ws']; m = int(W[0]['m'])
    gaps = np.flatnonzero(np.diff(tf) > 500.)
    cov = AC.Coverage(trade_start_ms=ws-2050., trade_end_ms=ws+300950., mark_end_ms=float(tf[-1]),
        expected_legs=len(info), book_start_ms=float(tf[0]), book_end_ms=float(tf[-1]),
        book_gaps=tuple(AC.MissingLeg(f'{m}_{j}', float(tf[j]), float(tf[j+1]), 'book_gap') for j in gaps))
    led = AC.Ledger(W, r, submitted_quantity=C.SIZE, shift_ms=C.SHIFT, leg_info=info, coverage=cov)
    rows = []; chosen = {0: [], 1: []}; valid = []
    for key, t in AC.generation_probes(led, day=day, ages_ms=C.AGES):
        s, rung, generation = key; w = W[s]
        row = dict(day=day, m=m, side=s, rung=rung, generation=generation, decision_ms=t,
                   phase=min(5, int((t-ws)//60000)+1), status='pending')
        # Future availability is checked for every potential action, even if it has no fill.
        support = mark_support(w, t)
        x = AC.label_action(led, key, t, day=day, settings=C.SETTINGS)
        row['status'] = x['status']
        if x['status'] in ('eligible', 'eligible_estimated_timing') and support != 'ok': row['status'] = support
        if row['status'] in ('eligible', 'eligible_estimated_timing'):
            k = int(np.searchsorted(w['tg'], t)); chosen[s].append(k)
            row.update(k=k, **{f'x_{a}': v for a,v in x['features'].items()})
            y = x['label']; row.update(own_quantity=y['filled_quantity'], value_cents=y['value_cents'],
                severe=float(y['value_cents'] < -30.), own_fill=float(y['filled_quantity'] > 1e-10),
                at_price_value_cents=y['at_price_value_cents'], through_price_value_cents=y['through_price_value_cents'],
                common_prefix_quantity=y['common_prefix_quantity'],
                ws_fill_quantity=sum(z['quantity'] for z in y['tranches'] if info[z['leg']].timing == 'ws'),
                imputed_fill_quantity=sum(z['quantity'] for z in y['tranches'] if info[z['leg']].uncertainty_ms==2000.))
            # Separate arithmetic audit against the neutral order's raw fills.
            mt, mm = AC.valid_marks(w, s); q = value = 0.
            for ft, fq in led.orders[key]['fl']:
                if t + 37. <= ft <= t + 1000.:
                    q += fq
                    value += fq * (1 if s == 0 else -1) * (mm[np.searchsorted(mt, ft+1000., side='right')-1] - x['features']['price'])
            C.require(np.isclose(q, row['own_quantity'], rtol=1e-10, atol=1e-8) and
                      np.isclose(value, row['value_cents'], rtol=1e-10, atol=1e-7), 'independent tranche accounting')
            valid.append((row, key, t))
        rows.append(row)
    feats = feature_rows(W, fair, depth, chosen)
    for row, key, t in valid:
        s = key[0]; x, scale = feats[s][row['k']]
        C.require(np.isfinite(x[C.MARKET.index('pf_mid')]), 'invalid feature midpoint')
        row.update(zip(C.MARKET, map(float, x)))
        O = C.V.order_inputs(pd.DataFrame([row]), [scale], [row['pf_mid']])
        row.update({name: float(O[name][0]) for name in C.ORDER})
        row.update(fp_size=C.SIZE, fp_remaining=row['x_known_remaining'], fp_queue=row['x_queue_ahead_known'],
                   fp_shown=row['x_shown_known'], fp_scale=scale, fp_phase=float(row['phase']))
        row.update(public_target(W[s], t, row['x_price'], row['pf_mid']))
        # Brute public-volume arithmetic is independent of the vectorized mask.
        q = sum(float(v) for a,p,v in zip(W[s]['anc'], W[s]['ptl'], W[s]['shl'])
                if t+37. <= a-C.SHIFT <= t+1000. and (p<=row['x_price'] if s==0 else p>=row['x_price']))
        C.require(np.isclose(q,row['flow_quantity'],rtol=1e-10,atol=1e-7), 'public quantity reconciliation')
        row['flow_class'],row['move_class'],row['joint_class'] = [int(z) for z in C.classes(row['flow_quantity'],row['move_cents'])]
        row['window_imputed_legs'] = sum(int(g.imputed.sum()) for g in src.values())
        row['window_estimated_legs'] = sum(int((~g.ws_timed).sum()) for g in src.values())
    causal_checks=0
    if valid and m % 17 == 0:
        row, key, t=valid[len(valid)//2];s=key[0];k=row['k'];rng=np.random.default_rng(m)
        mutated={a:I.M.perturb_future(w,int(w['jk'][k]),k,rng) for a,w in W.items()}
        future=depth['k']>t;changed_depth={a:v.copy() for a,v in depth.items()}
        for a in changed_depth:
            if a!='k':changed_depth[a][future]+=rng.uniform(1,5,int(future.sum()))
        choice={0:[],1:[]};choice[s]=[k]
        after=feature_rows(mutated,I.M.perturb_pb(fair,t,rng),changed_depth,choice)[s][k]
        before=feats[s][k]
        C.require(np.array_equal(after[0],before[0],equal_nan=True) and np.isclose(after[1],before[1],equal_nan=True),'future feature leak')
        # Same-time information must be able to change the instrument.
        changed=dict(W[s]);changed['X']=np.array(W[s]['X'],copy=True);changed['X'][k,0]+=10.
        bad=dict(W);bad[s]=changed
        flagged=feature_rows(bad,fair,depth,choice)[s][k][0]
        C.require(not np.array_equal(flagged,before[0],equal_nan=True),'feature control did not fire')
        order=led.orders[key];features=AC.known_state(W,order,key,t,C.SIZE,terminal_ms=led.end,settings=C.SETTINGS)
        future_order=dict(order,fl=list(order['fl'])+[(t+100.,.01)])
        unchanged=AC.known_state(W,future_order,key,t,C.SIZE,terminal_ms=led.end,settings=C.SETTINGS)
        C.require(all((np.isnan(v) and np.isnan(unchanged[a])) or v==unchanged[a] for a,v in features.items()),'unreceived fill leak')
        causal_checks=1
    return rows, dict(m=m, generations=len(rows), raw_legs=len(info), engine_fills=len(r['fl']),causal_checks=causal_checks,
                     statuses=dict(Counter(x['status'] for x in rows)), neutral_reference=r['local_reference'])


def run(day, smoke=False):
    C.verify(); C.require(day in C.DAYS, 'undeclared date')
    name=('smoke_' if smoke else '')+day; out=C.OUT/name
    C.require(not out.exists(), 'output exists'); out.mkdir(parents=True)
    start=time.time(); inputs={}
    def track(p):
        p=Path(p); C.require(p.exists(),'missing '+str(p));inputs[str(p)]=C.sha(p)
    s1=Path(I.K.IO)/'step1'/day; s5=Path(I.K.IO)/'step5'/day
    for p in [s1/'registry.parquet', s1/'clock_rl_d2.json',s1/'legs_timed2.parquet',s1/'legs_tk.parquet',
              s5/'preds_t60k.parquet',s5/'rows_rl_d2_pk.parquet']: track(p)
    imp=Path(C.V.E.OUTD)/f'day_{day}_imp.json';track(imp);offset=json.loads(imp.read_text())['untimed']['imp_off_ms']
    reg=pd.read_parquet(s1/'registry.parquet')
    lo=I._ms(f'{day[:4]}-{day[4:6]}-{day[6:]}');hi=lo+86400000
    good=(reg.ws*1000>=max(lo,I.M.boundary_cut(day)))&(reg.ws*1000<hi)&reg.outcome.isin(['Up','Down'])
    selected=reg.loc[good].sort_values('ws').mid.astype(int).tolist()
    if smoke: selected=selected[:2]
    W0={int(x.mid):float(x.ws)*1000 for x in reg.itertuples() if int(x.mid) in selected}
    C.require(W0 and not (I.SEAL_MS[0]<=min(W0.values())<I.SEAL_MS[1]) and max(W0.values())<I.FWD_MS,'protected date')
    fair,fair_stat=I.fair_fix(day,W0);hours=I.X5.depth_hours(day)
    for h in hours:
        for p in [Path(IO.BN_DIR)/(h+'.csv.gz'),Path(IO.REF_RUN)/'reference_prices'/(h+'.jsonl.gz'),Path(I.X5.DEPTH_DIR)/(h+'.csv.gz')]:
            if p.exists():track(p)
    for p in sorted((s1/'books').glob('part_*.parquet')):
        if I.part_allowed(str(p),day) and I.part_overlaps(p,W0):track(p)
    ck=json.loads((s1/'clock_rl_d2.json').read_text())
    depth,depth_stat=I.X5.depth_arrays(hours,float(ck['bn_in'])+float(ck['d_dec']))
    C.require(depth is not None,'depth unavailable');C.guard('day inputs')
    statuses=[];frames=[];load_stats=[]
    for start_at in range(0,len(selected),24):
        D=I.load_day(day,set(selected[start_at:start_at+24]),offset)
        C.require(D['stat']['untimed_outside_context_legs']==0 and D['stat']['anchored_outside_context_legs']==0,
                  'raw leg context exclusions require explicit ledger support')
        seen=[]
        for W in I.pairs(D):
            m=int(W[0]['m']);seen.append(m)
            rows,receipt=build_window(day,D,W,fair[m],depth)
            statuses.append(receipt);frames.append(pd.DataFrame(rows))
        missing=sorted(set(selected[start_at:start_at+24])-set(seen))
        for m in missing:statuses.append(dict(m=m,status='missing_frames_preds_rows',generations=0))
        load_stats.append(dict(D['stat']));del D;gc.collect();C.guard('batch')
        print(json.dumps(dict(day=day,markets_done=len(statuses),seconds=round(time.time()-start),rss_mib=round(I.K.rss_mib()))),flush=True)
    frame=pd.concat(frames,ignore_index=True);del frames
    C.require(not frame.duplicated(['day','m','side','rung','generation']).any(),'duplicate action')
    eligible=frame.status.isin(['eligible','eligible_estimated_timing'])
    C.require(not np.isinf(frame.loc[eligible,C.FEATURES].to_numpy(float)).any(),'infinite features')
    frame.to_parquet(out/'actions.parquet',index=False)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),day=day,smoke=smoke,selected_markets=len(selected),
        selected_ids=selected,registry_exclusions=int((~good).sum()),rows=len(frame),eligible=int(eligible.sum()),
        statuses=frame.status.value_counts().to_dict(),by_phase=frame.groupby(['phase','status']).size().to_dict(),
        markets=statuses,loader=load_stats,source_manifest=C.sha(C.REVIEW/'freeze_v1.json'),input_hashes=inputs,
        fair=fair_stat,depth=depth_stat,imputation_offset_ms=offset,
        output_sha256=C.sha(out/'actions.parquet'),seconds=time.time()-start))
    print(json.dumps(dict(done=name,rows=len(frame),eligible=int(eligible.sum()),seconds=round(time.time()-start))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('day');p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.day,a.smoke)
