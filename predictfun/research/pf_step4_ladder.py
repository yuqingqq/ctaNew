"""LADDER ENGINE (2026-11-05). A copy of pf_step4_live.py with opt-in ladder / improve / per-minute-line rules: PM<key> per-minute pull,
size AM:lo:hi:<key>, IM<key> improve inside a >= 2-tick spread, LD<K> rung one tick behind the best (kept while within K ticks),
LL<key> rung line (default the pull line), LH rung hold (a rung resting behind the best is pulled only by the Binance
guards; the guard line applies when it is placed and once it is at the best). Lines from step4/minute_lines_t60k.json. Without these tokens every arm reproduces
pf_step4_live.py. Fills carry lane / ahead0 / age_ms; invariants own_dup_price and own_cross are counted per arm.
Original docstring follows.
LIVE-DEVIATION COPY of pf_step4.py (2026-11-02, request from the pmmmrust execution-port session). pf_step4.py stays
byte-identical (C2's frozen hash 5d9bdf87); this copy adds live rules as rule tokens, every one OFF unless named:
  XNA        no amend: a same-price size-down is a cancel + new order at the back (queue = displayed size at activation)
  XL<ms>A    removal lock: a cancelled / replaced order stays fillable until decision + max(20, ms), every cancel
  XL<ms>R    same, but only for the same-price size-down replace (the A1 shrink)
  XMIN1      minimum order value 1 USD: bid p*size, ask (1-p)*size (a Down buy); smaller orders are never placed
  XS<ms>     SEQUENTIAL removal lock: a cancelled order stays fillable until decision + max(20, ms); the side stays
             empty after that and its new order (the then-current intent) activates 20 ms later; activations arriving
             while the side waits are absorbed
  XP<ms>     PLACEMENT-RELATIVE lock (Predict's removalLockedUntil), sequential: a cancelled order stays fillable until
             max(decision + 20, its activation + ms); the side's then-current intent activates 20 ms after that
  XQ<ms>     PLACEMENT-RELATIVE lock, CONCURRENT: a cancelled order stays fillable (beside its replacement, which
             activates at decision + 20 ms) until max(decision + 20, its activation + ms); G6 does not see it
  XT<r>      request throttle (exec-port spec): places = bucket 4 refilling r/s, a place needs a whole token; cancels =
             bucket 8 refilling 2/s, always sent (may go negative); a place also needs cancel tokens >= open orders + 1
             and < 3 unanswered places on its side (answered after 40 ms); a refused place retries while its intent
             stands after 250 ms, doubling to 4 s; a changed intent resets the backoff
  XV / XV<rpm>  venue-sized throttle (XV<rpm>: refill <rpm> per minute instead of 500) (Predict's create/remove headers: 40 per s, 500 per min): one shared bucket of 40
             refilling 500/60 per s; cancels always sent (may go negative); a place needs tokens >= open orders + 2;
             < 3 unanswered places per side; retry 250 ms doubling to 4 s while the intent stands
  XCAP<usd>  total-loss cap: a place is refused if realised + this window's worst case (open buys on the losing side,
             incl. the new order) < -usd; realised <= -usd halts all later windows. Windows run in start-time order;
             state carries across days through PF_LIVE_CAPSTATE_IN -> <out>/capstate.json.
Scenario 'tokyoreal' (2026-11-02, user: real latency values): the measured Tokyo clock of pf_clock_rl (Predict frames
per frame at Tokyo's 53.1 ms median delay + 10 ms decision, Binance E + 1.5 + 10, orders and cancels live 22 ms after
sending, own fills known 30 ms later); needs step1/<day>/clock_rl.json from pf_step5_features_rl.
Split latency (tokyoreal with PF_RL_LPLACE / PF_RL_LCANCEL set; 2026-11-03, measured from Predict's wallet-event stamps):
requests are sent at the decision; a place goes live L_PLACE later (post-only checked then), a cancel takes effect L_CANCEL
after sending, or L_CANCEL after the order goes live if it was sent while the create was pending (the venue processes the
remove after the create). Concurrent: the old order stays fillable until its cancel takes effect, the new one is live
from its own activation (the side can be empty in between). Sequential (XS0): the side's next order is sent only when
the cancel has taken effect. Live-like arms only (no amend).
Round O3 tokens (2026-11-03, optimisation at the Tokyo clock; every one off unless named):
  B1T03      G1 (Binance 1 s) threshold 0.3 bps instead of 0.5
  B2T03 / B2W100 / B2W100T03   G2 (Binance short leg) threshold 0.3 bps / window 100 ms / both, instead of 250 ms, 0.5 bps
  BKA / BK<s> / BKM            stand back: quote one tick behind the best price (bid b1 - 1, ask a1 + 1) always / from
             <s> seconds into the window / when the 1 s prediction is mildly against the side (>= 0.1 c); G4 skips such quotes
  HY<ms>     reprice dead-band: a new price for a quoted side is sent only after it has persisted <ms> (pulls, first
             places and same-price size changes act at once)
  size A7:lo:hi:par            A1 sizing (hi when the 1 s prediction favours the side, >= 0) before <par> s, lo after
Round O4 tokens (2026-11-03, sweep protection; off unless named):
  BKD2       with BK*: stand back two ticks (bid b1 - 2, ask a1 + 2) instead of one
  CD<ms>     touch cooldown: when the known best price on a side moves away (bid touch down / ask touch up), withhold
             that side for <ms>
  FC<ms>     fill cooldown: after an own fill becomes known on a side, withhold that side for <ms>
Leftover-control tokens (2026-11-03, user: capture spread, keep leftovers small; off unless named):
  CAP<n>     inventory cap: the side that would increase |net| is withheld once |known net| >= n shares
  RO<s>      reduce-only from <s> seconds: only the side that reduces |net| is quoted, sized <= |net|
Always recorded (no behaviour change): per-window P&L marked to the venue mid at stop + 100 ms (mtm_stop: spread captured
minus adverse selection, without the outcome) and the remaining carry to settlement (carry_stop = net x (outcome - mid)),
plus: requests (place / cancel / amend, at decision time), per-window min worst case at
place attempts, order size per fill, zombie (locked) fills, Binance raw-row gaps > 1 s and Predict frame gaps > 5 s / 30 s
inside live windows, registration lead (ws - first_seen). Output under step4/live_<scenario>_...
Original docstring follows.

Step 4 (exploratory): baseline + D1 rules one at a time, under a Tokyo-host latency scenario.

Clock: venue/UTC ms. Latency scenarios (user 2026-11-01: Tokyo host, ~20 ms to Predict):
  tokyo20  Predict frame known at updateTimestampMs + 20; Binance row known at E + 5; order/cancel active 20 ms after decision
  tape     Predict and Binance known at this host's receipt - 72 ms (both venues ~74 ms one way from here; keeps
           Predict's measured publication delay, ~58 ms median); orders active 20 ms after decision
Decisions happen at every known Predict frame, at every Binance guard flip (D1's extra syncs) and at 240 s.

Rules (D1 pieces, each switchable):
  G1 Binance 1 s against-side withhold: m = (mid_i/mid_j - 1)*1e4, i = last row known <= t, j = last <= t - 1 s;
     BUY withheld if m <= -0.5, SELL if m >= +0.5; unavailable (i older than 1 s) = not adverse
  G2 Binance 250 ms leg (TICK_UNION bps part): per row, 1e4*ln(mid_i/mid_j), j = last row <= row - 250 ms; same thresholds
  G3 gap guard: desired level at the touch and the next level behind >= 2 ticks away (or none) -> withhold
  G4 NO_OFF_TOUCH: desired level behind the touch -> withhold
  G5 STOP240: no quotes from 240 s
  G6 inventory TAPER10_HARD25: increasing side withheld at |net| >= 10; hard 25 incl. live size; final 60 s reduce-only
Fill model as step 3 (FIFO behind displayed size, trades at our price advance us, displayed-size cap,
sweep fills up to the sweeping leg's size, post-only refusal at activation, timed legs exact,
untimed at S-0.5 / S / S+0.5 s).

Additions 2026-11-01 (size / optimisation round; default arms unchanged):
  arms carry an order size; inventory caps scale with it (soft 2 x size, hard 5 x size = D1's 10/25 at size 5)
  M<d> = 1 s model guard at theta 0.<d...> c style tokens: M05 0.5, M10 1.0, M025 0.25, M075 0.75
  Q<d> = 250 ms model guard (pred_250), same token style; S<sec> = stop time for G5 and G6's final rule (default 240)
  SBV   = with V<d>: a flagged side is quoted one tick behind the best (bid b1 - 1, ask a1 + 1) instead of withheld (2026-11-04)
  SBP   = with V<d>: a flagged side is re-priced from the 1 s prediction p: bid min(b1 - 1, floor(mid + p - 0.5)), ask
          max(a1 + 1, ceil(mid + p + 0.5)), at most 5 ticks behind the best; one tick when no fresh prediction (2026-11-04)
  V<d> = per-side guard (2026-11-04, stage 3): the bid is pulled when v_bid <= -theta, the ask when v_ask <= -theta, from the same
         predictions file (columns v_bid / v_ask, e.g. preds_g20: a model's value of keeping each side, shifted so that 0 is the
         chosen pull quantile); same staleness rule as M (100 ms); arms without V tokens are unchanged
  PF_PREDS = prediction file stem under step5/<day>/ (default preds); PF_SIZES for the size armset
Adaptive size (2026-11-01, round O2): size spec "K:lo:hi:par" chosen per side at each decision; caps use hi.
  A1 model: hi if the 1 s prediction favours the side (bid: pred >= par, ask: pred <= -par), else lo
  A3 depth: clip(round(par x displayed size at our price), lo, hi)
  A4 inventory: hi on the side that reduces |net|, else lo
  A6 time: hi before par seconds into the window, else lo
  AV guard value (2026-11-04): hi if the side's trained-guard value (v_bid / v_ask of the predictions file, fresh <= 100 ms)
     is >= par c, else lo; decisions also fire at its crossings
  A live order at the same price is amended DOWN in place (queue kept) or cancel/replaced UP (queue reset).
Fill model v2 (default since the 2026-11-01 code review; PF_FILLMODEL=v1 reproduces the earlier numbers):
  - queue position is capped by the displayed size of every venue frame STRICTLY BEFORE the event (a frame stamped at a
    trade's own ms already shows that trade, so v1 counted it twice); frames where our level is beyond the 5 shown
    levels do not cap; activation and post-only use the frame strictly before activation
  - a fill swept through our level leaves no queue ahead of us
  - our own fills are known L_FILL = 20 ms later (inventory, caps, A4, repost after a full fill)
  - decisions also fire when a model prediction goes stale / resumes, and at A1's size threshold crossings
  - reduce-only quotes after the stop are capped at |inventory|
  G6's hard cap (5 x size) never binds given the soft cap (2 x size): |net| < 2c, live <= c, new <= c.
Usage: python3 -m predictfun.research.pf_step4 <day> <scenario> <armset>
  armset: singles  (base, base+G1..G6 for JOIN and MICRO; timing 'mid'; base also early/late)
          stack    (cumulative D1 order G4, G2, G6, G5, G3, G1 for MICRO and JOIN; all three timings)
          ends     (base and ALL only, all three timings)
          size     (MICRO + D1 stack, and + M05 at all timings, for each size in PF_SIZES)
          opt      (MICRO + D1 stack + M05 variants: theta, 250 ms guard, stop time, leave-one-rule-out; mid, size 5)
          custom   (PF_ARMS = "POLICY|R1+R2+...|timing|size;..."; output dir suffix PF_TAG)
"""
import sys, os, math, bisect, heapq, collections, json, itertools
import numpy as np, pandas as pd
from . import pf_io as io

day, SCEN, ARMSET = sys.argv[1], sys.argv[2], sys.argv[3]
S1 = f'{io.OUT}/step1/{day}'
PREDS = os.environ.get('PF_PREDS', 'preds')
OUT = (f"{io.OUT}/step4/ladder_{SCEN}_{ARMSET}{'' if os.environ.get('PF_TIMING', 'v1') == 'v1' else '_' + os.environ['PF_TIMING']}"
       f"{'' if PREDS == 'preds' else '_' + PREDS}{('_' + os.environ['PF_TAG']) if os.environ.get('PF_TAG') else ''}/{day}")
os.makedirs(OUT, exist_ok=True)
SIZE = 5.0; NET_MS = 72.0; L_O = 20.0
P_IN = 20.0; B_IN = 5.0
LEGACY_INTENT = os.environ.get('PF_LEGACY_INTENT') == '1'   # diagnostic: reproduce pre-O2 intent handling
FILLMODEL = os.environ.get('PF_FILLMODEL', 'v2')              # v1 = pre-review fill handling (regression only)
V2 = FILLMODEL != 'v1'
L_FILL = 20.0
SPLIT = False
if SCEN == 'tokyoreal':
    from . import pf_clock_rl as CK
    L_O, L_FILL, B_IN = CK.L_ORDER, CK.L_FILL, CK.BN_IN + CK.D_DEC
    RL_SHIFT = CK.load_shift(S1)
    SPLIT = CK.L_PLACE is not None
    if SPLIT:
        L_PL, L_CA, L_ANS = CK.L_PLACE, CK.L_CANCEL, CK.L_ANSWER

reg = pd.read_parquet(f'{S1}/registry.parquet').set_index('mid')
TXTIME = os.environ.get('PF_TXTIME', '')        # '' (per-leg times, default) / 'ws' / 'min': one time per taker transaction
assert TXTIME in ('', 'ws', 'min'), TXTIME
TXSHIFT = float(os.environ.get('PF_TXSHIFT', '0')); TXSHIFT_E2 = float(os.environ.get('PF_TXSHIFT_E2', '0'))
assert TXSHIFT_E2 == 0 or TXTIME == 'ws', 'PF_TXSHIFT_E2 needs PF_TXTIME=ws'
assert TXSHIFT == 0 or TXTIME, 'PF_TXSHIFT needs PF_TXTIME'
TXSTAT = None
# 2026-11-05 review fixes (user-reported, reproduced on synthetic inputs):
#  PF_FILLPRIO=price (default): our own orders on a side trade best price first, then oldest first at one price (was: cancel-pending
#    orders first regardless of price, so an old 49 c bid could fill before its live 50 c replacement); 'old' = regression only.
#  PF_OWNCROSS=on (default): post-only at activation also refuses an order that would cross our own resting opposite order, live or
#    cancel-pending (was: the venue book only); 'off' = regression only. Counted: postonly_refused_own, own_cross (incl. cancel-pending).
#  GS token: no guard value for a side (missing or > 100 ms old) withholds that side (default arms: quote, as before).
#  PF_LEGORDER=price (default): legs at one time on one side run the maker side's best price first (bids high -> low, asks
#    low -> high); was ascending price on both sides, so on the bid side a worse leg's through-fill cleared our queue ahead before
#    the better leg ran (overfill); 'old' = regression only. Counted: leg_groups_reordered.
#  PF_REJRESET=on (default): a post-only rejection against the venue book clears the lane's intent, so the next decision re-sends
#    the quote if still wanted (was: intent kept, same quote never re-sent); 'off' = regression only. Counted: rej_reset.
#  PF_POLICYVIEW=known (default): the strategy sees its orders through acknowledgements - an order enters its view at activation and
#    leaves when the strategy cancels it or when the fill notification (L_FILL later) reports it done; partial fills reduce the open
#    size the strategy sees (inventory caps) only when notified (was: the matcher's state, so a fill was acted on at once -
#    look-ahead); 'physical' = regression only. Counted: acted_on_unknown_fill (a cancel sent for an order already filled, not yet
#    reported), view_filled_unknown (decision reads where the venue had filled the order and the strategy did not know yet).
#  Request reservations, including cancel-pending creates, expire only on observable cancellation/rejection/fill acknowledgements.
#  POLICYVIEW=known also values the loss cap using notified cash/inventory and policy-visible open orders.
POLICYVIEW = os.environ.get('PF_POLICYVIEW', 'known'); assert POLICYVIEW in ('known', 'physical'), POLICYVIEW
FILLPRIO = os.environ.get('PF_FILLPRIO', 'price'); OWNCROSS = os.environ.get('PF_OWNCROSS', 'on')
LEGORDER = os.environ.get('PF_LEGORDER', 'price'); REJRESET = os.environ.get('PF_REJRESET', 'on')
assert FILLPRIO in ('price', 'old') and OWNCROSS in ('on', 'off') and LEGORDER in ('price', 'old') and REJRESET in ('on', 'off'), (FILLPRIO, OWNCROSS, LEGORDER, REJRESET)
QACT = os.environ.get('PF_QACT', '') == 'max'   # review sensitivity (2026-11-04): queue at activation = max(frame before, first frame after <= 100 ms)
# second review (2026-11-05): PF_TXKEY=tk groups legs by the real taker order (step1/<day>/legs_tk.parquet from pf_txmap; unmapped legs
# keep their own time); PF_TXJIT=frame moves each transaction back by U(0, 1) x the gap to the previous frame of its market (capped
# 100 ms; one draw per transaction, seed = day): the match happened between that frame and the one that shows it.
TXKEY = os.environ.get('PF_TXKEY', ''); TXJIT = os.environ.get('PF_TXJIT', '')
assert TXKEY in ('', 'tk') and TXJIT in ('', 'frame'), (TXKEY, TXJIT)
assert not (TXKEY or TXJIT) or TXTIME, 'PF_TXKEY / PF_TXJIT need PF_TXTIME'
SEALED = ('20261028', '20261029', '20261030'); FORWARD = '20261104'   # no input from a sealed or forward day (padding included)
assert day not in SEALED and day < FORWARD, f'REFUSED: {day} is sealed or forward'
ok_day = lambda d8: d8 not in SEALED and d8 < FORWARD
_D0 = pd.Timestamp(f'{day[:4]}-{day[4:6]}-{day[6:]}', tz='UTC').value // 1_000_000
T_LO = _D0 if not ok_day((pd.Timestamp(_D0 - 1, unit='ms')).strftime('%Y%m%d')) else -np.inf      # ms bounds of admissible input
T_HI = _D0 + 86_400_000 if not ok_day((pd.Timestamp(_D0 + 86_400_000, unit='ms')).strftime('%Y%m%d')) else np.inf
GID = None
TIMING = os.environ.get('PF_TIMING', 'v1')     # v1: WS exact + bracket; e2: WS + E2 point estimate; e4: WS + E4 exact + bracket
POLICIES = os.environ.get('PF_POLICIES', 'JOIN,MICRO').split(',')
if TIMING == 'v1':
    LT = pd.read_parquet(f'{S1}/legs_timed.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how'])
else:
    TXK = ['first_recv_ns', 'taker_index', 'taker_quote', 'price_executed']
    LT = pd.read_parquet(f'{S1}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how', 't_e2', 't_e4'] + (TXK if TXTIME else []))
    alt = LT[f't_{TIMING}'].values
    use = (LT.how.values == 0) & np.isfinite(alt)
    if TXTIME:                                       # one time per taker transaction (2026-11-04 review): all its legs at once
        ws_t = np.where(LT.how.values > 0, LT.t_ms.values.astype(float), np.nan); e2_t = np.where(use, alt, np.nan)
        key = ['mid', 'exec_s'] + TXK
        if TXKEY == 'tk':
            tkf = pd.read_parquet(f'{S1}/legs_tk.parquet')
            assert len(tkf) == len(LT), 'legs_tk not aligned with legs_timed2'
            LT['tkg'] = np.where(tkf.tk.values >= 0, tkf.tk.values, -1 - np.arange(len(LT)))   # unmapped: a group of its own
            n_tk_status = {int(k): int(v) for k, v in zip(*np.unique(tkf.tk_status.values, return_counts=True))}; del tkf
            n_oldkey = int(LT.groupby(key, sort=False, dropna=False).ngroups); key = ['tkg']
        if TXTIME == 'ws':                           # earliest WS-timed leg, else earliest estimated leg
            a_ = pd.Series(ws_t, index=LT.index).groupby([LT[c] for c in key], sort=False, dropna=False).transform('min').values
            b_ = pd.Series(e2_t, index=LT.index).groupby([LT[c] for c in key], sort=False, dropna=False).transform('min').values
            anc = np.where(np.isfinite(a_), a_, b_)
            anc = anc + np.where(np.isfinite(a_), 0.0, TXSHIFT_E2)   # PF_TXSHIFT_E2: E2-only transactions earlier (E2 error sensitivity)
        else:                                        # 'min': earliest leg of either kind
            anc = pd.Series(np.fmin(ws_t, e2_t), index=LT.index).groupby([LT[c] for c in key], sort=False, dropna=False).transform('min').values
        old = np.where(LT.how.values > 0, ws_t, e2_t); ch = np.isfinite(anc) & (np.abs(anc - old) > 1e-6)
        TXSTAT = dict(mode=TXTIME, legs=int(len(LT)), legs_retimed=int(ch.sum()), shares_retimed=float(LT.shares.values[ch].sum()),
                      untimed_given_time=int((np.isfinite(anc) & ~np.isfinite(old)).sum()),
                      abs_shift_ms_p50=float(np.median(np.abs(anc - old)[ch])) if ch.any() else 0.0)
        if TXKEY == 'tk':
            TXSTAT.update(key='taker order (tx hash + taker hash)', tk_status_legs=n_tk_status, groups_old_key=n_oldkey, groups_tk=int(LT.groupby(key, sort=False).ngroups))
        tim = np.isfinite(anc); anc = anc + TXSHIFT          # PF_TXSHIFT: sensitivity, match earlier than the frame that shows it
        kk = [LT[c] for c in key]                            # instrument: transactions whose timed legs carry > 1 distinct time
        nb = pd.Series(np.where(np.isfinite(old), old, np.nan)).groupby(kk, sort=False, dropna=False).nunique()
        na = pd.Series(np.where(tim, anc, np.nan)).groupby(kk, sort=False, dropna=False).nunique()
        TXSTAT['tx_multi_time_before'] = int((nb > 1).sum()); TXSTAT['tx_multi_time_after'] = int((na > 1).sum())   # 'after' = 0 by construction (invariant, not evidence)
        GID = pd.factorize(pd.MultiIndex.from_arrays(kk) if len(kk) > 1 else kk[0])[0]
        if TXSTAT['tx_multi_time_after']:
            sys.exit(f'REFUSED: transaction timing left {TXSTAT["tx_multi_time_after"]} transactions with more than one time')
        LT['t_ms'] = np.where(tim, anc, LT.t_ms.values); LT['how'] = np.where(tim, np.where(LT.how.values > 0, LT.how.values, 3), LT.how.values)
        LT = LT.drop(columns=TXK + (['tkg'] if TXKEY == 'tk' else [])); print('[tx timing]', TXSTAT, flush=True)
    else:
        LT['t_ms'] = np.where(use, alt, LT.t_ms.values); LT['how'] = np.where(use, 3, LT.how.values)
    LT = LT.drop(columns=['t_e2', 't_e4'])
LT_BY = {m: g for m, g in LT.groupby('mid', sort=False)}
PRED = {}; PRED250 = {}; PREDVB = {}; PREDVA = {}
if ARMSET in ('model', 'size', 'opt', 'custom'):
    pr = pd.read_parquet(f'{io.OUT}/step5/{day}/{PREDS}.parquet', columns=['mid', 'tau', 'pred_1000', 'pred_250'])
    for col, dst in (('pred_1000', PRED), ('pred_250', PRED250)):
        q_ = pr[pr[col].notna()]
        dst.update({m: (g.tau.values.astype(np.float64), g[col].values.astype(np.float64)) for m, g in q_.groupby('mid', sort=False)})
    del pr, q_
    import pyarrow.parquet as _pq
    if 'v_bid' in _pq.ParquetFile(f'{io.OUT}/step5/{day}/{PREDS}.parquet').schema.names:
        pv = pd.read_parquet(f'{io.OUT}/step5/{day}/{PREDS}.parquet', columns=['mid', 'tau', 'v_bid', 'v_ask'])
        for col, dst in (('v_bid', PREDVB), ('v_ask', PREDVA)):
            q_ = pv[pv[col].notna()]
            dst.update({m: (g.tau.values.astype(np.float64), g[col].values.astype(np.float64)) for m, g in q_.groupby('mid', sort=False)})
        del pv, q_
cols = ['mid', 't_ms', 'recv_ns'] + [f'{s}{i}_{k}' for s in 'ba' for i in range(1, 6) for k in 'ps']
books = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=cols) for f in sorted(os.listdir(f'{S1}/books'))], ignore_index=True)
books.sort_values(['mid', 't_ms', 'recv_ns'], inplace=True, kind='stable')
_nb0 = len(books); books = books[(books.t_ms.values >= T_LO) & (books.t_ms.values < T_HI)].reset_index(drop=True)
INPUT_BOUNDS = dict(t_lo_ms=float(T_LO), t_hi_ms=float(T_HI), book_frames_dropped=int(_nb0 - len(books)))
if TXJIT:                                            # within-frame match time (see PF_TXJIT above)
    FRM = {m: np.unique(g.t_ms.values.astype(float)) for m, g in books[['mid', 't_ms']].groupby('mid', sort=False)}
    assert GID is not None and GID.min() >= 0, 'transaction ids'
    u = np.random.default_rng(int(day)).random(int(GID.max()) + 1)[GID]
    t0 = LT.t_ms.values.astype(float); t1 = t0.copy(); gap = np.zeros(len(LT))
    for m, ix in LT.groupby('mid', sort=False).indices.items():
        f = FRM.get(m)
        if f is None:
            continue
        tt = t0[ix]; fin = np.isfinite(tt); j = np.searchsorted(f, tt[fin], side='left') - 1
        g_ = np.where(j >= 0, tt[fin] - f[np.clip(j, 0, None)], 0.0); gg = np.zeros(len(ix)); gg[fin] = np.clip(g_, 0.0, 100.0); gap[ix] = gg
    t1 = np.where(np.isfinite(t0), t0 - u * gap, t0); sh_ = (t0 - t1)[np.isfinite(t0)]
    TXSTAT.update(jitter='frame', jitter_mean_ms=float(sh_.mean()), jitter_p50_ms=float(np.median(sh_)), jitter_zero_gap_legs=int((gap[np.isfinite(t0)] == 0).sum()))
    assert 0 < TXSTAT['jitter_mean_ms'] <= 50, TXSTAT
    LT['t_ms'] = t1; LT_BY = {m: g for m, g in LT.groupby('mid', sort=False)}; print('[tx jitter]', {k: TXSTAT[k] for k in TXSTAT if k.startswith('jitter')}, flush=True)

# ---- Binance futures: knowledge time (ms) + mid, whole day +-1 h ----
# Only rows where the mid or the 250 ms-leg class changes are kept. Exact for both guards: G1 between kept
# rows sees an unchanged mid (m = 0, or "unavailable" = not adverse, same decision); G2's class is held per row.
kb, mb, cb = [], [], []
carry_k = np.empty(0); carry_m = np.empty(0); last_m = np.nan; last_c = 99
n_raw = 0
BN_GAPS = []; last_raw = -np.inf
G2V = {'B2T03': (250.0, 0.3), 'B2W100': (100.0, 0.5), 'B2W100T03': (100.0, 0.3)}   # round O3 short-leg variants
NEEDV = sorted({G2V[x] for x in G2V if x in os.environ.get('PF_ARMS', '').replace('|', '+').split('+')})
cbv = {v: [] for v in NEEDV}; last_cv = {v: 99 for v in NEEDV}
for h in [h for h in io.hours_of(day, extra_after=1, extra_before=1) if ok_day(h[:8])]:
    p = f'{io.BN_DIR}/{h}.csv.gz'
    if not os.path.exists(p):
        continue
    df = pd.read_csv(p, header=None, names=['recv', 'E', 'T', 'u', 'bid', 'bq', 'ask', 'aq'], usecols=['recv', 'E', 'u', 'bid', 'ask'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df = df[(df.bid > 0) & (df.ask > df.bid)].drop_duplicates('u', keep='first')
    k = (df.E.values + B_IN) if SCEN in ('tokyo20', 'tokyoreal') else (df.recv.values / 1e6 - NET_MS)
    m = (df.bid.values + df.ask.values) / 2
    del df
    o = np.argsort(k, kind='stable'); k = k[o].astype(np.float64); m = m[o].astype(np.float64)
    n_raw += len(k)
    kk = np.r_[last_raw, k]; gi = np.flatnonzero(np.diff(kk) > 1000.0)      # live rule: Binance silent > 1 s
    BN_GAPS.extend(zip(kk[gi].tolist(), kk[gi + 1].tolist())); last_raw = k[-1]
    K = np.concatenate([carry_k, k]); M = np.concatenate([carry_m, m])
    j = np.searchsorted(K, k - 250.0, side='right') - 1
    bp = np.where(j >= 0, 1e4 * np.log(m / M[np.clip(j, 0, None)]), np.nan)
    c = np.where(bp <= -0.5, -1, np.where(bp >= 0.5, 1, 0)).astype(np.int8)
    keep = (m != np.r_[last_m, m[:-1]]) | (c != np.r_[last_c, c[:-1]])
    cvs = {}
    for (w_, th_) in NEEDV:
        jv = np.searchsorted(K, k - w_, side='right') - 1
        bpv = np.where(jv >= 0, 1e4 * np.log(m / M[np.clip(jv, 0, None)]), np.nan)
        cvs[(w_, th_)] = cv = np.where(bpv <= -th_, -1, np.where(bpv >= th_, 1, 0)).astype(np.int8)
        keep |= cv != np.r_[last_cv[(w_, th_)], cv[:-1]]
    kb.append(k[keep]); mb.append(m[keep]); cb.append(c[keep])
    for v_, cv in cvs.items():
        cbv[v_].append(cv[keep]); last_cv[v_] = cv[-1]
    last_m = m[-1]; last_c = c[-1]
    tail = k >= k[-1] - 1000.0; carry_k = k[tail]; carry_m = m[tail]
    del K, M, j, bp, c, keep, k, m
    io.mem_check(f'bn {h}')
KB = np.concatenate(kb); MB = np.concatenate(mb); BC = np.concatenate(cb)
o = np.argsort(KB, kind='stable'); KB = KB[o]; MB = MB[o]; BC = BC[o]
BCV = {v_: np.concatenate(x)[o] for v_, x in cbv.items()}
del kb, mb, cb, o
print(f'[in] {day} {SCEN} books {len(books)} legs {len(LT)} binance raw {n_raw} kept {len(KB)} [mem] {io.rss_mib()} MiB', flush=True)


def guard_state(t, use1, use2, g1=0.5, g2v=None):
    """(buy_adverse, sell_adverse) at knowledge time t (g1 = G1 threshold bps; g2v = short-leg variant key or None)."""
    i = int(np.searchsorted(KB, t, side='right')) - 1
    if i < 0:
        return False, False
    ba = sa = False
    if use1 and t - KB[i] <= 1000.0:
        j = int(np.searchsorted(KB, t - 1000.0, side='right')) - 1
        if j >= 0:
            m = (MB[i] / MB[j] - 1.0) * 1e4
            ba |= m <= -g1; sa |= m >= g1
    if use2:
        bc_ = BC if g2v is None else BCV[g2v]
        ba |= bc_[i] == -1; sa |= bc_[i] == 1
    return ba, sa


def guard_states_vec(t, use1, use2, g1=0.5, g2v=None):
    """Vectorised guard_state over an array of knowledge times -> (buy_adverse, sell_adverse) bool arrays."""
    i = np.searchsorted(KB, t, side='right') - 1
    ok = i >= 0; ic = np.clip(i, 0, None)
    ba = np.zeros(len(t), bool); sa = np.zeros(len(t), bool)
    if use1:
        j = np.searchsorted(KB, t - 1000.0, side='right') - 1
        av = ok & (j >= 0) & (t - KB[ic] <= 1000.0)
        m = (MB[ic] / MB[np.clip(j, 0, None)] - 1.0) * 1e4
        ba |= av & (m <= -g1); sa |= av & (m >= g1)
    if use2:
        bc_ = BC if g2v is None else BCV[g2v]
        ba |= ok & (bc_[ic] == -1); sa |= ok & (bc_[ic] == 1)
    return ba, sa


_FLIP_CACHE = {}
def guard_flips(t0, t1, use1, use2, g1=0.5, g2v=None):
    """Times in [t0, t1] where the guard state changes (candidates: row knowledge times, and +1 s for G1)."""
    key = (t0, use1, use2, g1, g2v)
    if key in _FLIP_CACHE:
        return _FLIP_CACHE[key]
    a = np.searchsorted(KB, t0 - 1000.0); b = np.searchsorted(KB, t1)
    cand = KB[a:b]
    if use1:
        cand = np.concatenate([cand, cand + 1000.0, cand + 1000.0 + 1e-6])
    cand = np.unique(np.concatenate([[t0], cand[(cand >= t0) & (cand <= t1)]]))
    ba, sa = guard_states_vec(cand, use1, use2, g1, g2v)
    ch = np.r_[False, (ba[1:] != ba[:-1]) | (sa[1:] != sa[:-1])]
    out = cand[ch].tolist()
    _FLIP_CACHE[key] = out
    return out


def model_theta(rules, key='M'):
    for r in rules:
        if r.startswith(key):
            d = r[1:]
            return float(d[0] + '.' + d[1:])   # M05 -> 0.5 c, M10 -> 1.0 c, M025 -> 0.25 c
    return None


def stop_time(rules):
    for r in rules:
        if r.startswith('S') and r[1:].isdigit():
            return float(r[1:])
    return 240.0


def model_state(mid, t, th, P=PRED):
    v = P.get(mid)
    if v is None:
        return False, False
    i = int(np.searchsorted(v[0], t, side='right')) - 1
    if i < 0 or t - v[0][i] > 100.0:
        return False, False
    p = v[1][i]
    return p <= -th, p >= th


def model_flips(mid, t0, t1, th, P=PRED):
    v = P.get(mid)
    if v is None:
        return []
    a = np.searchsorted(v[0], t0); b = np.searchsorted(v[0], t1)
    tt = v[0][a:b]; p = v[1][a:b]
    st = np.where(p <= -th, -1, np.where(p >= th, 1, 0))
    ch = np.r_[True, st[1:] != st[:-1]]
    out = tt[ch].tolist()
    if V2 and len(tt):
        gap = np.r_[np.diff(tt) > 100.0, True]                 # prediction goes stale after the last point of a run
        out += (tt[gap] + 100.0 + 1e-3).tolist() + tt[1:][gap[:-1]].tolist()   # ... and resumes at the next point
    return out


def parse_size(x):
    if isinstance(x, str) and ':' in x:
        k, lo, hi, par = x.split(':')
        if k == 'AM':                                  # hi when v >= the minute's line <par> (a minute_lines key)
            return dict(kind=k, lo=float(lo), hi=float(hi), par=par)
        return dict(kind=k, lo=float(lo), hi=float(hi), par=float(par))
    return dict(kind='fixed', lo=float(x), hi=float(x), par=0.0)


def model_value(mid, t, P=None):
    v = (PRED if P is None else P).get(mid)
    if v is None:
        return None
    i = int(np.searchsorted(v[0], t, side='right')) - 1
    if i < 0 or t - v[0][i] > 100.0:
        return None
    return v[1][i]


LINES = json.load(open(f'{io.OUT}/step4/minute_lines_t60k.json'))['lines']


def minute_line(key, rel_s):
    """the per-minute guard line <key> at rel_s seconds into the window (after 240 s: the 4th minute's)."""
    return LINES[key][min(max(int(rel_s // 60), 0), 3)]


def line_key(rules, pre):
    for r in rules:
        if r.startswith(pre) and r[len(pre):] in LINES:
            return r[len(pre):]
    return None


def ladder_k(rules):
    for r in rules:
        if r.startswith('LD') and r[2:].isdigit():
            return int(r[2:])
    return None


def line_flips(mid, ws_, t0, t1, key, P):
    """decision times where v crosses the minute's line <key> or goes stale / resumes, plus the minute boundaries."""
    v = P.get(mid)
    out = [ws_ + 60_000.0 * m for m in (1, 2, 3)]
    if v is None:
        return out
    for m in range(4):
        lo_, hi_ = max(t0, ws_ + 60_000.0 * m), (min(t1, ws_ + 60_000.0 * (m + 1)) if m < 3 else t1)
        a = np.searchsorted(v[0], lo_); b = np.searchsorted(v[0], hi_)
        tt = v[0][a:b]; p = v[1][a:b]
        if not len(tt):
            continue
        st = p > minute_line(key, 60.0 * m)
        ch = np.r_[True, st[1:] != st[:-1]]
        out += tt[ch].tolist()
        gap = np.r_[np.diff(tt) > 100.0, True]
        out += (tt[gap] + 100.0 + 1e-3).tolist() + tt[1:][gap[:-1]].tolist()
    return out


def live_opts(rules):
    o = dict(noamend='XNA' in rules, lock=0.0, lock_scope=None, min_usd=None, losscap=None, seq=None, thr=None, plock=None, qlock=None, venue=False, venue_rpm=500.0, bk=None, hy=None, bkd=1, cd=None, fc=None, capn=None, ro=None)
    for r in rules:
        if r.startswith('XS') and r[2:].isdigit():
            o['seq'] = float(r[2:])
        elif r.startswith('XQ') and r[2:].isdigit():
            o['qlock'] = float(r[2:])
        elif r.startswith('XP') and r[2:].isdigit():
            o['seq'] = 0.0; o['plock'] = float(r[2:])
        elif r.startswith('XT'):
            o['thr'] = float(r[2:])
        elif r == 'XV' or (r.startswith('XV') and r[2:].isdigit()):
            o['thr'] = 0.0; o['venue'] = True; o['venue_rpm'] = float(r[2:]) if len(r) > 2 else 500.0
        elif r == 'BKD2':
            o['bkd'] = 2
        elif r.startswith('BK'):
            o['bk'] = r[2:] if r[2:] in ('A', 'M') else float(r[2:])
        elif r.startswith('CAP') and r[3:].isdigit():
            o['capn'] = float(r[3:])
        elif r.startswith('RO') and r[2:].isdigit():
            o['ro'] = float(r[2:])
        elif r.startswith('CD') and r[2:].isdigit():
            o['cd'] = float(r[2:])
        elif r.startswith('FC') and r[2:].isdigit():
            o['fc'] = float(r[2:])
        elif r.startswith('HY') and r[2:].isdigit():
            o['hy'] = float(r[2:])
        if r.startswith('XL') and r[-1] in 'AR':
            o['lock'] = float(r[2:-1]); o['lock_scope'] = 'all' if r[-1] == 'A' else 'sizedown'
        elif r.startswith('XMIN'):
            o['min_usd'] = float(r[4:])
        elif r.startswith('XCAP'):
            o['losscap'] = float(r[4:])
    return o


def run_window(mid, F, kf, rules, policy, mode, size=SIZE, cap_state=None):
    SZ = parse_size(size); cap = SZ['hi']; LV = live_opts(rules); VSTRICT = 'GS' in rules
    PMK = line_key(rules, 'PM'); IMK = line_key(rules, 'IM'); LDK = ladder_k(rules)
    LLK = line_key(rules, 'LL') or PMK or 'zero'; LHOLD = 'LH' in rules
    LANES = (0, 1, 2, 3) if LDK is not None else (0, 1)
    LANES_OF = {s_: tuple(L_ for L_ in LANES if L_ % 2 == s_) for s_ in (0, 1)}
    if (LDK is not None or IMK is not None or PMK is not None) and (not SPLIT or LV['seq'] is not None or LV['hy'] or LEGACY_INTENT or LV['ro'] is not None or LV['capn'] is not None or LV['bk'] is not None):
        raise SystemExit('REFUSED: ladder / improve / per-minute rules are modelled for the split-latency live-like arms only')
    R = reg.loc[mid]
    if pd.isna(R.outcome) or R.outcome not in ('Up', 'Down'):
        return None
    y = 1.0 if R.outcome == 'Up' else 0.0
    ws = float(R.ws) * 1000.0
    if LV['losscap'] is not None and cap_state['halted']:
        return dict(pnl=0.0, net_end=0.0, n_fills=0, shares=0.0, fills=[], withheld=collections.Counter(), requests=[],
                    min_worst=np.nan, refused=0, halted=True, zombie_fills=0, live_ctr={}, delays=[], mtm_stop=0.0, carry_stop=0.0)
    realised0 = cap_state['realised'] if cap_state is not None else 0.0
    requests = []; zombies = {0: [], 1: []}; live_ctr = collections.Counter(); min_worst = [np.inf]
    busy = {L_: None for L_ in LANES}; TH = LV['thr'] is not None; TB = dict(pb=40.0 if LV['venue'] else 4.0, cb=8.0, last=ws)
    sent = {0: [], 1: []}; pend = {L_: None for L_ in LANES}; delays = []
    inflight = {L_: [] for L_ in LANES}; SEQ = LV['seq'] is not None             # split latency: sent but not yet live
    hy_pend = {L_: None for L_ in LANES}
    CDS = dict(b1=None, a1=None, until=[-np.inf, -np.inf], fc=[-np.inf, -np.inf]); SEQX = itertools.count(10 ** 9)
    if SPLIT and not LV['noamend']:
        raise SystemExit('REFUSED: the split-latency handler models live-like arms only (XNA)')
    tf = [r['t_ms'] for r in F]
    g = LT_BY.get(mid)
    if g is None:
        trades = []
    else:
        shift = {'early': -500.0, 'mid': 0.0, 'late': 499.0}[mode]
        tt = np.where(g.how.values > 0, g.t_ms.values, g.exec_s.values * 1000.0 + shift)
        trades = list(zip(tt.tolist(), g.side.tolist(), g.ptick.tolist(), g.shares.tolist(), (g.how.values > 0).tolist()))
        if LEGORDER == 'price':                       # one time, one side: the maker side's best price first
            trades.sort(key=lambda r_: (r_[0], r_[1], -r_[2] if r_[1] == 0 else r_[2], r_[3], r_[4]))
        else:
            trades.sort()
    use1, use2 = 'G1' in rules, 'G2' in rules
    n_reord = 0
    if LEGORDER == 'price' and len(trades) > 1:
        for a_, b_ in zip(trades[:-1], trades[1:]):
            n_reord += int(a_[0] == b_[0] and a_[1] == b_[1] == 0 and a_[2] != b_[2])
    G1TH = 0.3 if 'B1T03' in rules else 0.5; G2KEY = next((G2V[x] for x in G2V if x in rules), None)
    ev = []    # (time, prio, seq, kind, payload) ; prio: activation 0, trade 1, decision 2
    seq = 0
    for i, k in enumerate(kf):
        if ws <= k < ws + 300_000:
            ev.append((k, 2, seq, 'D', i)); seq += 1
    if use1 or use2:
        for t in guard_flips(ws, ws + 300_000, use1, use2, G1TH, G2KEY):
            ev.append((t, 2, seq, 'D', None)); seq += 1
    th = model_theta(rules); th2 = model_theta(rules, 'Q'); thv = model_theta(rules, 'V'); stop = stop_time(rules)
    if V2 and SZ['kind'] in ('A1', 'A7'):
        for t in model_flips(mid, ws, ws + 300_000, SZ['par'] if SZ['kind'] == 'A1' else 0.0):
            ev.append((t, 2, seq, 'D', None)); seq += 1
    if V2 and SZ['kind'] == 'AV':
        for P_ in (PREDVB, PREDVA):
            for t in model_flips(mid, ws, ws + 300_000, SZ['par'], P_):
                ev.append((t, 2, seq, 'D', None)); seq += 1
    if SZ['kind'] == 'A7':
        ev.append((ws + SZ['par'] * 1000.0, 2, seq, 'D', None)); seq += 1
    if LV['ro'] is not None:
        ev.append((ws + LV['ro'] * 1000.0, 2, seq, 'D', None)); seq += 1
    if LV['bk'] is not None and LV['bk'] not in ('A', 'M'):
        ev.append((ws + LV['bk'] * 1000.0, 2, seq, 'D', None)); seq += 1
    if LV['bk'] == 'M':
        for t in model_flips(mid, ws, ws + 300_000, 0.1):
            ev.append((t, 2, seq, 'D', None)); seq += 1
    if th is not None:
        for t in model_flips(mid, ws, ws + 300_000, th):
            ev.append((t, 2, seq, 'D', None)); seq += 1
    if th2 is not None:
        for t in model_flips(mid, ws, ws + 300_000, th2, PRED250):
            ev.append((t, 2, seq, 'D', None)); seq += 1
    if thv is not None:
        for P_ in (PREDVB, PREDVA):
            for t in model_flips(mid, ws, ws + 300_000, thv, P_):
                ev.append((t, 2, seq, 'D', None)); seq += 1
    for key_ in {k_ for k_ in (PMK, IMK, (LLK if LDK is not None else None), (SZ['par'] if SZ['kind'] == 'AM' else None)) if k_ is not None}:
        for P_ in (PREDVB, PREDVA):
            for t in line_flips(mid, ws, ws, ws + 300_000, key_, P_):
                ev.append((t, 2, seq, 'D', None)); seq += 1
    for t in sorted({ws + stop * 1000.0, ws + 240_000, ws + 300_000}):
        ev.append((t, 2, seq, 'D', None)); seq += 1
    for tr in trades:
        ev.append((tr[0], 1, seq, 'T', tr)); seq += 1
    heapq.heapify(ev)
    orders = {L_: None for L_ in LANES}; intent = {L_: None for L_ in LANES}
    known = {L_: None for L_ in LANES}                       # the strategy's view of each lane's order (acknowledgements)
    cancel_wait = {}                                        # request identity -> (policy order, cancel acknowledgement time)

    def pv(L_):
        if POLICYVIEW == 'known':
            if orders[L_] is None and known[L_] is not None:
                live_ctr['view_filled_unknown'] += 1
            return known[L_]
        return orders[L_]

    def pv_rem(o_):
        return o_['size'] - o_.get('kq', 0.0) if POLICYVIEW == 'known' else o_['rem']

    def reserve_cancel(o_, ack):
        if POLICYVIEW == 'known':
            cancel_wait.setdefault(id(o_), (o_, ack))

    def policy_open(now):
        """Orders reserved by client-visible lifecycle state; physical fills never release capacity."""
        if POLICYVIEW == 'known':
            expired = [key for key, (o_, ack) in cancel_wait.items() if ack <= now or pv_rem(o_) <= 1e-12]
            for key in expired:
                del cancel_wait[key]
            refs = [o_ for o_ in known.values() if o_ is not None]
            refs += [r_ for fl_ in inflight.values() for r_ in fl_ if not r_['cancelled']]
            refs += [o_ for o_, _ in cancel_wait.values()]
        else:                                               # explicit regression path
            refs = [o_ for o_ in orders.values() if o_ is not None]
            refs += [z for zs_ in zombies.values() for z in zs_ if z['t_end'] > now and z['rem'] > 1e-12]
            refs += [r_ for fl_ in inflight.values() for r_ in fl_ if not r_['cancelled']]
        return list({id(o_): o_ for o_ in refs}.values())

    net = 0.0; knet = 0.0; cash = 0.0; kcash = 0.0; fills = []; withheld = collections.Counter()
    live_ctr['leg_groups_reordered'] += n_reord

    def venue_book(t):
        i = bisect.bisect_right(tf, t) - 1
        return F[i] if i >= 0 else None

    def known_frame(t):
        i = bisect.bisect_right(kf, t) - 1
        return F[i] if i >= 0 else None

    def disp(f, side, pt):
        pre = 'b' if side == 0 else 'a'
        for i in range(1, 6):
            if f[f'{pre}{i}_p'] == pt:
                return f[f'{pre}{i}_s']
        return 0.0

    def disp_rng(f, side, pt):
        """Displayed size at pt, or None when pt lies beyond the 5 levels shown (size unknown)."""
        last = f['b5_p'] if side == 0 else f['a5_p']
        if last >= 0 and ((side == 0 and pt < last) or (side == 1 and pt > last)):
            return None
        return disp(f, side, pt)

    def side_size(t, rel, side, f, pq):
        k = SZ['kind']
        if k == 'fixed':
            return SZ['lo']
        lo, hi, par = SZ['lo'], SZ['hi'], SZ['par']
        if k == 'A1':
            v = model_value(mid, t)
            return hi if (v is not None and ((side == 0 and v >= par) or (side == 1 and v <= -par))) else lo
        if k == 'A7':
            if rel >= par:
                return lo
            v = model_value(mid, t)
            return hi if (v is not None and ((side == 0 and v >= 0) or (side == 1 and v <= 0))) else lo
        if k == 'A3':
            return float(min(hi, max(lo, round(par * disp(f, side, pq)))))
        if k == 'A4':
            return hi if ((side == 0 and knet < 0) or (side == 1 and knet > 0)) else lo
        if k == 'A6':
            return hi if rel < par else lo
        if k == 'AV':
            v = model_value(mid, t, PREDVB if side == 0 else PREDVA)
            return hi if (v is not None and v >= par) else lo
        if k == 'AM':
            v = model_value(mid, t, PREDVB if side == 0 else PREDVA)
            return hi if (v is not None and v >= minute_line(par, rel)) else lo
        raise ValueError(k)

    NOQ = tuple((None, 0.0) for L_ in LANES)

    def own_open(s_, t_):
        """our orders resting on side s_ at t_: live and cancel-pending (still fillable)."""
        return [orders[L_] for L_ in LANES_OF[s_] if orders[L_] is not None] + [z for z in zombies[s_] if z['t_end'] > t_ and z['rem'] > 1e-12]

    def cur_order(L_):
        """the order this lane has (latest sent and not cancelled, else live)."""
        fl = [r_ for r_ in inflight[L_] if not r_['cancelled']]
        return fl[-1] if fl else pv(L_)

    def desired(t):
        rel = (t - ws) / 1000.0
        if rel >= 300 or ('G5' in rules and rel >= stop):
            return NOQ
        f = known_frame(t)
        if f is None:
            return NOQ
        b1, bs, a1, as_ = f['b1_p'], f['b1_s'], f['a1_p'], f['a1_s']
        if b1 < 0 or a1 < 0 or a1 <= b1:
            return NOQ
        if policy == 'JOIN':
            q = [b1, a1]
        else:
            micro = (b1 * as_ + a1 * bs) / max(bs + as_, 1e-12); half = (a1 - b1) / 2.0
            qb = math.floor(micro - half + 1e-9); qa = math.ceil(micro + half - 1e-9)
            q = [qb if 1 <= qb < a1 else None, qa if b1 < qa <= 99 else None]
        bkside = [False, False]
        if LV['bk'] is not None:
            for s_ in (0, 1):
                if q[s_] is None:
                    continue
                if LV['bk'] == 'A':
                    do_ = True
                elif LV['bk'] == 'M':
                    v_ = model_value(mid, t); do_ = v_ is not None and ((s_ == 0 and v_ <= -0.1) or (s_ == 1 and v_ >= 0.1))
                else:
                    do_ = rel >= LV['bk']
                if do_:
                    np_ = b1 - LV['bkd'] if s_ == 0 else a1 + LV['bkd']   # one (BKD2: two) ticks behind the best price
                    q[s_] = np_ if 1 <= np_ <= 99 else None; bkside[s_] = q[s_] is not None
        if LV['cd'] is not None:                              # O4 touch cooldown
            pb_, pa_ = CDS['b1'], CDS['a1']
            if pb_ is not None and b1 < pb_:
                CDS['until'][0] = t + LV['cd']; heapq.heappush(ev, (t + LV['cd'] + 1e-3, 2, next(SEQX), 'D', None))
            if pa_ is not None and a1 > pa_:
                CDS['until'][1] = t + LV['cd']; heapq.heappush(ev, (t + LV['cd'] + 1e-3, 2, next(SEQX), 'D', None))
            CDS['b1'], CDS['a1'] = b1, a1
        for s_ in (0, 1):
            if q[s_] is not None and (t < CDS['until'][s_] or t < CDS['fc'][s_]):
                q[s_] = None; withheld['COOLDOWN'] += 1
        ba, sa = guard_state(t, use1, use2, G1TH, G2KEY) if (use1 or use2) else (False, False)
        btc_ = (ba, sa)                                     # Binance-only flags (LH: a resting rung is pulled only by these)
        if th is not None:
            mb_, ms_ = model_state(mid, t, th)
            ba = ba or mb_; sa = sa or ms_
        if th2 is not None:
            mb_, ms_ = model_state(mid, t, th2, PRED250)
            ba = ba or mb_; sa = sa or ms_
        if thv is not None:
            vb_, _ = model_state(mid, t, thv, PREDVB); va_, _ = model_state(mid, t, thv, PREDVA)
            if 'SBV' in rules:                                       # step back one tick instead of withholding the flagged side
                for s_, flag in ((0, vb_), (1, va_)):
                    if flag and q[s_] is not None:
                        np_ = b1 - 1 if s_ == 0 else a1 + 1
                        q[s_] = np_ if 1 <= np_ <= 99 else None; bkside[s_] = q[s_] is not None
            elif 'SBP' in rules:                                     # step back to the price the 1 s prediction points to
                pv_ = model_value(mid, t); m_ = (b1 + a1) / 2.0
                for s_, flag in ((0, vb_), (1, va_)):
                    if flag and q[s_] is not None:
                        if pv_ is None:
                            np_ = b1 - 1 if s_ == 0 else a1 + 1
                        elif s_ == 0:
                            np_ = int(max(b1 - 5, min(b1 - 1, math.floor(m_ + pv_ - 0.5))))
                        else:
                            np_ = int(min(a1 + 5, max(a1 + 1, math.ceil(m_ + pv_ + 0.5))))
                        q[s_] = np_ if 1 <= np_ <= 99 else None; bkside[s_] = q[s_] is not None
            else:
                ba = ba or vb_; sa = sa or va_
        vv_ = [None, None]
        if PMK is not None or IMK is not None or LDK is not None:
            vv_ = [model_value(mid, t, PREDVB), model_value(mid, t, PREDVA)]
        if PMK is not None:
            c_ = minute_line(PMK, rel)
            ba = ba or (vv_[0] is not None and vv_[0] <= c_); sa = sa or (vv_[1] is not None and vv_[1] <= c_)
        if VSTRICT:                                          # GS: no guard value (missing / > 100 ms old) -> withhold the side
            for s_, P_ in ((0, PREDVB), (1, PREDVA)):
                if model_value(mid, t, P_) is None:
                    withheld['GS_nocover'] += 1
                    if s_ == 0:
                        ba = True
                    else:
                        sa = True
        live = {s: sum(pv_rem(o_) for o_ in (pv(L_) for L_ in LANES_OF[s]) if o_) for s in (0, 1)}
        imp = [False, False]
        if IMK is not None and a1 - b1 >= 2:                 # improve one tick inside a >= 2-tick spread
            ci_ = minute_line(IMK, rel)
            ok_ = [vv_[s_] is not None and vv_[s_] >= ci_ for s_ in (0, 1)]
            if ok_[0] and ok_[1] and b1 + 1 == a1 - 1:     # both want the one inside price: the higher v takes it
                ok_[0 if vv_[0] < vv_[1] else 1] = False
            for s_, np_ in ((0, b1 + 1), (1, a1 - 1)):
                if ok_[s_]:
                    q[s_] = np_; imp[s_] = True; bkside[s_] = False
        rung = [None, None]; rz = [0.0, 0.0]
        if LDK is not None:                                  # ladder rung one tick behind the best
            cl_ = minute_line(LLK, rel)
            for s_ in (0, 1):
                L_ = s_ + 2; o_ = cur_order(L_); flag = ba if s_ == 0 else sa
                behind_ = o_ is not None and ((s_ == 0 and b1 - LDK <= o_['pt'] < b1) or (s_ == 1 and a1 < o_['pt'] <= a1 + LDK))
                if LHOLD and behind_:                        # LH: a rung resting behind the best ignores the guard line
                    if btc_[s_] or (VSTRICT and vv_[s_] is None):
                        continue
                    p_, z_ = o_['pt'], o_['size']
                elif flag or vv_[s_] is None or not (vv_[s_] > cl_):
                    continue
                elif o_ is not None and ((s_ == 0 and b1 - LDK <= o_['pt'] < a1) or (s_ == 1 and b1 < o_['pt'] <= a1 + LDK)):
                    p_, z_ = o_['pt'], o_['size']            # resting within K ticks (or promoted to / inside the best): keep
                else:
                    p_ = b1 - 1 if s_ == 0 else a1 + 1
                    if not 1 <= p_ <= 99:
                        continue
                    z_ = side_size(t, rel, s_, f, p_)
                if 'G6' in rules:
                    increasing = (s_ == 0 and knet >= 0) or (s_ == 1 and knet <= 0)
                    if increasing and knet != 0 and abs(knet) >= 2 * cap:
                        withheld['LD_G6_soft'] += 1; continue
                    if increasing and abs(knet) + live[s_] + z_ > 5 * cap:
                        withheld['LD_G6_hard'] += 1; continue
                if LV['min_usd'] is not None and (p_ if s_ == 0 else 100 - p_) * z_ / 100.0 < LV['min_usd'] - 1e-9:
                    withheld['LD_MIN_USD'] += 1; continue
                rung[s_] = p_; rz[s_] = z_
        zs = [0.0, 0.0]
        for side in (0, 1):
            p = q[side]
            if p is None:
                continue
            zs[side] = sz_ = side_size(t, rel, side, f, p)
            touch = b1 if side == 0 else a1
            if 'G4' in rules and not bkside[side] and ((side == 0 and p < b1) or (side == 1 and p > a1)):
                q[side] = None; withheld['G4'] += 1; continue
            if 'G3' in rules and p == touch:
                nxt = f['b2_p'] if side == 0 else f['a2_p']
                if nxt < 0 or abs(touch - nxt) >= 2:
                    q[side] = None; withheld['G3'] += 1; continue
            if (side == 0 and ba) or (side == 1 and sa):
                q[side] = None; withheld['BTC'] += 1; continue
            if 'G6' in rules:
                increasing = (side == 0 and knet >= 0) or (side == 1 and knet <= 0)
                reducing = (side == 0 and knet < 0) or (side == 1 and knet > 0)
                if rel >= stop and not reducing:
                    q[side] = None; withheld['G6_final'] += 1; continue
                if V2 and rel >= stop:
                    zs[side] = sz_ = min(sz_, abs(knet))           # reduce-only: never flip the position
                if increasing and knet != 0 and abs(knet) >= 2 * cap:
                    q[side] = None; withheld['G6_soft'] += 1; continue
                if increasing and abs(knet) + live[side] + sz_ > 5 * cap:
                    q[side] = None; withheld['G6_hard'] += 1; continue
            red_ = (side == 0 and knet < 0) or (side == 1 and knet > 0)
            if LV['ro'] is not None and rel >= LV['ro']:      # reduce-only phase
                if not red_:
                    q[side] = None; withheld['RO'] += 1; continue
                zs[side] = sz_ = min(sz_, abs(knet))
            if LV['capn'] is not None and not red_ and abs(knet) >= LV['capn'] - 1e-9:
                q[side] = None; withheld['CAPN'] += 1; continue
            if LV['min_usd'] is not None and (p if side == 0 else 100 - p) * sz_ / 100.0 < LV['min_usd'] - 1e-9:
                q[side] = None; withheld['MIN_USD'] += 1; continue  # live: an order under 1 USD is never sent
            if LDK is not None and rung[side] is not None and p == rung[side]:
                q[side] = None; withheld['LD_dup_touch'] += 1; continue   # our rung already holds this price
            if imp[side]:
                live_ctr['improve_wanted'] += 1
        out_ = [(q[i], zs[i] if q[i] is not None else 0.0) for i in (0, 1)]
        if LDK is not None:
            out_ += [(rung[i], rz[i] if rung[i] is not None else 0.0) for i in (0, 1)]
        return tuple(out_)

    def worst_case(extra=None):
        """This window's worst settlement value given fills so far and every open order filled on its losing side."""
        w_up = kcash + knet if POLICYVIEW == 'known' else cash + net
        w_dn = kcash if POLICYVIEW == 'known' else cash
        opens = policy_open(t) if POLICYVIEW == 'known' else [orders[L_] for L_ in LANES if orders[L_]] + [z for s_ in (0, 1) for z in zombies[s_] if z['t_end'] > t]
        if extra is not None:
            opens.append(extra)
        for o_ in opens:
            px_ = o_['pt'] / 100.0
            rem_ = pv_rem(o_) if POLICYVIEW == 'known' and o_ is not extra else o_['rem']
            if o_['side'] == 0:
                w_dn -= rem_ * px_                          # bought YES, settles at 0
            else:
                w_up -= rem_ * (1.0 - px_)                  # sold YES (bought NO), settles against us at 1
        return min(w_up, w_dn)

    def remove(side, o_, t_dec, why):
        """Take order o_ off side; under a removal lock it stays fillable (zombie) until t_dec + max(L_O, lock)."""
        locked = LV['lock'] > L_O and (LV['lock_scope'] == 'all' or why == 'sizedown')
        if locked:
            o_['t_end'] = t_dec + LV['lock']; zombies[side].append(o_); live_ctr['zombies'] += 1
        elif LV['qlock'] is not None and o_['t0'] + LV['qlock'] > t_dec + L_O:
            o_['t_end'] = o_['t0'] + LV['qlock']; zombies[side].append(o_); live_ctr['zombies'] += 1
            live_ctr['plock_held'] += 1                   # removable only from activation + lock; rests beside the new order
        reserve_cancel(o_, o_.get('t_end', t_dec + L_O))
        orders[side] = None; known[side] = None

    def refill(now):
        if now > TB['last']:
            dt_ = (now - TB['last']) / 1000.0
            if LV['venue']:
                TB['pb'] = min(40.0, TB['pb'] + LV['venue_rpm'] / 60.0 * dt_)
            else:
                TB['pb'] = min(4.0, TB['pb'] + LV['thr'] * dt_); TB['cb'] = min(8.0, TB['cb'] + 2.0 * dt_)
            TB['last'] = now

    def cancel(side, o_, t_dec, why):
        """Send a cancel. Sequential lock: the order stays on its side (fillable, seen by G6) until the cancel lands;
        returns True when the side must now wait (its next order is placed by the 'U' event)."""
        nonlocal seq
        requests.append((t_dec, side, 'cancel', side))
        if TH:
            refill(t_dec); TB['pb' if LV['venue'] else 'cb'] -= 1.0
        if LV['seq'] is not None:
            busy[side] = t_end = t_dec + max(LV['seq'], L_O)
            if LV['plock'] is not None:
                busy[side] = t_end = max(t_dec + L_O, o_['t0'] + LV['plock'])   # removable from activation + lock
                live_ctr['plock_held'] += int(o_['t0'] + LV['plock'] > t_dec + L_O)
            reserve_cancel(o_, t_end)
            heapq.heappush(ev, (t_end, 0, seq, 'U', side)); seq += 1
            return True
        remove(side, o_, t_dec, why)
        return False

    while ev:
        t, pr, _, kind, pl = heapq.heappop(ev)
        if kind == 'D':
            want = desired(t)
            for side in LANES:
                if LEGACY_INTENT and want[side][0] is None and intent[side] is None:
                    continue                              # old behaviour, diagnostic only (see PF_LEGACY_INTENT)
                if want[side] != intent[side]:
                    if (LV['hy'] and want[side][0] is not None and intent[side] is not None and intent[side][0] is not None
                            and want[side][0] != intent[side][0]):                     # O3 dead-band on reprices
                        pw = hy_pend[side]
                        if pw is None or pw[0] != want[side]:
                            hy_pend[side] = (want[side], t); heapq.heappush(ev, (t + LV['hy'], 2, seq, 'D', None)); seq += 1
                            continue
                        if t - pw[1] < LV['hy'] - 1e-9:
                            continue
                    hy_pend[side] = None
                    intent[side] = want[side]
                    heapq.heappush(ev, (t + (0.0 if SPLIT else L_O), 0, seq, 'A', (side,) + want[side])); seq += 1
                else:
                    hy_pend[side] = None
        elif kind == 'A' and SPLIT:                         # split latency: requests leave at the decision
            lane, p, zsz = pl; side = lane % 2
            if busy[lane] is not None:
                live_ctr['absorbed'] += 1; continue          # sequential: waiting for the cancel to take effect
            fl = [r_ for r_ in inflight[lane] if not r_['cancelled']]
            cur = fl[-1] if fl else pv(lane)
            if cur is None and p is None:
                continue
            if cur is not None and p is not None and cur['pt'] == p and abs(cur['size'] - zsz) < 1e-9:
                continue                                      # same price and size: keep
            if cur is not None:
                requests.append((t, side, 'cancel', lane))
                if TH:
                    refill(t); TB['pb' if LV['venue'] else 'cb'] -= 1.0
                pending_cur = any(cur is r_ for r_ in fl)
                t_rm = (cur['t_act'] if pending_cur else t) + L_CA   # a remove sent before the create is live waits for it
                if pending_cur and CK.L_PENDCANCEL is not None:
                    t_rm = max(t + L_CA, cur['t_send'] + CK.L_PENDCANCEL)   # measured: held until ~create answer + 55 ms
                reserve_cancel(cur, t_rm)
                if pending_cur:
                    cur['cancelled'] = True; cur['t_rm'] = t_rm
                if SEQ:
                    busy[lane] = t_rm; heapq.heappush(ev, (t_rm, 0, seq, 'U', lane)); seq += 1
                    continue
                if not pending_cur:
                    known[lane] = None                           # the strategy has cancelled it
                    if cur is orders[lane]:
                        cur['t_end'] = t_rm; zombies[side].append(cur); orders[lane] = None; live_ctr['zombies'] += 1
                    else:
                        live_ctr['acted_on_unknown_fill'] += 1   # already filled at the venue; the cancel finds nothing
            if p is None:
                continue
            wc = worst_case(dict(side=side, pt=p, rem=zsz))
            min_worst[0] = min(min_worst[0], wc)
            if LV['losscap'] is not None and realised0 + wc < -LV['losscap']:
                live_ctr['refused'] += 1; intent[lane] = None; continue
            if TH:
                refill(t); key = (p, zsz)
                open_n = len(policy_open(t))
                sent[side] = [x for x in sent[side] if x > t - L_ANS]
                short = (TB['pb'] < open_n + 2 - 1e-9) if LV['venue'] else (TB['pb'] < 1.0 - 1e-9 or TB['cb'] < open_n + 1 - 1e-9)
                if short or len(sent[side]) >= 3:
                    pd_ = pend[lane]
                    if pd_ is None or pd_['key'] != key:
                        live_ctr['place_never'] += int(pd_ is not None)
                        pend[lane] = pd_ = dict(key=key, t_first=t, bo=250.0)
                    else:
                        pd_['bo'] = min(4000.0, pd_['bo'] * 2.0)
                    live_ctr['throttled'] += 1
                    heapq.heappush(ev, (t + pd_['bo'], 0, seq, 'R', (lane, key))); seq += 1
                    continue
                TB['pb'] -= 1.0; sent[side].append(t); pd_ = pend[lane]
                if pd_ is not None and pd_['key'] == key:
                    live_ctr['place_delayed'] += 1; delays.append(t - pd_['t_first'])
                else:
                    live_ctr['place_never'] += int(pd_ is not None); live_ctr['place_immediate'] += 1
                pend[lane] = None
            requests.append((t, side, 'place', lane))
            rec = dict(side=side, lane=lane, pt=p, size=zsz, t_send=t, t_act=t + L_PL, cancelled=False)
            inflight[lane].append(rec); heapq.heappush(ev, (t + L_PL, 0, seq, 'P', rec)); seq += 1
        elif kind == 'P':                                   # split latency: a sent order goes live (post-only check)
            rec = pl; side = rec['side']; lane = rec['lane']
            inflight[lane] = [r_ for r_ in inflight[lane] if r_ is not rec]
            ci = bisect.bisect_left(tf, t) - 1; f = F[ci] if ci >= 0 else None
            if f is not None and ((side == 0 and f['a1_p'] >= 0 and rec['pt'] >= f['a1_p']) or (side == 1 and f['b1_p'] >= 0 and rec['pt'] <= f['b1_p'])):
                live_ctr['postonly_refused'] += 1
                cancel_wait.pop(id(rec), None)                # the create rejection is terminal and observable
                if REJRESET == 'on' and not rec['cancelled']:
                    intent[lane] = None; live_ctr['rej_reset'] += 1   # rejected: the next decision re-sends it if still wanted
                continue
            if OWNCROSS == 'on':                              # post-only: our own resting opposite orders are on the venue book too
                op_ = own_open(1 - side, t)
                if op_ and ((side == 0 and rec['pt'] >= min(x['pt'] for x in op_)) or (side == 1 and rec['pt'] <= max(x['pt'] for x in op_))):
                    live_ctr['postonly_refused_own'] += 1
                    cancel_wait.pop(id(rec), None)
                    if not rec['cancelled']:
                        intent[rec['lane']] = None                     # refused: the next decision re-places it if still wanted
                    continue
            o = dict(side=side, lane=lane, pt=rec['pt'], rem=rec['size'], size=rec['size'], ahead=disp(f, side, rec['pt']) if f is not None else 0.0, t0=t, ci=ci)
            if id(rec) in cancel_wait:
                _, ack = cancel_wait.pop(id(rec))
                cancel_wait[id(o)] = (o, ack)
            if QACT and ci + 1 < len(F) and tf[ci + 1] - t <= 100.0:   # pessimistic bound: orders that joined before our activation
                d2 = disp_rng(F[ci + 1], side, rec['pt'])                # but after the last frame are ahead of us
                if d2 is not None and d2 > o['ahead']:
                    o['ahead'] = d2; live_ctr['qact_raised'] += 1
            o['ahead0'] = o['ahead']
            if rec['cancelled'] and not SEQ:
                o['t_end'] = rec['t_rm']; zombies[side].append(o); live_ctr['zombies'] += 1   # live until its cancel lands
            else:
                orders[lane] = o; known[lane] = o               # sequential + cancelled: removed by its 'U' event
                lv_ = [orders[L_]['pt'] for L_ in LANES_OF[side] if orders[L_] is not None]
                live_ctr['own_dup_price'] += int(len(lv_) != len(set(lv_)))
                ob_ = [x['pt'] for x in own_open(0, t)]; oa_ = [x['pt'] for x in own_open(1, t)]   # incl. cancel-pending
                live_ctr['own_cross'] += int(bool(ob_) and bool(oa_) and max(ob_) >= min(oa_))
        elif kind == 'A':
            side, p, zsz = pl
            assert LANES == (0, 1), 'non-split handler: one order per side only'
            o = orders[side]; t_dec = t - L_O
            if busy[side] is not None:
                live_ctr['absorbed'] += 1; continue      # sequential: the side waits for its cancel to land
            if p is None:
                if o is not None:
                    cancel(side, o, t_dec, 'pull')
                continue
            why = 'replace'
            if o is not None and o['pt'] == p:
                if zsz < o['size']:
                    if not LV['noamend']:                 # amend down in place, queue kept
                        requests.append((t_dec, side, 'amend', side))
                        o['rem'] -= o['size'] - zsz; o['size'] = zsz
                        if o['rem'] <= 1e-12:
                            orders[side] = None; intent[side] = None
                        continue
                    why = 'sizedown'                      # live: no amend -> cancel + new order at the back
                elif zsz <= o['size']:
                    continue                              # same size: keep (D1's same-price keep)
                                                          # larger: cancel/replace below, queue reset
            ci = (bisect.bisect_left(tf, t) if V2 else bisect.bisect_right(tf, t)) - 1
            f = F[ci] if ci >= 0 else None
            if o is not None and cancel(side, o, t_dec, why):
                continue                                  # sequential: placed when the cancel lands
            wc = worst_case(dict(side=side, pt=p, rem=zsz))
            min_worst[0] = min(min_worst[0], wc)
            if LV['losscap'] is not None and realised0 + wc < -LV['losscap']:
                live_ctr['refused'] += 1; intent[side] = None; continue   # live: place refused by the total-loss cap
            if TH:
                refill(t_dec); key = (p, zsz)
                open_n = len(policy_open(t_dec))
                sent[side] = [x for x in sent[side] if x > t_dec - 2 * L_O]
                short = (TB['pb'] < open_n + 2 - 1e-9) if LV['venue'] else (TB['pb'] < 1.0 - 1e-9 or TB['cb'] < open_n + 1 - 1e-9)
                if short or len(sent[side]) >= 3:
                    pd_ = pend[side]
                    if pd_ is None or pd_['key'] != key:
                        live_ctr['place_never'] += int(pd_ is not None)
                        pend[side] = pd_ = dict(key=key, t_first=t_dec, bo=250.0)
                    else:
                        pd_['bo'] = min(4000.0, pd_['bo'] * 2.0)
                    live_ctr['throttled'] += 1
                    heapq.heappush(ev, (t + pd_['bo'] - L_O, 0, seq, 'R', (side, key))); seq += 1
                    continue
                TB['pb'] -= 1.0; sent[side].append(t_dec); pd_ = pend[side]
                if pd_ is not None and pd_['key'] == key:
                    live_ctr['place_delayed'] += 1; delays.append(t_dec - pd_['t_first'])
                else:
                    live_ctr['place_never'] += int(pd_ is not None); live_ctr['place_immediate'] += 1
                pend[side] = None
            requests.append((t_dec, side, 'place', side))
            if f is not None and ((side == 0 and f['a1_p'] >= 0 and p >= f['a1_p']) or (side == 1 and f['b1_p'] >= 0 and p <= f['b1_p'])):
                orders[side] = None
                if REJRESET == 'on':
                    intent[side] = None; live_ctr['rej_reset'] += 1
                continue
            if OWNCROSS == 'on':
                op_ = own_open(1 - side, t)
                if op_ and ((side == 0 and p >= min(x['pt'] for x in op_)) or (side == 1 and p <= max(x['pt'] for x in op_))):
                    live_ctr['postonly_refused_own'] += 1; orders[side] = None; intent[side] = None; continue
            orders[side] = dict(side=side, lane=side, pt=p, rem=zsz, size=zsz, ahead=disp(f, side, p) if f is not None else 0.0, t0=t, ci=ci)
            orders[side]['ahead0'] = orders[side]['ahead']; known[side] = orders[side]
        elif kind == 'U':                                   # sequential lock: our cancel has landed
            side = pl
            if busy[side] is not None and abs(busy[side] - t) < 1e-6:
                busy[side] = None; orders[side] = None; known[side] = None
                it = intent[side]
                if it is not None and it[0] is not None:
                    heapq.heappush(ev, (t + (0.0 if SPLIT else L_O), 0, seq, 'A', (side,) + it)); seq += 1
        elif kind == 'R':                                   # throttle: retry a refused place while its intent stands
            lane, key = pl
            if pend[lane] is not None and pend[lane]['key'] == key and intent[lane] == key:
                heapq.heappush(ev, (t + (0.0 if SPLIT else L_O), 0, seq, 'A', (lane,) + key)); seq += 1
        elif kind == 'K':                                   # v2: our fill becomes known
            lane, qd, done, o_k = pl; side = lane % 2
            knet += qd; kcash -= qd * o_k['pt'] / 100.0
            o_k['kq'] = o_k.get('kq', 0.0) + abs(qd)
            if LV['fc'] is not None:                          # O4 fill cooldown on the filled side
                CDS['fc'][side] = t + LV['fc']; heapq.heappush(ev, (t + LV['fc'] + 1e-3, 2, seq, 'D', None)); seq += 1
                heapq.heappush(ev, (t, 2, seq, 'D', None)); seq += 1
            if done and ((known[lane] is o_k) if POLICYVIEW == 'known' else orders[lane] is None):
                known[lane] = None; intent[lane] = None     # slot vacated as notified: re-place now (D1's repost)
                heapq.heappush(ev, (t, 2, seq, 'D', None)); seq += 1
        else:
            tt_, side, pt, sh, timed = pl
            if zombies[side]:
                zombies[side] = [z for z in zombies[side] if z['t_end'] > t and z['rem'] > 1e-12]
            book_ = zombies[side] + [orders[L_] for L_ in LANES_OF[side] if orders[L_] is not None]   # older (locked) orders first
            if FILLPRIO == 'price' and len(book_) > 1:       # price priority among our own orders, then time (stable sort)
                b0_ = [id(o_) for o_ in book_]
                book_.sort(key=lambda o_: -o_['pt'] if side == 0 else o_['pt'])
                live_ctr['prio_reordered'] += int([id(o_) for o_ in book_] != b0_)
            sh_left = sh
            for o in book_:
                is_z = not any(o is orders[L_] for L_ in LANES_OF[side])
                if V2:
                    j = bisect.bisect_left(tf, t) - 1           # frames strictly before the trade
                    for k in range(o['ci'] + 1, j + 1):
                        d = disp_rng(F[k], side, o['pt'])
                        if d is not None and d < o['ahead']:
                            o['ahead'] = d
                    if j > o['ci']:
                        o['ci'] = j
                else:
                    f = venue_book(t)
                    if f is not None:
                        o['ahead'] = min(o['ahead'], disp(f, side, o['pt']))
                through = (side == 0 and pt < o['pt']) or (side == 1 and pt > o['pt'])
                if pt == o['pt']:
                    use = min(sh_left, o['ahead']); o['ahead'] -= use; q = min(sh_left - use, o['rem'])
                elif through:
                    q = min(sh_left, o['rem'])
                    if V2:
                        o['ahead'] = 0.0                        # a sweep through our level consumed the queue ahead
                else:
                    q = 0.0
                if q > 1e-12:
                    sh_left -= q                                # a locked order ahead of ours takes its share first
                    o['rem'] -= q; px = o['pt'] / 100.0
                    if side == 0:
                        net += q; cash -= q * px
                    else:
                        net -= q; cash += q * px
                    fills.append((t, side, o['pt'], q, 'through' if through else 'at', timed, o['size'], is_z, o.get('lane', side), o.get('ahead0', np.nan), t - o['t0']))
                    live_ctr['zombie_fills'] += int(is_z)
                    done = (not is_z) and o['rem'] <= 1e-12
                    if done:
                        orders[o.get('lane', side)] = None
                    if V2:
                        heapq.heappush(ev, (t + L_FILL, 1, seq, 'K', (o.get('lane', side), q if side == 0 else -q, done, o))); seq += 1
                    else:
                        knet = net
                        if done:
                            intent[o.get('lane', side)] = None          # slot vacated: the next decision re-places (D1's repost)
    mids = np.array([((r['a1_p'] + r['b1_p']) / 2.0) if (r['b1_p'] >= 0 and r['a1_p'] > r['b1_p']) else np.nan for r in F])

    def mid_at(t):
        i = bisect.bisect_right(tf, t) - 1
        while i >= 0 and not np.isfinite(mids[i]):
            i -= 1
        return mids[i] if i >= 0 else np.nan
    rows = []
    for t, side, pt, q, kd, timed, osz, is_z, ln, a0, age in fills:
        sg = 1.0 if side == 0 else -1.0
        rows.append((mid, (t - ws) / 1000.0, side, pt, q, kd == 'through', timed, sg * (y * 100 - pt) * q / 100.0,
                     sg * (mid_at(t + 1000) - pt) * q / 100.0, sg * (mid_at(t + 60000) - pt) * q / 100.0,
                     0.25 * 0.02 * min(pt, 100 - pt) / 100.0 * q, osz, is_z, ln, a0, age))
    live_ctr['place_never'] += sum(pend[L_] is not None for L_ in LANES)   # still waiting at the window's end
    m_stop = mid_at(ws + stop * 1000.0 + 100.0)
    mtm_stop = cash + net * m_stop / 100.0 if np.isfinite(m_stop) else np.nan
    carry_stop = net * (y - m_stop / 100.0) if np.isfinite(m_stop) else np.nan
    return dict(pnl=cash + net * y, net_end=net, n_fills=len(fills), shares=sum(x[3] for x in fills), fills=rows, withheld=withheld,
                requests=requests, min_worst=(min_worst[0] if np.isfinite(min_worst[0]) else np.nan), refused=live_ctr['refused'],
                halted=False, zombie_fills=live_ctr['zombie_fills'], live_ctr=dict(live_ctr), delays=delays, mtm_stop=mtm_stop, carry_stop=carry_stop)


BASE = {'JOIN': [], 'MICRO': []}
SINGLES = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
STACK = ['G4', 'G2', 'G6', 'G5', 'G3', 'G1']
arms = []
if ARMSET == 'singles':
    for pol in [p for p in ('JOIN', 'MICRO') if p in POLICIES]:
        arms += [(pol, (), m) for m in ('early', 'mid', 'late')]
        arms += [(pol, (r,), 'mid') for r in SINGLES if not (pol == 'JOIN' and r == 'G4')]
        arms += [(pol, ('G1', 'G2'), 'mid')]
elif ARMSET == 'stack':
    for pol in [p for p in ('MICRO', 'JOIN') if p in POLICIES]:
        arms += [(pol, (), m) for m in ('early', 'mid', 'late')]
        for n in range(1, len(STACK) + 1):
            rs = tuple(STACK[:n])
            if pol == 'JOIN' and rs == ('G4',):
                continue
            for m in ('early', 'mid', 'late'):
                arms.append((pol, rs, m))
elif ARMSET == 'model':
    NOBTC = ('G4', 'G6', 'G5', 'G3')
    sets = [NOBTC, tuple(STACK)] + [NOBTC + (f'M{k}',) for k in ('05', '10')] + [tuple(STACK) + (f'M{k}',) for k in ('05', '10')]
    for rs in sets:
        for m in ('early', 'mid', 'late'):
            arms.append(('MICRO', rs, m))
elif ARMSET == 'size':
    for sz in [float(x) for x in os.environ.get('PF_SIZES', '5,10,25,50,100').split(',')]:
        arms.append(('MICRO', tuple(STACK), 'mid', sz))
        for m in ('early', 'mid', 'late'):
            arms.append(('MICRO', tuple(STACK) + ('M05',), m, sz))
elif ARMSET == 'opt':
    ST = tuple(STACK)
    sets = [ST, ST + ('M05',), ST + ('M025',), ST + ('M075',), ST + ('M10',),
            ST + ('M05', 'Q025'), ST + ('M05', 'Q05'), ST + ('Q05',),
            ST + ('M05', 'S270'), ST + ('M05', 'S300')]
    sets += [tuple(g for g in STACK if g != drop) + ('M05',) for drop in ('G1', 'G2', 'G3', 'G4', 'G6')]
    arms = [('MICRO', rs, 'mid') for rs in sets]
elif ARMSET == 'custom':
    for spec in os.environ['PF_ARMS'].split(';'):
        pol, rs, m, sz = spec.split('|')
        arms.append((pol, tuple(x for x in rs.split('+') if x), m, sz if ':' in sz else float(sz)))
else:
    for pol in [p for p in ('MICRO', 'JOIN') if p in POLICIES]:
        for rs in ((), tuple(STACK)):
            for m in ('early', 'mid', 'late'):
                arms.append((pol, rs, m))
arms = [a if len(a) == 4 else a + (SIZE,) for a in arms]
if any(r[0] in 'MQ' for a in arms for r in a[1]) and not PRED:
    sys.exit(f'REFUSED: model-guard arms but no predictions loaded from step5/{day}/{PREDS}.parquet')
if any(r[0] == 'Q' for a in arms for r in a[1]) and not PRED250:
    sys.exit(f'REFUSED: 250 ms guard arms but no pred_250 in step5/{day}/{PREDS}.parquet')
if any(r[0] == 'V' for a in arms for r in a[1]) and not (PREDVB and PREDVA):
    sys.exit(f'REFUSED: per-side guard arms but no v_bid / v_ask in step5/{day}/{PREDS}.parquet')
if any(line_key(a[1], p_) for a in arms for p_ in ('PM', 'IM', 'LL')) and not (PREDVB and PREDVA):
    sys.exit(f'REFUSED: per-minute line arms but no v_bid / v_ask in step5/{day}/{PREDS}.parquet')
if any(isinstance(a[3], str) and a[3].startswith('AM:') for a in arms) and not (PREDVB and PREDVA):
    sys.exit(f'REFUSED: per-minute size arms but no v_bid / v_ask in step5/{day}/{PREDS}.parquet')
if any(isinstance(a[3], str) and a[3].startswith('AV:') for a in arms) and not (PREDVB and PREDVA):
    sys.exit(f'REFUSED: guard-value size arms but no v_bid / v_ask in step5/{day}/{PREDS}.parquet')

FCOLS = ['mid', 't', 'side', 'pt', 'q', 'through', 'timed', 'settle', 'mo1', 'mo60', 'rebate', 'osz', 'zombie', 'lane', 'ahead0', 'age_ms']
WCOLS = ['mid', 'ws', 'pnl', 'net_end', 'n_fills', 'shares', 'min_worst', 'refused', 'halted', 'zombie_fills', 'mtm_stop', 'carry_stop']
res = collections.defaultdict(list); allf = collections.defaultdict(list); wh = collections.defaultdict(collections.Counter)
reqs = collections.defaultdict(list); LCT = collections.defaultdict(collections.Counter); DLY = collections.defaultdict(list)
MAXW = int(os.environ.get('PF_MAX_WINDOWS', '0')); nw = 0


def arm_name(pol, rs, m, sz):
    return f"{pol}{'+' + '+'.join(rs) if rs else ''}/{m}{('@' + sz) if isinstance(sz, str) else ('' if sz == SIZE else '@%g' % sz)}"


CAPIN = os.environ.get('PF_LIVE_CAPSTATE_IN')
cap_in = json.load(open(CAPIN)) if CAPIN else {}
cstate = {arm_name(*a): dict(cap_in.get(arm_name(*a), dict(realised=0.0, halted=False, halted_at=None, windows_seen=0))) for a in arms}
GIDX = books.groupby('mid', sort=False).indices
ORDER = sorted((m_ for m_ in GIDX if m_ in reg.index), key=lambda m_: float(reg.loc[m_].ws))   # start-time order (loss cap)
diag = []                                           # live-rule data diagnostics per window (no behaviour change)
BG = np.array(BN_GAPS, dtype=np.float64).reshape(-1, 2)
for mid in ORDER:
    g = books.iloc[GIDX[mid]]
    nw += 1
    if MAXW and nw > MAXW:
        break
    F = g.to_dict('records')
    kf = ((g.t_ms.values + P_IN) if SCEN == 'tokyo20' else CK.pf_known(g.t_ms.values, g.recv_ns.values, RL_SHIFT) if SCEN == 'tokyoreal'
          else (g.recv_ns.values / 1e6 - NET_MS))
    kf = np.maximum.accumulate(kf.astype(np.float64)).tolist()   # knowledge times non-decreasing in frame order
    ws_ = float(reg.loc[mid].ws) * 1000.0; q0, q1 = ws_, ws_ + 240_000.0
    bn_off = sum(max(0.0, min(q1, b + 2000.0) - max(q0, a + 1000.0)) for a, b in BG if b + 2000.0 > q0 and a + 1000.0 < q1)
    kk = np.array([x for x in kf if q0 - 30_000.0 <= x < q1], dtype=np.float64)
    kk = np.r_[kk, q1] if len(kk) else np.array([q0 - 30_000.0, q1])
    dk = np.diff(kk); gi = np.flatnonzero(dk > 5000.0)
    pf_off = sum(max(0.0, min(q1, kk[i_ + 1] + 2000.0) - max(q0, kk[i_] + 5000.0)) for i_ in gi)
    diag.append(dict(mid=int(mid), ws=ws_, bn_off_ms=bn_off, pf_off_ms=pf_off, pf_gap30=bool((dk > 30_000.0).any()),
                     reg_lead_s=float(reg.loc[mid].ws) - float(reg.loc[mid].first_seen_ns) / 1e9 if pd.notna(reg.loc[mid].first_seen_ns) else np.nan))
    for pol, rs, m, sz in arms:
        name = arm_name(pol, rs, m, sz); cs = cstate[name]
        r = run_window(mid, F, kf, set(rs), pol, m, sz, cap_state=cs)
        if r is None:
            continue
        cs['realised'] += r['pnl']; cs['windows_seen'] += 1
        cap_ = live_opts(set(rs))['losscap']
        if cap_ is not None and not cs['halted'] and cs['realised'] <= -cap_:
            cs['halted'] = True; cs['halted_at'] = dict(day=day, mid=int(mid), ws=ws_, realised=round(cs['realised'], 2))
        res[name].append((mid, ws_, r['pnl'], r['net_end'], r['n_fills'], r['shares'], r['min_worst'], r['refused'], r['halted'], r['zombie_fills'], r['mtm_stop'], r['carry_stop']))
        wh[name].update(r['withheld'])
        allf[name].extend(r['fills'])
        reqs[name].extend((t_, s_, k_, mid, l_) for t_, s_, k_, l_ in r['requests'])
        LCT[name].update(r['live_ctr']); DLY[name].extend(r['delays'])
    _FLIP_CACHE.clear()
    io.mem_check(f'win {mid}')
os.makedirs(f'{OUT}/fills', exist_ok=True)
import hashlib
summary = dict(day=day, scenario=SCEN, armset=ARMSET, timing=TIMING, tx_timing=TXSTAT, input_bounds=INPUT_BOUNDS,
               code_sha256={f: hashlib.sha256(open(f'{os.path.dirname(os.path.abspath(__file__))}/{f}', 'rb').read()).hexdigest() for f in ('pf_step4_ladder.py', 'pf_clock_rl.py', 'pf_io.py')},
               env={k: v for k, v in sorted(os.environ.items()) if k.startswith('PF_')}, preds=PREDS, fill_model=FILLMODEL, mode='EXPLORATORY', size=SIZE, order_latency_ms=L_O,
               order_live_ms=(L_PL if SPLIT else L_O), cancel_effective_ms=(L_CA if SPLIT else L_O),
               pending_cancel_after_create_ms=(CK.L_PENDCANCEL if SPLIT else None), decision_ms=(CK.D_DEC if SCEN == 'tokyoreal' else 0.0),
               predict_in_ms=(P_IN if SCEN == 'tokyo20' else f'tokyo-measured (receipt - {RL_SHIFT:.1f}, >= update + 13.2) + 10' if SCEN == 'tokyoreal' else 'recv-72'),
               binance_in_ms=(B_IN if SCEN in ('tokyo20', 'tokyoreal') else 'recv-72'), fill_known_ms=L_FILL, arms={})
for name, v in res.items():
    W = pd.DataFrame(v, columns=WCOLS)
    stem = name.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
    os.makedirs(f'{OUT}/win', exist_ok=True); os.makedirs(f'{OUT}/requests', exist_ok=True)
    W.to_parquet(f'{OUT}/win/{stem}.parquet', index=False)
    Q = pd.DataFrame(reqs.pop(name), columns=['t_ms', 'side', 'kind', 'mid', 'lane']); Q.to_parquet(f'{OUT}/requests/{stem}.parquet', index=False)
    f = pd.DataFrame(allf.pop(name), columns=FCOLS)
    f.to_parquet(f"{OUT}/fills/{name.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')}.parquet", index=False)
    f['kind'] = np.where(f.through, 'through', 'at')
    summary['arms'][name] = dict(
        windows=int(len(W)), gross_usd=round(float(W.pnl.sum()), 2), windows_positive=int((W.pnl > 0).sum()),
        fills=int(W.n_fills.sum()), shares=round(float(W.shares.sum()), 1), abs_net_end_p95=round(float(W.net_end.abs().quantile(0.95)), 1),
        mo1_usd=round(float(f.mo1.sum()), 2), mo60_usd=round(float(f.mo60.sum()), 2), rebate_usd_unverified=round(float(f.rebate.sum()), 2),
        by_kind={k: dict(shares=round(float(x.q.sum()), 1), mo1=round(float(x.mo1.sum()), 2), settle=round(float(x.settle.sum()), 2)) for k, x in f.groupby('kind')} if len(f) else {},
        by_lane={int(k): dict(fills=int(len(x)), shares=round(float(x.q.sum()), 1), mo1=round(float(x.mo1.sum()), 2), settle=round(float(x.settle.sum()), 2),
                              through_share=round(float((x.q * x.through).sum() / max(x.q.sum(), 1e-9)), 3), ahead0_median=round(float(x.ahead0.median()), 1))
                 for k, x in f.groupby('lane')} if len(f) else {},
        by_phase_60s=f.assign(ph=(f.t // 60 * 60).astype(int)).groupby('ph').agg(shares=('q', 'sum'), settle=('settle', 'sum')).round(2).to_dict('index') if len(f) else {},
        withheld_decisions=dict(wh[name]),
        mtm_stop_usd=round(float(W.mtm_stop.sum()), 2), carry_stop_usd=round(float(W.carry_stop.sum()), 2),
        carry_stop_sd=round(float(W.carry_stop.std()), 2), abs_net_stop_mean=round(float(W.net_end.abs().mean()), 1),
        live=dict(requests={k: int(c) for k, c in Q.kind.value_counts().items()}, refused_events=int(W.refused.sum()),
                  windows_with_refusal=int((W.refused > 0).sum()), halted_windows=int(W.halted.sum()), zombie_fills=int(W.zombie_fills.sum()),
                  cap_state_end=cstate[name], counters=dict(LCT[name]),
                  place_delay_ms=(dict(n=len(DLY[name]), p50=float(np.percentile(DLY[name], 50)), p90=float(np.percentile(DLY[name], 90)),
                                       max=float(np.max(DLY[name]))) if DLY[name] else None)))
    del f
D_ = pd.DataFrame(diag)
summary['live_data_diag'] = dict(windows=int(len(D_)), bn_silent_gt1s_quote_s=round(float(D_.bn_off_ms.sum()) / 1000.0, 2),
                                 bn_windows_hit=int((D_.bn_off_ms > 0).sum()), pf_silent_gt5s_quote_s=round(float(D_.pf_off_ms.sum()) / 1000.0, 2),
                                 pf_windows_hit=int((D_.pf_off_ms > 0).sum()), pf_windows_gap30=int(D_.pf_gap30.sum()),
                                 reg_lead_s=dict(min=round(float(D_.reg_lead_s.min()), 2), p01=round(float(D_.reg_lead_s.quantile(0.01)), 2),
                                                 p50=round(float(D_.reg_lead_s.median()), 2)),
                                 reg_lead_lt5s=int((D_.reg_lead_s < 5).sum()), reg_lead_nan=int(D_.reg_lead_s.isna().sum()))
io.write_json(f'{OUT}/capstate.json', cstate)
io.write_json(f'{OUT}/summary.json', summary)
print('[done]', day, SCEN, ARMSET, {k: (v['gross_usd'], v['fills']) for k, v in summary['arms'].items()})
