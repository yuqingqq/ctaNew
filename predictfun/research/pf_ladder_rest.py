"""LADDER_REST_INVENTORY_v1 OVER FULL MARKETS (P-2026-005; 2026-11-21, session aea5bfb9). The single current plan
(reviews/automatic/20261121_default_post_ladder/plan.md, commit cf4eee75), its metric contract (reviews/automatic/20261121_ladder_post_analysis/
metrics.md, e3c2b998) and the plan review (reviews/automatic/20261121_ladder_plan_review/report.md, a8f863af: design ready; log structural
subreasons and reconstructible decision states in the same batch). USER: large resting orders WITH inventory management, two price levels,
100 shares per level, quoting through the market, one experiment. One strategy, one run; no comparison arm, PI0, model, guard or fit; zero /
no trading is an arithmetic reference. Consumed Oct 24-27 only; sampled development data: no validation, full-day return or interval claim.
STRATEGY (pf_branch_ladder LADDER_REST_INVENTORY_v1 + LadderInventory; q0 = 100 per level, two levels per side (bid: best, best - 1 tick;
 ask: best, best + 1 tick), shared side cap C = Qside = 200, S = 100, R = 50, target 0; declared, not fitted): sizes q = floor(min(q0 f,
 D - B_side, room)) per vacant level after every command; the restriction mode pulls every live order of the restricted side (the inventory
 pull) and blocks its new orders; orders keep their gid, price, birth rung, quantity and queue (no re-centring, top-up, resize, eviction,
 price skew, taker hedge, liquidation or timer); duplicate prices and quotes crossing our own unclosed opposite orders are refused; the
 frozen structural rule per gid with its subreasons; M50 timing, 58 / 37 / 100 ms, the post-only refusal unchanged. The registry carries
 no venue quantity increment or minimum: the research convention (whole shares, minimum 1) is retained. Tick 1 cent: the registry's
 decimal_precision 2 and whole-cent tape prices are verified per market (status tick_unverified otherwise).
POPULATION (build_manifest, frozen before any simulation): the unique (day, market) ids of the ENTRY-EXPOSURE manifest (parent; its committed
 receipt copy, sha256 recorded); the opening = the registry window start ws (pf_joint_moment_data.metadata verifies the raw input hashes;
 registry sha256 recorded); cells = day x the UTC hour of the opening; per cell the minimum sha256('large_rest_inventory_full_market_v1|' +
 canonical {day, m}), ties by the key; every cell's candidates and the day's registry / parent market counts recorded; at most 96. Pilot:
 the first two selected markets of each date in canonical (m) order (in the sample). Partitions: one per date. No outcome enters; no backfill.
PATH (market_path): the first grid step t >= ws whose last RECEIVED frame holds a valid book starts the episode (nothing backdated; the
 opening wait split into steps with no frame received - missing capture - and with an invalid received book - not yet quotable);
 QP.build_path(W, t_start, ws + 298,000) with its rules unchanged (frames to the end, no venue frame gap > 1 s, a valid mark within 1 s
 before the end); any failure makes the market INCOMPLETE with that status (never a shorter slice, never bridged). END = ws + 298,000 -
 t_start (the supported pre-expiry endpoint; the last 2 s and settlement are not modelled). Boundary marks at ws + 60 s j (j = 1..4), the
 same builder; a missing one is a status and its two minutes are one interval.
RECORDS per market (run_market): the ledger (fills with the trade rows they consumed, cash, position, value, orders with quantities and
 birth rungs, requests, decisions, fallbacks, constraints, closures), the mode log, the structural log (every subreason flag, the primary
 reason, gid, rung, received frame and time, distance, mode, request, effect, acknowledgement), the command log of every client command
 (tick, side, live / slot / fallback, gid, rung, price, quantity, action, reason); per tick and side (post-decision) the client counts
 (post-pending, live, cancel-pending), venue-active orders and shares, distinct active prices, client / venue at-best flags; per tick the
 client / engine positions, reservations B / A, the mode, the engine cash and the venue mid; per fill the venue mid of the last frame with
 venue time <= the fill (and its age) and at t + h for h in MARKOUTS when t + h <= END (with ages); SNAPSHOTS (frozen sampling rule):
 every executed entry, each generation's first live review (fixed) and its minimum-rank live decision (rank = sha256(PREFIX|live|{day, m,
 side, gid, t}); eligible count = its live decisions), each with the 16 client context fields of the ladder client (ctx_*: own generation,
 age, ordinal, I, B, A, own state, opposite side's most active state), the view and its command index.
FEATURES (pass two, features_at): at the snapshots the 107 inputs rebuilt on the true clock: the 71 market inputs by the pipeline's
 real_features at the decision's grid step; the order inputs by known_steps_q (pf_pipeline_export.known_steps with the order's quantity,
 birth rung and the market horizon) and FC.V.order_inputs; fp_size = the order's quantity, fp_phase = the market phase at the decision,
 t_to_end = END - g; the 16 context fields; fair_edge (the archived 5-second fair vector) is NOT stretched: NaN by declaration. Statuses
 off_grid / no_market_row / no_mid. Per generation the client queue estimate at each of its live decisions (known_steps_q).
CHECKS IN THE RUN: the legacy reproduction on each selected market's first parent world with a 5-second path (pf_branch_ladder LEGACY +
 FixedPostKeep against pf_branch_engine.Branch + POST_KEEP: fills, decisions, requests, events, closures, constraints, fallbacks, value; and
 features_at in the legacy convention against pf_r2_teacher.Teacher.inputs, 107 columns, float32 bits); the reconstruction of every
 snapshot's client view (I, B, A, mode, the order's known fill and state) from the saved events alone (reconstruct_view).
MEASUREMENT (measure(); gross, zero cost, pre-expiry marked, cents; per day, side, minute and birth rung; equal market weight; complete
 markets only, every status counted): per fill execution edge s q (mid(t) - p), movement s q (mid(t + h) - mid(t)), markout s q (mid(t + h)
 - p) on each horizon's valid fills (and their common valid subset), END value s q (mark_END - p), summed once to cash + inventory x mark;
 the equity path at the venue mid per tick (gaps explicit) and its maximum drawdown; minute increments between the boundary marks
 (carry of the inventory held + the new fills), telescoping to the final value; generations (submitted / accepted, any-fill and quantity
 fill fractions on actual shares, distinct fill events, first / full fill delays from activation with censoring, queue estimates at the first
 live decision, time at the estimated front, rung-1 orders that later sit at the client best and their subsequent value, multi-level
 sweeps, removals by reason, full fill before removal, pending at END, first-review pulls, replacements and their delay, survivors, locked
 value before the earliest possible cancel effect, severe < -30 c per generation and < -1 c per submitted share); participation (bid / ask
 / either / both venue-active and at-best time on the full window and on valid-book time; downtime by a declared exclusive precedence with
 every flag kept); inventory (venue and client paths and discrepancy, time-average and peak |I|, time |I| >= 100, recovery from entering a
 restriction to |I| <= 50 with censoring, reversals and their END value, reservations, posted sizes, restricted / cap time, terminal I);
 losses (contributions by class, worst markets, top-5 market share of the gross negative market value with the losing market count, the
 value without severe generations as an EX-POST decomposition only); the decision-state map (the dictionary inputs at entries and sampled
 live states, quintile bins of the nonzero finite values + zero + missing, fixed on Oct 24-25, applied unchanged to every day).
READINGS (predicates(), computed patterns, not causal verdicts): the headline (final marked value, filled shares, both-side active time,
 severe-loss contribution per day); value sign per day; repeated fills; the carry versus new-fill sign split; the review's five patterns as
 flags (edge_then_adverse, short_positive_later_negative, low_participation, tail_erases_ordinary, deep_level_rarely_filled).
Usage (run_light): pf_ladder_rest.py selftest | manifest | pilot <day> | run <day> | measure"""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_entry_exposure as EX                           # noqa: E402  frozen: the parent manifest's plans and file rules
import pf_branch_ladder as LB                            # noqa: E402  the multi-order client
import pf_adv2_replay as AR                              # noqa: E402  frozen: mid_at
import pf_queue_place_test as QP                         # noqa: E402  frozen: build_path, the registry (JD.metadata)

C, pd, T, LS, L, R2, KP, A, BE, X = EX.C, EX.pd, EX.T, EX.LS, EX.L, EX.R2, EX.KP, EX.A, EX.BE, EX.X
FC = R2.FC
VERSION = 'ladder_rest_inventory_full_market_v1'
PREFIX = 'large_rest_inventory_full_market_v1|'             # the plan's population domain (section 3), unchanged from the single-level plan
DAYS = ('20261024', '20261025', '20261026', '20261027')
DISCOVERY = ('20261024', '20261025')
STEP, PLACE, CANCEL, NOTICE = float(BE.STEP), float(BE.PLACE), float(BE.CANCEL), float(BE.FILL_NOTICE)
SUPPORTED_END = LB.MARKET_END_MAX
MINUTE = 60000.
MARKOUTS = (50., 100., 1000., 5000., 30000.)
SEVERE_ABS, SEVERE_PER_SHARE = 30., 1.
PILOT_PER_DAY = 2
THIN = 30
STATES = ('absent', 'post_pending', 'live', 'cancel_pending', 'terminal')
MODE_CODE = {None: -1, 'UNRESTRICTED': 0, 'LONG_RESTRICTED': 1, 'SHORT_RESTRICTED': 2}
CMD_KINDS = ('live', 'slot')
CMD_ACTIONS = ('POST', 'WAIT', 'KEEP', 'PULL')
CMD_REASONS = ('post', 'keep', 'inventory_pull', 'restricted', 'zero_desired', 'exposure_full', 'no_room', 'cap', 'near_end', 'own_cross_blocked',
               'unavailable_price', 'fallback')
DOWNTIME = ('invalid_book', 'restricted', 'own_cross_blocked', 'near_end', 'cap', 'no_room', 'exposure_full', 'zero_desired', 'unavailable_price',
            'activation_latency', 'awaiting_notice', 'other')       # the declared exclusive precedence (metrics.md 3)
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_rest_20261121'
PARENT_REL = 'predictfun/reviews/entry_exposure_20261121/manifest.json'
OUT = A.DATA / 'ladder_rest_v1_20261121'
SOURCES = ('pf_ladder_rest.py', 'pf_ladder_rest_selftest.py', 'pf_branch_ladder.py', 'pf_branch_engine.py', 'pf_entry_exposure.py', 'pf_r2_shadow.py',
           'pf_r2_teacher.py', 'pf_r2_labels.py', 'pf_r2_label_sample.py', 'pf_pipeline_export.py', 'pf_queue_place_test.py', 'pf_adv2_replay.py')
# The declared feature dictionary of the decision-state map (metrics.md 4: spread, distance from best, depth / queue, age and remaining,
# signed flow, imbalance and changes, movement / volatility, source ages, remaining time, inventory, reservations, pending flags). Every
# column is one of the 107 inputs, except inv_signed (derived); adv_ = signed against the order's side (positive = adverse).
DICTIONARY = [
    ('pf_spread', 'spread', 'Predict spread (cents)', 'the last received Predict frame'),
    ('o_depth', 'distance from best', 'order price behind the received best (ticks; 0 = at best)', 'the last received Predict frame'),
    ('r_s1', 'displayed depth', 'own best level size / the window liquidity scale', 'the last received Predict frame'),
    ('fp_queue', 'queue', 'client-known queue ahead (shares; the shown size at the price at an entry)', 'frames received since activation (rule M)'),
    ('o_q_rel', 'queue', 'client-known queue ahead relative to the window scale', 'frames received since activation'),
    ('ctx_generation_age_ms', 'age', 'ms since the generation was submitted (0 at an entry)', 'the client clock'),
    ('fp_remaining', 'remaining', 'client-known remaining quantity (shares)', 'fill notices (+100 ms)'),
    ('adv_bn_flow1000', 'signed flow', 'Binance trade flow over the last 1 s, adverse-signed', 'Binance trades received by the decision'),
    ('adv_ofi250', 'signed flow', 'Predict order-flow imbalance over 250 ms, adverse-signed', 'Predict frames received by the decision'),
    ('r_own_pend', 'signed flow', 'own-side pending Predict flow over 1 s / the window scale', 'Predict trades received by the decision'),
    ('adv_pf_imb1', 'imbalance', 'Predict level-1 depth imbalance against the side', 'the last received Predict frame'),
    ('adv_d_bnimb250', 'imbalance change', 'Binance top-of-book imbalance change over 250 ms, adverse-signed', 'Binance book received by the decision'),
    ('own_d250', 'imbalance change', 'own-side best-level size change over 250 ms (negative = depletion)', 'received Predict frames'),
    ('adv_pf_dm1000', 'movement', 'Predict mid change over 1 s, adverse-signed', 'received Predict frames'),
    ('adv_bn_r250', 'movement', 'Binance mid return over 250 ms, adverse-signed', 'Binance book received by the decision'),
    ('adv_bn_r1000', 'movement', 'Binance mid return over 1 s, adverse-signed', 'Binance book received by the decision'),
    ('sigma', 'volatility', 'Binance volatility (unsigned)', 'Binance history received by the decision'),
    ('bn_ntr1000', 'volatility', 'Binance trades in the last 1 s (unsigned activity)', 'Binance trades received by the decision'),
    ('gap_adv', 'fair price', 'fair-price gap against the side', 'fair prices stamped by the decision'),
    ('pf_age', 'source age', 'age of the received Predict book state (ms)', 'the client clock'),
    ('bn_age', 'source age', 'age of the received Binance state (ms)', 'the client clock'),
    ('t_to_end', 'remaining time', 'ms to the supported market end', 'the client clock'),
    ('inv_signed', 'inventory', 'client-known inventory in the order\'s direction (bid +I, ask -I; shares)', 'fill notices (+100 ms)'),
    ('ctx_bid_reserved_shares', 'reservations', 'known remaining of every unclosed bid (shares)', 'the client state'),
    ('ctx_ask_reserved_shares', 'reservations', 'known remaining of every unclosed ask (shares)', 'the client state'),
    ('ctx_other_state_post_pending', 'pending flags', 'the opposite side\'s most active state is post-pending (0 / 1)', 'the client state')]
DICT_COLS = [d[0] for d in DICTIONARY]
C.require(float(X.SETTINGS.notification_ms) == NOTICE, 'the pipeline notice delay differs from the engine\'s')
C.require(all(c in T.FEATS for c in DICT_COLS if c != 'inv_signed'), 'a dictionary column outside the 107 inputs')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def committed_blob(rel):
    r = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{rel}'], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def committed_equal(rel, local):
    b = committed_blob(rel)
    return b is not None and Path(local).exists() and b == Path(local).read_bytes()


def frozen():
    """The runner, the client, the selftest, the declaration, the manifest and the launcher are the committed ones."""
    return bool(all(committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_rest.py', 'pf_ladder_rest_selftest.py', 'pf_branch_ladder.py'))
                and all(committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in ('declaration.md', 'manifest.json', 'launch_ladder_rest.sh')))


# ---------------------------------------------------------------------------------------------------------------------- the population
def rank(kind, key):
    return hashlib.sha256((PREFIX + (f'{kind}|' if kind else '') + LS.ckey(key)).encode()).hexdigest()


def registry(day):
    """The identity-bound registry of the day (raw input hashes verified) -> ({m: (opening ms, decimal_precision)}, registry sha256, metadata
    receipt sha256)."""
    raw, reg = QP.JD.metadata(day, True); p = Path(QP.I.K.IO) / 'step1' / day / 'registry.parquet'
    return ({int(r.mid): (float(r.ws) * 1000., int(r.decimal_precision)) for r in reg.itertuples()}, C.sha(p), C.sha(Path(QP.JD.C.OUT) / day / 'receipt.json'))


def utc_hour(day, t_ms):
    return LS.utc_hour(day, int(round(t_ms)))


def select_markets(cands, opens):
    """cands: {day: sorted unique market ids}; opens: {day: {m: (opening ms, precision)}} -> (cells, selected in canonical (day, m) order)."""
    cells, sel = [], []
    for day in DAYS:
        by = {}
        for m in cands.get(day, []):
            C.require(m in opens[day], f'{day} market {m} has no registry opening'); by.setdefault(utc_hour(day, opens[day][m][0]), []).append(m)
        for h in sorted(by):
            ms = sorted(by[h], key=lambda m: (rank('', dict(day=day, m=int(m))), LS.ckey(dict(day=day, m=int(m)))))
            cells.append(dict(day=day, hour=int(h), candidates=len(ms), candidate_ids=sorted(int(m) for m in ms)))
            sel.append(dict(day=day, m=int(ms[0]), hour=int(h), ws=float(opens[day][ms[0]][0]), decimal_precision=int(opens[day][ms[0]][1]),
                            rank=rank('', dict(day=day, m=int(ms[0]))), cell_candidates=len(ms)))
    sel.sort(key=lambda x: (x['day'], x['m']))
    return cells, sel


def build_manifest(path=None, parent_path=None, reg_fn=None, plans=None, identity=True):
    """The frozen population (header) -> REPO/RECEIPT/manifest.json, write-once. reg_fn / plans: hand inputs in the selftest only."""
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    parent_path = Path(parent_path or REPO / PARENT_REL); par = json.loads(parent_path.read_text())
    if parent_path == REPO / PARENT_REL:
        C.require(committed_equal(PARENT_REL, parent_path), 'the parent manifest is not the committed one')
    reg_fn = reg_fn or registry; plans = plans or {k: json.loads((Path(p['dir']) / 'plan.json').read_text()) for k, p in EX.PLANS.items()}
    cands, worlds, kinds = {}, {}, {}
    for kind, S in par['sets'].items():
        for f in S['forks']:
            d, m = str(f['day']), int(f['m']); cands.setdefault(d, set()).add(m); kinds[d] = kind; worlds.setdefault((d, m), set()).add(int(f['t_ms']))
    C.require(all(d in cands for d in DAYS), 'a declared day has no parent market')
    opens, regs = {}, {}
    for day in DAYS:
        o_, rsha, msha = reg_fn(day); opens[day] = o_
        regs[day] = dict(registry_sha256=rsha, metadata_receipt_sha256=msha, registry_markets=len(o_), parent_markets=len(cands.get(day, ())))
    cells, sel = select_markets({d: sorted(v) for d, v in cands.items()}, opens)
    for x in sel:
        x['parent_worlds'] = sorted(worlds[(x['day'], x['m'])]); x['set'] = kinds[x['day']]
    pilot = {day: [x['m'] for x in sel if x['day'] == day][:PILOT_PER_DAY] for day in DAYS}
    inputs = {day: dict(set=kinds[day], kp_dir=plans[kinds[day]]['kp_dir'], files=EX.PLANS[kinds[day]]['files'], inputs=plans[kinds[day]]['sample'][day]['inputs'],
                        plan_sha256=C.sha(Path(EX.PLANS[kinds[day]]['dir']) / 'plan.json') if identity else None) for day in DAYS}
    man = dict(version=VERSION, prefix=PREFIX, parent=dict(rel=PARENT_REL, sha256=C.sha(parent_path)), days=list(DAYS), discovery=list(DISCOVERY),
               config=LB.CONFIGS[LB.LADDER], legacy=LB.CONFIGS[LB.LEGACY], contract=LB.CONTRACT, settings=dict(BE.SETTINGS), supported_end_ms=SUPPORTED_END,
               markouts=list(MARKOUTS), cells=cells, selected=sel, pilot=pilot, registry=regs, inputs=inputs,
               dictionary=[dict(col=c, family=f, meaning=m_, clock=k_) for c, f, m_, k_ in DICTIONARY],
               sampling=dict(entries='every executed entry', first='each generation\'s first live review', live='per generation the minimum sha256(PREFIX|live|{day, m, side, gid, t}) live decision; eligible = its live decisions'),
               feature_schema=dict(n=len(T.FEATS), sha256=hashlib.sha256(LS.ckey(list(T.FEATS)).encode()).hexdigest()), asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(json.loads(json.dumps(man, default=str)), indent=1, sort_keys=True) + '\n')
    return man


# ---------------------------------------------------------------------------------------------------------------------- the path
def market_start(W, ws):
    """(the first grid step >= ws whose last received frame holds a valid book, or None; steps before it with a frame received / without)."""
    tg = np.asarray(W[0]['tg'], float); jk = np.asarray(W[0]['jk']); b, a = np.asarray(W[0]['P5'], float)[:, 0], np.asarray(W[1]['P5'], float)[:, 0]
    seen = none = 0
    for k in range(len(tg)):
        t = float(tg[k])
        if t < ws:
            continue
        if t >= ws + SUPPORTED_END:
            break
        j = int(jk[k])
        if j >= 0 and 0 <= b[j] < a[j] <= 100:
            return t, seen, none
        seen += int(j >= 0); none += int(j < 0)
    return None, seen, none


def whole_cents(path):
    """Every book price and trade price of the path is a whole cent (the tick convention)."""
    f = np.asarray(path['frames'], float); pr = f[:, [2, 3, 4, 5, 6, 12, 13, 14, 15, 16]]; tr = np.asarray(path['trades'], float)
    return bool(np.all((pr < 0) | (np.abs(pr - np.round(pr)) < 1e-9)) and np.all(np.abs(tr[:, 2] - np.round(tr[:, 2])) < 1e-9))


def market_path(W, ws, precision=2):
    """The complete supported path of one market (header) -> (path, status, info)."""
    t0, seen, none = market_start(W, ws); info = dict(ws=float(ws), wait_steps_frame_received=int(seen), wait_steps_no_frame=int(none))
    if t0 is None:
        return None, ('no_quotable_book' if seen else 'no_frame_received'), info
    k0 = int(np.searchsorted(np.asarray(W[0]['tg'], float), t0)); j0 = int(np.asarray(W[0]['jk'])[k0])
    info.update(t_start=float(t0), opening_wait_ms=float(t0 - ws), known_frame_receipt_rel_ms=float(np.asarray(W[0]['kf'], float)[j0] - ws))
    p, st = QP.build_path(W, t0, ws + SUPPORTED_END)
    if p is None:
        return None, st, info
    if int(precision) != 2 or not whole_cents(p):
        return None, 'tick_unverified', info
    marks = {}
    for j in range(1, 5):
        T_ = ws + j * MINUTE
        if T_ > t0:
            pj, sj = QP.build_path(W, t0, T_); marks[str(j)] = dict(t_rel=float(T_ - t0), mark=float(pj['mark']) if pj is not None else None, status=sj)
    info.update(end_rel=float(ws + SUPPORTED_END - t0), boundary_marks=marks)
    return p, 'complete', info


# ---------------------------------------------------------------------------------------------------------------------- snapshots
def ladder_ctx(B, v, o):
    """The 16 client context fields on the ladder client (header): own generation (the live order; none at an entry), its age and ordinal,
    I, B, A, the own state and the opposite side's most active state (live > cancel_pending > post_pending > absent). For one level they are
    pf_r2_shadow.client_snapshot's (selftest)."""
    s, g = int(v['side']), float(v['t']); side_g = [int(x['gid']) for x in B.E.o if int(x['side']) == s]
    other = [B.gid_state(gid, g) for gid in B.unclosed[1 - s]]
    ost = next((z for z in ('live', 'cancel_pending', 'post_pending') if z in other), 'absent'); st = o['state'] if o else 'absent'
    x = dict(ctx_has_generation=float(o is not None), ctx_generation_age_ms=float(g - o['submit']) if o else 0.,
             ctx_generation_ordinal=float(side_g.index(int(o['gid'])) + 1) if o else float(len(side_g) + 1), ctx_inventory_shares=float(v['position']),
             ctx_bid_reserved_shares=float(v['B']), ctx_ask_reserved_shares=float(v['A']))
    for nm, z_ in (('self', st), ('other', ost)):
        for z in STATES:
            x[f'ctx_{nm}_state_{z}'] = float(z_ == z)
    return x


def snap(B, v, o, slot, kind):
    """A decision snapshot (header): the order (live) or the slot (price, rung, q) of an entry."""
    s, g = int(v['side']), float(v['t'])
    if o is not None:
        price, rung, q, act, gid = float(o['price']), float(o['rung']), float(o['size']), float(o['act']), int(o['gid'])
    else:
        (price, rung, q), act, gid = slot, g + PLACE, None
    return dict(kind=kind, t=g, side=s, gid=gid, price=float(price), rung=float(rung), q=float(q), act=float(act), position=float(v['position']), B=float(v['B']),
                A=float(v['A']), mode=v['mode'], bid=float(v['bid']), ask=float(v['ask']), shown_best=float(v['bid_size'] if s == 0 else v['ask_size']),
                state=o['state'] if o else 'absent', known_filled=float(o['known_filled']) if o else 0., cmd_index=len(B.cmd_log), ctx=ladder_ctx(B, v, o))


class Recorded(LB.Policy):
    """The strategy (inner, its actions unchanged) with the snapshots of the frozen sampling rule (header)."""

    def __init__(self, inner, B, key, teacher=None):
        super().__init__(inner.name, inner.horizon); self.inner, self.B, self.key, self.teacher = inner, B, dict(key), teacher
        self.entries, self.first, self.live_s, self.n_live = {}, {}, {}, {}

    def live(self, v, o):
        a = self.inner.live(v, o); s, g, gid = int(v['side']), float(v['t']), int(o['gid']); kk = (s, gid); self.n_live[kk] = self.n_live.get(kk, 0) + 1
        if kk not in self.first:
            self.first[kk] = snap(self.B, v, o, None, 'first')
        r = rank('live', dict(self.key, side=s, gid=gid, t=int(round(g))))
        if kk not in self.live_s or r < self.live_s[kk]['rank']:
            self.live_s[kk] = dict(snap(self.B, v, o, None, 'live'), rank=r)
        return a

    def slot(self, v, r, price):
        a, q, why = self.inner.slot(v, r, price); s, g = int(v['side']), float(v['t'])
        if a == 'POST':
            room = (self.B.cap - v['position'] - v['B']) if s == 0 else (self.B.cap + v['position'] - v['A'])
            if q <= room + LB.TOL and g + PLACE < self.B.end:
                self.entries[(s, int(round(g)), int(r))] = snap(self.B, v, None, (price, r, q), 'entry')
        return a, q, why


class Observer:
    """Per tick (post-decision) the client counts and venue diagnostics, read from the branch only (header)."""

    def __init__(self, B):
        nt = LB.n_ticks(B); self.nt = nt; z = lambda dt: np.zeros((nt, 2), dt)
        self.npp, self.nlive, self.ncp, self.nva = z(np.int8), z(np.int8), z(np.int8), z(np.int8)
        self.qva, self.nprice, self.bestc, self.bestv, self.bookok = z(np.float32), z(np.int8), z(np.int8), z(np.int8), z(np.int8)
        self.posc, self.pose, self.cash, self.mid = (np.zeros(nt) for _ in range(4)); self.res = np.zeros((nt, 2)); self.mode = np.full(nt, -1, np.int8)
        self.fi = self.ki = self.jv = 0; self.pe = self.pk = self.ce = 0.; self.kfill = {}

    def __call__(self, B, k):
        g = k * STEP; E = B.E; fl = E.fills
        while self.fi < len(fl) and fl[self.fi][0] <= g:
            t, gid, s_, q, p, _ = fl[self.fi]; sg = 1. if s_ == 0 else -1.; self.pe += sg * q; self.ce -= sg * q * p; self.fi += 1
        while self.ki < self.fi and fl[self.ki][0] + NOTICE <= g:
            t, gid, s_, q, _, _ = fl[self.ki]; self.pk += (1. if s_ == 0 else -1.) * q; self.kfill[gid] = self.kfill.get(gid, 0.) + q; self.ki += 1
        while self.jv + 1 < E.nf and E.f[self.jv + 1, 0] <= g:
            self.jv += 1
        fk, fv = E.f[E.known], E.f[self.jv]
        for s in (0, 1):
            best_c, best_v = float(fk[2 + 10 * s]), float(fv[2 + 10 * s]); self.bookok[k, s] = int(LB.book_ok_side(fk, s)); prices = set(); r_ = 0.
            for gid in B.unclosed[s]:
                o = E.o[gid]; st = B.gid_state(gid, g); r_ += o['size'] - self.kfill.get(gid, 0.)
                self.npp[k, s] += st == 'post_pending'; self.nlive[k, s] += st == 'live'; self.ncp[k, s] += st == 'cancel_pending'
                va = (not o['refused']) and o['act'] <= g < o['remove'] and o['rem'] > LB.TOL
                if va:
                    self.nva[k, s] += 1; self.qva[k, s] += o['rem']; prices.add(o['price']); self.bestv[k, s] |= int(o['price'] == best_v)
                if st in ('live', 'cancel_pending') and o['price'] == best_c:
                    self.bestc[k, s] = 1
            self.nprice[k, s] = len(prices); self.res[k, s] = r_
        self.posc[k], self.pose[k], self.cash[k] = self.pk, self.pe, self.ce
        b_, a_ = fv[2], fv[12]; self.mid[k] = (b_ + a_) / 2. if (b_ >= 0 and a_ >= 0) else np.nan; self.mode[k] = MODE_CODE[B.mode]

    def arrays(self):
        return dict(npp=self.npp, nlive=self.nlive, ncp=self.ncp, nva=self.nva, qva=self.qva, nprice=self.nprice, bestc=self.bestc, bestv=self.bestv,
                    bookok=self.bookok, posc=self.posc, pose=self.pose, cash=self.cash, mid=self.mid, res=self.res, mode=self.mode)


def jl(x):
    return json.loads(json.dumps(list(x), default=str))


def mids_of(frames, fills, end):
    """Per fill [mid(t), age(t)] + per horizon [mid(t + h), age(t + h)] (None beyond END or without a two-sided frame)."""
    out = []
    for f in fills:
        t = float(f[0]); row = []
        for h in (0.,) + MARKOUTS:
            tt = t + h
            if tt > end:
                row += [None, None]; continue
            j = int(np.searchsorted(frames[:, 0], tt, side='right')) - 1; m_ = AR.mid_at(frames, tt)
            row += [m_, None if (j < 0 or m_ is None) else float(tt - frames[j, 0])]
        out.append(row)
    return out


def ledger_record(B, Lg):
    """The finished ladder branch's evidence (header); decisions and the command log go to the arrays."""
    return dict(status=Lg['status'], cash=Lg.get('cash'), position=Lg.get('position'), value=Lg.get('value'), mark=B.mark, end=B.end, fills=jl(B.E.fills),
                fallbacks=jl(B.fallbacks), constraints=jl(B.constraints), requests=jl([e_ for e_ in B.E.events if e_[1].endswith('_request')]),
                closed={str(k): list(v) for k, v in B.closed.items()}, open_at_end={str(k): v for k, v in (Lg.get('open_at_end') or {}).items()},
                open_at_end_gid={str(k): v for k, v in (Lg.get('open_at_end_gid') or {}).items()}, mode_log=jl(B.mode_log), mode_end=B.mode, struct_log=jl(B.struct_log),
                orders=[dict(gid=o['gid'], side=o['side'], rung=o['rung'], price=o['price'], size=o['size'], rem=o['rem'], act=o['act'], remove=o['remove'],
                             submit=o['submit'], refused=bool(o['refused']), req=o['req'], why=o['why'], ahead_final=o['ahead']) for o in B.E.o],
                settings=dict(B.settings), ident=dict(B.ident), tape_id=BE.tape_id(B.E.f, B.E.tr, B.mark), fingerprint=LB.fingerprint(B), n_decisions=len(B.decisions),
                n_commands=len(B.cmd_log))


def command_arrays(B):
    """The decisions and the command log as arrays (codes: STATES, CMD_ACTIONS, CMD_KINDS, CMD_REASONS)."""
    dc = np.array([[d[0], d[1], STATES.index(d[2]), CMD_ACTIONS.index(d[3]), -1 if d[4] is None else d[4]] for d in B.decisions], float).reshape(-1, 5)
    cm = np.array([[c[0], c[1], CMD_KINDS.index(c[2]), c[3], c[4], c[5], c[6], CMD_ACTIONS.index(c[7]), CMD_REASONS.index(c[8])] for c in B.cmd_log], float).reshape(-1, 9)
    return dict(decisions=dc, commands=cm)


def run_ladder(path, key, ident):
    """One market under LADDER_REST_INVENTORY_v1 from its supported start to END -> (branch, ledger record, recorder, observer). The initial
    queue ahead of each generation is read right after its submission tick (the engine's add9 value; trades apply at the next tick)."""
    B = LB.LadderBranch(path, LB.LADDER, end=ident['end_rel'], episode=0, ident=ident); pol = Recorded(LB.LadderInventory(B), B, key); obs = Observer(B); ahead0 = {}

    def observe(B_, k):
        obs(B_, k)
        while len(ahead0) < len(B_.E.o):
            o = B_.E.o[len(ahead0)]; ahead0[int(o['gid'])] = float(o['ahead'])
    LB.run_episode(B, pol, observe=observe); Lg = LB.finish(B); rec = ledger_record(B, Lg); rec['ahead0'] = {str(k): v for k, v in ahead0.items()}
    used = sorted({int(f[5]) for f in B.E.fills})
    rec['trade_rows'] = {str(i): [float(x) for x in B.E.tr[i]] for i in used}
    rec['trades_total'] = {str(s): dict(n=int((B.E.tr[:, 1] == s).sum()), q=float(B.E.tr[B.E.tr[:, 1] == s, 3].sum())) for s in (0, 1)}
    rec['fill_mids'] = mids_of(np.asarray(B.E.f, float), B.E.fills, B.end)
    return B, rec, pol, obs


# ---------------------------------------------------------------------------------------------------------------------- reconstruction
def reconstruct_view(rec, cmds, sn):
    """A snapshot's client view recomputed from the saved events alone (no simulation): I from the fills notified by t; the unclosed orders at
    the snapshot (submitted before t, or at t by an earlier command, and not closed by t); B / A their known remaining; the mode after the
    transitions up to t; the order's known fill and state (header)."""
    t = float(sn['t']); fills = rec['fills']; orders = {int(o['gid']): o for o in rec['orders']}; closed = {int(k): v for k, v in rec['closed'].items()}
    I = sum((1. if int(f[2]) == 0 else -1.) * float(f[3]) for f in fills if float(f[0]) + NOTICE <= t)
    created_at_t = {int(c[3]) for c in cmds[:int(sn['cmd_index'])] if int(c[7]) == CMD_ACTIONS.index('POST') and float(c[0]) == t and int(c[3]) >= 0}
    kf = lambda gid: sum(float(f[3]) for f in fills if int(f[1]) == gid and float(f[0]) + NOTICE <= t)
    res = {0: 0., 1: 0.}
    for gid, o in orders.items():
        alive = (float(o['submit']) < t or gid in created_at_t) and not (gid in closed and float(closed[gid][1]) <= t)
        if alive:
            res[int(o['side'])] += float(o['size']) - kf(gid)
    mode = 'UNRESTRICTED' if rec['settings'].get('inventory') else None
    for g_, I_, old, new in rec['mode_log']:
        if float(g_) <= t:
            mode = new
    out = dict(position=I, B=res[0], A=res[1], mode=mode)
    if sn.get('kind') != 'entry' and sn.get('gid') is not None:
        o = orders[int(sn['gid'])]; out['known_filled'] = kf(int(sn['gid']))
        out['state'] = 'post_pending' if t < float(o['act']) + BE.POST_NOTICE else ('cancel_pending' if (o['req'] is not None and float(o['req']) < t) else 'live')
    return out


def reconstruct_ok(rec, cmds, sn):
    r = reconstruct_view(rec, cmds, sn); keys = ['position', 'B', 'A', 'mode'] + (['known_filled', 'state'] if (sn.get('kind') != 'entry' and sn.get('gid') is not None) else [])
    return all((abs(float(r[k]) - float(sn[k])) <= 1e-9) if isinstance(r[k], float) else r[k] == sn[k] for k in keys)


# ---------------------------------------------------------------------------------------------------------------------- the features
def known_steps_q(w, side, ta, pt, fl, d, terminal, size, exposure, rung=0., notice=NOTICE):
    """pf_pipeline_export.known_steps (its production branch: the running-minimum queue, the v2 pre-activation POST state) with the order's
    submitted quantity, birth rung and horizon as parameters; equal to it at size 30, rung 0, exposure 5,000 (selftest)."""
    clock = np.asarray(w['kf'], float); d = np.asarray(d, float); ft, fq = (np.asarray(x, float) for x in fl)
    C.require(bool(np.isfinite(clock).all() and (np.diff(clock) >= 0).all()), 'nonmonotone known frame clock')
    ja = int(np.searchsorted(clock, ta, side='right') - 1); jt = (np.searchsorted(clock, d, side='right') - 1).astype(int)
    filled = np.array([sum(float(b) for a, b in zip(ft, fq) if a + notice <= x) for x in d], float)
    lo, hi = max(ja, 0), int(jt.max()) if len(jt) else -1
    sh = np.array([X.AC.shown_at(w, f, side, pt) for f in range(lo, hi + 1)], float) if hi >= lo else np.zeros(0)
    fin = np.where(np.isfinite(sh), sh, np.inf); run = np.minimum.accumulate(fin) if len(fin) else fin
    shown = np.array([(sh[j - lo] if j >= lo else X.AC.shown_at(w, j, side, pt)) if j >= 0 else np.nan for j in jt], float)
    ok = (ja >= 0) & (ja <= jt)
    initial = np.where(ok, sh[0] if len(sh) else np.nan, np.nan)
    ahead = np.where(ok, np.array([run[j - lo] if (o_ and 0 <= j - lo < len(run)) else np.inf for j, o_ in zip(jt, ok)]), np.nan)
    ahead = np.where(np.isfinite(ahead), ahead, np.nan); pre = d < ta
    initial, ahead = np.where(pre, shown, initial), np.where(pre, shown, ahead)
    best = np.array([float(w['P5'][j][0]) if j >= 0 else np.nan for j in jt])
    return dict(side=np.full(len(d), float(side)), price=np.full(len(d), float(pt)), rung=np.full(len(d), float(rung)), depth=(best - pt) if side == 0 else (pt - best),
                age_ms=d - ta, submitted_quantity=np.full(len(d), float(size)), known_remaining=np.maximum(0., float(size) - filled), known_filled=filled,
                queue_initial_known=initial, queue_ahead_known=ahead, shown_known=shown, queue_initial_unknown=(~np.isfinite(initial)).astype(float),
                effective_horizon_ms=np.minimum(exposure, terminal - d))


def features_at(W_, s, t_abs0, pt, end, ws, market, fills, legacy=None):
    """The 107 inputs at one snapshot (header): fills = the generation's fills [(t rel, q)]; market = {grid step: (x, scale)} of side s; legacy
    = dict(fv, phase) for the legacy convention (else the full-market one) -> (row or None, status)."""
    sg = 1. if s == 0 else -1.; g = float(pt['t']); d = t_abs0 + g; tg = np.asarray(W_[s]['tg'], float); kk = min(int(np.searchsorted(tg, d)), len(tg) - 1)
    if abs(tg[kk] - d) > 1e-6:
        return None, 'off_grid'
    if kk not in market:
        return None, 'no_market_row'
    xm, scale = market[kk]; ref = float(xm[FC.MARKET.index('pf_mid')])
    if not np.isfinite(ref):
        return None, 'no_mid'
    ft = np.array([t_abs0 + float(t) for t, q in fills], float); fq = np.array([float(q) for t, q in fills], float)
    kn = known_steps_q(W_[s], s, t_abs0 + float(pt['act']), float(pt['price']), (ft, fq), np.array([d]), t_abs0 + end, float(pt['q']), end, rung=float(pt['rung']))
    oi = FC.V.order_inputs(pd.DataFrame({f'x_{a}': kn[a] for a in X.KNOWN}), np.array([scale]), np.array([ref]))
    out = {nm: float(xm[i_]) for i_, nm in enumerate(FC.MARKET)}; out.update({nm: float(np.asarray(oi[nm], float)[0]) for nm in FC.ORDER})
    k_ep = int(round(g / STEP))
    if legacy is not None:
        fv = legacy['fv']; fe = sg * (float(fv[min(k_ep, len(fv) - 1)]) - float(pt['price'])); ph = float(legacy['phase'])
    else:
        fe = np.nan; ph = float(min(5, int((d - ws) // MINUTE) + 1))
    out.update(fp_size=float(pt['q']), fp_remaining=float(kn['known_remaining'][0]), fp_queue=float(kn['queue_ahead_known'][0]), fp_shown=float(kn['shown_known'][0]),
               fp_scale=float(scale), fp_phase=ph, fair_edge=fe, t_to_end=end - g)
    out.update(pt['ctx'])
    return out, 'ok'


def market_rows(W_, fair_m, depth, feat_fn, t_abs0, pts):
    """The pipeline's market inputs at the grid steps of the snapshots (one builder call)."""
    chosen = {0: set(), 1: set()}
    for s, g in pts:
        tg = np.asarray(W_[s]['tg'], float); chosen[s].add(min(int(np.searchsorted(tg, t_abs0 + g)), len(tg) - 1))
    F = feat_fn(W_, fair_m, depth, {s: sorted(v) for s, v in chosen.items()})
    return {s: F.get(s, {}) for s in (0, 1)}


def fills_by_gid(fills):
    out = {}
    for f in fills:
        out.setdefault(int(f[1]), []).append((float(f[0]), float(f[3])))
    return out


def all_snaps(pol, rec):
    """Every snapshot of the frozen sampling rule with its generation and denominators."""
    sub = {(int(o['side']), int(round(o['submit'])), int(round(o['rung']))): int(o['gid']) for o in rec['orders']}; out = []
    for (s, g, r), sn in sorted(pol.entries.items()):
        out.append(dict(sn, gid=sub.get((s, g, r)), n_eligible=None, rank=None))
    for kk, sn in sorted(pol.first.items()):
        out.append(dict(sn, n_eligible=pol.n_live[kk], rank=None))
    for kk, sn in sorted(pol.live_s.items()):
        out.append(dict(sn, n_eligible=pol.n_live[kk]))
    return out


def point_rows(W_, fair_m, depth, feat_fn, t_abs0, ws, rec, snaps):
    """Pass two: the 107 inputs at every snapshot (statuses kept)."""
    mr = market_rows(W_, fair_m, depth, feat_fn, t_abs0, [(sn['side'], sn['t']) for sn in snaps]); fb = fills_by_gid(rec['fills']); rows = []
    for sn in snaps:
        row, st = features_at(W_, sn['side'], t_abs0, sn, rec['end'], ws, mr[sn['side']], fb.get(sn['gid'], []) if sn['gid'] is not None else [])
        x = dict(kind=sn['kind'], side=int(sn['side']), gid=-1 if sn['gid'] is None else int(sn['gid']), t=float(sn['t']), q=float(sn['q']), rung=float(sn['rung']),
                 price=float(sn['price']), position=float(sn['position']), mode=sn['mode'], shown_best=sn['shown_best'], n_eligible=-1 if sn['n_eligible'] is None else int(sn['n_eligible']),
                 rank=sn.get('rank') or '', status=st)
        if row is not None:
            x.update({c: float(np.float32(row[c])) for c in T.FEATS})
        rows.append(x)
    return rows


def queue_summary(W_, rec, cmds, t_abs0):
    """Per generation the client queue estimate at each of its live commands (known_steps_q): count, at the estimated front (queue ahead 0),
    unknown, the first / last estimates and, for a rung-1 order, its first live command at the received best (aged deeper order at best)."""
    by = {}
    for c in cmds:
        if int(c[2]) == CMD_KINDS.index('live') and int(c[3]) >= 0:
            by.setdefault(int(c[3]), []).append((float(c[0]), float(c[5])))
    fb = fills_by_gid(rec['fills']); out = {}
    for o in rec['orders']:
        ts = by.get(int(o['gid']))
        if not ts:
            continue
        s = int(o['side']); fl = fb.get(int(o['gid']), []); ft = np.array([t_abs0 + t for t, q in fl], float); fq = np.array([q for t, q in fl], float)
        kn = known_steps_q(W_[s], s, t_abs0 + float(o['act']), float(o['price']), (ft, fq), t_abs0 + np.asarray([t for t, _ in ts], float), t_abs0 + float(rec['end']),
                           float(o['size']), float(rec['end']), rung=float(o['rung']))
        a = kn['queue_ahead_known']; dep = kn['depth']; at_best = [i for i in range(len(ts)) if np.isfinite(dep[i]) and dep[i] == 0.]
        fi = lambda x: None if not np.isfinite(x) else float(x)
        out[str(o['gid'])] = dict(n=len(ts), front=int(np.sum(a == 0)), unknown=int(np.sum(~np.isfinite(a))), first=fi(a[0]), last=fi(a[-1]), last_t=float(ts[-1][0]),
                                  first_best_t=float(ts[at_best[0]][0]) if (at_best and float(o['rung']) >= 1) else None,
                                  first_best_queue=fi(a[at_best[0]]) if (at_best and float(o['rung']) >= 1) else None)
    return out


# ---------------------------------------------------------------------------------------------------------------------- the legacy reproduction
class LegacyRec(BE.Policy):
    """POST_KEEP on the FROZEN client (its action unchanged) recording, at every executable entry and first live review, pf_r2_teacher
    .Teacher.inputs on the branch (a feature-only teacher: no model)."""

    def __init__(self, B, world):
        super().__init__(self.fn, 'post_keep_legacy_record', horizon=B.end); self.B = B; self.tf = T.Teacher(None, None); self.tf.bind(B, world); self.pts = {}; self.seen = set()

    def fn(self, v):
        s, g = int(v['side']), float(v['t'])
        if v['state'] == 'absent' and v['room'] >= -LB.TOL and g + PLACE < self.B.end:
            self.pts[('entry', int(round(g)), s)] = self.tf.inputs(v, self.B, s, g)
        elif v['state'] == 'live' and v['gid'] not in self.seen:
            self.seen.add(v['gid']); self.pts[('first', int(round(g)), s)] = self.tf.inputs(v, self.B, s, g)
        return BE.POST_KEEP(v)


def legacy_check(day, w, world5, feat_fn):
    """The bounded legacy reproduction on one 5-second world (header) -> its readings."""
    path = world5['path']; ident = R2.episode_ident(day, w['o'][0])
    B1 = BE.Branch(path, 0, ident=ident); lr = LegacyRec(B1, world5)
    for k in range(int(np.ceil(B1.end / STEP))):
        BE.step(B1, k, lr)
    L1 = BE.finish(B1)
    B2 = LB.LadderBranch(path, LB.LEGACY, ident=ident); rc = Recorded(LB.FixedPostKeep(B2), B2, dict(day=day, m=int(w['m']))); LB.run_episode(B2, rc); L2 = LB.finish(B2)
    eq = dict(fills=jl(L1['fills']) == jl(L2['fills']), decisions=jl(L1['decisions']) == jl(L2['decisions']), requests=jl(L1['requests']) == jl(L2['requests']),
              events=jl(B1.E.events) == jl(B2.E.events), closed=B1.closed == B2.closed, constraints=jl(L1['constraints']) == jl(L2['constraints']),
              fallbacks=jl(L1['fallbacks']) == jl(L2['fallbacks']), value=L1.get('value') == L2.get('value'),
              open_at_end={int(k): v for k, v in L1['open_at_end'].items()} == {int(k): v for k, v in L2['open_at_end'].items()})
    t0 = float(w['o'][0].t); C.require(float(w['o'][1].t) == t0, 'a world whose two originals start at different times')
    snaps = [dict(sn, kind='entry') for sn in rc.entries.values()] + [dict(sn, kind='first') for sn in rc.first.values()]
    mr = market_rows(world5['W'], world5['fair_m'], world5['depth'], feat_fn, t0, [(sn['side'], sn['t']) for sn in snaps]); fb = fills_by_gid(L2['fills'])
    n_eq = n_bad = n_both_na = n_missing = 0; bad_cols = {}
    for sn in snaps:
        s = sn['side']; key = (sn['kind'], int(round(sn['t'])), s)
        if key not in lr.pts:
            n_missing += 1; continue
        tch = lr.pts[key]
        row, st = features_at(world5['W'], s, t0, sn, BE.END, None, mr[s], fb.get(sn['gid'], []) if sn['gid'] is not None else [],
                              legacy=dict(fv=world5['fv'], phase=w['o'][s].phase))
        if tch is None and row is None:
            n_both_na += 1; continue
        if tch is None or row is None:
            n_bad += 1; continue
        cols = [c for c in T.FEATS if np.float32(row[c]).tobytes() != np.float32(tch[c]).tobytes()]
        n_bad += int(bool(cols)); n_eq += int(not cols)
        for c in cols:
            bad_cols[c] = bad_cols.get(c, 0) + 1
    return dict(m=int(w['m']), t_ms=int(w['t_ms']), engine_equal=eq, engine_all=bool(all(eq.values())), points=len(snaps), teacher_points=len(lr.pts),
                points_unmatched=n_missing + (len(lr.pts) - (len(snaps) - n_missing)), features_equal=n_eq, features_bad=n_bad, features_both_unavailable=n_both_na,
                bad_cols=bad_cols, value=L2.get('value'), fills=len(L2['fills']))


# ---------------------------------------------------------------------------------------------------------------------- one market, one day
def run_market(day, x, prs, fair, depth, feat_fn, allw, opens):
    """One selected market: the legacy reproduction on its first parent world, the full supported path, the strategy, pass two (header)."""
    m = int(x['m']); key = dict(day=day, m=m); r = dict(key, hour=int(x['hour']), ws=float(x['ws'])); tm = {}; t = time.time()
    C.require(abs(opens[m][0] - float(x['ws'])) < 1e-6, f'{day} {m}: the registry opening differs from the manifest')
    t_par = min(int(v) for v in x['parent_worlds']); aw = allw[(m, t_par)]; w = dict(m=m, t=aw['t'], t_ms=t_par, o=aw['o'])
    world5, st5 = L.world_bind(day, w, prs, fair, depth, feat_fn)
    r['legacy'] = legacy_check(day, w, world5, feat_fn) if world5 is not None else dict(m=m, t_ms=t_par, status=f'world:{st5}'); world5 = None; tm['legacy'] = time.time() - t
    W_ = prs.get(m)
    if W_ is None:
        return dict(r, status='missing_market_tape', timing=tm), {}, []
    C.require(abs(float(W_[0]['ws']) - float(x['ws'])) < 1e-6, f'{day} {m}: the tape window start differs from the registry opening')
    t = time.time(); path, st, info = market_path(W_, float(x['ws']), opens[m][1]); r['path_info'] = info; tm['path'] = time.time() - t
    if path is None:
        return dict(r, status=f'incomplete:{st}', timing=tm), {}, []
    t_abs0, end = float(info['t_start']), float(info['end_rel'])
    ident = dict(day=day, instrument_id=m, start_abs_ms=t_abs0, end_abs_ms=t_abs0 + end, end_rel=end, opening_ms=float(x['ws']),
                 mark_source='QP.build_path supported-end mark', tape_id=BE.tape_id(path['frames'], path['trades'], path['mark']))
    t = time.time(); B, rec, pol, obs = run_ladder(path, key, ident); tm['run'] = time.time() - t
    t = time.time(); arr = dict(obs.arrays(), **command_arrays(B)); snaps = all_snaps(pol, rec)
    rows = point_rows(W_, fair.get(m), depth, feat_fn, t_abs0, float(x['ws']), rec, snaps)
    rec['queue'] = queue_summary(W_, rec, arr['commands'], t_abs0); tm['features'] = time.time() - t
    rec['snapshots'] = [{k: v for k, v in sn.items() if k != 'ctx'} for sn in snaps]
    rec['reconstruct'] = dict(checked=len(snaps), failed=sum(0 if reconstruct_ok(rec, arr['commands'], sn) else 1 for sn in snaps))
    rows = [dict(rw, day=day, m=m) for rw in rows]
    return dict(r, status='complete' if rec['status'] == 'ok' else f"ledger:{rec['status']}", ledger=rec, timing=tm), arr, rows


def day_paths(day, pilot, out=None):
    sub = Path(out or OUT) / ('pilot' if pilot else 'days')
    return {k: sub / f'{day}{s}' for k, s in (('rec', '.json'), ('gz', '.jsonl.gz'), ('npz', '_arrays.npz'), ('feat', '_features.parquet'), ('status', '_status.json'))}


def load_day(day, members, man):
    """The identity-checked inputs of one day for the members (pf_entry_exposure.run_part's checks) -> (pairs, fair, depth, feat_fn, parent
    worlds, registry)."""
    inp = man['inputs'][day]; kp = Path(inp['kp_dir']); src = KP.Tape(); ff = EX.files_fn(inp['set'])
    C.require(LS.identity_now(day, kp, src, ff, LS.hours_tracked) == inp['inputs'], f'{day}: the inputs differ from the manifest\'s')
    opens, rsha, msha = registry(day)
    C.require(rsha == man['registry'][day]['registry_sha256'] and msha == man['registry'][day]['metadata_receipt_sha256'], f'{day}: the registry differs from the manifest')
    keys = [(int(x['m']), min(int(v) for v in x['parent_worlds'])) for x in members]
    allw = {(int(w['m']), int(w['t_ms'])): w for w in LS.worlds_all(day, kp, keys=keys)}
    C.require(set(allw) == set(keys), f'{day}: a parent world is not one of the day\'s worlds')
    mk = sorted({int(x['m']) for x in members}); fair, depth, hrs, trk = X.real_inputs(day, mk)
    C.require(LS.day_identity(day, kp, src, hrs, trk, ff) == inp['inputs'], f"{day}: real_inputs' hours / tracked differ from the manifest")
    data = src.load(day, mk, src.offset(day)); prs = {int(W_[0]['m']): W_ for W_ in src.pairs(data)}; data = None
    return prs, fair, depth, X.real_features, allw, opens


def projection_allows():
    p = REPO / RECEIPT / 'logs' / 'pilot_projection.json'
    return p.exists() and json.loads(p.read_text()).get('run_rest') is True


def run_day(day, pilot=False, binder=None, man_path=None, require_frozen=True, out=None):
    """One date's partition (pilot: its PILOT_PER_DAY markets) -> its record and outputs, write-once and source-bound (header)."""
    C.require(day in DAYS, f'{day} is not a declared day')
    if require_frozen:
        C.require(frozen(), 'the runner / client / selftest / declaration / manifest / launcher are not the committed ones: no market runs before the freeze')
        C.require(pilot or projection_allows(), 'the full run needs the write-once pilot projection with run_rest true')
    mp = Path(man_path or REPO / RECEIPT / 'manifest.json'); man = json.loads(mp.read_text()); msha = C.sha(mp)
    members = [x for x in man['selected'] if x['day'] == day]
    if pilot:
        members = [x for x in members if int(x['m']) in {int(m) for m in man['pilot'][day]}]
    P = day_paths(day, pilot, out)
    if P['rec'].exists():
        rec = json.loads(P['rec'].read_text()); C.require(rec['manifest_sha256'] == msha and rec['sources'] == sources(), f'{day}: an existing record of another identity')
        return 'DONE'
    P['rec'].parent.mkdir(parents=True, exist_ok=True); t0 = time.time(); tim = dict(load=None, markets=[])
    G = LS.Guard(1200. + 900. * max(1, len(members)), 900.); mon = LS.RN.Sampler(P['status'], LS.GUARD); mon.start()
    try:
        prs, fair, depth, feat_fn, allw, opens = (binder or (lambda: load_day(day, members, man)))(); tim['load'] = time.time() - t0; G.check('load')
        recs, arrs, feats = [], {}, []
        for x in members:
            tm = time.time(); r, arr, rows = run_market(day, x, prs, fair, depth, feat_fn, allw, opens); recs.append(r); feats += rows
            arrs.update({f"{int(x['m'])}__{k}": v for k, v in arr.items()})
            tim['markets'].append(dict(m=int(x['m']), s=time.time() - tm, status=r['status'], parts=r.get('timing'))); G.check('market', time.time() - tm)
        b = LS.gz_bytes(recs); P['gz'].write_bytes(b); np.savez_compressed(P['npz'], **arrs)
        pd.DataFrame(feats if feats else [dict(kind='none')]).to_parquet(P['feat'], index=False)
        LS.RN.atomic_json(P['rec'], dict(version=VERSION, day=day, pilot=bool(pilot), members=[int(x['m']) for x in members], records=len(recs),
                                         sha256=dict(gz=hashlib.sha256(b).hexdigest(), npz=C.sha(P['npz']), feat=C.sha(P['feat'])), manifest_sha256=msha,
                                         sources=sources(), timing=tim, elapsed_s=time.time() - t0, hwm_mib=LS.hwm(), asof=C.stamp()))
        return 'DAY_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(P['status'], dict(state='STOP_RESOURCE', day=day, latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


# ---------------------------------------------------------------------------------------------------------------------- measurement
def load_outputs(pilot=False, man_path=None, out=None):
    """Every day's record verified (output hashes, manifest, sources, members) -> (manifest, market records, arrays by key, feature rows)."""
    mp = Path(man_path or REPO / RECEIPT / 'manifest.json'); man = json.loads(mp.read_text()); msha = C.sha(mp); recs, arrs, feats = [], {}, []
    for day in DAYS:
        P = day_paths(day, pilot, out)
        if pilot and not P['rec'].exists():
            continue
        rec = json.loads(P['rec'].read_text()); C.require(rec['manifest_sha256'] == msha and rec['sources'] == sources(), f'{day}: a record of another identity')
        b = P['gz'].read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256']['gz'] and C.sha(P['npz']) == rec['sha256']['npz'] and C.sha(P['feat']) == rec['sha256']['feat'],
                  f'{day}: an output differs from its record')
        rr = LS.gz_rows(b); C.require([int(r['m']) for r in rr] == [int(m) for m in rec['members']], f'{day}: the records are not the members')
        recs += rr; z = np.load(P['npz']); arrs.update({f"{day}|{k}": z[k] for k in z.files}); f = pd.read_parquet(P['feat']); feats.append(f[f.kind != 'none'])
    return man, recs, arrs, (pd.concat(feats, ignore_index=True) if feats else pd.DataFrame())


def sgn(s):
    return 1. if int(s) == 0 else -1.


def arr_of(arrs, r, name):
    return arrs[f"{r['day']}|{int(r['m'])}__{name}"]


def gen_rows(r, arrs):
    """The generation table of one complete market (header)."""
    L_ = r['ledger']; mark, end = float(L_['mark']), float(L_['end']); t0, ws = float(r['path_info']['t_start']), float(r['ws']); fb = {}
    for f, mm in zip(L_['fills'], L_['fill_mids']):
        fb.setdefault(int(f[1]), []).append((float(f[0]), int(f[5]), float(f[3]), float(f[4]), mm))
    by_t = {}
    for f in L_['fills']:
        by_t.setdefault((int(f[2]), float(f[0])), set()).add(int(f[1]))
    struct = {int(x['gid']): x for x in L_['struct_log']}; cm = arr_of(arrs, r, 'commands'); first_cmd = {}
    for c in cm:
        if int(c[2]) == CMD_KINDS.index('live') and int(c[3]) >= 0 and int(c[3]) not in first_cmd:
            first_cmd[int(c[3])] = CMD_ACTIONS[int(c[7])]
    rows = []
    for o in L_['orders']:
        gid, s = int(o['gid']), int(o['side']); sg = sgn(s); fl = sorted(fb.get(gid, []), key=lambda x: (x[0], x[1])); size = float(o['size'])
        filled = sum(x[2] for x in fl); cum = 0.; full = None
        for t, i, q, p, mm in fl:
            cum += q
            if full is None and cum >= size - LB.TOL:
                full = t
        val = sum(sg * q * (mark - p) for t, i, q, p, mm in fl); av = [(t, q, p, mm[0]) for t, i, q, p, mm in fl if mm[0] is not None]
        ev = sorted({t for t, *_ in fl}); first_t = ev[0] if ev else None; e_g = math.ceil(float(o['act']) / STEP) * STEP + CANCEL
        cl = L_['closed'].get(str(gid)); req, eff = o['req'], float(o['remove']); qs = L_.get('queue', {}).get(str(gid), {})
        requested = req is not None; reason = None if not requested else ('inventory' if o['why'] == 'policy' else (struct.get(gid, {}).get('primary') or str(o['why'])))
        eff_removal = bool(requested and eff <= end and (full is None or full >= eff)); full_before = bool(requested and full is not None and full < eff)
        pend_end = bool(requested and (eff > end or eff + BE.CANCEL_ACK > end) and not full_before)
        fb_t = qs.get('first_best_t'); sweep = any(len(by_t.get((s, t), ())) >= 2 for t in ev)
        vend = min(eff, full if full is not None else np.inf, end)
        rows.append(dict(day=r['day'], m=int(r['m']), gid=gid, side=s, rung=int(round(float(o['rung']))), size=size, price=float(o['price']), submit=float(o['submit']),
                         act=float(o['act']), refused=bool(o['refused']), minute=int((t0 + float(o['submit']) - ws) // MINUTE), filled=filled, fill_fraction=filled / size,
                         any_fill=bool(filled > LB.TOL), n_fill_rows=len(fl), n_events=len(ev), first_fill_after_act=None if first_t is None else first_t - float(o['act']),
                         full_fill_after_act=None if full is None else full - float(o['act']), censored_ms=None if full is not None or o['refused'] else vend - float(o['act']),
                         value=val, value_first_event=sum(sg * q * (mark - p) for t, i, q, p, mm in fl if t == first_t),
                         value_repeat_events=sum(sg * q * (mark - p) for t, i, q, p, mm in fl if t != first_t),
                         edge=sum(sg * q * (m0 - p) for t, q, p, m0 in av), move=sum(sg * q * (mark - m0) for t, q, p, m0 in av), n_mid_missing=len(fl) - len(av),
                         earliest_cancel_effect=e_g, value_locked=sum(sg * q * (mark - p) for t, i, q, p, mm in fl if t < e_g),
                         value_later=sum(sg * q * (mark - p) for t, i, q, p, mm in fl if t >= e_g), lifetime=0. if o['refused'] else max(0., vend - float(o['act'])),
                         requested=requested, request_reason=reason, effective_removal=eff_removal, full_fill_before_removal=full_before, pending_at_end=pend_end,
                         reviewed=gid in first_cmd, first_review_pull=first_cmd.get(gid) == 'PULL', survivor_end=bool((not o['refused']) and eff > end and filled < size - LB.TOL),
                         closed_why=cl[0] if cl else None, closed_t=float(cl[1]) if cl else None, ahead0=float(L_['ahead0'].get(str(gid), np.nan)),
                         queue_n=qs.get('n', 0), queue_front=qs.get('front', 0), queue_unknown=qs.get('unknown', 0), queue_first=qs.get('first'), queue_last=qs.get('last'),
                         became_best=fb_t is not None, became_best_t=fb_t, became_best_queue=qs.get('first_best_queue'),
                         value_after_best=sum(sg * q * (mark - p) for t, i, q, p, mm in fl if fb_t is not None and t >= fb_t), sweep=sweep,
                         severe_abs=bool(val < -SEVERE_ABS), severe_per_share=bool(val / size < -SEVERE_PER_SHARE),
                         cls='nofill' if filled <= LB.TOL else ('gain' if val > LB.TOL else 'zero' if val >= -LB.TOL else 'severe' if val < -SEVERE_ABS else 'ordinary')))
    return rows


def fill_rows(r):
    """The fill table of one complete market (header): edge, movement and markout per horizon with their availability, END value."""
    L_ = r['ledger']; mark = float(L_['mark']); t0, ws = float(r['path_info']['t_start']), float(r['ws'])
    od = {int(o['gid']): o for o in L_['orders']}; rows = []; by_t = {}
    for f in L_['fills']:
        by_t.setdefault((int(f[2]), float(f[0])), set()).add(int(f[1]))
    for f, mm in zip(L_['fills'], L_['fill_mids']):
        t, gid, s, q, p, i = float(f[0]), int(f[1]), int(f[2]), float(f[3]), float(f[4]), int(f[5]); sg = sgn(s); m0, a0 = mm[0], mm[1]
        x = dict(day=r['day'], m=int(r['m']), gid=gid, side=s, rung=int(round(float(od[gid]['rung']))), t=t, q=q, p=p, trade=i, minute=int((t0 + t - ws) // MINUTE),
                 age=t - float(od[gid]['act']), mid0=m0, mid0_age=a0, value=sg * q * (mark - p), edge=None if m0 is None else sg * q * (m0 - p),
                 move_end=None if m0 is None else sg * q * (mark - m0), sweep=len(by_t[(s, t)]) >= 2)
        for j, h in enumerate(MARKOUTS):
            mh, ah = mm[2 + 2 * j], mm[3 + 2 * j]; c = f'{int(h)}'
            x[f'mo{c}'] = None if mh is None else sg * q * (mh - p); x[f'mv{c}'] = None if (mh is None or m0 is None) else sg * q * (mh - m0); x[f'age{c}'] = ah
        rows.append(x)
    return rows


def minute_rows(r):
    """Per interval between the valid boundary marks (the start, the market minutes, END): the equity increment E(T_b) - E(T_a) = the carry of
    the inventory held at T_a + the interval's new fills, at the interval's end mark; telescoping to the final value (header)."""
    L_ = r['ledger']; end, mark = float(L_['end']), float(L_['mark']); bm = r['path_info'].get('boundary_marks', {})
    bounds = [(0., None, 'start')] + [(float(v['t_rel']), v['mark'], v['status']) for k, v in sorted(bm.items(), key=lambda kv: int(kv[0]))] + [(end, mark, 'end')]
    ok = [b for b in bounds if b[2] == 'start' or b[1] is not None]; fills = sorted(L_['fills'], key=lambda f: (float(f[0]), int(f[5]))); rows = []
    pc = lambda T_: (sum(sgn(f[2]) * float(f[3]) for f in fills if float(f[0]) <= T_), -sum(sgn(f[2]) * float(f[3]) * float(f[4]) for f in fills if float(f[0]) <= T_))
    for (ta, ma, _), (tb, mb, _) in zip(ok[:-1], ok[1:]):
        (pa, ca), (pb, cb) = pc(ta), pc(tb); va = 0. if ma is None else ca + pa * ma; vb = cb + pb * mb
        carry = 0. if ma is None else pa * (mb - ma); new = sum(sgn(f[2]) * float(f[3]) * (mb - float(f[4])) for f in fills if ta < float(f[0]) <= tb)
        rows.append(dict(day=r['day'], m=int(r['m']), t_from=ta, t_to=tb, minute_from=int((float(r['path_info']['t_start']) + ta - float(r['ws'])) // MINUTE),
                         dv=vb - va, carry=carry, new=new, pos_from=pa, pos_to=pb, n_fills=sum(1 for f in fills if ta < float(f[0]) <= tb)))
    return rows, [b[2] for b in bounds if b[1] is None and b[2] != 'start']


def equity_row(r, arrs):
    """The equity path at the venue mid of each tick (cash + engine inventory x mid; gaps where no two-sided frame) and its maximum drawdown."""
    cash, pos, mid = arr_of(arrs, r, 'cash'), arr_of(arrs, r, 'pose'), arr_of(arrs, r, 'mid'); E_ = cash + pos * mid; ok = np.isfinite(E_)
    dd = 0.; peak = -np.inf
    for e in E_[ok]:
        peak = max(peak, e); dd = max(dd, peak - e)
    return dict(day=r['day'], m=int(r['m']), ticks=int(len(E_)), gaps=int((~ok).sum()), max_drawdown=float(dd), equity_last_mid=float(E_[ok][-1]) if ok.any() else None,
                value=float(r['ledger']['value']))


def downtime_of(flags):
    """The exclusive downtime reason: the first of DOWNTIME (the declared precedence) among the applicable flags, else other."""
    return next((d for d in DOWNTIME if d in flags), 'other')


def tick_reasons(r, arrs):
    """Per tick and side when no order is venue-active: (the exclusive DOWNTIME reason by precedence, the set of every applicable flag)."""
    nva, npp, nlive, ncp = (arr_of(arrs, r, k) for k in ('nva', 'npp', 'nlive', 'ncp')); nt = nva.shape[0]
    fl = [[set(), set()] for _ in range(nt)]
    for c in arr_of(arrs, r, 'commands'):
        k, s, why = int(round(c[0] / STEP)), int(c[1]), CMD_REASONS[int(c[8])]
        if 0 <= k < nt and why in ('fallback', 'restricted', 'own_cross_blocked', 'near_end', 'cap', 'no_room', 'exposure_full', 'zero_desired', 'unavailable_price'):
            fl[k][s].add('invalid_book' if why == 'fallback' else why)
    out = np.full((nt, 2), 'quoting', object); flags = {}
    for k in range(nt):
        for s in (0, 1):
            if nva[k, s] > 0:
                continue
            f = set(fl[k][s])
            if npp[k, s] > 0:
                f.add('activation_latency')
            if nlive[k, s] + ncp[k, s] > 0:
                f.add('awaiting_notice')
            out[k, s] = downtime_of(f)
            for d in f:
                flags[(s, d)] = flags.get((s, d), 0) + 1
    return out, flags


def part_rows(r, arrs):
    """Participation per market minute and side (and joint) on the full window and on valid-book time, from the tick arrays."""
    nva, bc, bv, bo, npr = (arr_of(arrs, r, k) for k in ('nva', 'bestc', 'bestv', 'bookok', 'nprice')); nt = nva.shape[0]; va = nva > 0
    t0, ws = float(r['path_info']['t_start']), float(r['ws']); mins = ((t0 + np.arange(nt) * STEP - ws) // MINUTE).astype(int); why, _ = tick_reasons(r, arrs); rows = []
    for mi in sorted(set(mins.tolist())):
        sel = mins == mi; n = int(sel.sum()); vb = sel & (bo[:, 0] > 0) & (bo[:, 1] > 0)
        x = dict(day=r['day'], m=int(r['m']), minute=int(mi), ticks=n, valid_book_ticks=int(vb.sum()), both=float((va[sel, 0] & va[sel, 1]).mean()),
                 either=float((va[sel, 0] | va[sel, 1]).mean()), both_at_best=float(((bv[sel, 0] > 0) & (bv[sel, 1] > 0)).mean()),
                 both_valid=float((va[vb, 0] & va[vb, 1]).mean()) if vb.any() else None)
        for s, nm in ((0, 'bid'), (1, 'ask')):
            x.update({f'{nm}_active': float(va[sel, s].mean()), f'{nm}_best_client': float((bc[sel, s] > 0).mean()), f'{nm}_best_venue': float((bv[sel, s] > 0).mean()),
                      f'{nm}_levels': float(nva[sel, s].mean()), f'{nm}_prices': float(npr[sel, s].mean()), f'{nm}_active_valid': float(va[vb, s].mean()) if vb.any() else None})
            for d in DOWNTIME:
                x[f'{nm}_{d}'] = float((why[sel, s] == d).mean())
        rows.append(x)
    return rows


def inventory_row(r, arrs):
    """Inventory, mode, recovery and reversal readings of one complete market (header)."""
    L_ = r['ledger']; pe, pc, md, res = arr_of(arrs, r, 'pose'), arr_of(arrs, r, 'posc'), arr_of(arrs, r, 'mode'), arr_of(arrs, r, 'res')
    end, mark = float(L_['end']), float(L_['mark']); S, R = LB.CONFIGS[LB.LADDER]['inventory']['S'], LB.CONFIGS[LB.LADDER]['inventory']['R']
    eps, cur, t_in = [], None, None
    for g, I, old, new in L_['mode_log']:
        if cur is not None and new != cur:
            eps.append(dict(mode=cur, start=t_in, end=float(g), censored=False, exit=new)); cur, t_in = None, None
        if new != 'UNRESTRICTED':
            cur, t_in = new, float(g)
    if cur is not None:
        eps.append(dict(mode=cur, start=t_in, end=end, censored=True, exit=None))
    pos = 0.; rev = []
    for f in sorted(L_['fills'], key=lambda f: (float(f[0]), int(f[5]))):
        s, q, p = int(f[2]), float(f[3]), float(f[4]); new = pos + sgn(s) * q
        if abs(pos) > LB.TOL and abs(new) > LB.TOL and np.sign(pos) != np.sign(new):
            rev.append(dict(t=float(f[0]), before=pos, after=new, over=abs(new), value_over=sgn(s) * abs(new) * (mark - p)))
        pos = new
    n = max(1, len(pe))
    return dict(day=r['day'], m=int(r['m']), peak_abs=float(np.max(np.abs(pe))) if len(pe) else 0., mean_abs=float(np.mean(np.abs(pe))) if len(pe) else 0.,
                share_abs_ge_S=float(np.mean(np.abs(pe) >= S - LB.TOL)) if len(pe) else 0., share_abs_ge_S_client=float(np.mean(np.abs(pc) >= S - LB.TOL)) if len(pc) else 0.,
                discrepancy_mean=float(np.mean(np.abs(pe - pc))) if len(pe) else 0., discrepancy_max=float(np.max(np.abs(pe - pc))) if len(pe) else 0.,
                share_long_mode=float(np.mean(md == 1)), share_short_mode=float(np.mean(md == 2)), transitions=len(L_['mode_log']), episodes=len(eps),
                episodes_censored=sum(e['censored'] for e in eps), recovery_ms=[e['end'] - e['start'] for e in eps if not e['censored']],
                reversals=len(rev), reversal_value=float(sum(x['value_over'] for x in rev)), terminal_engine=float(L_['position']), terminal_client=float(pc[-1]) if len(pc) else 0.,
                terminal_abs_gt_R=bool(abs(float(L_['position'])) > R + LB.TOL), value=float(L_['value']), cash=float(L_['cash']),
                mean_bid_reserved=float(res[:, 0].mean()), mean_ask_reserved=float(res[:, 1].mean()), ticks=int(n))


def replacement_rows(r, gens):
    """Per removed generation (closed by an acknowledged removal) the delay to the side's next submission and the queue estimate change."""
    rows = []
    for s in (0, 1):
        gs = sorted([g for g in gens if g['side'] == s], key=lambda g: g['submit'])
        for g in gs:
            if g['closed_why'] != 'removed':
                continue
            nxt = next((h for h in gs if h['submit'] >= g['closed_t']), None)
            rows.append(dict(day=r['day'], m=int(r['m']), side=s, gid=g['gid'], reason=g['request_reason'], closed_t=g['closed_t'], next_gid=None if nxt is None else nxt['gid'],
                             delay_ms=None if nxt is None else nxt['submit'] - g['closed_t'], queue_last=g['queue_last'], next_queue_first=None if nxt is None else nxt['queue_first']))
    return rows


def remaining_value(fills, gid, d, mark):
    """A live snapshot's avoidable exposure: the generation's fills at or after the modeled cancel effect d + 37 ms, at the END mark."""
    return sum(sgn(f[2]) * float(f[3]) * (mark - float(f[4])) for f in fills if int(f[1]) == int(gid) and float(f[0]) >= float(d) + CANCEL)


def edges_of(x):
    """Quintile edges of the finite nonzero values (the four interior edges), or [] with fewer than 5 distinct values."""
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x != 0.)]
    return [] if len(np.unique(x)) < 5 else [float(e) for e in np.quantile(x, [.2, .4, .6, .8])]


def bin_of(x, e):
    if x is None or not np.isfinite(x):
        return 'missing'
    if x == 0.:
        return 'zero'
    return 'nonzero' if not e else f'q{1 + int(np.searchsorted(np.asarray(e), x, side="right"))}'


def state_rows(F, gens, recs):
    """The decision-state rows: every snapshot with its outcome (an entry: the generation's END value; a live state: its avoidable exposure)."""
    if not len(F):
        return pd.DataFrame()
    G = {(r['day'], r['m'], r['gid']): r for r in gens}; byk = {(r['day'], int(r['m'])): r['ledger'] for r in recs if r['status'] == 'complete'}; out = []
    for x in F.to_dict('records'):
        Lg = byk.get((x['day'], int(x['m'])))
        if Lg is None or int(x['gid']) < 0:
            continue
        g = G[(x['day'], int(x['m']), int(x['gid']))]; y = g['value'] if x['kind'] == 'entry' else remaining_value(Lg['fills'], x['gid'], x['t'], Lg['mark'])
        row = dict(day=x['day'], m=int(x['m']), kind=x['kind'], side=int(x['side']), gid=int(x['gid']), rung=int(round(x['rung'])), t=float(x['t']), status=x['status'],
                   y=float(y), size=float(g['size']), filled=float(g['filled']), n_eligible=int(x['n_eligible']), severe_abs=bool(y < -SEVERE_ABS),
                   severe_per_share=bool(y / float(g['size']) < -SEVERE_PER_SHARE))
        for c in DICT_COLS:
            v = x.get('ctx_inventory_shares') if c == 'inv_signed' else x.get(c)
            row[c] = None if v is None or not np.isfinite(v) else (sgn(x['side']) * float(v) if c == 'inv_signed' else float(v))
        out.append(row)
    return pd.DataFrame(out)


def state_map(S_):
    """Per kind, side and dictionary input the Oct 24-25 bins and, per day and bin, the support and the outcome sums."""
    if not len(S_):
        return pd.DataFrame(), {}
    edges, rows = {}, []
    for (kind, s), g in S_.groupby(['kind', 'side']):
        dsc = g[g.day.isin(DISCOVERY) & (g.status == 'ok')]
        for c in DICT_COLS:
            e = edges_of(dsc[c].astype(float).values); edges[f'{kind}|{s}|{c}'] = e; b = g[c].astype(float).map(lambda v: bin_of(v, e))
            for (day, bn), h in g.assign(_b=b).groupby(['day', '_b']):
                y = h.y.astype(float)
                rows.append(dict(kind=kind, side=int(s), feature=c, day=day, bin=bn, n=int(len(h)), generations=int(len(set(zip(h.m, h.gid)))), markets=int(h.m.nunique()),
                                 y_sum=float(y.sum()), y_pos=float(y[y > 0].sum()), y_neg=float(y[y < 0].sum()), n_zero=int((y.abs() <= LB.TOL).sum()),
                                 severe_abs=int(h.severe_abs.sum()), severe_per_share=int(h.severe_per_share.sum()), fill_rate=float((h.filled > LB.TOL).mean())))
    return pd.DataFrame(rows), edges


def severe_timelines(recs, G_, n=5):
    """The n worst generations of each day: observations, commands, cancel effect, fills and the inventory path (header)."""
    out = []
    byk = {(r['day'], int(r['m'])): r for r in recs if r['status'] == 'complete'}
    for day in DAYS:
        g = G_[(G_.day == day) & (G_.cls == 'severe')].sort_values('value').head(n) if len(G_) else G_
        for x in g.to_dict('records'):
            r = byk[(x['day'], int(x['m']))]; L_ = r['ledger']; fl = sorted([f for f in L_['fills'] if int(f[1]) == int(x['gid'])], key=lambda f: float(f[0]))
            pos_at = lambda T_: sum(sgn(f[2]) * float(f[3]) for f in L_['fills'] if float(f[0]) <= T_)
            out.append(dict(day=day, m=int(x['m']), gid=int(x['gid']), side=int(x['side']), rung=int(x['rung']), size=x['size'], value=x['value'], submit=x['submit'],
                            act=x['act'], earliest_cancel_effect=x['earliest_cancel_effect'], request_reason=x['request_reason'], value_locked=x['value_locked'],
                            value_later=x['value_later'], fills=[dict(t=float(f[0]), q=float(f[3]), p=float(f[4]), notice=float(f[0]) + NOTICE, pos_after=pos_at(float(f[0])))
                                                                 for f in fl], position_end=float(L_['position'])))
    return out


def top5_share(x):
    x = np.asarray(x, float); neg = -x[x < 0]
    return None if not len(neg) or neg.sum() <= 0 else float(np.sort(neg)[::-1][:5].sum() / neg.sum())


def summaries(M_, G_, Fi, I_, P_, E_, R_):
    """Per day x side (and joint 'both') x birth rung ('all' / 0 / 1): the dashboard (header, metrics.md)."""
    rows = []
    for day in DAYS + ('all',):
        mk = M_[(M_.status == 'complete') & ((M_.day == day) if day != 'all' else True)]; keys = set(zip(mk.day, mk.m))
        sel = lambda D: D[[(d, m) in keys for d, m in zip(D.day, D.m)]] if len(D) else D
        g_, f_, i_, p_, e_, r_ = sel(G_), sel(Fi), sel(I_), sel(P_), sel(E_), sel(R_)
        for side in ('bid', 'ask', 'both'):
            for rung in ('all', 0, 1):
                gs = g_ if side == 'both' else g_[g_.side == (0 if side == 'bid' else 1)]; fs = f_ if side == 'both' else f_[f_.side == (0 if side == 'bid' else 1)]
                if rung != 'all':
                    gs = gs[gs.rung == rung] if len(gs) else gs; fs = fs[fs.rung == rung] if len(fs) else fs
                val = float(fs.value.sum()) if len(fs) else 0.; sub = float(gs['size'].sum()) if len(gs) else 0.; fq = float(fs.q.sum()) if len(fs) else 0.
                acc = gs[~gs.refused] if len(gs) else gs
                x = dict(day=day, side=side, rung=rung, markets=int(len(mk)), generations=int(len(gs)), accepted=int(len(acc)), refused=int(gs.refused.sum()) if len(gs) else 0,
                         fills=int(len(fs)), filled_shares=fq, submitted_shares=sub, accepted_shares=float(acc['size'].sum()) if len(acc) else 0., value=val,
                         value_per_market=val / len(mk) if len(mk) else None, value_per_submitted=val / sub if sub else None, value_per_filled=val / fq if fq else None,
                         any_fill_per_submitted=float(gs.any_fill.mean()) if len(gs) else None, any_fill_per_accepted=float(acc.any_fill.mean()) if len(acc) else None,
                         quantity_fill_fraction=float(gs.filled.sum() / sub) if sub else None, mean_size=float(gs['size'].mean()) if len(gs) else None)
                av = fs[fs.edge.notna()] if len(fs) else fs
                x.update(edge=float(av.edge.sum()) if len(av) else 0., move_end=float(av.move_end.sum()) if len(av) else 0., fills_mid=int(len(av)), q_mid=float(av.q.sum()) if len(av) else 0.,
                         fills_mid_missing=int(len(fs) - len(av)))
                for h in MARKOUTS:
                    c = f'{int(h)}'; a_ = fs[fs[f'mo{c}'].notna()] if len(fs) else fs
                    x.update({f'mo{c}': float(a_[f'mo{c}'].sum()) if len(a_) else 0., f'mv{c}': float(a_[f'mv{c}'].fillna(0).sum()) if len(a_) else 0., f'mo{c}_fills': int(len(a_)),
                              f'mo{c}_q': float(a_.q.sum()) if len(a_) else 0., f'mo{c}_missing': int(len(fs) - len(a_)),
                              f'mo{c}_age_median': float(a_[f'age{c}'].median()) if len(a_) else None})
                cmn = fs.dropna(subset=['edge'] + [f'mo{int(h)}' for h in MARKOUTS]) if len(fs) else fs
                x.update(common_fills=int(len(cmn)), common_q=float(cmn.q.sum()) if len(cmn) else 0., common_value=float(cmn.value.sum()) if len(cmn) else 0.,
                         **{f'common_mo{int(h)}': float(cmn[f'mo{int(h)}'].sum()) if len(cmn) else 0. for h in MARKOUTS}, common_edge=float(cmn.edge.sum()) if len(cmn) else 0.)
                for cl in ('nofill', 'gain', 'zero', 'ordinary', 'severe'):
                    h_ = gs[gs.cls == cl] if len(gs) else gs; x[f'n_{cl}'] = int(len(h_)); x[f'v_{cl}'] = float(h_.value.sum()) if len(h_) else 0.
                fg = gs[gs.filled > LB.TOL] if len(gs) else gs; rv = gs[gs.reviewed] if len(gs) else gs
                x.update(n_severe_per_share=int(gs.severe_per_share.sum()) if len(gs) else 0, v_severe_per_share=float(gs[gs.severe_per_share].value.sum()) if len(gs) else 0.,
                         value_without_severe_expost=val - (float(gs[gs.cls == 'severe'].value.sum()) if len(gs) else 0.),
                         top5_generation_loss_share=top5_share(gs.value) if len(gs) else None, worst_generation=float(gs.value.min()) if len(gs) else None,
                         value_locked=float(gs.value_locked.sum()) if len(gs) else 0., value_later=float(gs.value_later.sum()) if len(gs) else 0.,
                         severe_locked=float(gs[gs.cls == 'severe'].value_locked.sum()) if len(gs) else 0., severe_value=float(gs[gs.cls == 'severe'].value.sum()) if len(gs) else 0.,
                         filled_generations=int(len(fg)), repeat_generations=int((fg.n_events >= 2).sum()) if len(fg) else 0, mean_events=float(fg.n_events.mean()) if len(fg) else None,
                         value_repeat_events=float(fg.value_repeat_events.sum()) if len(fg) else 0., value_first_events=float(fg.value_first_event.sum()) if len(fg) else 0.,
                         full_fills=int(fg.full_fill_after_act.notna().sum()) if len(fg) else 0, median_first_fill_ms=float(fg.first_fill_after_act.median()) if len(fg) else None,
                         median_full_fill_ms=float(fg.full_fill_after_act.median()) if len(fg) and fg.full_fill_after_act.notna().any() else None,
                         censored=int(gs.censored_ms.notna().sum()) if len(gs) else 0, became_best=int(gs.became_best.sum()) if len(gs) else 0,
                         value_after_best=float(gs.value_after_best.sum()) if len(gs) else 0., sweep_generations=int(gs.sweep.sum()) if len(gs) else 0,
                         sweep_value=float(gs[gs.sweep].value.sum()) if len(gs) else 0., reviewed=int(len(rv)), first_review_pulls=int(rv.first_review_pull.sum()) if len(rv) else 0,
                         requests=int(gs.requested.sum()) if len(gs) else 0, effective_removals=int(gs.effective_removal.sum()) if len(gs) else 0,
                         full_fill_before_removal=int(gs.full_fill_before_removal.sum()) if len(gs) else 0, pending_at_end=int(gs.pending_at_end.sum()) if len(gs) else 0,
                         survivors_end=int(gs.survivor_end.sum()) if len(gs) else 0, queue_front_share=float(gs.queue_front.sum() / gs.queue_n.sum()) if len(gs) and gs.queue_n.sum() else None)
                for rs in ('inventory', 'no_book', 'distance', 'cross'):
                    x[f'requests_{rs}'] = int((gs.request_reason == rs).sum()) if len(gs) else 0
                if rung == 'all' and len(p_):
                    w = p_.ticks.astype(float)
                    cols = [f'{side}_active', f'{side}_best_client', f'{side}_best_venue', f'{side}_levels', f'{side}_prices'] if side != 'both' else ['both', 'either', 'both_at_best']
                    for c in cols:
                        x[c] = float((p_[c] * w).sum() / w.sum())
                    if side != 'both':
                        for d in DOWNTIME:
                            x[f'down_{d}'] = float((p_[f'{side}_{d}'] * w).sum() / w.sum())
                if side == 'both' and rung == 'all' and len(i_):
                    x.update(losing_markets=int((i_.value < 0).sum()), top5_market_loss_share=top5_share(i_.value), worst_market=float(i_.value.min()), peak_abs_mean=float(i_.peak_abs.mean()),
                             mean_abs_inventory=float(i_.mean_abs.mean()), share_abs_ge_S=float(i_.share_abs_ge_S.mean()), discrepancy_mean=float(i_.discrepancy_mean.mean()),
                             restricted_markets=int((i_.transitions > 0).sum()), transitions=int(i_.transitions.sum()), reversals=int(i_.reversals.sum()),
                             reversal_value=float(i_.reversal_value.sum()), terminal_abs_mean=float(i_.terminal_engine.abs().mean()), terminal_gt_R=int(i_.terminal_abs_gt_R.sum()),
                             recovery_median_ms=float(np.median([v for l_ in i_.recovery_ms for v in l_])) if any(len(l_) for l_ in i_.recovery_ms) else None,
                             episodes=int(i_.episodes.sum()), episodes_censored=int(i_.episodes_censored.sum()), mean_bid_reserved=float(i_.mean_bid_reserved.mean()),
                             mean_ask_reserved=float(i_.mean_ask_reserved.mean()), max_drawdown_mean=float(e_.max_drawdown.mean()) if len(e_) else None,
                             replacement_delay_median_ms=float(r_.delay_ms.dropna().median()) if len(r_) and r_.delay_ms.notna().any() else None)
                rows.append(x)
    return pd.DataFrame(rows)


def predicates(Sm, Mi):
    """The declared readings (header): the headline, value signs, repeated fills, the carry / new-fill split and the review's five patterns."""
    S0 = Sm[Sm.rung.astype(str) == 'all']; both = S0[S0.side == 'both'].set_index('day'); days = [d for d in DAYS if d in both.index and both.loc[d, 'markets'] > 0]
    full = len(days) == len(DAYS); allr = both.loc['all'] if 'all' in both.index else None; out = dict(days=days, complete_days=full)
    out['headline'] = {d: dict(value=float(both.loc[d, 'value']), filled_shares=float(both.loc[d, 'filled_shares']), both_active=both.loc[d].get('both'),
                               severe_value=float(both.loc[d, 'v_severe'])) for d in days + (['all'] if allr is not None else [])}
    sv = [float(both.loc[d, 'value']) for d in days]
    out['strategy_value'] = dict(answer='INCOMPLETE' if not full else 'POSITIVE_EVERY_DAY' if all(v > 0 for v in sv) else 'NEGATIVE_EVERY_DAY' if all(v < 0 for v in sv) else 'MIXED',
                                 value_by_day=dict(zip(days, sv)))
    rep = int(allr['repeat_generations']) if allr is not None else 0; rv = [float(both.loc[d, 'value_repeat_events']) for d in days]
    out['repeated_fills'] = dict(answer='REPEATED_THIN' if rep < THIN or not full else 'REPEATED_EARN' if all(v > 0 for v in rv) else 'REPEATED_LOSE' if all(v < 0 for v in rv) else 'REPEATED_MIXED',
                                 repeat_generations=rep, value_by_day=dict(zip(days, rv)))
    ca = {d: float(Mi[Mi.day == d].carry.sum()) if len(Mi) else 0. for d in days}; nw = {d: float(Mi[Mi.day == d].new.sum()) if len(Mi) else 0. for d in days}
    out['loss_source'] = dict(answer='INCOMPLETE' if not full else 'CARRY_DOMINATES' if all(ca[d] < 0 and abs(ca[d]) >= abs(nw[d]) for d in days) else
                              'NEW_FILLS_DOMINATE' if all(nw[d] < 0 and abs(nw[d]) >= abs(ca[d]) for d in days) else 'MIXED', carry=ca, new=nw)
    if allr is not None:
        g = lambda k: float(allr.get(k) or 0.)
        out['patterns'] = dict(
            edge_then_adverse=bool(g('common_edge') > 0 and g('common_mo1000') < g('common_edge')),
            short_positive_later_negative=bool(g('mo100') > 0 and g('value') < 0),
            low_participation=bool(allr.get('both') is not None and float(allr['both']) < .5),
            tail_erases_ordinary=bool(g('v_gain') + g('v_ordinary') > 0 and g('value') < 0 and g('v_severe') < 0),
            deep_level_rarely_filled=None)
        r0 = S0[(S0.day == 'all') & (S0.side == 'both')]; R0 = Sm[(Sm.day == 'all') & (Sm.side == 'both') & (Sm.rung.astype(str) == '0')]
        R1 = Sm[(Sm.day == 'all') & (Sm.side == 'both') & (Sm.rung.astype(str) == '1')]
        if len(R0) and len(R1) and R0.iloc[0]['any_fill_per_accepted'] is not None and R1.iloc[0]['any_fill_per_accepted'] is not None:
            out['patterns']['deep_level_rarely_filled'] = bool(float(R1.iloc[0]['any_fill_per_accepted']) < .5 * float(R0.iloc[0]['any_fill_per_accepted']))
        out['patterns_note'] = 'investigation guides from the review, not causal verdicts; one policy identifies no ladder or inventory-control gain'
    return out


def measure(out=OUT, man_path=None):
    """The descriptive measurement (header) -> out/measure/*.csv + audit.json; write-once."""
    md = Path(out) / 'measure'; C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)'); md.mkdir(parents=True, exist_ok=True)
    man, recs, arrs, F = load_outputs(False, man_path, out); comp = [r for r in recs if r['status'] == 'complete']
    C.require(sorted((x['day'], int(x['m'])) for x in man['selected']) == sorted((r['day'], int(r['m'])) for r in recs), 'the records are not the selected markets')
    gens, fills, mins, skipped, reps = [], [], [], {}, []
    for r in comp:
        g_ = gen_rows(r, arrs); gens += g_; fills += fill_rows(r); mr, sk = minute_rows(r); mins += mr; skipped[f"{r['day']}_{r['m']}"] = sk; reps += replacement_rows(r, g_)
    parts = [p for r in comp for p in part_rows(r, arrs)]; invs = [inventory_row(r, arrs) for r in comp]; eqs = [equity_row(r, arrs) for r in comp]
    M_ = pd.DataFrame([dict(day=r['day'], m=int(r['m']), hour=r['hour'], status=r['status'], opening_wait_ms=(r.get('path_info') or {}).get('opening_wait_ms'),
                            wait_frame_received=(r.get('path_info') or {}).get('wait_steps_frame_received'), wait_no_frame=(r.get('path_info') or {}).get('wait_steps_no_frame'),
                            end_rel=(r.get('path_info') or {}).get('end_rel'), legacy_engine_all=(r.get('legacy') or {}).get('engine_all'),
                            legacy_features_bad=(r.get('legacy') or {}).get('features_bad'), legacy_features_equal=(r.get('legacy') or {}).get('features_equal'),
                            legacy_status=(r.get('legacy') or {}).get('status', 'ok'), reconstruct_checked=((r.get('ledger') or {}).get('reconstruct') or {}).get('checked'),
                            reconstruct_failed=((r.get('ledger') or {}).get('reconstruct') or {}).get('failed')) for r in recs])
    G_, Fi, Mi, P_, I_, E_, R_ = (pd.DataFrame(x) for x in (gens, fills, mins, parts, invs, eqs, reps))
    S_ = state_rows(F, gens, recs); SM, edges = state_map(S_); Sm = summaries(M_, G_, Fi, I_, P_, E_, R_); pr = predicates(Sm, Mi); tl = severe_timelines(recs, G_)
    names = ('markets', 'generations', 'fills', 'minutes', 'participation', 'inventory', 'equity', 'replacements', 'state_rows', 'state_map', 'summary')
    for nm, D in zip(names, (M_, G_, Fi, Mi, P_, I_, E_, R_, S_, SM, Sm)):
        D.to_csv(md / f'{nm}.csv', index=False)
    C.dump(md / 'severe_timelines.json', tl); st = M_.status.value_counts().to_dict()
    C.dump(md / 'audit.json', dict(version=VERSION, asof=C.stamp(), statuses=st, complete=len(comp), selected=len(recs), skipped_boundaries=skipped, edges=edges, predicates=pr,
                                   outputs={f'{nm}.csv': C.sha(md / f'{nm}.csv') for nm in names}, manifest_sha256=C.sha(Path(man_path or REPO / RECEIPT / 'manifest.json')),
                                   sources=sources()))
    print(json.dumps(dict(statuses=st, predicates=pr), indent=1, default=str)[:4000])
    return pr


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_rest_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        m_ = build_manifest(); print(json.dumps(dict(selected=len(m_['selected']), cells=len(m_['cells']), pilot=m_['pilot']), indent=1))
    elif cmd in ('pilot', 'run'):
        print(run_day(sys.argv[2], pilot=(cmd == 'pilot')))
    elif cmd == 'measure':
        measure()
    else:
        sys.exit('usage: pf_ladder_rest.py selftest | manifest | pilot <day> | run <day> | measure')
