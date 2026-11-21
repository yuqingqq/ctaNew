"""BINANCE-PRESSURE-GUARD-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review 2c5dbd53 (target 84a4f999) accepted the ask-reservation guard
as a consumed-day diagnostic (not promoted) and named ONE next priority: one external-market signal against the original default-post ladder
(LADDER_REST_INVENTORY_v1, frozen 47d2eec1) on the same 19 complete markets, all 96 statuses preserved; the second guard candidate of this
ladder line (the reservation guard is neither tuned nor stacked). A declared new hypothesis on consumed Oct 24-27 development days (the
audit's adv_bn_r250 failed its all-eight-cells lead rule); no profitability, significance, validation or promotion claim.
RULE (on_value + PressureGuard; the sha256 of their source is in the manifest): x_s = the unchanged causal builder's float32 adv_bn_r250
 (FC.MARKET[20]; pf_pipeline_export.real_features = pf_flow_price_data.feature_rows: the step row's bn_r250 times -1 for bids, +1 for asks)
 at the decision's receipt-clock grid step of side s's world (pf_ladder_rest.features_at's lookup: |tg - (t_start + g)| <= 1e-6). ON_s iff
 x_s is finite, x_s > 0 and pf_ladder_rest.bin_of(x_s, the frozen live|s|adv_bn_r250 edges) == 'q5' (x_s >= 0.4666730940341952 bids,
 0.285456120967865 asks; the edge file's sha256 ca688917... is required; quantiles are never recomputed). On side s while ON_s: a live order
 the baseline would KEEP is PULLed (pressure_pull); a slot the baseline would POST is WAITed (pressure_withhold, a withheld slot proposal,
 not an order). Structural removals (pre_tick) and the inventory rule keep precedence and their reasons; post-pending orders become eligible
 at their first live review; re-entry under the normal rules at the first decision after clearance (no timer); the other side keeps its
 policy unless its own signal is ON. A missing value (off_grid / no_market_row / nan / beyond_cache) takes no guard action and is counted
 with the builder's bn_age. Every intervention is logged (t, side, kind, gid, rung, x, bn_age, the known remaining or proposed quantity,
 the client-known position, mode, reason). DISABLED (enabled=False, the same class) delegates every action and reason and must reproduce the
 accepted baseline exactly.
CACHE (pressure_cache): one builder call per complete market for every tick of both sides (ceil(END / 50 ms) ticks); checked in the run
 against every saved baseline feature row of the market (snapshot_parity: the cached float32 bits at its side / tick = the saved value's),
 the side sign (side_sign: x_0 = -x_1 at every tick both sides have) and, on CAUSAL_N = 16 decisions per date (rank sha256(CAUSAL_PREFIX
 |day|m|s|k) over the date's ok ticks), causal_one: the builder's future-only perturbation (pf_pipeline_export.real_perturb, seeded by the
 rank) leaves the market row and the signal unchanged; a delivered adverse-move control (the step row's bn_r250 at the decision set to the
 adverse CONTROL_ADVERSE when OFF, to 0 when ON) flips the signal.
QUEUE (review 2c5dbd53 finding 2): venue diagnostics per generation (the engine's queue ahead / remaining after its submission tick, at its
 first live review tick and at a pressure pull tick, read after the tick; ahead_final) apart from the receipt-known estimates
 (pf_ladder_rest.queue_summary); for every pressure pull the same side's next generation (re-entry): ahead after submission, price, gap.
RUN (run_day; pilot = the first two complete markets of each date in the manifest's (day, m) order): per complete market the baseline path
 (its record's path_info must agree), the cache and its checks, the DISABLED and the ENABLED arm on fresh branches with the baseline's ident;
 the disabled arm's fingerprint, fills, orders, requests, value, initial queue and receipt-known queue estimates must equal the baseline's.
SELECTION CONTROL (selection(); the audit's saved live states of both sides, one hash-selected state per generation; uncensored +1 s windows;
 no fill / value enters the matching): selected = on_value of the state's float32 adv_bn_r250 under its side's frozen edges; strata = day x
 market x side x remaining band x age band x inventory sign x [float32(ctx_bid_reserved_shares) >= 200] (ASK-RESERVATION-GUARD-v1's bands:
 fp_remaining [0, 25) [25, 50) [50, 75) [75, inf) shares; ctx_generation_age_ms [0, 500) [500, 2000) [2000, 10000) [10000, inf) ms; inv_known
 < -1e-9 / within / > 1e-9; a missing value its own band); statistic = the mean window +1 s per state (per valid window share apart, with
 valid / total / excluded quantities); NULL_DRAWS = 500 with SEED = 20261121, in every stratum the selected count drawn uniformly without
 replacement from its uncensored states; INCONCLUSIVE below 5 movable strata or 30 movable selected states. Per-side figures descriptive.
 Not a null for the full feedback policy's P&L.
MEASUREMENT (measure(); gross, zero cost, cents; complete markets; every status counted): per arm / market the END value (the paired main
 comparison), the execution edge and +50 / +100 ms markouts with their valid quantities and censor counts, the +1 s markout on the shared
 fill-time cutoff t <= END - 1,000 ms (valid quantity, censored fills; late fills stay in END); by side and rung; good / bad fills by the
 +1 s sign; the identity END = eligible +1 s markout + movement from each eligible fill's +1 s mid to the END mark + late / missing-horizon
 END (eligible = cutoff and a +1 s mid); fills lost (baseline only) and added (candidate only) by the multiset of (t, side, price,
 quantity), their positive / negative +1 s and END contributions (guard - baseline END = added - lost); pressure pulls / withholds (those
 at g + PLACE >= END apart) and reducing-side suppressions (bids at I < 0, asks at I > 0); removals by reason; generations; the first-review
 pull rate; lifetimes (min(remove, full fill, END) - activation); the exact venue inventory from the fills (mean / peak |I|, time at |I| >=
 100, terminal) and exact both-side uptime (pf_ladder_audit.durations); the queue diagnostics; the missing-signal counts.
READINGS (predicates(), computed): END_paired and mo1s_per_share (IMPROVES_EVERY_DAY / WORSENS_EVERY_DAY / MIXED; per side too);
 participation and inventory deltas; selection (SELECTION_BEYOND_EXPOSURE when the selected mean +1 s is below the null p05;
 NO_SELECTION_BEYOND_EXPOSURE = not demonstrated by this screen; INCONCLUSIVE); interpretation: better_selection iff END and per-share
 improve every day and the selection control is beyond exposure; quoting_less iff the filled shares fall on every day and per-share does not
 improve every day; mixed otherwise. Not a promotion.
Usage (run_light): pf_ladder_pressure.py selftest | manifest | pilot <day> | run <day> | selection [out] | measure [out] (out: the rehearsal on the pilot copy)"""
import collections
import gzip
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_rest as M                                # noqa: E402  frozen: the baseline runner
import pf_ladder_audit as AU                              # noqa: E402  frozen: durations, fill_view
import pf_ladder_guard as GD                              # noqa: E402  frozen: manifest_members, band, inv_sign, the bands and null settings

C, pd, LB, BE, LS, A, X = M.C, M.pd, M.LB, M.BE, M.LS, M.A, M.X
VERSION = 'binance_pressure_guard_v1'
FEATURE = 'adv_bn_r250'
FI, AGE_I = M.FC.MARKET.index(FEATURE), M.FC.MARKET.index('bn_age')
EDGE_KEYS = {0: 'live|0|adv_bn_r250', 1: 'live|1|adv_bn_r250'}
EDGES_SHA256 = 'ca688917cd7ddc7c40e94031bee646f4fc1339511bb7c48fbe5270797a38bf80'
LAST_EDGES = {0: 0.4666730940341952, 1: 0.285456120967865}
REASONS = M.CMD_REASONS + ('pressure_pull', 'pressure_withhold')
ST_CODES = ('ok', 'off_grid', 'no_market_row', 'nan')
NULL_DRAWS, SEED, MIN_MOVABLE_STRATA, MIN_MOVABLE_ROWS = GD.NULL_DRAWS, GD.SEED, GD.MIN_MOVABLE_STRATA, GD.MIN_MOVABLE_ROWS
REM_BANDS, AGE_BANDS, B_EDGE = GD.REM_BANDS, GD.AGE_BANDS, 200.
CAUSAL_N, CAUSAL_PREFIX, BN_COL, CONTROL_ADVERSE = 16, 'binance_pressure_guard_v1|causal|', X.I.M.X37.index('bn_r250'), 10.
PILOT_PER_DAY, STEP, S_LIM = GD.PILOT_PER_DAY, M.STEP, 100.
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_pressure_20261121'
OUT = A.DATA / 'ladder_pressure_v1_20261121'
SOURCES = ('pf_ladder_pressure.py', 'pf_ladder_pressure_selftest.py', 'pf_ladder_guard.py', 'pf_ladder_rest.py', 'pf_branch_ladder.py', 'pf_branch_engine.py',
           'pf_ladder_audit.py', 'pf_pipeline_export.py', 'pf_flow_price_data.py', 'pf_flow_price_common.py', 'pf_stop_ev_minutes_inputs.py', 'pf_stop_model.py',
           'pf_stop_depth.py', 'pf_pull_depth.py', 'pf_influence.py')
FROZEN_RECEIPT = ('declaration.md', 'manifest.json', 'launch_pressure.sh')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def frozen():
    return bool(all(M.committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_pressure.py', 'pf_ladder_pressure_selftest.py'))
                and all(M.committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in FROZEN_RECEIPT))


def frozen_edges(path=None):
    """The frozen live|s|adv_bn_r250 bins of the accepted measurement (its sha256 required; never recomputed)."""
    p = Path(path or M.OUT / 'measure' / 'audit.json'); C.require(C.sha(p) == EDGES_SHA256, 'the edge file differs from the frozen one')
    e = json.loads(p.read_text())['edges']; out = {s: [float(v) for v in e[EDGE_KEYS[s]]] for s in (0, 1)}
    C.require(all(out[s][-1] == LAST_EDGES[s] for s in (0, 1)), 'the last edges differ from the declared ones'); return out


def on_value(x, edges_s):
    """ON for one value (header): finite, positive and q5 under the side's frozen bins."""
    return bool(x is not None and np.isfinite(x) and x > 0. and M.bin_of(float(x), edges_s) == 'q5')


class PressureGuard(LB.Policy):
    """The candidate (header): the baseline LadderInventory plus the side-specific Binance pressure guard; enabled=False delegates everything."""

    def __init__(self, B, cache, edges, enabled=True):
        super().__init__('BINANCE_PRESSURE_GUARD_v1' if enabled else 'pressure_guard_disabled', B.end)
        self.inner, self.cache, self.enabled, self.B = LB.LadderInventory(B), cache, bool(enabled), B
        self.edges = {int(s): [float(v) for v in e] for s, e in edges.items()}
        self.log, self.q_first, self.q_pull, self.seen, self.missing = [], [], [], set(), collections.Counter()

    def value(self, v):
        s = int(v['side']); k = int(round(float(v['t']) / STEP)); xs = self.cache['x'][s]
        if not 0 <= k < len(xs):
            return np.nan, 'beyond_cache', np.nan
        return float(xs[k]), ST_CODES[int(self.cache['status'][s][k])], float(self.cache['age'][s][k])

    def signal(self, v):
        x, st, ag = self.value(v)
        if st != 'ok':
            self.missing[(int(v['side']), st)] += 1
        return bool(self.enabled and st == 'ok' and on_value(x, self.edges[int(v['side'])])), x, ag

    def _note(self, v, kind, gid, rung, q, x, ag, act):
        self.log.append((float(v['t']), int(v['side']), kind, -1 if gid is None else int(gid), float(rung), float(x), float(ag), float(q), float(v['position']), v['mode'], act))

    def live(self, v, o):
        if int(o['gid']) not in self.seen:
            self.seen.add(int(o['gid'])); self.q_first.append((int(o['gid']), float(v['t'])))
        a, why = self.inner.live(v, o)
        if a == 'KEEP':
            on, x, ag = self.signal(v)
            if on:
                self._note(v, 'live', o['gid'], o['rung'], float(o['size']) - float(o['known_filled']), x, ag, 'pressure_pull'); self.q_pull.append((int(o['gid']), float(v['t'])))
                return 'PULL', 'pressure_pull'
        return a, why

    def slot(self, v, r, price):
        a, q, why = self.inner.slot(v, r, price)
        if a == 'POST':
            on, x, ag = self.signal(v)
            if on:
                self._note(v, 'slot', None, r, q, x, ag, 'pressure_withhold'); return 'WAIT', None, 'pressure_withhold'
        return a, q, why


def rule_sha():
    """The frozen rule's identity: the sha256 of on_value's and PressureGuard's source."""
    return hashlib.sha256((inspect.getsource(on_value) + inspect.getsource(PressureGuard)).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------- the cache and its checks
def pressure_cache(W_, fair_m, depth, t_abs0, end, feat_fn=None):
    """One builder call for every tick of both sides (header) -> dict(x, age, status, kk per side, nt, grids_equal)."""
    feat_fn = feat_fn or X.real_features; nt = int(np.ceil(end / STEP)); d = t_abs0 + STEP * np.arange(nt); kk, ch = {}, {}
    for s in (0, 1):
        tg = np.asarray(W_[s]['tg'], float); j = np.minimum(np.searchsorted(tg, d), len(tg) - 1); ok = np.abs(tg[j] - d) <= 1e-6
        kk[s] = np.where(ok, j, -1).astype(np.int64); ch[s] = sorted({int(v) for v in j[ok]})
    F = feat_fn(W_, fair_m, depth, ch); out = dict(x={}, age={}, status={}, kk=kk, nt=nt)
    for s in (0, 1):
        x = np.full(nt, np.nan, np.float32); ag = np.full(nt, np.nan, np.float32); st = np.zeros(nt, np.int8); Fs = F.get(s, {})
        for k in range(nt):
            j = int(kk[s][k])
            if j < 0:
                st[k] = ST_CODES.index('off_grid'); continue
            row = Fs.get(j)
            if row is None:
                st[k] = ST_CODES.index('no_market_row'); continue
            x[k] = np.float32(row[0][FI]); ag[k] = np.float32(row[0][AGE_I])
            if not np.isfinite(x[k]):
                st[k] = ST_CODES.index('nan')
        out['x'][s], out['age'][s], out['status'][s] = x, ag, st
    t0, t1 = np.asarray(W_[0]['tg'], float), np.asarray(W_[1]['tg'], float)
    out['grids_equal'] = bool(len(t0) == len(t1) and np.array_equal(t0, t1)); return out


def bits(v):
    return np.asarray([v], np.float32).view(np.uint32)[0]


def snapshot_parity(cache, rows):
    """Every saved baseline feature row of the market (side, t, value) against the cache: the float32 bits must agree."""
    n = eq = off = 0; bad = []
    for s, t, v in rows:
        n += 1; k = int(round(float(t) / STEP))
        if abs(k * STEP - float(t)) > 1e-6 or not 0 <= k < cache['nt'] or int(cache['status'][int(s)][k]) != 0:
            off += 1; bad.append([int(s), float(t), 'off_tick_or_status']); continue
        if bits(cache['x'][int(s)][k]) == bits(v):
            eq += 1
        else:
            bad.append([int(s), float(t), float(cache['x'][int(s)][k]), float(v)])
    return dict(rows=n, equal=eq, unavailable=off, mismatched=n - eq - off, first_bad=bad[:5])


def side_sign(cache):
    """x_0 = -x_1 at every tick both sides have (the builder's side sign on the same Binance row)."""
    both = (cache['status'][0] == 0) & (cache['status'][1] == 0); eq = cache['x'][0][both] == -cache['x'][1][both]
    return dict(ticks=int(both.sum()), equal=int(eq.sum()), mismatched=int((~eq).sum()))


def causal_rank(day, m, s, k):
    return hashlib.sha256(f'{CAUSAL_PREFIX}{day}|{int(m)}|{int(s)}|{int(k)}'.encode()).hexdigest()


def causal_pick(day, caches, n=CAUSAL_N):
    """The n minimum-rank (rank, m, s, k) over the date's ok ticks."""
    return sorted((causal_rank(day, m, s, k), int(m), s, int(k)) for m, c in caches.items() for s in (0, 1) for k in np.nonzero(c['status'][s] == 0)[0])[:n]


def causal_one(W_, fair_m, depth, cache, s, k, edges_s, rank, feat_fn=None, perturb=None):
    """One decision (header): the future-only perturbation leaves the market row and the signal unchanged; the delivered control flips it."""
    feat_fn, perturb = feat_fn or X.real_features, perturb or X.real_perturb
    j = int(cache['kk'][s][k]); t = float(np.asarray(W_[s]['tg'], float)[j]); ch = {0: [], 1: []}; ch[s] = [j]
    row = feat_fn(W_, fair_m, depth, ch)[s][j]; x = np.float32(row[0][FI]); on = on_value(float(x), edges_s)
    mut, fair2, dep2 = perturb(W_, fair_m, depth, j, t, np.random.default_rng(int(rank[:16], 16))); aft = feat_fn(mut, fair2, dep2, ch)[s][j]
    same = bool(np.array_equal(np.asarray(aft[0], np.float32).view(np.uint32), np.asarray(row[0], np.float32).view(np.uint32))
                and np.isclose(aft[1], row[1], equal_nan=True))
    ctl = 0. if on else (-CONTROL_ADVERSE if s == 0 else CONTROL_ADVERSE); w2 = dict(W_[s]); w2['X'] = np.array(W_[s]['X'], copy=True); w2['X'][j, BN_COL] = ctl
    w2['cache'] = {}; w2.pop('entry_local_cache', None); W2 = dict(W_); W2[s] = w2; x3 = np.float32(feat_fn(W2, fair_m, depth, ch)[s][j][0][FI])
    return dict(s=int(s), k=int(k), j=j, x=float(x), on=on, cached_equal=bool(bits(x) == bits(cache['x'][s][k])), same=same, same_signal=bool(on_value(float(np.float32(aft[0][FI])), edges_s) == on),
                control=ctl, x_control=float(x3), fired=bool(on_value(float(x3), edges_s) != on))


# ---------------------------------------------------------------------------------------------------------------------- the arms
class QueueObs:
    """Per tick: pf_ladder_rest.Observer plus the venue queue diagnostics (header), read after the tick (the state at the decision)."""

    def __init__(self, B, pol):
        self.obs, self.pol, self.ahead0, self.first, self.pull = M.Observer(B), pol, {}, {}, {}

    def __call__(self, B_, k):
        self.obs(B_, k)
        while len(self.ahead0) < len(B_.E.o):
            o = B_.E.o[len(self.ahead0)]; self.ahead0[int(o['gid'])] = [float(o['submit']), float(o['ahead']), float(o['rem'])]
        for lst, dst in ((self.pol.q_first, self.first), (self.pol.q_pull, self.pull)):
            for g, t in lst:
                o = B_.E.o[g]; dst[g] = [t, float(o['ahead']), float(o['rem'])]
            lst.clear()

    def arrays(self):
        return self.obs.arrays()

    def queue(self):
        return dict(ahead0={str(g): v for g, v in self.ahead0.items()}, first={str(g): v for g, v in self.first.items()}, pull={str(g): v for g, v in self.pull.items()})


def command_arrays(B):
    dc = np.array([[d[0], d[1], M.STATES.index(d[2]), M.CMD_ACTIONS.index(d[3]), -1 if d[4] is None else d[4]] for d in B.decisions], float).reshape(-1, 5)
    cm = np.array([[c[0], c[1], M.CMD_KINDS.index(c[2]), c[3], c[4], c[5], c[6], M.CMD_ACTIONS.index(c[7]), REASONS.index(c[8])] for c in B.cmd_log], float).reshape(-1, 9)
    return dict(decisions=dc, commands=cm)


def run_arm(path, ident, cache, edges, enabled):
    """One arm of one market on a fresh branch (the baseline's construction, the pressure policy) -> (branch, ledger record, policy, observer)."""
    B = LB.LadderBranch(path, LB.LADDER, end=ident['end_rel'], episode=0, ident=ident); pol = PressureGuard(B, cache, edges, enabled); obs = QueueObs(B, pol)
    LB.run_episode(B, pol, observe=obs); Lg = LB.finish(B); rec = M.ledger_record(B, Lg)
    rec['fill_mids'] = M.mids_of(np.asarray(B.E.f, float), B.E.fills, B.end); rec['guard_log'] = [list(x) for x in pol.log]
    rec['missing'] = {f'{s}|{st}': int(n) for (s, st), n in pol.missing.items()}; rec['queue_venue'] = obs.queue()
    return B, rec, pol, obs


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    man_run = json.loads((REPO / M.RECEIPT / 'manifest.json').read_text()); order, pilot = GD.manifest_members(man_run); e = frozen_edges()
    man = dict(version=VERSION, rule_sha256=rule_sha(), feature=FEATURE, edges={str(s): e[s] for s in (0, 1)}, edge_keys={str(s): EDGE_KEYS[s] for s in (0, 1)},
               edges_source=str(M.OUT / 'measure' / 'audit.json'), edges_sha256=EDGES_SHA256, sources=sources(),
               baseline=dict(freeze='47d2eec1', results='869fa85f', manifest_sha256=C.sha(REPO / M.RECEIPT / 'manifest.json')), days=man_run['days'], complete=order, pilot=pilot,
               causal=dict(n_per_date=CAUSAL_N, prefix=CAUSAL_PREFIX, bn_col=BN_COL, control_adverse=CONTROL_ADVERSE),
               null=dict(draws=NULL_DRAWS, seed=SEED, rem_bands=list(REM_BANDS), age_bands_ms=list(AGE_BANDS), b_edge=B_EDGE, min_movable_strata=MIN_MOVABLE_STRATA,
                         min_movable_rows=MIN_MOVABLE_ROWS), asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def day_paths(day, pilot, out=None):
    sub = Path(out or OUT) / ('pilot' if pilot else 'days')
    return {k: sub / f'{day}{s}' for k, s in (('rec', '.json'), ('gz', '.jsonl.gz'), ('npz', '_arrays.npz'), ('status', '_status.json'))}


def gz_records(p):
    return [json.loads(x) for x in gzip.decompress(Path(p).read_bytes()).decode().splitlines() if x]


def run_day(day, pilot=False, out=None, require_frozen=True, man_path=None, proj_path=None):
    """One date's complete markets (pilot: its first two) under both arms -> records and arrays, write-once (header)."""
    C.require(day in M.DAYS, f'{day} is not a declared day')
    if require_frozen:
        C.require(frozen(), 'the pressure runner / selftest / declaration / manifest / launcher are not the committed ones')
    man_path = Path(man_path or REPO / RECEIPT / 'manifest.json'); gm = json.loads(man_path.read_text())
    C.require(gm['rule_sha256'] == rule_sha(), 'the rule differs from the frozen one')
    C.require(gm['sources'] == sources(), 'a source (runner or builder) differs from the frozen one')
    proj_path = Path(proj_path or REPO / RECEIPT / 'logs' / 'pilot_projection.json')
    C.require(pilot or (proj_path.exists() and json.loads(proj_path.read_text()).get('run_rest') is True), 'the full run needs the write-once pilot projection with run_rest true')
    P = day_paths(day, pilot, out); C.require(not P['rec'].exists(), f'{day}: the output exists (write-once)'); P['rec'].parent.mkdir(parents=True, exist_ok=True)
    edges = frozen_edges(); mans = [int(m) for m in (gm['pilot'] if pilot else gm['complete'])[day]]; man_run = json.loads((REPO / M.RECEIPT / 'manifest.json').read_text())
    base = {int(r['m']): r for r in gz_records(M.day_paths(day, False)['gz'])}
    feat = pd.read_parquet(M.day_paths(day, False)['feat'], columns=['m', 'side', 't', 'status', FEATURE])
    members = [x for x in man_run['selected'] if x['day'] == day and int(x['m']) in mans]; t0 = time.time(); tim = dict(load=None, markets=[], causal=None)
    G = LS.Guard(1200. + 900. * max(1, len(members)), 900.); mon = LS.RN.Sampler(P['status'], LS.GUARD); mon.start()
    try:
        prs, fair, depth, feat_fn, allw, opens = M.load_day(day, members, man_run); tim['load'] = time.time() - t0; G.check('load')
        recs, arrs, caches, ctx = [], {}, {}, {}
        for x in members:
            tm = time.time(); m = int(x['m']); b = base[m]; W_ = prs[m]; path, st, info = M.market_path(W_, float(x['ws']), opens[m][1])
            C.require(st == 'complete' and info == b['path_info'], f'{day} {m}: the path differs from the baseline record')
            t_abs0, end = float(info['t_start']), float(info['end_rel'])
            ident = dict(day=day, instrument_id=m, start_abs_ms=t_abs0, end_abs_ms=t_abs0 + end, end_rel=end, opening_ms=float(x['ws']),
                         mark_source='QP.build_path supported-end mark', tape_id=BE.tape_id(path['frames'], path['trades'], path['mark']))
            cache = pressure_cache(W_, fair.get(m), depth, t_abs0, end, feat_fn); caches[m] = cache; ctx[m] = (W_, fair.get(m)); fm = feat[(feat.m == m) & (feat.status == 'ok')]
            out_m = dict(day=day, m=m, path_info=info, cache=dict(nt=cache['nt'], grids_equal=cache['grids_equal'],
                                                                  status={str(s): {c: int((cache['status'][s] == i).sum()) for i, c in enumerate(ST_CODES)} for s in (0, 1)},
                                                                  parity=snapshot_parity(cache, zip(fm.side, fm.t, fm[FEATURE])), sign=side_sign(cache)))
            arrs.update({f'{m}__cache__{k}{s}': cache[k][s] for k in ('x', 'age', 'status') for s in (0, 1)})
            for arm, en in (('disabled', False), ('pressure', True)):
                B, rec, pol, obs = run_arm(path, ident, cache, edges, en); cm = command_arrays(B); rec['queue_known'] = M.queue_summary(W_, rec, cm['commands'], t_abs0)
                out_m[arm] = rec; arrs.update({f'{m}__{arm}__{k}': v for k, v in dict(obs.arrays(), **cm).items()})
            d_, bl = out_m['disabled'], b['ledger']
            out_m['control'] = dict(fingerprint=d_['fingerprint'] == bl['fingerprint'], fills=d_['fills'] == bl['fills'], value=d_['value'] == bl['value'],
                                    orders=d_['orders'] == bl['orders'], requests=d_['requests'] == bl['requests'], queue_known=d_['queue_known'] == bl['queue'],
                                    ahead0={g: v[1] for g, v in d_['queue_venue']['ahead0'].items()} == bl['ahead0'])
            out_m['control_all'] = bool(all(out_m['control'].values())); recs.append(out_m); tim['markets'].append(dict(m=m, s=time.time() - tm)); G.check('market', time.time() - tm)
        tc = time.time(); causal = []
        for rank, m, s, k in causal_pick(day, caches):
            W_, fm_ = ctx[m]
            causal.append(dict(causal_one(W_, fm_, depth, caches[m], s, k, edges[s], rank, feat_fn), m=m, rank=rank, status='ok') if caches[m]['grids_equal']
                          else dict(m=m, s=int(s), k=int(k), rank=rank, status='grids_differ'))
        tim['causal'] = time.time() - tc
        b_ = LS.gz_bytes(recs); P['gz'].write_bytes(b_); np.savez_compressed(P['npz'], **arrs)
        LS.RN.atomic_json(P['rec'], dict(version=VERSION, day=day, pilot=bool(pilot), members=mans, causal=causal, sha256=dict(gz=hashlib.sha256(b_).hexdigest(), npz=C.sha(P['npz'])),
                                         manifest_sha256=C.sha(man_path), rule_sha256=rule_sha(), sources=sources(), timing=tim, elapsed_s=time.time() - t0, hwm_mib=LS.hwm(),
                                         asof=C.stamp()))
        return 'DAY_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(P['status'], dict(state='STOP_RESOURCE', day=day, latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


# ---------------------------------------------------------------------------------------------------------------------- selection control
def selection_table(S_, edges):
    """The live states of both sides with the pressure mask, strata and window values (uncensored only for the statistic; censored counted)."""
    g = S_[S_.kind == 'live'].copy()
    g['selected'] = [on_value(float(np.float32(v)), edges[int(s)]) if np.isfinite(v) else False for v, s in zip(g[FEATURE].astype(float), g.side.astype(int))]
    b200 = ['missing' if not np.isfinite(v) else int(float(np.float32(v)) >= B_EDGE) for v in g.ctx_bid_reserved_shares.astype(float)]
    g['stratum'] = [json.dumps([d, int(m), int(s), GD.band(r, REM_BANDS), GD.band(a, AGE_BANDS), GD.inv_sign(i), b]) for d, m, s, r, a, i, b in
                    zip(g.day, g.m, g.side, g.fp_remaining.astype(float), g.ctx_generation_age_ms.astype(float), g.inv_known.astype(float), b200)]
    g['uncensored'] = (~g.censored_1s.astype(bool)) & (g.win_mo1s.notna() | (g.win_n == 0))
    g['y'] = g.win_mo1s.fillna(0.).astype(float); g['valid_q'] = np.where(g.uncensored, g.win_q.astype(float), 0.)
    return g


def selection(S_, edges, draws=NULL_DRAWS, seed=SEED):
    """The matched-draw selection control (header) -> dict (the pooled answer; per-side figures descriptive)."""
    g = selection_table(S_, edges); u = g[g.uncensored]; sel = u[u.selected]
    J = float(sel.y.mean()) if len(sel) else None; Jq = float(sel.y.sum() / sel.valid_q.sum()) if len(sel) and sel.valid_q.sum() > 0 else None
    strata = {k: (int(v.selected.sum()), int(len(v))) for k, v in u.groupby('stratum')}; idx = {k: v.index.values for k, v in u.groupby('stratum')}
    movable = {k: (s, n) for k, (s, n) in strata.items() if 0 < s < n}; rng = np.random.default_rng(seed); null, nullq, nside = [], [], {0: [], 1: []}
    for _ in range(int(draws)):
        pick = [i for k, (s, n) in strata.items() if s > 0 for i in rng.choice(idx[k], size=s, replace=False)]
        y = u.loc[pick, 'y']; q = u.loc[pick, 'valid_q']; sd = u.loc[pick, 'side'].astype(int); null.append(float(y.mean())); nullq.append(float(y.sum() / q.sum()) if q.sum() > 0 else np.nan)
        for s in (0, 1):
            nside[s].append(float(y[sd == s].mean()) if (sd == s).any() else np.nan)
    null = np.asarray(null); nullq = np.asarray(nullq)
    thin = len(movable) < MIN_MOVABLE_STRATA or sum(s for s, n in movable.values()) < MIN_MOVABLE_ROWS
    ans = 'INCONCLUSIVE' if thin or J is None else ('SELECTION_BEYOND_EXPOSURE' if J < float(np.quantile(null, .05)) else 'NO_SELECTION_BEYOND_EXPOSURE')
    per_side = {}
    for s in (0, 1):
        ss = sel[sel.side.astype(int) == s]; ns = np.asarray(nside[s], float)
        per_side[str(s)] = dict(selected=int(len(ss)), J=float(ss.y.mean()) if len(ss) else None, null_mean=float(np.nanmean(ns)) if np.isfinite(ns).any() else None,
                                null_p05=float(np.nanquantile(ns, .05)) if np.isfinite(ns).any() else None,
                                percentile=float(np.nanmean(ns <= ss.y.mean())) if len(ss) and np.isfinite(ns).any() else None)
    return dict(answer=ans, selected=int(len(sel)), states=int(len(u)), censored=int((~g.uncensored).sum()), selected_censored=int((g.selected & ~g.uncensored).sum()),
                J=J, J_per_valid_share=Jq, valid_window_q=float(sel.valid_q.sum()), total_window_q=float(g[g.selected].win_q.astype(float).sum()),
                excluded_window_q=float(g[g.selected & ~g.uncensored].win_q.astype(float).sum()), null_mean=float(null.mean()), null_p05=float(np.quantile(null, .05)),
                null_p95=float(np.quantile(null, .95)), percentile=float((null <= J).mean()) if J is not None else None, null_q_p05=float(np.nanquantile(nullq, .05)),
                strata=len(strata), movable_strata=len(movable), movable_selected=int(sum(s for s, n in movable.values())),
                degenerate_strata=int(sum(1 for s, n in strata.values() if s == n and s > 0)), missing_feature=int((~np.isfinite(g[FEATURE].astype(float))).sum()),
                per_side=per_side, draws=int(draws), seed=int(seed))


def run_selection(out=OUT, require_frozen=True):
    out = Path(out); p = out / 'selection.json'
    if require_frozen:
        C.require(frozen(), 'not the committed pressure runner')
    C.require(not p.exists(), 'the selection control exists (write-once)'); out.mkdir(parents=True, exist_ok=True)
    S_ = pd.read_csv(AU.OUT / 'states.csv', float_precision='round_trip', dtype={'day': str}); res = selection(S_, frozen_edges())
    res['inputs'] = dict(states_sha256=C.sha(AU.OUT / 'states.csv'), edges_sha256=EDGES_SHA256); C.dump(p, res); print(json.dumps(res, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------------------- measurement
SG = {0: 1., 1: -1.}
FCOLS = ['t', 'gid', 'side', 'rung', 'q', 'p', 'end', 'edge', 'mo50', 'mo100', 'mo1s', 'move_end']


def fill_frame(rec):
    """One arm's fills with their contributions (cents): END s q (mark - p); edge / +50 / +100 / +1 s s q (mid - p); s q (mark - mid(+1 s))."""
    mark = float(rec['mark']); od = {int(o['gid']): o for o in rec['orders']}; rows = []
    for f, mm in zip(rec['fills'], rec['fill_mids']):
        t, gid, s, q, p = float(f[0]), int(f[1]), int(f[2]), float(f[3]), float(f[4]); sg = SG[s]
        v = lambda i: np.nan if mm[i] is None else sg * q * (float(mm[i]) - p)
        rows.append([t, gid, s, int(round(float(od[gid]['rung']))), q, p, sg * q * (mark - p), v(0), v(2), v(4), v(6), np.nan if mm[6] is None else sg * q * (mark - float(mm[6]))])
    return pd.DataFrame(rows, columns=FCOLS)


def inventory_exact(rec, end):
    """The venue inventory from the fills, piecewise constant on [0, END]: mean |I|, peak |I|, time share |I| >= S_LIM."""
    ev = sorted((float(f[0]), SG[int(f[2])] * float(f[3])) for f in rec['fills']); t_prev, I_, area, high, peak = 0., 0., 0., 0., 0.
    for t, dq in ev + [(end, 0.)]:
        t = min(max(t, 0.), end); area += abs(I_) * (t - t_prev); high += (t - t_prev) if abs(I_) >= S_LIM - 1e-9 else 0.; t_prev = t; I_ += dq; peak = max(peak, abs(I_))
    return dict(mean_abs_inv=area / end, peak_abs_inv=peak, share_abs_ge_S=high / end, terminal_inv=float(rec['position']))


def lifetimes(rec):
    fv = AU.fill_view(rec); end = float(rec['end']); out = {0: [], 1: []}
    for o in rec['orders']:
        if o['refused']:
            continue
        cum, full = 0., None
        for y in fv.get(int(o['gid']), []):
            cum += y[3]
            if full is None and cum >= float(o['size']) - LB.TOL:
                full = y[0]
        out[int(o['side'])].append(min(float(o['remove']), full if full is not None else np.inf, end) - float(o['act']))
    return out


def queue_metrics(rec):
    """The queue diagnostics (header): venue ahead after submission / at the first live review / at a pressure pull; the receipt-known first
    estimate; for each pressure pull the same side's next generation (re-entry)."""
    qv, qk = rec['queue_venue'], rec['queue_known']; od = {int(o['gid']): o for o in rec['orders']}
    a0 = [v[1] for v in qv['ahead0'].values()]; af = [v[1] for v in qv['first'].values()]; kf = [v['first'] for v in qk.values() if v['first'] is not None]
    re_ = []
    for g, v in qv['pull'].items():
        o = od[int(g)]; nxt = sorted((float(x['submit']), int(x['gid'])) for x in rec['orders'] if int(x['side']) == int(o['side']) and float(x['submit']) > v[0])
        if nxt:
            g2 = nxt[0][1]; re_.append(dict(gap=nxt[0][0] - v[0], ahead_pull=v[1], ahead_re=qv['ahead0'][str(g2)][1], same_price=float(od[g2]['price']) == float(o['price'])))
    mean = lambda z: float(np.mean(z)) if len(z) else np.nan
    return dict(q_ahead0_mean=mean(a0), q_first_venue_mean=mean(af), q_first_known_mean=mean(kf), q_pulls=len(qv['pull']), q_pull_venue_mean=mean([v[1] for v in qv['pull'].values()]),
                q_reentries=len(re_), q_reentry_gap_mean=mean([r['gap'] for r in re_]), q_reentry_ahead_mean=mean([r['ahead_re'] for r in re_]),
                q_surrendered_mean=mean([r['ahead_re'] - r['ahead_pull'] for r in re_]), q_reentry_same_price=mean([float(r['same_price']) for r in re_]))


def arm_metrics(rec, a, end):
    """One arm of one market (header)."""
    F = fill_frame(rec); x = dict(value=float(rec['value']), fills=len(F)); cut = (F.t <= end - 1000.).values; v1 = F.mo1s.notna().values; el = cut & v1
    for nm, ms in (('all', np.ones(len(F), bool)), ('bid', (F.side == 0).values), ('ask', (F.side == 1).values)):
        D_ = F[ms]; x.update({f'{nm}_q': float(D_.q.sum()), f'{nm}_end': float(D_.end.sum())})
        for h in ('edge', 'mo50', 'mo100'):
            ok = D_[h].notna(); x.update({f'{nm}_{h}': float(D_[h][ok].sum()), f'{nm}_{h}_valid_q': float(D_.q[ok].sum()), f'{nm}_{h}_censored': int((~ok).sum())})
        cz = F[ms & cut]; ve = F[ms & el]
        x.update({f'{nm}_cut_q': float(cz.q.sum()), f'{nm}_cut_mo1s': float(ve.mo1s.sum()), f'{nm}_cut_valid_q': float(ve.q.sum()), f'{nm}_cut_censored': int(cz.mo1s.isna().sum()),
                  f'{nm}_good_mo1s': float(ve.mo1s[ve.mo1s > 0].sum()), f'{nm}_bad_mo1s': float(ve.mo1s[ve.mo1s < 0].sum()), f'{nm}_id_mo1s': float(ve.mo1s.sum()),
                  f'{nm}_id_move': float(ve.move_end.sum()), f'{nm}_id_late': float(F[ms & ~el].end.sum())})
    for r_ in (0, 1):
        D_ = F[F.rung == r_]; x[f'rung{r_}_q'] = float(D_.q.sum()); x[f'rung{r_}_end'] = float(D_.end.sum())
    gl = rec.get('guard_log', [])
    for s, nm in ((0, 'bid'), (1, 'ask')):
        z = [e for e in gl if int(e[1]) == s]
        x.update({f'{nm}_pulls': sum(1 for e in z if e[10] == 'pressure_pull'), f'{nm}_withholds': sum(1 for e in z if e[10] == 'pressure_withhold'),
                  f'{nm}_reducing_suppressed': sum(1 for e in z if (e[8] < -1e-9 if s == 0 else e[8] > 1e-9))})
    x.update(pulls=x['bid_pulls'] + x['ask_pulls'], withholds=x['bid_withholds'] + x['ask_withholds'], reducing_suppressed=x['bid_reducing_suppressed'] + x['ask_reducing_suppressed'],
             withholds_near_end=sum(1 for e in gl if e[10] == 'pressure_withhold' and e[0] + BE.PLACE >= end))
    cm = a['commands']; pull = cm[:, 7] == M.CMD_ACTIONS.index('PULL')
    for rs in ('inventory_pull', 'pressure_pull'):
        for s, nm in ((0, 'bid'), (1, 'ask')):
            x[f'{nm}_req_{rs}'] = int((pull & (cm[:, 8] == REASONS.index(rs)) & (cm[:, 1] == s)).sum())
    first = {}
    for c in cm[(cm[:, 2] == M.CMD_KINDS.index('live')) & (cm[:, 3] >= 0)]:
        first.setdefault(int(c[3]), c)
    x.update(reviewed=len(first), first_review_pulls=sum(1 for c in first.values() if int(c[7]) == M.CMD_ACTIONS.index('PULL')),
             first_review_pressure=sum(1 for c in first.values() if int(c[8]) == REASONS.index('pressure_pull')))
    life = lifetimes(rec)
    for s, nm in ((0, 'bid'), (1, 'ask')):
        oo = [o for o in rec['orders'] if int(o['side']) == s]; lt = life[s]
        x.update({f'{nm}_generations': len(oo), f'{nm}_structural': sum(1 for o in oo if o['why'] == 'structural'), f'{nm}_policy': sum(1 for o in oo if o['why'] == 'policy'),
                  f'{nm}_life_mean_ms': float(np.mean(lt)) if lt else np.nan, f'{nm}_life_median_ms': float(np.median(lt)) if lt else np.nan})
    x.update(generations=len(rec['orders']), **inventory_exact(rec, end), **queue_metrics(rec))
    ms_ = rec.get('missing', {}); x.update(missing_bid=sum(v for k, v in ms_.items() if k.startswith('0|')), missing_ask=sum(v for k, v in ms_.items() if k.startswith('1|')))
    return x


def lost_added(rb, rg, end):
    """Fills lost (baseline only) and added (candidate only) by the multiset of (t, side, price, quantity) (header)."""
    key = lambda f: (round(float(f[0]), 6), int(f[2]), round(float(f[4]), 6), round(float(f[3]), 6))

    def unmatched(fills, other):
        rem = collections.Counter(map(key, other)); out = []
        for f in fills:
            k = key(f); out.append(rem[k] <= 0); rem[k] -= 1 if rem[k] > 0 else 0
        return np.array(out, bool)
    Fb, Fg = fill_frame(rb), fill_frame(rg); lo, ad = Fb[unmatched(rb['fills'], rg['fills'])], Fg[unmatched(rg['fills'], rb['fills'])]; out = {}
    for nm, D_ in (('lost', lo), ('added', ad)):
        el = (D_.t <= end - 1000.) & D_.mo1s.notna()
        out.update({f'{nm}_n': len(D_), f'{nm}_q': float(D_.q.sum()), f'{nm}_mo1s_pos': float(D_.mo1s[el & (D_.mo1s > 0)].sum()), f'{nm}_mo1s_neg': float(D_.mo1s[el & (D_.mo1s < 0)].sum()),
                    f'{nm}_end_pos': float(D_.end[D_.end > 0].sum()), f'{nm}_end_neg': float(D_.end[D_.end < 0].sum()), f'{nm}_end': float(D_.end.sum())})
    out['delta_value'] = float(rg['value']) - float(rb['value']); out['identity_gap'] = out['delta_value'] - (out['added_end'] - out['lost_end'])
    return out


def predicates(R_, sel_answer):
    """The declared readings (header)."""
    g = R_[R_.arm == 'pressure'].set_index(['day', 'm']); b = R_[R_.arm == 'disabled'].set_index(['day', 'm']); days = sorted(R_.day.unique()); out = dict(days=days)

    def cat(vals):
        return 'IMPROVES_EVERY_DAY' if all(v > 0 for v in vals) else 'WORSENS_EVERY_DAY' if all(v < 0 for v in vals) else 'MIXED'

    def per(X_, d, side):
        q = X_.loc[d][f'{side}_cut_valid_q'].sum(); return float(X_.loc[d][f'{side}_cut_mo1s'].sum() / q) if q > 0 else np.nan
    dEND = {d: float(g.loc[d].value.sum() - b.loc[d].value.sum()) for d in days}
    dEs = {side: {d: float(g.loc[d][f'{side}_end'].sum() - b.loc[d][f'{side}_end'].sum()) for d in days} for side in ('bid', 'ask')}
    dps = {side: {d: per(g, d, side) - per(b, d, side) for d in days} for side in ('all', 'bid', 'ask')}
    dq = {d: float(g.loc[d].all_q.sum() - b.loc[d].all_q.sum()) for d in days}; mk = g.value - b.value
    out.update(END_paired=dict(answer=cat(list(dEND.values())), by_day=dEND, total=float(sum(dEND.values())), markets_better=int((mk > 1e-9).sum()),
                               markets_worse=int((mk < -1e-9).sum()), markets_equal=int((mk.abs() <= 1e-9).sum())),
               END_by_side={side: dict(answer=cat(list(dEs[side].values())), by_day=dEs[side]) for side in ('bid', 'ask')},
               mo1s_per_share={side: dict(answer=cat(list(dps[side].values())) if all(np.isfinite(v) for v in dps[side].values()) else 'MIXED', by_day=dps[side])
                               for side in ('all', 'bid', 'ask')},
               participation=dict(filled_q_delta_by_day=dq, both_exact_delta=float(g.both_exact.mean() - b.both_exact.mean()), bid_exact_delta=float(g.bid_exact.mean() - b.bid_exact.mean()),
                                  ask_exact_delta=float(g.ask_exact.mean() - b.ask_exact.mean()), pulls=int(g.pulls.sum()), withholds=int(g.withholds.sum()),
                                  reducing_suppressed=int(g.reducing_suppressed.sum())),
               inventory=dict(mean_abs_delta=float(g.mean_abs_inv.mean() - b.mean_abs_inv.mean()), share_abs_ge_S_delta=float(g.share_abs_ge_S.mean() - b.share_abs_ge_S.mean()),
                              terminal_abs_delta=float(g.terminal_inv.abs().mean() - b.terminal_inv.abs().mean())),
               selection=sel_answer)
    better = (out['END_paired']['answer'] == 'IMPROVES_EVERY_DAY' and out['mo1s_per_share']['all']['answer'] == 'IMPROVES_EVERY_DAY' and sel_answer == 'SELECTION_BEYOND_EXPOSURE')
    less = all(v < 0 for v in dq.values()) and out['mo1s_per_share']['all']['answer'] != 'IMPROVES_EVERY_DAY'
    out['interpretation'] = 'better_selection' if better else ('quoting_less' if less else 'mixed')
    out['note'] = 'the second guard candidate of the ladder line on 19 consumed-day markets; no profitability, significance, validation or promotion claim'
    return out


def measure(out=OUT, man_path=None):
    out = Path(out); md = out / 'measure'; C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)')
    C.require((out / 'selection.json').exists(), 'the selection control must run before the measurement'); md.mkdir(parents=True, exist_ok=True)
    gm = json.loads(Path(man_path or REPO / RECEIPT / 'manifest.json').read_text()); sel = json.loads((out / 'selection.json').read_text()); rows, ctl, la, cz = [], [], [], []
    for day in gm['days']:
        P = day_paths(day, False, out); rec = json.loads(P['rec'].read_text()); b = P['gz'].read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256']['gz'] and C.sha(P['npz']) == rec['sha256']['npz'] and rec['rule_sha256'] == gm['rule_sha256'], f'{day}: identity')
        z = np.load(P['npz']); cz += [dict(c, day=day) for c in rec['causal']]
        for r in LS.gz_rows(b):
            m = int(r['m']); end = float(r['path_info']['end_rel']); pa, sg_ = r['cache']['parity'], r['cache']['sign']
            ctl.append(dict(day=day, m=m, **r['control'], control_all=r['control_all'], parity_rows=pa['rows'], parity_equal=pa['equal'], parity_unavailable=pa['unavailable'],
                            parity_mismatched=pa['mismatched'], sign_ticks=sg_['ticks'], sign_mismatched=sg_['mismatched'], grids_equal=r['cache']['grids_equal'],
                            **{f'cache_{s}_{c}': n for s, v in r['cache']['status'].items() for c, n in v.items()}))
            for arm in ('disabled', 'pressure'):
                a = {k.split('__')[2]: z[k] for k in z.files if k.startswith(f'{m}__{arm}__')}
                du = AU.durations(dict(day=day, m=m, ledger=r[arm]), {f'{day}|{m}__nva': a['nva']})
                rows.append(dict(day=day, m=m, arm=arm, both_exact=du['both_exact'], ask_exact=du['ask_exact'], bid_exact=du['bid_exact'], **arm_metrics(r[arm], a, end)))
            la.append(dict(day=day, m=m, **lost_added(r['disabled'], r['pressure'], end)))
    R_ = pd.DataFrame(rows); CT = pd.DataFrame(ctl); LA = pd.DataFrame(la); CZ = pd.DataFrame(cz); pr = predicates(R_, sel['answer'])
    for n, D_ in (('arms.csv', R_), ('control.csv', CT), ('lost_added.csv', LA), ('causal.csv', CZ)):
        D_.to_csv(md / n, index=False)
    ok_c = CZ[CZ.status == 'ok']
    integ = dict(control_all=bool(CT.control_all.all()), parity_rows=int(CT.parity_rows.sum()), parity_mismatched=int(CT.parity_mismatched.sum()),
                 parity_unavailable=int(CT.parity_unavailable.sum()), sign_mismatched=int(CT.sign_mismatched.sum()), causal_n=int(len(CZ)), causal_ok=int(len(ok_c)),
                 causal_same=int(ok_c.same.sum()), causal_same_signal=int(ok_c.same_signal.sum()), causal_fired=int(ok_c.fired.sum()), causal_cached_equal=int(ok_c.cached_equal.sum()),
                 identity_gap_max=float(LA.identity_gap.abs().max()))
    C.dump(md / 'audit.json', dict(version=VERSION, asof=C.stamp(), predicates=pr, integration=integ, markets=int(R_.m.nunique()), selection=sel,
                                   outputs={n: C.sha(md / n) for n in ('arms.csv', 'control.csv', 'lost_added.csv', 'causal.csv')}, sources=sources(), rule_sha256=gm['rule_sha256']))
    print(json.dumps(dict(predicates=pr, integration=integ), indent=1, default=str))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_pressure_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1500])
    elif cmd in ('pilot', 'run'):
        print(run_day(sys.argv[2], pilot=(cmd == 'pilot')))
    elif cmd == 'selection':
        run_selection(*(sys.argv[2:3] or [OUT]))
    elif cmd == 'measure':
        measure(*(sys.argv[2:3] or [OUT]))
    else:
        sys.exit('usage: pf_ladder_pressure.py selftest | manifest | pilot <day> | run <day> | selection [out] | measure [out]')
