"""ASK-RESERVATION-GUARD-v1 (P-2026-005; 2026-11-21, session aea5bfb9). Review f8c5e579 (target 6b14fbf9) accepted the lead-time audit and named ONE
next priority: one declared candidate against the frozen default-post ladder (LADDER_REST_INVENTORY_v1, frozen 47d2eec1) on the same 19 complete
markets, all 96 statuses preserved. An adaptive consumed-day mechanism test, counted as ONE new candidate; no bid mirror, threshold sweep, new
model or data day. No profitability, significance or validation claim.
RULE (AskGuard, frozen with its sha256 in the manifest): signal ON iff float32(B) >= EDGE, B = the client-known reservation of every unclosed
 bid at the decision (pf_branch_ladder side_view, after the notices and every earlier command of the tick), EDGE = the last frozen Oct 24-25
 edge of live|1|ctx_bid_reserved_shares (the frozen measurement's audit.json; pf_ladder_rest.bin_of's q5 = x >= that edge). On the ASK side
 only: a live ask the baseline would KEEP is PULLed (guard_pull); a slot the baseline would POST is WAITed (guard_withhold). Everything else -
 bids, the inventory rule (it keeps precedence and its own reasons), sizes, structural rules, reservations, tick / notice order, paths and
 marks - is the baseline's; post-pending asks become eligible at their first live review; no timer (the signal is re-read at every command);
 the 37 ms cancel latency and the 50 ms grid are the engine's. Every intervention is logged with its time, gid / rung, B, the client-known
 position and the mode. DISABLED (the same class, enabled=False) delegates every action and reason: it must reproduce the accepted baseline.
RUN (run_day; pilot = the first two complete markets of each date in the manifest's (day, m) order): per complete market the baseline path
 (pf_ladder_rest.market_path, its record's path_info must agree), the DISABLED and the ENABLED arm on fresh branches with the baseline's ident,
 pf_ladder_rest.Observer and mids_of; the disabled arm's ledger fingerprint, fills and value must equal the saved baseline's (control).
SELECTION CONTROL (selection(); the audit's saved live states of ask generations, one hash-selected state each; uncensored +1 s windows; no
 fill / value enters the matching): selected = q5 of ctx_bid_reserved_shares by the frozen edges; strata = day x market x remaining band
 (fp_remaining [0, 25), [25, 50), [50, 75), [75, inf)) x age band (ctx_generation_age_ms [0, 500), [500, 2000), [2000, 10000), [10000, inf)) x
 inventory sign (the client-known inventory inv_known = ctx_inventory_shares: < -1e-9 / within 1e-9 / > 1e-9); missing values their own band; statistic = the mean window +1 s per state
 (and per valid window share, the quantity of the same uncensored states); NULL_DRAWS = 500 draws with SEED, in every stratum the selected count
 drawn uniformly without replacement from its uncensored states. Movable strata: 0 < selected < states; INCONCLUSIVE below MIN_MOVABLE_STRATA
 movable strata or MIN_MOVABLE_ROWS movable selected states. Not a null for the full feedback policy's P&L.
MEASUREMENT (measure(); gross, zero cost, cents; complete markets): paired per market and per day END value (guard - baseline, the main
 comparison); edge and +50 / +100 / +1,000 ms markouts total and per filled share with valid quantities and censor counts; the +1 s comparison on
 the shared fill-time cutoff t <= END - 1,000 ms (late fills stay in END); by side and rung; good / bad fill contributions by the +1 s sign;
 guard pulls / withholds and the suppressed reducing-side ask opportunities (position > 0); structural / inventory removals; generations
 (reposts) and lifetimes; mean / peak |I|, time at |I| >= 100, terminal inventory; exact both-side uptime (pf_ladder_audit.durations).
READINGS (predicates(), computed): END_paired and mo1s_per_share (IMPROVES_EVERY_DAY / WORSENS_EVERY_DAY / MIXED; by side too); participation
 deltas; selection (SELECTION_BEYOND_EXPOSURE when the selected mean is below the null p05, NO_SELECTION_BEYOND_EXPOSURE, or INCONCLUSIVE);
 interpretation: better_selection iff END and per-share improve every day and the selection control is beyond exposure; quoting_less iff the
 filled shares fall on every day and per-share does not improve every day; mixed otherwise. Not a promotion.
CORRECTED SUMMARIES (review f8c5e579): (1) per-share terms publish valid_window_q (uncensored states), total and excluded quantity; (3) the
 generation +1 s censor count = generations with fills whose lifetime +1 s is missing, checked 0 <= count <= generations.
Usage (run_light): pf_ladder_guard.py selftest | manifest | pilot <day> | run <day> | selection | measure"""
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pf_ladder_rest as M                                # noqa: E402  frozen: the baseline runner
import pf_ladder_audit as AU                              # noqa: E402  frozen: the audit (durations, windows)

C, pd, LB, BE, LS, A = M.C, M.pd, M.LB, M.BE, M.LS, M.A
VERSION = 'ask_reservation_guard_v1'
EDGE_KEY = 'live|1|ctx_bid_reserved_shares'
REASONS = M.CMD_REASONS + ('guard_pull', 'guard_withhold')
NULL_DRAWS, SEED, MIN_MOVABLE_STRATA, MIN_MOVABLE_ROWS = 500, 20261121, 5, 30
REM_BANDS, AGE_BANDS = (25., 50., 75.), (500., 2000., 10000.)
PILOT_PER_DAY = 2
STEP = M.STEP
REPO = HERE.parents[1]
RECEIPT = 'predictfun/reviews/ladder_guard_20261121'
OUT = A.DATA / 'ladder_guard_v1_20261121'
SOURCES = ('pf_ladder_guard.py', 'pf_ladder_guard_selftest.py', 'pf_ladder_rest.py', 'pf_branch_ladder.py', 'pf_branch_engine.py', 'pf_ladder_audit.py')


def sources():
    return {n: C.sha(HERE / n) for n in SOURCES}


def frozen():
    return bool(all(M.committed_equal(f'predictfun/research/{n}', HERE / n) for n in ('pf_ladder_guard.py', 'pf_ladder_guard_selftest.py'))
                and all(M.committed_equal(f'{RECEIPT}/{n}', REPO / RECEIPT / n) for n in ('declaration.md', 'manifest.json', 'launch_guard.sh')))


def frozen_edge():
    return float(json.loads((M.OUT / 'measure' / 'audit.json').read_text())['edges'][EDGE_KEY][-1])


class AskGuard(LB.Policy):
    """The candidate (header): the baseline LadderInventory plus the ask-side reservation guard; enabled=False delegates everything."""

    def __init__(self, B, edge, enabled=True):
        super().__init__('ASK_RESERVATION_GUARD_v1' if enabled else 'ask_guard_disabled', B.end)
        self.inner, self.edge, self.enabled, self.B = LB.LadderInventory(B), float(edge), bool(enabled), B; self.log = []

    def signal(self, v):
        b = np.float32(v['B'])
        return bool(self.enabled and np.isfinite(b) and float(b) >= self.edge)

    def _note(self, v, kind, gid, rung, act):
        self.log.append((float(v['t']), int(v['side']), kind, -1 if gid is None else int(gid), float(rung), float(np.float32(v['B'])), float(v['position']), v['mode'], act))

    def live(self, v, o):
        a, why = self.inner.live(v, o)
        if int(v['side']) == 1 and a == 'KEEP' and self.signal(v):
            self._note(v, 'live', o['gid'], o['rung'], 'guard_pull'); return 'PULL', 'guard_pull'
        return a, why

    def slot(self, v, r, price):
        a, q, why = self.inner.slot(v, r, price)
        if int(v['side']) == 1 and a == 'POST' and self.signal(v):
            self._note(v, 'slot', None, r, 'guard_withhold'); return 'WAIT', None, 'guard_withhold'
        return a, q, why


def rule_sha():
    """The frozen rule's identity: the sha256 of AskGuard's source."""
    return hashlib.sha256(inspect.getsource(AskGuard).encode()).hexdigest()


def command_arrays(B):
    dc = np.array([[d[0], d[1], M.STATES.index(d[2]), M.CMD_ACTIONS.index(d[3]), -1 if d[4] is None else d[4]] for d in B.decisions], float).reshape(-1, 5)
    cm = np.array([[c[0], c[1], M.CMD_KINDS.index(c[2]), c[3], c[4], c[5], c[6], M.CMD_ACTIONS.index(c[7]), REASONS.index(c[8])] for c in B.cmd_log], float).reshape(-1, 9)
    return dict(decisions=dc, commands=cm)


def run_arm(path, ident, edge, enabled):
    """One arm of one market on a fresh branch (the baseline's construction, the guard policy) -> (branch, ledger record, policy, observer)."""
    B = LB.LadderBranch(path, LB.LADDER, end=ident['end_rel'], episode=0, ident=ident); pol = AskGuard(B, edge, enabled); obs = M.Observer(B)
    LB.run_episode(B, pol, observe=obs); Lg = LB.finish(B); rec = M.ledger_record(B, Lg)
    rec['fill_mids'] = M.mids_of(np.asarray(B.E.f, float), B.E.fills, B.end); rec['guard_log'] = [list(x) for x in pol.log]
    return B, rec, pol, obs


def manifest_members(man_run):
    """The complete markets of the baseline run (its records) in the manifest's (day, m) order, and the pilot."""
    comp = {}
    for day in man_run['days']:
        rr = [json.loads(x) for x in __import__('gzip').decompress(M.day_paths(day, False)['gz'].read_bytes()).decode().splitlines() if x]
        comp[day] = [int(r['m']) for r in rr if r['status'] == 'complete']
    order = {day: [int(x['m']) for x in sorted(man_run['selected'], key=lambda x: (x['day'], int(x['m']))) if x['day'] == day and int(x['m']) in comp[day]] for day in man_run['days']}
    return order, {day: v[:PILOT_PER_DAY] for day, v in order.items()}


def build_manifest(path=None):
    path = Path(path or REPO / RECEIPT / 'manifest.json'); C.require(not path.exists(), 'the manifest exists (write-once)')
    man_run = json.loads((REPO / M.RECEIPT / 'manifest.json').read_text()); order, pilot = manifest_members(man_run)
    edge = frozen_edge(); C.require(edge == 200., f'the frozen edge {edge} is not the 200 shares the review declared')
    man = dict(version=VERSION, rule_sha256=rule_sha(), edge=edge, edge_key=EDGE_KEY, edges_source=str(M.OUT / 'measure' / 'audit.json'), edges_sha256=C.sha(M.OUT / 'measure' / 'audit.json'),
               baseline=dict(freeze='47d2eec1', results='869fa85f', manifest_sha256=C.sha(REPO / M.RECEIPT / 'manifest.json')), days=man_run['days'], complete=order, pilot=pilot,
               null=dict(draws=NULL_DRAWS, seed=SEED, rem_bands=list(REM_BANDS), age_bands_ms=list(AGE_BANDS), min_movable_strata=MIN_MOVABLE_STRATA, min_movable_rows=MIN_MOVABLE_ROWS),
               asof=C.stamp())
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(man, indent=1, sort_keys=True) + '\n'); return man


def day_paths(day, pilot, out=None):
    sub = Path(out or OUT) / ('pilot' if pilot else 'days')
    return {k: sub / f'{day}{s}' for k, s in (('rec', '.json'), ('gz', '.jsonl.gz'), ('npz', '_arrays.npz'), ('status', '_status.json'))}


def run_day(day, pilot=False, out=None, require_frozen=True):
    """One date's complete markets (pilot: its first two) under both arms -> records and arrays, write-once (header)."""
    C.require(day in M.DAYS, f'{day} is not a declared day')
    if require_frozen:
        C.require(frozen(), 'the guard runner / selftest / declaration / manifest / launcher are not the committed ones')
    gm = json.loads((REPO / RECEIPT / 'manifest.json').read_text()); C.require(gm['rule_sha256'] == rule_sha(), 'the rule differs from the frozen one')
    C.require(pilot or (REPO / RECEIPT / 'logs' / 'pilot_projection.json').exists() and json.loads((REPO / RECEIPT / 'logs' / 'pilot_projection.json').read_text()).get('run_rest') is True,
              'the full run needs the write-once pilot projection with run_rest true')
    P = day_paths(day, pilot, out); C.require(not P['rec'].exists(), f'{day}: the output exists (write-once)'); P['rec'].parent.mkdir(parents=True, exist_ok=True)
    mans = [int(m) for m in (gm['pilot'] if pilot else gm['complete'])[day]]; man_run = json.loads((REPO / M.RECEIPT / 'manifest.json').read_text())
    base = {int(r['m']): r for r in (json.loads(x) for x in __import__('gzip').decompress(M.day_paths(day, False)['gz'].read_bytes()).decode().splitlines() if x)}
    members = [x for x in man_run['selected'] if x['day'] == day and int(x['m']) in mans]; t0 = time.time(); tim = dict(load=None, markets=[])
    G = LS.Guard(1200. + 900. * max(1, len(members)), 900.); mon = LS.RN.Sampler(P['status'], LS.GUARD); mon.start()
    try:
        prs, fair, depth, feat_fn, allw, opens = M.load_day(day, members, man_run); tim['load'] = time.time() - t0; G.check('load')
        recs, arrs = [], {}
        for x in members:
            tm = time.time(); m = int(x['m']); b = base[m]; W_ = prs[m]; path, st, info = M.market_path(W_, float(x['ws']), opens[m][1])
            C.require(st == 'complete' and info == b['path_info'], f'{day} {m}: the path differs from the baseline record')
            t_abs0, end = float(info['t_start']), float(info['end_rel'])
            ident = dict(day=day, instrument_id=m, start_abs_ms=t_abs0, end_abs_ms=t_abs0 + end, end_rel=end, opening_ms=float(x['ws']),
                         mark_source='QP.build_path supported-end mark', tape_id=BE.tape_id(path['frames'], path['trades'], path['mark']))
            out_m = dict(day=day, m=m, path_info=info)
            for arm, en in (('disabled', False), ('guard', True)):
                B, rec, pol, obs = run_arm(path, ident, gm['edge'], en); out_m[arm] = rec
                arrs.update({f'{m}__{arm}__{k}': v for k, v in dict(obs.arrays(), **command_arrays(B)).items()})
            d_, bl = out_m['disabled'], b['ledger']
            out_m['control'] = dict(fingerprint=d_['fingerprint'] == bl['fingerprint'], fills=d_['fills'] == bl['fills'], value=d_['value'] == bl['value'],
                                    orders=d_['orders'] == bl['orders'], requests=d_['requests'] == bl['requests'])
            out_m['control_all'] = bool(all(out_m['control'].values())); recs.append(out_m); tim['markets'].append(dict(m=m, s=time.time() - tm)); G.check('market', time.time() - tm)
        b_ = LS.gz_bytes(recs); P['gz'].write_bytes(b_); np.savez_compressed(P['npz'], **arrs)
        LS.RN.atomic_json(P['rec'], dict(version=VERSION, day=day, pilot=bool(pilot), members=mans, sha256=dict(gz=hashlib.sha256(b_).hexdigest(), npz=C.sha(P['npz'])),
                                         manifest_sha256=C.sha(REPO / RECEIPT / 'manifest.json'), rule_sha256=rule_sha(), sources=sources(), timing=tim, elapsed_s=time.time() - t0,
                                         hwm_mib=LS.hwm(), asof=C.stamp()))
        return 'DAY_DONE'
    except LS._Latch as e:
        LS.RN.atomic_json(P['status'], dict(state='STOP_RESOURCE', day=day, latch=e.args[0], asof=C.stamp())); return 'STOP_RESOURCE'
    finally:
        mon.live = False


# ---------------------------------------------------------------------------------------------------------------------- selection control
def band(x, edges):
    if x is None or not np.isfinite(x):
        return 'missing'
    return int(np.searchsorted(np.asarray(edges, float), float(x), side='right'))


def inv_sign(x):
    if x is None or not np.isfinite(x):
        return 'missing'
    return -1 if x < -1e-9 else (1 if x > 1e-9 else 0)


def selection_table(S_, edges):
    """The ask live states with their selection flag, strata and window values (uncensored only for the statistic; censored counted)."""
    g = S_[(S_.kind == 'live') & (S_.side == 1)].copy(); e = edges[EDGE_KEY]
    g['selected'] = [M.bin_of(float(np.float32(v)) if v is not None and np.isfinite(v) else np.nan, e) == 'q5' for v in g.ctx_bid_reserved_shares.astype(float)]
    g['stratum'] = [json.dumps([d, int(m), band(r, REM_BANDS), band(a, AGE_BANDS), inv_sign(i)]) for d, m, r, a, i in
                    zip(g.day, g.m, g.fp_remaining.astype(float), g.ctx_generation_age_ms.astype(float), g.inv_known.astype(float))]
    g['uncensored'] = (~g.censored_1s.astype(bool)) & (g.win_mo1s.notna() | (g.win_n == 0))
    g['y'] = g.win_mo1s.fillna(0.).astype(float); g['valid_q'] = np.where(g.uncensored, g.win_q.astype(float), 0.)
    return g


def selection(S_, edges, draws=NULL_DRAWS, seed=SEED):
    """The matched-draw selection control (header) -> dict."""
    g = selection_table(S_, edges); u = g[g.uncensored]; sel = u[u.selected]
    J = float(sel.y.mean()) if len(sel) else None; Jq = float(sel.y.sum() / sel.valid_q.sum()) if len(sel) and sel.valid_q.sum() > 0 else None
    strata = {k: (int(v.selected.sum()), int(len(v))) for k, v in u.groupby('stratum')}
    movable = {k: (s, n) for k, (s, n) in strata.items() if 0 < s < n}; rng = np.random.default_rng(seed); null, nullq = [], []
    idx = {k: v.index.values for k, v in u.groupby('stratum')}
    for _ in range(int(draws)):
        pick = [i for k, (s, n) in strata.items() if s > 0 for i in rng.choice(idx[k], size=s, replace=False)]
        y = u.loc[pick, 'y']; q = u.loc[pick, 'valid_q']; null.append(float(y.mean())); nullq.append(float(y.sum() / q.sum()) if q.sum() > 0 else np.nan)
    null = np.asarray(null); nullq = np.asarray(nullq)
    thin = len(movable) < MIN_MOVABLE_STRATA or sum(s for s, n in movable.values()) < MIN_MOVABLE_ROWS
    ans = 'INCONCLUSIVE' if thin or J is None else ('SELECTION_BEYOND_EXPOSURE' if J < float(np.quantile(null, .05)) else 'NO_SELECTION_BEYOND_EXPOSURE')
    return dict(answer=ans, selected=int(len(sel)), states=int(len(u)), censored=int((~g.uncensored).sum()), selected_censored=int((g.selected & ~g.uncensored).sum()),
                J=J, J_per_valid_share=Jq, valid_window_q=float(sel.valid_q.sum()), total_window_q=float(g[g.selected].win_q.astype(float).sum()),
                excluded_window_q=float(g[g.selected & ~g.uncensored].win_q.astype(float).sum()), null_mean=float(null.mean()), null_p05=float(np.quantile(null, .05)),
                null_p95=float(np.quantile(null, .95)), percentile=float((null <= J).mean()) if J is not None else None, null_q_p05=float(np.nanquantile(nullq, .05)),
                strata=len(strata), movable_strata=len(movable), movable_selected=int(sum(s for s, n in movable.values())), degenerate_strata=int(sum(1 for s, n in strata.values() if s == n and s > 0)),
                draws=int(draws), seed=int(seed))


def censor_count(G_):
    """Finding 3: generations with fills whose lifetime +1 s is missing (0 <= count <= generations)."""
    c = int((G_.lifetime_mo1s.isna() & (G_.n_fills > 0)).sum()); C.require(0 <= c <= len(G_), 'a censor count outside [0, generations]'); return c


def run_selection(out=OUT):
    out = Path(out); p = out / 'selection.json'; C.require(frozen(), 'not the committed guard'); C.require(not p.exists(), 'the selection control exists (write-once)'); out.mkdir(parents=True, exist_ok=True)
    S_ = pd.read_csv(AU.OUT / 'states.csv', float_precision='round_trip', dtype={'day': str}); edges = json.loads((M.OUT / 'measure' / 'audit.json').read_text())['edges']
    G_ = pd.read_csv(AU.OUT / 'generations_v2.csv', float_precision='round_trip', dtype={'day': str}); res = selection(S_, edges); res['generation_censor_count'] = censor_count(G_)
    res['inputs'] = dict(states_sha256=C.sha(AU.OUT / 'states.csv'), generations_sha256=C.sha(AU.OUT / 'generations_v2.csv'), edges_sha256=C.sha(M.OUT / 'measure' / 'audit.json'))
    C.dump(p, res); print(json.dumps(res, indent=1, default=str))


# ---------------------------------------------------------------------------------------------------------------------- measurement
def arm_metrics(rec, a, end):
    """One arm of one market: values, markouts, the shared +1 s cutoff, side / rung splits, guard and removal counts, inventory."""
    mark = float(rec['mark']); od = {int(o['gid']): o for o in rec['orders']}; x = dict(value=float(rec['value']), fills=len(rec['fills']))
    sg = lambda s: 1. if int(s) == 0 else -1.; rows = []
    for f, mm in zip(rec['fills'], rec['fill_mids']):
        t, gid, s, q, p = float(f[0]), int(f[1]), int(f[2]), float(f[3]), float(f[4])
        rows.append(dict(t=t, side=s, rung=int(round(float(od[gid]['rung']))), q=q, end=sg(s) * q * (mark - p), edge=None if mm[0] is None else sg(s) * q * (mm[0] - p),
                         mo50=None if mm[2] is None else sg(s) * q * (mm[2] - p), mo100=None if mm[4] is None else sg(s) * q * (mm[4] - p), mo1s=None if mm[6] is None else sg(s) * q * (mm[6] - p)))
    F = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['t', 'side', 'rung', 'q', 'end', 'edge', 'mo50', 'mo100', 'mo1s'])
    cut = F[F.t <= end - 1000.]
    for nm, D_ in (('all', F), ('bid', F[F.side == 0]), ('ask', F[F.side == 1])):
        cz = cut[cut.side == (0 if nm == 'bid' else 1)] if nm != 'all' else cut; v1 = cz[cz.mo1s.notna()]
        x.update({f'{nm}_q': float(D_.q.sum()), f'{nm}_end': float(D_.end.sum()), f'{nm}_edge': float(D_.edge.dropna().sum()), f'{nm}_mo50': float(D_.mo50.dropna().sum()),
                  f'{nm}_mo100': float(D_.mo100.dropna().sum()), f'{nm}_cut_q': float(cz.q.sum()), f'{nm}_cut_mo1s': float(v1.mo1s.sum()), f'{nm}_cut_valid_q': float(v1.q.sum()),
                  f'{nm}_cut_censored': int(cz.mo1s.isna().sum()), f'{nm}_good_mo1s': float(v1.mo1s[v1.mo1s > 0].sum()), f'{nm}_bad_mo1s': float(v1.mo1s[v1.mo1s < 0].sum())})
    for r_ in (0, 1):
        D_ = F[F.rung == r_]; x[f'rung{r_}_q'] = float(D_.q.sum()); x[f'rung{r_}_end'] = float(D_.end.sum())
    gl = rec.get('guard_log', []); x.update(guard_pulls=sum(1 for z in gl if z[8] == 'guard_pull'), guard_withholds=sum(1 for z in gl if z[8] == 'guard_withhold'),
                                            reducing_suppressed=sum(1 for z in gl if z[6] > 1e-9))
    why = [o['why'] for o in rec['orders'] if o['req'] is not None]
    x.update(generations=len(rec['orders']), ask_generations=sum(1 for o in rec['orders'] if int(o['side']) == 1), structural_requests=sum(1 for w in why if w == 'structural'),
             policy_requests=sum(1 for w in why if w == 'policy'))
    pe = a['pose']; x.update(mean_abs_inv=float(np.mean(np.abs(pe))), peak_abs_inv=float(np.max(np.abs(pe))), share_abs_ge_S=float(np.mean(np.abs(pe) >= 100. - 1e-9)),
                             terminal_inv=float(rec['position']))
    return x


def measure(out=OUT):
    out = Path(out); md = out / 'measure'; C.require(not (md / 'audit.json').exists(), 'the measurement exists (write-once)'); md.mkdir(parents=True, exist_ok=True)
    gm = json.loads((REPO / RECEIPT / 'manifest.json').read_text()); rows = []; ctl = []
    for day in gm['days']:
        P = day_paths(day, False, out); rec = json.loads(P['rec'].read_text()); b = P['gz'].read_bytes()
        C.require(hashlib.sha256(b).hexdigest() == rec['sha256']['gz'] and C.sha(P['npz']) == rec['sha256']['npz'] and rec['rule_sha256'] == gm['rule_sha256'], f'{day}: identity')
        z = np.load(P['npz'])
        for r in LS.gz_rows(b):
            end = float(r['path_info']['end_rel']); ctl.append(dict(day=day, m=int(r['m']), **r['control'], control_all=r['control_all']))
            for arm in ('disabled', 'guard'):
                a = {k.split('__')[2]: z[k] for k in z.files if k.startswith(f"{int(r['m'])}__{arm}__")}
                du = AU.durations(dict(day=day, m=int(r['m']), ledger=r[arm]), {f"{day}|{int(r['m'])}__nva": a['nva']})
                rows.append(dict(day=day, m=int(r['m']), arm=arm, both_exact=du['both_exact'], ask_exact=du['ask_exact'], bid_exact=du['bid_exact'], **arm_metrics(r[arm], a, end)))
    R_ = pd.DataFrame(rows); CT = pd.DataFrame(ctl)
    pr = predicates(R_); R_.to_csv(md / 'arms.csv', index=False); CT.to_csv(md / 'control.csv', index=False)
    C.dump(md / 'audit.json', dict(version=VERSION, asof=C.stamp(), predicates=pr, markets=int(R_.m.nunique()), control_all=bool(CT.control_all.all()),
                                   outputs={n: C.sha(md / n) for n in ('arms.csv', 'control.csv')}, sources=sources(), rule_sha256=gm['rule_sha256']))
    print(json.dumps(pr, indent=1, default=str))


def predicates(R_, selection_res=None):
    """The declared readings (header)."""
    sel = selection_res or (json.loads((OUT / 'selection.json').read_text()) if (OUT / 'selection.json').exists() else None)
    g = R_[R_.arm == 'guard'].set_index(['day', 'm']); b = R_[R_.arm == 'disabled'].set_index(['day', 'm']); days = sorted(R_.day.unique()); out = dict(days=days)

    def cat(vals):
        return 'IMPROVES_EVERY_DAY' if all(v > 0 for v in vals) else 'WORSENS_EVERY_DAY' if all(v < 0 for v in vals) else 'MIXED'
    dEND = {d: float((g.loc[d].value - b.loc[d].value).sum()) for d in days}
    per = lambda X, d, side: float(X.loc[d][f'{side}_cut_mo1s'].sum() / X.loc[d][f'{side}_cut_valid_q'].sum()) if X.loc[d][f'{side}_cut_valid_q'].sum() > 0 else np.nan
    dps = {side: {d: per(g, d, side) - per(b, d, side) for d in days} for side in ('all', 'bid', 'ask')}
    dq = {d: float(g.loc[d].all_q.sum() - b.loc[d].all_q.sum()) for d in days}
    mk = (g.value - b.value)
    out.update(END_paired=dict(answer=cat(list(dEND.values())), by_day=dEND, markets_better=int((mk > 1e-9).sum()), markets_worse=int((mk < -1e-9).sum()), markets_equal=int((mk.abs() <= 1e-9).sum())),
               mo1s_per_share={side: dict(answer=cat([v for v in dps[side].values() if np.isfinite(v)]) if all(np.isfinite(v) for v in dps[side].values()) else 'MIXED', by_day=dps[side])
                               for side in ('all', 'bid', 'ask')},
               participation=dict(filled_q_delta_by_day=dq, both_exact_delta=float(g.both_exact.mean() - b.both_exact.mean()), ask_exact_delta=float(g.ask_exact.mean() - b.ask_exact.mean()),
                                  ask_q_delta=float(g.ask_q.sum() - b.ask_q.sum()), guard_pulls=int(g.guard_pulls.sum()), guard_withholds=int(g.guard_withholds.sum()),
                                  reducing_suppressed=int(g.reducing_suppressed.sum())),
               inventory=dict(mean_abs_delta=float(g.mean_abs_inv.mean() - b.mean_abs_inv.mean()), share_abs_ge_S_delta=float(g.share_abs_ge_S.mean() - b.share_abs_ge_S.mean()),
                              terminal_abs_delta=float(g.terminal_inv.abs().mean() - b.terminal_inv.abs().mean())),
               selection=None if sel is None else sel['answer'])
    better = (out['END_paired']['answer'] == 'IMPROVES_EVERY_DAY' and out['mo1s_per_share']['all']['answer'] == 'IMPROVES_EVERY_DAY' and out['selection'] == 'SELECTION_BEYOND_EXPOSURE')
    less = all(v < 0 for v in dq.values()) and out['mo1s_per_share']['all']['answer'] != 'IMPROVES_EVERY_DAY'
    out['interpretation'] = 'better_selection' if better else ('quoting_less' if less else 'mixed')
    out['note'] = 'one adaptive consumed-day candidate on 19 markets; no profitability, significance, validation or promotion claim'
    return out


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'selftest':
        import pf_ladder_guard_selftest as ST
        ST.main(sys.modules[__name__])
    elif cmd == 'manifest':
        print(json.dumps(build_manifest(), indent=1)[:1500])
    elif cmd in ('pilot', 'run'):
        print(run_day(sys.argv[2], pilot=(cmd == 'pilot')))
    elif cmd == 'selection':
        run_selection()
    elif cmd == 'measure':
        measure()
    else:
        sys.exit('usage: pf_ladder_guard.py selftest | manifest | pilot <day> | run <day> | selection | measure')
