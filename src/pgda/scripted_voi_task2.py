# 21-scripted_voi_task2.py — Task-2 pilot gate, Stage 1: scripted value of information
# Physically-Grounded Diffusion Augmentation (PGDA)
#
# NO RL. A scripted pusher answers: on task 2, how much is knowing the material
# worth, and can a policy without the cue get the same by probing?
#
# PER EPISODE (coupled_no_cue physics, seeded, natural material mix)
#   1. Reset; park the pusher on the push line, 1.5 cm (measured) behind the
#      object on the side away from the goal.
#   2. Direct pushes: for every commanded speed v in V_GRID and n = 1..N_MAX,
#      push along the goal direction for the onset step plus n-1 more steps,
#      end the window (the env's own retract + coast to rest) and record where
#      the object comes to rest.
#   3. Probes: strike at a gentle probe speed, read the object's speed after
#      the onset step (feature: ice keeps sliding, rust stops), then stop or
#      continue with (v2, j), j = 1..N_MAX-1.
#   Every rollout is a FRESH episode: seeded reset, then the recorded approach
#   actions replayed, then its own push. This is bit-exact (the self-test, [H],
#   checks it). PyBullet snapshots are not used: restoring one does not
#   reproduce a fresh run once objects touch (up to 1 cm on ice).
#   About 118 rollouts per episode (probe rollouts that equal a direct push
#   are reused).
#   Rest positions are stored relative to the start in the push frame (s along
#   the goal direction, l lateral), so success can be evaluated afterwards for
#   ANY goal distance D on the push line and ANY tolerance delta.
#
# AFTER THE RUN (analysis, also runnable alone with ANALYSE_ONLY = True)
#   For every design (K in 0..N_MAX-1, goal band of D, delta), averaging over D
#   in the band; all strategies may depend on D because the policy sees the goal:
#     S_single : best (v, n<=K+1) per D, same for all materials (no cue)
#     S_probe  : best probe speed, with the continuation chosen per D and per
#                bin of the probe feature (2 or 3 bins, thresholds optimised
#                in-sample, which overstates S_probe and makes the gate harder)
#     S_cat    : best strategy per D and per material (perfect category cue);
#                it may use every strategy the probing policy can, so the cue
#                is never compared against a larger strategy set
#     S_oracle : best strategy per D and per episode (exact physics)
#   Gate (fixed before running): S_cat - max(S_single, S_probe) >= 15 pp and
#   S_cat >= 60 % on every material.
#   Pre-registered choice among passing designs: the largest worst-case margin
#   min(gap - 15, min_c S_cat_c - 60); ties -> delta = 5 cm, then smaller K.
#   The chosen design also gets a bootstrap 95% CI of its gap (episodes
#   resampled, all strategies re-optimised in each resample).
#
# HOW TO RUN
#   1. Put this file next to pushing_env.py and pushing_env_task2.py.
#   2. DRY_RUN = True: builds the env, times one episode, prints the projected
#      run time. Nothing is saved.
#   3. DRY_RUN = False, QUICK = True for a first look (120 episodes), then
#      QUICK = False for the gate (960 episodes, about 300 per material).
#      Results are saved every 10 episodes; re-running resumes. QUICK and full
#      runs use the same episodes (the quick run is the first 120 of the full).
#   Upload task2_voi_results.json.

import os
import sys
import json
import time
import random
import itertools
import multiprocessing as mp

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
DRY_RUN       = False
QUICK         = False         # True: first 120 episodes only
ANALYSE_ONLY  = False          # True: skip simulation, re-analyse the JSON

N_EPISODES       = 960         # natural material mix: about 300 per material
N_EPISODES_QUICK = 120
PILOT_SEED_BASE  = 2_000_000   # disjoint from evaluation seeds 10000+1000*s+k
MODE             = "coupled_no_cue"
N_WORKERS        = max(1, (os.cpu_count() or 2) - 1)
RESULTS_FILE     = "task2_voi_results.json"
ENV_CONFIG       = {}          # task2_config overrides; leave empty for the gate

V_GRID   = [0.05, 0.08, 0.12, 0.16, 0.20, 0.30, 0.45, 0.60]   # commanded m/s
N_MAX    = 6                   # push steps n = 1..N_MAX  (K up to N_MAX-1)
PROBE_V  = [0.08, 0.16]        # probe strike speeds, m/s
GAP      = 0.015               # pre-push gap between pusher and object, m
GUARD    = 40                  # max steps to reach contact during a push

D_GRID   = [round(0.04 + 0.01 * i, 2) for i in range(21)]        # 4..24 cm
BANDS    = [(0.06, 0.10), (0.08, 0.12), (0.10, 0.14), (0.12, 0.16),
            (0.14, 0.18), (0.16, 0.20), (0.18, 0.22), (0.20, 0.24),
            (0.08, 0.15), (0.10, 0.20)]
DELTAS   = [0.03, 0.04, 0.05]
K_VALUES = list(range(N_MAX))
SPLIT_Q  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

GATE_GAP_PP      = 15.0
GATE_MIN_CAT_PCT = 60.0
REFERENCE_DESIGN = {"K": 3, "band": (0.08, 0.15), "delta": 0.05}   # brief's start
N_BOOT           = 200

CATEGORIES = ["Heavy Rust", "Wet Ice", "Coarse Sandpaper"]
STOP_BEYOND = max(D_GRID) + max(DELTAS) + 0.05   # overshoot beyond any goal: stop early
BIG_WINDOW = 10 ** 6

CONFIG = {"mode": MODE, "seed_base": PILOT_SEED_BASE, "v_grid": V_GRID, "n_max": N_MAX,
          "probe_v": PROBE_V, "gap": GAP, "env_config": ENV_CONFIG,
          "rollouts": "seeded_replay", "approach": "closest_point_gap"}


def key(*parts):
    return "|".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in parts)


# ── Simulation (runs in worker processes) ─────────────────────────────────────
_ENV = None


def _init_worker(mode, env_config):
    global _ENV
    import contextlib
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        from pushing_env_task2 import UR3ePushingTask2Env
        _ENV = UR3ePushingTask2Env(render_mode=None, randomization_mode=mode,
                                   reward_weights={"success_threshold": 0.05},
                                   task2_config=env_config)


def _pack(env, info):
    """Rest position in the push frame + flags + coast time + peak speed."""
    pos, _ = env._obj_pose()
    rel = pos[:2] - env._obj_start[:2]
    u = env._u_goal
    s, l = float(rel @ u), float(rel @ np.array([-u[1], u[0]]))
    flags = int(info["at_rest"] > 0.5) | (int(info["early_exit"] > 0.5) << 1)
    return [round(s, 5), round(l, 5), flags, round(info["coast_time"], 3),
            round(info["peak_speed"], 4)]


def _push_until_onset(env, a):
    from pushing_env_task2 import APPROACH, PUSH
    n0 = env._n_steps
    while env._phase == APPROACH and env._n_steps - n0 < GUARD:
        env.step(a)
    return env._phase == PUSH


class ReplayError(RuntimeError):
    pass


def _fresh(env, seed):
    np.random.seed(seed)
    random.seed(seed)
    env.reset()
    env.set_push_window(BIG_WINDOW)


def _onset_feature(env):
    lin, _ = env._obj_vel()
    pos, _ = env._obj_pose()
    return [round(float(np.hypot(lin[0], lin[1])), 5),
            round(float((pos[:2] - env._obj_start[:2]) @ env._u_goal), 5)]


def run_episode(k):
    from pushing_env_task2 import scripted_approach, push_action, CONTROL_DT, APPROACH
    env = _ENV
    t0 = time.time()
    seed = PILOT_SEED_BASE + k
    _fresh(env, seed)
    rec = env.episode_record()
    out = {"k": k, "seed": seed, "category": rec["category"], "object": rec["object"],
           "mass": rec["mass"], "mu": rec["mu"], "izz_scale": rec["izz_scale"],
           "goal_theta_deg": rec["goal_theta_deg"], "start": rec["start"]}
    appr = scripted_approach(env, gap=GAP)
    actions = appr.pop("actions")
    out["approach"] = appr
    if not appr["ok"]:
        out["valid"] = False
        out["t_sec"] = round(time.time() - t0, 2)
        return out
    ee_ref = env._ee_actual().copy()
    state = {"at_pre_push": True, "rollouts": 0}

    def prepare():
        """Bring the env to the pre-push point of this episode, exactly."""
        state["rollouts"] += 1
        if state["at_pre_push"]:
            state["at_pre_push"] = False
            return
        _fresh(env, seed)
        for act in actions:
            env.step(np.asarray(act))
        if env._phase != APPROACH or not np.array_equal(env._ee_actual(), ee_ref):
            raise ReplayError(f"episode {k}: the seeded replay did not reproduce the approach "
                              f"exactly; results would not be exact (see self-test [H])")

    def check_feature(v, f):
        if feature.setdefault(key(v), f) != f:
            raise ReplayError(f"episode {k}: onset differs between replays at v={v}")

    # Direct pushes
    direct, ee_speed, feature = {}, {}, {}
    for v in V_GRID:
        a = push_action(env, v)
        for n in range(1, N_MAX + 1):
            prepare()
            if not _push_until_onset(env, a):
                break                              # never touched the object at this v
            check_feature(v, _onset_feature(env))
            disp = []
            for _ in range(n - 1):
                e0 = env._ee_actual()[:2]
                env.step(a)
                disp.append(float((env._ee_actual()[:2] - e0) @ env._u_goal))
            *_, info = env.finish_window(stop_beyond=STOP_BEYOND)
            direct[key(v, n)] = _pack(env, info)
            if n == N_MAX and disp:
                ee_speed[key(v)] = round(float(np.median(disp)) / CONTROL_DT, 4)

    # Probes: strike, read the feature, then stop or continue with (v2, j).
    # A probe that stops, or continues at its own speed, IS a direct push.
    probe = {}
    for ap in PROBE_V:
        a1 = push_action(env, ap)
        for v2, j in [("stop", 0)] + [(v2, j) for v2 in V_GRID for j in range(1, N_MAX)]:
            pk = key(ap, "stop", 0) if v2 == "stop" else key(ap, v2, j)
            same = key(ap, 1) if v2 == "stop" else (key(ap, j + 1) if v2 == ap else None)
            if same is not None and same in direct:
                probe[pk] = direct[same]
                continue
            prepare()
            if not _push_until_onset(env, a1):
                break
            check_feature(ap, _onset_feature(env))
            a2 = push_action(env, v2) if v2 != "stop" else None
            for _ in range(j):
                env.step(a2)
            *_, info = env.finish_window(stop_beyond=STOP_BEYOND)
            probe[pk] = _pack(env, info)
    out.update({"valid": True, "direct": direct, "ee_speed": ee_speed, "probe": probe,
                "feature": feature, "rollouts": state["rollouts"],
                "t_sec": round(time.time() - t0, 2)})
    return out


# ── Analysis ──────────────────────────────────────────────────────────────────
# Every strategy is an "option": a direct push (v, n), or a probe strike at ap
# followed by a stop or a continuation (v2, j). j = push steps after the onset
# step, so under window K an option is allowed if j <= K. The cue-holder
# (S_cat, S_oracle) may use every option, like the probing no-cue policy.
PROBE_OPTS = [("stop", 0)] + [(v2, j) for v2 in V_GRID for j in range(1, N_MAX)]
OPTIONS = ([("direct", None, v, n - 1) for v in V_GRID for n in range(1, N_MAX + 1)]
           + [("probe", ap, v2, j) for ap in PROBE_V for (v2, j) in PROBE_OPTS])
OPT_J = np.array([o[3] for o in OPTIONS])
OPT_DIRECT = np.array([o[0] == "direct" for o in OPTIONS])
OPT_PROBE = [np.array([o[0] == "probe" and o[1] == ap for o in OPTIONS]) for ap in PROBE_V]


def option_label(o):
    if o[0] == "direct":
        return f"push {o[2]:.2f} m/s for {o[3] + 1} step(s)"
    if o[2] == "stop":
        return f"probe {o[1]:.2f} m/s, then stop"
    return f"probe {o[1]:.2f} m/s, then {o[2]:.2f} m/s for {o[3]} step(s)"


def _option_key(o):
    if o[0] == "direct":
        return "direct", key(o[2], o[3] + 1)
    return "probe", (key(o[1], "stop", 0) if o[2] == "stop" else key(o[1], o[2], o[3]))


class Data:
    def __init__(self, res):
        eps = sorted([e for e in res["episodes"].values() if e.get("valid")],
                     key=lambda e: e["k"])
        self.eps = eps
        E, O = len(eps), len(OPTIONS)
        self.cat = np.array([e["category"] for e in eps])
        self.S, self.L = np.full((E, O), np.nan), np.full((E, O), np.nan)
        self.OK = np.zeros((E, O), bool)
        self.F = np.full((E, len(PROBE_V)), np.nan)
        for ei, e in enumerate(eps):
            for oi, o in enumerate(OPTIONS):
                fam, kk = _option_key(o)
                r = e[fam].get(kk)
                if r:
                    self.S[ei, oi], self.L[ei, oi] = r[0], r[1]
                    self.OK[ei, oi] = bool(r[2] & 1) and not bool(r[2] & 2)
            for ai, ap in enumerate(PROBE_V):
                f = e["feature"].get(key(ap))
                if f:
                    self.F[ei, ai] = f[0]
        self._cache = {}

    def succ(self, delta, D):
        k = (delta, D)
        if k not in self._cache:
            with np.errstate(invalid="ignore"):
                self._cache[k] = self.OK & (((self.S - D) ** 2 + self.L ** 2) < delta ** 2)
        return self._cache[k]


def _probe_value(F, P):
    """Best no-cue policy after a probe: 1, 2 or 3 bins of the probe feature F,
    with the best continuation per bin. P = (E, options) success. Episodes in
    which the probe never touched the object form their own bin."""
    E = len(F)
    if E == 0 or P.shape[1] == 0:
        return 0.0
    P = P.astype(float)
    nanbin = np.isnan(F)
    base = P[nanbin].mean(0).max() * nanbin.sum() if nanbin.any() else 0.0
    f, Pf = F[~nanbin], P[~nanbin]
    if len(f) == 0:
        return base / E
    order = np.argsort(f, kind="stable")
    fs, Ps = f[order], Pf[order]
    cum = np.vstack([np.zeros((1, Ps.shape[1])), np.cumsum(Ps, 0)])

    def seg(a, b):
        return (cum[b] - cum[a]).max()
    best = seg(0, len(fs))                                     # one bin
    cuts = sorted(set(int(np.searchsorted(fs, q, side="right"))
                      for q in np.quantile(f, SPLIT_Q)))
    cuts = [c for c in cuts if 0 < c < len(fs)]
    for c in cuts:
        best = max(best, seg(0, c) + seg(c, len(fs)))
    for c1, c2 in itertools.combinations(cuts, 2):
        best = max(best, seg(0, c1) + seg(c1, c2) + seg(c2, len(fs)))
    return (best + base) / E


def design_metrics(data, delta, K, band, ix=None):
    """Mean over D in the band of S_single, S_probe, S_cat (overall and per
    material) and S_oracle, in %. ix: episode indices (bootstrap)."""
    ix = np.arange(len(data.eps)) if ix is None else ix
    cat = data.cat[ix]
    allowed = OPT_J <= K
    Ds = [D for D in D_GRID if band[0] - 1e-9 <= D <= band[1] + 1e-9]
    acc = {"single": [], "probe": [], "cat": [], "oracle": [],
           **{f"cat_{c}": [] for c in CATEGORIES}}
    p_c = {c: float(np.mean(cat == c)) for c in CATEGORIES}
    for D in Ds:
        s = data.succ(delta, D)[ix]
        sa = s[:, allowed]
        acc["single"].append(s[:, allowed & OPT_DIRECT].mean(0).max())
        acc["oracle"].append(sa.any(1).mean())
        sc = 0.0
        for c in CATEGORIES:
            m = cat == c
            v = sa[m].mean(0).max() if m.any() else 0.0
            acc[f"cat_{c}"].append(v)
            sc += p_c[c] * v
        acc["cat"].append(sc)
        acc["probe"].append(max(_probe_value(data.F[ix, ai], s[:, allowed & OPT_PROBE[ai]])
                                for ai in range(len(PROBE_V))))
    out = {k: 100 * float(np.mean(v)) for k, v in acc.items()}
    out["gap"] = out["cat"] - max(out["single"], out["probe"])
    out["min_cat"] = min(out[f"cat_{c}"] for c in CATEGORIES)
    out["passes"] = bool(out["gap"] >= GATE_GAP_PP and out["min_cat"] >= GATE_MIN_CAT_PCT)
    out["score"] = min(out["gap"] - GATE_GAP_PP, out["min_cat"] - GATE_MIN_CAT_PCT)
    return out


def cross_matrix(data, delta, K, band):
    """Rows: the strategy that is best for material r (per D); columns: its
    success on material c. Also returns the best strategy per material and D."""
    Ds = [D for D in D_GRID if band[0] - 1e-9 <= D <= band[1] + 1e-9]
    allowed = np.where(OPT_J <= K)[0]
    M, best = np.zeros((3, 3)), {}
    for D in Ds:
        s = data.succ(delta, D)[:, allowed]
        for ri, r in enumerate(CATEGORIES):
            oi = int(np.argmax(s[data.cat == r].mean(0)))
            best.setdefault(r, []).append(f"D={D:.2f}: {option_label(OPTIONS[allowed[oi]])}")
            for ci, c in enumerate(CATEGORIES):
                M[ri, ci] += s[data.cat == c][:, oi].mean() / len(Ds)
    return (100 * M).round(1).tolist(), best


def analyse(res):
    data = Data(res)
    E = len(data.eps)
    n_all = len(res["episodes"])
    counts = {c: int(np.sum(data.cat == c)) for c in CATEGORIES}
    designs = []
    for delta, K, band in itertools.product(DELTAS, K_VALUES, BANDS):
        m = design_metrics(data, delta, K, band)
        designs.append({"delta": delta, "K": K, "band": list(band), **m})
    passing = [d for d in designs if d["passes"]]
    passing.sort(key=lambda d: (-d["score"], d["delta"] != 0.05, d["K"]))
    chosen = passing[0] if passing else None
    ref = next(d for d in designs if d["K"] == REFERENCE_DESIGN["K"]
               and tuple(d["band"]) == REFERENCE_DESIGN["band"]
               and d["delta"] == REFERENCE_DESIGN["delta"])
    out = {"n_episodes": n_all, "n_valid": E, "counts": counts, "designs": designs,
           "n_passing": len(passing), "chosen": chosen, "reference": ref}
    for name, dsg in (("chosen", chosen), ("reference", ref)):
        if dsg is None:
            continue
        M, best = cross_matrix(data, dsg["delta"], dsg["K"], tuple(dsg["band"]))
        dsg["cross_matrix"], dsg["best_strategy_per_D"] = M, best
    if chosen is not None:
        rng = np.random.default_rng(0)
        gaps = []
        for _ in range(N_BOOT):
            ix = rng.integers(0, E, E)
            gaps.append(design_metrics(data, chosen["delta"], chosen["K"],
                                       tuple(chosen["band"]), ix)["gap"])
        chosen["gap_boot_ci95"] = [float(np.percentile(gaps, 2.5)),
                                   float(np.percentile(gaps, 97.5))]
    # Achieved pusher speed and post-window object speed per commanded v
    spd = {}
    for v in V_GRID:
        ee = [e["ee_speed"].get(key(v)) for e in data.eps if e["ee_speed"].get(key(v)) is not None]
        pk = {c: [e["direct"][key(v, N_MAX)][4] for e in data.eps
                  if e["category"] == c and key(v, N_MAX) in e["direct"]] for c in CATEGORIES}
        spd[key(v)] = {"ee_speed_median": float(np.median(ee)) if ee else None,
                       **{f"peak_obj_speed_{c}": float(np.median(x)) if x else None
                          for c, x in pk.items()}}
    out["speeds"] = spd
    out["approach_failures"] = n_all - E
    out["early_exit_rate"] = float(np.mean([(r[2] & 2) > 0 for e in data.eps
                                            for r in e["direct"].values()])) if E else None
    return out


def print_summary(a):
    print(f"\n{'#'*96}\n  TASK-2 SCRIPTED VALUE OF INFORMATION  ({a['n_valid']} valid of "
          f"{a['n_episodes']} episodes; {a['counts']})\n{'#'*96}")
    print("\n  Commanded push speed -> achieved pusher speed and object peak speed after the window (medians, m/s)")
    for v, s in a["speeds"].items():
        print(f"    v = {v} : pusher {s['ee_speed_median'] or float('nan'):.3f}  | object "
              + "  ".join(f"{c.split()[-1]} {s[f'peak_obj_speed_{c}'] or float('nan'):.3f}" for c in CATEGORIES))
    hdr = (f"\n  {'K':>2} {'band (cm)':>11} {'delta':>5} | {'single':>6} {'probe':>6} {'cat':>6} "
           f"{'oracle':>6} | {'Rust':>5} {'Ice':>5} {'Sand':>5} | {'gap':>6}  pass")

    def row(d):
        return (f"  {d['K']:2d} {d['band'][0]*100:5.0f}-{d['band'][1]*100:<5.0f} {d['delta']*100:5.0f} | "
                f"{d['single']:6.1f} {d['probe']:6.1f} {d['cat']:6.1f} {d['oracle']:6.1f} | "
                + " ".join(f"{d[f'cat_{c}']:5.1f}" for c in CATEGORIES)
                + f" | {d['gap']:+6.1f}  {'PASS' if d['passes'] else '-'}")
    print(f"\n  Reference design (brief): K={REFERENCE_DESIGN['K']}, D {REFERENCE_DESIGN['band']}, "
          f"delta {REFERENCE_DESIGN['delta']}" + hdr)
    print(row(a["reference"]))
    top = sorted(a["designs"], key=lambda d: -d["score"])[:12]
    print(f"\n  Designs closest to the gate (sorted by worst-case margin; "
          f"{a['n_passing']} of {len(a['designs'])} pass)" + hdr)
    for d in top:
        print(row(d))
    c = a["chosen"]
    if c is None:
        print("\n  NO DESIGN PASSES THE GATE. Do not train PGDA on Design A; see the reply for the fallback.")
    else:
        print(f"\n  CHOSEN (pre-registered rule): K = {c['K']}, D = {c['band'][0]*100:.0f}-{c['band'][1]*100:.0f} cm, "
              f"delta = {c['delta']*100:.0f} cm")
        print(f"    gap {c['gap']:+.1f} pp, bootstrap 95% CI [{c['gap_boot_ci95'][0]:+.1f}, {c['gap_boot_ci95'][1]:+.1f}]")
        print("    strategy best for row material, applied to column material (success %):")
        print("      " + " " * 18 + "".join(f"{x:>18s}" for x in CATEGORIES))
        for r, vals in zip(CATEGORIES, c["cross_matrix"]):
            print(f"      {r:18s}" + "".join(f"{v:18.1f}" for v in vals))
        print("    best strategy per material at the band's middle distance:")
        for r in CATEGORIES:
            lst = c["best_strategy_per_D"][r]
            print(f"      {r:18s} {lst[len(lst) // 2]}")
    print(f"\n  Approach failures: {a['approach_failures']}; early-exit rate (far overshoots): "
          f"{a['early_exit_rate']*100 if a['early_exit_rate'] is not None else float('nan'):.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"config": CONFIG, "episodes": {}}


def save_results(res):
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, RESULTS_FILE)


if __name__ == "__main__":
    n_target = N_EPISODES_QUICK if QUICK else N_EPISODES
    print(f"{'='*96}\n  Task-2 scripted VoI | DRY_RUN={DRY_RUN} QUICK={QUICK} | {n_target} episodes | "
          f"{N_WORKERS} workers\n{'='*96}")
    res = load_results()
    if ANALYSE_ONLY:
        a = analyse(res)
        res["analysis"] = a
        save_results(res)
        print_summary(a)
        sys.exit(0)

    if res.get("config") != json.loads(json.dumps(CONFIG)) and res["episodes"]:
        raise SystemExit(f"{RESULTS_FILE} was made with a different configuration; "
                         f"rename it or restore the old settings.")

    # Check the env and time one episode in this process
    _init_worker(MODE, ENV_CONFIG)
    t0 = time.time()
    first = run_episode(0)
    dt = time.time() - t0
    ap = first["approach"]
    print(f"  env mode '{MODE}': OK | episode 0: {first['category']}, valid={first['valid']}, "
          f"{dt:.1f} s for {first.get('rollouts', 0)} rollouts")
    print(f"  approach: {ap.get('steps')} steps, gap {ap.get('gap', float('nan')) * 100:.2f} cm, "
          f"{ap.get('lateral_offset', float('nan')) * 1000:+.1f} mm off the push line")
    if first["valid"]:
        print("  achieved pusher speed per commanded v (m/s): "
              + ", ".join(f"{k}->{v}" for k, v in first["ee_speed"].items()))
    eta = dt * n_target / max(1, N_WORKERS)
    print(f"  projected time for {n_target} episodes on {N_WORKERS} workers: about {eta/60:.0f} min "
          f"(rough: episodes differ, and laptop cores do not all run at full speed)")
    if DRY_RUN:
        print("\n  DRY_RUN = True: nothing saved. Set DRY_RUN = False to run.")
        sys.exit(0)

    res["episodes"].setdefault("0", first)
    pending = [k for k in range(n_target) if str(k) not in res["episodes"]]
    print(f"  {len(res['episodes'])} episodes already done, {len(pending)} to go")
    t_start = time.time()
    if pending:
        ctx = mp.get_context("spawn")
        with ctx.Pool(N_WORKERS, initializer=_init_worker, initargs=(MODE, ENV_CONFIG)) as pool:
            for i, out in enumerate(pool.imap_unordered(run_episode, pending), 1):
                res["episodes"][str(out["k"])] = out
                if i % 10 == 0 or i == len(pending):
                    save_results(res)
                    el = time.time() - t_start
                    print(f"    {i}/{len(pending)} done, {el/60:.1f} min elapsed, "
                          f"~{el/i*(len(pending)-i)/60:.0f} min left")
    save_results(res)
    sub = {"config": res["config"],
           "episodes": {k: v for k, v in res["episodes"].items() if int(k) < n_target}}
    a = analyse(sub)
    res["analysis"] = a
    save_results(res)
    print_summary(a)
    print(f"\n  Saved -> {RESULTS_FILE}")