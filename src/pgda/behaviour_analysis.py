# 20-behaviour_analysis.py — What does the policy DO with the material code?
# Physically-Grounded Diffusion Augmentation (PGDA)
#
# DESIGN (no training; same seeded episodes as 13-eval_coupled_and_category.py)
#   PGDA is run on the coupled dynamics under four codes in dims 9-10:
#     encoder          : as trained (the encoder's prediction)
#     code_Heavy Rust  : the Heavy Rust code written into every episode
#     code_Wet Ice     : the Wet Ice code written into every episode
#     code_Coarse Sandpaper : the Coarse Sandpaper code in every episode
#   The true physics never changes, so the three forced codes give a clean
#   code x true-material matrix: the diagonal is PGDA with a PERFECT encoder,
#   the off-diagonal cells are mismatches. The No-Cue agent (E) is the
#   reference: it cannot condition on a code, so any difference between
#   materials in its behaviour comes from the physics alone.
#
# WHAT IS MEASURED (per episode, from the observation and the action)
#   A "push step" is a step after which the object moves (speed > V_MOVE).
#   - push_effort  : mean commanded end-effector displacement on push steps,
#                    in cm per control step (the action is a displacement in
#                    metres, bounded to +/-0.05 m per axis; max 7.07 cm/step)
#   - strike_offset: mean perpendicular distance (cm) between the push line and
#                    the object centre on push steps (off-centre pushing)
#   - spin         : mean |object angular velocity| on push steps (rad/s)
#   - bouts        : number of separate pushing phases (re-approaches)
#   - yaw_change   : total |change in object yaw| over the episode (deg)
#   - success, steps to success, final distance to goal
#
# OBSERVATION LAYOUT (verified in pushing_env.py, _get_obs_from_components):
#   dims 0-2 EE xyz, 3-4 EE minus object (xy), 5-6 object velocity (xy),
#   7 object spin, 8 object yaw, 9-10 physics code;
#   achieved_goal = object xy, desired_goal = goal xy.
#   Action: Box(-0.05, 0.05, (2,)) = EE displacement in metres per step.
#
# Resumable: results saved after every (condition, seed).

import os
import sys
import glob
import json
import random
import numpy as np

# ── numpy 2.x checkpoint compatibility (same shim as 15 / 17) ────────────────
if int(np.__version__.split(".")[0]) < 2:
    import numpy.core
    import numpy.core.numeric
    import numpy.core.multiarray
    try:
        import numpy.core._multiarray_umath as _mau
    except ImportError:
        _mau = None
    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    if _mau is not None:
        sys.modules.setdefault("numpy._core._multiarray_umath", _mau)
    if not hasattr(np, "_core"):
        np._core = numpy.core
    import numpy.random._pickle as _np_random_pickle
    _orig_ctor = _np_random_pickle.__bit_generator_ctor

    def _patched_ctor(bit_generator_name="MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return _orig_ctor(bit_generator_name)

    _np_random_pickle.__bit_generator_ctor = _patched_ctor

from stable_baselines3 import SAC
from gymnasium.wrappers import TimeLimit
from pushing_env import UR3ePushingEnv

# ── Configuration ─────────────────────────────────────────────────────────────
DRY_RUN = False

SEEDS             = [42, 123, 456, 789, 1011]
NUM_EPISODES      = 100
EVAL_THRESHOLD    = 0.05
EPISODE_SEED_BASE = 10_000
RESULTS_FILE      = "behaviour_results.json"
ENCODER_PATH      = ("/Users/mohammedalfandi/Documents/Robotics and AI /"
                     "MSc Project /Codes-2026/texture_encoder.pt")
V_MOVE            = 0.02          # m/s: object counts as moving above this
CODE_DIMS         = [9, 10]

CATEGORIES = ["Heavy Rust", "Wet Ice", "Coarse Sandpaper"]
CODES = {                          # same values as 17-baseline_label_check.py
    "Coarse Sandpaper": [0.400, 0.832],
    "Heavy Rust":       [0.800, 0.597],
    "Wet Ice":          [0.200, 0.013],
}

RUN_IDS = {
    "experimental": {42: "sgl94py3", 123: "1ktap4xi", 456: "lqckfjem",
                     789: "n8cfyu4x", 1011: "289a4luz"},
    "coupled_no_cue": {42: "lt56b7lo", 123: "t72zdfyj", 456: "azkfw077",
                       789: "k2nomi95", 1011: "7djx8mih"},
}

# condition -> (checkpoint type, env mode, needs encoder, forced code or None)
CONDITIONS = {
    "PGDA encoder":          ("experimental", "experimental", True, None),
    "PGDA code=Heavy Rust":  ("experimental", "experimental", True, "Heavy Rust"),
    "PGDA code=Wet Ice":     ("experimental", "experimental", True, "Wet Ice"),
    "PGDA code=Sandpaper":   ("experimental", "experimental", True, "Coarse Sandpaper"),
    "No-Cue (E)":            ("coupled_no_cue", "coupled_no_cue", False, None),
}
METRICS = ["push_effort", "strike_offset", "spin", "bouts", "yaw_change", "steps_to_success"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def ckpt_path(ck, seed):
    rid = RUN_IDS.get(ck, {}).get(seed, "")
    if rid:
        return f"models/{rid}/final_model_{ck}_seed{seed}_v2.zip"
    c = glob.glob(f"models/*/final_model_{ck}_seed{seed}_v2.zip")
    return max(c, key=os.path.getctime) if c else None


def make_env(mode, needs_encoder):
    env = UR3ePushingEnv(render_mode=None, randomization_mode=mode,
                         reward_weights={"success_threshold": EVAL_THRESHOLD},
                         encoder_path=ENCODER_PATH if needs_encoder else None)
    env.success_threshold = EVAL_THRESHOLD
    return TimeLimit(env, max_episode_steps=200)


def load_model(path, env):
    return SAC.load(path, env=env, device="cpu", custom_objects={
        "action_space": env.action_space,
        "observation_space": env.observation_space,
        "lr_schedule": lambda _: 0.0})


def with_code(obs, code):
    if code is None:
        return obs
    out = {k: v.copy() for k, v in obs.items()}
    out["observation"][CODE_DIMS] = np.asarray(code, dtype=out["observation"].dtype)
    return out


def episode_metrics(steps, success, steps_to_suc, final_dist):
    """steps: list of (obs_before, action, obs_after)."""
    effort, offset, spin, moving = [], [], [], []
    yaws = [float(steps[0][0]["observation"][8])] if steps else []
    for ob, a, oa in steps:
        o, o2 = ob["observation"], oa["observation"]
        yaws.append(float(o2[8]))
        is_push = float(np.hypot(o2[5], o2[6])) > V_MOVE
        moving.append(is_push)
        if not is_push:
            continue
        a = np.asarray(a, dtype=float).reshape(-1)[:2]
        r = np.asarray(ob["achieved_goal"], float)[:2] - np.asarray(o[:2], float)  # EE -> object
        na = np.linalg.norm(a)
        effort.append(na * 100.0)                                   # cm per step
        if na > 1e-6:
            offset.append(abs(a[0] * r[1] - a[1] * r[0]) / na * 100.0)   # cm
        spin.append(abs(float(o2[7])))
    bouts = sum(1 for i, m in enumerate(moving) if m and (i == 0 or not moving[i - 1]))
    yaw = np.unwrap(np.array(yaws)) if len(yaws) > 1 else np.zeros(1)
    nan = float("nan")
    return {
        "success": int(success),
        "steps_to_success": steps_to_suc if success else nan,
        "final_distance_cm": final_dist * 100.0,
        "push_steps": int(sum(moving)),
        "push_effort": float(np.mean(effort)) if effort else nan,
        "strike_offset": float(np.mean(offset)) if offset else nan,
        "spin": float(np.mean(spin)) if spin else nan,
        "bouts": int(bouts),
        "yaw_change": float(np.degrees(np.sum(np.abs(np.diff(yaw))))),
    }


def run_condition(model, env, seed, forced):
    records = []
    for ep in range(NUM_EPISODES):
        ep_seed = EPISODE_SEED_BASE + 1000 * seed + ep
        np.random.seed(ep_seed)
        random.seed(ep_seed)
        obs, _ = env.reset()
        cat = getattr(env.unwrapped, "_current_category", None) or "unknown"
        code = CODES[forced] if forced else None
        done = truncated = False
        steps, n, steps_to_suc, success = [], 0, None, False
        while not (done or truncated) and n < 200:
            action, _ = model.predict(with_code(obs, code), deterministic=True)
            obs2, _, done, truncated, info = env.step(action)
            steps.append(({k: np.array(v, copy=True) for k, v in obs.items()},
                          np.array(action, copy=True),
                          {k: np.array(v, copy=True) for k, v in obs2.items()}))
            obs = obs2
            n += 1
            if info.get("is_success", False) and steps_to_suc is None:
                steps_to_suc, success = n, True
        genuine = bool(success and (steps_to_suc is None or steps_to_suc > 1))
        final = float(np.linalg.norm(np.asarray(obs["achieved_goal"])[:2]
                                     - np.asarray(obs["desired_goal"])[:2]))
        m = episode_metrics(steps, genuine, steps_to_suc, final)
        m.update({"episode": ep, "category": cat})
        records.append(m)
    return records


# ── Summary ───────────────────────────────────────────────────────────────────

def nanmean(x):
    x = np.array([v for v in x if v is not None and not np.isnan(v)], float)
    return float(x.mean()) if x.size else float("nan")


def summarise(res):
    try:
        from scipy import stats
    except Exception:
        stats = None
    conds = [c for c in CONDITIONS if c in res]

    def recs(cond, cat=None):
        out = []
        for r in res.get(cond, {}).values():
            out += [e for e in r["episodes"] if cat is None or e["category"] == cat]
        return out

    print(f"\n{'#'*86}\n  SUCCESS (%) — rows: condition, columns: TRUE material "
          f"(physics never changes)\n{'#'*86}")
    print(f"  {'':24s}" + "".join(f"{c:>18s}" for c in CATEGORIES) + f"{'All':>10s}")
    for cond in conds:
        row = [100 * nanmean([e['success'] for e in recs(cond, c)]) for c in CATEGORIES]
        allr = 100 * nanmean([e['success'] for e in recs(cond)])
        print(f"  {cond:24s}" + "".join(f"{v:18.1f}" for v in row) + f"{allr:10.1f}")
    print("  (diagonal of the three forced-code rows = PGDA with a perfect encoder)")

    for metric, unit in [("push_effort", "cm per step, max 7.07"), ("strike_offset", "cm"),
                         ("spin", "rad/s"), ("bouts", "count"), ("yaw_change", "deg")]:
        print(f"\n  {metric} ({unit})")
        print(f"  {'':24s}" + "".join(f"{c:>18s}" for c in CATEGORIES))
        for cond in conds:
            row = [nanmean([e[metric] for e in recs(cond, c)]) for c in CATEGORIES]
            print(f"  {cond:24s}" + "".join(f"{v:18.3f}" for v in row))

    # Does the code change behaviour with physics held fixed? Seed-level test.
    print(f"\n{'#'*86}\n  EFFECT OF THE CODE WITH PHYSICS FIXED (seed-level means, paired over seeds)\n{'#'*86}")
    for cat in CATEGORIES:
        true_cond = {"Heavy Rust": "PGDA code=Heavy Rust", "Wet Ice": "PGDA code=Wet Ice",
                     "Coarse Sandpaper": "PGDA code=Sandpaper"}[cat]
        for other in [c for c in conds if c.startswith("PGDA code=") and c != true_cond]:
            for metric in ["success", "push_effort", "strike_offset"]:
                a, b = [], []
                for s in SEEDS:
                    ra = res.get(true_cond, {}).get(f"seed_{s}")
                    rb = res.get(other, {}).get(f"seed_{s}")
                    if not ra or not rb:
                        continue
                    va = nanmean([e[metric] for e in ra["episodes"] if e["category"] == cat])
                    vb = nanmean([e[metric] for e in rb["episodes"] if e["category"] == cat])
                    if not (np.isnan(va) or np.isnan(vb)):
                        a.append(va)
                        b.append(vb)
                if len(a) < 2:
                    continue
                scale = 100 if metric == "success" else 1
                d = (np.array(b) - np.array(a)) * scale
                p = (f", p = {stats.ttest_rel(b, a).pvalue:.3f}"
                     if stats is not None and len(a) >= 3 else "")
                print(f"  true {cat:16s} | {other.replace('PGDA ', ''):22s} vs true code: "
                      f"{metric:13s} {d.mean():+8.3f}{p}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'='*86}\n  Behaviour analysis | DRY_RUN = {DRY_RUN}\n{'='*86}")
    missing = 0
    for cond, (ck, mode, enc, forced) in CONDITIONS.items():
        for s in SEEDS:
            p_ = ckpt_path(ck, s)
            ok = p_ is not None and os.path.exists(p_)
            missing += not ok
            if cond.endswith("encoder") or cond.startswith("No-Cue"):
                print(f"  {cond:24s} seed {s:4d}: {p_}  [{'OK' if ok else 'MISSING'}]")
    if missing:
        print(f"\n  {missing} checkpoint(s) missing — fix RUN_IDS.")
    if DRY_RUN:
        print("\n  DRY_RUN = True: nothing evaluated. Set DRY_RUN = False to run.")
        sys.exit(0)

    res = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            res = json.load(f)

    for cond, (ck, mode, enc, forced) in CONDITIONS.items():
        for s in SEEDS:
            key = f"seed_{s}"
            if key in res.get(cond, {}):
                continue
            path = ckpt_path(ck, s)
            if not path or not os.path.exists(path):
                continue
            env = make_env(mode, enc)
            model = load_model(path, env)
            records = run_condition(model, env, s, forced)
            env.close()
            res.setdefault(cond, {})[key] = {"checkpoint": path, "episodes": records}
            with open(RESULTS_FILE, "w") as f:
                json.dump(res, f)
            sr = 100 * np.mean([e["success"] for e in records])
            print(f"  {cond:24s} seed {s:4d}: success {sr:5.1f}%")

    summarise(res)
    print(f"\n  Saved -> {RESULTS_FILE}")
