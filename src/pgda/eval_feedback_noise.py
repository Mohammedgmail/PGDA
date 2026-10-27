# 15-eval_feedback_noise.py — Does the appearance cue help when interaction
# feedback is unreliable?
# Physically-Grounded Diffusion Augmentation (PGDA)
#
# QUESTION
#   Without a cue, a policy can only infer material from how the object
#   responds to pushes (object velocity, spin, yaw). PGDA also receives a
#   per-episode material code that does not depend on those channels.
#   If the feedback channels are corrupted (as with camera tracking: noise and
#   latency), does PGDA degrade less than the no-cue policies?
#
# WHAT IS CORRUPTED (policy input only; the simulation itself is untouched)
#   observation dims 5-6 (object velocity x, y), 7 (object spin), 8 (object yaw)
#   Positions (dims 0-4, achieved/desired goal) and the code (dims 9-10) are
#   left clean, so only the interaction-feedback channels are affected.
#
# CONDITIONS
#   clean      : no corruption (should reproduce 13-eval_..._results exactly)
#   noise_a    : Gaussian noise, sigma = a x (std of that channel in clean
#                rollouts), redrawn every control step
#   delay_k    : the four channels are k control steps old (k / 12 s latency)
#   Noise sigmas are calibrated once from clean rollouts and saved, so every
#   agent receives identical noise magnitudes and identical noise sequences.
#
# EPISODES: the same seeded episodes as 13-eval_coupled_and_category.py.
# Resumable: results saved after every (condition, agent, seed).

import os
import sys
import glob
import json
import random
from collections import deque
import numpy as np

# ── numpy 2.x checkpoint compatibility (same shim as 9-evaluate_agents.py) ───
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
DRY_RUN = False         # True: list checkpoints only, no evaluation

SEEDS             = [42, 123, 456, 789, 1011]
NUM_EPISODES      = 100
EVAL_THRESHOLD    = 0.05
EPISODE_SEED_BASE = 10_000        # same as 13-eval_coupled_and_category.py
NOISE_SEED_OFFSET = 777_777       # separate stream for the corruption noise
RESULTS_FILE      = "feedback_noise_results.json"
CALIB_FILE        = "feedback_noise_calibration.json"
CALIB_EPISODES    = 30            # clean episodes per calibration agent
ENCODER_PATH      = ("/Users/mohammedalfandi/Documents/Robotics and AI /"
                     "MSc Project /Codes-2026/texture_encoder.pt")

FEEDBACK_DIMS = [5, 6, 7, 8]      # obj vel x, obj vel y, obj spin, obj yaw

# (name, kind, level). Remove entries to shorten the run.
CONDITIONS = [
    ("clean",     "none",  0),
    ("noise_0.5", "noise", 0.5),
    ("noise_1.0", "noise", 1.0),
    ("delay_2",   "delay", 2),     # ~167 ms at 12 Hz
    ("delay_4",   "delay", 4),     # ~333 ms
]

AGENTS_TO_RUN = ["pgda", "agent_e", "baseline_on_coupled"]

# Copy the run IDs you used in 13-eval_coupled_and_category.py
RUN_IDS = {
    "baseline": {42: "8nmpi7rn", 123: "re3t6y0u", 456: "e135k9g9",
                 789: "nkf3hbh8", 1011: "bmnmufqc"},
    "experimental": {42: "sgl94py3", 123: "1ktap4xi", 456: "lqckfjem",
                     789: "n8cfyu4x", 1011: "289a4luz"},
    "coupled_no_cue": {42: "lt56b7lo", 123: "t72zdfyj", 456: "azkfw077",
                       789: "k2nomi95", 1011: "7djx8mih"},
}

# name -> (checkpoint type, env mode, needs encoder, label)
AGENTS = {
    "pgda":                ("experimental", "experimental", True,
                            "PGDA (B)"),
    "agent_e":             ("coupled_no_cue", "coupled_no_cue", False,
                            "Coupled, no cue (E)"),
    "baseline_on_coupled": ("baseline", "coupled_no_cue", False,
                            "Baseline (A) on coupled"),
}
CATEGORIES = ["Heavy Rust", "Wet Ice", "Coarse Sandpaper"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def ckpt_path(ckpt_type, seed):
    rid = RUN_IDS.get(ckpt_type, {}).get(seed, "")
    if rid:
        return f"models/{rid}/final_model_{ckpt_type}_seed{seed}_v2.zip"
    c = glob.glob(f"models/*/final_model_{ckpt_type}_seed{seed}_v2.zip")
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


class Corruptor:
    """Corrupts FEEDBACK_DIMS of the policy's observation; env is untouched."""

    def __init__(self, kind, level, sigma):
        self.kind, self.level = kind, level
        self.sigma = np.asarray(sigma, dtype=np.float32)

    def reset(self, obs, noise_seed):
        self.rng = np.random.default_rng(noise_seed)
        self.buf = deque([obs["observation"][FEEDBACK_DIMS].copy()]
                         * (int(self.level) + 1 if self.kind == "delay" else 1),
                         maxlen=(int(self.level) + 1 if self.kind == "delay" else 1))
        return self(obs, first=True)

    def __call__(self, obs, first=False):
        if self.kind == "none":
            return obs
        out = {k: v.copy() for k, v in obs.items()}
        clean = obs["observation"][FEEDBACK_DIMS].copy()
        if self.kind == "noise":
            out["observation"][FEEDBACK_DIMS] = clean + (
                self.level * self.sigma *
                self.rng.standard_normal(len(FEEDBACK_DIMS))).astype(np.float32)
        elif self.kind == "delay":
            if not first:
                self.buf.append(clean)
            out["observation"][FEEDBACK_DIMS] = self.buf[0]   # k steps old
        return out


def run_episodes(model, env, seed, corr, n, record=None):
    per_cat = {c: [0, 0] for c in CATEGORIES}
    succ = 0
    for ep in range(n):
        ep_seed = EPISODE_SEED_BASE + 1000 * seed + ep
        np.random.seed(ep_seed)
        random.seed(ep_seed)
        obs, _ = env.reset()
        cat = getattr(env.unwrapped, "_current_category", None) or "unknown"
        pol_obs = corr.reset(obs, NOISE_SEED_OFFSET + ep_seed)
        done = truncated = False
        steps, steps_to_suc, success = 0, None, False
        while not (done or truncated) and steps < 200:
            if record is not None:
                record.append(obs["observation"][FEEDBACK_DIMS].copy())
            action, _ = model.predict(pol_obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            pol_obs = corr(obs)
            steps += 1
            if info.get("is_success", False) and steps_to_suc is None:
                steps_to_suc, success = steps, True
        genuine = int(success and (steps_to_suc is None or steps_to_suc > 1))
        succ += genuine
        per_cat.setdefault(cat, [0, 0])
        per_cat[cat][0] += genuine
        per_cat[cat][1] += 1
    return succ, per_cat


def calibrate():
    """Std of each feedback channel in clean rollouts (PGDA + E, seed 42)."""
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE) as f:
            c = json.load(f)
        print(f"  calibration loaded from {CALIB_FILE}: sigma = {c['sigma']}")
        return c["sigma"]
    rec = []
    for name in ["pgda", "agent_e"]:
        ck, mode, enc, label = AGENTS[name]
        env = make_env(mode, enc)
        model = load_model(ckpt_path(ck, 42), env)
        run_episodes(model, env, 42, Corruptor("none", 0, [0] * 4),
                     CALIB_EPISODES, record=rec)
        env.close()
    sigma = np.std(np.array(rec), axis=0).tolist()
    with open(CALIB_FILE, "w") as f:
        json.dump({"dims": FEEDBACK_DIMS,
                   "names": ["obj_vel_x", "obj_vel_y", "obj_spin", "obj_yaw"],
                   "sigma": sigma, "n_steps": len(rec)}, f, indent=2)
    print(f"  calibration: sigma = {[round(s, 4) for s in sigma]} "
          f"from {len(rec)} clean steps -> {CALIB_FILE}")
    return sigma


def summarise(res):
    print(f"\n{'#'*78}\n  SUCCESS (%) mean ± sd over seeds; change vs clean in brackets\n{'#'*78}")
    conds = [c[0] for c in CONDITIONS if c[0] in res]
    print("  " + " " * 26 + "".join(f"{c:>17s}" for c in conds))
    means = {}
    for a in AGENTS_TO_RUN:
        row = []
        for c in conds:
            runs = res.get(c, {}).get(a, {})
            if not runs:
                row.append(" " * 17)
                continue
            v = np.array([r["success_rate"] * 100 for r in runs.values()])
            means[(a, c)] = v
            base = means.get((a, "clean"))
            delta = f"({v.mean() - base.mean():+.1f})" if base is not None and c != "clean" else ""
            row.append(f"{v.mean():5.1f}±{v.std(ddof=1) if len(v) > 1 else 0:3.1f}{delta:>8s}")
        print(f"  {AGENTS[a][3]:26s}" + "".join(f"{x:>17s}" for x in row))
    if "pgda" in AGENTS_TO_RUN and "agent_e" in AGENTS_TO_RUN:
        print("\n  PGDA - E per condition (seed-paired, same episodes):")
        for c in conds:
            if ("pgda", c) in means and ("agent_e", c) in means \
                    and len(means[("pgda", c)]) == len(means[("agent_e", c)]):
                d = means[("pgda", c)] - means[("agent_e", c)]
                print(f"    {c:12s} {d.mean():+5.1f} pp   per seed {d.astype(int).tolist()}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'='*78}\n  Feedback-corruption evaluation | DRY_RUN = {DRY_RUN}\n{'='*78}")
    missing = 0
    for a in AGENTS_TO_RUN:
        for s in SEEDS:
            p_ = ckpt_path(AGENTS[a][0], s)
            ok = p_ is not None and os.path.exists(p_)
            missing += not ok
            print(f"  {AGENTS[a][3]:26s} seed {s:4d}: {p_}  [{'OK' if ok else 'MISSING'}]")
    if missing:
        print(f"\n  {missing} checkpoint(s) missing — fix RUN_IDS.")
    if DRY_RUN:
        print("\n  DRY_RUN = True: nothing evaluated. Set DRY_RUN = False to run.")
        sys.exit(0)

    sigma = calibrate()
    res = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            res = json.load(f)

    for cname, kind, level in CONDITIONS:
        for a in AGENTS_TO_RUN:
            ck, mode, enc, label = AGENTS[a]
            for s in SEEDS:
                key = f"seed_{s}"
                if key in res.get(cname, {}).get(a, {}):
                    continue
                path = ckpt_path(ck, s)
                if not path or not os.path.exists(path):
                    continue
                env = make_env(mode, enc)
                model = load_model(path, env)
                succ, per_cat = run_episodes(model, env, s,
                                             Corruptor(kind, level, sigma),
                                             NUM_EPISODES)
                env.close()
                res.setdefault(cname, {}).setdefault(a, {})[key] = {
                    "successes": succ, "episodes": NUM_EPISODES,
                    "success_rate": succ / NUM_EPISODES,
                    "per_category": {c: {"successes": v[0], "episodes": v[1]}
                                     for c, v in per_cat.items()},
                    "checkpoint": path}
                with open(RESULTS_FILE, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"  {cname:10s} | {label:26s} | seed {s:4d}: {succ}/{NUM_EPISODES}  "
                      + "  ".join(f"{c.split()[-1]}={v[0]}/{v[1]}"
                                  for c, v in per_cat.items() if v[1]))

    summarise(res)
    print(f"\n  Saved -> {RESULTS_FILE}  (noise sigmas in {CALIB_FILE})")
