# 17-baseline_label_check.py — Does writing the material label into the
# Baseline's observation change its behaviour?
# Physically-Grounded Diffusion Augmentation (PGDA)
#
# WHY
#   On the real robot (14-deploy_policy.py --agent A --condition matched) the
#   true category code was written into observation dims 9-10 for the Baseline
#   too. The Baseline was trained with dims 9-10 always at zero, so the network
#   weights attached to those two inputs never received a gradient and are still
#   at their random initial values. A non-zero code therefore reaches the
#   Baseline's policy through untrained weights. If that perturbs its actions,
#   part of its real-robot drop (98% in simulation -> 70% / 63%) would come from
#   the label input itself rather than from sim-to-real transfer.
#
# WHAT IT DOES (simulation only, no lab needed)
#   Baseline (A) checkpoints, coupled dynamics, the SAME seeded episodes as
#   13-eval_coupled_and_category.py, two conditions:
#     zero  : dims 9-10 = 0, as in training (should reproduce 83.6% exactly)
#     label : dims 9-10 = the true category code, as on the robot
#   Reported overall and per category; Heavy Rust and Coarse Sandpaper are the
#   two categories tested on hardware.
#
# Resumable: results saved after every (condition, seed).

import glob
import json
import os
import random
import sys

import numpy as np

# ── numpy 2.x checkpoint compatibility (same shim as 15-eval_feedback_noise.py)
if int(np.__version__.split(".")[0]) < 2:
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric

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

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC

from pgda.env import UR3ePushingEnv

# ── Configuration ─────────────────────────────────────────────────────────────
DRY_RUN = False  # True: list checkpoints only, no evaluation

SEEDS = [42, 123, 456, 789, 1011]
NUM_EPISODES = 100
EVAL_THRESHOLD = 0.05
EPISODE_SEED_BASE = 10_000  # same as 13-eval_coupled_and_category.py
RESULTS_FILE = "baseline_label_check.json"
CODE_DIMS = [9, 10]

# Same Baseline run IDs as 13 / 15
RUN_IDS = {42: "8nmpi7rn", 123: "re3t6y0u", 456: "e135k9g9", 789: "nkf3hbh8", 1011: "bmnmufqc"}

# Category code = normalised midpoint (Appendix C). These must match what
# 14-deploy_policy.py writes; check them against that script before running.
CODES = {
    "Coarse Sandpaper": [0.400, 0.832],
    "Heavy Rust": [0.800, 0.597],
    "Wet Ice": [0.200, 0.013],
}
CONDITIONS = ["zero", "label"]
CATEGORIES = ["Heavy Rust", "Wet Ice", "Coarse Sandpaper"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def ckpt_path(seed):
    rid = RUN_IDS.get(seed, "")
    if rid:
        return f"models/{rid}/final_model_baseline_seed{seed}_v2.zip"
    c = glob.glob(f"models/*/final_model_baseline_seed{seed}_v2.zip")
    return max(c, key=os.path.getctime) if c else None


def make_env():
    env = UR3ePushingEnv(
        render_mode=None,
        randomization_mode="coupled_no_cue",
        reward_weights={"success_threshold": EVAL_THRESHOLD},
        encoder_path=None,
    )
    env.success_threshold = EVAL_THRESHOLD
    return TimeLimit(env, max_episode_steps=200)


def load_model(path, env):
    return SAC.load(
        path,
        env=env,
        device="cpu",
        custom_objects={
            "action_space": env.action_space,
            "observation_space": env.observation_space,
            "lr_schedule": lambda _: 0.0,
        },
    )


def with_code(obs, code):
    """Copy of obs with dims 9-10 set to `code` (policy input only)."""
    if code is None:
        return obs
    out = {k: v.copy() for k, v in obs.items()}
    out["observation"][CODE_DIMS] = np.asarray(code, dtype=out["observation"].dtype)
    return out


def run_episodes(model, env, seed, condition, n):
    per_cat = {c: [0, 0] for c in CATEGORIES}
    succ = 0
    for ep in range(n):
        ep_seed = EPISODE_SEED_BASE + 1000 * seed + ep
        np.random.seed(ep_seed)
        random.seed(ep_seed)
        obs, _ = env.reset()
        cat = getattr(env.unwrapped, "_current_category", None) or "unknown"
        code = CODES.get(cat) if condition == "label" else None
        if condition == "label" and code is None:
            raise RuntimeError(f"No code for category '{cat}'")
        done = truncated = False
        steps, steps_to_suc, success = 0, None, False
        while not (done or truncated) and steps < 200:
            action, _ = model.predict(with_code(obs, code), deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            steps += 1
            if info.get("is_success", False) and steps_to_suc is None:
                steps_to_suc, success = steps, True
        genuine = int(success and (steps_to_suc is None or steps_to_suc > 1))
        succ += genuine
        per_cat.setdefault(cat, [0, 0])
        per_cat[cat][0] += genuine
        per_cat[cat][1] += 1
    return succ, per_cat


def summarise(res):
    print(
        f"\n{'#' * 72}\n  BASELINE (A) ON COUPLED DYNAMICS: label in dims 9-10 vs zero\n{'#' * 72}"
    )
    rates = {}
    for cond in CONDITIONS:
        runs = res.get(cond, {})
        if not runs:
            continue
        v = np.array(
            [runs[f"seed_{s}"]["success_rate"] * 100 for s in SEEDS if f"seed_{s}" in runs]
        )
        rates[cond] = v
        cats = []
        for c in CATEGORIES:
            S = sum(r["per_category"].get(c, {}).get("successes", 0) for r in runs.values())
            N = sum(r["per_category"].get(c, {}).get("episodes", 0) for r in runs.values())
            cats.append(f"{c.split()[-1]} {100 * S / max(N, 1):5.1f}% ({S}/{N})")
        S = sum(
            r["per_category"].get(c, {}).get("successes", 0)
            for r in runs.values()
            for c in ["Heavy Rust", "Coarse Sandpaper"]
        )
        N = sum(
            r["per_category"].get(c, {}).get("episodes", 0)
            for r in runs.values()
            for c in ["Heavy Rust", "Coarse Sandpaper"]
        )
        print(
            f"  {cond:6s} overall {v.mean():5.1f} ± {v.std(ddof=1) if len(v) > 1 else 0:3.1f}%   "
            + "   ".join(cats)
            + f"   Rust+Sand {100 * S / max(N, 1):5.1f}%"
        )
    if "zero" in rates and "label" in rates and len(rates["zero"]) == len(rates["label"]):
        d = rates["label"] - rates["zero"]
        try:
            from scipy import stats

            p = stats.ttest_rel(rates["label"], rates["zero"]).pvalue
            ptxt = f", seed-paired p = {p:.3f}"
        except Exception:
            ptxt = ""
        print(
            f"\n  label - zero: {d.mean():+.1f} pp overall (per seed {d.astype(int).tolist()}){ptxt}"
        )
        print(
            "  Expected for 'zero': 83.6% overall, Rust 97.3%, Sandpaper 98.0% "
            "(13-eval_coupled_and_category.py)."
        )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'=' * 72}\n  Baseline label check | DRY_RUN = {DRY_RUN}\n{'=' * 72}")
    missing = 0
    for s in SEEDS:
        p_ = ckpt_path(s)
        ok = p_ is not None and os.path.exists(p_)
        missing += not ok
        print(f"  Baseline (A) seed {s:4d}: {p_}  [{'OK' if ok else 'MISSING'}]")
    if missing:
        print(f"\n  {missing} checkpoint(s) missing — fix RUN_IDS.")
    if DRY_RUN:
        print("\n  DRY_RUN = True: nothing evaluated. Set DRY_RUN = False to run.")
        sys.exit(0)

    res = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            res = json.load(f)

    for cond in CONDITIONS:
        for s in SEEDS:
            key = f"seed_{s}"
            if key in res.get(cond, {}):
                continue
            path = ckpt_path(s)
            if not path or not os.path.exists(path):
                continue
            env = make_env()
            model = load_model(path, env)
            succ, per_cat = run_episodes(model, env, s, cond, NUM_EPISODES)
            env.close()
            res.setdefault(cond, {})[key] = {
                "successes": succ,
                "episodes": NUM_EPISODES,
                "success_rate": succ / NUM_EPISODES,
                "per_category": {
                    c: {"successes": v[0], "episodes": v[1]} for c, v in per_cat.items()
                },
                "checkpoint": path,
            }
            with open(RESULTS_FILE, "w") as f:
                json.dump(res, f, indent=2)
            print(
                f"  {cond:6s} | seed {s:4d}: {succ}/{NUM_EPISODES}  "
                + "  ".join(f"{c.split()[-1]}={v[0]}/{v[1]}" for c, v in per_cat.items() if v[1])
            )

    summarise(res)
    print(f"\n  Saved -> {RESULTS_FILE}")
