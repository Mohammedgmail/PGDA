# 19-eval_specialists.py — Specialists vs generalists on the SAME episodes,
# and the largest gain any material cue could add
# Physically-Grounded Diffusion Augmentation (PGDA)
#
# WHAT IT DOES
#   For TARGET (e.g. "Wet Ice"), it walks through the seeded episodes used by
#   13-eval_coupled_and_category.py (seed = EPISODE_SEED_BASE + 1000*s + k),
#   keeps the first N_PER_SEED episodes whose category is TARGET, and evaluates
#   on exactly those episodes:
#     - the specialists trained by 18-train_specialist.py, and
#     - the generalists: No-Cue (E), PGDA (B), Oracle (C), Baseline (A).
#   Episode k >= 100 continues the same seeding scheme, so the first ~third of
#   the target episodes are literally the ones in Table III.
#
# THE BOUND (printed at the end)
#   A cue-conditioned policy can at best act like the specialist on each
#   material, so the largest gain any cue could add over the No-Cue agent is
#       Delta_max = sum_c p_c * (S_c(specialist) - S_c(No-Cue))
#   For categories without specialists yet, S_c(specialist) is capped at 100%,
#   which makes the bound conservative (an over-estimate of the cue's value).
#   p_c and the No-Cue rates for those categories come from
#   coupled_and_category_results.json (script 13), if present.
#
# Run once per TARGET. Results accumulate in specialist_results.json,
# keyed by category, so the bound tightens as you add specialists.
# Resumable: results saved after every (agent, seed).

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
DRY_RUN = False          # True: list checkpoints and episode counts only

TARGET            = "Wet Ice"     # must match CATEGORY used in 18-train_specialist.py
SEEDS             = [42, 123, 456, 789, 1011]
N_PER_SEED        = 100           # target-category episodes per seed
MAX_K             = 999           # stay inside each seed's block of 1000 seeds
EVAL_THRESHOLD    = 0.05
EPISODE_SEED_BASE = 10_000        # same as 13 / 15 / 17
RESULTS_FILE      = "specialist_results.json"
TABLE3_FILE       = "coupled_and_category_results.json"   # from script 13
ENCODER_PATH      = ("/Users/mohammedalfandi/Documents/Robotics and AI /"
                     "MSc Project /Codes-2026/texture_encoder.pt")
CATEGORIES        = ["Heavy Rust", "Wet Ice", "Coarse Sandpaper"]

# Specialist run IDs per category and seed (printed at the end of training).
# Blank = auto-discover the newest models/*/final_model_specialist_<slug>_seed<s>.zip
SPECIALIST_RUN_IDS = {
    "Wet Ice":          {42: "cxz4zstx", 123: "0ud8ouef", 456: "bd1uxkzg", 789: "mfk1vhmc", 1011: "9ore1pfd"},
    "Heavy Rust":       {},
    "Coarse Sandpaper": {},
}

# Generalists: same run IDs as 13 / 15 / 17 (blank = auto-discover, as in 13)
RUN_IDS = {
    "baseline": {42: "8nmpi7rn", 123: "re3t6y0u", 456: "e135k9g9",
                 789: "nkf3hbh8", 1011: "bmnmufqc"},
    "experimental": {42: "sgl94py3", 123: "1ktap4xi", 456: "lqckfjem",
                     789: "n8cfyu4x", 1011: "289a4luz"},
    "coupled_no_cue": {42: "lt56b7lo", 123: "t72zdfyj", 456: "azkfw077",
                       789: "k2nomi95", 1011: "7djx8mih"},
    "oracle": {42: "y0ruahqi", 123: "", 456: "", 789: "", 1011: ""},
}

# name -> (checkpoint type, env mode, needs encoder, label)  [as in script 13]
GENERALISTS = {
    "agent_e":             ("coupled_no_cue", "coupled_no_cue", False, "No-Cue (E)"),
    "pgda":                ("experimental", "experimental", True, "PGDA (B)"),
    "oracle":              ("oracle", "oracle", True, "Oracle (C)"),
    "baseline_on_coupled": ("baseline", "coupled_no_cue", False, "Baseline (A)"),
}
AGENTS_TO_RUN = ["specialist", "agent_e", "pgda", "oracle", "baseline_on_coupled"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def slug(cat):
    return cat.lower().replace(" ", "_")


def ckpt_path(agent, seed):
    if agent == "specialist":
        rid = SPECIALIST_RUN_IDS.get(TARGET, {}).get(seed, "")
        name = f"final_model_specialist_{slug(TARGET)}_seed{seed}.zip"
        if rid:
            return f"models/{rid}/{name}"
        c = glob.glob(f"models/*/{name}")
        return max(c, key=os.path.getctime) if c else None
    ck = GENERALISTS[agent][0]
    rid = RUN_IDS.get(ck, {}).get(seed, "")
    if rid:
        return f"models/{rid}/final_model_{ck}_seed{seed}_v2.zip"
    c = glob.glob(f"models/*/final_model_{ck}_seed{seed}_v2.zip")
    return max(c, key=os.path.getctime) if c else None


def agent_setup(agent):
    """(env mode, needs encoder, label)"""
    if agent == "specialist":
        return "coupled_no_cue", False, f"Specialist ({TARGET})"
    _, mode, enc, label = GENERALISTS[agent]
    return mode, enc, label


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


def seeded_reset(env, seed, k):
    ep_seed = EPISODE_SEED_BASE + 1000 * seed + k
    np.random.seed(ep_seed)
    random.seed(ep_seed)
    obs, _ = env.reset()
    return obs, getattr(env.unwrapped, "_current_category", None)


def target_episode_ks(seed):
    """Indices k of the first N_PER_SEED episodes whose category is TARGET.
    Uses a cue-free environment; categories do not depend on the agent
    (Table III has identical per-category counts for every agent)."""
    env = make_env("coupled_no_cue", False)
    ks = []
    for k in range(MAX_K + 1):
        _, cat = seeded_reset(env, seed, k)
        if cat == TARGET:
            ks.append(k)
            if len(ks) == N_PER_SEED:
                break
    env.close()
    return ks


def run_episodes(model, env, seed, ks):
    succ, outcomes = 0, []
    for k in ks:
        obs, cat = seeded_reset(env, seed, k)
        if cat != TARGET:
            raise RuntimeError(f"seed {seed}, k={k}: expected {TARGET}, got {cat}. "
                               f"Categories differ between environment modes.")
        done = truncated = False
        steps, steps_to_suc, success = 0, None, False
        while not (done or truncated) and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            steps += 1
            if info.get("is_success", False) and steps_to_suc is None:
                steps_to_suc, success = steps, True
        genuine = int(success and (steps_to_suc is None or steps_to_suc > 1))
        succ += genuine
        outcomes.append(genuine)
    return succ, outcomes


# ── Summary and bound ─────────────────────────────────────────────────────────

def table3_stats():
    """p_c and No-Cue per-category success from script 13, if available."""
    if not os.path.exists(TABLE3_FILE):
        return None, None
    with open(TABLE3_FILE) as f:
        t3 = json.load(f)
    counts, e_rate = {c: 0 for c in CATEGORIES}, {}
    for c in CATEGORIES:
        S = N = 0
        for r in t3.get("agent_e", {}).values():
            pc = r.get("per_category", {}).get(c, {})
            S += pc.get("successes", 0)
            N += pc.get("episodes", 0)
        counts[c] = N
        e_rate[c] = 100 * S / N if N else None
    tot = sum(counts.values())
    p = {c: counts[c] / tot for c in CATEGORIES} if tot else None
    return p, e_rate


def summarise(res):
    try:
        from scipy import stats
    except Exception:
        stats = None
    print(f"\n{'#'*76}\n  SUCCESS ON THE SAME TARGET-CATEGORY EPISODES (mean ± sd over seeds)\n{'#'*76}")
    for cat, block in res.items():
        print(f"\n  {cat}  ({N_PER_SEED} episodes per seed)")
        rates = {}
        for agent in AGENTS_TO_RUN:
            runs = block.get(agent, {})
            if not runs:
                continue
            v = {int(s.split('_')[1]): round(r["success_rate"] * 100, 1) for s, r in runs.items()}
            rates[agent] = v
            arr = np.array(list(v.values()))
            label = agent_setup(agent)[2] if agent != "specialist" else f"Specialist ({cat})"
            print(f"    {label:24s} {arr.mean():5.1f} ± {arr.std(ddof=1) if len(arr) > 1 else 0:4.1f}%"
                  f"   per seed {dict(sorted(v.items()))}")
        if "specialist" in rates:
            for other in ["agent_e", "pgda", "oracle"]:
                if other not in rates:
                    continue
                common = sorted(set(rates["specialist"]) & set(rates[other]))
                if len(common) < 2:
                    continue
                d = np.array([rates["specialist"][s] - rates[other][s] for s in common])
                ptxt = ""
                if stats is not None and len(common) >= 3:
                    ptxt = f", seed-paired p = {stats.ttest_rel([rates['specialist'][s] for s in common], [rates[other][s] for s in common]).pvalue:.3f}"
                print(f"    Specialist − {agent_setup(other)[2]:12s} {d.mean():+5.1f} pp "
                      f"(seeds {common}){ptxt}")

    # Bound
    p, e_t3 = table3_stats()
    if p is None:
        p = {c: 1 / 3 for c in CATEGORIES}
        print("\n  (coupled_and_category_results.json not found: using p_c = 1/3)")
    print(f"\n{'#'*76}\n  LARGEST GAIN ANY CUE COULD ADD OVER THE NO-CUE AGENT\n{'#'*76}")
    total, notes = 0.0, []
    for c in CATEGORIES:
        block = res.get(c, {})
        spec = block.get("specialist", {})
        e_here = block.get("agent_e", {})
        common = sorted(set(spec) & set(e_here))
        if spec:
            s_spec = np.mean([r["success_rate"] * 100 for r in spec.values()])
            src = "measured"
        else:
            s_spec, src = 100.0, "capped at 100%"
        if spec and common:      # same seeds and episodes for both
            s_spec = np.mean([spec[k]["success_rate"] * 100 for k in common])
            s_e = np.mean([e_here[k]["success_rate"] * 100 for k in common])
        elif e_here:
            s_e = np.mean([r["success_rate"] * 100 for r in e_here.values()])
        elif e_t3 and e_t3.get(c) is not None:
            s_e = e_t3[c]
        else:
            s_e = None
        if s_e is None:
            notes.append(f"{c}: no No-Cue rate available")
            continue
        gap = s_spec - s_e
        total += p[c] * gap
        print(f"    {c:17s} p = {p[c]:.3f}   specialist {s_spec:5.1f}% ({src:14s})  "
              f"No-Cue {s_e:5.1f}%   gap {gap:+5.1f} pp")
    print(f"\n    Delta_max = {total:+.1f} pp  "
          f"(any cue, even a perfect one, could raise success by at most this much)")
    for n in notes:
        print(f"    note: {n}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'='*76}\n  Specialist evaluation | TARGET = {TARGET} | DRY_RUN = {DRY_RUN}\n{'='*76}")
    plan = []
    for agent in AGENTS_TO_RUN:
        seeds = (sorted(SPECIALIST_RUN_IDS.get(TARGET, {})) or SEEDS) \
            if agent == "specialist" else SEEDS
        for s in seeds:
            p_ = ckpt_path(agent, s)
            ok = p_ is not None and os.path.exists(p_)
            print(f"  {agent_setup(agent)[2]:24s} seed {s:4d}: {p_}  [{'OK' if ok else 'MISSING'}]")
            if ok:
                plan.append((agent, s, p_))

    res = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            res = json.load(f)
    ks_cache = res.setdefault("_episode_indices", {}).setdefault(TARGET, {})

    if DRY_RUN:
        s0 = SEEDS[0]
        ks = target_episode_ks(s0)
        print(f"\n  Seed {s0}: {len(ks)} {TARGET} episodes found within k <= {ks[-1] if ks else '-'}.")
        print("  DRY_RUN = True: nothing evaluated. Set DRY_RUN = False to run.")
        sys.exit(0)

    for agent, s, path in plan:
        block = res.setdefault(TARGET, {}).setdefault(agent, {})
        key = f"seed_{s}"
        if key in block:
            continue
        if str(s) not in ks_cache:
            ks_cache[str(s)] = target_episode_ks(s)
        ks = ks_cache[str(s)]
        mode, enc, label = agent_setup(agent)
        env = make_env(mode, enc)
        model = load_model(path, env)
        succ, outcomes = run_episodes(model, env, s, ks)
        env.close()
        block[key] = {"successes": succ, "episodes": len(ks),
                      "success_rate": succ / len(ks), "outcomes": outcomes,
                      "checkpoint": path}
        with open(RESULTS_FILE, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  {label:24s} seed {s:4d}: {succ}/{len(ks)}")

    summarise({k: v for k, v in res.items() if not k.startswith("_")})
    print(f"\n  Saved -> {RESULTS_FILE}")
