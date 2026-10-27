# 9-evaluate_agents.py — Deterministic Agent Evaluation + RQ2 Mismatch + Ablations
# Physically-Grounded Diffusion Augmentation for Robust VLA (MSc Project)
#
# THREE MODES — set EVAL_MODE below:
#
#   'standard'  — Evaluates Agent A (baseline) vs Agent B (experimental) under
#                 their training conditions. Produces the primary RQ1 result.
#                 Run this for each seed after training.
#
#   'mismatch'  — RQ2 mechanistic experiment. Evaluates ONLY Agent B models
#                 (experimental) under three conditions:
#                   1. matched    — normal experimental mode (baseline comparison)
#                   2. mismatched — texture from one category, physics from another
#                   3. no_texture — grey objects, baseline uncorrelated physics
#                 If Agent B success drops significantly under mismatched vs matched,
#                 the policy was using texture as a physics predictor (RQ2 = YES).
#                 If success is unchanged, the policy ignores texture (RQ2 = NO).
#                 No retraining needed. Run once across all 5 Agent B seeds.
#
#   'ablation'  — Evaluates the Oracle (Agent C), No-DR (Agent D), and
#                 Coupled-No-Cue (Agent E) models alongside the existing
#                 baseline and experimental results.
#                 Produces the full performance spectrum for RQ1:
#                   No-DR ≤ Baseline ≤ [Coupled-No-Cue] ≤ Experimental ≤ Oracle
#                 Oracle           uses mode='oracle'           (ground-truth
#                                  physics in obs).
#                 No-DR            uses mode='no_dr'            (fixed
#                                  physics, grey object).
#                 Coupled-No-Cue   uses mode='coupled_no_cue'   (same coupled
#                                  distribution + asymmetric Izz as
#                                  Experimental, but dims 9-10 permanently
#                                  zero — no encoder inference ever runs).
#                                  Isolates whether the coupled distribution
#                                  ALONE — independent of any visual cue —
#                                  accounts for the Baseline-vs-Experimental
#                                  gap, or whether the cue itself is doing
#                                  real work (Reviewer 2's requested control).
#                 Run after training all ablation models.
#
# Usage:
#   Standard:  set EVAL_MODE = 'standard', set EVAL_SEED and run IDs
#   Mismatch:  set EVAL_MODE = 'mismatch', set AGENT_B_MODEL_PATHS
#   Ablation:  set EVAL_MODE = 'ablation', set ABLATION_SEED and run IDs

import glob
import os
from collections import defaultdict

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC

from pgda.env import UR3ePushingEnv
from pgda.paths import ENCODER_PATH

# ── Configuration ─────────────────────────────────────────────────────────────
EVAL_MODE = os.environ.get("PGDA_EVAL_MODE", "ablation")  # 'standard' | 'mismatch' | 'ablation'

# ── Standard mode configuration ───────────────────────────────────────────────
EVAL_SEED = 42
BASELINE_RUN_ID = "bmnmufqc"  # run ID for baseline model
EXPERIMENTAL_RUN_ID = "289a4luz"  # run ID for experimental model

# ── Mismatch mode configuration ───────────────────────────────────────────────
# Paths to the 5 Agent B (experimental) final models.
AGENT_B_MODEL_PATHS = {
    42: "models/sgl94py3/final_model_experimental_seed42_v2.zip",
    123: "models/1ktap4xi/final_model_experimental_seed123_v2.zip",
    456: "models/lqckfjem/final_model_experimental_seed456_v2.zip",
    789: "models/n8cfyu4x/final_model_experimental_seed789_v2.zip",
    1011: "models/289a4luz/final_model_experimental_seed1011_v2.zip",
}

# ── Ablation mode configuration ───────────────────────────────────────────────
# Seed to evaluate for the ablation study.
# Start with seed 42.  After seeing oracle results, decide whether to run
# remaining seeds.
ABLATION_SEED = 42

# Run IDs from 8-train_agents.py for the oracle, no_dr, and coupled_no_cue
# training runs. Fill these in after training completes — check your
# models/ directory.
ORACLE_RUN_ID = "y0ruahqi"  # e.g. "abc123xy"  ← after oracle training
NO_DR_RUN_ID = "gcg1543h"  # e.g. "def456zw"  ← after no_dr training
COUPLED_NO_CUE_RUN_ID = ""  # e.g. "ghi789ab"  ← after coupled_no_cue
#   training (Agent E )

# The ablation evaluator also re-runs baseline and experimental for the same
# seed so all four results are on identical episode samples.
# Set these to the run IDs you already have for seed 42 (or ABLATION_SEED).
ABLATION_BASELINE_RUN_ID = "8nmpi7rn"  # same as BASELINE_RUN_ID above
ABLATION_EXPERIMENTAL_RUN_ID = "sgl94py3"  # same as the seed-42 experimental

NUM_EVAL_EPISODES = 100
EVAL_THRESHOLD = 0.05  # metres

# Path to the trained texture encoder weights.
# Set this to the same path used in 8-train_agents.py.
# Only used when evaluating experimental agents — baseline always gets None.
ENCODER_PATH = str(ENCODER_PATH)


# ── Statistical helpers ───────────────────────────────────────────────────────


def bootstrap_ci(successes, n_bootstrap=5000, alpha=0.05):
    arr = np.array(successes, dtype=float)
    n = len(arr)
    means = [np.mean(np.random.choice(arr, size=n, replace=True)) for _ in range(n_bootstrap)]
    return (
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def two_proportion_z_test(s_a, n_a, s_b, n_b):
    p_a = s_a / n_a
    p_b = s_b / n_b
    p_pool = (s_a + s_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) + 1e-12
    z = (p_a - p_b) / se
    from math import erfc, sqrt

    return float(z), float(erfc(abs(z) / sqrt(2)))


# ── Core evaluation function ──────────────────────────────────────────────────


def evaluate(
    model_path, mode, num_episodes=100, eval_threshold=0.05, label=None, encoder_path=None
):
    """
    Run deterministic evaluation of a trained SAC+HER agent.

    Parameters
    ----------
    model_path     : path to .zip model file
    mode           : randomization_mode for the environment
                     ('baseline', 'experimental', 'mismatched',
                      'no_texture', 'oracle', 'no_dr', 'coupled_no_cue')
    num_episodes   : number of evaluation episodes
    eval_threshold : success distance threshold in metres
    label          : display label (defaults to mode if None)
    encoder_path   : path to texture_encoder.pt.
                     Required for 'experimental' and 'oracle' modes.
                     None for baseline, no_dr, mismatched, no_texture,
                     and coupled_no_cue (that mode never calls the encoder —
                     dims 9-10 are zeroed by construction, not predicted).
    """
    display = label or mode.upper()
    print(f"\n{'=' * 58}")
    print(f"  Evaluating: {display}")
    print(f"  Model  : {model_path}")
    print(f"  Mode   : {mode}  |  Episodes: {num_episodes}  |  Threshold: {eval_threshold}m")
    print(f"{'=' * 58}")

    env = UR3ePushingEnv(
        render_mode=None,
        randomization_mode=mode,
        reward_weights={"success_threshold": eval_threshold},
        encoder_path=encoder_path,  # None for baseline, path for experimental
    )
    env.success_threshold = eval_threshold
    env = TimeLimit(env, max_episode_steps=200)

    model = SAC.load(model_path, env=env, device="cpu")

    success_flags = []
    final_distances = []
    steps_to_suc_log = []
    steps_per_ep = []
    per_object = defaultdict(lambda: {"success": 0, "total": 0})

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = truncated = False
        step_count = 0
        steps_to_suc = None
        success = False

        while not (done or truncated) and step_count < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            step_count += 1
            if info.get("is_success", False) and steps_to_suc is None:
                steps_to_suc = step_count
                success = True

        import pybullet as p_eval

        try:
            inner = env.env  # unwrap TimeLimit
            obj_pos = p_eval.getBasePositionAndOrientation(
                inner.object_uid, physicsClientId=inner._client
            )[0]
            final_dist = float(np.linalg.norm(np.array(obj_pos[:2]) - np.array(inner.goal_pos[:2])))
            obj_name = inner.current_object_name
        except Exception:
            final_dist = float("nan")
            obj_name = "unknown"

        genuine = int(success and (steps_to_suc is None or steps_to_suc > 1))
        success_flags.append(genuine)
        final_distances.append(final_dist)
        steps_per_ep.append(step_count)
        if steps_to_suc:
            steps_to_suc_log.append(steps_to_suc)
        per_object[obj_name]["total"] += 1
        per_object[obj_name]["success"] += genuine

        status = "SUCCESS" if genuine else "FAILURE"
        print(
            f"  Ep {ep + 1:3d}/{num_episodes} | {status} | "
            f"dist={final_dist:.4f}m | steps={step_count:3d} | "
            f"obj={obj_name}"
        )

    env.close()

    n_success = sum(success_flags)
    success_rate = n_success / num_episodes
    ci_lo, ci_hi = bootstrap_ci(success_flags)
    mean_dist = float(np.nanmean(final_distances))
    mean_steps = float(np.mean(steps_to_suc_log)) if steps_to_suc_log else float("nan")

    print(f"\n{'─' * 58}")
    print(f"  {display} Results")
    print(f"  Genuine success rate : {success_rate * 100:.1f}%")
    print(f"  Success count        : {n_success}/{num_episodes}")
    print(f"  95% CI               : [{ci_lo * 100:.1f}%, {ci_hi * 100:.1f}%]")
    print(f"  Mean Final Δ         : {mean_dist * 100:.2f} cm from goal")
    print(f"  Mean Steps to success: {mean_steps:.1f}")
    print(f"{'─' * 58}")

    print(f"\n  Per-object ({display}):")
    for name, counts in sorted(
        per_object.items(), key=lambda x: x[1]["success"] / max(x[1]["total"], 1)
    ):
        sr = counts["success"] / counts["total"] if counts["total"] else 0
        bar = "█" * int(sr * 10) + "░" * (10 - int(sr * 10))
        print(
            f"    {name:35s} [{bar}] {counts['success']:2d}/{counts['total']:2d} ({sr * 100:.0f}%)"
        )

    return {
        "label": display,
        "mode": mode,
        "success_rate": success_rate,
        "successes": n_success,
        "episodes": num_episodes,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "per_object_results": dict(per_object),
        "final_distances": final_distances,
        "steps_to_success": steps_to_suc_log,
        "mean_final_distance": mean_dist,
        "mean_steps_to_success": mean_steps,
    }


# ── Mismatch experiment ───────────────────────────────────────────────────────


def run_mismatch_experiment(model_paths_by_seed, num_episodes=100):
    """
    RQ2 experiment: evaluate each Agent B model under three conditions.
    Reports per-seed and pooled results.
    """
    import json

    print(f"\n{'#' * 60}")
    print("  RQ2 MISMATCH EXPERIMENT")
    print("  Testing whether Agent B uses texture as a physics cue.")
    print(f"{'#' * 60}")

    all_results = {}

    for seed, model_path in model_paths_by_seed.items():
        if model_path is None:
            print(f"\n  Seed {seed}: model path not set — skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  SEED {seed}")
        print(f"{'=' * 60}")

        seed_results = {}

        # Condition 1: Matched (normal experimental — baseline comparison)
        seed_results["matched"] = evaluate(
            model_path,
            "experimental",
            num_episodes,
            encoder_path=ENCODER_PATH,
            label=f"Seed {seed} — MATCHED (experimental)",
        )

        # Condition 2: Mismatched (texture ≠ physics)
        seed_results["mismatched"] = evaluate(
            model_path,
            "mismatched",
            num_episodes,
            encoder_path=ENCODER_PATH,  # encoder still runs; physics are mismatched
            label=f"Seed {seed} — MISMATCHED (broken correlation)",
        )

        # Condition 3: No texture (grey objects, baseline physics)
        seed_results["no_texture"] = evaluate(
            model_path,
            "no_texture",
            num_episodes,
            encoder_path=None,  # no texture → no encoder prediction
            label=f"Seed {seed} — NO TEXTURE (grey, control)",
        )

        # Per-seed summary
        r_m = seed_results["matched"]
        r_mis = seed_results["mismatched"]
        r_nt = seed_results["no_texture"]

        z_mis, p_mis = two_proportion_z_test(
            r_m["successes"], r_m["episodes"], r_mis["successes"], r_mis["episodes"]
        )
        z_nt, p_nt = two_proportion_z_test(
            r_m["successes"], r_m["episodes"], r_nt["successes"], r_nt["episodes"]
        )

        print(f"\n  ── Seed {seed} Mismatch Summary ──")
        print(
            f"  Matched    : {r_m['success_rate'] * 100:.1f}%  "
            f"CI [{r_m['ci_lower'] * 100:.1f}%, {r_m['ci_upper'] * 100:.1f}%]"
        )
        print(
            f"  Mismatched : {r_mis['success_rate'] * 100:.1f}%  "
            f"CI [{r_mis['ci_lower'] * 100:.1f}%, {r_mis['ci_upper'] * 100:.1f}%]"
            f"  Δ={(r_mis['success_rate'] - r_m['success_rate']) * 100:+.1f}pp"
            f"  p={p_mis:.4f}"
            f"  {'SIGNIFICANT' if p_mis < 0.05 else 'not significant'}"
        )
        print(
            f"  No texture : {r_nt['success_rate'] * 100:.1f}%  "
            f"CI [{r_nt['ci_lower'] * 100:.1f}%, {r_nt['ci_upper'] * 100:.1f}%]"
            f"  Δ={(r_nt['success_rate'] - r_m['success_rate']) * 100:+.1f}pp"
            f"  p={p_nt:.4f}"
            f"  {'SIGNIFICANT' if p_nt < 0.05 else 'not significant'}"
        )

        if p_mis < 0.05:
            delta = (r_mis["success_rate"] - r_m["success_rate"]) * 100
            print(
                f"\n  ✓ RQ2 SIGNAL: Mismatch caused {abs(delta):.1f}pp drop "
                f"(p={p_mis:.4f}). Policy was using texture as physics cue."
            )
        else:
            print(
                "\n  ✗ NO RQ2 SIGNAL: Mismatch had no significant effect. "
                "Policy ignores texture — relies on state vector alone."
            )

        seed_results["stats"] = {
            "matched_vs_mismatched": {
                "z": z_mis,
                "p": p_mis,
                "delta_pp": (r_mis["success_rate"] - r_m["success_rate"]) * 100,
                "significant": p_mis < 0.05,
            },
            "matched_vs_no_texture": {
                "z": z_nt,
                "p": p_nt,
                "delta_pp": (r_nt["success_rate"] - r_m["success_rate"]) * 100,
                "significant": p_nt < 0.05,
            },
        }
        all_results[f"seed_{seed}"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "final_distances"}
            for k, v in seed_results.items()
        }

    # ── Pooled summary across seeds ───────────────────────────────────────────
    valid = [v for v in all_results.values() if "matched" in v]
    if len(valid) > 1:
        print(f"\n{'#' * 60}")
        print("  POOLED MISMATCH RESULTS")
        print(f"{'#' * 60}")
        for cond in ["matched", "mismatched", "no_texture"]:
            rates = [v[cond]["success_rate"] for v in valid]
            print(
                f"  {cond:12s}: "
                f"mean={np.mean(rates) * 100:.1f}%  "
                f"std={np.std(rates) * 100:.1f}%  "
                f"(n={len(rates)} seeds)"
            )
        sig_mis = sum(1 for v in valid if v["stats"]["matched_vs_mismatched"]["significant"])
        print(f"\n  Mismatch significant in {sig_mis}/{len(valid)} seeds.")
        if sig_mis >= 2:
            print("  → RQ2 ANSWER: YES — policy uses texture as physics predictor.")
        elif sig_mis == 0:
            print("  → RQ2 ANSWER: NO — policy ignores texture entirely.")
        else:
            print("  → RQ2 ANSWER: PARTIAL — inconsistent signal across seeds.")

    # Save
    out_path = "mismatch_experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved → {out_path}")
    return all_results


# ── Standard evaluation (RQ1) ─────────────────────────────────────────────────


def run_standard_evaluation():
    import json

    if BASELINE_RUN_ID and EXPERIMENTAL_RUN_ID:
        baseline_path = f"models/{BASELINE_RUN_ID}/final_model_baseline_seed{EVAL_SEED}_v2.zip"
        exp_path = f"models/{EXPERIMENTAL_RUN_ID}/final_model_experimental_seed{EVAL_SEED}_v2.zip"
    else:
        baseline_models = glob.glob(
            f"models/*/final_model_baseline_seed{EVAL_SEED}.zip"
        ) + glob.glob("models/*/final_model_baseline.zip")
        experimental_models = glob.glob(
            f"models/*/final_model_experimental_seed{EVAL_SEED}.zip"
        ) + glob.glob("models/*/final_model_experimental.zip")

        if not baseline_models or not experimental_models:
            print("Error: model files not found.")
            print(f"  Baseline    : {baseline_models}")
            print(f"  Experimental: {experimental_models}")
            raise SystemExit(1)

        baseline_path = max(baseline_models, key=os.path.getctime)
        exp_path = max(experimental_models, key=os.path.getctime)

    print(f"  Baseline model     : {baseline_path}")
    print(f"  Experimental model : {exp_path}")

    results_baseline = evaluate(
        baseline_path,
        "baseline",
        num_episodes=NUM_EVAL_EPISODES,
        eval_threshold=EVAL_THRESHOLD,
        encoder_path=None,  # baseline never uses encoder
        label="BASELINE (Agent A)",
    )
    results_exp = evaluate(
        exp_path,
        "experimental",
        num_episodes=NUM_EVAL_EPISODES,
        eval_threshold=EVAL_THRESHOLD,
        encoder_path=ENCODER_PATH,  # experimental uses encoder
        label="EXPERIMENTAL (Agent B)",
    )

    print(f"\n{'=' * 58}")
    print("  STATISTICAL COMPARISON (RQ1)")
    print(f"{'=' * 58}")

    z, p_val = two_proportion_z_test(
        results_baseline["successes"],
        results_baseline["episodes"],
        results_exp["successes"],
        results_exp["episodes"],
    )

    print(
        f"  Baseline     : {results_baseline['success_rate'] * 100:.1f}%  "
        f"CI [{results_baseline['ci_lower'] * 100:.1f}%, "
        f"{results_baseline['ci_upper'] * 100:.1f}%]"
    )
    print(
        f"  Experimental : {results_exp['success_rate'] * 100:.1f}%  "
        f"CI [{results_exp['ci_lower'] * 100:.1f}%, "
        f"{results_exp['ci_upper'] * 100:.1f}%]"
    )
    print(f"  z-statistic  : {z:+.3f}")
    print(
        f"  p-value      : {p_val:.6f}  "
        f"({'significant' if p_val < 0.05 else 'not significant'} α=0.05)"
    )
    delta = results_exp["success_rate"] - results_baseline["success_rate"]
    print(f"  Δ success    : {delta * 100:+.1f} pp ({'Exp > Base' if delta > 0 else 'Base > Exp'})")

    output = {
        "seed": EVAL_SEED,
        "baseline": {k: v for k, v in results_baseline.items() if k != "final_distances"},
        "experimental": {k: v for k, v in results_exp.items() if k != "final_distances"},
        "comparison": {
            "z_statistic": z,
            "p_value": p_val,
            "delta_success": delta,
            "significant": p_val < 0.05,
        },
    }
    out_path = f"evaluation_results_seed{EVAL_SEED}_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")
    print(f"{'=' * 58}")


# ── Ablation evaluation (Oracle + No-DR, RQ1 performance spectrum) ───────────


def run_ablation_evaluation():
    """
    Evaluates all five agents for the same seed to produce the full
    performance spectrum:

        No-DR (D) ≤ Baseline (A) ≤ Coupled-No-Cue (E) ≤ Experimental (B) ≤ Oracle (C)

    All five are evaluated on the same number of episodes so results are
    directly comparable. Oracle, No-DR, and Coupled-No-Cue use the
    pushing_env.py modes 'oracle', 'no_dr', and 'coupled_no_cue' respectively.

    Key scientific questions answered:
      Oracle - Experimental gap        → how much is lost to encoder
                                          imperfection?
      Baseline - No-DR gap             → how much does DR difficulty alone
                                          explain?
      Baseline - Coupled-No-Cue gap    → how much of the Baseline-vs-
                                          Experimental gap is just the coupled
                                          distribution + asymmetric Izz,
                                          BEFORE any cue is added? (Reviewer 2)
      Experimental - Coupled-No-Cue gap → holding the coupled distribution and
                                          Izz fixed, how much does the cue
                                          itself buy? (the cleanest isolation
                                          of the cue's value)
    """
    import json

    seed = ABLATION_SEED
    print(f"\n{'#' * 62}")
    print(f"  ABLATION EVALUATION — Seed {seed}")
    print("  Full performance spectrum: No-DR / Baseline / Exp / Oracle")
    print(f"{'#' * 62}\n")

    # ── Build model paths ─────────────────────────────────────────────────────
    def _model_path(run_id, mode_name, s):
        """Construct model path from run ID and mode name."""
        return f"models/{run_id}/final_model_{mode_name}_seed{s}_v2.zip"

    paths = {}

    # Baseline (Agent A) ── already trained, use existing run ID
    if ABLATION_BASELINE_RUN_ID:
        paths["baseline"] = _model_path(ABLATION_BASELINE_RUN_ID, "baseline", seed)
    else:
        candidates = glob.glob(f"models/*/final_model_baseline_seed{seed}_v2.zip")
        if not candidates:
            raise FileNotFoundError(
                f"No baseline model found for seed {seed}. Set ABLATION_BASELINE_RUN_ID."
            )
        paths["baseline"] = max(candidates, key=os.path.getctime)

    # Experimental (Agent B) ── already trained
    if ABLATION_EXPERIMENTAL_RUN_ID:
        paths["experimental"] = _model_path(ABLATION_EXPERIMENTAL_RUN_ID, "experimental", seed)
    else:
        candidates = glob.glob(f"models/*/final_model_experimental_seed{seed}_v2.zip")
        if not candidates:
            raise FileNotFoundError(
                f"No experimental model found for seed {seed}. Set ABLATION_EXPERIMENTAL_RUN_ID."
            )
        paths["experimental"] = max(candidates, key=os.path.getctime)

    # Oracle (Agent C) ── newly trained
    if ORACLE_RUN_ID:
        paths["oracle"] = _model_path(ORACLE_RUN_ID, "oracle", seed)
    else:
        candidates = glob.glob(f"models/*/final_model_oracle_seed{seed}_v2.zip")
        if not candidates:
            raise FileNotFoundError(
                f"No oracle model found for seed {seed}. "
                f"Set ORACLE_RUN_ID or check models/ directory. "
                f"Run 8-train_agents.py with MODE='oracle' first."
            )
        paths["oracle"] = max(candidates, key=os.path.getctime)

    # No-DR (Agent D) ── newly trained
    if NO_DR_RUN_ID:
        paths["no_dr"] = _model_path(NO_DR_RUN_ID, "no_dr", seed)
    else:
        candidates = glob.glob(f"models/*/final_model_no_dr_seed{seed}_v2.zip")
        if not candidates:
            raise FileNotFoundError(
                f"No no_dr model found for seed {seed}. "
                f"Set NO_DR_RUN_ID or check models/ directory. "
                f"Run 8-train_agents.py with MODE='no_dr' first."
            )
        paths["no_dr"] = max(candidates, key=os.path.getctime)

    # Coupled-No-Cue (Agent E) ── newly trained (Reviewer 2 control)
    if COUPLED_NO_CUE_RUN_ID:
        paths["coupled_no_cue"] = _model_path(COUPLED_NO_CUE_RUN_ID, "coupled_no_cue", seed)
    else:
        candidates = glob.glob(f"models/*/final_model_coupled_no_cue_seed{seed}_v2.zip")
        if not candidates:
            raise FileNotFoundError(
                f"No coupled_no_cue model found for seed {seed}. "
                f"Set COUPLED_NO_CUE_RUN_ID or check models/ directory. "
                f"Run 8-train_agents.py with MODE='coupled_no_cue' first."
            )
        paths["coupled_no_cue"] = max(candidates, key=os.path.getctime)

    print("  Model paths:")
    for name, p in paths.items():
        exists = "✓" if os.path.exists(p) else "✗ MISSING"
        print(f"    {name:14s}: {p}  [{exists}]")
    print()

    # ── Run evaluations ───────────────────────────────────────────────────────
    # encoder_path is needed for 'experimental' and 'oracle' modes:
    #   - experimental: encoder predicts physics from texture image
    #   - oracle:       env loads encoder so texture renders correctly,
    #                   but _physics_pred is overridden with ground truth
    #                   (the encoder is NOT called for the prediction itself)
    results = {}

    results["no_dr"] = evaluate(
        paths["no_dr"],
        "no_dr",
        num_episodes=NUM_EVAL_EPISODES,
        eval_threshold=EVAL_THRESHOLD,
        encoder_path=None,  # no texture, no encoder needed
        label=f"Agent D — NO-DR (fixed physics, seed {seed})",
    )

    results["baseline"] = evaluate(
        paths["baseline"],
        "baseline",
        num_episodes=NUM_EVAL_EPISODES,
        eval_threshold=EVAL_THRESHOLD,
        encoder_path=None,  # baseline never uses encoder
        label=f"Agent A — BASELINE (standard DR, seed {seed})",
    )

    results["coupled_no_cue"] = evaluate(
        paths["coupled_no_cue"],
        "coupled_no_cue",
        num_episodes=NUM_EVAL_EPISODES,
        eval_threshold=EVAL_THRESHOLD,
        encoder_path=None,  # cue withheld by design — never calls encoder
        label=f"Agent E — COUPLED, NO CUE (Reviewer 2 control, seed {seed})",
    )

    results["experimental"] = evaluate(
        paths["experimental"],
        "experimental",
        num_episodes=NUM_EVAL_EPISODES,
        eval_threshold=EVAL_THRESHOLD,
        encoder_path=ENCODER_PATH,  # encoder prediction in dims 9-10
        label=f"Agent B — EXPERIMENTAL (semantic DR, seed {seed})",
    )

    results["oracle"] = evaluate(
        paths["oracle"],
        "oracle",
        num_episodes=NUM_EVAL_EPISODES,
        eval_threshold=EVAL_THRESHOLD,
        encoder_path=ENCODER_PATH,  # needed so env renders textures correctly
        label=f"Agent C — ORACLE (ground-truth physics, seed {seed})",
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  ABLATION SUMMARY — Seed {seed}")
    print("  Performance spectrum: No-DR → Baseline → Coupled-No-Cue → Experimental → Oracle")
    print(f"{'=' * 62}")

    order = ["no_dr", "baseline", "coupled_no_cue", "experimental", "oracle"]
    labels = {
        "no_dr": "No-DR          (Agent D)",
        "baseline": "Baseline       (Agent A)",
        "coupled_no_cue": "Coupled-NoCue  (Agent E)",
        "experimental": "Experimental   (Agent B)",
        "oracle": "Oracle         (Agent C)",
    }

    for name in order:
        r = results[name]
        print(
            f"  {labels[name]} : "
            f"{r['success_rate'] * 100:5.1f}%  "
            f"CI [{r['ci_lower'] * 100:.1f}%, {r['ci_upper'] * 100:.1f}%]  "
            f"mean_dist={r['mean_final_distance'] * 100:.1f}cm"
        )

    print()

    # Key diagnostic gaps
    r_nd = results["no_dr"]["success_rate"]
    r_bl = results["baseline"]["success_rate"]
    r_e = results["coupled_no_cue"]["success_rate"]
    r_exp = results["experimental"]["success_rate"]
    r_or = results["oracle"]["success_rate"]

    gap_enc = (r_or - r_exp) * 100  # encoder imperfection cost
    gap_task = (r_bl - r_exp) * 100  # task difficulty cost (conflates
    # distribution + Izz + cue — see the
    # decomposition below)
    gap_dr = (r_bl - r_nd) * 100  # benefit of DR over no-DR

    # Decomposition of gap_task into "distribution alone" vs "cue alone",
    # using Agent E (coupled distribution + Izz, cue withheld) as the pivot.
    gap_dist_alone = (r_bl - r_e) * 100  # cost of coupled distribution
    # + asymmetric Izz, with NO cue
    gap_cue_value = (r_exp - r_e) * 100  # value the cue adds, holding the
    # coupled distribution + Izz fixed

    print("  Key gaps:")
    print(f"    Oracle - Experimental   = {gap_enc:+.1f}pp  ← cost of encoder imperfection")
    print(
        f"    Baseline - Experimental = {gap_task:+.1f}pp  "
        f"← cost of harder task (distribution + Izz + cue, combined —"
    )
    print("                                see decomposition below)")
    print(f"    Baseline - No-DR        = {gap_dr:+.1f}pp  ← benefit of DR over fixed physics")
    print()
    print("  Decomposition (Reviewer 2 control — Agent E):")
    print(
        f"    Baseline - Coupled-NoCue   = {gap_dist_alone:+.1f}pp  "
        f"← cost of the coupled distribution + asymmetric Izz"
    )
    print("                                  ALONE, before any cue is added")
    print(
        f"    Experimental - Coupled-NoCue = {gap_cue_value:+.1f}pp  "
        f"← value the cue itself adds, holding distribution"
    )
    print("                                  and Izz fixed (cleanest isolation")
    print("                                  of what the cue buys)")
    print()

    # Interpretation
    print("  Interpretation:")
    if r_or >= 0.90:
        print("  ✓ Oracle ≥ 90%: the task CAN be solved with perfect physics")
        print("    signal. Encoder quality is the primary remaining bottleneck.")
        print("    → Lab sim-to-real encoder test is strongly motivated.")
        print("    → Improving encoder (more images, SDXL) would close the gap.")
    elif r_or >= 0.80:
        print("  ~ Oracle 80-90%: task difficulty is a significant factor.")
        print("    Even perfect physics knowledge does not fully close the gap.")
        print("    → Both encoder quality AND task complexity limit performance.")
    else:
        print("  ✗ Oracle < 80%: the RL algorithm cannot exploit the signal")
        print("    even when it is perfect. Algorithmic limitation confirmed.")
        print("    → Consider reducing Izz asymmetry or denser reward shaping.")

    if r_nd >= 0.95:
        print("\n  ✓ No-DR ≥ 95%: env is solvable without any randomisation.")
        print(f"    Baseline's {r_bl * 100:.0f}% confirms DR adds difficulty but is")
        print("    well within expected range.")
    elif r_nd < 0.85:
        print("\n  ✗ No-DR < 85%: WARNING — something is wrong. A fully")
        print("    deterministic env should be trivially solvable. Check")
        print("    that no_dr mode is correctly implemented.")

    print("\n  Reviewer 2 interpretation (is the cue doing real work, or is")
    print("  the coupled distribution alone the whole story?):")
    if gap_cue_value >= gap_dist_alone and gap_cue_value > 5.0:
        print(f"  ✓ Cue value ({gap_cue_value:+.1f}pp) exceeds distribution-alone")
        print(f"    cost ({gap_dist_alone:+.1f}pp): most of the Baseline-vs-")
        print("    Experimental gap reflects the cue doing real work, not just")
        print("    a harder marginal distribution.")
    elif gap_dist_alone > gap_cue_value and gap_dist_alone > 5.0:
        print(f"  ~ Distribution-alone cost ({gap_dist_alone:+.1f}pp) exceeds cue")
        print(f"    value ({gap_cue_value:+.1f}pp): most of the Baseline-vs-")
        print("    Experimental gap is the harder coupled distribution and")
        print("    asymmetric Izz, not the cue itself. This does not undermine")
        print("    the mismatch result (a within-condition test, unaffected by")
        print("    this), but §III-D's framing should reflect it honestly.")
    else:
        print(f"  ~ Distribution-alone cost ({gap_dist_alone:+.1f}pp) and cue value")
        print(f"    ({gap_cue_value:+.1f}pp) are both small/comparable — inconclusive")
        print("    at this seed; check the pooled result across all 5 seeds.")

    # ── Statistical comparisons ───────────────────────────────────────────────
    print("\n  Statistical comparisons (two-proportion z-test):")
    pairs = [
        ("baseline", "experimental", "Baseline      vs Experimental  "),
        ("oracle", "experimental", "Oracle        vs Experimental  "),
        ("baseline", "no_dr", "Baseline      vs No-DR         "),
        ("oracle", "baseline", "Oracle        vs Baseline      "),
        ("baseline", "coupled_no_cue", "Baseline      vs Coupled-NoCue "),
        ("experimental", "coupled_no_cue", "Experimental  vs Coupled-NoCue "),
    ]
    comparisons = {}
    for a, b, label_pair in pairs:
        z, p = two_proportion_z_test(
            results[a]["successes"],
            results[a]["episodes"],
            results[b]["successes"],
            results[b]["episodes"],
        )
        delta = (results[a]["success_rate"] - results[b]["success_rate"]) * 100
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    {label_pair}: Δ={delta:+.1f}pp  p={p:.4e}  {sig}")
        comparisons[f"{a}_vs_{b}"] = {"z": z, "p": p, "delta_pp": delta, "significant": p < 0.05}

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "seed": seed,
        "no_dr": {k: v for k, v in results["no_dr"].items() if k != "final_distances"},
        "baseline": {k: v for k, v in results["baseline"].items() if k != "final_distances"},
        "coupled_no_cue": {
            k: v for k, v in results["coupled_no_cue"].items() if k != "final_distances"
        },
        "experimental": {
            k: v for k, v in results["experimental"].items() if k != "final_distances"
        },
        "oracle": {k: v for k, v in results["oracle"].items() if k != "final_distances"},
        "comparisons": comparisons,
        "gaps": {
            "oracle_minus_experimental_pp": gap_enc,
            "baseline_minus_experimental_pp": gap_task,
            "baseline_minus_nodr_pp": gap_dr,
            "baseline_minus_coupled_no_cue_pp": gap_dist_alone,
            "experimental_minus_coupled_no_cue_pp": gap_cue_value,
        },
    }

    out_path = f"ablation_results_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full ablation results saved → {out_path}")
    print(f"{'=' * 62}")
    return output


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if EVAL_MODE == "standard":
        run_standard_evaluation()

    elif EVAL_MODE == "mismatch":
        valid_paths = {seed: path for seed, path in AGENT_B_MODEL_PATHS.items() if path is not None}
        if not valid_paths:
            print(
                "Error: no AGENT_B_MODEL_PATHS set. Fill in the model paths at the top of the file."
            )
            raise SystemExit(1)
        run_mismatch_experiment(valid_paths, num_episodes=NUM_EVAL_EPISODES)

    elif EVAL_MODE == "ablation":
        run_ablation_evaluation()

    else:
        print(f"Error: EVAL_MODE must be 'standard', 'mismatch', or 'ablation', got '{EVAL_MODE}'")
        raise SystemExit(1)
