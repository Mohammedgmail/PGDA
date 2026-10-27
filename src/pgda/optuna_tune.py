# 7-optuna_tune.py — SAC+HER Hyperparameter Optimisation
# Physically-Grounded Diffusion Augmentation for Robust VLA (MSc Project)
#
# With SAC+HER and sparse reward (0/-1), there are NO reward weights to tune.
# HER provides the learning signal by relabelling goals — dense shaping is
# not needed. Optuna's job is purely to find the best SAC hyperparameters:
#
#   learning_rate   — how fast the actor/critic networks update
#   tau             — soft target network update rate
#   gamma           — discount factor
#   batch_size      — transitions sampled per gradient step
#   learning_starts — random exploration steps before SAC begins updating
#   n_sampled_goal  — HER virtual goals per real transition
#
# Runtime: ~30 trials × ~15 min/trial ≈ 7–8 hours
# (single env per trial; SAC+HER requires non-vectorised environment)

import json

import numpy as np
import optuna
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from pgda.env import UR3ePushingEnv

PARAMS_FILE = "optuna_best_params.json"
N_TRIALS = 30
TUNE_STEPS = 20_000  # steps per trial
TUNE_EPISODE_STEPS = 200  # matches training config
TUNE_EVAL_EPISODES = 20  # enough episodes for a reliable success signal
SUCCESS_THRESHOLD = 0.05  # metres — same as training and evaluation


# ── Trial success tracker ─────────────────────────────────────────────────────


class TrialSuccessCallback(BaseCallback):
    """Tracks genuine success count during a trial (used for diagnostics)."""

    def __init__(self):
        super().__init__(verbose=0)
        self._ep_success = 0
        self._ep_total = 0

    def _on_step(self) -> bool:
        for info, done in zip(
            self.locals.get("infos", [{}]), self.locals.get("dones", [False]), strict=False
        ):
            if done:
                self._ep_total += 1
                self._ep_success += int(info.get("is_success", False))
        return True


# ── Environment factory ───────────────────────────────────────────────────────


def make_trial_env(seed: int = 42) -> Monitor:
    """
    Single monitored GoalEnv instance for one Optuna trial.
    SAC+HER requires a non-vectorised (single) environment — no SubprocVecEnv.
    """
    env = UR3ePushingEnv(
        render_mode=None,
        randomization_mode="baseline",
        reward_weights={"success_threshold": SUCCESS_THRESHOLD},
    )
    env = TimeLimit(env, max_episode_steps=TUNE_EPISODE_STEPS)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ── Objective function ────────────────────────────────────────────────────────


def objective(trial: optuna.Trial) -> float:

    # ── SAC hyperparameters to tune ───────────────────────────────────────────
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    tau = trial.suggest_categorical("tau", [0.005, 0.01, 0.02, 0.05])
    gamma = trial.suggest_categorical("gamma", [0.90, 0.95, 0.98, 0.99])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    learning_starts = trial.suggest_categorical("learning_starts", [500, 1000, 2000])
    n_sampled_goal = trial.suggest_categorical("n_sampled_goal", [2, 4, 6])

    # ── Build environment ─────────────────────────────────────────────────────
    try:
        env = make_trial_env(seed=42 + trial.number)
    except Exception as e:
        print(f"  [Trial {trial.number}] Env init failed: {e}")
        return -999.0

    # ── Train ─────────────────────────────────────────────────────────────────
    try:
        model = SAC(
            "MultiInputPolicy",
            env,
            verbose=0,
            learning_rate=lr,
            buffer_size=50_000,  # small for tuning — keeps RAM manageable
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            ent_coef="auto",  # automatic entropy tuning
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy="future",
            ),
            device="cpu",
            seed=42,
        )
        success_cb = TrialSuccessCallback()
        model.learn(
            total_timesteps=TUNE_STEPS,
            callback=success_cb,
            log_interval=99999,  # suppress per-step stdout
        )
    except Exception as e:
        print(f"  [Trial {trial.number}] Training failed: {e}")
        env.close()
        return -999.0

    # ── Evaluate ──────────────────────────────────────────────────────────────
    eval_env = UR3ePushingEnv(
        render_mode=None,
        randomization_mode="baseline",
        reward_weights={"success_threshold": SUCCESS_THRESHOLD},
    )
    eval_env.success_threshold = SUCCESS_THRESHOLD
    eval_env = TimeLimit(eval_env, max_episode_steps=TUNE_EPISODE_STEPS)

    success_count = 0
    total_dist = 0.0

    try:
        for _ep in range(TUNE_EVAL_EPISODES):
            obs, _ = eval_env.reset()
            done = truncated = False
            steps = 0
            ep_success = False

            while not (done or truncated) and steps < TUNE_EPISODE_STEPS:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = eval_env.step(action)
                steps += 1
                if info.get("is_success", False):
                    ep_success = True

            # Genuine push only — exclude step-1 spawn artefacts
            if ep_success and steps > 1:
                success_count += 1

            # Track final object→goal distance as tiebreaker
            import pybullet as p

            obj_pos = p.getBasePositionAndOrientation(
                eval_env.env.object_uid, physicsClientId=eval_env.env._client
            )[0]
            dist = float(
                np.linalg.norm(np.array(obj_pos[:2]) - np.array(eval_env.env.goal_pos[:2]))
            )
            total_dist += dist

    except Exception as e:
        print(f"  [Trial {trial.number}] Eval failed: {e}")
    finally:
        eval_env.close()
        env.close()

    success_rate = success_count / TUNE_EVAL_EPISODES
    mean_dist = total_dist / TUNE_EVAL_EPISODES

    # Primary score: success rate (0–1 range scaled to 0–100).
    # Tiebreaker: mean distance (closer is better).
    # max(0, 0.30 - mean_dist) rewards getting within 30 cm of goal.
    score = 100.0 * success_rate + max(0.0, 0.30 - mean_dist) * 10.0

    print(
        f"  Trial {trial.number:3d} | "
        f"successes={success_count}/{TUNE_EVAL_EPISODES} "
        f"({success_rate * 100:.0f}%) | "
        f"mean_dist={mean_dist * 100:.1f}cm | "
        f"score={score:.3f} | "
        f"lr={lr:.2e} tau={tau} gamma={gamma} "
        f"bs={batch_size} starts={learning_starts} n_goal={n_sampled_goal}"
    )
    return score


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  Optuna v3 — SAC+HER, Sparse Reward, Fixed Environment")
    print(f"  {N_TRIALS} trials × {TUNE_STEPS:,} steps  (single env per trial)")
    print("  Tuning: learning_rate, tau, gamma, batch_size,")
    print("          learning_starts, n_sampled_goal")
    print("  No reward weights — HER provides the learning signal.")
    print("=" * 65)

    sampler = optuna.samplers.TPESampler(n_startup_trials=8, seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="pushing_sac_her_v3",
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print(f"\n{'=' * 65}")
    print(f"  Best trial : #{best.number}  score={best.value:.4f}")
    print(f"{'=' * 65}")

    # ── Save in format expected by 8-train_agents.py ──────────────────────────
    sac_keys = {"learning_rate", "tau", "gamma", "batch_size", "learning_starts", "n_sampled_goal"}
    sac_params = {k: v for k, v in best.params.items() if k in sac_keys}

    output = {
        "best_trial": best.number,
        "best_score": best.value,
        "best_reward": best.value,  # alias kept for compatibility
        "sac_hyperparams": sac_params,
        "reward_weights": {"success_threshold": SUCCESS_THRESHOLD},
        "optuna_version": "v3_sac_her_sparse",
        "notes": (
            "SAC+HER with sparse reward (0/-1). No reward weights needed. "
            "Tuned: learning_rate, tau, gamma, batch_size, "
            "learning_starts, n_sampled_goal."
        ),
    }

    with open(PARAMS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Params saved → {PARAMS_FILE}")
    print("\n  SAC hyperparameters:")
    for k, v in sac_params.items():
        print(f"    {k:22s}: {v}")
    print("\n  → Update 8-train_agents.py with these values, then train seed 42.")
