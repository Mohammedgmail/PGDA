# 18-train_specialist.py — Train a single-material specialist (no cue)
# Physically-Grounded Diffusion Augmentation (PGDA)
#
# PURPOSE
#   A policy trained on ONE material only is the best a cue-conditioned policy
#   could hope to do on that material (same algorithm, same budget). Comparing
#   specialists with the generalist agents bounds how much ANY material cue,
#   even a perfect one, could add (see 19-eval_specialists.py).
#
# WHAT IT TRAINS
#   Identical to 8-train_agents.py (SAC+HER, same hyperparameters loaded from
#   optuna_best_params.json, fixed ent_coef 0.1, 1M steps, 200-step episodes),
#   but on the coupled dynamics restricted to a single category:
#     - environment mode 'coupled_no_cue' (category-coupled mass, friction and
#       inertia scaling; observation dims 9-10 held at zero), and
#     - a wrapper that re-draws the episode until its category is CATEGORY.
#   The specialist therefore sees exactly the physics PGDA and the No-Cue agent
#   see for that material, with all 1M steps spent on it.
#
# RUN ORDER (priority)
#   CATEGORY = "Wet Ice", SEED = 42, 123, 456     <- decisive for the bound
#   (optional) "Heavy Rust" and "Coarse Sandpaper", same seeds, to tighten it
#
# IMPORTANT: train every specialist in the SAME Python environment as the
#   No-Cue agent (E), i.e. the environment you use now, and keep it fixed.

import os
import json
import numpy as np
import gymnasium as gym
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback)
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from pushing_env import UR3ePushingEnv

# ── 1. Configuration ──────────────────────────────────────────────────────────
CATEGORY = "Wet Ice"       # "Wet Ice" | "Heavy Rust" | "Coarse Sandpaper"
SEED     = 1011              # 42, 123, 456 (add 789, 1011 if time allows)
MODE     = "coupled_no_cue"

MAX_EPISODE_STEPS = 200
TOTAL_TIMESTEPS   = 1_000_000
GOAL_SELECTION_STRATEGY = "future"
BUFFER_SIZE = 1_000_000
ENT_COEF    = 0.1
SUCCESS_THRESHOLD      = 0.05
CHECKPOINT_EVERY_STEPS = 50_000
WANDB_PROJECT = "physically-grounded-diffusion"   # same project as 8-train_agents.py

PARAMS_FILE = "optuna_best_params.json"           # same file as 8-train_agents.py
_SAC_DEFAULTS = {"learning_rate": 1e-3, "tau": 0.005, "gamma": 0.95,
                 "batch_size": 256, "learning_starts": 1000, "n_sampled_goal": 4}
if os.path.exists(PARAMS_FILE):
    with open(PARAMS_FILE) as _f:
        _SAC_PARAMS = json.load(_f).get("sac_hyperparams", _SAC_DEFAULTS)
    print(f"[Config] Loaded SAC hyperparameters from {PARAMS_FILE}")
else:
    raise SystemExit(f"{PARAMS_FILE} not found: the specialists must use the same "
                     f"hyperparameters as the other agents. Run from the folder "
                     f"that contains it.")
LEARNING_RATE   = float(_SAC_PARAMS["learning_rate"])
TAU             = float(_SAC_PARAMS["tau"])
GAMMA           = float(_SAC_PARAMS["gamma"])
BATCH_SIZE      = int(_SAC_PARAMS["batch_size"])
LEARNING_STARTS = int(_SAC_PARAMS["learning_starts"])
N_SAMPLED_GOAL  = int(_SAC_PARAMS["n_sampled_goal"])

SLUG = CATEGORY.lower().replace(" ", "_")          # e.g. "wet_ice"
CATEGORIES = ["Heavy Rust", "Wet Ice", "Coarse Sandpaper"]
assert CATEGORY in CATEGORIES, f"CATEGORY must be one of {CATEGORIES}"


# ── 2. Single-category wrapper ────────────────────────────────────────────────

class CategoryFilter(gym.Wrapper):
    """Re-draws the episode until the environment picks `category`.

    The environment samples the category itself (texture -> category ->
    coupled physics), so this keeps its within-category distribution exactly
    as in the other agents' training. Only the first reset of a call receives
    the seed; re-draws continue the random stream.
    """

    def __init__(self, env, category, max_tries=500):
        super().__init__(env)
        self.category, self.max_tries = category, max_tries
        self.n_draws, self.n_kept = 0, 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        kwargs.pop("seed", None)
        tries = 1
        while getattr(self.env.unwrapped, "_current_category", None) != self.category:
            if tries >= self.max_tries:
                raise RuntimeError(
                    f"No '{self.category}' episode in {tries} draws — check the "
                    f"category name against env.unwrapped._current_category.")
            obs, info = self.env.reset(**kwargs)
            tries += 1
        self.n_draws += tries
        self.n_kept += 1
        return obs, info


# ── 3. Callbacks ──────────────────────────────────────────────────────────────

class SuccessRateCallback(BaseCallback):
    """Same logging as 8-train_agents.py (success/rate every 100 episodes)."""

    def __init__(self, log_every=100, verbose=1):
        super().__init__(verbose=verbose)
        self.log_every, self._ep_count, self._ep_successes, self._window_start = \
            log_every, 0, 0, 0

    def _on_step(self):
        for info, done in zip(self.locals.get("infos", [{}]),
                              self.locals.get("dones", [False])):
            if done:
                self._ep_count += 1
                self._ep_successes += int(info.get("is_success", 0))
        window = self._ep_count - self._window_start
        if window >= self.log_every:
            rate = self._ep_successes / window
            self.logger.record("success/rate", rate)
            self.logger.record("success/episodes_total", self._ep_count)
            if self.verbose:
                print(f"  [Step {self.num_timesteps:,}]  {CATEGORY} success "
                      f"(last {window} eps): {rate * 100:.1f}%")
            self._window_start, self._ep_successes = self._ep_count, 0
        return True


# ── 4. Environment ────────────────────────────────────────────────────────────

def make_env(seed):
    env = UR3ePushingEnv(render_mode=None, randomization_mode=MODE,
                         reward_weights={"success_threshold": SUCCESS_THRESHOLD},
                         encoder_path=None)
    env = CategoryFilter(env, CATEGORY)            # inside TimeLimit
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ── 5. Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_name = f"Specialist_{SLUG}_SAC_HER_seed{SEED}"
    run = wandb.init(
        project=WANDB_PROJECT, name=run_name, mode="online", save_code=True,
        config=dict(algorithm="SAC+HER", seed=SEED, total_steps=TOTAL_TIMESTEPS,
                    max_episode_steps=MAX_EPISODE_STEPS, randomization_mode=MODE,
                    specialist_category=CATEGORY, physics_signal="none",
                    agent_label=f"Specialist_{SLUG}",
                    success_threshold=SUCCESS_THRESHOLD,
                    n_sampled_goal=N_SAMPLED_GOAL,
                    goal_selection_strategy=GOAL_SELECTION_STRATEGY,
                    learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
                    learning_starts=LEARNING_STARTS, batch_size=BATCH_SIZE,
                    tau=TAU, gamma=GAMMA, ent_coef=ENT_COEF,
                    strategy="single_category_specialist"),
        sync_tensorboard=True)

    print(f"\n{'='*60}\n  [{run_name}]  category = {CATEGORY}, seed = {SEED}\n"
          f"  lr = {LEARNING_RATE:.3e}, batch = {BATCH_SIZE}, gamma = {GAMMA}, "
          f"tau = {TAU}\n{'='*60}\n")

    env = make_env(SEED)
    os.makedirs(f"models/{run.id}/checkpoints", exist_ok=True)

    model = SAC("MultiInputPolicy", env, verbose=1,
                learning_rate=LEARNING_RATE, buffer_size=BUFFER_SIZE,
                learning_starts=LEARNING_STARTS, batch_size=BATCH_SIZE,
                tau=TAU, gamma=GAMMA, ent_coef=ENT_COEF, target_entropy=-2.0,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=N_SAMPLED_GOAL,
                    goal_selection_strategy=GOAL_SELECTION_STRATEGY),
                tensorboard_log=f"runs/{run.id}", seed=SEED, device="cpu")

    callbacks = CallbackList([
        SuccessRateCallback(log_every=100, verbose=1),
        CheckpointCallback(save_freq=CHECKPOINT_EVERY_STEPS,
                           save_path=f"models/{run.id}/checkpoints/",
                           name_prefix=f"ckpt_{run_name}", verbose=1),
        WandbCallback(gradient_save_freq=5_000,
                      model_save_path=f"models/{run.id}", verbose=2),
    ])

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks,
                    log_interval=10)
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving.")

    out = f"models/{run.id}/final_model_specialist_{SLUG}_seed{SEED}"
    model.save(out)
    filt = env.env.env   # Monitor -> TimeLimit -> CategoryFilter
    if isinstance(filt, CategoryFilter) and filt.n_kept:
        print(f"  Category filter: {filt.n_kept} episodes kept from "
              f"{filt.n_draws} draws ({100 * filt.n_kept / filt.n_draws:.0f}%).")
    env.close()
    wandb.finish()
    print(f"\n[{run_name}] done.\n  Model -> {out}.zip\n  Run ID: {run.id} "
          f"(paste it into SPECIALIST_RUN_IDS in 19-eval_specialists.py)")
