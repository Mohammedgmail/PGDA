# 8-train_agents.py — SAC + HER Training Pipeline
# Physically-Grounded Diffusion Augmentation for Robust VLA (MSc Project)
#
# Supports five modes:
#   baseline       → Agent A: uncorrelated DR, no encoder
#   experimental   → Agent B: asymmetric Izz + encoder prediction in obs
#   oracle         → Agent C: asymmetric Izz + GROUND-TRUTH physics in obs
#   no_dr          → Agent D: fixed physics, grey object, no randomisation
#   coupled_no_cue → Agent E: coupled distribution/inertia, no encoder cue
#
# Run order for ablation study:
#   1. oracle   (seed 42 first) — upper bound on what encoder can achieve
#   2. no_dr    (seed 42 first) — lower bound / sanity check
#   Then run remaining seeds for whichever condition oracle result motivates.

import os
os.environ["TORCH_DYNAMO_DISABLE"] = "1"
import numpy as np
import wandb
import json
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback)
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from pushing_env import UR3ePushingEnv

# ── 1. Configuration ──────────────────────────────────────────────────────────
# ↓↓↓  CHANGE THIS LINE ONLY BETWEEN RUNS  ↓↓↓
MODE = "coupled_no_cue"   # 'baseline' | 'experimental' | 'coupled_no_cue'
# ↑↑↑─────────────────────────────────────────↑↑↑

ENCODER_PATH = '/Users/mohammedalfandi/Documents/Robotics and AI /MSc Project /Codes-2026/texture_encoder.pt'

# Change SEED for each run: 42, 123, 456, 789, 1011
SEED = 1011

MAX_EPISODE_STEPS  = 200
TOTAL_TIMESTEPS    = 1_000_000   # 1M — oracle/no_dr plateau at same rate

# Oracle and no_dr do NOT use the encoder at inference time.
# Encoder path is passed to env but only used if MODE == "experimental".
_ENCODER_MODES = {"experimental", "oracle"}   # modes that load encoder weights

# ── Load SAC hyperparameters ──────────────────────────────────────────────────
PARAMS_FILE = "optuna_best_params.json"

_SAC_DEFAULTS = {
    "learning_rate":   1e-3,
    "tau":             0.005,
    "gamma":           0.95,
    "batch_size":      256,
    "learning_starts": 1000,
    "n_sampled_goal":  4,
}

if os.path.exists(PARAMS_FILE):
    with open(PARAMS_FILE) as _f:
        _loaded = json.load(_f)
    _SAC_PARAMS = _loaded.get("sac_hyperparams", _SAC_DEFAULTS)
    print(f"[Config] Loaded SAC hyperparameters from {PARAMS_FILE}")
else:
    _SAC_PARAMS = _SAC_DEFAULTS
    print(f"[Config] {PARAMS_FILE} not found — using defaults.")

LEARNING_RATE   = float(_SAC_PARAMS.get("learning_rate",   _SAC_DEFAULTS["learning_rate"]))
TAU             = float(_SAC_PARAMS.get("tau",             _SAC_DEFAULTS["tau"]))
GAMMA           = float(_SAC_PARAMS.get("gamma",           _SAC_DEFAULTS["gamma"]))
BATCH_SIZE      = int(  _SAC_PARAMS.get("batch_size",      _SAC_DEFAULTS["batch_size"]))
LEARNING_STARTS = int(  _SAC_PARAMS.get("learning_starts", _SAC_DEFAULTS["learning_starts"]))
N_SAMPLED_GOAL  = int(  _SAC_PARAMS.get("n_sampled_goal",  _SAC_DEFAULTS["n_sampled_goal"]))

BUFFER_SIZE = 1_000_000
ENT_COEF    = 0.1

SUCCESS_THRESHOLD      = 0.05
CHECKPOINT_EVERY_STEPS = 50_000


# ── 2. Success rate callback ──────────────────────────────────────────────────

class SuccessRateCallback(BaseCallback):
    def __init__(self, log_every: int = 100, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.log_every     = log_every
        self._ep_count     = 0
        self._ep_successes = 0
        self._window_start = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        dones = self.locals.get("dones", [False])
        for info, done in zip(infos, dones):
            if done:
                self._ep_count     += 1
                self._ep_successes += int(info.get("is_success", 0))

        window = self._ep_count - self._window_start
        if window >= self.log_every:
            rate = self._ep_successes / window
            self.logger.record("success/rate",           rate)
            self.logger.record("success/episodes_total", self._ep_count)
            if self.verbose:
                print(f"  [Step {self.num_timesteps:,}]  "
                      f"Success rate (last {window} eps): {rate * 100:.1f}%  "
                      f"({self._ep_successes}/{window})")
            self._window_start = self._ep_count
            self._ep_successes = 0
        return True


# ── 3. Environment factory ────────────────────────────────────────────────────

def make_env(seed: int = 42, encoder_path: str = None) -> Monitor:
    """
    Builds a single monitored GoalEnv instance.
    encoder_path is only used when MODE is in _ENCODER_MODES.
    For oracle mode the env loads the encoder weights but does NOT
    call the encoder during inference — it uses ground-truth physics.
    For no_dr the encoder is not loaded at all.
    """
    env = UR3ePushingEnv(
        render_mode=None,
        randomization_mode=MODE,
        reward_weights={'success_threshold': SUCCESS_THRESHOLD},
        encoder_path=encoder_path,
    )
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


# ── 4. Main execution block ───────────────────────────────────────────────────

if __name__ == "__main__":

    AGENT_LABELS = {
        'baseline':       'A',
        'experimental':   'B',
        'oracle':         'C',
        'no_dr':          'D',
        'coupled_no_cue': 'E',
    }
    agent_label = AGENT_LABELS.get(MODE, MODE)
    run_name    = f"Agent_{agent_label}_{MODE}_SAC_HER_seed{SEED}"

    # Encoder is passed to env only for modes that need it.
    # oracle: env loads weights so texture can still be rendered,
    #         but _physics_pred comes from ground truth, not encoder.
    # no_dr:  no texture, no encoder needed.
    encoder_arg = ENCODER_PATH if MODE in _ENCODER_MODES else None

    run = wandb.init(
        project="physically-grounded-diffusion",
        name=run_name,
        mode="offline",          # safer: avoids "Crashed" on connection loss
        save_code=True,
        config=dict(
            algorithm="SAC+HER",
            seed=SEED,
            total_steps=TOTAL_TIMESTEPS,
            max_episode_steps=MAX_EPISODE_STEPS,
            randomization_mode=MODE,
            success_threshold=SUCCESS_THRESHOLD,
            n_sampled_goal=N_SAMPLED_GOAL,
            goal_selection_strategy="future",
            learning_rate=LEARNING_RATE,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=GAMMA,
            ent_coef=ENT_COEF,
            encoder_path=encoder_arg,
            strategy="S1+S2_ablation",
            agent_label=f"Agent_{agent_label}",
            physics_signal=(
                "ground_truth" if MODE == "oracle" else
                "encoder_pred" if MODE == "experimental" else
                "none"
            ),
        ),
        sync_tensorboard=True,
    )

    print(f"\n{'='*60}")
    print(f"  [{run_name}]")
    print(f"  Algorithm      : SAC + HER")
    print(f"  Mode           : {MODE}")
    print(f"  Agent          : Agent {agent_label}")
    print(f"  Seed           : {SEED}")
    print(f"  Total steps    : {TOTAL_TIMESTEPS:,}")
    print(f"  Episode length : {MAX_EPISODE_STEPS} steps")
    print(f"  Physics signal : {'ground-truth (oracle)' if MODE == 'oracle' else 'encoder prediction' if MODE == 'experimental' else 'none'}")
    print(f"  Encoder path   : {encoder_arg or 'N/A'}")
    print(f"{'='*60}\n")

    env = make_env(seed=SEED, encoder_path=encoder_arg)

    os.makedirs(f"models/{run.id}/checkpoints", exist_ok=True)

    print(f"  learning_rate   = {LEARNING_RATE:.2e}")
    print(f"  tau             = {TAU}")
    print(f"  gamma           = {GAMMA}")
    print(f"  batch_size      = {BATCH_SIZE}")
    print(f"  learning_starts = {LEARNING_STARTS}")
    print(f"  n_sampled_goal  = {N_SAMPLED_GOAL}")

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        target_entropy=-2.0,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=N_SAMPLED_GOAL,
            goal_selection_strategy="future",
        ),
        tensorboard_log=f"runs/{run.id}",
        seed=SEED,
        device="cpu",
    )

    success_cb = SuccessRateCallback(log_every=100, verbose=1)

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY_STEPS,
        save_path=f"models/{run.id}/checkpoints/",
        name_prefix=f"ckpt_{run_name}",
        verbose=1,
    )

    wandb_cb = WandbCallback(
        gradient_save_freq=5_000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    callbacks = CallbackList([success_cb, checkpoint_cb, wandb_cb])

    print(f"[{run_name}] Starting {TOTAL_TIMESTEPS:,}-step SAC+HER training...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            log_interval=10,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving checkpoint...")

    save_name = f"final_model_{MODE}_seed{SEED}_v2"
    model.save(f"models/{run.id}/{save_name}")
    env.close()

    # Sync offline run to WandB
    import subprocess
    subprocess.run(
        ["wandb", "sync", f"wandb/{run.id}"],
        check=False)

    wandb.finish()

    print(f"\n[{run_name}] Training complete.")
    print(f"  Final model → models/{run.id}/{save_name}.zip")
    print(f"  Checkpoints → models/{run.id}/checkpoints/")

    # Sync any remaining offline runs
    import glob
    for run_dir in glob.glob(os.path.join(os.getcwd(), "wandb", "offline-run-*")):
        print(f"Syncing: {run_dir}")
        subprocess.run(["wandb", "sync", run_dir], check=False)