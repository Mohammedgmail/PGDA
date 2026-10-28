# PGDA: Physically-Grounded Diffusion Augmentation

[![CI](https://github.com/Mohammedgmail/PGDA/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohammedgmail/PGDA/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C)
![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-4B5563)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Code for the paper:** *"Coupling Appearance and Physics in Domain Randomisation: Do
Manipulation Policies Rely on Visual Material Cues?"*, submitted to *Robotics and Autonomous Systems* (Elsevier).

---

## Overview

Domain randomisation (DR) typically samples visual appearance and dynamics **independently**. Real objects don't behave this way: an object's appearance and its dynamics share a common cause, its **material**. Steel looks heavy and ice looks slippery.

**PGDA (Physically-Grounded Diffusion Augmentation)** restores this link. Each diffusion-generated texture belongs to a material category that sets the object's mass, friction and inertia ranges. A physics-prediction encoder recognises the category from the texture and passes the corresponding code to a reinforcement-learning policy (SAC+HER) for planar pushing with a UR3e arm.

We then ask two questions: does the policy **rely** on the visual material cue, and does it **need** it?

| Test | Question | Answer |
|------|----------|--------|
| **Mismatch**: wrong material code, physics unchanged | Does the policy *rely* on the cue? | **Yes**: success drops by 11.0 pp in simulation (every seed) and from 96.7% to 20.0% on a real UR3e |
| **No-Cue control**: same coupled dynamics, cue held at zero | Does the policy *need* the cue? | **No**: the No-Cue agent does at least as well (84.0% vs 79.0%) |
| **Oracle**: ground-truth physics in the observation | Would a better cue help? | **No**: equivalent to PGDA within ±5 pp |
| **Specialist**: trained on Wet Ice alone | How much could any cue add? | **At most 3.4 pp** over the No-Cue agent |

On a second task where materials need **conflicting pushes**, a perfect cue adds only 5.1 pp to scripted pushes that react to contact and 11.8 pp to committed ones, because **the first contact reveals the material within one control step**.

On a real UR3e, label-conditioned control transfers (96.7% with the correct label), but the encoder trained on synthetic textures classifies real materials with only 46.7% accuracy: **perception, not control, limits autonomous deployment**.

---

## Installation

```bash
git clone https://github.com/Mohammedgmail/PGDA.git
cd PGDA
pip install -e ".[dev]"          # package, tests and linting
pip install -e ".[eval]"         # analysis scripts (matplotlib, seaborn, scipy)
pip install -e ".[textures]"     # texture generation with Stable Diffusion
```

Requires Python 3.10 to 3.13. CI tests Linux on all four versions.

**PyBullet wheels.** PyBullet publishes prebuilt Linux wheels for Python 3.10 and 3.11 only. On Python 3.12 and 3.13, pip builds it from source, which needs a C++ compiler and takes a few minutes.

**macOS.** PyBullet has no macOS wheels, and its source build fails with current Apple compilers. Install it from conda-forge first, then install this package:

```bash
conda install -c conda-forge pybullet
pip install -e ".[dev]"
```

### Simulation assets (not included)

The repository contains the physics mapping (`assets/semantic_physics_mapping.json`). The robot model, the object models and the texture bank are not included. All paths are resolved from the repository root; set `PGDA_ROOT` to use another folder.

```
PGDA/
├── ur_description/urdf/ur3e_rg2.urdf     # UR3e arm with RG2 gripper
├── assets/
│   ├── semantic_physics_mapping.json     # included
│   ├── train_objects/<name>/<name>.urdf  # 17 Google Scanned Objects as URDF
│   └── texture_bank/<Category>/*.png     # 1,000 textures per category
└── texture_encoder.pt                    # written by python -m pgda.train_encoder
```

- **Texture bank:** generate it with `python -m pgda.texture_bank` (needs the `textures` extra and a GPU), or see [assets/README.md](assets/README.md).
- **Encoder:** `python -m pgda.train_encoder` writes `texture_encoder.pt` to the repository root. Set `PGDA_ENCODER_PATH` to load or save it elsewhere.

The simulation tests are skipped, with a reason, when the robot or object models are missing.

---

## Quick start

### 1. Train the texture encoder

```bash
python -m pgda.train_encoder
```

### 2. Train agents

Each script reads its settings from constants at the top of the file. The most common ones can also be set with environment variables:

```bash
# Agent B (PGDA): coupled DR + encoder cue
PGDA_MODE=experimental PGDA_SEED=42 python -m pgda.train_agents

# Agent A (Baseline): uncorrelated DR, no cue
PGDA_MODE=baseline PGDA_SEED=42 python -m pgda.train_agents

# Agents C, D, E: Oracle, No-DR, No-Cue
PGDA_MODE=oracle         PGDA_SEED=42 python -m pgda.train_agents
PGDA_MODE=no_dr          PGDA_SEED=42 python -m pgda.train_agents
PGDA_MODE=coupled_no_cue PGDA_SEED=42 python -m pgda.train_agents

# Single-material specialist
PGDA_CATEGORY="Wet Ice" PGDA_SEED=42 python -m pgda.train_specialist
```

The paper uses seeds 42, 123, 456, 789 and 1011, with 1M steps per run.

### 3. Evaluate

The evaluation scripts load trained checkpoints from the paths and run IDs set at the top of each file; update these before running.

```bash
PGDA_EVAL_MODE=standard python -m pgda.evaluate_agents
PGDA_EVAL_MODE=mismatch python -m pgda.evaluate_agents
PGDA_EVAL_MODE=ablation python -m pgda.evaluate_agents
```

| Script | Paper result |
|--------|--------------|
| `pgda.evaluate_agents` | Coupled-dynamics evaluation and mismatch experiments (Table 3, Table F.1) |
| `pgda.eval_specialists` | Specialist comparison and the bound on any cue (Section 4.4, Table I.1) |
| `pgda.behaviour_analysis` | Success and push behaviour under each material code (Table J.1; wrong-code average in Table 4) |
| `pgda.eval_feedback_noise` | Corrupted-feedback robustness (Table G.1) |
| `pgda.baseline_label_check` | Baseline with the supplied label (Table H.1) |
| `pgda.scripted_voi_task2`, `pgda.voi_followup_task2` | Task 2 pilot gate (Table 5, Tables K.1 and K.2) |
| `pgda.optuna_tune` | Hyperparameter search (Table A.1) |

---

## Key results (from the paper)

### Task 1: planar pushing, all agents on the coupled dynamics

Five seeds × 100 episodes, mean ± s.d. over seeds; within a seed, all agents face the same episodes.

| Agent | Description | Success (%) |
|-------|-------------|-------------|
| Baseline (A) | Uncorrelated DR, no cue | 83.6 ± 4.6 |
| **PGDA (B)** | **Coupled DR + encoder cue** | **79.0 ± 5.9** |
| Oracle (C) | Coupled DR + ground-truth physics | 80.6 ± 4.6 |
| No-DR (D) | Fixed physics, no cue | 82.4 ± 4.0 |
| No-Cue (E) | Coupled DR, cue held at zero | 84.0 ± 4.5 |

On their own uncoupled training distribution, Baseline and No-DR reach 98.4 ± 0.5 and 98.8 ± 1.0.

- **Mismatch (B):** 81.0 ± 4.0 with the true code, 70.0 ± 4.3 with a wrong code (−11.0 pp, lower in every seed, p = 0.004).
- **Bound:** a Wet Ice specialist (52.8 ± 8.2) does no better than the No-Cue agent on Wet Ice (59.8 ± 4.6), so any cue could add at most 3.4 pp.

### Task 2: single-push sliding (scripted policies)

| Push | Gain from a perfect category cue (95% interval) |
|------|-------------------------------------------------|
| With feedback (react to contact) | +5.1 pp [+3.7, +6.7] |
| Committed strike | +11.8 pp [+10.3, +13.4] |

One control step after a gentle strike, the object's speed identifies Wet Ice in 95.0% of episodes.

### Sim-to-real (UR3e, 30 episodes per cell)

| Condition | Heavy Rust | Coarse Sandpaper |
|-----------|------------|------------------|
| PGDA + correct label | 96.7% | 96.7% |
| PGDA + wrong label | 20.0% | 20.0% |
| PGDA + no label | 3.3% | 0.0% |
| Baseline + label | 70.0% | 63.3% |

The encoder classifies real materials with 46.7% accuracy (N = 90; chance is 33.3%).

---

## Project structure

```
PGDA/
├── src/pgda/
│   ├── env.py                    # UR3e planar pushing environment (PyBullet + Gymnasium)
│   ├── env_task2.py              # Task 2: single-push sliding
│   ├── paths.py                  # where data files are looked up (PGDA_ROOT)
│   ├── texture_bank.py           # Stable Diffusion texture generation
│   ├── train_encoder.py          # ResNet-18 material encoder
│   ├── train_agents.py           # SAC+HER training (agents A to E)
│   ├── train_specialist.py       # single-material specialist
│   ├── optuna_tune.py            # hyperparameter search
│   ├── evaluate_agents.py        # standard, mismatch and ablation evaluation
│   ├── eval_specialists.py       # specialists and the bound on any cue
│   ├── eval_feedback_noise.py    # corrupted-feedback robustness
│   ├── baseline_label_check.py   # Baseline with the supplied label
│   ├── behaviour_analysis.py     # how the policy uses the material code
│   ├── learning_speed_check.py   # learning-speed check
│   ├── scripted_voi_task2.py     # Task 2 pilot gate
│   ├── voi_followup_task2.py     # Task 2 follow-up analysis
│   └── texture_test.py           # interactive texture demo (opens a PyBullet window)
├── tests/                        # unit tests
├── examples/                     # example configurations
├── assets/                       # physics mapping; texture bank and objects go here
├── .github/workflows/ci.yml      # CI: lint + tests on Linux, Python 3.10 to 3.13
├── pyproject.toml
└── LICENSE
```

---

## Citation

```bibtex
@article{alfandi2026pgda,
  title   = {Used but Not Needed: Testing Whether Manipulation Policies Rely on Visual Material Cues in Domain Randomisation},
  author  = {Alfandi, Mohammed and Wang, Shidong and Shao, Zhuang and Adhikari, Kabita},
  journal = {Robotics and Autonomous Systems},
  note    = {Submitted},
  year    = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
