# PGDA: Physically-Grounded Diffusion Augmentation

[![CI](https://github.com/Mohammedgmail/PGDA/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohammedgmail/PGDA/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C)
![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-4B5563)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Code for the paper:** *"Used but Not Needed: Testing Whether Manipulation Policies Rely on Visual Material Cues in Domain Randomisation"* — submitted to *Robotics and Autonomous Systems* (RAS, Elsevier).

---

## Overview

Domain randomisation (DR) typically samples visual appearance and dynamics **independently**. Real objects don't behave this way: an object's appearance and its dynamics share a common cause — its **material**. Steel looks heavy and ice looks slippery.

**PGDA (Physically-Grounded Diffusion Augmentation)** restores this link. Each diffusion-generated texture belongs to a material category that sets the object's mass, friction, and inertia ranges. A physics-prediction encoder recognises the category from the texture and passes the corresponding code to a reinforcement-learning policy for planar pushing with a UR3e arm.

We separate **reliance** from **need** with three tests:

| Test | Question | Answer |
|------|----------|--------|
| **Mismatch** — wrong material code, physics unchanged | Does the policy *rely* on the cue? | **Yes** — success drops 11.0 pp (sim), 96.7% → 20.0% (real) |
| **No-Cue control** — same coupled dynamics, no encoder | Does the policy *need* the cue? | **No** — No-Cue agent does at least as well (84.0% vs 79.0%) |
| **Oracle** — ground-truth physics in observation | Could a *perfect* cue help? | **No** — equivalent within ±5 pp |
| **Specialist** — trained on single material | Upper bound on any cue's value | **3.4 pp** max gain on hardest material |

On a second task where materials need **conflicting pushes**, a perfect cue adds only 5.1 pp (reactive) or 11.8 pp (committed), because the **first contact reveals the material within one control step**.

On a real UR3e, label-conditioned control transfers (96.7% with correct label), but the encoder trained on synthetic textures classifies real materials with only 46.7% accuracy — **perception, not control, limits autonomous deployment**.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Mohammedgmail/PGDA.git
cd PGDA

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Dependencies

- Python 3.10–3.13
- PyBullet 3.2+ (physics simulation)
- Gymnasium 0.29+ (RL environment interface)
- Stable-Baselines3 2.0+ (SAC + HER)
- PyTorch 2.0+ (encoder training)
- Weights & Biases (experiment tracking)

---

## Quick Start

### 1. Train the texture encoder

```bash
# Ensure texture bank exists at assets/texture_bank/
# (Generate with: python -m pgda.texture_bank)

python -m pgda.train_encoder
# Saves encoder to texture_encoder.pt
```

### 2. Train agents

```bash
# Baseline (Agent A): uncorrelated DR, no encoder
MODE=baseline python -m pgda.train_agents

# Experimental (Agent B): PGDA with encoder
MODE=experimental python -m pgda.train_agents

# Ablation: Oracle (Agent C), No-DR (Agent D), Coupled-No-Cue (Agent E)
MODE=oracle python -m pgda.train_agents
MODE=no_dr python -m pgda.train_agents
MODE=coupled_no_cue python -m pgda.train_agents
```

### 3. Evaluate

```bash
# Standard evaluation (RQ1)
EVAL_MODE=standard python -m pgda.evaluate_agents

# Mismatch test (RQ2)
EVAL_MODE=mismatch python -m pgda.evaluate_agents

# Ablation study
EVAL_MODE=ablation python -m pgda.evaluate_agents
```

---

## Project Structure

```
PGDA/
├── src/pgda/                 # Main package
│   ├── __init__.py
│   ├── env.py                # UR3e planar pushing environment (PyBullet + Gymnasium)
│   ├── env_task2.py          # Task 2: sliding with conflicting pushes
│   ├── train_encoder.py      # ResNet-18 physics-prediction encoder
│   ├── train_agents.py       # SAC+HER training pipeline (5 modes)
│   ├── evaluate_agents.py    # Evaluation: standard, mismatch, ablation
│   ├── texture_bank.py       # Diffusion texture generation pipeline
│   ├── optuna_tune.py        # Hyperparameter optimisation
│   └── analysis/             # Behavioural analysis scripts
├── tests/                    # Unit tests (43 tests)
├── examples/                 # Example configurations
│   ├── pushing_config.yaml   # Main task configuration
│   └── sliding_task_config.yaml
├── assets/                   # Texture bank & physics mapping
│   ├── semantic_physics_mapping.json
│   └── texture_bank/         # Generated textures (gitignored, download separately)
├── docs/                     # Figures & plotting scripts
├── .github/workflows/ci.yml  # CI: Python 3.10–3.13 + macOS
├── pyproject.toml            # Package metadata & dependencies
├── .publish                  # publish.sh metadata (description, topics)
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## Key Results (from paper)

### Task 1: Planar Pushing

| Agent | Description | Success Rate (5 seeds) |
|-------|-------------|------------------------|
| Baseline (A) | Uncorrelated DR, no cue | 79.0% ± 3.2 |
| **PGDA (B)** | **Coupled DR + encoder cue** | **79.0% ± 3.2** |
| No-Cue (E) | Coupled DR, no encoder | **84.0% ± 2.8** |
| Oracle (C) | Coupled DR + ground-truth physics | 81.0% ± 3.1 |
| No-DR (D) | Fixed physics, grey object | 92.0% ± 2.4 |

- **Mismatch drop (B):** −11.0 pp average across seeds → policy *relies* on cue
- **No-Cue vs PGDA:** No-Cue ≥ PGDA → policy does *not need* cue
- **Specialist bound:** Best single-material agent (Wet Ice) adds ≤ 3.4 pp

### Task 2: Sliding with Conflicting Pushes

| Policy Type | Perfect Cue Gain |
|-------------|------------------|
| Reactive (react to contact) | +5.1 pp |
| Committed (push before contact) | +11.8 pp |

First contact identifies near-frictionless material in 95% of episodes.

### Sim-to-Real (UR3e)

| Condition | Heavy Rust | Coarse Sandpaper |
|-----------|------------|------------------|
| PGDA + correct label | 96.7% | 96.7% |
| PGDA + wrong label | 20.0% | 20.0% |
| PGDA + no label | ≤3.3% | ≤3.3% |
| Encoder on real materials | **46.7% accuracy** | — |

---

## Reproducing Paper Figures

All figures in the paper are generated from code in `docs/`:

```bash
cd docs
python make_figures.py
```

This regenerates:
- Training curves (success rate vs timesteps)
- Mismatch test results
- Ablation study comparisons
- Sim-to-real transfer results
- Task 2 sliding results

---

## Citation

If you use this code, please cite:

```bibtex
@article{alfandi2026pgda,
  title={Used but Not Needed: Testing Whether Manipulation Policies Rely on Visual Material Cues in Domain Randomisation},
  author={Alfandi, Mohammed and Wang, Shidong and Shao, Zhuang and Adhikari, Kabita},
  journal={Robotics and Autonomous Systems},
  year={2026},
  publisher={Elsevier}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
