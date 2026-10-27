# Changelog

## 0.1.1

Fixes to the initial release. Simulation behaviour, defaults and the random-number
streams used for the reported results are unchanged.

### Fixed
- **Encoder silently disabled.** `env.py` imported the encoder as `train_encoder`
  inside a `try` block; in the package this import failed, the error was caught,
  and the environment fell back to a zero physics code. Experimental (PGDA) runs
  therefore trained without a cue. It now imports `pgda.train_encoder`.
- **Broken imports.** Nine scripts imported `pushing_env` and three functions in
  `scripted_voi_task2.py` imported `pushing_env_task2`, the module names before
  packaging. They now import `pgda.env` and `pgda.env_task2`.
- **File paths.** The environment looked for its data next to `env.py`
  (`src/pgda/assets/`), not in the repository's `assets/` folder, and several
  scripts used absolute paths from the author's computer. All paths now come from
  `pgda/paths.py`, rooted at the repository (override with `PGDA_ROOT` and
  `PGDA_ENCODER_PATH`).
- **README commands.** `MODE=... python -m pgda.train_agents` had no effect.
  `train_agents`, `train_specialist` and `evaluate_agents` now read optional
  `PGDA_MODE`, `PGDA_SEED`, `PGDA_CATEGORY` and `PGDA_EVAL_MODE` variables; the
  defaults are the previous values.
- **README results.** The results table now matches the paper.
- **Dependencies.** `scipy` added to the `eval` extra; new `textures` extra for
  Stable Diffusion texture generation.

### Removed
- `docs/make_figures.py`, which drew placeholder data rather than the paper's
  results.

### CI
- Lint (ruff check and format) runs once in its own job; the code passes both.
- Tests run on Linux with Python 3.10 to 3.13, using CPU-only PyTorch and a pip
  cache. macOS is no longer tested: PyBullet has no macOS wheels and its source
  build fails with current Apple compilers (use conda-forge, see README).
- Simulation tests are skipped, with a reason, when the robot and object models
  are not present. New tests check that every import in the package resolves,
  and that the physics mapping and inertia scaling match the paper.
- The legacy `np.random` API is kept (ruff rule NPY002 disabled): evaluation seeds
  the global generator per episode, and changing the API would change every
  episode.
