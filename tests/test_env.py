"""Tests for the PGDA environment and package layout.

The simulation tests need the robot and object models (ur_description/ and
assets/train_objects/), which are not distributed with the repository; see the
README. They are skipped, with a reason, when those files are missing, so CI
checks everything that can run without them.
"""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from pgda import paths

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = REPO_ROOT / "src" / "pgda"

needs_sim_assets = pytest.mark.skipif(
    not paths.simulation_assets_available(),
    reason=f"robot/object models not found under {paths.ROOT} (see README: Simulation assets)",
)

# Values reported in the paper (Table B.1 and Section 3.2).
PAPER_BOUNDS = {
    "Heavy Rust": {"mass": [2.0, 3.0], "lateral_friction": [0.80, 1.00]},
    "Wet Ice": {"mass": [0.5, 1.5], "lateral_friction": [0.01, 0.05]},
    "Coarse Sandpaper": {"mass": [1.0, 2.0], "lateral_friction": [1.00, 1.50]},
}
PAPER_IZZ_SCALE = {"Heavy Rust": 0.15, "Wet Ice": 4.0, "Coarse Sandpaper": 1.0}

# Third-party packages that only some scripts need and that are not installed
# by `pip install -e ".[dev]"` (e.g. texture generation).
OPTIONAL_PACKAGES = {"diffusers", "transformers", "accelerate", "seaborn", "cv2", "scipy"}


def test_env_import():
    """The environment module imports."""
    from pgda.env import UR3ePushingEnv

    assert UR3ePushingEnv is not None


def test_all_imports_resolve():
    """Every import in the package points at a module that exists.

    Guards against scripts importing modules by their names before packaging
    (e.g. `pushing_env`), which fail at run time, or, inside a try block,
    silently disable the texture encoder.
    """
    unresolved = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split(".")[0] in OPTIONAL_PACKAGES:
                    continue
                if importlib.util.find_spec(name) is None:
                    unresolved.append(f"{path.name}:{node.lineno}: {name}")
    assert not unresolved, "unresolvable imports:\n" + "\n".join(unresolved)


def test_encoder_class_importable_from_env_path():
    """env.py loads the encoder class from pgda.train_encoder."""
    from pgda.train_encoder import TextureEncoder

    assert TextureEncoder is not None


def test_paths_point_into_repository():
    """By default, data files are resolved from the repository root."""
    assert paths.SEMANTIC_MAP_FILE.is_file()
    assert paths.ASSETS_DIR == paths.ROOT / "assets"


def test_material_categories():
    """Category inertia scaling matches the paper."""
    from pgda.env import CATEGORY_IZZ_SCALE

    assert CATEGORY_IZZ_SCALE == PAPER_IZZ_SCALE


def test_semantic_physics_mapping():
    """The physics mapping file matches the bounds reported in the paper."""
    with open(paths.SEMANTIC_MAP_FILE) as f:
        mapping = json.load(f)

    assert mapping == PAPER_BOUNDS
    for bounds in mapping.values():
        assert bounds["mass"][0] < bounds["mass"][1]
        assert bounds["lateral_friction"][0] < bounds["lateral_friction"][1]


@needs_sim_assets
def test_env_creation():
    """The environment can be created, reset and stepped."""
    from pgda.env import UR3ePushingEnv

    env = UR3ePushingEnv(render_mode=None, randomization_mode="baseline")
    assert env.observation_space is not None
    assert env.action_space is not None

    obs, info = env.reset(seed=42)
    assert obs is not None

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    env.close()


@needs_sim_assets
@pytest.mark.parametrize(
    "mode",
    [
        "baseline",
        "experimental",
        "oracle",
        "no_dr",
        "coupled_no_cue",
        "mismatched",
        "no_texture",
    ],
)
def test_randomization_modes(mode):
    """Every randomisation mode can be created and reset."""
    from pgda.env import UR3ePushingEnv

    env = UR3ePushingEnv(render_mode=None, randomization_mode=mode)
    obs, info = env.reset(seed=42)
    assert obs is not None
    env.close()
