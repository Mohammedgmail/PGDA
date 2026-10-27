"""Tests for PGDA environment."""

import pytest
import numpy as np


def test_env_import():
    """Test that the environment can be imported."""
    from pgda.env import UR3ePushingEnv
    assert UR3ePushingEnv is not None


def test_env_creation():
    """Test that the environment can be created."""
    from pgda.env import UR3ePushingEnv
    
    env = UR3ePushingEnv(render_mode=None, randomization_mode="baseline")
    assert env is not None
    
    # Check observation and action spaces
    assert env.observation_space is not None
    assert env.action_space is not None
    
    # Check reset
    obs, info = env.reset(seed=42)
    assert obs is not None
    
    # Check step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    
    env.close()


def test_randomization_modes():
    """Test all randomization modes can be created."""
    from pgda.env import UR3ePushingEnv
    
    modes = ["baseline", "experimental", "oracle", "no_dr", "coupled_no_cue", "mismatched", "no_texture"]
    
    for mode in modes:
        env = UR3ePushingEnv(render_mode=None, randomization_mode=mode)
        obs, info = env.reset(seed=42)
        assert obs is not None
        env.close()


def test_material_categories():
    """Test that material categories are defined."""
    from pgda.env import CATEGORY_IZZ_SCALE
    
    assert "Heavy Rust" in CATEGORY_IZZ_SCALE
    assert "Wet Ice" in CATEGORY_IZZ_SCALE
    assert "Coarse Sandpaper" in CATEGORY_IZZ_SCALE
    
    # Check values from paper
    assert CATEGORY_IZZ_SCALE["Heavy Rust"] == 0.15
    assert CATEGORY_IZZ_SCALE["Wet Ice"] == 4.0
    assert CATEGORY_IZZ_SCALE["Coarse Sandpaper"] == 1.0


def test_semantic_physics_mapping():
    """Test that semantic physics mapping loads correctly."""
    import json
    from pathlib import Path
    
    mapping_path = Path(__file__).parent.parent / "assets" / "semantic_physics_mapping.json"
    assert mapping_path.exists()
    
    with open(mapping_path) as f:
        mapping = json.load(f)
    
    assert "Heavy Rust" in mapping
    assert "Wet Ice" in mapping
    assert "Coarse Sandpaper" in mapping
    
    for cat, bounds in mapping.items():
        assert "mass" in bounds
        assert "lateral_friction" in bounds
        assert len(bounds["mass"]) == 2
        assert len(bounds["lateral_friction"]) == 2
        assert bounds["mass"][0] < bounds["mass"][1]
        assert bounds["lateral_friction"][0] < bounds["lateral_friction"][1]
