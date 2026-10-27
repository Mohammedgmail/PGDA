"""Pytest configuration for PGDA tests."""

import os
import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set up headless rendering for PyBullet
os.environ["PYBULLET_EGL"] = "1"
