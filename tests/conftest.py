"""Pytest configuration for PGDA tests."""

import sys
from pathlib import Path

# Add src to path for testing
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set up headless rendering for PyBullet
import os
os.environ["PYBULLET_EGL"] = "1"
