"""Where PGDA looks for its data files.

All scripts resolve their inputs and outputs from one root folder. By default
this is the repository root (two levels above this file), which is correct for
an editable install (``pip install -e .``). Set ``PGDA_ROOT`` to use another
folder, and ``PGDA_ENCODER_PATH`` to load or save the encoder elsewhere.

Expected layout under the root::

    assets/semantic_physics_mapping.json   (included in the repository)
    assets/texture_bank/<Category>/*.png   (generate or download, see README)
    assets/train_objects/<name>/<name>.urdf  (object models, see README)
    ur_description/urdf/ur3e_rg2.urdf      (robot model, see README)
    texture_encoder.pt                     (written by python -m pgda.train_encoder)
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("PGDA_ROOT", Path(__file__).resolve().parents[2]))
ASSETS_DIR = ROOT / "assets"
SEMANTIC_MAP_FILE = ASSETS_DIR / "semantic_physics_mapping.json"
TEXTURE_BANK_DIR = ASSETS_DIR / "texture_bank"
TRAIN_OBJECTS_DIR = ASSETS_DIR / "train_objects"
ROBOT_URDF = ROOT / "ur_description" / "urdf" / "ur3e_rg2.urdf"
ENCODER_PATH = Path(os.environ.get("PGDA_ENCODER_PATH", ROOT / "texture_encoder.pt"))


def simulation_assets_available() -> bool:
    """True if the robot model and at least one object model are present."""
    return ROBOT_URDF.is_file() and any(TRAIN_OBJECTS_DIR.glob("*/*.urdf"))
