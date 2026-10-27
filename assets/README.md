# Assets

This directory contains the physics-to-material mapping and placeholder for the texture bank.

## semantic_physics_mapping.json

Defines the physical parameter ranges for each material category:

```json
{
  "Heavy Rust": {
    "mass": [2.0, 3.0],
    "lateral_friction": [0.8, 1.0]
  },
  "Wet Ice": {
    "mass": [0.5, 1.5],
    "lateral_friction": [0.01, 0.05]
  },
  "Coarse Sandpaper": {
    "mass": [1.0, 2.0],
    "lateral_friction": [1.0, 1.5]
  }
}
```

## texture_bank/

The texture bank contains 1000+ diffusion-generated images per category, organised as:

```
texture_bank/
├── Heavy_Rust/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── Wet_Ice/
│   ├── image_001.png
│   └── ...
└── Coarse_Sandpaper/
    ├── image_001.png
    └── ...
```

**Note:** The actual texture images are not included in this repository due to size (~3 GB). To generate or download them:

### Option 1: Generate locally
```bash
python -m pgda.texture_bank
```
This uses Stable Diffusion to generate textures. Requires a GPU with 8GB+ VRAM.

### Option 2: Download pre-generated bank
Download from: [Link to be added - e.g., Zenodo/Google Drive]

Place the extracted folders in `assets/texture_bank/`.

### Option 3: Use your own textures
Organise your textures in the same structure and update `semantic_physics_mapping.json` if needed.
