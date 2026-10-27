# train_encoder.py — Supervised Pre-training of Texture Physics Encoder
# PGDA

# Usage:
#   1. Ensure your texture bank exists at TEXTURE_BANK_DIR (1000+ PNG images
#      organised in subdirectories named by category, e.g.:
#        textures/Heavy_Rust/image_001.png
#        textures/Wet_Ice/image_002.png
#        textures/Coarse_Sandpaper/image_003.png
#   2. Ensure semantic_physics_mapping.json exists (same format as in env).
#   3. Run:  python train_encoder.py
#   4. Encoder saved to texture_encoder.pt
#   5. Set ENCODER_PATH = "texture_encoder.pt" in 8-train_agents.py
#
# Target: validation R² > 0.85 on both mass and friction before training.
# If R² < 0.70 after 50 epochs, increase N_EPOCHS or check image quality.

import glob
import json
import os

# ── Configuration ─────────────────────────────────────────────────────────────
# Set these three paths to match project layout before running.
#
# TEXTURE_BANK_DIR : folder containing subdirectories named by category, e.g.:
#     textures/Heavy_Rust/image_001.png
#     textures/Wet_Ice/image_002.png
#     textures/Coarse_Sandpaper/image_003.png
#
# SEMANTIC_MAP_FILE : your existing semantic_physics_mapping.json
#     (the same file pushing_env.py uses — just give the full path)
#
# ENCODER_OUTPUT_PATH : where to save the trained encoder weights
#     (then set ENCODER_PATH in 8-train_agents.py to this same path)
import os as _os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

from pgda import paths

# ── EDIT THESE THREE LINES ────────────────────────────────────────────────────
TEXTURE_BANK_DIR = str(paths.TEXTURE_BANK_DIR)
SEMANTIC_MAP_FILE = str(paths.SEMANTIC_MAP_FILE)
ENCODER_OUTPUT_PATH = str(paths.ENCODER_PATH)
# ─────────────────────────────────────────────────────────────────────────────

# Auto-resolve relative paths from the script's own directory so the script
# works correctly when run from any working directory.
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
if not _os.path.isabs(TEXTURE_BANK_DIR):
    TEXTURE_BANK_DIR = _os.path.join(_script_dir, TEXTURE_BANK_DIR)
if not _os.path.isabs(SEMANTIC_MAP_FILE):
    SEMANTIC_MAP_FILE = _os.path.join(_script_dir, SEMANTIC_MAP_FILE)
if not _os.path.isabs(ENCODER_OUTPUT_PATH):
    ENCODER_OUTPUT_PATH = _os.path.join(_script_dir, ENCODER_OUTPUT_PATH)

IMAGE_SIZE = 64  # resize all textures to 64×64
BATCH_SIZE = 64
N_EPOCHS = 60
LEARNING_RATE = 3e-4
VAL_FRACTION = 0.15  # 15% of images held out for validation
RANDOM_SEED = 42

# Physics normalisation bounds (computed from all categories in mapping)
# Will be computed automatically from semantic_physics_mapping.json
MASS_MIN = MASS_MAX = FRICTION_MIN = FRICTION_MAX = None


# ── Dataset ───────────────────────────────────────────────────────────────────


class TexturePhysicsDataset(Dataset):
    """
    Loads texture images and labels them with normalised (mass, friction)
    midpoint values derived from their semantic category bounds.

    Labels are the midpoint of the category's range, normalised to [0,1]
    across all categories. This gives a deterministic, consistent supervision
    signal for each image.
    """

    def __init__(self, texture_dir, semantic_map, transform=None):
        self.samples = []  # list of (image_path, [norm_mass, norm_friction])
        self.transform = transform

        # Compute global bounds for normalisation
        all_masses = []
        all_frictions = []
        for bounds in semantic_map.values():
            all_masses.extend(bounds["mass"])
            all_frictions.extend(bounds["lateral_friction"])
        global_mass_min = min(all_masses)
        global_mass_max = max(all_masses)
        global_fric_min = min(all_frictions)
        global_fric_max = max(all_frictions)

        print("  Physics normalisation bounds:")
        print(f"    mass     : [{global_mass_min:.2f}, {global_mass_max:.2f}] kg")
        print(f"    friction : [{global_fric_min:.2f}, {global_fric_max:.2f}]")

        # Category index mapping (consistent order)
        self.categories = sorted(semantic_map.keys())
        self.physics_map = {}  # cat_idx → [norm_mass, norm_friction]

        # Build sample list
        n_found = 0
        for cat_idx, (category, bounds) in enumerate(sorted(semantic_map.items())):
            # Normalised midpoint label
            mass_mid = (bounds["mass"][0] + bounds["mass"][1]) / 2
            fric_mid = (bounds["lateral_friction"][0] + bounds["lateral_friction"][1]) / 2
            norm_mass = (mass_mid - global_mass_min) / (global_mass_max - global_mass_min + 1e-8)
            norm_fric = (fric_mid - global_fric_min) / (global_fric_max - global_fric_min + 1e-8)
            # Classification label: integer category index
            # We store the float physics values for inference-time mapping too
            label = cat_idx
            self.physics_map[cat_idx] = [float(norm_mass), float(norm_fric)]

            # Match directory name: category "Heavy Rust" → dir "Heavy_Rust"
            dir_name = category.replace(" ", "_")
            cat_dir = os.path.join(texture_dir, dir_name)

            if not os.path.isdir(cat_dir):
                print(
                    f"  WARNING: directory not found for category '{category}' → tried '{cat_dir}'"
                )
                continue

            images = (
                glob.glob(os.path.join(cat_dir, "*.png"))
                + glob.glob(os.path.join(cat_dir, "*.jpg"))
                + glob.glob(os.path.join(cat_dir, "*.jpeg"))
            )

            for img_path in images:
                self.samples.append((img_path, label))
            n_found += len(images)
            print(
                f"  Category '{category}': {len(images)} images, "
                f"label=({norm_mass:.3f}, {norm_fric:.3f})"
            )

        print(f"\n  Total samples: {n_found}")
        if n_found == 0:
            raise RuntimeError(
                f"No images found in '{texture_dir}'. "
                "Check TEXTURE_BANK_DIR and subdirectory naming."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)  # class index


# ── Model ─────────────────────────────────────────────────────────────────────


class TextureEncoder(nn.Module):
    """
    Lightweight CNN: ResNet-18 backbone re-headed as a 3-class texture
    classifier (Heavy Rust / Wet Ice / Coarse Sandpaper).

    Why classification not regression:
        All images in a category share identical physics labels, making MSE
        regression degenerate (R² ≈ 0.17). Classification is the correct
        formulation — the model learns visual features that distinguish the
        3 material appearances. At inference time the predicted class index
        is mapped back to (norm_mass, norm_friction) from the physics map.

    N_CLASSES is set at construction time from the semantic mapping.
    """

    def __init__(self, n_classes=3):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),  # logits — no softmax here (CrossEntropy)
        )
        self.net = backbone
        self.n_classes = n_classes

    def forward(self, x):
        return self.net(x)  # returns raw logits (N, n_classes)

    def predict_physics(self, x, physics_map):
        """
        Runs inference and maps predicted class → (norm_mass, norm_friction).
        Returns a numpy array of shape (2,) clipped to [0,1].
        Used by pushing_env.py at every episode reset.
        """
        with torch.no_grad():
            logits = self.forward(x)
            class_idx = int(logits.argmax(dim=-1).item())
        return np.array(physics_map.get(class_idx, [0.0, 0.0]), dtype=np.float32)


# ── Training ──────────────────────────────────────────────────────────────────


def compute_accuracy(logits, targets):
    """Top-1 classification accuracy."""
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float().sum()
    return float(correct / len(targets))


def train():
    print("=" * 60)
    print("  Texture Physics Encoder — Supervised Pre-training")
    print("=" * 60)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load semantic mapping
    if not _os.path.exists(SEMANTIC_MAP_FILE):
        raise FileNotFoundError(
            f"\n  semantic_physics_mapping.json not found at:\n"
            f"    {SEMANTIC_MAP_FILE}\n\n"
            f"  Fix: set SEMANTIC_MAP_FILE at the top of train_encoder.py\n"
            f"  to the full path of your existing mapping file, e.g.:\n"
            f"    SEMANTIC_MAP_FILE = '/Users/yourname/.../semantic_physics_mapping.json'"
        )
    with open(SEMANTIC_MAP_FILE) as f:
        semantic_map = json.load(f)
    print(f"\n  Loaded {len(semantic_map)} categories from {SEMANTIC_MAP_FILE}")
    if not _os.path.exists(TEXTURE_BANK_DIR):
        raise FileNotFoundError(
            f"\n  Texture bank not found at: {TEXTURE_BANK_DIR}\n"
            f"  Fix: set TEXTURE_BANK_DIR at the top of train_encoder.py\n"
            f"  to the folder containing your category subdirectories."
        )
    print(f"  Texture bank: {TEXTURE_BANK_DIR}")

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    # Dataset
    print(f"\n  Loading texture bank from '{TEXTURE_BANK_DIR}'...")
    full_dataset = TexturePhysicsDataset(TEXTURE_BANK_DIR, semantic_map, transform=train_transform)

    n_val = max(1, int(len(full_dataset) * VAL_FRACTION))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(RANDOM_SEED)
    )
    val_ds.dataset.transform = val_transform  # no augmentation for val

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n  Train: {n_train} samples | Val: {n_val} samples")
    print(f"  Batch size: {BATCH_SIZE} | Epochs: {N_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")

    # Model, loss, optimiser
    n_classes = len(semantic_map)
    model = TextureEncoder(n_classes=n_classes)
    criterion = nn.CrossEntropyLoss()  # correct loss for classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"  Device: {device}\n")

    best_val_r2 = -999.0
    best_epoch = 0

    print(f"  {'Epoch':>5s}  {'Train CE':>10s}  {'Val CE':>9s}  {'Val Acc':>7s}  {'Best?'}")
    print("  " + "-" * 45)

    for epoch in range(1, N_EPOCHS + 1):
        # ── Training pass ─────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= n_train
        scheduler.step()

        # ── Validation pass ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                val_loss += criterion(preds, labels).item() * images.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        val_loss /= n_val
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_acc = compute_accuracy(all_preds, all_labels)

        is_best = val_acc > best_val_r2  # reusing variable name for best score
        if is_best:
            best_val_r2 = val_acc
            best_epoch = epoch
            # Also save physics_map alongside weights so env can load it
            torch.save(
                {
                    "state_dict": model.cpu().state_dict(),
                    "physics_map": full_dataset.physics_map,
                    "categories": full_dataset.categories,
                    "n_classes": n_classes,
                },
                ENCODER_OUTPUT_PATH,
            )
            model = model.to(device)

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(
                f"  {epoch:5d}  {train_loss:10.6f}  "
                f"{val_loss:9.6f}  {val_acc:7.4f}  "
                f"{'✓ BEST' if is_best else ''}"
            )

    print(f"\n{'=' * 60}")
    print("  Training complete.")
    print(f"  Best validation Acc : {best_val_r2 * 100:.1f}%  (epoch {best_epoch})")
    print(f"  Encoder saved to    : {ENCODER_OUTPUT_PATH}")

    if best_val_r2 >= 0.90:
        print("  ✓ Accuracy ≥ 90% — encoder is reliable. Ready for RL training.")
        print(f"  → ENCODER_PATH = '{ENCODER_OUTPUT_PATH}' is already set in 8-train_agents.py")
    elif best_val_r2 >= 0.75:
        print("  ⚠  Accuracy in [75%, 90%] — usable but not ideal.")
        print("     Consider increasing N_EPOCHS to 100 or adding more texture images.")
    else:
        print("  ✗ Accuracy < 75% — encoder needs improvement before RL training.")
        print("     Check: (1) subdirectory naming matches semantic_physics_mapping.json")
        print("            (2) increase N_EPOCHS to 100")
        print("            (3) ensure ≥ 200 images per category")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
