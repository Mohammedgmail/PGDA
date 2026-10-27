import os
import time

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image  # Requires: pip install Pillow


# --- PART 1: GENERATE A DUMMY TEXTURE ---
# In your real project, Stable Diffusion will do this.
# For now, we create a simple "noisy" texture to simulate rust/rough material.
def create_dummy_texture(filename="procedural_rust.png"):
    width, height = 256, 256
    # Generate random noise (simulating a rough surface)
    noise_data = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)

    # Add a reddish tint (simulating rust)
    noise_data[:, :, 0] = np.clip(noise_data[:, :, 0] + 50, 0, 255)  # Red channel

    img = Image.fromarray(noise_data, "RGB")
    img.save(filename)
    return os.path.abspath(filename)


# --- PART 2: SETUP SIMULATION ---
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Clean view

# Load Ground
planeId = p.loadURDF("plane.urdf")

# Load UR3e (Your existing setup)
robot_folder = "ur_description"
urdf_file = os.path.join(robot_folder, "urdf/ur3e.urdf")
current_dir = os.path.dirname(os.path.abspath(__file__))
p.setAdditionalSearchPath(current_dir)

try:
    robotId = p.loadURDF(urdf_file, [0, 0, 0], useFixedBase=True)
except Exception:
    print("UR3e not found, make sure 'ur_description' folder is here.")

# --- PART 3: THE GENERATIVE AUGMENTATION LOGIC ---
# This mimics your project's "Experimental Mode"

# 1. Create the Visual/Physical Object (A simple cube)
colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])
visualBoxId = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05])

# Spawn it on the table in front of the robot
boxId = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=colBoxId,
    baseVisualShapeIndex=visualBoxId,
    basePosition=[0.4, 0, 0.05],
)

# 2. Load and Apply the Texture (The "Visual" part)
texture_path = create_dummy_texture()
textureId = p.loadTexture(texture_path)
p.changeVisualShape(boxId, -1, textureUniqueId=textureId)
print(f"Applied Texture: {texture_path}")

# 3. Apply the Physics (The "Grounded" part) [cite: 27]
# Since it looks like 'Rust', we give it High Friction and High Mass
p.changeDynamics(
    boxId,
    -1,
    mass=2.5,  # Heavier than normal
    lateralFriction=1.2,  # High friction (rough surface)
    rollingFriction=0.01,
)

print("Applied Physics: Mass=2.5kg, Friction=1.2 (Simulating Heavy Rust)")

# --- PART 4: SIMULATION LOOP ---
# Focus camera on the new cube
p.resetDebugVisualizerCamera(
    cameraDistance=0.8, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0.4, 0, 0.05]
)

while True:
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
