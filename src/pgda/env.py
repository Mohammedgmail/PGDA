import glob
import json
import os
import random
import time

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from pgda.paths import ROOT as PGDA_ROOT

# ── Tuning knobs ──────────────────────────────────────────────────────────────
SIM_STEPS_PER_ACTION = 20
SETTLE_STEPS = 300

# ── Workspace ─────────────────────────────────────────────────────────────────
FIXED_Z = 0.19  # Calibrated from visual_test.py measurements:
#   tip_offset_below_tool0 = 0.165 m  (measured)
#   object_centre_z = 0.025 m → FIXED_Z = 0.025 + 0.165 = 0.19
# Ground clipping at 0.19m is SOLVED by disabling robot↔plane
# collision in _setup_robot (arm is IK-controlled, needs no
# physics floor contact). Object↔plane contact unaffected.
EE_X_BOUNDS = (0.22, 0.46)
EE_Y_BOUNDS = (-0.16, 0.16)

# ── RG2 gripper closed positions ──────────────────────────────────────────────
GRIPPER_JOINT_CLOSED = {
    "finger_joint": +0.785398,
    "left_inner_knuckle_joint": -0.558505,
    "left_inner_finger_joint": +0.785398,
    "right_outer_knuckle_joint": -0.558505,
    "right_inner_knuckle_joint": -0.558505,
    "right_inner_finger_joint": +0.785398,
}

# ── [S1] Asymmetric CoM: category → Izz multiplier ──────────────────────────
# Heavy Rust: Izz × 0.15 → low rotational inertia → spins easily if pushed
#   off-centre. Correct strike point (near CoM) is visually identifiable.
# Wet Ice:    Izz × 4.0  → high rotational inertia → translates cleanly.
# Sandpaper:  Izz × 1.0  → neutral, intermediate behaviour.
CATEGORY_IZZ_SCALE = {
    "Heavy Rust": 0.15,
    "Wet Ice": 4.00,
    "Coarse Sandpaper": 1.00,
}

# ── [P3] Reward weights (overridable via reward_weights dict) ─────────────────
DEFAULT_REWARD_WEIGHTS = {
    "w_push": 2.0,  # potential-based obj→goal shaping
    "w_align": 0.3,  # anti-goal positioning bonus
    "w_contact": 0.05,  # per-step contact bonus
    "w_spin": 0.5,  # angular velocity penalty
    "w_action": 0.1,  # action magnitude penalty (energy)
    "r_success": 5.0,  # terminal success reward (rescaled from 100)
    "success_threshold": 0.05,  # metres; exposed for curriculum annealing
}


class UR3ePushingEnv(gym.Env):
    """
    Custom PyBullet Gymnasium environment for planar pushing.

    Observation space:
        end_effector_pos : (3,)  float32  — EE Cartesian position
        state_vector     : (10,) float32  — privileged low-dim state [P2]
        wrist_rgbd       : (64,64,4) uint8
        global_rgbd      : (64,64,4) uint8

    Action space:
        (dx, dy) : float32 in [-0.05, 0.05] m — planar delta EE position

    Randomization modes:
        'baseline'     — mass and friction sampled uniformly (uncorrelated)
        'experimental' — mass and friction sampled from semantic-physics map;
                         dims 9-10 filled with encoder prediction.
        'mismatched'   — texture from one category, physics from a DIFFERENT
                         category. Breaks the visual-physics correlation while
                         keeping everything else identical. Used for RQ2 mismatch
                         experiment on trained Agent B models (no retraining).
        'no_texture'   — uniform grey object, physics from baseline uncorrelated
                         distribution. Control condition for RQ2: tests whether
                         removing texture entirely affects a trained policy.
        'oracle'       — ABLATION: identical physics to experimental (asymmetric
                         Izz, category-correlated mass/friction), but dims 9-10
                         receive the GROUND-TRUTH normalised physics values
                         instead of encoder predictions. Upper bound on what a
                         perfect encoder could achieve. Isolates encoder quality
                         from task difficulty.
        'no_dr'        — ABLATION: fixed physics (midpoint of baseline bounds),
                         uniform grey object, no randomisation of any kind.
                         Lower bound / sanity check. Tests whether any gap
                         remains when the task is fully deterministic.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    HOME_JOINT_ANGLES = [0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0]

    def __init__(
        self,
        render_mode=None,
        randomization_mode="experimental",
        reward_weights=None,
        encoder_path=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.randomization_mode = randomization_mode

        # ── [S2] Pre-trained texture encoder ─────────────────────────────────
        # Experimental mode: encoder predicts (norm_mass, norm_friction) from
        # the texture image applied in reset(). Result appended to observation.
        # Baseline mode: encoder_path=None → _physics_pred = [0,0] always.
        # Both agents have identical 11-dim obs; only the content differs.
        self.encoder = None
        self._physics_pred = np.zeros(2, dtype=np.float32)
        if encoder_path is not None:
            try:
                import torch

                from pgda.train_encoder import TextureEncoder

                enc = TextureEncoder()
                checkpoint = torch.load(encoder_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    enc.load_state_dict(checkpoint["state_dict"])
                    self._encoder_physics_map = checkpoint.get("physics_map", {})
                else:
                    enc.load_state_dict(checkpoint)  # fallback: raw state_dict
                    self._encoder_physics_map = {}
                print(f"[Env] Encoder physics map: {self._encoder_physics_map}")
                print(f"[Env] n_classes: {enc.n_classes}")
                enc.eval()
                self.encoder = enc
                self._torch = torch
                print(f"[Env] Texture encoder loaded: {encoder_path}")
            except Exception as e:
                print(f"[Env] WARNING: encoder load failed: {e}. Physics pred = zeros.")

        # ── [HER] Sparse reward — threshold only ─────────────────────────────
        # SAC+HER uses sparse reward (0/-1). Only the success threshold matters.
        rw = {**DEFAULT_REWARD_WEIGHTS, **(reward_weights or {})}
        self.success_threshold = float(rw.get("success_threshold", 0.05))

        # ── PyBullet client ───────────────────────────────────────────────────
        if self.render_mode == "human":
            self._client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._client)
        else:
            self._client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)

        # ── Action & observation spaces ───────────────────────────────────────
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(2,), dtype=np.float32)

        # ── [HER] GoalEnv observation space ──────────────────────────────────
        # HER requires keys: observation, achieved_goal, desired_goal.
        # observation (9-dim, goal-free):
        #   [0-2] EE xyz, [3-4] EE→obj xy, [5-6] obj vel xy,
        #   [7] obj spin z, [8] obj yaw
        # achieved_goal: object (x,y) — relabelled by HER after each episode
        # desired_goal : target (x,y) — relabelled by HER after each episode
        # [S2] dims 9-10: encoder physics prediction (norm mass, norm friction)
        # Baseline mode: always [0,0]. Experimental: predicted from texture.
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(low=-10.0, high=10.0, shape=(11,), dtype=np.float32),
                "achieved_goal": spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32),
                "desired_goal": spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32),
            }
        )

        # ── File paths ────────────────────────────────────────────────────────
        self.current_dir = str(PGDA_ROOT)  # see pgda/paths.py; override with PGDA_ROOT
        self.urdf_file = os.path.join(self.current_dir, "ur_description", "urdf", "ur3e_rg2.urdf")
        self.train_objects_dir = os.path.join(self.current_dir, "assets", "train_objects")
        self.texture_dir = os.path.join(self.current_dir, "assets", "texture_bank")
        self.mapping_path = os.path.join(
            self.current_dir, "assets", "semantic_physics_mapping.json"
        )

        # ── Object dataset (with geometry filter) ────────────────────────────
        all_urdfs = glob.glob(os.path.join(self.train_objects_dir, "*", "*.urdf"))
        forbidden_keywords = [
            "cola",
            "water",
            "tea",
            "can",
            "frother",
            "wagon",
            "cart",
            "turtle",
            "shoe",
            "mens",
            "dish",
            "cup",
            "lunch_box",
        ]
        self.object_urdfs = [
            u
            for u in all_urdfs
            if not any(
                kw in os.path.basename(os.path.dirname(u)).lower() for kw in forbidden_keywords
            )
        ]
        if not self.object_urdfs:
            print("Warning: geometry filter left no objects — reverting to all.")
            self.object_urdfs = all_urdfs

        # ── Texture & semantic mapping ────────────────────────────────────────
        self.texture_files = glob.glob(os.path.join(self.texture_dir, "*", "*.png"))
        with open(self.mapping_path) as f:
            self.semantic_mapping = json.load(f)

        # Baseline (uncorrelated) sampling bounds
        self.baseline_mass_bounds = [0.1, 5.0]
        self.baseline_friction_bounds = [0.1, 1.2]

        # ── Runtime state ─────────────────────────────────────────────────────
        self.goal_pos = None
        self.goal_uid = -1
        self.prev_dist_obj_goal = 0.0
        self.object_uid = -1
        self.current_object_name = ""
        # [P1] Store last sampled physics for SysID logging
        self._last_mass = 1.0
        self._last_friction = 0.5
        self._last_inertia = [1e-4, 1e-4, 1e-4]

        self.fixed_z = FIXED_Z
        self.fixed_orientation = None  # set from URDF at home pose
        self.ee_link_index = -1
        self.gripper_tip_index = -1
        self.current_ee_pos = [0.3, 0.0, self.fixed_z]

        self.cam_width = 64
        self.cam_height = 64
        self.cam_fov = 60
        self.cam_near = 0.02
        self.cam_far = 2.0

    # ── PyBullet helpers ──────────────────────────────────────────────────────

    def _pb(self):
        return {"physicsClientId": self._client}

    def _connected(self):
        try:
            return p.isConnected(self._client)
        except Exception:
            return False

    def _find_link(self, robot_id, names):
        for i in range(p.getNumJoints(robot_id, **self._pb())):
            if p.getJointInfo(robot_id, i, **self._pb())[12].decode() in names:
                return i
        return p.getNumJoints(robot_id, **self._pb()) - 1

    def _get_movable_joints(self, robot_id):
        return [
            i
            for i in range(p.getNumJoints(robot_id, **self._pb()))
            if p.getJointInfo(robot_id, i, **self._pb())[2] != p.JOINT_FIXED
        ]

    def _joint_id_by_name(self, robot_id, name):
        for i in range(p.getNumJoints(robot_id, **self._pb())):
            if p.getJointInfo(robot_id, i, **self._pb())[1].decode() == name:
                return i
        raise ValueError(f"Joint '{name}' not found")

    # ── Robot setup ───────────────────────────────────────────────────────────

    def _apply_home_pose(self):
        for idx, jidx in enumerate(self.joint_indices):
            angle = self.HOME_JOINT_ANGLES[idx] if idx < len(self.HOME_JOINT_ANGLES) else 0.0
            p.resetJointState(self.robotId, jidx, angle, **self._pb())
            p.setJointMotorControl2(
                self.robotId,
                jidx,
                p.POSITION_CONTROL,
                targetPosition=angle,
                force=150.0,
                **self._pb(),
            )

    def _lock_gripper_closed(self):
        for jname, jval in GRIPPER_JOINT_CLOSED.items():
            try:
                jid = self._joint_id_by_name(self.robotId, jname)
            except ValueError:
                continue
            p.resetJointState(self.robotId, jid, targetValue=jval, **self._pb())
            p.setJointMotorControl2(
                self.robotId,
                jid,
                p.POSITION_CONTROL,
                targetPosition=jval,
                force=5000,
                maxVelocity=0.0,
                **self._pb(),
            )

    def _setup_robot(self):
        p.setAdditionalSearchPath(self.current_dir, **self._pb())
        self.robotId = p.loadURDF(self.urdf_file, [0, 0, 0], useFixedBase=True, **self._pb())

        # Disable robot↔plane collision for ALL robot links.
        # The arm is IK-controlled and never needs physics contact with the
        # floor. Without this, the gripper finger mesh clips the ground at
        # FIXED_Z=0.19m, generating contact forces that overcome the 5000N
        # motor lock and drive the finger joints out of position.
        # Object↔plane contact is completely unaffected (separate body pair).
        num_links = p.getNumJoints(self.robotId, **self._pb())
        for link_idx in range(-1, num_links):
            p.setCollisionFilterPair(self.robotId, self.planeId, link_idx, -1, 0, **self._pb())

        self.ee_link_index = self._find_link(self.robotId, {"tool0", "ee_link", "flange"})
        self.gripper_tip_index = self._find_link(self.robotId, {"gripper_tip_link"})

        all_movable = self._get_movable_joints(self.robotId)
        self.joint_indices = all_movable[:6]  # arm joints only for IK
        self._all_movable = all_movable  # all 12 DOF for IK limits

        self._apply_home_pose()

        # Capture true tool orientation from URDF at home pose (no hardcoding)
        ee_state = p.getLinkState(self.robotId, self.ee_link_index, **self._pb())
        self.fixed_orientation = ee_state[1]

        # Disable gripper self-collisions
        gn = p.getNumJoints(self.robotId, **self._pb())
        for i in range(self.ee_link_index, gn):
            for j in range(i + 1, gn):
                p.setCollisionFilterPair(self.robotId, self.robotId, i, j, 0, **self._pb())

        self._lock_gripper_closed()
        self._move_to_target(self.current_ee_pos, settle_steps=SETTLE_STEPS)
        # Re-lock after arm settling: arm movement can impart inertial forces
        # on gripper joints during the settle steps. A second pass ensures the
        # gripper is fully locked before any episode interaction begins.
        self._lock_gripper_closed()

    # ── Motion control ────────────────────────────────────────────────────────

    def _move_to_target(self, target_pos, settle_steps: int = SIM_STEPS_PER_ACTION):
        target_pos[2] = self.fixed_z

        # Full 12-DOF IK limits (arm + gripper) — fixes null-space bias
        lo = [p.getJointInfo(self.robotId, j, **self._pb())[8] for j in self._all_movable]
        hi = [p.getJointInfo(self.robotId, j, **self._pb())[9] for j in self._all_movable]
        jr = [
            upper - lower if upper > lower else 2 * np.pi
            for upper, lower in zip(hi, lo, strict=False)
        ]

        gripper_closed_vals = list(GRIPPER_JOINT_CLOSED.values())
        rest_poses = [
            p.getJointState(self.robotId, j, **self._pb())[0] for j in self.joint_indices
        ] + gripper_closed_vals

        joint_poses = p.calculateInverseKinematics(
            self.robotId,
            self.ee_link_index,
            target_pos,
            self.fixed_orientation,
            lowerLimits=lo,
            upperLimits=hi,
            jointRanges=jr,
            restPoses=rest_poses,
            maxNumIterations=200,
            residualThreshold=1e-5,
            **self._pb(),
        )

        for i, jidx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                bodyIndex=self.robotId,
                jointIndex=jidx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=150.0,
                maxVelocity=1.0,
                **self._pb(),
            )

        for _ in range(settle_steps):
            if not self._connected():
                raise RuntimeError("PyBullet disconnected — use Ctrl+C to stop.")
            p.stepSimulation(**self._pb())
            if self.render_mode == "human":
                time.sleep(1.0 / 240.0)

        # z-lock correction pass
        actual_z = p.getLinkState(self.robotId, self.ee_link_index, **self._pb())[0][2]
        if abs(actual_z - self.fixed_z) > 0.01:
            home_rest = self.HOME_JOINT_ANGLES + gripper_closed_vals
            corr = p.calculateInverseKinematics(
                self.robotId,
                self.ee_link_index,
                [target_pos[0], target_pos[1], self.fixed_z],
                self.fixed_orientation,
                lowerLimits=lo,
                upperLimits=hi,
                jointRanges=jr,
                restPoses=home_rest,
                maxNumIterations=400,
                residualThreshold=1e-6,
                **self._pb(),
            )
            for i, jidx in enumerate(self.joint_indices):
                p.setJointMotorControl2(
                    bodyIndex=self.robotId,
                    jointIndex=jidx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=corr[i],
                    force=150.0,
                    maxVelocity=0.5,
                    **self._pb(),
                )
            for _ in range(settle_steps):
                if not self._connected():
                    raise RuntimeError("PyBullet disconnected.")
                p.stepSimulation(**self._pb())

    # ── Camera rendering ──────────────────────────────────────────────────────

    def _safe_camera_image(self, eye, target, up):
        eye = np.array(eye, dtype=float)
        target = np.array(target, dtype=float)
        up = np.array(up, dtype=float)
        look = target - eye
        if np.linalg.norm(look) < 1e-6:
            target = eye + np.array([0.0, 0.0, -0.1])
            look = target - eye
        look_hat = look / np.linalg.norm(look)
        up_hat = up / (np.linalg.norm(up) + 1e-12)
        if abs(np.dot(look_hat, up_hat)) > 0.99:
            up = (
                np.array([1.0, 0.0, 0.0])
                if abs(look_hat[0]) < abs(look_hat[1])
                else np.array([0.0, 1.0, 0.0])
            )
        proj = p.computeProjectionMatrixFOV(self.cam_fov, 1.0, self.cam_near, self.cam_far)
        vm = p.computeViewMatrix(eye.tolist(), target.tolist(), up.tolist())
        return p.getCameraImage(
            self.cam_width, self.cam_height, vm, proj, renderer=p.ER_TINY_RENDERER, **self._pb()
        )

    def _render_cameras(self):
        if not self._connected():
            blank = np.zeros((self.cam_height, self.cam_width, 4), np.uint8)
            return blank, blank

        g_result = self._safe_camera_image([0.3, 0.0, 1.2], [0.3, 0.0, 0.0], [0.0, 1.0, 0.0])

        ee_pos = np.array(p.getLinkState(self.robotId, self.ee_link_index, **self._pb())[0])
        w_result = self._safe_camera_image(
            ee_pos, ee_pos + np.array([0.0, 0.0, -0.5]), [0.0, 1.0, 0.0]
        )

        def to_rgbd(r):
            if r is None:
                return np.zeros((self.cam_height, self.cam_width, 4), np.uint8)
            _, _, rgb, dep, _ = r
            rgb_arr = np.reshape(rgb, (self.cam_height, self.cam_width, 4))[:, :, :3]
            dep_arr = (np.reshape(dep, (self.cam_height, self.cam_width, 1)) * 255).astype(np.uint8)
            return np.concatenate((rgb_arr, dep_arr), axis=2)

        return to_rgbd(w_result), to_rgbd(g_result)

    # ── [P2] State vector computation ─────────────────────────────────────────

    def _compute_state_and_reward_components(self):
        """
        Returns a dict of all quantities shared between _get_obs and step.
        Computing them once avoids duplicate PyBullet calls per timestep.

        State vector (10 floats):
          [0-1] obj→goal displacement (x, y)
          [2-3] ee→obj displacement (x, y)
          [4-5] object linear velocity (x, y)
          [6]   object angular velocity z (spin)
          [7]   object yaw on plane
          [8-9] (cos, sin) of approach angle:
                angle between (obj→ee) and (obj→goal) vectors

        Approach angle geometry:
          cos_approach = -1 when EE is on anti-goal side (optimal push position)
          cos_approach = +1 when EE is on goal side (pushing away from goal)
          r_align = -cos_approach rewards positioning on the correct side
        """
        ee_pos = np.array(self.current_ee_pos)
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.object_uid, **self._pb())
        obj_pos = np.array(obj_pos)
        goal = np.array(self.goal_pos)

        obj_vel, obj_ang_vel = p.getBaseVelocity(self.object_uid, **self._pb())

        # 2D displacement vectors
        vec_obj_goal = goal[:2] - obj_pos[:2]  # object→goal
        vec_obj_ee = ee_pos[:2] - obj_pos[:2]  # object→EE

        norm_og = np.linalg.norm(vec_obj_goal) + 1e-8
        norm_oe = np.linalg.norm(vec_obj_ee) + 1e-8

        cos_approach = float(np.dot(vec_obj_ee, vec_obj_goal) / (norm_oe * norm_og))
        sin_approach = float(np.cross(vec_obj_ee, vec_obj_goal) / (norm_oe * norm_og))

        obj_yaw = float(p.getEulerFromQuaternion(obj_orn)[2])
        spin = float(obj_ang_vel[2])

        state_vec = np.array(
            [
                obj_pos[0] - goal[0],  # 0: obj→goal x
                obj_pos[1] - goal[1],  # 1: obj→goal y
                ee_pos[0] - obj_pos[0],  # 2: ee→obj x
                ee_pos[1] - obj_pos[1],  # 3: ee→obj y
                float(obj_vel[0]),  # 4: object vel x
                float(obj_vel[1]),  # 5: object vel y
                spin,  # 6: object spin rate
                obj_yaw,  # 7: object yaw
                cos_approach,  # 8: approach alignment cos
                sin_approach,  # 9: approach alignment sin
            ],
            dtype=np.float32,
        )

        # Contact detection
        contact_pts = p.getContactPoints(bodyA=self.robotId, bodyB=self.object_uid, **self._pb())
        in_contact = contact_pts is not None and len(contact_pts) > 0

        # Distance for reward
        dist_obj_goal = float(np.linalg.norm(obj_pos[:2] - goal[:2]))

        return {
            "state_vec": state_vec,
            "obj_pos": obj_pos,
            "cos_approach": cos_approach,
            "spin": spin,
            "in_contact": in_contact,
            "dist_obj_goal": dist_obj_goal,
        }

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def step(self, action):
        if not self._connected():
            raise RuntimeError("PyBullet disconnected — use Ctrl+C, not the X button.")

        dx, dy = float(action[0]), float(action[1])
        self.current_ee_pos[0] = float(np.clip(self.current_ee_pos[0] + dx, *EE_X_BOUNDS))
        self.current_ee_pos[1] = float(np.clip(self.current_ee_pos[1] + dy, *EE_Y_BOUNDS))
        self.current_ee_pos[2] = self.fixed_z
        self._move_to_target(self.current_ee_pos)

        # ── [HER] Sparse reward ──────────────────────────────────────────────
        # 0.0 on success, -1.0 otherwise.
        # HER's replay buffer calls compute_reward() after each episode to
        # relabel failed transitions as virtual successes toward the achieved
        # object position. This converts 100% of transitions into useful
        # learning signal — dense shaping is not needed.
        s = self._compute_state_and_reward_components()

        terminated = bool(s["dist_obj_goal"] < self.success_threshold)
        reward = 0.0 if terminated else -1.0
        info = {"is_success": float(terminated)}

        obs = self._get_obs_from_components(s)
        return obs, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self._connected():
            raise RuntimeError("PyBullet server disconnected.")

        p.resetSimulation(**self._pb())
        p.setGravity(0, 0, -9.81, **self._pb())
        p.setPhysicsEngineParameter(numSolverIterations=50, **self._pb())
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), **self._pb())

        self.planeId = p.loadURDF("plane.urdf", **self._pb())
        p.changeDynamics(
            self.planeId,
            -1,
            lateralFriction=1.0,
            spinningFriction=0.01,
            rollingFriction=0.01,
            **self._pb(),
        )

        self.current_ee_pos = [0.3, 0.0, self.fixed_z]
        self._setup_robot()

        # ── Object selection ──────────────────────────────────────────────────
        if not self.object_urdfs:
            raise RuntimeError(f"No object URDFs found in {self.train_objects_dir}")
        if not self.texture_files:
            raise RuntimeError(f"No textures found in {self.texture_dir}")

        selected_urdf = random.choice(self.object_urdfs)
        self.current_object_name = os.path.basename(os.path.dirname(selected_urdf))
        # Object spawns on the left side of workspace
        # Spawn at z=0.005m (just above ground) so the object settles
        # immediately without falling through the gripper at z≈0.19m.
        # Spawning at z=0.1m caused the falling object to collide with
        # the gripper and get knocked outside the EE workspace.
        spawn_pos = [np.random.uniform(0.25, 0.32), np.random.uniform(-0.1, 0.1), 0.005]
        # Goal spawns on the right side — guaranteed separation
        self.goal_pos = [np.random.uniform(0.36, 0.44), np.random.uniform(-0.1, 0.1), 0.001]

        # ── Dynamic AABB scaling ──────────────────────────────────────────────
        temp_uid = p.loadURDF(selected_urdf, basePosition=[0, 0, 10], **self._pb())
        p.stepSimulation(**self._pb())

        min_pt = np.array(p.getAABB(temp_uid, -1, **self._pb())[0])
        max_pt = np.array(p.getAABB(temp_uid, -1, **self._pb())[1])
        for i in range(p.getNumJoints(temp_uid, **self._pb())):
            l_min, l_max = p.getAABB(temp_uid, i, **self._pb())
            min_pt = np.minimum(min_pt, np.array(l_min))
            max_pt = np.maximum(max_pt, np.array(l_max))
        p.removeBody(temp_uid, **self._pb())

        max_dim = max(max_pt - min_pt)
        if max_dim < 0.001:
            max_dim = 0.1
        target_size = 0.06
        dynamic_scale = float(np.clip(target_size / max_dim, 0.05, 5.0))

        self.object_uid = p.loadURDF(
            selected_urdf, basePosition=spawn_pos, globalScaling=dynamic_scale, **self._pb()
        )

        # ── Texture application ───────────────────────────────────────────────
        selected_texture_path = random.choice(self.texture_files)
        texture_id = p.loadTexture(selected_texture_path, **self._pb())
        p.changeVisualShape(
            self.object_uid, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=texture_id, **self._pb()
        )

        # ── Physics randomization (mass + friction) ───────────────────────────
        if self.randomization_mode == "baseline":
            mass = float(np.random.uniform(*self.baseline_mass_bounds))
            lateral_friction = float(np.random.uniform(*self.baseline_friction_bounds))
            # Baseline: zero physics prediction — agent cannot use encoder signal
            self._physics_pred = np.zeros(2, dtype=np.float32)
            self._current_category = None

        elif self.randomization_mode == "experimental":
            category_dir = os.path.basename(os.path.dirname(selected_texture_path))
            category = category_dir.replace("_", " ")
            if category not in self.semantic_mapping:
                raise ValueError(f"Semantic category '{category}' not found in mapping.")
            mass_bounds = self.semantic_mapping[category]["mass"]
            friction_bounds = self.semantic_mapping[category]["lateral_friction"]
            mass = float(np.random.uniform(*mass_bounds))
            lateral_friction = float(np.random.uniform(*friction_bounds))

            # ── [S2] Encoder inference ────────────────────────────────────────
            # Run the pre-trained texture encoder on the selected texture image
            # to produce a 2-dim physics prediction appended to observation.
            # Normalised to [0,1]: 0=min of all categories, 1=max.
            if self.encoder is not None:
                try:
                    import torchvision.transforms as T
                    from PIL import Image

                    transform = T.Compose(
                        [
                            T.Resize((64, 64)),
                            T.ToTensor(),
                            T.Normalize([0.5] * 3, [0.5] * 3),
                        ]
                    )
                    img = Image.open(selected_texture_path).convert("RGB")
                    inp = transform(img).unsqueeze(0)  # (1, 3, 64, 64)
                    with self._torch.no_grad():
                        logits = self.encoder(inp)
                        class_idx = int(logits.argmax(dim=-1).item())
                    pred = np.array(
                        self._encoder_physics_map.get(class_idx, [0.0, 0.0]), dtype=np.float32
                    )
                    self._physics_pred = pred
                except Exception:
                    self._physics_pred = np.zeros(2, dtype=np.float32)
            else:
                self._physics_pred = np.zeros(2, dtype=np.float32)

            # ── [S1] Asymmetric CoM via category-specific Izz scaling ─────────
            # Overwrites the uniform-box Izz computed in SysID below.
            # Store category so SysID block can apply the correct multiplier.
            self._current_category = category

        elif self.randomization_mode == "coupled_no_cue":
            # ── ABLATION: coupled distribution and inertia, cue withheld ──────
            # Same physics sampling and asymmetric Izz as 'experimental', but
            # dims 9-10 are permanently zero -- no encoder inference is ever
            # run. Isolates whether the coupled distribution ALONE (independent
            # of any visual cue) accounts for the Baseline-vs-PGDA gap.
            # Unlike 'no_texture', physics here come from the CATEGORY-SPECIFIC
            # bounds (same as experimental/oracle), not the baseline bounds --
            # this is the true same-distribution, no-cue control.
            category_dir = os.path.basename(os.path.dirname(selected_texture_path))
            category = category_dir.replace("_", " ")
            if category not in self.semantic_mapping:
                raise ValueError(f"coupled_no_cue mode: category '{category}' not in mapping.")
            mass_bounds = self.semantic_mapping[category]["mass"]
            friction_bounds = self.semantic_mapping[category]["lateral_friction"]
            mass = float(np.random.uniform(*mass_bounds))
            lateral_friction = float(np.random.uniform(*friction_bounds))

            # No encoder inference -- the cue is withheld by design, not by error.
            self._physics_pred = np.zeros(2, dtype=np.float32)

            # Set category so the asymmetric Izz block (below) applies the same
            # scaling as experimental/oracle -- dynamics must match exactly.
            self._current_category = category

        elif self.randomization_mode == "mismatched":
            # ── RQ2 mismatch experiment ───────────────────────────────────────
            # Texture is already applied above from the full texture bank.
            # Physics are sampled from a DIFFERENT category than the texture,
            # deliberately breaking the visual-physics correlation.
            # If the trained Agent B is using texture as a physics cue, its
            # success rate should drop significantly vs the matched condition.
            texture_category_dir = os.path.basename(os.path.dirname(selected_texture_path))
            texture_category = texture_category_dir.replace("_", " ")
            all_categories = list(self.semantic_mapping.keys())
            # Remove the texture category so physics always come from elsewhere
            other_categories = [c for c in all_categories if c != texture_category]
            if not other_categories:
                # Fallback: if only one category exists, use baseline
                mass = float(np.random.uniform(*self.baseline_mass_bounds))
                lateral_friction = float(np.random.uniform(*self.baseline_friction_bounds))
            else:
                physics_category = random.choice(other_categories)
                mass_bounds = self.semantic_mapping[physics_category]["mass"]
                friction_bounds = self.semantic_mapping[physics_category]["lateral_friction"]
                mass = float(np.random.uniform(*mass_bounds))
                lateral_friction = float(np.random.uniform(*friction_bounds))
            # Encoder still produces a prediction based on the VISIBLE texture
            # (correct for the texture shown, but wrong for the actual physics).
            # This is the intended mismatch: agent acts on a misleading signal.
            if self.encoder is not None:
                try:
                    import torchvision.transforms as T
                    from PIL import Image

                    transform = T.Compose(
                        [
                            T.Resize((64, 64)),
                            T.ToTensor(),
                            T.Normalize([0.5] * 3, [0.5] * 3),
                        ]
                    )
                    img = Image.open(selected_texture_path).convert("RGB")
                    inp = transform(img).unsqueeze(0)
                    with self._torch.no_grad():
                        logits = self.encoder(inp)
                        class_idx = int(logits.argmax(dim=-1).item())
                    self._physics_pred = np.array(
                        self._encoder_physics_map.get(class_idx, [0.0, 0.0]), dtype=np.float32
                    )
                except Exception:
                    self._physics_pred = np.zeros(2, dtype=np.float32)
            else:
                self._physics_pred = np.zeros(2, dtype=np.float32)
            # No asymmetric Izz in mismatch — physics already deliberately mismatched
            self._current_category = None

        elif self.randomization_mode == "no_texture":
            # ── RQ2 control condition ─────────────────────────────────────────
            # Remove all texture information: apply uniform grey colour.
            # Physics sampled from the same baseline uncorrelated distribution
            # as Agent A. Tests whether the trained policy uses texture at all:
            # if success rate is unchanged vs matched condition, the policy
            # ignores texture and relies purely on the 9-dim state vector.
            p.changeVisualShape(
                self.object_uid,
                -1,
                rgbaColor=[0.6, 0.6, 0.6, 1.0],  # uniform grey — no texture
                textureUniqueId=-1,
                **self._pb(),
            )
            mass = float(np.random.uniform(*self.baseline_mass_bounds))
            lateral_friction = float(np.random.uniform(*self.baseline_friction_bounds))
            self._physics_pred = np.zeros(2, dtype=np.float32)
            self._current_category = None

        elif self.randomization_mode == "oracle":
            # ── ABLATION: Oracle physics signal ───────────────────────────────
            # Identical physics setup to experimental mode:
            #   - texture is already applied above from the full texture bank
            #   - category derived from the selected texture path
            #   - mass + friction sampled from category-specific bounds
            #   - asymmetric Izz applied (via _current_category below)
            #
            # The ONLY difference from experimental: dims 9-10 of the
            # observation receive the GROUND-TRUTH normalised physics values
            # instead of the encoder's prediction. This is the upper bound —
            # it answers "how well would the agent do with a perfect encoder?"
            category_dir = os.path.basename(os.path.dirname(selected_texture_path))
            category = category_dir.replace("_", " ")
            if category not in self.semantic_mapping:
                raise ValueError(f"Oracle mode: category '{category}' not in mapping.")
            mass_bounds = self.semantic_mapping[category]["mass"]
            friction_bounds = self.semantic_mapping[category]["lateral_friction"]
            mass = float(np.random.uniform(*mass_bounds))
            lateral_friction = float(np.random.uniform(*friction_bounds))

            # Normalise ground-truth physics
            norm_mass = float(
                np.clip(
                    (mass - self.baseline_mass_bounds[0])
                    / (self.baseline_mass_bounds[1] - self.baseline_mass_bounds[0] + 1e-8),
                    0.0,
                    1.0,
                )
            )
            norm_fric = float(
                np.clip(
                    (lateral_friction - self.baseline_friction_bounds[0])
                    / (self.baseline_friction_bounds[1] - self.baseline_friction_bounds[0] + 1e-8),
                    0.0,
                    1.0,
                )
            )
            self._physics_pred = np.array([norm_mass, norm_fric], dtype=np.float32)
            # Set category so asymmetric Izz block applies correctly
            self._current_category = category

        elif self.randomization_mode == "no_dr":
            # ── ABLATION: No domain randomisation ─────────────────────────────
            # Fully deterministic environment — neither physics nor texture
            # varies across episodes.
            #   - Uniform grey object (no texture, no encoder signal)
            #   - Fixed mass and friction at the midpoint of baseline bounds
            #   - Izz unchanged (no asymmetric scaling — category is None)
            #   - dims 9-10 always [0, 0]
            #
            # This is the lower bound / sanity check. Any agent should solve
            # this easily since there is nothing to adapt to. If it does not,
            # the architecture or algorithm has a fundamental problem.
            p.changeVisualShape(
                self.object_uid,
                -1,
                rgbaColor=[0.7, 0.7, 0.7, 1.0],  # uniform grey
                textureUniqueId=-1,
                **self._pb(),
            )
            # Fixed at midpoint of the baseline sampling range
            mass = float(
                (self.baseline_mass_bounds[0] + self.baseline_mass_bounds[1]) / 2.0
            )  # 2.55 kg
            lateral_friction = float(
                (self.baseline_friction_bounds[0] + self.baseline_friction_bounds[1]) / 2.0
            )  # 0.65
            self._physics_pred = np.zeros(2, dtype=np.float32)
            self._current_category = None

        else:
            raise ValueError(
                "randomization_mode must be one of: 'baseline', "
                "'experimental', 'mismatched', 'no_texture', "
                "'oracle', 'no_dr', 'coupled_no_cue'"
            )

        # ── [P1] SysID: Physically-correct inertia + stochastic CoM offset ───
        #
        # Raw GSO objects carry b3Warning "No inertial data" → PyBullet defaults
        # to m=1 kg, I=diag(1,1,1) kg⋅m².  For a 6cm object this is ~7 orders
        # of magnitude too large, suppressing tipping in training but not eval.
        #
        # Uniform box inertia for a cube of side 2·half:
        #   I_xx = I_yy = I_zz = (m/12)·2·(2·half)²  = m·(2·half)²/6
        #
        # Anisotropy scale [0.5, 2.0] per axis models non-uniform photogrammetry
        # mass distribution without requiring internal mesh geometry.
        #
        # CoM offset models objects with eccentric mass (handles, hollow bodies).
        # |Δcom| ≤ 0.3 · half_extent keeps CoM physically inside the object.
        # Z-component forced positive: CoM is always above the base plane.
        half = (target_size * dynamic_scale) / 2.0  # ~0.03 m for 6cm object

        base_inertia = (mass / 12.0) * (2.0 * half) ** 2  # uniform cube
        anisotropy = np.random.uniform(0.5, 2.0, 3)
        Ixx = float(base_inertia * anisotropy[0])
        Iyy = float(base_inertia * anisotropy[1])
        Izz = float(base_inertia * anisotropy[2])

        com_offset = np.random.uniform(-0.3 * half, 0.3 * half, 3)
        com_offset[2] = abs(com_offset[2])  # CoM above base

        # ── [S1] Asymmetric Izz: category-specific rotational inertia ───────
        # Experimental mode: scales Izz so texture visually predicts spin
        # resistance. Heavy Rust (low Izz) → spins easily if pushed off-centre.
        # Wet Ice (high Izz) → resists spin, translates cleanly.
        # Baseline mode: _current_category is None → Izz unchanged (random).
        cat = getattr(self, "_current_category", None)
        if cat is not None and cat in CATEGORY_IZZ_SCALE:
            Izz = float(Izz * CATEGORY_IZZ_SCALE[cat])

        p.changeDynamics(
            bodyUniqueId=self.object_uid,
            linkIndex=-1,
            mass=mass,
            lateralFriction=lateral_friction,
            localInertiaDiagonal=[Ixx, Iyy, Izz],
            **self._pb(),
        )

        # Record for optional logging / debugging
        self._last_mass = mass
        self._last_friction = lateral_friction
        self._last_inertia = [Ixx, Iyy, Izz]

        # Settle object onto plane
        for _ in range(100):
            p.stepSimulation(**self._pb())

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[0.3, 0, 0.3],
                **self._pb(),
            )

        # ── Goal marker ───────────────────────────────────────────────────────
        # goal_pos already set to far zone [0.36, 0.44] above (spawn fix).
        # Do NOT resample here — that would reinstate the overlap bug.
        goal_visual = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.04, length=0.002, rgbaColor=[1, 0, 0, 0.5], **self._pb()
        )
        self.goal_uid = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=goal_visual, basePosition=self.goal_pos, **self._pb()
        )

        # Initialise potential-based reward baseline
        obj_pos_init = np.array(p.getBasePositionAndOrientation(self.object_uid, **self._pb())[0])
        self.prev_dist_obj_goal = float(
            np.linalg.norm(obj_pos_init[:2] - np.array(self.goal_pos)[:2])
        )

        return self._get_obs(), {}

    # ── Observation helpers ───────────────────────────────────────────────────

    def _get_obs(self):
        """Standard entry point — computes state components then packages obs."""
        s = self._compute_state_and_reward_components()
        return self._get_obs_from_components(s)

    def _get_obs_from_components(self, s):
        """
        [HER] Returns GoalEnv-format observation dict.
        observation  : 9-dim goal-free state
        achieved_goal: object (x,y) — what HER relabels as achieved
        desired_goal : target (x,y) — what HER relabels as desired
        """
        if self.ee_link_index != -1:
            ee_pos = np.array(
                p.getLinkState(self.robotId, self.ee_link_index, **self._pb())[0], dtype=np.float32
            )
        else:
            ee_pos = np.zeros(3, dtype=np.float32)

        obj_pos = np.array(
            p.getBasePositionAndOrientation(self.object_uid, **self._pb())[0], dtype=np.float32
        )

        obs_vec = np.array(
            [
                ee_pos[0],
                ee_pos[1],
                ee_pos[2],  # 0-2: EE xyz
                ee_pos[0] - obj_pos[0],  # 3:   EE→obj x
                ee_pos[1] - obj_pos[1],  # 4:   EE→obj y
                s["state_vec"][4],
                s["state_vec"][5],  # 5-6: obj vel xy
                s["state_vec"][6],  # 7:   obj spin z
                s["state_vec"][7],  # 8:   obj yaw
                self._physics_pred[0],  # 9:   pred norm_mass
                self._physics_pred[1],  # 10:  pred norm_friction
            ],
            dtype=np.float32,
        )

        return {
            "observation": obs_vec,
            "achieved_goal": obj_pos[:2].copy(),
            "desired_goal": np.array(self.goal_pos[:2], dtype=np.float32),
        }

    # ── Curriculum interface ──────────────────────────────────────────────────

    def set_success_threshold(self, value: float) -> None:
        """
        Setter for the success threshold, called by CurriculumCallback via
        SubprocVecEnv.env_method().

        SubprocVecEnv runs each environment in an isolated subprocess.
        The parent process cannot access env attributes directly (no .envs
        list). Instead it sends an interprocess message via env_method(), which
        the worker receives and dispatches as getattr(env, method_name)(args).
        gymnasium.Wrapper.__getattr__ delegates unknown attribute lookups to
        the wrapped env, so this method is reachable through the TimeLimit
        wrapper without any changes to the wrapper itself.
        """
        self.success_threshold = float(value)

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info
    ) -> np.ndarray:
        """
        [HER] Required by GoalEnv interface.
        Called by HerReplayBuffer to relabel rewards in batches.
        achieved_goal, desired_goal: arrays of shape (N, 2) or (2,).
        Returns 0.0 for success, -1.0 otherwise — matching sparse reward.
        """
        dist = np.linalg.norm(np.array(achieved_goal) - np.array(desired_goal), axis=-1)
        return -(dist >= self.success_threshold).astype(np.float32)

    def render(self):
        pass

    def close(self):
        if self._connected():
            p.disconnect(self._client)
