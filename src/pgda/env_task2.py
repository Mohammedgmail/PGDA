# pushing_env_task2.py — Task 2: slide the object to a target with a single push
# Physically-Grounded Diffusion Augmentation (PGDA)
#
# WHAT CHANGES FROM TASK 1 (pushing_env.py is NOT modified; this is a subclass)
#   Robot, simulator, objects, materials, textures, encoder, physics sampling,
#   code in obs dims 9-10 and the action space are inherited unchanged.
#   Only the task is new:
#
#   Phase 0  APPROACH  The policy moves the pusher as in task 1.
#   Phase 1  PUSH      Starts at the first control step after which the robot
#                      touches the object, or the object moves (speed > 1 cm/s,
#                      displacement > 3 mm or yaw change > 0.03 rad). From then
#                      on the policy has K_WINDOW more control steps.
#   Phase 2  COAST     The arm retracts 3 cm away from the push direction, and
#                      the simulation runs until the object is at rest
#                      (linear < 5 mm/s and angular < 0.05 rad/s for 12
#                      consecutive 240 Hz steps) or 3 s have passed.
#                      The episode ends here (terminated=True).
#   Success: the object is at rest and its rest position is within delta
#   (success_threshold) of the goal. No success is possible before the coast.
#
#   Geometry (drawn AFTER the base reset from the same global RNG, so every
#   mode sees identical objects, textures, physics, starts and goals):
#     object start x in OBJ_X, y in OBJ_Y (re-placed after the base reset);
#     goal = start + D * (cos th, sin th), D in D_RANGE, |th| <= THETA_MAX.
#     The pusher starts parked at START_EE, behind every possible object.
#
# REWARD AND HER (requirement R6)
#   r = 0                 at the end of the coast if success
#   r = -1/(1-gamma)      at the end of the coast if not success
#   r = -1                on every other step
#   A failed end of coast is an absorbing failure: -1 per step forever has
#   value -1/(1-gamma), which is what a never-successful task-1 episode is
#   worth. Without it, ending the episode quickly with any push would be
#   cheaper than a careful approach (each extra step costs -1, a miss only 1).
#   compute_reward() reproduces this rule exactly from (achieved goal, desired
#   goal, info["rest_end"], info["at_rest"]), so HER relabelling stays
#   consistent: a transition can only be a success if it is the end of a coast
#   with the object at rest, and its relabelled goal lies within delta.
#   IMPORTANT: SB3 passes empty infos to compute_reward unless the training
#   script sets   replay_buffer_kwargs=dict(..., copy_info_dict=True).
#   compute_reward() raises an error if it receives infos without the flags,
#   so a misconfigured run fails at the first gradient step instead of
#   training silently on wrong rewards.
#
# OBSERVATION
#   dims 0-10 exactly as task 1 (dims 9-10: physics code), plus
#   dim 11: 1.0 once the push window has started, else 0.0
#   dim 12: push steps left / K_WINDOW (1.0 during the approach)
#
# ORACLE (task-2 version)
#   dims 9-10 = true (mass, friction) normalised over the UNION of the three
#   category ranges (m in [0.5, 3.0], mu in [0.01, 1.50]) without clipping,
#   i.e. on the same scale as the category codes. ORACLE_SCALE = "union_log"
#   uses log(mu) instead (Wet Ice then spans 0.00-0.32 of the channel instead
#   of 0.000-0.027). Decide before training; see the reply that came with
#   this file.
#
# MODES: 'experimental', 'oracle', 'coupled_no_cue', 'baseline'.
#   'mismatched' and 'no_texture' are refused (rule 1: interventions go
#   through with_code() only). 'no_dr' is refused because it draws no physics
#   random numbers, so its episodes would not match the other modes.
#
# SELF-TEST
#   python pushing_env_task2.py --selftest
#   Prints per-episode physics, code and phase logic in every mode, checks HER
#   consistency, measures the arm's real push speed, the scripted pusher's gap
#   to the object, the free-coast distance against v^2/(2 mu g), seeded-replay
#   determinism and timing, and writes task2_selftest.json. Exit code 1 if any
#   check fails.
#
# VERSION 2 (13 Oct 2026), after the first self-test on the real assets
#   - Common-episode check: the Baseline settles its object up to ~0.05 mm
#     differently (other mass and friction), so it is now compared on the drawn
#     values (object, spawn point, D, direction) exactly and on the settled
#     start within 1 mm. Coupled modes must still match exactly.
#   - PyBullet's saveState/restoreState does not reproduce a fresh run once
#     objects touch (up to 1 cm on ice), so the snapshot helpers are removed.
#     The pilot (script 21) now replays every branch from a seeded reset,
#     which is bit-exact; section [H] checks this.
#   - The scripted approach measures the pusher-object gap with PyBullet's
#     closest-point query instead of link bounding boxes, which overestimated
#     the RG2's reach by 3-4 cm and pushed the start point off the push line.
#   The environment itself (phases, reward, observation) is unchanged.

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pybullet as p
from gymnasium import spaces

from pushing_env import (UR3ePushingEnv, EE_X_BOUNDS, EE_Y_BOUNDS,
                         CATEGORY_IZZ_SCALE)

# ── Task-2 configuration (defaults; the pilot gate decides the final values) ──
TASK2_DEFAULTS = {
    "k_window":      3,              # control steps the policy keeps after onset
    "d_range":       (0.08, 0.15),   # goal distance from the object's start (m)
    "theta_max_deg": 20.0,           # goal direction within +-theta of +x
    "obj_x":         (0.29, 0.33),   # object start region (m)
    "obj_y":         (-0.08, 0.08),
    "start_ee":      (0.22, 0.0),    # parked pusher, behind every object
    "spawn_z":       0.005,          # same spawn height as the base reset
    "v_onset":       0.01,           # m/s
    "d_onset":       0.003,          # m
    "yaw_onset":     0.03,           # rad
    "contact_tol":   0.0005,         # m; contact points closer than this count
    "v_rest":        0.005,          # m/s
    "w_rest":        0.05,           # rad/s
    "n_rest":        12,             # consecutive 240 Hz steps below both
    "coast_max_s":   3.0,
    "settle_max_s":  2.0,            # extra settling at reset (ice can drift)
    "retract":       0.03,           # m, arm retraction at the end of the window
    "gamma":         0.95,           # MUST equal the training gamma
    "oracle_scale":  "union_linear", # or "union_log"
}
TASK2_MODES = ("experimental", "oracle", "coupled_no_cue", "baseline")
APPROACH, PUSH, DONE = 0, 1, 2
PHASE_NAME = {APPROACH: "approach", PUSH: "push", DONE: "done"}
SIM_HZ = 240.0
CONTROL_DT = 20 / SIM_HZ              # one control step = 20 sim steps

_INFO_HELP = ("compute_reward() received an info dict without 'rest_end'. "
              "Set replay_buffer_kwargs=dict(..., copy_info_dict=True) in the "
              "SAC constructor; otherwise SB3 passes empty infos to HER and "
              "every relabelled reward would be wrong.")


class UR3ePushingTask2Env(UR3ePushingEnv):
    """Task 2: a single push that must leave the object at rest on the goal."""

    def __init__(self, render_mode=None, randomization_mode="experimental",
                 reward_weights=None, encoder_path=None, task2_config=None):
        cfg = dict(TASK2_DEFAULTS)
        unknown = set(task2_config or {}) - set(TASK2_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown task2_config keys: {sorted(unknown)}")
        cfg.update(task2_config or {})
        if randomization_mode not in TASK2_MODES:
            raise ValueError(f"Task 2 supports modes {TASK2_MODES}, not "
                             f"'{randomization_mode}' (see header).")
        if cfg["oracle_scale"] not in ("union_linear", "union_log"):
            raise ValueError("oracle_scale must be 'union_linear' or 'union_log'")
        self.cfg = cfg
        super().__init__(render_mode=render_mode,
                         randomization_mode=randomization_mode,
                         reward_weights=reward_weights,
                         encoder_path=encoder_path)
        self.k_window = int(cfg["k_window"])
        self.fail_reward = -1.0 / (1.0 - float(cfg["gamma"]))
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-10.0, high=10.0, shape=(13,),
                                      dtype=np.float32),
            "achieved_goal": spaces.Box(low=-2.0, high=2.0, shape=(2,),
                                        dtype=np.float32),
            "desired_goal": spaces.Box(low=-2.0, high=2.0, shape=(2,),
                                       dtype=np.float32),
        })
        m = self.semantic_mapping
        self._union = ((min(v["mass"][0] for v in m.values()),
                        max(v["mass"][1] for v in m.values())),
                       (min(v["lateral_friction"][0] for v in m.values()),
                        max(v["lateral_friction"][1] for v in m.values())))
        (m0, m1), (f0, f1) = self._union
        self.category_codes = {   # union-normalised category midpoints
            c: [(np.mean(v["mass"]) - m0) / (m1 - m0),
                (np.mean(v["lateral_friction"]) - f0) / (f1 - f0)]
            for c, v in m.items()}
        self._phase, self._push_left, self._n_steps = APPROACH, self.k_window, 0
        self._onset_step, self._onset_reason = None, ""
        self._obj_start = np.zeros(3)
        self._yaw_start = 0.0
        self._u_goal = np.array([1.0, 0.0])
        self._goal_dist, self._goal_theta = 0.0, 0.0
        self._spawn_xy = (0.0, 0.0)
        self._reset_info = {}

    # ── Robot start pose ──────────────────────────────────────────────────────
    def _setup_robot(self):
        # The base reset sets current_ee_pos = (0.3, 0) right before calling
        # this, a point inside the object spawn region. Task 2 parks the pusher
        # behind the object region instead, so nothing is touched at reset.
        sx, sy = self.cfg["start_ee"]
        self.current_ee_pos = [float(sx), float(sy), self.fixed_z]
        super()._setup_robot()

    # ── Small helpers ─────────────────────────────────────────────────────────
    def _obj_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.object_uid, **self._pb())
        return np.array(pos), float(p.getEulerFromQuaternion(orn)[2])

    def _obj_vel(self):
        lin, ang = p.getBaseVelocity(self.object_uid, **self._pb())
        return np.array(lin), np.array(ang)

    def _ee_actual(self):
        return np.array(p.getLinkState(self.robotId, self.ee_link_index,
                                       **self._pb())[0])

    def _robot_object_contact(self):
        pts = p.getContactPoints(bodyA=self.robotId, bodyB=self.object_uid,
                                 **self._pb()) or []
        return any(c[8] <= self.cfg["contact_tol"] for c in pts)

    def _is_still(self):
        lin, ang = self._obj_vel()
        return (np.linalg.norm(lin) < self.cfg["v_rest"]
                and np.linalg.norm(ang) < self.cfg["w_rest"])

    def _settle_until_rest(self):
        pb, n_need = self._pb(), int(self.cfg["n_rest"])
        max_steps, run = int(round(self.cfg["settle_max_s"] * SIM_HZ)), 0
        for n in range(max_steps + 1):
            run = run + 1 if self._is_still() else 0
            if run >= n_need:
                return n, True
            p.stepSimulation(**pb)
        p.resetBaseVelocity(self.object_uid, [0, 0, 0], [0, 0, 0], **pb)
        for _ in range(24):
            p.stepSimulation(**pb)
        return max_steps, False

    def _oracle_code(self):
        (m0, m1), (f0, f1) = self._union
        cm = (self._last_mass - m0) / (m1 - m0)
        if self.cfg["oracle_scale"] == "union_log":
            cf = (np.log(self._last_friction) - np.log(f0)) / (np.log(f1) - np.log(f0))
        else:
            cf = (self._last_friction - f0) / (f1 - f0)
        return np.array([cm, cf], dtype=np.float32)   # no clipping

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        self._phase, self._push_left = APPROACH, self.k_window
        super().reset(seed=seed, options=options)
        cfg, pb = self.cfg, self._pb()

        # Task-2 geometry: four extra draws from the global RNG, identical in
        # every supported mode (the base reset makes the same number of draws).
        ox = float(np.random.uniform(*cfg["obj_x"]))
        oy = float(np.random.uniform(*cfg["obj_y"]))
        dist = float(np.random.uniform(*cfg["d_range"]))
        th = float(np.radians(np.random.uniform(-cfg["theta_max_deg"],
                                                cfg["theta_max_deg"])))
        self._spawn_xy = (ox, oy)

        # Re-place the object exactly as the base spawns it (same height, same
        # orientation, zero velocity), which also erases any contact with the
        # parked pusher during the base reset.
        p.resetBasePositionAndOrientation(self.object_uid, [ox, oy, cfg["spawn_z"]],
                                          [0, 0, 0, 1], **pb)
        p.resetBaseVelocity(self.object_uid, [0, 0, 0], [0, 0, 0], **pb)
        for _ in range(100):
            p.stepSimulation(**pb)
        n_settle, settled = self._settle_until_rest()

        pos, yaw = self._obj_pose()
        self._obj_start, self._yaw_start = pos, yaw
        self._u_goal = np.array([np.cos(th), np.sin(th)])
        self._goal_dist, self._goal_theta = dist, th
        self.goal_pos = [float(pos[0] + dist * np.cos(th)),
                         float(pos[1] + dist * np.sin(th)), 0.001]
        p.resetBasePositionAndOrientation(self.goal_uid, self.goal_pos,
                                          [0, 0, 0, 1], **pb)
        self.prev_dist_obj_goal = dist
        if self.randomization_mode == "oracle":
            self._physics_pred = self._oracle_code()

        self._phase, self._push_left, self._n_steps = APPROACH, self.k_window, 0
        self._onset_step, self._onset_reason = None, ""
        self._reset_info = {"settle_steps": int(n_settle), "settled": bool(settled),
                            "contact_at_reset": bool(self._robot_object_contact())}
        return self._get_obs(), {}

    def step(self, action):
        if self._phase == DONE:
            raise RuntimeError("step() called after the episode ended; call reset().")
        if not self._connected():
            raise RuntimeError("PyBullet disconnected.")
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:2],
                    self.action_space.low, self.action_space.high)
        self.current_ee_pos[0] = float(np.clip(self.current_ee_pos[0] + a[0], *EE_X_BOUNDS))
        self.current_ee_pos[1] = float(np.clip(self.current_ee_pos[1] + a[1], *EE_Y_BOUNDS))
        self.current_ee_pos[2] = self.fixed_z
        self._move_to_target(self.current_ee_pos)
        self._n_steps += 1

        finished = False
        if self._phase == APPROACH:
            hit, why = self._onset_check()
            if hit:
                self._phase, self._onset_step = PUSH, self._n_steps
                self._onset_reason, self._push_left = why, self.k_window
                finished = self._push_left <= 0
        elif self._phase == PUSH:
            self._push_left -= 1
            finished = self._push_left <= 0
        coast = self._stop_and_coast() if finished else None
        return self._finish_step(coast)

    def _onset_check(self):
        pos, yaw = self._obj_pose()
        lin, _ = self._obj_vel()
        why = []
        if self._robot_object_contact():
            why.append("contact")
        if np.hypot(lin[0], lin[1]) > self.cfg["v_onset"]:
            why.append("speed")
        if np.hypot(*(pos[:2] - self._obj_start[:2])) > self.cfg["d_onset"]:
            why.append("displacement")
        if abs((yaw - self._yaw_start + np.pi) % (2 * np.pi) - np.pi) > self.cfg["yaw_onset"]:
            why.append("yaw")
        return bool(why), "+".join(why)

    def _stop_and_coast(self, stop_beyond=None):
        """End of the push window: retract the arm, simulate until rest."""
        cfg, pb = self.cfg, self._pb()
        pos, _ = self._obj_pose()
        d = pos[:2] - self._obj_start[:2]
        u = d / np.linalg.norm(d) if np.linalg.norm(d) > 0.002 else self._u_goal
        ee = self._ee_actual()
        tgt = [float(np.clip(ee[0] - cfg["retract"] * u[0], *EE_X_BOUNDS)),
               float(np.clip(ee[1] - cfg["retract"] * u[1], *EE_Y_BOUNDS)),
               self.fixed_z]
        self.current_ee_pos = list(tgt)
        self._move_to_target(list(tgt), settle_steps=0)   # command only

        lin, _ = self._obj_vel()
        v_end = float(np.hypot(lin[0], lin[1]))
        peak, run, at_rest, early, n = v_end, 0, False, False, 0
        for n in range(1, int(round(cfg["coast_max_s"] * SIM_HZ)) + 1):
            p.stepSimulation(**pb)
            if self.render_mode == "human":
                time.sleep(1.0 / SIM_HZ)
            lin, ang = self._obj_vel()
            peak = max(peak, float(np.hypot(lin[0], lin[1])))
            still = (np.linalg.norm(lin) < cfg["v_rest"]
                     and np.linalg.norm(ang) < cfg["w_rest"])
            run = run + 1 if still else 0
            if run >= cfg["n_rest"]:
                at_rest = True
                break
            if stop_beyond is not None:     # pilot only: outcome already decided
                q, _ = self._obj_pose()
                s = float(np.dot(q[:2] - self._obj_start[:2], self._u_goal))
                if s > stop_beyond and float(np.dot(lin[:2], self._u_goal)) > 0:
                    early = True
                    break
        self._phase = DONE
        return {"at_rest": at_rest, "early_exit": early, "coast_steps": n,
                "coast_time": n / SIM_HZ, "v_window_end": v_end, "peak_speed": peak}

    def _finish_step(self, coast):
        obs = self._get_obs()
        if coast is None:
            info = {"is_success": 0.0, "rest_end": 0.0, "at_rest": 0.0}
        else:
            dist = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
            ok = coast["at_rest"] and dist < self.success_threshold
            info = {"is_success": float(ok), "rest_end": 1.0,
                    "at_rest": float(coast["at_rest"]), "rest_dist": dist,
                    "push_steps": int(self._n_steps - (self._onset_step or self._n_steps) + 1),
                    "onset_step": int(self._onset_step or self._n_steps),
                    "coast_time": float(coast["coast_time"]),
                    "v_window_end": float(coast["v_window_end"]),
                    "peak_speed": float(coast["peak_speed"]),
                    "early_exit": float(coast["early_exit"])}
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))
        return obs, reward, coast is not None, False, info

    def _get_obs_from_components(self, s):
        obs = super()._get_obs_from_components(s)
        if self._phase == APPROACH:
            extra = [0.0, 1.0]
        elif self._phase == PUSH:
            extra = [1.0, self._push_left / max(self.k_window, 1)]
        else:
            extra = [1.0, 0.0]
        obs["observation"] = np.concatenate(
            [obs["observation"], np.array(extra, dtype=np.float32)]).astype(np.float32)
        return obs

    # ── HER-consistent reward ─────────────────────────────────────────────────
    @staticmethod
    def _info_field(info, key, shape):
        if isinstance(info, dict):
            if key not in info:
                raise RuntimeError(_INFO_HELP)
            return np.full(shape, float(info[key]))
        items = list(np.ravel(np.asarray(info, dtype=object)))
        if any((not isinstance(i, dict)) or key not in i for i in items):
            raise RuntimeError(_INFO_HELP)
        return np.array([float(i[key]) for i in items]).reshape(shape)

    def compute_reward(self, achieved_goal, desired_goal, info):
        dist = np.linalg.norm(np.asarray(achieved_goal, dtype=np.float64)
                              - np.asarray(desired_goal, dtype=np.float64), axis=-1)
        end = self._info_field(info, "rest_end", dist.shape) > 0.5
        rest = self._info_field(info, "at_rest", dist.shape) > 0.5
        success = end & rest & (dist < self.success_threshold)
        return np.where(success, 0.0,
                        np.where(end, self.fail_reward, -1.0)).astype(np.float32)

    # ── Helpers for the scripted pilot (not used by RL training) ──────────────
    def set_push_window(self, k):
        """Change K for the current episode (pilot only)."""
        self.k_window = int(k)
        if self._phase == APPROACH:
            self._push_left = self.k_window

    def finish_window(self, stop_beyond=None):
        """End the push window now (pilot only); same return as step()."""
        if self._phase == DONE:
            raise RuntimeError("episode already ended")
        return self._finish_step(self._stop_and_coast(stop_beyond))

    def episode_record(self):
        cat = getattr(self, "_current_category", None)
        return {"category": cat, "object": self.current_object_name,
                "mass": float(self._last_mass), "mu": float(self._last_friction),
                "inertia": [float(v) for v in self._last_inertia],
                "izz_scale": (CATEGORY_IZZ_SCALE.get(cat) if cat else None),
                "code": [float(v) for v in self._physics_pred],
                "spawn": [float(v) for v in self._spawn_xy],
                "start": [float(v) for v in self._obj_start[:2]],
                "goal": [float(v) for v in self.goal_pos[:2]],
                "goal_dist": float(self._goal_dist),
                "goal_theta_deg": float(np.degrees(self._goal_theta)),
                "reset": dict(self._reset_info)}


# ═════════════════════════════════════════════════════════════════════════════
# Scripted pusher (shared by the self-test and 21-scripted_voi_task2.py)
# ═════════════════════════════════════════════════════════════════════════════

def robot_object_gap(env, max_d=0.3):
    """Smallest distance (m) between any robot link and the object, from
    PyBullet's closest-point query on the actual collision shapes."""
    pts = p.getClosestPoints(bodyA=env.robotId, bodyB=env.object_uid,
                             distance=max_d, **env._pb()) or []
    return float(min((c[8] for c in pts), default=max_d))


def move_to(env, target_xy, step_max=0.012, tol=0.002, max_steps=40, log=None,
            compensate=False):
    """Move the pusher towards target_xy during the approach phase.
    Arrives when the command has reached the target and the ACTUAL end effector
    is within `tol` of it, or (compensate=False) has stopped moving. With
    compensate=True a static inverse-kinematics offset is cancelled by shifting
    the command once the arm has stopped. Every action sent is appended to
    `log`, so the motion can be replayed exactly. Returns False if the push
    window opened or the target was not reached in max_steps."""
    target = np.asarray(target_xy, dtype=float)
    lo_b = np.array([EE_X_BOUNDS[0], EE_Y_BOUNDS[0]])
    hi_b = np.array([EE_X_BOUNDS[1], EE_Y_BOUNDS[1]])
    goal, prev = target.copy(), env._ee_actual()[:2]
    for _ in range(max_steps):
        cmd = np.clip(goal - np.asarray(env.current_ee_pos[:2]), -step_max, step_max)
        env.step(cmd)
        if log is not None:
            log.append([float(cmd[0]), float(cmd[1])])
        if env._phase != APPROACH:
            return False
        act = env._ee_actual()[:2]
        commanded = np.linalg.norm(goal - np.asarray(env.current_ee_pos[:2])) < 1e-6
        stopped = np.linalg.norm(act - prev) < 5e-5
        prev = act
        if commanded and np.linalg.norm(target - act) < tol:
            return True
        if commanded and stopped:
            if not compensate:
                return True
            goal = np.clip(goal + (target - act), lo_b, hi_b)
    return False


def _line_s_min(u, l0):
    """Smallest s for which p(s) = s*u + l0*n lies inside the EE bounds."""
    n = np.array([-u[1], u[0]])
    lo = -np.inf
    for ax, (bmin, bmax) in enumerate((EE_X_BOUNDS, EE_Y_BOUNDS)):
        a, b = float(u[ax]), float(l0 * n[ax])            # coordinate = a*s + b
        if abs(a) > 1e-12:
            lo = max(lo, min((bmin - b) / a, (bmax - b) / a))
    return lo


def scripted_approach(env, gap=0.015, backoff=0.08, tol=0.001, max_iter=25):
    """Park the pusher on the push line (through the object's centre along the
    goal direction), `gap` metres from the object.
      1. slide sideways along the back edge of the workspace to the line,
      2. move to a point on the line `backoff` behind the object's centre
         (or the nearest point inside the workspace),
      3. close in along the line until the closest-point distance between the
         robot and the object equals `gap`.
    Returns a dict; 'actions' replays the approach exactly after a seeded reset."""
    u = env._u_goal
    n = np.array([-u[1], u[0]])
    c = env._obj_start[:2]
    s_c, l0 = float(c @ u), float(c @ n)
    s_far = max(s_c - backoff, _line_s_min(u, l0) + 0.006)   # room to compensate
    p_far = s_far * u + l0 * n
    log = []
    ok = (move_to(env, [env.current_ee_pos[0], p_far[1]], tol=0.006, log=log)
          and move_to(env, p_far, tol=0.0005, log=log, compensate=True))
    far_gap = robot_object_gap(env)
    iters = 0
    while ok and iters < max_iter:
        d = robot_object_gap(env)
        if d - gap <= tol:
            break
        tgt = env._ee_actual()[:2] + min(d - gap, 0.012) * u
        ok = move_to(env, tgt, tol=0.0005, log=log, compensate=True)
        iters += 1
    ee = env._ee_actual()[:2]
    return {"ok": bool(ok and env._phase == APPROACH),
            "gap": robot_object_gap(env), "gap_at_far_point": far_gap,
            "lateral_offset": float(ee @ n - l0),
            "limited_by_workspace": bool(s_far > s_c - backoff + 1e-9),
            "pre_push": [float(v) for v in ee], "steps": int(env._n_steps),
            "iterations": iters, "onset_during_approach": env._phase != APPROACH,
            "actions": log}


def push_action(env, v):
    """Constant commanded push at v m/s along the goal direction."""
    return env._u_goal * (v * CONTROL_DT)


# ═════════════════════════════════════════════════════════════════════════════
# Self-test
# ═════════════════════════════════════════════════════════════════════════════

SELFTEST_ENCODER_PATH = ("/Users/mohammedalfandi/Documents/Robotics and AI /"
                         "MSc Project /Codes-2026/texture_encoder.pt")
SELFTEST_SEED_BASE = 3_000_000
SELFTEST_FILE = "task2_selftest.json"


class _Checks:
    def __init__(self):
        self.items = []

    def __call__(self, name, ok, detail=""):
        self.items.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        if not ok:
            print(f"    FAIL  {name}  {detail}")
        return ok

    def summary(self):
        bad = [c for c in self.items if not c["ok"]]
        return len(self.items) - len(bad), bad


def _appr_summary(appr):
    return {k: v for k, v in appr.items() if k != "actions"}


def _run_to_end(env, a, max_steps=200):
    done, n = False, 0
    while not done and n < max_steps:
        _, _, done, _, info = env.step(a)
        n += 1
    return done


def _seeded_reset(env, k):
    s = SELFTEST_SEED_BASE + k
    np.random.seed(s)
    random.seed(s)
    return env.reset()


def _park_object_far(env):
    """Move the object out of reach (arm-speed test) without starting the window."""
    pb = env._pb()
    p.resetBasePositionAndOrientation(env.object_uid, [0.9, 0.9, env.cfg["spawn_z"]],
                                      [0, 0, 0, 1], **pb)
    p.resetBaseVelocity(env.object_uid, [0, 0, 0], [0, 0, 0], **pb)
    for _ in range(100):
        p.stepSimulation(**pb)
    env._settle_until_rest()
    env._obj_start, env._yaw_start = env._obj_pose()


def _quiet(fn, *a, **kw):
    """Run fn with C-level stdout silenced (PyBullet prints b3 warnings)."""
    sys.stdout.flush()
    fd, saved = sys.stdout.fileno(), os.dup(sys.stdout.fileno())
    with open(os.devnull, "w") as dn:
        os.dup2(dn.fileno(), fd)
        try:
            return fn(*a, **kw)
        finally:
            sys.stdout.flush()
            os.dup2(saved, fd)
            os.close(saved)


def selftest(args):
    chk, out = _Checks(), {"config": {k: (list(v) if isinstance(v, tuple) else v)
                                      for k, v in TASK2_DEFAULTS.items()}}
    enc_ok = os.path.exists(args.encoder)
    modes = [m for m in args.modes if m != "experimental" or enc_ok]
    if "experimental" in args.modes and not enc_ok:
        chk("encoder file found for 'experimental'", False, args.encoder)
    print(f"\n{'='*92}\n  TASK 2 SELF-TEST  |  modes {modes}  |  {args.episodes} seeded episodes per mode\n{'='*92}")

    # ── A. Physics, code and reset state per mode ────────────────────────────
    records, timing = {}, {"reset_s": []}
    for mode in modes:
        env = _quiet(UR3ePushingTask2Env, randomization_mode=mode,
                     encoder_path=args.encoder if mode == "experimental" else None)
        print(f"\n  [A] mode = {mode}")
        if mode == "experimental":
            chk("encoder loaded (experimental)", env.encoder is not None,
                "" if env.encoder is not None else
                "env.encoder is None: dims 9-10 would silently be zero")
        print(f"    {'k':>2} {'category':17s} {'mass':>6} {'mu':>6} {'Izz x':>6} "
              f"{'code 9-10':>16} {'start (x,y)':>15} {'D':>6} {'th':>6} {'settle':>6}")
        for k in range(args.episodes):
            t0 = time.time()
            obs, _ = _quiet(_seeded_reset, env, k)
            timing["reset_s"].append(time.time() - t0)
            r = env.episode_record()
            records.setdefault(mode, []).append(r)
            cat = r["category"]
            print(f"    {k:2d} {str(cat):17s} {r['mass']:6.3f} {r['mu']:6.3f} "
                  f"{str(r['izz_scale']):>6} [{r['code'][0]:6.3f},{r['code'][1]:6.3f}] "
                  f"({r['start'][0]:.3f},{r['start'][1]:+.3f}) {r['goal_dist']:6.3f} "
                  f"{r['goal_theta_deg']:+6.1f} {r['reset']['settle_steps']:6d}")
            tag = f"{mode} k={k}"
            if mode == "baseline":
                chk(f"{tag}: no category", cat is None)
                chk(f"{tag}: baseline mass/mu bounds", 0.1 <= r["mass"] <= 5.0 and 0.1 <= r["mu"] <= 1.2)
                chk(f"{tag}: no Izz scaling", r["izz_scale"] is None)
            else:
                b = env.semantic_mapping.get(cat, {})
                chk(f"{tag}: category in mapping", cat in env.semantic_mapping, cat)
                chk(f"{tag}: mass within category", b and b["mass"][0] <= r["mass"] <= b["mass"][1])
                chk(f"{tag}: mu within category", b and b["lateral_friction"][0] <= r["mu"] <= b["lateral_friction"][1])
                chk(f"{tag}: Izz scale applied", r["izz_scale"] == CATEGORY_IZZ_SCALE.get(cat))
            code = np.array(r["code"])
            if mode in ("baseline", "coupled_no_cue"):
                chk(f"{tag}: code is zero", np.allclose(code, 0))
            elif mode == "oracle":
                chk(f"{tag}: oracle code = union-normalised truth (no clip)",
                    np.allclose(code, env._oracle_code(), atol=1e-6))
            elif mode == "experimental":
                dists = {c: np.abs(code - np.array(v)).max() for c, v in env.category_codes.items()}
                pred = min(dists, key=dists.get)
                chk(f"{tag}: code is one of the three category codes", dists[pred] < 2e-3, code)
                r["encoder_pred"], r["encoder_correct"] = pred, pred == cat
            chk(f"{tag}: object settled at reset", r["reset"]["settled"])
            chk(f"{tag}: no robot-object contact at reset", not r["reset"]["contact_at_reset"])
            chk(f"{tag}: object start in region",
                env.cfg["obj_x"][0] - 0.02 <= r["start"][0] <= env.cfg["obj_x"][1] + 0.02
                and env.cfg["obj_y"][0] - 0.02 <= r["start"][1] <= env.cfg["obj_y"][1] + 0.02, r["start"])
            chk(f"{tag}: goal at distance D", abs(np.hypot(*(np.subtract(r["goal"], r["start"]))) - r["goal_dist"]) < 1e-5)
            chk(f"{tag}: pusher at start pose", np.linalg.norm(env._ee_actual()[:2] - np.array(env.cfg["start_ee"])) < 0.01,
                env._ee_actual()[:2])
            chk(f"{tag}: obs is 13-dim with phase [0,1]",
                obs["observation"].shape == (13,) and np.allclose(obs["observation"][11:], [0, 1]))
        if mode == "experimental":
            acc = np.mean([r["encoder_correct"] for r in records[mode]])
            print(f"    encoder correct in {acc*100:.0f}% of these episodes (expected about 67%)")
        env.close()

    # ── B. Common episodes across modes ──────────────────────────────────────
    # Every mode draws the same object, spawn point, D and direction. Coupled
    # modes share the physics, so their settled start must match exactly. The
    # Baseline's other mass and friction let its object settle slightly
    # differently; that is reported and only has to stay below 1 mm.
    print("\n  [B] common episodes across modes")
    coupled = [m for m in modes if m != "baseline"]
    base_dev = []
    for k in range(args.episodes):
        ref = records[modes[0]][k]
        for m in modes[1:]:
            r = records[m][k]
            chk(f"k={k}: same object, spawn point, D and direction in {modes[0]} and {m}",
                r["object"] == ref["object"] and r["spawn"] == ref["spawn"]
                and r["goal_dist"] == ref["goal_dist"]
                and r["goal_theta_deg"] == ref["goal_theta_deg"])
        if len(coupled) > 1:
            r0 = records[coupled[0]][k]
            for m in coupled[1:]:
                r = records[m][k]
                chk(f"k={k}: same physics and settled start in {coupled[0]} and {m}",
                    r["category"] == r0["category"] and r["mass"] == r0["mass"]
                    and r["mu"] == r0["mu"] and r["inertia"] == r0["inertia"]
                    and r["start"] == r0["start"] and r["goal"] == r0["goal"])
        if "baseline" in modes and modes[0] != "baseline":
            dev = float(np.hypot(*np.subtract(records["baseline"][k]["start"], ref["start"])))
            base_dev.append(dev)
            chk(f"k={k}: Baseline settled start within 1 mm of {modes[0]}", dev < 0.001,
                f"{dev * 1000:.3f} mm")
    print(f"    compared {args.episodes} episodes across {modes}")
    if base_dev:
        print(f"    Baseline settled start differs by at most {max(base_dev) * 1000:.3f} mm "
              f"(other mass and friction); the Baseline is evaluated in coupled_no_cue mode,\n"
              f"    where every episode matches exactly")
    out["baseline_settle_dev_m"] = base_dev

    # ── C. Phase logic with a scripted push (coupled_no_cue) ─────────────────
    env = _quiet(UR3ePushingTask2Env, randomization_mode="coupled_no_cue")
    K = env.k_window
    print(f"\n  [C] phase logic, K = {K}, push 1.5 cm/step (coupled_no_cue); fail reward = {env.fail_reward:.1f}")
    phase_out, step_times, coast_times, appr_steps = [], [], [], []
    for k in range(args.episodes):
        _quiet(_seeded_reset, env, k)
        r = env.episode_record()
        appr = scripted_approach(env)
        chk(f"C k={k}: scripted approach reached pre-push point", appr["ok"], _appr_summary(appr))
        if not appr["ok"]:
            continue
        appr_steps.append(appr["steps"])
        a = push_action(env, 0.18)
        trace, done = [], False
        while not done and env._n_steps < 120:
            t0 = time.time()
            obs, rew, done, trunc, info = env.step(a)
            step_times.append(time.time() - t0)
            lin, _ = env._obj_vel()
            trace.append((env._n_steps, PHASE_NAME[env._phase], env._push_left,
                          float(np.hypot(lin[0], lin[1])), rew, obs["observation"][11:].tolist()))
            chk(f"C k={k} step {env._n_steps}: reward == compute_reward",
                abs(rew - float(env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info))) < 1e-6)
        print(f"    k={k} {r['category']:16s} mu={r['mu']:.3f} D={r['goal_dist']:.3f}  "
              f"approach {appr['steps']} steps (gap {appr['gap'] * 100:.1f} cm, "
              f"{appr['lateral_offset'] * 1000:+.1f} mm off the line), onset at step "
              f"{env._onset_step} ({env._onset_reason})")
        for st in trace[-(K + 2):]:
            print(f"      step {st[0]:3d}  phase {st[1]:8s} left {st[2]:2d}  obj speed {st[3]:.3f} m/s  "
                  f"reward {st[4]:7.2f}  obs[11:13] {np.round(st[5], 2).tolist()}")
        chk(f"C k={k}: episode terminated", done)
        if done:
            chk(f"C k={k}: K steps after onset", env._n_steps - env._onset_step == K,
                (env._onset_step, env._n_steps))
            chk(f"C k={k}: reward 0 iff success", (rew == 0.0) == bool(info["is_success"]))
            chk(f"C k={k}: terminal reward is 0 or fail reward", rew in (0.0, np.float32(env.fail_reward)))
            coast_times.append(info["coast_time"])
            print(f"      end: at rest {bool(info['at_rest'])}, coast {info['coast_time']:.2f} s, "
                  f"speed at window end {info['v_window_end']:.3f} m/s, rest error "
                  f"{info['rest_dist']*100:.1f} cm, success {bool(info['is_success'])}, reward {rew:.1f}")
            phase_out.append({"k": k, "category": r["category"], "mu": r["mu"], "D": r["goal_dist"],
                              "onset_step": env._onset_step, "onset_reason": env._onset_reason,
                              "info": info, "reward": rew})
    # K = 0 must end on the onset step
    _quiet(_seeded_reset, env, 0)
    env.set_push_window(0)
    if scripted_approach(env)["ok"]:
        done = False
        while not done and env._n_steps < 120:
            _, _, done, _, _ = env.step(push_action(env, 0.18))
        chk("C: K=0 ends on the onset step", done and env._n_steps == env._onset_step,
            (env._onset_step, env._n_steps))
    env.set_push_window(TASK2_DEFAULTS["k_window"])
    out["phase_logic"] = phase_out

    # ── D. HER consistency ────────────────────────────────────────────────────
    print("\n  [D] HER relabelling (compute_reward on batches)")
    d = env.success_threshold
    ag = np.array([[0.3, 0.0]] * 5)
    dg = np.array([[0.3, 0.0], [0.3, 0.5 * d], [0.3, 2 * d], [0.3, 0.0], [0.3, 0.0]])
    infos = np.array([{"rest_end": 0.0, "at_rest": 0.0}, {"rest_end": 1.0, "at_rest": 1.0},
                      {"rest_end": 1.0, "at_rest": 1.0}, {"rest_end": 1.0, "at_rest": 0.0},
                      {"rest_end": 1.0, "at_rest": 1.0}], dtype=object)
    got = env.compute_reward(ag, dg, infos)
    want = np.array([-1.0, 0.0, env.fail_reward, env.fail_reward, 0.0], dtype=np.float32)
    print(f"    cases: mid-episode, rest within delta, rest outside, coast timeout, rest on goal")
    print(f"    got  {np.round(got, 2).tolist()}\n    want {np.round(want, 2).tolist()}")
    chk("D: batched compute_reward matches the rule", np.allclose(got, want))
    try:
        env.compute_reward(ag, dg, np.array([{}] * 5, dtype=object))
        chk("D: empty infos (copy_info_dict=False) raise", False)
    except RuntimeError:
        chk("D: empty infos (copy_info_dict=False) raise", True)

    # ── E. Arm speed: commanded vs achieved ──────────────────────────────────
    print("\n  [E] arm tracking: commanded vs achieved EE displacement per control step (object removed)")
    speed = {}
    for direction, vec in (("+x", np.array([1.0, 0.0])), ("+y", np.array([0.0, 1.0]))):
        for cmd in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
            _quiet(_seeded_reset, env, 0)
            _park_object_far(env)
            if direction == "+y":
                move_to(env, [0.32, -0.14])
            xs = [env._ee_actual()[:2] @ vec]
            for _ in range(5):
                env.step(vec * cmd)
                xs.append(env._ee_actual()[:2] @ vec)
            dd = np.diff(xs) * 100
            speed[f"{direction} {cmd*100:.1f}"] = dd.tolist()
            print(f"    {direction} cmd {cmd*100:4.1f} cm/step -> achieved {np.round(dd, 2).tolist()} cm/step"
                  f"   (steps 2-5 mean {dd[1:].mean():.2f} = {dd[1:].mean()/100/CONTROL_DT:.2f} m/s)")
    out["arm_speed_cm_per_step"] = speed
    vmax_x = max(np.mean(v[1:]) for k_, v in speed.items() if k_.startswith("+x")) / 100 / CONTROL_DT
    print(f"    => sustained max EE speed along +x is about {vmax_x:.2f} m/s "
          f"(nominal 0.05 m / {CONTROL_DT:.4f} s = 0.60 m/s)")
    out["vmax_x_mps"] = float(vmax_x)

    # ── F. Scripted approach: measured gap and travel to first contact ──────
    print("\n  [F] scripted approach: closest-point gap to the object (target 1.5 cm), "
          "then creep 1 mm/step to first contact")
    gaps = []
    for k in range(min(3, args.episodes)):
        _quiet(_seeded_reset, env, k)
        appr = scripted_approach(env)
        summ = _appr_summary(appr)
        if not chk(f"F k={k}: approach ok", appr["ok"], summ):
            continue
        s0 = float(env._ee_actual()[:2] @ env._u_goal)
        while env._phase == APPROACH and env._n_steps < 150:
            env.step(env._u_goal * 0.001)
        travel = float(env._ee_actual()[:2] @ env._u_goal) - s0
        gaps.append({**summ, "travel_to_contact": travel, "onset": env._onset_reason})
        print(f"    k={k}: gap {appr['gap'] * 100:.2f} cm, pusher {appr['lateral_offset'] * 1000:+.1f} mm "
              f"off the push line, "
              f"first contact after {travel * 100:.2f} cm ({env._onset_reason})")
        chk(f"F k={k}: gap within 2 mm of 1.5 cm", abs(appr["gap"] - 0.015) < 0.002,
            f"{appr['gap'] * 100:.2f} cm")
        chk(f"F k={k}: pusher within 2 mm of the push line", abs(appr["lateral_offset"]) < 0.002,
            f"{appr['lateral_offset'] * 1000:+.1f} mm")
        chk(f"F k={k}: no contact before the measured gap is closed", travel > appr["gap"] - 0.0015,
            f"travel {travel * 100:.2f} cm, gap {appr['gap'] * 100:.2f} cm")
    out["approach"] = gaps

    # ── G. Free coast vs v^2 / (2 mu g) ──────────────────────────────────────
    print("\n  [G] free coast after a 0.15 m/s kick, vs v^2/(2 mu g)")
    coast, per_cat, k = [], {}, 0
    while min([per_cat.get(c, 0) for c in env.semantic_mapping] or [0]) < 2 and k < 60:
        _quiet(_seeded_reset, env, 100 + k)
        k += 1
        cat = env._current_category
        if per_cat.get(cat, 0) >= 2:
            continue
        per_cat[cat] = per_cat.get(cat, 0) + 1
        v0 = 0.15
        p.resetBaseVelocity(env.object_uid, [v0 * env._u_goal[0], v0 * env._u_goal[1], 0], [0, 0, 0], **env._pb())
        t0 = time.time()
        res = env._stop_and_coast()
        coast_times.append(time.time() - t0)
        pos, _ = env._obj_pose()
        dist = float(np.linalg.norm(pos[:2] - env._obj_start[:2]))
        theory = v0 ** 2 / (2 * env._last_friction * 9.81)
        coast.append({"category": cat, "mu": env._last_friction, "measured": dist, "theory": theory,
                      "at_rest": res["at_rest"], "object": env.current_object_name})
        print(f"    {cat:16s} {env.current_object_name[:24]:24s} mu={env._last_friction:.3f}  "
              f"measured {dist*100:6.2f} cm  theory {theory*100:6.2f} cm  ratio {dist/theory:4.2f}  "
              f"rest {res['at_rest']} after {res['coast_time']:.2f} s")
        if cat == "Wet Ice":
            chk(f"G: ice coast within 30% of theory ({env.current_object_name})", 0.7 < dist / theory < 1.3)
    out["free_coast"] = coast

    # ── H. Determinism of seeded replay ──────────────────────────────────────
    # Script 21 evaluates every (speed, steps) branch by replaying the episode
    # from a seeded reset; two replays with the same actions must be identical.
    # For reference only, the same run restored from a PyBullet snapshot is
    # also compared: saveState/restoreState does not reproduce a fresh run
    # exactly once objects touch, which is why script 21 does not use it.
    print("\n  [H] determinism: two seeded replays with the same actions "
          "(snapshot restore shown for reference)")
    ks = list(range(min(2, args.episodes)))
    for j in range(80):
        _quiet(_seeded_reset, env, 100 + j)
        if env._current_category == "Wet Ice":
            ks.append(100 + j)
            break
    det = []
    for k in ks:
        _quiet(_seeded_reset, env, k)
        cat, mu = env._current_category, env._last_friction
        appr = scripted_approach(env)
        if not chk(f"H k={k}: approach ok", appr["ok"], _appr_summary(appr)):
            continue
        a = push_action(env, 0.18)
        _run_to_end(env, a)
        rest1 = env._obj_pose()[0][:2]
        _quiet(_seeded_reset, env, k)
        for aa in appr["actions"]:
            env.step(np.array(aa))
        _run_to_end(env, a)
        rest2 = env._obj_pose()[0][:2]
        _quiet(_seeded_reset, env, k)
        for aa in appr["actions"]:
            env.step(np.array(aa))
        sid = p.saveState(**env._pb())
        py = (list(env.current_ee_pos), env._phase, env._push_left, env._n_steps,
              env._onset_step, env._onset_reason)
        p.restoreState(stateId=sid, **env._pb())
        (env.current_ee_pos, env._phase, env._push_left, env._n_steps,
         env._onset_step, env._onset_reason) = (list(py[0]),) + py[1:]
        _run_to_end(env, a)
        rest3 = env._obj_pose()[0][:2]
        p.removeState(sid, **env._pb())
        d_rep = float(np.linalg.norm(rest1 - rest2))
        d_res = float(np.linalg.norm(rest1 - rest3))
        slide = float(np.linalg.norm(rest1 - env._obj_start[:2]))
        det.append({"k": k, "category": cat, "mu": mu, "slide_m": slide,
                    "replay_diff_m": d_rep, "restore_diff_m": d_res})
        print(f"    k={k} {cat:16s} mu={mu:.3f} slide {slide * 100:5.1f} cm | replay vs replay "
              f"{d_rep * 1000:.4f} mm | snapshot restore vs fresh run {d_res * 1000:.3f} mm (reference)")
        chk(f"H k={k}: seeded replay reproduces the run exactly", d_rep == 0.0, f"{d_rep * 1000:.6f} mm")
    out["determinism"] = det

    # ── I. Timing ─────────────────────────────────────────────────────────────
    t_reset = float(np.mean(timing["reset_s"]))
    t_step = float(np.mean(step_times)) if step_times else float("nan")
    t_coast = float(np.mean(coast_times)) if coast_times else float("nan")
    n_appr = float(np.mean(appr_steps)) if appr_steps else 15.0
    leaf = t_reset + (n_appr + 3.5) * t_step + t_coast
    out["timing"] = {"reset_s": t_reset, "step_s": t_step, "coast_s": t_coast,
                     "approach_steps": n_appr, "rollout_s": leaf,
                     "pilot_episode_s_estimate": 118 * leaf}
    print(f"\n  [I] timing: reset {t_reset:.3f} s, control step {t_step * 1000:.1f} ms, "
          f"coast {t_coast:.3f} s, approach {n_appr:.0f} steps"
          f"\n      => one replayed rollout about {leaf:.2f} s; script 21 makes about 118 per episode, "
          f"about {118 * leaf:.0f} s per episode per worker\n"
          f"         (its DRY_RUN times a real episode)")
    env.close()

    n_ok, bad = chk.summary()
    out["checks"], out["records"] = chk.items, records
    with open(SELFTEST_FILE, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\n{'#'*92}\n  {n_ok} checks passed, {len(bad)} failed.  Saved -> {SELFTEST_FILE}\n{'#'*92}")
    for b in bad:
        print(f"  FAILED: {b['check']}  {b['detail']}")
    return 0 if not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--modes", nargs="+", default=["coupled_no_cue", "oracle", "experimental", "baseline"])
    ap.add_argument("--encoder", default=SELFTEST_ENCODER_PATH)
    a = ap.parse_args()
    if not a.selftest:
        ap.print_help()
        sys.exit(0)
    sys.exit(selftest(a))