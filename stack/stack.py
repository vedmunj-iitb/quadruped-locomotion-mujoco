"""The whole stack in one controller, run at every 2 ms physics step:

  1. gait scheduler: which feet are down now and over the MPC horizon (trot, 0.36 s cycle)
  2. foothold planner: at each lift-off, the Raibert point is moved to the cheapest safe spot on
     the heightmap, and a Bezier swing curve is planned to it
  3. convex MPC: ground forces for the stance feet (33 Hz, and at every touchdown and lift-off)
  4. whole body QP: the MPC forces and the swing foot accelerations become joint torques

Written in 2026.
"""
import mujoco
import numpy as np

from common import go2
from footholds.planner import select_foothold
from gait.scheduler import bezier, bezier_acceleration, bezier_velocity, swing_curve
from mpc.controller import STAND, TROT, MPCController, contact_forces
from wbc.wbc import WholeBodyQP


class StackController(MPCController):
    def __init__(self, model, terrain=None, select=True, step_height=0.08, **kw):
        self.terrain = terrain             # footholds.terrain.Terrain, or None for flat ground
        self.select = select               # False: land on the Raibert point, whatever is there
        super().__init__(model, step_height=step_height, **kw)
        # MPCController measured the standing centre of mass height after a reset, which here
        # already stands on the course's ground. torques() adds the ground height, so take it off.
        self.com_height -= self.height(0.0, 0.0)
        self.wbc = WholeBodyQP(model)
        self.ground = np.zeros(4)          # terrain height under each planted foot (m)
        self.curves = [None] * 4           # Bezier control points of each leg's current swing
        self.steps = []                    # (time, leg, Raibert point, chosen foothold) of every step
        self.wbc_times = []

    def height(self, x, y):
        return 0.0 if self.terrain is None else float(self.terrain.height(x, y))

    def reset(self, data, x=0.0):
        """Home pose on the ground under x (the course starts at x = 0)."""
        super().reset(data)
        data.qpos[0] = x
        data.qpos[2] += self.height(x, 0.0)
        mujoco.mj_forward(self.model, data)
        feet = data.geom_xpos[self.feet_geoms]
        self.ground = np.array([self.height(f[0], f[1]) for f in feet])

    def plan_step(self, leg, data, v, t_swing):
        """At lift-off: Raibert point (as in the MPC stage), then the planner picks the foothold."""
        nominal = self.touchdown_point(leg, data, v, t_swing)[:2]
        if self.terrain is None:
            chosen = np.array([nominal[0], nominal[1], 0.0])
        elif not self.select:
            chosen = np.array([nominal[0], nominal[1], self.height(*nominal)])
        else:
            # Where this leg's hip will be when the foot lands, for the planner's reach checks.
            heading = go2.rot_z(self.yaw + self.cmd[2] * t_swing)
            hip = data.qpos[:3] + v * t_swing + heading @ (self.home[leg] * [1, 1, 0])
            hip[2] = self.ground.mean() + 0.30
            chosen = select_foothold(self.terrain, nominal, hip)
        self.steps.append((data.time, leg, nominal, chosen))
        return np.array([chosen[0], chosen[1], chosen[2] + go2.FOOT_RADIUS])

    def torques(self, data, dt):
        model = self.model
        step = self.accel * np.array([1.0, 1.0, 2.0]) * dt     # rate limit of the commands
        self.cmd += np.clip(self.target - self.cmd, -step, step)
        self.yaw_des += self.cmd[2] * dt
        self.update_gait(dt)
        rpy, com, omega, v = self.measure(data)
        feet = data.geom_xpos[self.feet_geoms].copy()
        t_swing = (1.0 - self.sched.gait.duty) * self.sched.gait.period
        stance = [self.sched.in_stance(leg) for leg in range(4)]

        # 1-2. Gait and footholds: plan each swing once, when its foot lifts off.
        for leg in range(4):
            if not stance[leg] and self.was_stance[leg]:
                self.lift_off[leg] = feet[leg]
                self.touchdown[leg] = self.plan_step(leg, data, v * [1, 1, 0], t_swing)
                self.curves[leg] = swing_curve(self.lift_off[leg], self.touchdown[leg],
                                               self.step_height, np.zeros(3), t_swing)
            if stance[leg] and not self.was_stance[leg]:            # touchdown
                self.ground[leg] = self.touchdown[leg, 2] - go2.FOOT_RADIUS

        # 3. MPC, with the body held com_height above the mean ground under the feet.
        self.since_solve += 1
        if stance != self.was_stance or self.since_solve >= self.every:
            self.since_solve = 0
            self.forces, prediction = self.mpc.solve(
                rpy, com, omega, v, np.where(np.array(stance)[:, None], feet, self.touchdown),
                self.sched.contact_plan(self.mpc.N, self.mpc.dt), self.velocity_command(),
                self.cmd[2], self.com_height + self.ground.mean(), yaw_ref=self.yaw_des)
            self.solve_times.append(self.mpc.info.run_time)   # OSQP update, solve and polish
            if prediction is not None:
                self.prediction = prediction

        # 4. Swing targets from the curves, then joint torques.
        swing = {}
        for leg in range(4):
            if not stance[leg]:
                s = self.sched.progress(leg)
                c = self.curves[leg]
                swing[leg] = (bezier(c, s), bezier_velocity(c, s, t_swing),
                              bezier_acceleration(c, s, t_swing))
        tau = self.wbc.torques(data, stance, f_des=self.forces, swing=swing)
        self.wbc_times.append(self.wbc.solve_time)
        self.was_stance = stance
        return np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def run_stack(schedule, seconds, terrain=None, frame_hook=None, model=None, data=None, **kw):
    """Run the stack for `seconds`. schedule(t) gives (vx, vy, yaw_rate, gait).

    Returns the log (arrays) and the controller. frame_hook(data, controller) is called after
    every physics step (for video).
    """
    if model is None:
        model = go2.load_model(None if terrain is None else terrain.add_to_spec)
    data = mujoco.MjData(model) if data is None else data
    ctrl = StackController(model, terrain=terrain, **kw)
    ctrl.reset(data)
    dt = model.opt.timestep
    keys = ("t", "pos", "com", "rpy", "target", "cmd", "vel", "yaw_rate", "forces",
            "contact_forces", "plan", "contact", "tau", "feet")
    log = {k: [] for k in keys}
    for _ in range(int(round(seconds / dt))):
        ctrl.command(*schedule(data.time))
        data.ctrl[:] = ctrl.torques(data, dt)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        R = data.xmat[1].reshape(3, 3)
        heading = go2.rot_z(go2.body_rpy(data)[2])
        log["t"].append(data.time)
        log["pos"].append(data.qpos[:3].copy())
        log["com"].append(data.subtree_com[1].copy())
        log["rpy"].append(go2.body_rpy(data))
        log["target"].append(ctrl.target.copy())
        log["cmd"].append(ctrl.cmd.copy())
        log["vel"].append(heading.T @ data.qvel[:3])
        log["yaw_rate"].append((R @ data.qvel[3:6])[2])
        log["forces"].append(ctrl.forces.copy())
        log["contact_forces"].append(contact_forces(model, data, ctrl.feet_geoms))
        log["plan"].append([ctrl.sched.in_stance(leg) for leg in range(4)])
        log["contact"].append(go2.feet_in_contact(model, data, ctrl.feet_geoms))
        log["tau"].append(data.ctrl.copy())
        log["feet"].append(data.geom_xpos[ctrl.feet_geoms].copy())
        if frame_hook is not None:
            frame_hook(data, ctrl)
    out = {k: np.array(v) for k, v in log.items()}
    out["solve_times"] = np.array(ctrl.solve_times)
    out["wbc_times"] = np.array(ctrl.wbc_times)
    return out, ctrl


__all__ = ["StackController", "run_stack", "STAND", "TROT"]
