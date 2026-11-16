"""Gait stage controller: foot targets from the gait, then leg IK, then joint PD torques.

This is the simplest way to drive the gait: no force planning. The MPC and WBC stages replace
the joint PD with planned forces. Written in 2026.
"""
import mujoco
import numpy as np

from common import go2
from gait.scheduler import GAITS, GaitScheduler, bezier, swing_curve


class GaitController:
    def __init__(self, model, gait, speed=0.0, turn=0.0, height=0.30, step_height=0.08,
                 kp=80.0, kd=2.0, k_yaw=2.0, sway=0.0, ramp=1.0):
        self.model = model
        self.sched = GaitScheduler(GAITS[gait] if isinstance(gait, str) else gait)
        self.speed, self.turn = speed, turn   # forward speed (m/s) and turn rate (rad/s) to reach
        self.step_height = step_height        # top of the swing above the ground (m)
        self.kp, self.kd = kp, kd             # joint PD gains (N m/rad, N m s/rad)
        self.k_yaw = k_yaw                    # heading hold: extra turn rate per rad of heading error
        self.sway = sway                      # how far the body leans towards the planted feet (0 to 1)
        self.ramp = ramp                      # seconds to reach the commanded speed from rest
        self.mass = model.body_subtreemass[1]
        self.foot_geoms = go2.foot_geoms(model)
        # Home: each foot under its thigh joint, ball centre (height - radius) below the body.
        self.home = go2.HIP + np.outer(go2.SIDE, [0.0, go2.L_HIP, 0.0])
        self.home[:, 2] = -(height - go2.FOOT_RADIUS)
        self.feet = self.home.copy()          # foot targets in the body frame (m)
        self.swing = [None] * 4               # control points of each leg's current swing
        self.was_stance = [True] * 4
        self.shift = np.zeros(3)              # current sideways lean of the body (m)
        self.yaw_ref = 0.0                    # heading the robot should have now (rad)
        self.q_des = None

    def reset(self, data):
        """Put the robot on the floor in the home pose, all feet down."""
        mujoco.mj_resetData(self.model, data)
        data.qpos[2] = -self.home[0, 2] + go2.FOOT_RADIUS
        data.qpos[3] = 1.0
        for leg in range(4):
            data.qpos[7 + 3 * leg: 10 + 3 * leg] = go2.leg_ik(leg, self.home[leg])
        mujoco.mj_forward(self.model, data)

    def touchdown_point(self, leg, v, w):
        """Half a stance stride ahead of home, so the stance is centred under the hip."""
        t_stance = self.sched.gait.duty * self.sched.gait.period
        p = go2.rot_z(w[2] * t_stance / 2) @ self.home[leg] + v * t_stance / 2
        p[2] = self.home[leg, 2]
        return p

    def update_feet(self, v, w, dt):
        """Advance the gait clock by dt and move every foot target along its stance or swing."""
        self.sched.advance(dt)
        t_swing = (1.0 - self.sched.gait.duty) * self.sched.gait.period
        for leg in range(4):
            stance = self.sched.in_stance(leg)
            planted = -(v + np.cross(w, self.feet[leg]))   # how a planted foot moves past the body
            if stance:
                self.feet[leg] += planted * dt
                self.feet[leg, 2] = self.home[leg, 2]
            else:
                if self.was_stance[leg]:                     # lift-off: plan this swing
                    self.swing[leg] = swing_curve(self.feet[leg], self.touchdown_point(leg, v, w),
                                                  self.step_height, planted, t_swing)
                self.feet[leg] = bezier(self.swing[leg], self.sched.progress(leg))
            self.was_stance[leg] = stance

    def torques(self, data, dt):
        """Joint torques for this control step, in the model's actuator order."""
        # Commands, raised from zero over the first `ramp` seconds.
        k = min(1.0, self.sched.t / self.ramp) if self.ramp > 0 else 1.0
        self.yaw_ref += k * self.turn * dt
        yaw_error = (self.yaw_ref - go2.body_rpy(data)[2] + np.pi) % (2 * np.pi) - np.pi
        v = np.array([k * self.speed, 0.0, 0.0])
        w = np.array([0.0, 0.0, k * self.turn + self.k_yaw * yaw_error])
        self.update_feet(v, w, dt)

        # Sway (walk): lean sideways towards the feet that will be planted 0.15 s from now, so the
        # centre of mass is over their triangle before the next leg lifts.
        soon = [leg for leg in range(4) if self.sched.in_stance(leg, self.sched.t + 0.15)]
        lean = np.zeros(3)
        if self.sway and soon:                                 # no lean without planted feet
            lean[1] = self.sway * self.home[soon, 1].mean()
        self.shift += (lean - self.shift) * dt / 0.1          # first-order filter, 0.1 s
        self.targets = self.feet - self.shift                  # moving the feet moves the body

        # Leg IK, then PD on the joints.
        q_des = np.concatenate([go2.leg_ik(leg, self.targets[leg]) for leg in range(4)])
        qd_des = np.zeros(12) if self.q_des is None else (q_des - self.q_des) / dt
        self.q_des = q_des
        tau = self.kp * (q_des - data.qpos[7:]) + self.kd * (qd_des - data.qvel[6:])

        # Stance legs also carry the body weight, shared equally: tau = J^T f, where f is the
        # downward push of the foot on the ground.
        stance = [leg for leg in range(4) if self.sched.in_stance(leg)]
        jac = np.zeros((3, self.model.nv))
        for leg in stance:
            mujoco.mj_jacGeom(self.model, data, jac, None, self.foot_geoms[leg])
            cols = slice(6 + 3 * leg, 9 + 3 * leg)   # this leg's joints among the 18 velocities
            push = np.array([0.0, 0.0, -self.mass * 9.81 / len(stance)])
            tau[3 * leg: 3 * leg + 3] += jac[:, cols].T @ push
        return np.clip(tau, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])


def simulate(gait, seconds, speed=0.0, turn=0.0, frame_hook=None, model=None, data=None, **kw):
    """Run a gait in MuJoCo for `seconds` and return the logged signals as arrays.

    frame_hook(data), if given, is called after every physics step (to record video or update
    the viewer).
    """
    model = go2.load_model() if model is None else model
    data = mujoco.MjData(model) if data is None else data
    ctrl = GaitController(model, gait, speed=speed, turn=turn, **kw)
    ctrl.reset(data)
    dt = model.opt.timestep          # 2 ms: the controller runs at every physics step, 500 Hz
    log = {k: [] for k in ("t", "pos", "rpy", "vel", "feet", "feet_body", "target", "contact",
                           "plan", "tau")}
    for _ in range(int(round(seconds / dt))):
        data.ctrl[:] = ctrl.torques(data, dt)
        mujoco.mj_step(model, data)
        mujoco.mj_fwdPosition(model, data)    # positions and contacts of the new state, not the old
        R = data.xmat[1].reshape(3, 3)
        feet = data.geom_xpos[ctrl.foot_geoms].copy()
        log["t"].append(data.time)
        log["pos"].append(data.qpos[:3].copy())
        log["rpy"].append(go2.body_rpy(data))
        log["vel"].append(R.T @ data.qvel[:3])                 # body frame
        log["feet"].append(feet)                               # world frame
        log["feet_body"].append((feet - data.xpos[1]) @ R)     # body frame
        log["target"].append(ctrl.targets.copy())              # body frame
        log["contact"].append(go2.feet_in_contact(model, data, ctrl.foot_geoms))
        log["plan"].append([ctrl.sched.in_stance(leg) for leg in range(4)])
        log["tau"].append(data.ctrl.copy())
        if frame_hook is not None:
            frame_hook(data)
    return {k: np.array(v) for k, v in log.items()}


def steady_state(log, after=4.0):
    """Mean forward speed (body frame, m/s) and turn rate (rad/s) after the start-up."""
    sel = log["t"] >= after
    yaw = np.unwrap(log["rpy"][sel, 2])
    return log["vel"][sel, 0].mean(), (yaw[-1] - yaw[0]) / (log["t"][sel][-1] - log["t"][sel][0])
