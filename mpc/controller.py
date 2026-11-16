"""MPC stage controller: convex MPC forces for the stance feet (tau = -J^T f), Bezier swing
with Cartesian PD. The full stack (stack/) replaces this leg control with the WBC.

Written in 2026.
"""
import mujoco
import numpy as np

from common import go2
from gait.scheduler import Gait, GaitScheduler, bezier, bezier_velocity, swing_curve
from mpc.mpc import ConvexMPC

TROT = Gait("trot", 0.36, 0.5, (0.0, 0.5, 0.5, 0.0))     # diagonal pairs, 12 MPC steps a cycle
STAND = Gait("stand", 1.0, 1.0, (0.0, 0.0, 0.0, 0.0))    # every foot always on the ground


class MPCController:
    def __init__(self, model, height=0.30, step_height=0.06, horizon=12, dt_mpc=0.03,
                 kp_swing=4000.0, kd_swing=70.0, k_raibert=0.15, mu=0.5, f_max=180.0, accel=1.0):
        self.model = model
        self.feet_geoms = go2.foot_geoms(model)
        self.step_height = step_height                     # top of the swing above the floor (m)
        self.kp_swing, self.kd_swing = kp_swing, kd_swing  # swing foot PD (N/m, N s/m)
        self.k_raibert = k_raibert          # extra step length per m/s of speed error (s)
        self.accel = accel                  # vx and vy change by at most this per second, the yaw rate by twice it
        self.every = int(round(dt_mpc / model.opt.timestep))   # physics steps per MPC solve
        # Home: each foot under its thigh joint, ball centre (height - radius) below the body.
        self.home = go2.HIP + np.outer(go2.SIDE, [0.0, go2.L_HIP, 0.0])
        self.home[:, 2] = -(height - go2.FOOT_RADIUS)
        # Mass, inertia and centre of mass height in the home pose, from the model.
        data = mujoco.MjData(model)
        self.reset(data)
        self.com_height = data.subtree_com[1, 2]
        self.mpc = ConvexMPC.from_model(model, data, horizon=horizon, dt=dt_mpc, mu=mu,
                                        f_max=f_max)
        # Run state.
        self.sched = GaitScheduler(STAND)
        self.next_gait = STAND
        self.target = np.zeros(3)           # commanded (vx, vy, yaw rate), heading frame
        self.cmd = np.zeros(3)              # the same after the rate limit
        self.forces = np.zeros((4, 3))      # MPC force of the ground on each foot, world frame
        self.prediction = None              # MPC's predicted states over the horizon
        self.lift_off = np.zeros((4, 3))    # where each swing started (world frame)
        self.touchdown = np.zeros((4, 3))   # where each swing will end (world frame)
        self.was_stance = [True] * 4
        self.yaw = 0.0                      # measured heading (rad)
        self.yaw_des = 0.0                  # heading the robot should have now (rad)
        self.since_solve = self.every       # so the first step solves
        self.solve_times = []

    def reset(self, data):
        """Put the robot on the floor in the home pose, all feet down."""
        mujoco.mj_resetData(self.model, data)
        data.qpos[2] = -self.home[0, 2] + go2.FOOT_RADIUS
        data.qpos[3] = 1.0
        for leg in range(4):
            data.qpos[7 + 3 * leg: 10 + 3 * leg] = go2.leg_ik(leg, self.home[leg])
        mujoco.mj_forward(self.model, data)

    def command(self, vx=0.0, vy=0.0, yaw_rate=0.0, gait=TROT):
        """Set the velocity to reach (m/s forward and left, rad/s) and the gait (TROT or STAND)."""
        self.target = np.array([vx, vy, yaw_rate])
        self.next_gait = gait

    def measure(self, data):
        """Body state from the simulator (no estimator): roll, pitch, yaw, centre of mass
        position, world angular velocity and centre of mass velocity."""
        R = data.xmat[1].reshape(3, 3)
        rpy = go2.body_rpy(data)
        self.yaw = rpy[2]
        mujoco.mj_subtreeVel(self.model, data)
        return rpy, data.subtree_com[1].copy(), R @ data.qvel[3:6], data.subtree_linvel[1].copy()

    def velocity_command(self):
        """Commanded velocity in the world frame."""
        return go2.rot_z(self.yaw) @ np.array([self.cmd[0], self.cmd[1], 0.0])

    def touchdown_point(self, leg, data, v, t_left):
        """Raibert foothold: where the hip will be at touchdown, plus half a stance stride, plus
        k (v - v_cmd), which steps further out when the body is faster than commanded."""
        t_stance = self.sched.gait.duty * self.sched.gait.period
        heading = self.yaw + self.cmd[2] * (t_left + t_stance / 2)
        p = data.qpos[:3] + go2.rot_z(heading) @ self.home[leg] + v * (t_left + t_stance / 2) \
            + self.k_raibert * (v - self.velocity_command())
        p[2] = go2.FOOT_RADIUS
        return p

    def update_gait(self, dt):
        """Advance the gait clock. A change from trot to stand waits for the end of a half
        cycle, when the swinging feet have just landed."""
        g = self.sched.gait
        if self.next_gait is not g and (g is STAND or self.sched.t % (g.period / 2) < dt):
            self.sched = GaitScheduler(self.next_gait)
        self.sched.advance(dt)

    def torques(self, data, dt):
        """Joint torques for this physics step, in the model's actuator order."""
        model = self.model
        step = self.accel * np.array([1.0, 1.0, 2.0]) * dt     # rate limit of the commands
        self.cmd += np.clip(self.target - self.cmd, -step, step)
        self.yaw_des += self.cmd[2] * dt
        self.update_gait(dt)
        rpy, com, omega, v = self.measure(data)
        feet = data.geom_xpos[self.feet_geoms].copy()
        t_swing = (1.0 - self.sched.gait.duty) * self.sched.gait.period
        stance = [self.sched.in_stance(leg) for leg in range(4)]
        for leg in range(4):
            if not stance[leg]:
                if self.was_stance[leg]:                 # lift-off: remember where the foot was
                    self.lift_off[leg] = feet[leg]
                t_left = (1.0 - self.sched.progress(leg)) * t_swing
                self.touchdown[leg] = self.touchdown_point(leg, data, v * [1, 1, 0], t_left)

        # MPC every dt_mpc (33 Hz), and at once when a foot lands or lifts, so new stance feet
        # get their forces straight away. A foot in the air pushes from where it will land.
        self.since_solve += 1
        if stance != self.was_stance or self.since_solve >= self.every:
            self.since_solve = 0
            self.forces, prediction = self.mpc.solve(
                rpy, com, omega, v, np.where(np.array(stance)[:, None], feet, self.touchdown),
                self.sched.contact_plan(self.mpc.N, self.mpc.dt), self.velocity_command(),
                self.cmd[2], self.com_height, yaw_ref=self.yaw_des)
            self.solve_times.append(self.mpc.info.run_time)   # OSQP update, solve and polish
            if prediction is not None:                   # else the MPC kept its last forces
                self.prediction = prediction

        tau = np.zeros(12)
        jac = np.zeros((3, model.nv))
        for leg in range(4):
            mujoco.mj_jacGeom(model, data, jac, None, self.feet_geoms[leg])
            J = jac[:, 6 + 3 * leg: 9 + 3 * leg]         # this leg's three joints
            if stance[leg]:
                # The foot pushes the ground with -f, so the ground pushes the robot with f.
                tau_leg = -J.T @ self.forces[leg]
            else:
                curve = swing_curve(self.lift_off[leg], self.touchdown[leg], self.step_height,
                                    np.zeros(3), t_swing)
                s = self.sched.progress(leg)
                p_des, v_des = bezier(curve, s), bezier_velocity(curve, s, t_swing)
                f = self.kp_swing * (p_des - feet[leg]) + self.kd_swing * (v_des - jac @ data.qvel)
                tau_leg = J.T @ f
            # Plus the torques for the leg's own weight and Coriolis forces (MuJoCo's bias force),
            # minus the model's passive joint damping (2 N m s/rad), which would brake the legs.
            dofs = slice(6 + 3 * leg, 9 + 3 * leg)
            tau[3 * leg: 3 * leg + 3] = tau_leg + data.qfrc_bias[dofs] - data.qfrc_passive[dofs]
            self.was_stance[leg] = stance[leg]
        return np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def contact_forces(model, data, geoms):
    """Force of the ground on each foot from MuJoCo's contacts (world frame)."""
    out = np.zeros((4, 3))
    f6 = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        for leg, g in enumerate(geoms):
            if g in (c.geom1, c.geom2):
                mujoco.mj_contactForce(model, data, i, f6)
                f = c.frame.reshape(3, 3).T @ f6[:3]       # contact frame to world
                out[leg] += f if c.geom2 == g else -f      # the normal points from geom1 to geom2
    return out


def simulate(schedule, seconds, frame_hook=None, model=None, data=None, **kw):
    """Run the controller for `seconds`. schedule(t) gives (vx, vy, yaw_rate, gait).

    Returns the logged signals as arrays. frame_hook(data, controller), if given, is called after
    every physics step (for video).
    """
    model = go2.load_model() if model is None else model
    data = mujoco.MjData(model) if data is None else data
    ctrl = MPCController(model, **kw)
    ctrl.reset(data)
    dt = model.opt.timestep          # 2 ms: the legs are controlled at every physics step
    keys = ("t", "pos", "com", "rpy", "target", "cmd", "vel", "yaw_rate", "forces",
            "contact_forces", "plan", "tau")
    log = {k: [] for k in keys}
    for _ in range(int(round(seconds / dt))):
        ctrl.command(*schedule(data.time))
        data.ctrl[:] = ctrl.torques(data, dt)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)    # positions, contacts and forces for the new state
        R = data.xmat[1].reshape(3, 3)
        heading = go2.rot_z(go2.body_rpy(data)[2])
        log["t"].append(data.time)
        log["pos"].append(data.qpos[:3].copy())
        log["com"].append(data.subtree_com[1].copy())
        log["rpy"].append(go2.body_rpy(data))
        log["target"].append(ctrl.target.copy())
        log["cmd"].append(ctrl.cmd.copy())
        log["vel"].append(heading.T @ data.qvel[:3])            # heading frame: forward, left
        log["yaw_rate"].append((R @ data.qvel[3:6])[2])
        log["forces"].append(ctrl.forces.copy())
        log["contact_forces"].append(contact_forces(model, data, ctrl.feet_geoms))
        log["plan"].append([ctrl.sched.in_stance(leg) for leg in range(4)])
        log["tau"].append(data.ctrl.copy())
        if frame_hook is not None:
            frame_hook(data, ctrl)
    out = {k: np.array(v) for k, v in log.items()}
    out["solve_times"] = np.array(ctrl.solve_times)
    return out
