"""A basic whole body controller (WBC) for the Go2: one QP per control step, solved with OSQP.

Written in 2026.

The unknowns are x = [qdd, f]: the 18 generalised accelerations (6 for the body, 12 for the
joints) and the ground force on each foot (3 x 4 = 12, world frame). Any foot can be a stance
foot (it stays put and pushes, inside its friction cone) or a swing foot (no force, and it
follows a target path). The joint torques follow from the joint rows of the equations of
motion.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation

from common import go2

NV, NF = 18, 12                    # generalised velocities, foot force components
NX = NV + NF                       # unknowns of the QP
NC = 6 + NF + 16 + 4 + 12          # constraint rows: body dynamics, feet, friction, normal, torque


@dataclass
class Target:
    """Where the body should be (world frame) and how it should move there."""
    pos: np.ndarray                                                  # body origin (m)
    rot: np.ndarray                                                  # orientation (3 x 3)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))     # m/s
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))     # m/s^2
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))   # rad/s
    alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))   # rad/s^2


def rpy_matrix(roll, pitch, yaw):
    return go2.rot_z(yaw) @ go2.rot_y(pitch) @ go2.rot_x(roll)


def target_at(path, t, h=1e-3):
    """Target from path(t) -> (position, (roll, pitch, yaw)). Velocities and accelerations come
    from central differences with step h."""
    def pos(s):
        return np.asarray(path(s)[0], float)

    def rot(s):
        return rpy_matrix(*path(s)[1])

    def omega(s):   # world frame angular velocity: rot(s + h) = exp(2h [omega]) rot(s - h)
        return Rotation.from_matrix(rot(s + h) @ rot(s - h).T).as_rotvec() / (2 * h)

    return Target(pos=pos(t), rot=rot(t), vel=(pos(t + h) - pos(t - h)) / (2 * h),
                  acc=(pos(t + h) - 2 * pos(t) + pos(t - h)) / h**2,
                  omega=omega(t), alpha=(omega(t + h) - omega(t - h)) / (2 * h))


def stand(model, data, height=0.30):
    """Put the robot on the floor, body level at `height`, each foot under its thigh joint."""
    mujoco.mj_resetData(model, data)
    feet = go2.HIP + np.outer(go2.SIDE, [0.0, go2.L_HIP, 0.0])
    feet[:, 2] = -(height - go2.FOOT_RADIUS)
    data.qpos[2] = height
    data.qpos[3] = 1.0
    for leg in range(4):
        data.qpos[7 + 3 * leg: 10 + 3 * leg] = go2.leg_ik(leg, feet[leg])
    mujoco.mj_forward(model, data)


def csc_values(dense, pattern):
    """The entries of `dense` in the storage order of the sparse (CSC) matrix `pattern`."""
    cols = np.repeat(np.arange(pattern.shape[1]), np.diff(pattern.indptr))
    return dense[pattern.indices, cols]


class WholeBodyQP:
    """Gains are for accelerations: kp in 1/s^2, kd in 1/s, ki in 1/s^3. Weights are relative,
    per (m/s^2)^2 or (rad/s^2)^2 for accelerations and per N^2 for forces."""

    def __init__(self, model, mu=0.5, fz_min=10.0, fz_max=250.0,
                 kp_pos=400.0, kd_pos=40.0, kp_rot=400.0, kd_rot=40.0, ki=400.0, i_max=0.005,
                 kp_swing=400.0, kd_swing=40.0, kp_joint=50.0, kd_joint=10.0, k_foot=50.0,
                 w_pos=1.0, w_height=10.0, w_rot=1.0, w_swing=10.0, w_joint=1e-3,
                 w_force=1e-5, w_force_track=1e-2, base_scale=0.1, friction_comp=True):
        self.model = model
        self.mu, self.fz_min, self.fz_max = mu, fz_min, fz_max
        self.kp_pos, self.kd_pos, self.kp_rot, self.kd_rot = kp_pos, kd_pos, kp_rot, kd_rot
        self.ki, self.i_max = ki, i_max          # integral gain, and the limit of each integral
        self.kp_swing, self.kd_swing = kp_swing, kd_swing
        self.kp_joint, self.kd_joint, self.k_foot = kp_joint, kd_joint, k_foot
        self.w_base = np.array([w_pos, w_pos, w_height, w_rot, w_rot, w_rot])
        self.w_swing, self.w_joint = w_swing, w_joint
        self.w_force, self.w_force_track = w_force, w_force_track
        self.base_scale = base_scale             # base task weights times this when f_des is given
        self.friction_comp = friction_comp
        self.dt = model.opt.timestep
        self.int_pos, self.int_rot = np.zeros(3), np.zeros(3)   # integrated errors
        self.foot_geoms = go2.foot_geoms(model)
        self.calves = [model.geom_bodyid[g] for g in self.foot_geoms]
        self.tau_lo, self.tau_hi = model.actuator_ctrlrange.T

        # Rows that never change. Friction pyramid |fx|, |fy| <= (mu / sqrt 2) fz, which lies
        # inside the friction cone |f_sideways| <= mu fz. The bounds on fz are set per step.
        k = mu / np.sqrt(2)
        self.friction = np.zeros((16, NF))
        for i in range(4):
            x, y, z = 3 * i, 3 * i + 1, 3 * i + 2
            self.friction[4 * i + 0, [x, z]] = [1.0, -k]
            self.friction[4 * i + 1, [x, z]] = [-1.0, -k]
            self.friction[4 * i + 2, [y, z]] = [1.0, -k]
            self.friction[4 * i + 3, [y, z]] = [-1.0, -k]
        self.normal = np.zeros((4, NF))
        self.normal[range(4), [2, 5, 8, 11]] = 1.0

        # Every entry of P (upper triangle) and A is kept, so the sparsity never changes.
        self.P_pattern = sparse.csc_matrix(np.triu(np.ones((NX, NX))))
        self.A_pattern = sparse.csc_matrix(np.ones((NC, NX)))
        self.solver = None
        self.eye = np.eye(NX)
        self.qdd, self.f = np.zeros(NV), np.zeros(NF)
        self.solve_time, self.failures = 0.0, 0

    def torques(self, data, stance=(True, True, True, True), f_des=None, swing=None, base=None):
        """Joint torques (12, actuator order) for this control step.

        stance: 4 bools, FL FR RL RR. A stance foot stays put and pushes on the ground.
        f_des:  (4, 3) ground forces to aim for (world frame, N), e.g. from an MPC, or None.
        swing:  {leg: (p, v, a)} target position, velocity and acceleration (world frame) of the
                centre of each swing foot's ball.
        base:   Target for the body, or None (allowed only with f_des: the forces move the body).
        """
        stance = np.asarray(stance, bool)
        swing = swing or {}
        if base is None and f_des is None:
            raise ValueError("give a base target, desired forces, or both")
        m, d = self.model, data
        mujoco.mj_forward(m, d)                  # bring M, bias forces and positions up to date
        M = np.zeros((NV, NV))
        mujoco.mj_fullM(m, d, M)                 # dense 18 x 18 mass matrix
        h = d.qfrc_bias - d.qfrc_passive         # gravity, Coriolis and centrifugal, joint damping
        qd = d.qvel
        R = d.xmat[1].reshape(3, 3)              # body to world

        # Foot Jacobians and Jdot qdot: at the bottom of the ball for a stance foot (where the
        # ground pushes), at the centre of the ball for a swing foot (the point that follows the
        # swing path).
        J = np.zeros((NF, NV))
        dJ = np.zeros((3, NV))
        dJ_qd = np.zeros(NF)
        centres = d.geom_xpos[self.foot_geoms]
        for i in range(4):
            point = centres[i] - [0.0, 0.0, go2.FOOT_RADIUS * stance[i]]
            mujoco.mj_jac(m, d, J[3 * i: 3 * i + 3], None, point, self.calves[i])
            mujoco.mj_jacDot(m, d, dJ, None, point, self.calves[i])
            dJ_qd[3 * i: 3 * i + 3] = dJ @ qd

        # Cost (1/2) x^T P x + q^T x, built from weighted least squares tasks w |rows x - b|^2.
        P = np.zeros((NX, NX))
        q = np.zeros(NX)

        def task(rows, b, w):
            P[:] += 2 * w * rows.T @ rows
            q[:] -= 2 * w * rows.T @ b

        def pick(cols):                          # rows that select some entries of x
            return self.eye[list(cols)]

        # Body: position and orientation PD (plus a small integral when there is no f_des).
        # For MuJoCo's free joint, qdd[0:3] is the acceleration of the body origin in the world
        # frame and qdd[3:6] the angular acceleration in the body frame.
        if base is not None:
            e_pos = base.pos - d.qpos[:3]
            e_rot = Rotation.from_matrix(R.T @ base.rot).as_rotvec()     # rotation still to go
            ki = self.ki if f_des is None else 0.0
            if ki:
                self.int_pos = np.clip(self.int_pos + e_pos * self.dt, -self.i_max, self.i_max)
                self.int_rot = np.clip(self.int_rot + e_rot * self.dt, -self.i_max, self.i_max)
            a_pos = (base.acc + self.kp_pos * e_pos + ki * self.int_pos
                     + self.kd_pos * (base.vel - qd[:3]))
            a_rot = (R.T @ base.alpha + self.kp_rot * e_rot + ki * self.int_rot
                     + self.kd_rot * (R.T @ base.omega - qd[3:6]))
            w = self.w_base * (self.base_scale if f_des is not None else 1.0)
            for j, a in enumerate(np.concatenate([a_pos, a_rot])):
                task(pick([j]), np.array([a]), w[j])

        # Swing feet: foot acceleration J qdd + Jdot qdot = PD on the target path.
        for leg, (p, v, a) in swing.items():
            rows = slice(3 * leg, 3 * leg + 3)
            a_des = (np.asarray(a, float) + self.kp_swing * (np.asarray(p, float) - centres[leg])
                     + self.kd_swing * (np.asarray(v, float) - J[rows] @ qd))
            task(np.hstack([J[rows], np.zeros((3, NF))]), a_des - dJ_qd[rows], self.w_swing)

        # Posture: the joint angles that would put each foot where it is now (stance) or at its
        # swing target, with the body at its target pose (leg IK from go2.py). It is a weak joint PD
        # with no velocity feedforward: it keeps the QP well posed and slightly slows moving targets.
        body_pos, body_rot = (base.pos, base.rot) if base is not None else (d.qpos[:3], R)
        aims = [np.asarray(swing[leg][0], float) if leg in swing else centres[leg]
                for leg in range(4)]
        q_ref = np.concatenate([go2.leg_ik(leg, body_rot.T @ (aims[leg] - body_pos))
                                for leg in range(4)])
        a_joint = self.kp_joint * (q_ref - d.qpos[7:]) - self.kd_joint * qd[6:]
        task(pick(range(6, NV)), a_joint, self.w_joint)

        # Forces: follow f_des on stance feet if it is given, otherwise keep them small.
        for i in range(4):
            cols = range(NV + 3 * i, NV + 3 * i + 3)
            if f_des is not None and stance[i]:
                task(pick(cols), np.asarray(f_des[i], float), self.w_force_track)
            else:
                task(pick(cols), np.zeros(3), self.w_force)

        # Constraints l <= A x <= u.
        A = np.zeros((NC, NX))
        l, u = np.zeros(NC), np.zeros(NC)
        A[0:6, :NV], A[0:6, NV:] = M[:6], -J[:, :6].T            # body rows: M qdd + h = J^T f
        l[0:6] = u[0:6] = -h[:6]
        for i in range(4):                                        # stance feet do not accelerate
            rows, foot = slice(6 + 3 * i, 9 + 3 * i), slice(3 * i, 3 * i + 3)
            if stance[i]:                                         # (and slow down if they slide)
                A[rows, :NV] = J[foot]
                l[rows] = u[rows] = -dJ_qd[foot] - self.k_foot * (J[foot] @ qd)
            else:
                l[rows], u[rows] = -np.inf, np.inf                # a row of zeros, always true
        A[18:34, NV:] = self.friction                             # friction pyramid
        l[18:34], u[18:34] = -np.inf, 0.0
        A[34:38, NV:] = self.normal                               # push, never pull. A swing
        l[34:38] = np.where(stance, self.fz_min, 0.0)             # foot gets fz = 0, and so
        u[34:38] = np.where(stance, self.fz_max, 0.0)             # fx = fy = 0 by the pyramid
        A[38:50, :NV], A[38:50, NV:] = M[6:], -J[:, 6:].T         # torque = M qdd + h - J^T f
        l[38:50], u[38:50] = self.tau_lo - h[6:], self.tau_hi - h[6:]

        # Solve. The sizes and sparsity never change, so OSQP is set up once, and later only the
        # numbers are updated (it also starts from the last solution).
        Px, Ax = csc_values(np.triu(P), self.P_pattern), csc_values(A, self.A_pattern)
        if self.solver is None:
            self.solver = osqp.OSQP()
            P_csc = sparse.csc_matrix((Px, self.P_pattern.indices, self.P_pattern.indptr),
                                      shape=(NX, NX))
            A_csc = sparse.csc_matrix((Ax, self.A_pattern.indices, self.A_pattern.indptr),
                                      shape=(NC, NX))
            self.solver.setup(P_csc, q, A_csc, l, u, verbose=False, eps_abs=1e-6, eps_rel=1e-6,
                              polishing=True, warm_starting=True)
        else:
            self.solver.update(Px=Px, q=q, Ax=Ax, l=l, u=u)
        res = self.solver.solve(raise_error=False)   # a failed solve is handled below
        self.solve_time = res.info.run_time       # OSQP update, solve and polish
        if res.info.status_val in (1, 2):        # solved, or solved to lower accuracy
            self.qdd, self.f = res.x[:NV], res.x[NV:]
        else:
            self.failures += 1                   # keep the last solution
        self.M, self.h, self.J, self.dJ_qd, self.stance = M, h, J, dJ_qd, stance   # for tests

        # Joint torques from the joint rows of the equations of motion.
        tau = M[6:] @ self.qdd + h[6:] - J[:, 6:].T @ self.f
        if self.friction_comp:   # the joints' dry friction (0.2 N m), pushed along the motion
            tau += m.dof_frictionloss[6:] * np.tanh(qd[6:] / 0.05)
        return np.clip(tau, self.tau_lo, self.tau_hi)


def foot_forces(model, data, geoms):
    """Ground force on each foot (world frame), summed over its contacts, from MuJoCo."""
    forces = np.zeros((4, 3))
    f6 = np.zeros(6)
    for k in range(data.ncon):
        c = data.contact[k]
        for leg, g in enumerate(geoms):
            if g in (c.geom1, c.geom2):
                mujoco.mj_contactForce(model, data, k, f6)
                f = c.frame.reshape(3, 3).T @ f6[:3]   # contact frame to world: force on geom2
                forces[leg] += f if c.geom2 == g else -f
    return forces
