"""Convex model predictive control of the body with the four foot forces, solved by OSQP.

The robot is one rigid body pushed by the forces of the feet on the ground, with the rotation
simplified to yaw only, as in "Dynamic Locomotion in the MIT Cheetah 3 Through Convex
Model-Predictive Control" (IROS 2018). That makes the dynamics linear, so the whole horizon is
one quadratic program (QP).

This module does not depend on any leg controller: give it the body state, the feet and the
contact plan, and it returns the force each foot should take from the ground.

Written in 2026.
"""
import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import expm

GRAVITY = 9.81
NX, NU = 13, 12   # state: roll, pitch, yaw, position (3), angular velocity (3), velocity (3), -g
                  # control: force of the ground on each foot, (fx, fy, fz) x 4, world frame

# State error weights in the order above, and the weight on the forces. These are tuned values.
STATE_WEIGHTS = np.array([50.0, 50.0, 10.0, 0.0, 0.0, 100.0, 2.0, 2.0, 1.0, 5.0, 5.0, 1.0, 0.0])
FORCE_WEIGHT = 1e-5


def skew(r):
    """Matrix [r]x with [r]x f = r x f."""
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def composite_inertia(model, data, base=1):
    """Inertia of the whole robot about its centre of mass, in the frame of body `base` (kg m^2).

    Parallel axis theorem over every body of a MuJoCo model in the pose of `data`: the body's
    own inertia, turned into the base frame, plus m (|d|^2 I - d d') for its offset d from the
    centre of mass.
    """
    R = data.xmat[base].reshape(3, 3)
    total = np.zeros((3, 3))
    for b in range(1, model.nbody):
        Rb = R.T @ data.ximat[b].reshape(3, 3)
        d = R.T @ (data.xipos[b] - data.subtree_com[base])
        total += Rb @ np.diag(model.body_inertia[b]) @ Rb.T
        total += model.body_mass[b] * (d @ d * np.eye(3) - np.outer(d, d))
    return total


class ConvexMPC:
    """Single rigid body MPC over N steps of dt seconds.

    mass (kg) and inertia (3 x 3, body frame, about the centre of mass) describe the robot.
    mu is the friction coefficient of the pyramid |fx|, |fy| <= mu fz, and f_max the largest
    normal force of one foot (N). ConvexMPC.from_model builds one from a MuJoCo model.
    """

    def __init__(self, mass, inertia, horizon=12, dt=0.03, mu=0.5, f_max=180.0,
                 state_weights=STATE_WEIGHTS, force_weight=FORCE_WEIGHT):
        self.mass = mass
        self.inertia = np.asarray(inertia, float)
        self.N, self.dt = horizon, dt
        self.mu, self.f_max = mu, f_max
        N = horizon
        # Decision variables z = [x_1 .. x_N, u_0 .. u_N-1]. The cost sum (x - x_ref)' Q (x - x_ref)
        # + u' R u is written as 1/2 z' P z + q' z with P = 2 diag(Q .. Q, R .. R).
        self.Q = np.diag(state_weights)
        P = 2 * sp.block_diag([sp.kron(sp.eye(N), self.Q), force_weight * sp.eye(NU * N)],
                              format="csc")
        # Fixed (row, column) pattern of the constraint matrix. Only its numbers change later, so
        # OSQP is set up once (with ones as placeholders) and then updated before every solve.
        self.rows, self.cols = self._pattern()
        A, l, u = self._constraints(np.ones((NX, NX)), np.ones((NX, NU)), np.zeros(NX),
                                    np.ones((N, 4), dtype=bool))
        self.solver = osqp.OSQP()
        self.solver.setup(P, np.zeros(P.shape[0]), A, l, u, verbose=False, warm_starting=True,
                          eps_abs=1e-4, eps_rel=1e-4, polishing=True, max_iter=4000)
        self.forces = np.zeros((4, 3))    # last good solution, kept if a solve fails
        self.info = None                  # OSQP info of the last solve (status_val, solve_time)

    @classmethod
    def from_model(cls, model, data, **kw):
        """Mass and inertia from a MuJoCo model in the pose of `data` (the robot is body 1)."""
        return cls(model.body_subtreemass[1], composite_inertia(model, data), **kw)

    def solve(self, rpy, pos, omega_world, vel_world, feet_world, contact_plan, v_cmd_world,
              yaw_rate_cmd, z_ref, yaw_ref=None):
        """Foot forces that make the body follow a velocity command.

        All vectors are in the world frame (z up). Feet are in the order FL, FR, RL, RR.
          rpy           roll, pitch, yaw of the body (rad, Z-Y-X Euler angles)
          pos           centre of mass position (m)
          omega_world   angular velocity of the body (rad/s)
          vel_world     velocity of the centre of mass (m/s)
          feet_world    (4, 3) foot positions (m). The controllers here pass the centres of the
                        foot balls, 22 mm above the ground contact. For a foot in the air, pass
                        where it will land: that is where its force acts once it is down.
          contact_plan  (N, 4) bool, row k for the step from t + k dt to t + (k + 1) dt,
                        True where the foot is on the ground
          v_cmd_world   commanded velocity of the centre of mass (m/s, z is ignored)
          yaw_rate_cmd  commanded turn rate (rad/s)
          z_ref         height of the centre of mass to hold (m)
          yaw_ref       heading to hold now (rad). None holds the current heading, so only the
                        turn rate is tracked.
        Returns
          forces        (4, 3) force of the ground on each foot (N), world frame, to apply
                        from now until the next solve. A foot planned in the air gets zero.
          prediction    (N, 13) predicted states at t + dt .. t + N dt, or None if OSQP failed
                        (then `forces` is the last good solution and self.info says why).
        """
        yaw = rpy[2]
        x0 = np.concatenate([rpy, pos, omega_world, vel_world, [-GRAVITY]])
        # Wanted states: level, at height z_ref, moving and turning as commanded. The position
        # starts from where the body is and has no weight in the cost, so speed is tracked.
        heading = yaw if yaw_ref is None else yaw + (yaw_ref - yaw + np.pi) % (2 * np.pi) - np.pi
        k = np.arange(1, self.N + 1)[:, None]
        v = np.array([v_cmd_world[0], v_cmd_world[1], 0.0])
        x_ref = np.zeros((self.N, NX))
        x_ref[:, 2:3] = heading + yaw_rate_cmd * k * self.dt
        x_ref[:, 3:6] = np.asarray(pos) + v * k * self.dt
        x_ref[:, 5] = z_ref
        x_ref[:, 8] = yaw_rate_cmd
        x_ref[:, 9:12] = v
        x_ref[:, 12] = -GRAVITY
        forces, prediction, self.info = self.solve_qp(x0, x_ref, np.asarray(feet_world) - pos,
                                                      np.asarray(contact_plan, bool))
        if forces is not None:
            self.forces = forces
        return self.forces, prediction

    def dynamics(self, yaw, arms):
        """Continuous A, B of the single rigid body model.

        arms: (4, 3) positions of the feet relative to the centre of mass, world frame.
        """
        A, B = np.zeros((NX, NX)), np.zeros((NX, NU))
        Rz = rot_z(yaw)
        A[0:3, 6:9] = Rz.T             # Euler angle rates from the world angular velocity (yaw only)
        A[3:6, 9:12] = np.eye(3)       # position changes with velocity
        A[11, 12] = 1.0                # the last state, -g, accelerates the body down
        I_inv = np.linalg.inv(Rz @ self.inertia @ Rz.T)   # inertia in the world frame
        for i in range(4):
            B[6:9, 3 * i:3 * i + 3] = I_inv @ skew(arms[i])   # torque r x f turns the body
            B[9:12, 3 * i:3 * i + 3] = np.eye(3) / self.mass  # force f / m accelerates it
        return A, B

    def discretize(self, A, B):
        """Exact zero-order hold over dt: expm([[A, B], [0, 0]] dt) = [[Ad, Bd], [0, I]]."""
        M = np.zeros((NX + NU, NX + NU))
        M[:NX, :NX], M[:NX, NX:] = A, B
        E = expm(M * self.dt)
        return E[:NX, :NX], E[:NX, NX:]

    def _pattern(self):
        """Row and column of every entry the constraint matrix can have, in a fixed order."""
        N, rows, cols = self.N, [], []
        xu = NX * N                                   # first column of the forces
        for k in range(N):
            r = np.arange(NX * k, NX * k + NX)
            rows.append(r); cols.append(NX * k + np.arange(NX))                       # + x_k+1
            if k > 0:
                rr, cc = np.meshgrid(r, NX * (k - 1) + np.arange(NX), indexing="ij")
                rows.append(rr.ravel()); cols.append(cc.ravel())                      # - Ad x_k
            rr, cc = np.meshgrid(r, xu + NU * k + np.arange(NU), indexing="ij")
            rows.append(rr.ravel()); cols.append(cc.ravel())                          # - Bd u_k
        first = NX * N                                # friction rows start here
        for k in range(N):
            for i in range(4):
                fx, fy, fz = xu + NU * k + 3 * i + np.arange(3)
                r = first + 5 * (4 * k + i)
                rows += [np.array([r, r, r + 1, r + 1, r + 2, r + 2, r + 3, r + 3, r + 4])]
                cols += [np.array([fx, fz, fx, fz, fy, fz, fy, fz, fz])]
        return np.concatenate(rows), np.concatenate(cols)

    def _constraints(self, Ad, Bd, x0, plan):
        """Constraint matrix and bounds l <= A z <= u for one solve.

        Dynamics, as equalities: x_k+1 - Ad x_k - Bd u_k = 0, and x_1 - Bd u_0 = Ad x_0.
        Each foot force, as a friction pyramid: |fx| <= mu fz, |fy| <= mu fz, and
        0 <= fz <= f_max on a planned stance foot, fz = 0 (so all of f = 0) on a swing foot.
        """
        N, mu = self.N, self.mu
        vals = []
        for k in range(N):
            vals.append(np.ones(NX))
            if k > 0:
                vals.append(-Ad.ravel())
            vals.append(-Bd.ravel())
        vals.append(np.tile([1.0, -mu, 1.0, mu, 1.0, -mu, 1.0, mu, 1.0], 4 * N))
        n_rows = NX * N + 5 * 4 * N
        A = sp.csc_matrix((np.concatenate(vals), (self.rows, self.cols)),
                          shape=(n_rows, (NX + NU) * N))
        l, u = np.zeros(n_rows), np.zeros(n_rows)
        l[:NX], u[:NX] = Ad @ x0, Ad @ x0
        stance = plan.reshape(-1).astype(float)              # (step, foot) in the row order
        inf = np.inf
        l[NX * N:] = np.column_stack([np.full(4 * N, -inf), np.zeros(4 * N), np.full(4 * N, -inf),
                                      np.zeros(4 * N), np.zeros(4 * N)]).ravel()
        u[NX * N:] = np.column_stack([np.zeros(4 * N), np.full(4 * N, inf), np.zeros(4 * N),
                                      np.full(4 * N, inf), self.f_max * stance]).ravel()
        return A, l, u

    def solve_qp(self, x0, x_ref, arms, plan):
        """The QP itself. x0: state (13), x_ref: (N, 13) wanted states at steps 1..N, arms:
        (4, 3) feet relative to the centre of mass, plan: (N, 4) bool.
        Returns (forces of the first step (4, 3), predicted states (N, 13), OSQP info), with
        None for the first two if OSQP did not solve it."""
        Ad, Bd = self.discretize(*self.dynamics(x0[2], arms))
        A, l, u = self._constraints(Ad, Bd, x0, plan)
        q = np.concatenate([-2 * (x_ref @ self.Q).ravel(), np.zeros(NU * self.N)])
        self.solver.update(q=q, l=l, u=u, Ax=A.data)
        res = self.solver.solve(raise_error=False)
        if res.x is None or res.info.status_val not in (1, 2):   # 1 solved, 2 solved inaccurate
            return None, None, res.info
        X = res.x[:NX * self.N].reshape(self.N, NX)
        U = res.x[NX * self.N:].reshape(self.N, NU)
        return U[0].reshape(4, 3), X, res.info
