"""The MPC model and QP on hand-made cases. Written in 2026."""
import numpy as np
import pytest

from mpc.mpc import GRAVITY, NU, NX, ConvexMPC, skew

MASS = 15.0
INERTIA = np.diag([0.2, 0.5, 0.5])
ARMS = np.array([[0.19, 0.14, -0.28], [0.19, -0.14, -0.28],
                 [-0.19, 0.14, -0.28], [-0.19, -0.14, -0.28]])   # feet relative to the CoM
ALL_DOWN = np.ones((12, 4), dtype=bool)


@pytest.fixture
def mpc():
    return ConvexMPC(MASS, INERTIA, horizon=12, dt=0.03, mu=0.5, f_max=180.0)


def at_rest(z=0.28):
    x = np.zeros(NX)
    x[5], x[12] = z, -GRAVITY
    return x


def test_force_blocks(mpc):
    A, B = mpc.dynamics(0.0, ARMS)
    for i in range(4):
        assert np.allclose(B[9:12, 3 * i:3 * i + 3], np.eye(3) / MASS)        # f / m
        assert np.allclose(B[6:9, 3 * i:3 * i + 3], np.linalg.inv(INERTIA) @ skew(ARMS[i]))
    assert A[11, 12] == 1.0                                                  # gravity state


def test_euler_rates_turn_with_yaw(mpc):
    A, _ = mpc.dynamics(np.pi / 2, ARMS)
    # Heading 90 deg: a world x angular velocity is a pitch rate of -1 (the body x axis is world y).
    assert np.allclose(A[0:3, 6:9] @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0])


def test_free_fall_one_step(mpc):
    Ad, _ = mpc.discretize(*mpc.dynamics(0.0, ARMS))
    x1 = Ad @ at_rest()
    assert np.isclose(x1[11], -GRAVITY * 0.03)                  # v = -g t
    assert np.isclose(x1[5], 0.28 - GRAVITY * 0.03**2 / 2)      # z = z0 - g t^2 / 2


def test_skew_is_cross_product():
    r, f = np.array([0.1, -0.2, 0.3]), np.array([4.0, 5.0, -6.0])
    assert np.allclose(skew(r) @ f, np.cross(r, f))


def test_standing_carries_the_weight(mpc):
    x0 = at_rest()
    forces, X, info = mpc.solve_qp(x0, np.tile(x0, (12, 1)), ARMS, ALL_DOWN)
    assert info.status_val in (1, 2)
    assert np.isclose(forces[:, 2].sum(), MASS * GRAVITY, rtol=0.01)
    assert np.allclose(forces[:, :2], 0.0, atol=0.5)
    # The body is predicted to stay put for the first half of the horizon. (Near the end the
    # plan lets it sag a little to save force, since nothing is weighted after step N.)
    assert np.abs(X[:6, 5] - 0.28).max() < 1e-3


def test_swing_feet_get_no_force_and_the_pyramid_holds(mpc):
    x0 = at_rest()
    x_ref = np.tile(x0, (12, 1))
    x_ref[:, 9] = 1.0                                           # ask for 1 m/s forward at once
    plan = ALL_DOWN.copy()
    plan[:6, [1, 2]] = False                                    # FR and RL in the air, then down
    plan[6:, [0, 3]] = False
    forces, X, info = mpc.solve_qp(x0, x_ref, ARMS, plan)
    assert info.status_val in (1, 2)
    assert np.allclose(forces[[1, 2]], 0.0, atol=0.05)          # swing feet push nothing
    for f in forces[[0, 3]]:
        assert -0.05 <= f[2] <= 180.0 + 0.05
        assert abs(f[0]) <= 0.5 * f[2] + 0.05 and abs(f[1]) <= 0.5 * f[2] + 0.05
    assert forces[[0, 3], 0].sum() > 10.0                       # and they push the body forward


def test_constraint_pattern_never_changes(mpc):
    """OSQP is set up once with placeholder ones and then only updated, so every real matrix
    must have the pattern of that first one, also at yaw 0 where more entries are exactly 0."""
    setup, _, _ = mpc._constraints(np.ones((NX, NX)), np.ones((NX, NU)), np.zeros(NX), ALL_DOWN)
    for yaw, arms in ((0.0, ARMS), (0.3, ARMS), (-2.0, ARMS * 1.1)):
        a, _, _ = mpc._constraints(*mpc.discretize(*mpc.dynamics(yaw, arms)), at_rest(), ALL_DOWN)
        assert np.array_equal(a.indices, setup.indices) and np.array_equal(a.indptr, setup.indptr)
    a1 = setup
    assert a1.shape == (NX * 12 + 5 * 4 * 12, (NX + NU) * 12)


def test_solve_api_in_world_coordinates(mpc):
    """The interface for a whole body controller: absolute positions, world frame."""
    com = np.array([2.0, -1.0, 0.28])
    forces, prediction = mpc.solve(np.zeros(3), com, np.zeros(3), np.zeros(3), com + ARMS,
                                   ALL_DOWN, np.zeros(3), 0.0, 0.28)
    assert prediction.shape == (12, NX)
    assert np.isclose(forces[:, 2].sum(), MASS * GRAVITY, rtol=0.01)


def test_heading_error_wraps_round(mpc):
    """Yaw -3.1 rad and a wanted heading of +3.1 rad are 0.083 rad apart, not 6.2 rad."""
    com = np.array([0.0, 0.0, 0.28])
    args = (np.zeros(3), np.zeros(3), com + ARMS, ALL_DOWN, np.zeros(3), 0.0, 0.28)
    f_wrap, _ = mpc.solve(np.array([0.0, 0.0, -3.1]), com, *args, yaw_ref=3.1)
    f_near, _ = mpc.solve(np.array([0.0, 0.0, -3.1]), com, *args, yaw_ref=-3.1 - 0.0832)
    assert np.allclose(f_wrap, f_near, atol=0.5)
