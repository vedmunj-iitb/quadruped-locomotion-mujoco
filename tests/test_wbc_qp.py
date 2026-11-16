"""The QP's solution against MuJoCo's own dynamics, and its limits. Written in 2026."""
import mujoco
import numpy as np
import pytest
from scipy import sparse

from common import go2
from wbc import scenarios as run
from wbc.wbc import WholeBodyQP, csc_values, rpy_matrix, stand, target_at


def controller_after(seconds, pushes=(), plan=None, path=run.hold):
    """A controller and its data after `seconds` of the given run, with one more step solved."""
    model = go2.load_model()
    data = mujoco.MjData(model)
    stand(model, data)
    wbc = WholeBodyQP(model)
    while True:
        data.xfrc_applied[1, :3] = run.push_at(pushes, data.time)
        inputs = plan(data.time, data) if plan else {}
        data.ctrl[:] = wbc.torques(data, base=target_at(path, data.time), **inputs)
        if data.time >= seconds - 1e-9:
            return model, data, wbc
        mujoco.mj_step(model, data)


def check_inverse_dynamics(model, data, wbc):
    """MuJoCo's inverse dynamics (contacts left out) for qdd from the QP needs exactly the
    solution's joint torques and foot forces: M qdd + h = S^T tau + J^T f, all 18 rows."""
    tau = wbc.M[6:] @ wbc.qdd + wbc.h[6:] - wbc.J[:, 6:].T @ wbc.f
    data.qacc[:] = wbc.qdd
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    mujoco.mj_inverse(model, data)
    applied = np.concatenate([np.zeros(6), tau]) + wbc.J.T @ wbc.f
    assert np.allclose(data.qfrc_inverse, applied, atol=1e-3)


@pytest.mark.parametrize("pushes", [(), ((0.5, 0.1, (0.0, 100.0, 0.0)),)])
def test_standing_solution_obeys_mujoco_inverse_dynamics(pushes):
    check_inverse_dynamics(*controller_after(0.55, pushes))


def test_three_feet_solution_obeys_mujoco_inverse_dynamics_and_swing_foot_is_free():
    model, data, wbc = controller_after(3.0, plan=run.three_leg_plan(go2.load_model()),
                                        path=run.lean)
    assert not wbc.stance[run.SWING_LEG]
    assert np.abs(wbc.f.reshape(4, 3)[run.SWING_LEG]).max() < 1e-3   # under 1 mN in the air
    check_inverse_dynamics(model, data, wbc)


def test_stance_feet_obey_the_foot_damping_constraint():
    """Stance feet: J qdd + Jdot qd = -k_foot J qd, so a foot at rest does not accelerate and a
    sliding foot is slowed down."""
    _, data, wbc = controller_after(0.3)
    foot_acc = wbc.J @ wbc.qdd + wbc.dJ_qd
    assert np.allclose(foot_acc, -wbc.k_foot * (wbc.J @ data.qvel), atol=1e-4)


def test_limits_hold_through_a_hard_push():
    """Friction cone, normal force and torque limits at every step of a 120 N push."""
    model = go2.load_model()
    data = mujoco.MjData(model)
    stand(model, data)
    wbc = WholeBodyQP(model)
    pushes = ((0.2, 0.1, (0.0, 120.0, 0.0)),)
    tol = 1e-3
    while data.time < 1.0:
        data.xfrc_applied[1, :3] = run.push_at(pushes, data.time)
        data.ctrl[:] = wbc.torques(data, base=target_at(run.hold, data.time))
        f = wbc.f.reshape(4, 3)
        assert np.all(np.linalg.norm(f[:, :2], axis=1) <= wbc.mu * f[:, 2] + tol)   # in the cone
        assert np.all(np.abs(f[:, :2]).max(axis=1) <= wbc.mu / np.sqrt(2) * f[:, 2] + tol)   # pyramid
        assert np.all(f[:, 2] >= wbc.fz_min - tol) and np.all(f[:, 2] <= wbc.fz_max + tol)
        tau = wbc.M[6:] @ wbc.qdd + wbc.h[6:] - wbc.J[:, 6:].T @ wbc.f
        assert np.all(tau >= wbc.tau_lo - tol) and np.all(tau <= wbc.tau_hi + tol)
        mujoco.mj_step(model, data)
    assert wbc.failures == 0


def test_needs_a_base_target_or_forces():
    model = go2.load_model()
    data = mujoco.MjData(model)
    stand(model, data)
    with pytest.raises(ValueError):
        WholeBodyQP(model).torques(data)


def test_csc_values_keeps_the_storage_order():
    dense = np.arange(12.0).reshape(3, 4)
    pattern = sparse.csc_matrix(np.ones((3, 4)))
    rebuilt = sparse.csc_matrix((csc_values(dense, pattern), pattern.indices, pattern.indptr),
                                shape=(3, 4))
    assert np.array_equal(rebuilt.toarray(), dense)


def test_target_derivatives():
    """Central differences of a known path match the analytic velocity and acceleration."""
    path = lambda t: ((0.0, 0.0, 0.3 + 0.05 * np.sin(2 * t)), (0.0, 0.2 * np.sin(3 * t), 0.0))
    tg = target_at(path, 0.4)
    assert np.isclose(tg.vel[2], 0.1 * np.cos(0.8), atol=1e-6)
    assert np.isclose(tg.acc[2], -0.2 * np.sin(0.8), atol=1e-4)
    assert np.allclose(tg.omega, [0.0, 0.6 * np.cos(1.2), 0.0], atol=1e-6)   # pitch rate, world y
    assert np.allclose(tg.alpha, [0.0, -1.8 * np.sin(1.2), 0.0], atol=1e-4)
    assert np.allclose(tg.rot, rpy_matrix(0.0, 0.2 * np.sin(1.2), 0.0))
