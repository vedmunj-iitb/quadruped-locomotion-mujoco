"""Leg kinematics against MuJoCo's own forward kinematics. Written in 2026."""
import mujoco
import numpy as np
import pytest

from common import go2

RNG = np.random.default_rng(0)


def random_angles(n):
    """Joint angles inside the model's limits, knee bent backwards."""
    return np.column_stack([RNG.uniform(-0.8, 0.8, n), RNG.uniform(-0.5, 2.0, n),
                            RNG.uniform(-2.6, -0.9, n)])


@pytest.mark.parametrize("leg", range(4))
def test_fk_undoes_ik(leg):
    """Points in a box round the walking stance: IK, then FK, gives the point back."""
    home = go2.HIP[leg] + [0.0, go2.SIDE[leg] * go2.L_HIP, -0.28]
    for p in home + RNG.uniform([-0.15, -0.05, -0.08], [0.15, 0.05, 0.10], (300, 3)):
        assert np.allclose(go2.foot_position(leg, go2.leg_ik(leg, p)), p, rtol=0, atol=1e-9)


@pytest.mark.parametrize("leg", range(4))
def test_ik_undoes_fk(leg):
    """The IK picks the foot below the thigh joint, so only those poses are checked."""
    for q in random_angles(300):
        in_plane = go2.rot_y(q[1]) @ (np.array([0.0, 0.0, -go2.L_THIGH]) + go2.rot_y(q[2]) @ go2.CALF)
        if in_plane[2] < -0.05:
            assert np.allclose(go2.leg_ik(leg, go2.foot_position(leg, q)), q, rtol=0, atol=1e-9)


def test_fk_matches_mujoco():
    model = go2.load_model()
    data = mujoco.MjData(model)
    geoms = go2.foot_geoms(model)
    for q in random_angles(50):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0                       # body at the origin, not rotated
        for leg in range(4):
            data.qpos[7 + 3 * leg: 10 + 3 * leg] = q
        mujoco.mj_kinematics(model, data)
        for leg in range(4):
            assert np.allclose(data.geom_xpos[geoms[leg]], go2.foot_position(leg, q), rtol=0, atol=1e-9)


def test_out_of_reach_goes_to_the_edge():
    q = go2.leg_ik(0, go2.HIP[0] + np.array([0.0, go2.L_HIP, -1.0]))   # 1 m below the hip
    assert np.allclose(q[1:], [0.0, -go2.KNEE_OFFSET], rtol=0, atol=1e-6)   # leg straight down
