"""The whole robot's inertia, checked against MuJoCo's mass matrix. Written in 2026."""
import mujoco
import numpy as np

from common import go2
from mpc.controller import MPCController
from mpc.mpc import composite_inertia


def test_inertia_matches_the_mass_matrix():
    """For the free joint, the rotational block of the mass matrix is the inertia of the whole
    robot about the body origin, in the body frame. Moving ours there with the parallel axis
    theorem must give the same matrix."""
    model = go2.load_model()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1)
    MPCController(model).reset(data)
    for trial in range(3):
        if trial:   # also in two random leg poses, with the body turned
            data.qpos[7:] += rng.uniform(-0.3, 0.3, 12)
            quat = rng.normal(size=4)
            data.qpos[3:7] = quat / np.linalg.norm(quat)
            mujoco.mj_forward(model, data)
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, M)
        R = data.xmat[1].reshape(3, 3)
        c = R.T @ (data.subtree_com[1] - data.qpos[:3])
        m = model.body_subtreemass[1]
        about_origin = composite_inertia(model, data) + m * (c @ c * np.eye(3) - np.outer(c, c))
        assert np.allclose(about_origin, M[3:6, 3:6], rtol=0, atol=1e-9)
        assert np.isclose(M[0, 0], m)
