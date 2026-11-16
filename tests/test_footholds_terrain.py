"""The heightmap, its maps, and the MuJoCo heightfield made from it. Written in 2026."""
import mujoco
import numpy as np

from common import go2
from footholds.terrain import GROUND, make_course

TERRAIN = make_course()


def test_heightfield_matches_heightmap():
    """Vertical rays in MuJoCo hit the heightfield at the heightmap's height (away from edges)."""
    model = go2.load_model(TERRAIN.add_to_spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    only_terrain = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)   # group 0: floor and course
    hit = np.array([-1], dtype=np.int32)
    rng = np.random.default_rng(0)
    checked = 0
    for iy, ix in zip(rng.integers(0, TERRAIN.ny, 3000), rng.integers(0, TERRAIN.nx, 3000)):
        if TERRAIN.edge_dist[iy, ix] < 0.03 or min(iy, ix, TERRAIN.ny - 1 - iy, TERRAIN.nx - 1 - ix) < 2:
            continue                                   # near a terrain edge or the map's border
        x, y = TERRAIN.point(iy, ix)
        dist = mujoco.mj_ray(model, data, np.array([x, y, 1.0]), np.array([0.0, 0.0, -1.0]),
                             only_terrain, 1, -1, hit)
        assert abs((1.0 - dist) - TERRAIN.h[iy, ix]) < 1e-6
        checked += 1
    assert checked > 2000


def test_course_heights():
    assert TERRAIN.height(0.0, 0.0) == GROUND
    assert TERRAIN.height(0.59, 0.1) == 0.0                     # first gap
    assert np.isclose(TERRAIN.height(1.2, 0.0), GROUND + 0.05)  # on the step


def test_edge_distance_and_slope():
    assert TERRAIN.edge_dist[TERRAIN.index(0.55, 0.0)] == 0.0           # gap edge
    assert np.isclose(TERRAIN.edge_dist[TERRAIN.index(0.40, 0.0)], 0.14, atol=0.011)
    assert np.isclose(TERRAIN.slope[TERRAIN.index(2.05, 0.0)], 0.05 / 0.30)   # ramp
    assert TERRAIN.slope[TERRAIN.index(0.2, 0.0)] == 0.0                # flat ground
