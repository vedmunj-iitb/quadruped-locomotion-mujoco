"""Raibert footholds and foothold selection. Written in 2026."""
import numpy as np

from footholds import planner
from footholds.terrain import GROUND, make_course

TERRAIN = make_course()


def hip_at(x, y):
    return np.array([x, y, GROUND + 0.30])


def test_raibert_point():
    hip, v = np.array([0.5, 0.14]), np.array([0.4, 0.0])
    assert np.allclose(planner.raibert_foothold(hip, v, 0.2), hip + v * 0.1)
    # Faster than commanded: the foot lands further ahead.
    faster = planner.raibert_foothold(hip, v, 0.2, v=np.array([0.5, 0.0]))
    assert np.isclose(faster[0] - (hip[0] + 0.04), 0.03 * 0.1)


def test_flat_ground_keeps_the_raibert_point():
    nominal = np.array([0.304, 0.142])
    chosen = planner.select_foothold(TERRAIN, nominal, hip_at(*nominal))
    assert np.linalg.norm(chosen[:2] - nominal) <= TERRAIN.res / np.sqrt(2) + 1e-9   # grid snap


def test_gap_is_avoided():
    for x in np.arange(0.55, 0.63, 0.01):
        nominal = np.array([x, 0.14])
        hip = hip_at(x - 0.04, 0.14)
        chosen = planner.select_foothold(TERRAIN, nominal, hip)
        iy, ix = TERRAIN.index(*chosen[:2])
        assert chosen[2] == GROUND                                   # on the ground, not in the gap
        assert TERRAIN.edge_dist[iy, ix] >= planner.EDGE_MIN
        assert np.linalg.norm(chosen[:2] - nominal) <= planner.WINDOW + 1e-9
        assert np.linalg.norm(chosen[:2] - hip[:2]) <= planner.HIP_REACH


def test_pothole_centre_moves_the_foot_out():
    nominal = np.array([2.45, 0.13])                                 # centre of a pothole
    chosen = planner.select_foothold(TERRAIN, nominal, hip_at(*nominal))
    assert chosen[2] == GROUND
    assert TERRAIN.edge_dist[TERRAIN.index(*chosen[:2])] >= planner.EDGE_MIN


def test_edge_cost_rises_towards_an_edge():
    (iy, ix), terms, _ = planner.foothold_costs(TERRAIN, np.array([0.45, 0.0]), hip_at(0.45, 0.0))
    x, _ = TERRAIN.point(iy, ix)
    near, far = terms["edge"][np.isclose(x, 0.52)], terms["edge"][np.isclose(x, 0.45)]
    assert near.min() > 0.0 and far.max() == 0.0


def test_no_feasible_foothold_returns_the_nominal():
    wide = TERRAIN.h.copy()
    wide[:, (np.arange(TERRAIN.nx) * TERRAIN.res + TERRAIN.x0 > 0.3)
            & (np.arange(TERRAIN.nx) * TERRAIN.res + TERRAIN.x0 < 0.9)] = 0.0   # 60 cm hole
    from footholds.terrain import Terrain
    t = Terrain(wide)
    nominal = np.array([0.6, 0.14])
    chosen = planner.select_foothold(t, nominal, hip_at(*nominal))
    assert np.allclose(chosen[:2], nominal)
