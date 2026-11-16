"""Gait timing and swing curves. Written in 2026."""
import numpy as np
import pytest

from gait.scheduler import GAITS, GaitScheduler, bezier, bezier_velocity, swing_curve

TIMES = np.linspace(0.0, 3.0, 1501)


def stance_sets(name):
    sched = GaitScheduler(GAITS[name])
    return {tuple(sched.in_stance(leg, t) for leg in range(4)) for t in TIMES}


def test_trot_moves_diagonal_pairs():          # FL+RR, then FR+RL
    assert stance_sets("trot") == {(True, False, False, True), (False, True, True, False)}


def test_pace_moves_side_pairs():               # FL+RL, then FR+RR
    assert stance_sets("pace") == {(True, False, True, False), (False, True, False, True)}


def test_bound_moves_front_and_rear_pairs():
    assert stance_sets("bound") == {(True, True, False, False), (False, False, True, True)}


def test_walk_keeps_three_feet_down_in_order():
    sched = GaitScheduler(GAITS["walk"])
    swinging = []
    for t in TIMES:
        down = [sched.in_stance(leg, t) for leg in range(4)]
        assert sum(down) == 3
        leg = down.index(False)
        if not swinging or swinging[-1] != leg:
            swinging.append(leg)
    assert swinging[:5] == [2, 0, 3, 1, 2]       # RL, FL, RR, FR, RL


@pytest.mark.parametrize("name", GAITS)
def test_progress_runs_from_0_to_1(name):
    """Within every full stance and swing, progress rises from near 0 to near 1."""
    sched = GaitScheduler(GAITS[name])
    for leg in range(4):
        p = np.array([sched.progress(leg, t) for t in TIMES])
        down = np.array([sched.in_stance(leg, t) for t in TIMES])
        assert p.min() >= 0.0 and p.max() < 1.0
        cuts = np.flatnonzero(down[1:] != down[:-1]) + 1
        for a, b in zip(cuts[:-1], cuts[1:]):         # whole stances and swings only
            assert np.all(np.diff(p[a:b]) > 0)
            assert p[a] < 0.05 and p[b - 1] > 0.95


def test_swing_curve_ends_and_height():
    start, end = np.array([0.15, 0.14, -0.28]), np.array([0.25, 0.14, -0.28])
    v = np.array([-0.5, 0.0, 0.3])          # the vertical part must be dropped
    pts = swing_curve(start, end, 0.08, v, 0.2)
    assert np.allclose(bezier(pts, 0.0), start) and np.allclose(bezier(pts, 1.0), end)
    assert np.isclose(bezier(pts, 0.5)[2] - start[2], 0.08)
    # Lift-off and touchdown at the stance velocity, with no vertical speed.
    for s in (0.0, 1.0):
        assert np.allclose(bezier_velocity(pts, s, 0.2), [v[0], v[1], 0.0], rtol=0, atol=1e-12)


def test_bezier_velocity_matches_finite_difference():
    pts = swing_curve([0.1, 0.1, -0.3], [0.2, 0.12, -0.25], 0.06, [-0.3, 0.05, 0.0], 0.15)
    h = 1e-6
    for s in (0.1, 0.4, 0.8):
        numeric = (bezier(pts, s + h) - bezier(pts, s - h)) / (2 * h) / 0.15
        assert np.allclose(bezier_velocity(pts, s, 0.15), numeric, atol=1e-6)


def test_contact_plan_matches_in_stance():
    sched = GaitScheduler(GAITS["trot"])
    sched.advance(0.13)
    plan = sched.contact_plan(10, 0.03)
    assert plan.shape == (10, 4)
    assert all(plan[k, leg] == sched.in_stance(leg, 0.13 + k * 0.03)
               for k in range(10) for leg in range(4))
