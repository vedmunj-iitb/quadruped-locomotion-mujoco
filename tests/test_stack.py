"""The whole stack in MuJoCo: the course with and without the foothold planner, and flat ground.
Written in 2026."""
import numpy as np

from common.metrics import fall_index, hole_drops
from footholds.terrain import make_course
from stack.demo import COURSE, schedule_from
from stack.stack import STAND, TROT, run_stack

TERRAIN = make_course()


def test_full_stack_crosses_the_first_gap_and_the_step():
    log, ctrl = run_stack(schedule_from(COURSE), 5.0, terrain=TERRAIN)
    assert fall_index(log, TERRAIN) is None
    assert log["pos"][-1, 0] > 1.3            # past the 8 cm gap and onto the step
    assert not hole_drops(log)                # no foot went into a hole
    assert ctrl.wbc.failures == 0


def test_without_the_planner_a_foot_drops_into_the_first_gap():
    log, _ = run_stack(schedule_from(COURSE), 4.0, terrain=TERRAIN, select=False)
    drops = hole_drops(log)
    assert drops
    t, leg = drops[0]
    i = int(round(t / (log["t"][1] - log["t"][0]))) - 1
    assert 0.55 <= log["feet"][i, leg, 0] <= 0.63


def test_flat_ground_trot_holds_its_speed_without_tipping():
    schedule = lambda t: (0.0, 0.0, 0.0, STAND) if t < 1.0 else (0.3, 0.0, 0.0, TROT)
    log, ctrl = run_stack(schedule, 8.0)
    sel = log["t"] >= 5.0
    speed = log["vel"][sel, 0].mean()
    assert 0.27 < speed < 0.40                # README: the stack runs about 15 % fast
    assert np.degrees(np.abs(log["rpy"][:, :2])).max() < 10.0
    assert ctrl.wbc.failures == 0
