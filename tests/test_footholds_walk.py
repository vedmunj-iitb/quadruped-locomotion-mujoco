"""The course in MuJoCo, with and without foothold selection. Written in 2026."""
from common.metrics import fall_index, hole_drops
from footholds.terrain import make_course
from footholds.walker import walk_course

TERRAIN = make_course()


def test_trot_with_selection_crosses_the_gap_and_the_step():
    log, ctrl = walk_course(seconds=6.0, gait="trot", speed=0.4, select=True, terrain=TERRAIN)
    assert fall_index(log, TERRAIN) is None
    assert not hole_drops(log)                # no foot went into a hole
    assert log["pos"][-1, 0] > 1.3            # past the first gap (0.55 m) and onto the step (1.05 m)


def test_trot_on_raibert_points_only_drops_a_foot_in_the_first_gap_and_falls():
    log, ctrl = walk_course(seconds=5.0, gait="trot", speed=0.4, select=False, terrain=TERRAIN)
    drops = hole_drops(log)
    assert drops, "no foot went into the gap"
    t_drop, leg = drops[0]
    i = int(round(t_drop / (log["t"][1] - log["t"][0]))) - 1
    assert 0.55 <= log["feet"][i, leg, 0] <= 0.63      # it was the first gap
    end = fall_index(log, TERRAIN)
    assert end is not None and log["pos"][end, 0] < 0.7
