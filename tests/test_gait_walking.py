"""Each gait in MuJoCo: the robot stays up and moves forward. Written in 2026."""
import numpy as np
import pytest

from gait.demo import DEFAULTS
from gait.joint_pd import simulate, steady_state


@pytest.mark.parametrize("name", DEFAULTS)
def test_gait_walks(name):
    settings = DEFAULTS[name]
    log = simulate(name, 6.0, **settings)
    speed, _ = steady_state(log, after=3.0)
    assert np.degrees(np.abs(log["rpy"][:, :2])).max() < 15.0     # never tips far
    assert log["pos"][:, 2].min() > 0.2                             # body never drops
    assert speed > 0.5 * settings["speed"]
    assert abs(log["pos"][-1, 1]) < 0.15                            # stays near its line


def test_trot_turns_at_the_commanded_rate():
    log = simulate("trot", 6.0, speed=0.4, turn=0.5)
    assert abs(steady_state(log, after=3.0)[1] - 0.5) < 0.05
