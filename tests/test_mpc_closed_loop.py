"""The whole controller in MuJoCo. Written in 2026."""
import numpy as np
import pytest

from mpc.controller import STAND, TROT, simulate


def run(vx=0.0, vy=0.0, wz=0.0, seconds=5.0):
    return simulate(lambda t: (0, 0, 0, STAND) if t < 1.0 else (vx, vy, wz, TROT), seconds)


def settled(log, after=3.0):
    return log["t"] >= after


def test_standing_feet_give_the_mpc_forces():
    """The stance torque -J' f must make the ground push back with f: checks the sign."""
    log = simulate(lambda t: (0, 0, 0, STAND), 1.0)
    sel = settled(log, 0.5)
    mpc, sim = log["forces"][sel].mean(axis=0), log["contact_forces"][sel].mean(axis=0)
    assert np.isclose(sim[:, 2].sum(), 15.2064 * 9.81, rtol=0.01)
    assert np.allclose(mpc, sim, atol=2.0)


@pytest.mark.parametrize("vx", [0.5, 1.0])
def test_trot_tracks_forward_speed(vx):
    log = run(vx=vx)
    sel = settled(log)
    assert abs(log["vel"][sel, 0].mean() - vx) < 0.1 * vx
    assert np.degrees(np.abs(log["rpy"][:, :2])).max() < 10.0
    assert log["pos"][:, 2].min() > 0.25


def test_trot_turns():
    log = run(vx=0.5, wz=0.8)
    assert abs(log["yaw_rate"][settled(log)].mean() - 0.8) < 0.05


def test_trot_steps_sideways():
    log = run(vy=0.3)
    assert abs(log["vel"][settled(log), 1].mean() - 0.3) < 0.1
