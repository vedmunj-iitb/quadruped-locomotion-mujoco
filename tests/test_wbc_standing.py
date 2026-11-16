"""Closed loop in MuJoCo: pose targets, pushes, desired forces and a swinging leg. Written in 2026."""
import numpy as np

from common import go2
from wbc import scenarios as run


def test_holds_a_pitch_and_height_target():
    path = lambda t: ((0.0, 0.0, 0.27), (0.0, np.radians(10), 0.0))
    log = run.simulate(1.5, path)
    assert abs(np.degrees(log["rpy"][-1, 1]) - 10.0) < 0.5
    assert abs(log["pos"][-1, 2] - 0.27) < 0.002


def test_pose_demo_tracks_within_two_degrees():
    log = run.simulate(run.POSE_SECONDS, run.dance)
    for name, (peak, _) in run.pose_errors(log).items():
        assert peak < (5.0 if name == "height" else 2.0), name    # mm for height, deg otherwise
    assert log["failures"] == 0


def test_recovers_from_a_sideways_push():
    log = run.simulate(2.0, run.hold, ((0.5, 0.1, (0.0, 80.0, 0.0)),))
    peak, tilt, recovery, slide = run.push_response(log, 0.5, 0.1, window=1.5)
    assert recovery < 0.5 and slide < 0.005 and tilt < 2.0


def test_too_strong_push_tips_it_over():
    """150 N from the side for 0.1 s is past what standing alone can take."""
    log = run.simulate(2.5, run.hold, run.TOO_STRONG)
    assert np.degrees(np.abs(log["rpy"][-1, :2])).max() > 45.0


def test_follows_desired_forces_while_standing():
    """f_des = m g / 4 on every foot: the QP's forces stay within 3 N of it and the body stays
    within 1 cm and 1 degree of its target."""
    model = go2.load_model()
    weight = model.body_subtreemass[1] * 9.81
    plan = lambda t, data: {"f_des": np.tile([0.0, 0.0, weight / 4], (4, 1))}
    log = run.simulate(2.0, run.hold, model=model, plan=plan)
    late = log["t"] >= 0.5
    assert np.abs(log["f_cmd"][late] - log["f_des"][late]).max() < 3.0
    assert np.linalg.norm(log["pos"] - log["pos_ref"], axis=1)[late].max() < 0.01
    assert np.degrees(np.abs(log["rpy"][late, :2])).max() < 1.0


def test_three_feet_and_a_swinging_leg():
    """The front left foot follows its path within 1 cm while the body holds its target."""
    model = go2.load_model()
    log = run.simulate(run.SWING_SECONDS, run.lean, model=model, plan=run.three_leg_plan(model))
    peak, rms, body, angle = run.swing_errors(log)
    assert peak < 10.0 and rms < 4.0 and body < 5.0 and angle < 2.0
    assert np.degrees(np.abs(log["rpy"][-1, :2])).max() < 1.0 and log["failures"] == 0
