"""WBC stage scenarios: body pose paths, pushes, the three-legged stand, the simulation loop and
the error measures used by the demo and the tests.

Written in 2026.
"""
import time

import mujoco
import numpy as np

from common import go2
from wbc.wbc import WholeBodyQP, foot_forces, stand, target_at

HEIGHT = 0.30                          # body height to hold (m)
MOVES = (("roll", np.radians(15)), ("pitch", np.radians(15)), ("yaw", np.radians(20)),
         ("height", 0.06), ("circle", np.radians(10)))
SEGMENT = 4.0                          # seconds per move: two cycles of 2 s
POSE_SECONDS = 1.0 + SEGMENT * len(MOVES) + 1.0
PUSHES = ((1.0, 0.1, (0.0, 80.0, 0.0)),     # (start s, duration s, force N, world frame)
          (3.5, 0.1, (150.0, 0.0, 0.0)),     # +y pushes the robot to its left, +x forwards
          (6.0, 0.1, (0.0, 120.0, 0.0)))
PUSH_SECONDS = 8.5
TOO_STRONG = ((1.0, 0.1, (0.0, 150.0, 0.0)),)    # sideways: tips it over

# Three-legged stand: the body leans towards the rear right over 1 s, the front left foot lifts
# 10 cm in 0.5 s, draws two vertical circles of 4 cm radius (1.5 s each), comes down in 0.5 s,
# and the body leans back.
LEAN = np.array([-0.04, -0.04, 0.0])
SWING_LEG = 0
T_LEAN, T_LIFT, T_CIRCLE, LIFT, RADIUS = 1.0, 0.5, 1.5, 0.10, 0.04
T_UP = 1.5                                        # the foot leaves the ground at 1.5 s
T_DOWN = T_UP + T_LIFT + 2 * T_CIRCLE             # starts down at 5.0 s
T_LANDED = T_DOWN + T_LIFT                        # back on the ground at 5.5 s
SWING_SECONDS = T_LANDED + 1.5


def smooth(s):
    """Smoothstep 3s^2 - 2s^3 on [0, 1] and its first two derivatives."""
    s = np.clip(s, 0.0, 1.0)
    return 3 * s**2 - 2 * s**3, 6 * s - 6 * s**2, 6 - 12 * s


def hold(t):
    """Stand still: body at HEIGHT, level."""
    return (0.0, 0.0, HEIGHT), (0.0, 0.0, 0.0)


def dance(t):
    """Body targets for the pose demo: one move every SEGMENT seconds after 1 s of standing.

    Each move is a 2 s sine, faded in and out over 0.5 s (smoothstep) so the target starts and
    ends at rest.
    """
    pos, rpy = np.array([0.0, 0.0, HEIGHT]), np.zeros(3)
    k, s = int((t - 1.0) // SEGMENT), (t - 1.0) % SEGMENT
    if t < 1.0 or k >= len(MOVES):
        return pos, rpy
    name, amplitude = MOVES[k]
    ramp = min(s, SEGMENT - s, 0.5) / 0.5
    a = amplitude * ramp * ramp * (3 - 2 * ramp)
    wave = 2 * np.pi * s / 2.0
    if name == "roll":
        rpy[0] = a * np.sin(wave)
    elif name == "pitch":
        rpy[1] = a * np.sin(wave)
    elif name == "yaw":
        rpy[2] = a * np.sin(wave)
    elif name == "height":
        pos[2] += a * np.sin(wave)
    else:                                   # circle: the body's top traces a cone
        rpy[0], rpy[1] = a * np.sin(wave), a * np.cos(wave)
    return pos, rpy


def lean(t):
    """Body target for the three-legged stand: lean over the three stance feet and back."""
    s_in = smooth(t / T_LEAN)[0]
    s_out = smooth((t - T_LANDED - 0.2) / T_LEAN)[0]
    return np.array([0.0, 0.0, HEIGHT]) + LEAN * (s_in - s_out), (0.0, 0.0, 0.0)


def swing_target(t, p0):
    """Target (p, v, a) of the swinging foot's ball centre at time t, world frame. p0 is where
    the foot stood."""
    top = p0 + [0.0, 0.0, LIFT]
    if t < T_UP + T_LIFT:                          # up
        s, ds, dds = smooth((t - T_UP) / T_LIFT)
        d = top - p0
        return p0 + s * d, ds / T_LIFT * d, dds / T_LIFT**2 * d
    if t < T_DOWN:                                 # circles, starting and ending at the top
        w = 2 * np.pi / T_CIRCLE
        th = w * (t - T_UP - T_LIFT)
        centre = top - [0.0, 0.0, RADIUS]
        p = centre + RADIUS * np.array([np.sin(th), 0.0, np.cos(th)])
        v = RADIUS * w * np.array([np.cos(th), 0.0, -np.sin(th)])
        a = -RADIUS * w**2 * np.array([np.sin(th), 0.0, np.cos(th)])
        return p, v, a
    s, ds, dds = smooth((t - T_DOWN) / T_LIFT)    # down to where it stood
    d = p0 - top
    return top + s * d, ds / T_LIFT * d, dds / T_LIFT**2 * d


def static_forces(model, data, stance):
    """Ground forces that hold the robot still on the stance feet: vertical, sharing its weight
    so that the moments about its centre of mass cancel. Used as a stand-in for an MPC's f_des."""
    feet = data.geom_xpos[go2.foot_geoms(model)]
    com = data.subtree_com[1]
    weight = model.body_subtreemass[1] * 9.81
    legs = np.flatnonzero(stance)
    A = np.vstack([np.ones(len(legs)), feet[legs, 0] - com[0], feet[legs, 1] - com[1]])
    fz = np.linalg.lstsq(A, [weight, 0.0, 0.0], rcond=None)[0]   # exact for 3 feet
    f = np.zeros((4, 3))
    f[legs, 2] = fz
    return f


def push_at(pushes, t):
    """Total push force (world frame) at time t."""
    force = np.zeros(3)
    for start, duration, f in pushes:
        if start <= t < start + duration:
            force += f
    return force


def simulate(seconds, path, pushes=(), frame_hook=None, model=None, plan=None, data=None,
             **wbc_settings):
    """Stand the robot up, then run the WBC for `seconds`. Returns the logged signals.

    path(t) gives the body target. A push acts on the trunk at its centre of mass.
    plan(t, data), if given, returns the other inputs of WholeBodyQP.torques as a dict
    (stance, f_des, swing). Without it all four feet stand and only the body target is used.
    frame_hook(data, wbc), if given, is called after every physics step (to record video or
    update the viewer).
    """
    model = go2.load_model() if model is None else model
    data = mujoco.MjData(model) if data is None else data
    stand(model, data, HEIGHT)
    wbc = WholeBodyQP(model, **wbc_settings)
    names = ("t", "pos", "rpy", "pos_ref", "rpy_ref", "f_cmd", "f_meas", "f_des", "feet", "slip",
             "swing_ref", "tau", "push", "qp_ms", "step_ms")
    log = {k: [] for k in names}
    while data.time < seconds - 1e-9:
        t = data.time
        data.xfrc_applied[1, :3] = push_at(pushes, t)
        inputs = plan(t, data) if plan is not None else {}
        start = time.perf_counter()
        data.ctrl[:] = wbc.torques(data, base=target_at(path, t), **inputs)
        log["step_ms"].append(1e3 * (time.perf_counter() - start))
        log["qp_ms"].append(1e3 * wbc.solve_time)
        log["f_cmd"].append(wbc.f.reshape(4, 3).copy())
        f_des = inputs.get("f_des")
        log["f_des"].append(np.full((4, 3), np.nan) if f_des is None else np.array(f_des))
        swing = inputs.get("swing") or {}
        log["swing_ref"].append(swing[SWING_LEG][0] if SWING_LEG in swing else np.full(3, np.nan))
        log["push"].append(data.xfrc_applied[1, :3].copy())
        mujoco.mj_step(model, data)
        mujoco.mj_fwdPosition(model, data)    # log positions of the new state, not the old one
        ref_pos, ref_rpy = path(data.time)
        log["t"].append(data.time)
        log["pos"].append(data.qpos[:3].copy())
        log["rpy"].append(go2.body_rpy(data))
        log["pos_ref"].append(np.array(ref_pos, float))
        log["rpy_ref"].append(np.array(ref_rpy, float))
        log["f_meas"].append(foot_forces(model, data, wbc.foot_geoms))
        log["feet"].append(data.geom_xpos[wbc.foot_geoms].copy())
        log["slip"].append(slip_speed(model, data, wbc.foot_geoms, wbc.calves))
        log["tau"].append(data.ctrl.copy())
        if frame_hook is not None:
            frame_hook(data, wbc)
    log = {k: np.array(v) for k, v in log.items()}
    log["failures"] = wbc.failures
    return log


def three_leg_plan(model, use_forces=False):
    """plan(t, data) for the three-legged stand: the swing leg's stance flag, its target path,
    and (optionally) static desired forces for the other three feet."""
    p0 = {}

    def plan(t, data):
        up = T_UP <= t < T_LANDED
        stance = [not (up and leg == SWING_LEG) for leg in range(4)]
        inputs = {"stance": stance}
        if up:
            if not p0:
                p0["p"] = data.geom_xpos[go2.foot_geoms(model)[SWING_LEG]].copy()
            inputs["swing"] = {SWING_LEG: swing_target(t, p0["p"])}
        if use_forces:
            inputs["f_des"] = static_forces(model, data, stance)
        return inputs
    return plan


def slip_speed(model, data, geoms, bodies):
    """How fast each foot slides on the floor (m/s): the horizontal speed of the foot's own
    material at the contact point. A ball that rolls without sliding gives zero."""
    speed = np.zeros(4)
    jac = np.zeros((3, model.nv))
    for k in range(data.ncon):
        c = data.contact[k]
        for leg, g in enumerate(geoms):
            if g in (c.geom1, c.geom2):
                mujoco.mj_jac(model, data, jac, None, c.pos, bodies[leg])
                speed[leg] = max(speed[leg], np.linalg.norm((jac @ data.qvel)[:2]))
    return speed


def pose_errors(log):
    """Largest and RMS tracking error during each move (degrees, or mm for the height)."""
    out = {}
    for k, (name, _) in enumerate(MOVES):
        sel = (log["t"] >= 1.0 + k * SEGMENT) & (log["t"] < 1.0 + (k + 1) * SEGMENT)
        if name == "height":
            e = 1e3 * (log["pos"][sel, 2] - log["pos_ref"][sel, 2])
        else:
            d = log["rpy"][sel] - log["rpy_ref"][sel]
            d = np.degrees((d + np.pi) % (2 * np.pi) - np.pi)
            e = np.linalg.norm(d[:, :2], axis=1) if name == "circle" else d[:, "rpy".index(name[0])]
        out[name] = (np.abs(e).max(), np.sqrt(np.mean(e**2)))
    return out


def swing_errors(log):
    """Foot tracking error while the leg is in the air (mm: largest, RMS), and the largest body
    position (mm) and angle (deg) errors over the whole run."""
    sel = (log["t"] > T_UP) & (log["t"] < T_LANDED)
    # The target is logged before the step and the foot after it, so compare with the next target.
    e = 1e3 * np.linalg.norm(log["feet"][sel, SWING_LEG][:-1] - log["swing_ref"][sel][1:], axis=1)
    body = 1e3 * np.linalg.norm(log["pos"] - log["pos_ref"], axis=1).max()
    ang = np.degrees(np.abs(log["rpy"] - log["rpy_ref"])).max()
    return e.max(), np.sqrt(np.mean(e**2)), body, ang


def push_response(log, start, duration, window=2.5, pos_tol=0.005, angle_tol=1.0):
    """Peak displacement (m), peak tilt (deg), recovery time (s) and the longest distance a
    foot slid on the floor (m), in the `window` seconds from the start of a push.

    Recovery: from the end of the push until the body stays within pos_tol of its target and
    within angle_tol of its target orientation. NaN if it does not within the window.
    """
    t = log["t"]
    i0 = np.searchsorted(t, start)
    i1 = np.searchsorted(t, start + window)
    t, win = t[:i1], slice(i0, i1)
    disp = np.linalg.norm(log["pos"][win] - log["pos"][i0 - 1], axis=1)
    tilt = np.degrees(np.abs(log["rpy"][win, :2])).max()
    err = np.linalg.norm(log["pos"][:i1] - log["pos_ref"][:i1], axis=1)
    ang = np.degrees(np.abs(log["rpy"][:i1] - log["rpy_ref"][:i1]).max(axis=1))
    after = t >= start + duration
    bad = np.where(after & ~((err < pos_tol) & (ang < angle_tol)))[0]
    if len(bad) and bad[-1] == len(t) - 1:
        recovery = np.nan
    else:
        recovery = t[bad[-1] + 1] - (start + duration) if len(bad) else 0.0
    dt = log["t"][1] - log["t"][0]
    slide = (log["slip"][win] * dt).sum(axis=0).max()
    return disp.max(), tilt, recovery, slide
