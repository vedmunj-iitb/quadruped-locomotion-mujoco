"""Unitree Go2 model from MuJoCo Menagerie, and closed-form leg kinematics.

Written in 2026.
"""
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "third_party/mujoco_menagerie/unitree_go2/scene.xml"

LEGS = ("FL", "FR", "RL", "RR")  # front left, front right, rear left, rear right (the model's order)
HIP = np.array([[0.1934, 0.0465, 0.0],    # abduction joint of each leg in the body frame (m)
                [0.1934, -0.0465, 0.0],
                [-0.1934, 0.0465, 0.0],
                [-0.1934, -0.0465, 0.0]])
SIDE = np.array([1.0, -1.0, 1.0, -1.0])   # +1 for a left leg, -1 for a right leg
L_HIP = 0.0955                            # abduction joint to thigh joint, sideways (m)
L_THIGH = 0.213                           # thigh joint to knee (m)
CALF = np.array([-0.002, 0.0, -0.213])    # knee to the centre of the foot ball, calf frame (m)
L_CALF = np.hypot(CALF[0], CALF[2])
KNEE_OFFSET = np.arctan2(-CALF[0], -CALF[2])  # the ball sits 2 mm behind the calf axis
FOOT_RADIUS = 0.022


def load_model(spec_edit=None):
    """Compile the Menagerie scene with a plainer floor (25 cm squares, no mirror effect), which
    keeps the GIFs small.

    spec_edit, if given, is called with the MjSpec before compiling (to add terrain, for example).
    """
    spec = mujoco.MjSpec.from_file(str(SCENE))
    floor = spec.texture("groundplane")
    floor.rgb1, floor.rgb2, floor.markrgb = [0.20, 0.27, 0.34], [0.17, 0.23, 0.30], [0.50, 0.56, 0.62]
    spec.material("groundplane").texrepeat = [2, 2]
    spec.material("groundplane").reflectance = 0.0
    sky = next(t for t in spec.textures if t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    sky.rgb1, sky.rgb2 = [0.35, 0.45, 0.55], [0.10, 0.14, 0.18]
    if spec_edit is not None:
        spec_edit(spec)
    return spec.compile()


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def foot_position(leg, q):
    """Centre of the foot ball in the body frame, for joint angles q = (abduction, hip, knee)."""
    q1, q2, q3 = q
    to_thigh = np.array([0.0, SIDE[leg] * L_HIP, 0.0])
    in_plane = rot_y(q2) @ (np.array([0.0, 0.0, -L_THIGH]) + rot_y(q3) @ CALF)
    return HIP[leg] + rot_x(q1) @ (to_thigh + in_plane)


def leg_ik(leg, p):
    """Joint angles (abduction, hip, knee) that put the foot ball centre at p (body frame).

    The knee is always bent backwards, as on the robot. A point out of reach gives the angles of
    the straight leg pointing at it (link lengths only, joint limits are not applied).
    """
    x, y, z = p - HIP[leg]
    d = SIDE[leg] * L_HIP
    # Abduction. Seen from the front, the thigh and calf swing in a plane L_HIP to the side of the
    # abduction axis, and the foot is zp below the thigh joint in that plane. Turning by q1 about x
    # maps (d, zp) onto (y, z).
    zp = -np.sqrt(max(y * y + z * z - L_HIP**2, 1e-9))
    q1 = np.arctan2(z, y) - np.arctan2(zp, d)
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    # Hip and knee: a two-link arm in the leg plane reaching (x, zp), law of cosines.
    c3 = (x * x + zp * zp - L_THIGH**2 - L_CALF**2) / (2 * L_THIGH * L_CALF)
    q3 = -np.arccos(np.clip(c3, -1.0, 1.0))
    a = L_THIGH + L_CALF * np.cos(q3)
    b = L_CALF * np.sin(q3)
    q2 = np.arctan2(-x, -zp) - np.arctan2(b, a)
    return np.array([q1, q2, q3 - KNEE_OFFSET])


def body_rpy(data):
    """Roll, pitch and yaw of the body (Z-Y-X order) from its rotation matrix."""
    R = data.xmat[1].reshape(3, 3)
    return np.array([np.arctan2(R[2, 1], R[2, 2]), -np.arcsin(np.clip(R[2, 0], -1, 1)),
                     np.arctan2(R[1, 0], R[0, 0])])


def foot_geoms(model):
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in LEGS]


def feet_in_contact(model, data, geoms):
    """Which feet touch something right now (from MuJoCo's contact list)."""
    pairs = data.contact.geom[:data.ncon]
    return np.array([np.any(pairs == g) for g in geoms])
