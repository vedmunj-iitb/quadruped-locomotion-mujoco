"""Measures shared by the demos. Written in 2026."""
import numpy as np


def fall_index(log, terrain=None, hold=0.5):
    """First sample of a fall, or None: the body tilts over 30 degrees or comes within 0.15 m of
    the ground under it, and stays so for `hold` seconds (or until the run ends). A tilt the robot
    recovers from is not a fall."""
    tilt = np.degrees(np.abs(log["rpy"][:, :2])).max(axis=1)
    ground = 0.0 if terrain is None else terrain.height(log["pos"][:, 0], log["pos"][:, 1])
    bad = (tilt > 30.0) | (log["pos"][:, 2] - ground < 0.15)
    n = int(round(hold / (log["t"][1] - log["t"][0])))
    i = 0
    while i < len(bad):
        if bad[i]:
            j = i
            while j < len(bad) and bad[j]:
                j += 1
            if j - i >= n or j == len(bad):
                return i
            i = j
        else:
            i += 1
    return None


def hole_drops(log, depth=0.10):
    """(time, leg) each time a foot ball centre goes below `depth` m: on the course that only
    happens in a hole (resting on the ground the centre is at 0.142 m)."""
    low = log["feet"][:, :, 2] < depth
    starts = np.argwhere(low[1:] & ~low[:-1])
    return [(log["t"][i + 1], leg) for i, leg in starts]
