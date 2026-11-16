"""Phase-based gait scheduler and Bezier swing curves.

Written in 2026.
"""
from dataclasses import dataclass
from math import comb

import numpy as np


@dataclass(frozen=True)
class Gait:
    name: str
    period: float   # one full cycle (s)
    duty: float     # fraction of the cycle a foot is on the ground
    offset: tuple   # phase offset of FL, FR, RL, RR (fraction of a cycle)


# A leg is on the ground while (phase + offset) mod 1 < duty.
GAITS = {
    "trot": Gait("trot", 0.40, 0.50, (0.0, 0.5, 0.5, 0.0)),        # diagonal pairs together
    "walk": Gait("walk", 1.00, 0.75, (0.5, 0.0, 0.75, 0.25)),      # one leg at a time: RL, FL, RR, FR
    "pace": Gait("pace", 0.25, 0.50, (0.0, 0.5, 0.0, 0.5)),        # left pair, then right pair
    "bound": Gait("bound", 0.30, 0.50, (0.0, 0.0, 0.5, 0.5)),      # front pair, then rear pair
}


class GaitScheduler:
    """Keeps the gait clock and says, for each leg, stance or swing and how far through it."""

    def __init__(self, gait):
        self.gait = gait
        self.t = 0.0

    def advance(self, dt):
        self.t += dt

    def leg_phase(self, leg, t=None):
        t = self.t if t is None else t
        return (t / self.gait.period + self.gait.offset[leg]) % 1.0

    def in_stance(self, leg, t=None):
        return self.leg_phase(leg, t) < self.gait.duty

    def progress(self, leg, t=None):
        """How far through its current stance or swing the leg is, from 0 to 1."""
        p, duty = self.leg_phase(leg, t), self.gait.duty
        return p / duty if p < duty else (p - duty) / (1.0 - duty)

    def contact_plan(self, horizon, dt):
        """Planned stance (True) of each leg at the next `horizon` steps of length dt."""
        return np.array([[self.in_stance(leg, self.t + k * dt) for leg in range(4)]
                         for k in range(horizon)])


def bezier(points, s):
    """Point at s (0 to 1) on the Bezier curve with these control points (Bernstein form)."""
    n = len(points) - 1
    w = np.array([comb(n, k) * s**k * (1 - s)**(n - k) for k in range(n + 1)])
    return w @ points


def bezier_velocity(points, s, duration):
    """d/dt of the curve when s goes from 0 to 1 in `duration` seconds.

    The derivative of a degree n Bezier curve is a degree n-1 Bezier curve with control points
    n (P[k+1] - P[k]).
    """
    n = len(points) - 1
    return n * bezier(np.diff(points, axis=0), s) / duration


def bezier_acceleration(points, s, duration):
    """d2/dt2 of the curve: a degree n-2 Bezier curve with control points
    n (n - 1) (P[k+2] - 2 P[k+1] + P[k]), divided by duration squared."""
    n = len(points) - 1
    return n * (n - 1) * bezier(np.diff(points, 2, axis=0), s) / duration**2


def swing_curve(start, end, height, foot_velocity, duration):
    """Six control points of a swing from `start` (lift-off) to `end` (touchdown).

    At both ends the foot moves with `foot_velocity`, the velocity a planted foot has relative to
    the body, so the swing joins the stance before and after it without a jump in speed. The
    height profile 10 s^2 (1-s)^2 peaks at s = 0.5 with 0.625 times its control point height.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    d = np.asarray(foot_velocity, float) * duration / 5    # end slope of a degree 5 curve: 5 (P1 - P0)
    d[2] = 0.0                                             # lift off and land with no vertical speed
    pts = np.array([start, start + d, start + d, end - d, end - d, end])
    pts[2:4, 2] += height / 0.625
    return pts
