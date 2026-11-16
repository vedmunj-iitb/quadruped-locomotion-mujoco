"""Footstep placement (Raibert heuristic) and foothold selection on a heightmap.

Written in 2026.
"""
import numpy as np

WINDOW = 0.10        # search radius round the nominal foothold (m)
EDGE_SAFE = 0.05     # footholds closer than this to an edge are penalised (m)
EDGE_MIN = 0.03      # and closer than this are not allowed (the foot ball has a 2.2 cm radius)
HIP_REACH = 0.22     # largest horizontal distance from the hip at touchdown (m)
LEG_RANGE = (0.20, 0.36)   # allowed height of the hip above the foot ball centre (m)
FOOT_RADIUS = 0.022
WEIGHTS = dict(slope=1.0, edge=2.0, reach=1.0)


def raibert_foothold(hip, v_cmd, t_stance, v=None, k=0.03):
    """Nominal foothold (x, y): the Raibert heuristic.

    hip: where the hip will be at touchdown. The foot lands half a stance stride ahead of it, so
    the stance is centred under the hip. If the measured velocity v is above the command, the
    foot lands further ahead, which brakes the body (and the reverse).
    """
    p = np.asarray(hip) + np.asarray(v_cmd) * t_stance / 2
    if v is not None:
        p = p + k * (np.asarray(v) - np.asarray(v_cmd))
    return p


def foothold_costs(terrain, nominal, hip):
    """Cost terms for every grid point within WINDOW of the nominal foothold.

    hip: where the hip will be at touchdown (x, y, z). Returns the grid indices of the
    candidates, a dict of cost terms and a mask of the candidates the leg can use.
    """
    iy0, ix0 = terrain.index(*nominal)
    r = int(np.ceil(WINDOW / terrain.res))
    iy, ix = np.mgrid[iy0 - r: iy0 + r + 1, ix0 - r: ix0 + r + 1]
    inside = (iy >= 0) & (iy < terrain.ny) & (ix >= 0) & (ix < terrain.nx)
    iy, ix = iy[inside], ix[inside]
    x, y = terrain.point(iy, ix)
    dist = np.hypot(x - nominal[0], y - nominal[1])
    keep = dist <= WINDOW
    iy, ix, x, y, dist = iy[keep], ix[keep], x[keep], y[keep], dist[keep]

    edge = terrain.edge_dist[iy, ix]
    terms = {
        "slope": terrain.slope[iy, ix],                                  # rise over run
        "edge": np.clip((EDGE_SAFE - edge) / EDGE_SAFE, 0.0, 1.0),       # 0 far from edges, 1 at one
        "reach": dist / WINDOW,                                          # 0 at the nominal point
    }
    leg = hip[2] - terrain.h[iy, ix] - FOOT_RADIUS          # hip height above the foot ball
    feasible = ((edge >= EDGE_MIN)                          # off the edges
                & (leg >= LEG_RANGE[0]) & (leg <= LEG_RANGE[1])   # in reach, so not down a gap
                & (np.hypot(x - hip[0], y - hip[1]) <= HIP_REACH))
    return (iy, ix), terms, feasible


def total_cost(terms):
    return sum(WEIGHTS[k] * v for k, v in terms.items())


def select_foothold(terrain, nominal, hip):
    """Cheapest feasible foothold near the nominal one, as (x, y, height).

    If no candidate is feasible, the nominal foothold is returned unchanged.
    """
    (iy, ix), terms, feasible = foothold_costs(terrain, nominal, hip)
    if not feasible.any():
        return np.array([nominal[0], nominal[1], terrain.height(*nominal)])
    cost = np.where(feasible, total_cost(terms), np.inf)
    best = np.argmin(cost)
    x, y = terrain.point(iy[best], ix[best])
    return np.array([x, y, terrain.h[iy[best], ix[best]]])
