"""A heightmap course with gaps, a step, a ramp and potholes, and the maps the planner reads.

Written in 2026.
"""
import mujoco
import numpy as np
from scipy import ndimage

RES = 0.01               # grid spacing (m)
X_RANGE = (-0.6, 4.6)    # course extent along x and y (m)
Y_RANGE = (-0.6, 0.6)
GROUND = 0.12            # height of the normal ground; gaps go down to 0 (m)


class Terrain:
    def __init__(self, heights, x0=X_RANGE[0], y0=Y_RANGE[0], res=RES):
        self.h = np.asarray(heights, float)   # h[iy, ix]: rows along y, columns along x, as MuJoCo
        self.x0, self.y0, self.res = x0, y0, res
        self.ny, self.nx = self.h.shape
        # Edge cells: np.gradient takes central differences, so a cell is an edge when the height
        # changes by more than 4 cm between its two neighbours. Every step of the course is 5 cm
        # or more, so the cells on both sides of every edge are marked.
        gy, gx = np.gradient(self.h, res)
        edge = np.hypot(gx, gy) * res > 0.02
        self.edge_dist = ndimage.distance_transform_edt(~edge) * res    # distance to an edge (m)
        # Slope (rise over run) of the surface, smoothed over the foot's 5 cm footprint.
        gy, gx = np.gradient(ndimage.uniform_filter(self.h, size=5), res)
        self.slope = np.hypot(gx, gy)

    def index(self, x, y):
        """Nearest grid point to (x, y), clipped to the map."""
        ix = np.clip(np.rint((np.asarray(x) - self.x0) / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(np.rint((np.asarray(y) - self.y0) / self.res).astype(int), 0, self.ny - 1)
        return iy, ix

    def point(self, iy, ix):
        return self.x0 + ix * self.res, self.y0 + iy * self.res

    def height(self, x, y):
        return self.h[self.index(x, y)]

    def add_to_spec(self, spec):
        """Add the heightmap to a MuJoCo model as a heightfield with the same grid."""
        # MuJoCo rescales heightfield data to 0..1 (lowest to highest), so give it that range and
        # place the geom at the lowest height.
        low = self.h.min()
        span = max(self.h.max() - low, 1e-6)
        rx = (self.nx - 1) * self.res / 2
        ry = (self.ny - 1) * self.res / 2
        spec.add_hfield(name="course", size=[rx, ry, span, 0.05], nrow=self.ny, ncol=self.nx,
                        userdata=((self.h - low) / span).ravel().tolist())
        mat = spec.add_material(name="course", rgba=[0.55, 0.5, 0.42, 1.0])
        spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="course",
                                material=mat.name, pos=[self.x0 + rx, self.y0 + ry, low])


def make_course():
    """The course of the README: gap, step up, gap on the step, ramp down, potholes."""
    x = X_RANGE[0] + RES * np.arange(int(round((X_RANGE[1] - X_RANGE[0]) / RES)) + 1)
    y = Y_RANGE[0] + RES * np.arange(int(round((Y_RANGE[1] - Y_RANGE[0]) / RES)) + 1)
    X, Y = np.round(np.meshgrid(x, y), 6)      # rounded, so that 1.45 >= 1.45 holds on the grid
    h = np.full(X.shape, GROUND)
    h[(X >= 0.55) & (X < 0.63)] = 0.0                          # gap, 8 cm
    h[(X >= 1.05) & (X < 1.90)] = GROUND + 0.05                # step up 5 cm
    h[(X >= 1.45) & (X < 1.52)] = 0.0                          # gap on the step, 7 cm
    ramp = (X >= 1.90) & (X < 2.20)                            # ramp down, 9.5 degrees
    h[ramp] = GROUND + 0.05 * (2.20 - X[ramp]) / 0.30
    for cx, cy in [(2.45, 0.13), (2.62, -0.15), (2.80, 0.14), (2.95, -0.12)]:   # potholes, 7 cm
        h[(abs(X - cx) < 0.035) & (abs(Y - cy) < 0.035)] = 0.0    # 7 x 7 grid points
    return Terrain(h)
