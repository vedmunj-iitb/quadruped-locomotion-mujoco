"""Walking the course in MuJoCo with the gait stage's joint PD controller, with every
touchdown point planned on the heightmap. This is the foothold stage's own demo. The full stack
(stack/) uses the same planner with the MPC and the WBC.

Written in 2026.
"""
import mujoco
import numpy as np

from common import go2
from footholds.planner import raibert_foothold, select_foothold
from footholds.terrain import make_course
from gait.joint_pd import GaitController
from gait.scheduler import bezier, swing_curve

BODY_HEIGHT = 0.30   # body origin above the mean height of the four footholds (m)


class TerrainWalker(GaitController):
    """GaitController whose feet land where the planner says, at the terrain's height."""

    def __init__(self, model, terrain, gait="walk", select=True, **kw):
        super().__init__(model, gait, height=BODY_HEIGHT, **kw)
        self.terrain = terrain
        self.select = select           # False: land on the Raibert point, whatever is there
        self.ground = None             # terrain height under each planted foot (m)
        self.landing = np.zeros(4)     # terrain height where each swinging foot will land (m)
        self.v_avg = np.zeros(2)       # body velocity (world x, y), low-pass filtered (one gait cycle)
        self.steps = []                # (time, leg, nominal xy, chosen xyz) of every step

    def reset(self, data, x=0.0):
        super().reset(data)
        data.qpos[0] = x
        data.qpos[2] += self.terrain.height(x, 0.0)
        mujoco.mj_forward(self.model, data)
        feet = data.geom_xpos[self.foot_geoms]
        self.ground = np.array([self.terrain.height(f[0], f[1]) for f in feet])

    def torques(self, data, dt):
        self.data = data               # update_feet needs the body's pose and velocity
        self.v_avg += (data.qvel[:2] - self.v_avg) * dt / self.sched.gait.period
        return super().torques(data, dt)

    def stance_height(self, ground):
        """Foot target height in the body frame for a foot on `ground`, with the body
        BODY_HEIGHT above the mean ground under the four feet."""
        return self.home[0, 2] + ground - self.ground.mean()

    def plan_step(self, leg, v_cmd, t_stance, t_swing):
        """World foothold for a leg that lifts off now, and its target in the heading (yaw-only)
        frame of the body."""
        d = self.data
        turn = go2.rot_z(go2.body_rpy(d)[2])
        # Where the body will be at touchdown, without the walk's sideways lean.
        body_then = d.qpos[:3] - turn @ self.shift
        body_then[:2] += self.v_avg * t_swing
        hip_then = body_then + turn @ (self.home[leg] * [1, 1, 0])
        hip_then[2] = self.ground.mean() + BODY_HEIGHT
        nominal = raibert_foothold(hip_then[:2], (turn @ v_cmd)[:2], t_stance,
                                   v=self.v_avg)
        if self.select:
            chosen = select_foothold(self.terrain, nominal, hip_then)
        else:
            chosen = np.array([*nominal, self.terrain.height(*nominal)])
        self.steps.append((d.time, leg, nominal, chosen))
        target = turn.T @ (np.array([chosen[0], chosen[1], 0.0]) - body_then)
        return chosen, target

    def update_feet(self, v, w, dt):
        self.sched.advance(dt)
        g = self.sched.gait
        t_stance, t_swing = g.duty * g.period, (1.0 - g.duty) * g.period
        for leg in range(4):
            stance = self.sched.in_stance(leg)
            planted = -(v + np.cross(w, self.feet[leg]))
            if stance:
                if not self.was_stance[leg]:                         # touchdown
                    self.ground[leg] = self.landing[leg]
                self.feet[leg] += planted * dt
                self.feet[leg, 2] = self.stance_height(self.ground[leg])
            else:
                if self.was_stance[leg]:                             # lift-off: plan the step
                    chosen, end = self.plan_step(leg, v, t_stance, t_swing)
                    self.landing[leg] = chosen[2]
                    end[2] = self.stance_height(chosen[2])
                    self.swing[leg] = swing_curve(self.feet[leg], end, self.step_height,
                                                  planted, t_swing)
                self.feet[leg] = bezier(self.swing[leg], self.sched.progress(leg))
            self.was_stance[leg] = stance


def walk_course(seconds=18.0, select=True, gait="walk", speed=0.2, frame_hook=None, model=None,
                terrain=None, on_start=None, data=None, **kw):
    """Walk the course and return the log and the controller (for its list of steps).

    on_start(controller), if given, is called once before the first step (to set up drawing).
    """
    terrain = make_course() if terrain is None else terrain
    model = go2.load_model(terrain.add_to_spec) if model is None else model
    data = mujoco.MjData(model) if data is None else data
    sway = 0.6 if gait == "walk" else 0.0
    ctrl = TerrainWalker(model, terrain, gait, select=select, speed=speed, sway=sway, **kw)
    ctrl.reset(data)
    if on_start is not None:
        on_start(ctrl)
    dt = model.opt.timestep
    log = {k: [] for k in ("t", "pos", "rpy", "feet", "contact")}
    for _ in range(int(round(seconds / dt))):
        data.ctrl[:] = ctrl.torques(data, dt)
        mujoco.mj_step(model, data)
        mujoco.mj_fwdPosition(model, data)    # positions and contacts of the new state
        log["t"].append(data.time)
        log["pos"].append(data.qpos[:3].copy())
        log["rpy"].append(go2.body_rpy(data))
        log["feet"].append(data.geom_xpos[ctrl.foot_geoms].copy())
        log["contact"].append(go2.feet_in_contact(model, data, ctrl.foot_geoms))
        if frame_hook is not None:
            frame_hook(data)
    return {k: np.array(v) for k, v in log.items()}, ctrl
