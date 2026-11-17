"""Three more pictures for the README, filmed with fixed cameras. Written in 2026.

- media/stack_with_without_planner.gif: the full stack on the course, with the foothold planner
  (left) and with each foot on its first guess (right), side by side at half speed
- media/stack_still.png: one sharp still of the full stack crossing the first gap
- media/wbc_push_front.gif: the push runs of make_media.py seen from the front right, 4 times
  slower around each push

    python extra_media.py
"""
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # offscreen rendering: must be set before MuJoCo loads

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import go2
from common.video import save_gif
from footholds.terrain import make_course
from stack.demo import COURSE, draw, schedule_from, summary
from stack.stack import run_stack
from wbc import scenarios
from wbc.demo import force_arrows

FONT = ImageFont.load_default(size=15)
MEDIA = Path(__file__).resolve().parent / "media"


def camera(lookat, distance, azimuth, elevation):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    return cam


def write(frame, *texts):
    """Write white text on a frame. Each text is (x, y, words)."""
    img = Image.fromarray(frame)
    pen = ImageDraw.Draw(img)
    for x, y, words in texts:
        pen.text((x, y), words, fill="white", font=FONT)
    return np.asarray(img)


# ---------- the course, with and without the foothold planner ----------

START, END, RATE = 1.3, 4.2, 25          # 25 frames per simulated second, shown at 12.5: half speed
PANEL_W, PANEL_H = 398, 300


def film_course(model, terrain, select):
    """Run the full stack on the course and film it from beside the first gap."""
    renderer = mujoco.Renderer(model, PANEL_H, PANEL_W)
    side = camera((0.80, 0.0, 0.20), 2.1, 103, -19)
    times = START + np.arange(int(round((END - START) * RATE)) + 1) / RATE
    frames, state = [], {"extra": None, "still_done": not select}

    def hook(data, ctrl):
        if state["extra"] is None:
            state["extra"] = draw(ctrl)   # green discs: chosen spots, arrows: the MPC's pushes
        if not state["still_done"] and data.time >= 3.1 - 1e-9:
            big = mujoco.Renderer(model, 720, 1280)
            x, y = data.qpos[0], data.qpos[1]
            ground = float(terrain.height(x, 0.0))
            big.update_scene(data, camera((x + 0.05, y, ground + 0.17), 1.25, 125, -20))
            state["extra"](big.scene, data)
            Image.fromarray(big.render()).save(MEDIA / "stack_still.png")
            big.close()
            state["still_done"] = True
        if len(frames) < len(times) and data.time + 1e-9 >= times[len(frames)]:
            renderer.update_scene(data, side)
            state["extra"](renderer.scene, data)
            frames.append(renderer.render().copy())

    log, ctrl = run_stack(schedule_from(COURSE), END + 0.05, terrain=terrain, frame_hook=hook,
                          model=model, select=select)
    renderer.close()
    s = summary(log, ctrl, terrain)
    fell = f"fell at {s['fell_at']:.1f} s" if s["fell_at"] is not None else "no fall"
    holes = ", ".join(f"{t:.1f} s" for t in s["drops"]) or "none"
    moved = log["pos"][-1, 0] - log["pos"][0, 0]
    print(f"   planner {'on ' if select else 'off'}: {moved:.2f} m by {END} s, {fell}, "
          f"feet in a hole at: {holes}")
    return frames


def course_clip():
    terrain = make_course()
    model = go2.load_model(terrain.add_to_spec)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1280, 720
    left = film_course(model, terrain, select=True)
    right = film_course(model, terrain, select=False)
    bar = np.full((PANEL_H, 4, 3), 25, np.uint8)
    frames = []
    for a, b in zip(left, right):
        a = write(a, (8, 6, "with the foothold planner"))
        b = write(b, (8, 6, "without it: first guess only"), (PANEL_W - 128, PANEL_H - 22, "half speed"))
        frames.append(np.concatenate([a, bar, b], axis=1))
    frames += [frames[-1]] * 15                            # hold the last frame for 1.2 s
    save_gif(frames, MEDIA / "stack_with_without_planner.gif", fps=12.5)


# ---------- the push runs, seen from the front ----------

SAMPLES, SLOW = 50, 4    # 50 frames per simulated second near a push (4 times slower), else 12.5


def film_pushes(model, seconds, pushes, captions, keep):
    renderer = mujoco.Renderer(model, 480, 640)
    arrows = force_arrows(model)            # red: the push, green: the ground force on each foot
    front = camera((0.0, 0.0, 0.24), 1.4, 140, -35)   # front right, from above: shows every push arrow
    frames, state = [], {"next": 0.0, "count": 0}

    def hook(data, wbc):
        if data.time + 1e-9 < state["next"]:
            return
        t, count = state["next"], state["count"]
        state["next"] += 1.0 / SAMPLES
        state["count"] += 1
        near = [i for i, (start, _, _) in enumerate(pushes) if start - 0.1 <= t < start + 0.7]
        if not keep[0] <= t <= keep[1] or (not near and count % SLOW):
            return
        renderer.update_scene(data, front)
        arrows(renderer.scene, data)
        words = f"{captions[near[0]]}, {SLOW}x slower" if near else "red: push, green: ground forces"
        frames.append(write(renderer.render().copy(), (8, 6, words)))

    log = scenarios.simulate(seconds, scenarios.hold, pushes, frame_hook=hook, model=model)
    renderer.close()
    tilt = np.degrees(np.abs(log["rpy"][:, :2])).max()
    print(f"   {len(pushes)} push(es) in {seconds} s: largest tilt {tilt:.1f} degrees")
    return frames


def push_clip():
    model = go2.load_model()
    captions = [f"{np.linalg.norm(f):.0f} N {'sideways' if f[0] == 0 else 'forwards'}, 0.1 s"
                for _, _, f in scenarios.PUSHES]
    frames = film_pushes(model, scenarios.PUSH_SECONDS, scenarios.PUSHES, captions, (0.5, 7.5))
    strong = np.linalg.norm(scenarios.TOO_STRONG[0][2])
    frames += film_pushes(model, 3.0, scenarios.TOO_STRONG,
                          [f"{strong:.0f} N sideways: too strong, it tips over"], (0.5, 2.5))
    save_gif(frames, MEDIA / "wbc_push_front.gif", fps=12.5)


if __name__ == "__main__":
    print("== course, with and without the foothold planner")
    course_clip()
    print("== pushes, seen from the front right")
    push_clip()
