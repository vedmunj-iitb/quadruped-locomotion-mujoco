"""Offscreen rendering of a MuJoCo run to GIF. Written in 2026."""
import os
import subprocess

# EGL renders on the GPU without a window. MuJoCo reads this only once, when it is first imported,
# so make_media.py and the demos (with --gif) set it before importing anything else.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np


class Recorder:
    """Renders a frame every 1/fps seconds of simulated time, with the camera following the body."""

    def __init__(self, model, width=640, height=360, fps=20, distance=1.4, azimuth=135,
                 elevation=-18, start=0.0, end=np.inf):
        self.renderer = mujoco.Renderer(model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.fps, self.start, self.end = fps, start, end
        self.next_time = start
        self.frames = []
        self.extra = None   # optional callback(scene, data) that adds shapes, such as force arrows

    def __call__(self, data):
        if data.time + 1e-9 < self.next_time or data.time > self.end:
            return
        self.next_time += 1.0 / self.fps
        self.cam.lookat[:] = [data.qpos[0] + 0.05, data.qpos[1], 0.25]
        self.renderer.update_scene(data, self.cam)
        if self.extra is not None:
            self.extra(self.renderer.scene, data)
        self.frames.append(self.renderer.render().copy())


def save_gif(frames, path, fps=20, colors=128):
    """Write RGB frames to a GIF with one shared palette (ffmpeg palettegen), which keeps it small."""
    h, w, _ = frames[0].shape
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-vf", f"split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
           "-loop", "0", str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(np.array(frames, dtype=np.uint8)).tobytes(),
                   check=True)


class WindowClosed(Exception):
    """The viewer window was closed, so the run stops."""


def viewer_hook(model, data, extra=None):
    """Open the MuJoCo viewer on data. Returns (hook, close).

    hook(data, *rest) redraws at 50 Hz and waits, so the run plays at about real time. If given,
    extra(scene, data, *rest) adds shapes (footholds, force arrows) to the viewer before each
    redraw, the same ones the GIFs show. When the window is closed, hook raises WindowClosed.
    close() shuts the viewer.
    """
    import time
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    start = time.time()

    def hook(d, *rest):
        if not viewer.is_running():
            raise WindowClosed
        if round(d.time / model.opt.timestep) % 10 == 0:
            if extra is not None:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    extra(viewer.user_scn, d, *rest)
            viewer.sync()
            time.sleep(max(0.0, d.time - (time.time() - start)))

    def close():
        viewer.close()
        time.sleep(0.5)   # let the viewer's thread finish first (MuJoCo 4.0.1 crashed otherwise)
    return hook, close
