"""Gait stage demo: one gait in the MuJoCo viewer or to a GIF, and the gait stage's media.

Written in 2026.

    python -m gait.demo trot                     # viewer, trot at 0.5 m/s
    python -m gait.demo walk --speed 0.15 --turn 0.3
    python -m gait.demo bound --gif bound.gif    # no window, writes a GIF
"""
import os
import sys

if "--gif" in sys.argv:   # a GIF renders offscreen: pick EGL before MuJoCo is first imported
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
from pathlib import Path

import numpy as np

from common import go2
from gait.joint_pd import GaitController, simulate, steady_state
from gait.scheduler import GAITS, swing_curve

MEDIA = Path(__file__).resolve().parents[1] / "media"
DEFAULTS = {"walk": dict(speed=0.2, sway=0.6), "trot": dict(speed=0.5),
            "pace": dict(speed=0.3), "bound": dict(speed=0.4)}
COLORS = {"FL": "#1f77b4", "FR": "#ff7f0e", "RL": "#2ca02c", "RR": "#d62728"}


def measure(log):
    """Mean forward speed, largest roll and pitch and how often the feet matched the plan."""
    sel = log["t"] >= 4.0
    rpy = log["rpy"][sel]
    return dict(speed=steady_state(log)[0],
                roll=np.degrees(np.abs(rpy[:, 0]).max()),
                pitch=np.degrees(np.abs(rpy[:, 1]).max()),
                match=100 * np.mean(log["contact"][sel] == log["plan"][sel]))


def gait_gif(clips):
    """Tile the four clips 2 x 2 and write the gait name and measured speed on each."""
    from PIL import Image, ImageDraw, ImageFont
    from common.video import save_gif
    font = ImageFont.load_default(size=14)
    tiles = []
    for name, (frames, stats) in clips.items():
        labelled = []
        for f in frames:
            img = Image.fromarray(f)
            ImageDraw.Draw(img).text((8, 6), f"{name}  {stats['speed']:.2f} m/s", fill="white",
                                     font=font)
            labelled.append(np.asarray(img))
        tiles.append(np.array(labelled))
    n = min(len(t) for t in tiles)
    top = np.concatenate([tiles[0][:n], tiles[1][:n]], axis=2)
    bottom = np.concatenate([tiles[2][:n], tiles[3][:n]], axis=2)
    save_gif(list(np.concatenate([top, bottom], axis=1)), MEDIA / "gait_all.gif", fps=15, colors=96)


def gait_diagram(logs):
    """Planned stance (pale bars) and measured foot contact (dark bars) over two gait cycles."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(logs), 1, figsize=(8, 7.5))
    for ax, (name, log) in zip(axes, logs.items()):
        period = GAITS[name].period
        t0 = 6.0
        sel = (log["t"] >= t0) & (log["t"] < t0 + 2 * period)
        t = log["t"][sel] - t0
        for leg, leg_name in enumerate(go2.LEGS):
            y = 3 - leg
            ax.fill_between(t, y - 0.38, y + 0.38, where=log["plan"][sel, leg], step="post",
                            color=COLORS[leg_name], alpha=0.25, lw=0)
            ax.fill_between(t, y - 0.12, y + 0.12, where=log["contact"][sel, leg], step="post",
                            color=COLORS[leg_name], lw=0)
        ax.set_yticks([3, 2, 1, 0], go2.LEGS)
        ax.set_xlim(0, 2 * period)
        ax.set_title(f"{name}: cycle {period} s, duty factor {GAITS[name].duty}", fontsize=10,
                     loc="left")
        ax.axvline(period, color="0.5", lw=0.8, ls="--")      # end of the first cycle
    axes[-1].set_xlabel("time (s), two gait cycles")
    fig.suptitle("Pale: planned stance.  Dark: foot touching the ground in the simulation.",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(MEDIA / "gait_diagram.png", dpi=110)
    plt.close(fig)


def foot_paths(log, model):
    """FL foot in the trot, relative to the body: target and simulated path, and the Bezier
    control points of one swing (rebuilt here for the steady speed)."""
    import matplotlib.pyplot as plt
    ctrl = GaitController(model, "trot", speed=DEFAULTS["trot"]["speed"])
    g = GAITS["trot"]
    t_stance, t_swing = g.duty * g.period, (1 - g.duty) * g.period
    v = np.array([DEFAULTS["trot"]["speed"], 0.0, 0.0])
    start = ctrl.home[0] - v * t_stance / 2
    end = ctrl.touchdown_point(0, v, np.zeros(3))
    pts = swing_curve(start, end, ctrl.step_height, -v, t_swing)
    ground = -(0.30 - go2.FOOT_RADIUS)

    sel = (log["t"] >= 6.0) & (log["t"] < 6.0 + 3 * g.period)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a1.plot(log["target"][sel, 0, 0], log["target"][sel, 0, 2] - ground, color="0.6", lw=3,
            label="foot target")
    a1.plot(log["feet_body"][sel, 0, 0], log["feet_body"][sel, 0, 2] - ground, color=COLORS["FL"],
            lw=1.2, label="foot in the simulation")
    a1.plot(pts[:, 0], pts[:, 2] - ground, "o--", color="k", ms=4, lw=0.7,
            label="Bezier control points")
    a1.set_xlabel("x, forward of the body origin (m)")
    a1.set_ylabel("height above the stance line (m)")
    a1.set_title("FL foot relative to the body, trot at 0.5 m/s", fontsize=10, loc="left")
    a1.set_aspect("equal")
    a1.set_ylim(-0.012, 0.175)
    a1.legend(fontsize=7.5, loc="upper center", ncol=3, handlelength=1.5)
    tt = log["t"][sel] - 6.0
    a2.plot(tt, log["target"][sel, 0, 2] - ground, color="0.6", lw=3, label="target")
    a2.plot(tt, log["feet_body"][sel, 0, 2] - ground, color=COLORS["FL"], lw=1.2, label="simulation")
    a2.set_xlabel("time (s)")
    a2.set_ylabel("foot height (m)")
    a2.set_title("FL foot height, three trot cycles", fontsize=10, loc="left")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MEDIA / "gait_foot_paths.png", dpi=110)
    plt.close(fig)


def media():
    """Run the four gaits for 12 s each, write the gait stage's media and print the numbers."""
    import matplotlib
    matplotlib.use("Agg")
    from common.video import Recorder
    MEDIA.mkdir(exist_ok=True)
    model = go2.load_model()
    logs, clips = {}, {}
    for name, settings in DEFAULTS.items():
        rec = Recorder(model, 280, 210, fps=15, distance=1.1, azimuth=100, elevation=-12,
                       start=4.0, end=7.0)
        logs[name] = simulate(name, 12.0, frame_hook=rec, model=model, **settings)
        stats = measure(logs[name])
        clips[name] = (rec.frames, stats)
        print(f"gait {name:5s}: commanded {settings['speed']:.2f} m/s, measured {stats['speed']:.3f} m/s, "
              f"max roll {stats['roll']:.1f} deg, max pitch {stats['pitch']:.1f} deg, "
              f"feet as planned {stats['match']:.0f} % of the time")
    gait_gif(clips)
    gait_diagram(logs)
    foot_paths(logs["trot"], model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gait", choices=list(DEFAULTS))
    ap.add_argument("--speed", type=float, help="forward speed command (m/s)")
    ap.add_argument("--turn", type=float, default=0.0, help="turn rate command (rad/s)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--gif", help="write a GIF here instead of opening the viewer")
    args = ap.parse_args()
    settings = dict(DEFAULTS[args.gait])
    if args.speed is not None:
        settings["speed"] = args.speed

    model = go2.load_model()
    if args.gif:
        from common.video import Recorder, save_gif
        rec = Recorder(model, 480, 320, end=args.seconds)
        log = simulate(args.gait, args.seconds, turn=args.turn, frame_hook=rec, model=model,
                       **settings)
        save_gif(rec.frames, args.gif)
    else:
        import mujoco
        from common.video import WindowClosed, viewer_hook
        data = mujoco.MjData(model)
        show, close = viewer_hook(model, data)
        try:
            log = simulate(args.gait, args.seconds, turn=args.turn, frame_hook=show, model=model,
                           data=data, **settings)
        except WindowClosed:
            print(f"viewer closed at t = {data.time:.1f} s")
            return
        finally:
            close()
    after = min(4.0, args.seconds / 2)
    speed, turn = steady_state(log, after=after)
    tilt = np.degrees(np.abs(log["rpy"][log["t"] >= after, :2]).max())
    print(f"{args.gait}: commanded {settings['speed']:.2f} m/s and {args.turn:.2f} rad/s, "
          f"measured {speed:.3f} m/s and {turn:.3f} rad/s, largest tilt {tilt:.1f} deg "
          f"(all from second {after:g} on)")


if __name__ == "__main__":
    main()
