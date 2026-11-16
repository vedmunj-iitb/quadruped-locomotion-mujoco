"""Makes every picture and clip in media/ and prints the numbers quoted in the README.

Written in 2026.

    python make_media.py                  # all five stages (a few minutes on this PC)
    python make_media.py gait mpc         # only some of them
"""
import importlib
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")   # offscreen rendering: must be set before MuJoCo loads

STAGES = ["gait", "footholds", "mpc", "wbc", "stack"]


def main():
    names = sys.argv[1:] or STAGES
    for name in names:
        if name not in STAGES:
            sys.exit(f"unknown stage {name!r}, pick from {STAGES}")
    for name in names:
        start = time.time()
        print(f"== {name}")
        importlib.import_module(f"{name}.demo").media()
        print(f"   ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
