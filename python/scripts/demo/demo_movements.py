"""Demo: moves the Minecraft character from Python through the McBot mod.

Usage:
    1. Start Minecraft with the mod and join a singleplayer world (ideally Superflat).
    2. uv run python scripts/demo_movements.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from mcbot import connect_with_retry

FRAMES_DIR = Path(__file__).resolve().parents[1] / "frames"


def report(label: str, info: dict, image=None) -> None:
    x, y, z = info["pos"]
    line = f"{label:<32} pos=({x:7.2f}, {y:6.2f}, {z:7.2f})  yaw={info['yaw']:7.1f}  pitch={info['pitch']:6.1f}"
    if image is not None:
        FRAMES_DIR.mkdir(exist_ok=True)
        path = FRAMES_DIR / f"{label.split()[0]}.png"
        Image.fromarray(image).save(path)
        line += f"  -> {path.name}"
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    with connect_with_retry(args.host, args.port) as mc:
        mc.wait_for_world()
        mc.config(width=320, height=180)
        print("Connected! Starting the demo (keep an eye on the Minecraft window).\n")

        image, info = mc.look(pitch=-mc.ping()["pitch"], ticks=5, frame=True)  # look straight ahead
        report("01_start", info, image)

        image, info = mc.forward(ticks=40, frame=True)
        report("02_forward_2s", info, image)

        image, info = mc.turn(90, ticks=10, frame=True)
        report("03_turn_right_90", info, image)

        image, info = mc.forward(ticks=40, jump=True, frame=True)
        report("04_forward_jumping", info, image)

        _, info = mc.turn(360, ticks=40)
        report("05_full_turn", info)

        image, info = mc.look(pitch=-60, ticks=15, frame=True)
        report("06_look_up", info, image)

        image, info = mc.look(pitch=120, ticks=20, frame=True)
        report("07_look_down", info, image)

        _, info = mc.look(pitch=-60, ticks=10)
        report("08_look_straight", info)

        _, info = mc.step(ticks=40, left=True)
        report("09_strafe_left", info)

        _, info = mc.step(ticks=40, back=True)
        report("10_back", info)

        image, info = mc.step(ticks=80, forward=True, yaw=360, frame=True)  # a circle in 4 s
        report("11_walk_in_circle", info, image)

        image, info = mc.forward(ticks=40, sprint=True, frame=True)
        report("12_sprint", info, image)

        _, info = mc.jump()
        report("13_jump_in_place", info)

        image, info = mc.stop(frame=True)
        report("14_stop", info, image)

    print(f"\nDemo finished. Images saved in {FRAMES_DIR}")


if __name__ == "__main__":
    try:
        main()
    except ConnectionError:
        print("\nConnection lost: Minecraft was closed (or the world was left abruptly).")
    except KeyboardInterrupt:
        print("\nDemo interrupted.")
