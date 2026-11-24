"""Automatic building example: a small house placed on the ground, in front of the player.

The script switches to build mode (creative), takes off, places the floor, the walls (with windows, a door and a
torch), then a pyramid roof, and finally steps back to take a picture of the house (frames/house.png).

Usage:
    1. Start Minecraft with the mod, join a singleplayer world (ideally Superflat) and stand on the ground,
       facing a free area of about 10 × 10 blocks.
    2. uv run python scripts/build_house.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from mcbot import Builder, Phase, Placement, connect_with_retry

FRAMES_DIR = Path(__file__).resolve().parents[1] / "frames"

SIZE = 7  # side of the house, in blocks
WALL_HEIGHT = 3
DISTANCE = 3  # distance between the player and the front wall

FLOOR = "stone_bricks"
WALL = "oak_planks"
PILLAR = "oak_log"
WINDOW = "glass_pane"
DOOR = "oak_door"
LIGHT = "torch"
ROOF = "dark_oak_planks"

Local = tuple[int, int, int]


# --- House plan, in local coordinates ----------------------------------------------------------
# x: from left to right (seen from the player), z: from the front wall to the back, y: 0 = house floor.

def is_window(x: int, z: int) -> bool:
    n, mid = SIZE - 1, (SIZE - 1) // 2
    if z == 0:
        return x in (1, n - 1)  # front: one window on each side of the door
    if z == n:
        return abs(x - mid) <= 1  # back: 3-block window
    return x in (0, n) and abs(z - mid) <= 1  # sides: 3-block window


def perimeter() -> list[tuple[int, int]]:
    """Cells around the house, in order (the builder walks around the wall)."""
    n = SIZE - 1
    return ([(x, 0) for x in range(SIZE)] + [(n, z) for z in range(1, SIZE)]
            + [(x, n) for x in range(n - 1, -1, -1)] + [(0, z) for z in range(n - 1, 0, -1)])


def house_plan() -> list[tuple[str, tuple[int, float, int], list[tuple[Local, str]]]]:
    """Building phases: (name, flying spot (cell x, feet height, cell z), blocks)."""
    n, mid = SIZE - 1, (SIZE - 1) // 2
    inside = (mid, 2.0, mid)  # in the centre, flying above the floor: the whole ground floor is within reach

    floor = [((x, 0, z), FLOOR) for z in range(SIZE) for x in range(SIZE)]

    walls = []
    for y in range(1, WALL_HEIGHT + 1):
        for x, z in perimeter():
            if (x, z) == (mid, 0) and y <= 2:
                continue  # door opening
            if x in (0, n) and z in (0, n):
                block = PILLAR
            elif y == 2 and is_window(x, z):
                block = WINDOW
            else:
                block = WALL
            walls.append(((x, y, z), block))
    details = [((mid, 1, 0), DOOR), ((mid, 1, n - 1), LIGHT)]

    phases = [("Floor", inside, floor), ("Walls, door and torch", inside, walls + details)]
    for i in range(mid + 1):  # pyramid roof: 7×7, 5×5, 3×3, 1×1
        y = WALL_HEIGHT + 1 + i
        layer = [((x, y, z), ROOF) for z in range(i, SIZE - i) for x in range(i, SIZE - i)]
        phases.append((f"Roof, level {i + 1}", (mid, y + 1.5, mid), layer))
    return phases


# --- From local coordinates to world coordinates -----------------------------------------------

@dataclass(frozen=True)
class Frame:
    """Frame of reference of the house: origin (front-left corner of the floor), "forward" and "right" directions."""

    origin: tuple[int, int, int]
    forward: tuple[int, int]
    right: tuple[int, int]

    def block(self, x: int, y: int, z: int) -> tuple[int, int, int]:
        ox, oy, oz = self.origin
        return (ox + self.right[0] * x + self.forward[0] * z, oy + y,
                oz + self.right[1] * x + self.forward[1] * z)

    def point(self, x: int, feet_y: float, z: int) -> tuple[float, float, float]:
        """Centre of cell (x, z) at feet height feet_y."""
        bx, by, bz = self.block(x, 0, z)
        return bx + 0.5, by + feet_y, bz + 0.5

    @property
    def yaw(self) -> float:
        """Orientation (Minecraft yaw) to look towards the back of the house."""
        return math.degrees(math.atan2(-self.forward[0], self.forward[1]))


def cardinal_direction(yaw: float) -> tuple[int, int]:
    """Direction (dx, dz) closest to where the player looks. Minecraft: yaw 0 = south (+z), 90 = west (-x)."""
    angle = math.radians(round(yaw / 90.0) * 90.0)
    return round(-math.sin(angle)), round(math.cos(angle))


def frame_in_front_of(pos: tuple[float, float, float], yaw: float) -> Frame:
    forward = cardinal_direction(yaw)
    right = (-forward[1], forward[0])
    px, py, pz = (math.floor(c) for c in pos)
    half = SIZE // 2
    origin = (px + forward[0] * DISTANCE - right[0] * half, py,
              pz + forward[1] * DISTANCE - right[1] * half)
    return Frame(origin, forward, right)


def house_phases(frame: Frame) -> list[Phase]:
    return [Phase(name, frame.point(*standpoint), [Placement(frame.block(*local), block) for local, block in blocks])
            for name, standpoint, blocks in house_plan()]


# --- Script ------------------------------------------------------------------------------------

def land(mc, max_ticks: int = 200) -> dict:
    """Stops flying and waits until the player is on the ground."""
    info = mc.set_flying(False)
    for _ in range(max_ticks // 5):
        if info["on_ground"]:
            break
        _, info = mc.step(ticks=5)
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--delay", type=int, default=2, help="ticks to wait after each block (default: 2)")
    args = parser.parse_args()

    with connect_with_retry(args.host, args.port) as mc:
        mc.wait_for_world()
        mc.config(width=640, height=360)

        info = mc.set_build_mode(True)
        print(f"Game mode: {info['game_mode']}")
        info = land(mc)  # the house floor is aligned with the height of the player's feet
        if not info["on_ground"]:
            raise SystemExit("The player must be on the ground to place the house.")

        frame = frame_in_front_of(info["pos"], info["yaw"])
        phases = house_phases(frame)
        total = sum(len(phase.blocks) for phase in phases)
        print(f"{SIZE}×{SIZE} house: {total} blocks, front-left corner at {frame.origin}\n")

        mc.set_flying(True)
        builder = Builder(mc, delay_ticks=args.delay)
        builder.build(phases)
        print(f"\nBuilding finished: {builder.summary()}")

        # Leave through the top, stand in front of the house and take a picture.
        mid = SIZE // 2
        mc.fly_to(*frame.point(mid, WALL_HEIGHT + 6, -9))
        mc.fly_to(*frame.point(mid, 3, -9))
        info = land(mc)
        yaw_delta = (frame.yaw - info["yaw"] + 180) % 360 - 180
        mc.step(ticks=10, yaw=yaw_delta, pitch=-10 - info["pitch"])

        mc.config(width=640, height=360, hide_gui=True)
        image, _ = mc.step(ticks=2, frame=True)
        mc.config(width=640, height=360, hide_gui=False)
        FRAMES_DIR.mkdir(exist_ok=True)
        Image.fromarray(image).save(FRAMES_DIR / "house.png")
        print(f"Picture saved: {FRAMES_DIR / 'house.png'}")


if __name__ == "__main__":
    try:
        main()
    except ConnectionError:
        print("\nConnection lost: Minecraft was closed (or the world was left abruptly).")
    except KeyboardInterrupt:
        print("\nBuilding interrupted.")
