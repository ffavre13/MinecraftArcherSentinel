"""Test: the player stands still and lists the entities around it, again and again.

Every `--every` ticks (default 20 = 1 second), prints the entities within `--radius` blocks, nearest first, with
their name and coordinates. Entities that appeared since the previous scan are marked `+`, and the ones that
disappeared are listed after the table. Stop with Ctrl+C.

Usage:
    1. Start Minecraft with the mod and join a singleplayer world.
    2. uv run python scripts/watch_entities.py [--radius 32] [--every 20] [--scans 0]
"""

from __future__ import annotations

import argparse

from mcbot import TICKS_PER_SECOND, Entity, connect_with_retry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--radius", type=float, default=32, help="distance from the player, in blocks (default: 32)")
    parser.add_argument("--every", type=int, default=TICKS_PER_SECOND,
                        help=f"ticks between two scans (default: {TICKS_PER_SECOND})")
    parser.add_argument("--scans", type=int, default=0, help="number of scans, 0 = until Ctrl+C (default: 0)")
    args = parser.parse_args()

    with connect_with_retry(args.host, args.port) as mc:
        mc.wait_for_world()
        previous: dict[int, Entity] = {}
        scan = 0
        try:
            while args.scans == 0 or scan < args.scans:
                scan += 1
                # No key held: the player stands still while the ticks pass.
                _, info = mc.step(ticks=args.every, entities={"radius": args.radius})
                current = {entity.id: entity for entity in info["entities"]}
                show(scan, info, current, previous, args.radius)
                previous = current
        except KeyboardInterrupt:
            print("\nStopped.")


def show(scan: int, info: dict, current: dict[int, Entity], previous: dict[int, Entity], radius: float) -> None:
    px, py, pz = info["pos"]
    print(f"\n=== Scan {scan}: player at ({px:.1f}, {py:.1f}, {pz:.1f}), "
          f"{len(current)} entities within {radius:g} blocks ===")
    if current:
        print(f"  {'id':>6}  {'type':<18} {'name':<24} {'x':>8} {'y':>7} {'z':>8} {'dist':>6}")
    for entity in current.values():
        mark = "+" if scan > 1 and entity.id not in previous else " "
        x, y, z = entity.pos
        print(f"{mark} {entity.id:>6}  {entity.type:<18} {display_name(entity):<24} "
              f"{x:8.1f} {y:7.1f} {z:8.1f} {entity.distance:6.1f}")
    for entity in previous.values():
        if entity.id not in current:
            print(f"- {entity.id:>6}  {entity.type:<18} {display_name(entity):<24} gone")


def display_name(entity: Entity) -> str:
    """Custom name or player name; for a dropped item, what it holds; otherwise empty."""
    if entity.name:
        return entity.name
    if entity.item is not None:
        return f"{entity.count} x {entity.item}"
    return ""


if __name__ == "__main__":
    main()
