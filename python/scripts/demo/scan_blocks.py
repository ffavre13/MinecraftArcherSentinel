"""Demo: reads the blocks around the player and prints a top-down map.

For each column around the player, the map shows the highest block of the box that was read
(north up, east right, the player is "@"), then the number of blocks of each kind.

Usage:
    1. Start Minecraft with the mod and join a singleplayer world.
    2. uv run python scripts/scan_blocks.py [--radius 8]
"""

from __future__ import annotations

import argparse
import math
from collections import Counter

import numpy as np

from mcbot import BlockVolume, connect_with_retry

AIR = ("air", "cave_air", "void_air")
SYMBOLS = "#%=+*o~^x&$:;!?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--radius", type=int, default=8, help="half-side of the cube read around the player (default: 8)")
    args = parser.parse_args()

    with connect_with_retry(args.host, args.port) as mc:
        info = mc.wait_for_world()
        volume = mc.get_blocks_around(radius=args.radius)
    show(volume, *(math.floor(c) for c in info["pos"]))


def show(volume: BlockVolume, x: int, y: int, z: int) -> None:
    """Prints the top-down map and the block counts; (x, y, z) = block of the player's feet."""
    print(f"Player at ({x}, {y}, {z}), box of {volume.shape} blocks from {volume.origin}")
    print(f"Under the feet: {volume.block_at(x, y - 1, z)}\n")

    # Highest block of each column (x, z) of the box; None if the column is empty.
    solid = ~np.isin(volume.ids, [i for i, name in enumerate(volume.palette) if name in AIR])
    top = volume.ids.shape[1] - 1 - np.argmax(solid[:, ::-1, :], axis=1)
    columns = {(i, k): volume.palette[volume.ids[i, top[i, k], k]] if solid[i, :, k].any() else None
               for i in range(volume.shape[0]) for k in range(volume.shape[2])}

    names = [name for name, _ in Counter(n for n in columns.values() if n).most_common()]
    symbols = {name: SYMBOLS[i] if i < len(SYMBOLS) else "?" for i, name in enumerate(names)}
    player = (x - volume.origin[0], z - volume.origin[2])
    for k in range(volume.shape[2]):
        row = ("@" if (i, k) == player else symbols.get(columns[(i, k)], " ") for i in range(volume.shape[0]))
        print(" ".join(row))

    print("\nLegend (top-down view):")
    for name in names:
        print(f"  {symbols[name]}  {name}")

    print("\nBlocks in the box:")
    counts = np.bincount(volume.ids.ravel(), minlength=len(volume.palette))
    for i in np.argsort(-counts):
        if counts[i] and volume.palette[i] not in AIR:
            print(f"  {counts[i]:6}  {volume.palette[i]}")


if __name__ == "__main__":
    main()
