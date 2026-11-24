"""Checks `set_block` / `set_blocks` in the game, then cleans up after itself.

Runs in an empty box of air (40 × 25 × 36 blocks) a few blocks from the player, higher up if needed, and writes
air back over the whole box at the end. Each check prints PASS, FAIL or INFO:

1. round trip     exact states written, then read back with get_blocks(states=True) right away
2. defaults       a block without properties gets its default state
3. validation     a batch with one bad entry raises McBotError and writes nothing
4. no updates     neighbours are not updated: a torch on air stays, two fences side by side stay unconnected
5. large write    8,000 blocks (two batches), checked, timed, then removed
6. self-changing  blocks the game changes right after they are written: sand, redstone wire, water
7. far away       a block 600 blocks away, outside the area the client has loaded

Usage:
    1. Start Minecraft with the mod and join a singleplayer world (any game mode, cheats not needed).
    2. uv run python scripts/check_set_blocks.py [--skip-far]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Callable

import numpy as np

from mcbot import McBotError, MinecraftClient, connect_with_retry

Pos = tuple[int, int, int]

BOX_SIZE = (40, 25, 36)
AIR = {"air", "cave_air", "void_air"}
FAR_DISTANCE = 600

# States written by the round trip, with every property given: the game must keep them exactly.
ROUND_TRIP = [
    *[f"oak_stairs[facing={facing},half={half},shape=straight,waterlogged=false]"
      for facing in ("north", "south", "east", "west") for half in ("bottom", "top")],
    "oak_log[axis=x]", "oak_log[axis=y]", "oak_log[axis=z]",
    "oak_slab[type=top,waterlogged=false]", "oak_slab[type=double,waterlogged=false]",
    "oak_trapdoor[facing=south,half=top,open=true,powered=false,waterlogged=false]",
    "oak_fence[east=true,north=false,south=false,waterlogged=false,west=true]",
    "chest[facing=east,type=single,waterlogged=false]",
    "furnace[facing=west,lit=false]",
    "glass",
]
DOOR = ["oak_door[facing=east,half=lower,hinge=left,open=true,powered=false]",
        "oak_door[facing=east,half=upper,hinge=left,open=true,powered=false]"]
STAIRS_DEFAULT = "oak_stairs[facing=north,half=bottom,shape=straight,waterlogged=false]"


class Checks:
    def __init__(self):
        self.failed = 0

    def report(self, status: str, name: str, detail: str = "") -> None:
        if status == "FAIL":
            self.failed += 1
        print(f"[{status}] {name}" + (f": {detail}" if detail else ""))

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.report("PASS" if ok else "FAIL", name, detail)
        return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--skip-far", action="store_true", help="skip check 7 (loads a chunk 600 blocks away)")
    args = parser.parse_args()

    checks = Checks()
    with connect_with_retry(args.host, args.port) as mc:
        info = mc.wait_for_world()
        origin = find_empty_box(mc, info["pos"])
        print(f"Test box: {origin} to {corner(origin, BOX_SIZE)}\n")
        try:
            run(checks, "1. round trip", lambda: check_round_trip(mc, checks, origin))
            run(checks, "2. defaults", lambda: check_defaults(mc, checks, offset(origin, 0, 0, 3)))
            run(checks, "3. validation", lambda: check_validation(mc, checks, offset(origin, 0, 0, 5)))
            run(checks, "4. no updates", lambda: check_no_updates(mc, checks, offset(origin, 0, 0, 7)))
            run(checks, "5. large write", lambda: check_large_write(mc, checks, offset(origin, 0, 4, 15)))
            run(checks, "6. self-changing", lambda: check_self_changing(mc, checks, offset(origin, 0, 0, 9)))
            if not args.skip_far:
                run(checks, "7. far away", lambda: check_far(mc, checks, info["pos"], origin))
        finally:
            print("\nCleaning up the test box...")
            clear(mc, origin, corner(origin, BOX_SIZE))
    print(f"\n{'All checks passed.' if checks.failed == 0 else f'{checks.failed} check(s) failed.'}")
    sys.exit(1 if checks.failed else 0)


def run(checks: Checks, name: str, check: Callable[[], None]) -> None:
    """Runs one group of checks; an unexpected error fails the group without stopping the others."""
    try:
        check()
    except McBotError as e:
        checks.report("FAIL", name, f"unexpected error from the mod: {e}")


# --- Checks ---------------------------------------------------------------------------------------


def check_round_trip(mc: MinecraftClient, checks: Checks, origin: Pos) -> None:
    entries = [(offset(origin, 2 * i, 0, 0), state) for i, state in enumerate(ROUND_TRIP)]
    door = offset(origin, 2 * len(ROUND_TRIP), 0, 0)
    entries += [(door, DOOR[0]), (offset(door, 0, 1, 0), DOOR[1])]

    start = time.perf_counter()
    info = mc.set_blocks(entries)
    elapsed = time.perf_counter() - start
    volume = mc.get_blocks(origin, offset(door, 0, 1, 0), states=True)  # read right after writing

    returned_ok = info["blocks"] == [state for _, state in entries]
    checks.check(returned_ok, "1. round trip: states returned by set_blocks are the ones requested",
                 "" if returned_ok else _diff([s for _, s in entries], info["blocks"]))
    read = [volume.block_at(*pos) for pos, _ in entries]
    read_ok = read == [state for _, state in entries]
    checks.check(read_ok, f"1. round trip: {len(entries)} states read back identical, right after writing "
                          f"({elapsed * 1000:.0f} ms)", "" if read_ok else _diff([s for _, s in entries], read))


def check_defaults(mc: MinecraftClient, checks: Checks, pos: Pos) -> None:
    info = mc.set_block(*pos, "minecraft:oak_stairs")
    read = mc.get_blocks(pos, pos, states=True).block_at(*pos)
    checks.check(info["block"] == STAIRS_DEFAULT and read == STAIRS_DEFAULT,
                 "2. defaults: 'minecraft:oak_stairs' becomes the default state",
                 f"returned {info['block']}, read {read}")


def check_validation(mc: MinecraftClient, checks: Checks, origin: Pos) -> None:
    cells = [offset(origin, i, 0, 0) for i in range(3)]
    before = mc.get_blocks(cells[0], cells[-1], states=True)
    bad_batches = {
        "unknown block": "not_a_block",
        "invalid property value": "oak_stairs[facing=up]",
        "unknown property": "stone[color=red]",
    }
    for name, bad in bad_batches.items():
        try:
            mc.set_blocks([(cells[0], "stone"), (cells[1], bad), (cells[2], "stone")])
            raised = ""
        except McBotError as e:
            raised = str(e)
        after = mc.get_blocks(cells[0], cells[-1], states=True)
        unchanged = np.array_equal(_names(before), _names(after))
        checks.check(bool(raised) and unchanged, f"3. validation: {name} ({bad!r}) is refused, nothing written",
                     f"error: {raised or 'none'}; box {'unchanged' if unchanged else 'CHANGED'}")
        if not unchanged:
            clear(mc, cells[0], cells[-1])

    x, _, z = origin
    try:
        mc.set_block(x, 10_000, z, "stone")
        checks.check(False, "3. validation: a position above the world height is refused", "no error")
    except McBotError as e:
        checks.check(True, "3. validation: a position above the world height is refused", str(e))


def check_no_updates(mc: MinecraftClient, checks: Checks, origin: Pos) -> None:
    torch, fence1, fence2 = origin, offset(origin, 2, 0, 0), offset(origin, 3, 0, 0)
    mc.set_blocks([(torch, "torch"), (fence1, "oak_fence"), (fence2, "oak_fence")])
    mc.step(ticks=20)
    volume = mc.get_blocks(torch, fence2, states=True)
    checks.check(volume.block_at(*torch) == "torch", "4. no updates: a torch on air stays after 1 s",
                 f"cell holds {volume.block_at(*torch)}")
    states = volume.block_at(*fence1), volume.block_at(*fence2)
    unconnected = all("east=false" in s and "west=false" in s for s in states)
    checks.check(unconnected, "4. no updates: two fences written side by side stay unconnected", ", ".join(states))


def check_self_changing(mc: MinecraftClient, checks: Checks, origin: Pos) -> None:
    """Blocks the game changes as soon as they are written. The call must still answer, without a timeout."""
    # Sand two blocks above a stone: it should fall onto it.
    stone, sand = origin, offset(origin, 0, 2, 0)
    mc.set_block(*stone, "stone")
    _timed_write(mc, checks, "6. self-changing: sand over air is written", [(sand, "sand")])
    mc.step(ticks=40)
    volume = mc.get_blocks(stone, sand)
    landed = volume.block_at(*offset(stone, 0, 1, 0))
    checks.report("INFO", "6. self-changing: after 2 s, the sand is "
                  + ("on the stone (it fell)" if landed == "sand" else f"still there ({volume.block_at(*sand)})"))

    # Redstone wire with power 7 and no power source: the game sets its power back to 0 straight away.
    wire = offset(origin, 3, 0, 0)
    _timed_write(mc, checks, "6. self-changing: redstone wire with power=7 and no source is written",
                 [(wire, "redstone_wire[east=none,north=none,power=7,south=none,west=none]")])
    checks.report("INFO", f"6. self-changing: the wire is now {mc.get_blocks(wire, wire, states=True).block_at(*wire)}")

    # Water in a stone basin (5 × 5 floor, rim one block high): does it spread?
    basin = offset(origin, 6, 0, 0)
    floor = [(offset(basin, dx, 0, dz), "stone") for dx in range(5) for dz in range(5)]
    rim = [(offset(basin, dx, 1, dz), "stone") for dx in range(5) for dz in range(5)
           if dx in (0, 4) or dz in (0, 4)]
    mc.set_blocks(floor + rim)
    source = offset(basin, 2, 1, 2)
    _timed_write(mc, checks, "6. self-changing: a water source is written", [(source, "water")])
    mc.step(ticks=40)
    volume = mc.get_blocks(offset(basin, 1, 1, 1), offset(basin, 3, 1, 3))
    wet = int(volume.mask("water").sum())
    checks.report("INFO", f"6. self-changing: after 2 s, {wet} of the 9 cells of the basin hold water "
                  + ("(it spreads)" if wet > 1 else "(it does not spread)"))


def check_large_write(mc: MinecraftClient, checks: Checks, origin: Pos) -> None:
    size = 20
    cells = [offset(origin, dx, dy, dz) for dx in range(size) for dy in range(size) for dz in range(size)]
    expected = ["stone" if (x + y + z) % 2 else "oak_planks" for x, y, z in cells]

    start = time.perf_counter()
    mc.set_blocks(zip(cells, expected))
    elapsed = time.perf_counter() - start
    far_corner = offset(origin, size - 1, size - 1, size - 1)
    volume = mc.get_blocks(origin, far_corner)
    names = _names(volume)
    x, y, z = np.meshgrid(*(np.arange(o, o + size) for o in origin), indexing="ij")
    wanted = np.where((x + y + z) % 2 == 1, "stone", "oak_planks")
    wrong = int((names != wanted).sum())
    checks.check(wrong == 0, f"5. large write: {len(cells)} blocks written and checked",
                 f"{elapsed:.2f} s, {len(cells) / elapsed:.0f} blocks/s" + (f", {wrong} wrong" if wrong else ""))

    start = time.perf_counter()
    clear(mc, origin, far_corner)
    elapsed = time.perf_counter() - start
    empty = mc.get_blocks(origin, far_corner)
    checks.check(bool(np.isin(_names(empty), list(AIR)).all()), "5. large write: the cube is removed with air",
                 f"{elapsed:.2f} s")


def check_far(mc: MinecraftClient, checks: Checks, player_pos, origin: Pos) -> None:
    """The server writes anywhere, but the client only has the chunks around the player."""
    x, _, z = (math.floor(c) for c in player_pos)
    pos = (x + FAR_DISTANCE, origin[1] + BOX_SIZE[1] - 1, z)  # same height as the top of the (empty) test box
    print(f"   writing glass at {pos}, {FAR_DISTANCE} blocks away (the server may need to generate the chunk)...")
    ok = _timed_write(mc, checks, f"7. far away: a block {FAR_DISTANCE} blocks away is written", [(pos, "glass")])
    if not ok:
        checks.report("INFO", "7. far away: the server has probably written it anyway; fly to "
                              f"{pos} to see (one glass block in the air)")
    try:
        mc.set_block(*pos, "air")
    except McBotError as e:
        checks.report("INFO", f"7. far away: could not remove it ({e})")


# --- Helpers --------------------------------------------------------------------------------------


def _timed_write(mc: MinecraftClient, checks: Checks, name: str, entries) -> bool:
    start = time.perf_counter()
    try:
        info = mc.set_blocks(entries)
    except McBotError as e:
        return checks.check(False, name, f"{e} after {time.perf_counter() - start:.1f} s")
    return checks.check(True, name, f"{time.perf_counter() - start:.2f} s, returned {info['blocks']}")


def find_empty_box(mc: MinecraftClient, player_pos) -> Pos:
    """A box of air a few blocks from the player, starting 3 blocks above the feet and moving up if needed."""
    x, y, z = (math.floor(c) for c in player_pos)
    origin = (x + 4, y + 3, z + 4)
    for _ in range(10):
        try:
            volume = mc.get_blocks(origin, corner(origin, BOX_SIZE))
        except McBotError as e:
            sys.exit(f"Cannot use the box at {origin}: {e}. Move somewhere else (lower, or in an open area).")
        if np.isin(_names(volume), list(AIR)).all():
            return origin
        origin = offset(origin, 0, 10, 0)
    sys.exit("No empty box found above the player. Move to an open area and try again.")


def clear(mc: MinecraftClient, corner1: Pos, corner2: Pos) -> None:
    (x0, y0, z0), (x1, y1, z1) = corner1, corner2
    mc.set_blocks(((x, y, z), "air") for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
                  for z in range(z0, z1 + 1))


def offset(pos: Pos, dx: int, dy: int, dz: int) -> Pos:
    return pos[0] + dx, pos[1] + dy, pos[2] + dz


def corner(origin: Pos, size: Pos) -> Pos:
    return offset(origin, size[0] - 1, size[1] - 1, size[2] - 1)


def _names(volume) -> np.ndarray:
    """Block name of every cell of a BlockVolume, as an array of strings."""
    return np.array(volume.palette, dtype=object)[volume.ids]


def _diff(expected: list[str], actual: list[str]) -> str:
    return "; ".join(f"expected {e}, got {a}" for e, a in zip(expected, actual) if e != a)


if __name__ == "__main__":
    main()
