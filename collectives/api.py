# Each Device holds a set of shards
from __future__ import annotations

import heapq
from typing import Callable, List, Optional
import numpy as np

from .datatypes import ShardRequest, AssociativeReduction
from .shardbuffer import ShardBuffer


class Device:
    def __init__(
        self,
        rank: int,
        world_size: int,
        scheduler: "Scheduler",
        shard_size: int,
        hop_latency: int,
        left: Optional["Device"] = None,
        right: Optional["Device"] = None,
    ):
        self.scheduler = scheduler
        self.shards = ShardBuffer(world_size, shard_size)
        self.rank = rank
        self.world_size = world_size
        self.shard_size = shard_size
        self.hop_latency = hop_latency
        # Define topology
        self.left = left
        self.right = right

        # precompute directions for sending
        self._dir = {}
        for i in range(world_size):
            cw = (i - rank) % world_size
            ccw = (rank - i) % world_size
            self._dir[i] = -1 if cw <= ccw else 1

    def put(self, i, shard: np.ndarray) -> None:
        # places a shard in a slot and overwrites
        self.shards.put(i, shard)

    def reset(self) -> None:
        # clear all shard slots
        self.shards.reset()

    def send(
        self,
        i,
        to: "Device",
        time: int,
        reduction: Optional[AssociativeReduction] = None,
        auto_forward: bool = False,
        blocking: bool = False,
        retain: bool = True,
        direction: Optional[int] = None,
    ) -> None:
        shard = self.shards[i]
        assert not np.isnan(shard).all(), f"Shard slot={i} on rank={self.rank} is empty"
        if blocking:
            to.recv(
                i,
                time,
                shard,
                reduction=reduction,
                auto_forward=auto_forward,
                direction=direction,
            )
        else:
            print(f"DEBUG(send):rank={self.rank} sending shard={i} to {to.rank}")
            self.scheduler.schedule(
                time=time,
                latency=self.hop_latency,
                i=i,
                src=self,
                dst=to,
                payload=shard.copy(),
                reduction=reduction,
                auto_forward=auto_forward,
                retain=retain,
                direction=direction,
            )
        if not retain:
            # Once we send the shard (ref), we don't need it anymore
            # so we can free this slot
            self.shards.clear(i)

    def recv(
        self,
        i: int,
        time: int,
        shard: np.ndarray,
        reduction: Optional[AssociativeReduction] = None,
        auto_forward: bool = False,
        blocking: bool = False,
        retain: bool = True,
        direction: Optional[int] = None,
    ) -> None:
        """
        Place `shard` into slot `i`. If `reduction` is specified, reduce
        with the existing shard; otherwise, slot must be empty.
        """
        existing = self.shards[i]
        if reduction is None:
            # assert existing is None, f"attempting overwrite at {i} when reduction=None"
            if not np.all(np.isnan(existing)):
                return
            self.shards[i] = shard.copy()
        else:
            if not np.all(np.isnan(existing)):
                if reduction == "sum":
                    self.shards[i] = existing + shard
                elif reduction == "max":
                    self.shards[i] = np.maximum(existing, shard)
                elif reduction == "min":
                    self.shards[i] = np.minimum(existing, shard)
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")
            else:
                self.shards[i] = shard
                return

        if auto_forward:
            if direction is None:
                direction = self._dir[i]

            print(
                f"DEBUG(recv):rank={self.rank} got shard={i}, sending to={'right' if direction == 1 else 'left'}"
            )

            # Event-driven firing off a send to the next device
            if direction == -1 and self.left is not None:
                self.send(
                    i,
                    self.left,
                    time,
                    reduction=reduction,
                    auto_forward=auto_forward,
                    blocking=blocking,
                    retain=retain,
                    direction=direction,
                )
            elif self.right is not None:
                self.send(
                    i,
                    self.right,
                    time,
                    reduction=reduction,
                    auto_forward=auto_forward,
                    blocking=blocking,
                    retain=retain,
                    direction=direction,
                )
            else:
                raise RuntimeError("attempted to send along nonexistent ICI")

    def __str__(self) -> str:
        # build rows of slot content
        MAX_DISPLAY = 4
        rows = []
        for shard in self.shards.data:
            if shard is None:
                content = ""
            else:
                arr = shard.tolist()
                n = len(arr)
                if n <= MAX_DISPLAY:
                    content = " ".join(str(x) for x in arr)
                else:
                    half = MAX_DISPLAY // 2
                    front = " ".join(str(x) for x in arr[:half])
                    back = " ".join(str(x) for x in arr[-half:])
                    content = f"{front} … {back}"
            rows.append(f"[{content}]")

        # figure out box width
        all_lines = [f"Device rank={self.rank}", *rows]
        width = max(len(line) for line in all_lines) + 2  # padding

        # box-drawing
        top = "┌" + "─" * width + "┐"
        header = "│" + all_lines[0].center(width) + "│"
        sep = "├" + "─" * width + "┤"
        body = "\n".join("│" + line.ljust(width) + "│" for line in rows)
        bot = "└" + "─" * width + "┘"

        return "\n".join([top, header, sep, body, bot])

    def __repr__(self):
        return self.__str__()


class Scheduler:
    def __init__(self):
        self._pipe: List[ShardRequest] = []

    def schedule(
        self,
        time: int,
        latency: int,
        i: int,
        src: Device,
        dst: Device,
        payload: np.ndarray,
        auto_forward: bool,
        reduction: Optional[AssociativeReduction],
        retain: bool,
        direction: Optional[int],
    ):
        heapq.heappush(
            self._pipe,
            ShardRequest(
                arrival=time + latency,
                i=i,
                src=src,
                dst=dst,
                payload=payload,
                auto_forward=auto_forward,
                reduction=reduction,
                retain=retain,
                direction=direction,
            ),
        )

    def run(self, start: Device, stopping_condition: Callable = lambda: False):
        while self._pipe and not stopping_condition():
            # Run a step of our scheduler:
            # 1) pop a ShardRequest, perform blocking recv
            req = heapq.heappop(self._pipe)
            req.dst.recv(
                i=req.i,
                time=req.arrival,
                shard=req.payload,  # type: ignore
                reduction=req.reduction,
                auto_forward=req.auto_forward,
                retain=req.retain,
                direction=req.direction,
            )
            visualize_topology(start)


def visualize_topology(start: Device, ring=True):
    seen = set()
    cur = start
    order = []
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        order.append(cur)
        cur = cur.right

    blocks = [d.__str__().splitlines() for d in order]
    height = max(len(b) for b in blocks)

    # render
    for row in range(height):
        line_parts = []
        for i, blk in enumerate(blocks):
            line_parts.append(blk[row])
            if row == height // 2:
                if i < len(order) - 1:
                    line_parts.append(" <-> ")
                elif ring:
                    line_parts.append("  ↺  ")
            else:
                line_parts.append(" " * 5)
        if ring:
            if row == height // 2:
                line_parts.insert(0, "  ↺  ")
            else:
                line_parts.insert(0, " " * 5)
        print("".join(line_parts))


def create_node(world_size: int, shard_size: int, hop_latency: int) -> tuple["Scheduler", list["Device"]]:
    scheduler = Scheduler()
    head = Device(0, world_size, scheduler, shard_size, hop_latency)
    prev = head
    cur = None

    device_list = [head]

    # Implement the chaining
    for rank in range(1, world_size):
        cur = Device(rank, world_size, scheduler, shard_size, hop_latency, left=prev)
        device_list.append(cur)
        prev.right = cur
        prev = cur

    # Wraparound
    tail = prev
    head.left = tail
    tail.right = head

    return scheduler, device_list
