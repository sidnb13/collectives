# Each Device holds a set of shards
from __future__ import annotations

import heapq
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .datatypes import AssociativeReduction, ShardRequest
from .shardbuffer import ShardBuffer, ShardedArray

logger = logging.getLogger(__name__)


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
        add_communication_noise: bool = False,
        communication_noise_mean: float = 0.0,
        communication_noise_std: float = 0.01,
    ):
        self.scheduler = scheduler
        self.shards = ShardedArray(world_size, shard_size)
        self.rank = rank
        self.world_size = world_size
        self.shard_size = shard_size
        self.hop_latency = hop_latency
        # Define topology
        self.left = left
        self.right = right
        self.add_communication_noise = add_communication_noise
        self.communication_noise_mean = communication_noise_mean
        self.communication_noise_std = communication_noise_std

    def put(self, i, shard: ShardBuffer) -> None:
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
        shard = self.shards.get(i)
        assert not np.isnan(shard).all(), (
            f"Shard slot={i} on rank={self.rank} is empty"
        )

        _metadata = {
            "checksum": shard.checksum,
        }

        if blocking:
            to.recv(
                i,
                time,
                shard,
                reduction=reduction,
                auto_forward=auto_forward,
                direction=direction,
                metadata=_metadata,
            )
        else:
            logger.debug(f"rank={self.rank} sending shard={i} to {to.rank}")
            self.scheduler.schedule(
                time=time,
                latency=self.hop_latency,
                i=i,
                src=self,
                dst=to,
                payload=shard.copy(),
                metadata=_metadata,
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
        shard: ShardBuffer,
        reduction: Optional[AssociativeReduction] = None,
        auto_forward: bool = False,
        blocking: bool = False,
        retain: bool = True,
        direction: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Place `shard` into slot `i`. If `reduction` is specified, reduce
        with the existing shard; otherwise, slot must be empty.
        """

        if metadata is not None and metadata.get("checksum") != shard.checksum:
            logger.debug("SDC detected, checksums do not match")

        existing = self.shards.get(i)
        if reduction is None:
            # assert existing is None, f"attempting overwrite at {i} when reduction=None"
            if not np.all(np.isnan(existing)):
                return
            self.shards.put(i, shard.copy())
        else:
            if not np.all(np.isnan(existing)):
                if reduction == "sum":
                    self.shards.put(i, ShardBuffer(existing + shard))
                elif reduction == "max":
                    self.shards.put(
                        i, ShardBuffer(np.maximum(existing, shard))
                    )
                elif reduction == "min":
                    self.shards.put(
                        i, ShardBuffer(np.minimum(existing, shard))
                    )
                else:
                    raise ValueError(f"Unknown reduction: {reduction}")
            else:
                self.shards.put(i, shard.copy())
                return

        if auto_forward:
            if direction is None:
                direction = self.scheduler.get_direction(self.rank, i, self.world_size)

            logger.debug(
                f"rank={self.rank} got shard={i}, sending to={'right' if direction == 1 else 'left'}"
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
        for shard in self.shards:
            if shard is None:
                content = ""
            else:
                arr = shard.tolist()
                n = len(arr)
                if n <= MAX_DISPLAY:
                    content = " ".join(
                        f"{x:.3f}"
                        if isinstance(x, float) or isinstance(x, np.floating)
                        else str(x)
                        for x in arr
                    )
                else:
                    half = MAX_DISPLAY // 2
                    front = " ".join(
                        f"{x:.3f}"
                        if isinstance(x, float) or isinstance(x, np.floating)
                        else str(x)
                        for x in arr[:half]
                    )
                    back = " ".join(
                        f"{x:.3f}"
                        if isinstance(x, float) or isinstance(x, np.floating)
                        else str(x)
                        for x in arr[-half:]
                    )
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

    @staticmethod
    def get_direction(src_rank: int, dst_rank: int, world_size: int) -> int:
        cw = (dst_rank - src_rank) % world_size
        ccw = (src_rank - dst_rank) % world_size
        return -1 if cw <= ccw else 1

    def schedule(
        self,
        time: int,
        latency: int,
        i: int,
        src: Device,
        dst: Device,
        payload: ShardBuffer,
        metadata: Dict[str, Any],
        auto_forward: bool,
        reduction: Optional[AssociativeReduction],
        retain: bool,
        direction: Optional[int],
    ):
        # Add simulated inflight communication noise to the shard if needed
        if src.add_communication_noise:
            payload = ShardBuffer(
                payload
                + np.random.normal(
                    src.communication_noise_mean,
                    src.communication_noise_std,
                    payload.shape,
                )
            )
        heapq.heappush(
            self._pipe,
            ShardRequest(
                arrival=time + latency,
                i=i,
                src=src,
                dst=dst,
                payload=payload,
                metadata=metadata,  # type: ignore
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
                metadata=req.metadata,  # type: ignore
            )
