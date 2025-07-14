import numpy as np

from .api import (
    Device,
    Scheduler,
)
from .datatypes import AssociativeReduction


def reset_state_allgather(device_list: list[Device], world_size: int, shard_size: int):
    for i, device in enumerate(device_list):
        device.reset()
        device.put(i, np.ones(shard_size) * i)


def all_gather(
    scheduler: Scheduler,
    device_list: list[Device],
    world_size: int,
    original_state: np.ndarray,
    hop_latency: int,
):
    def stopping_condition():
        device_asserts = [True] * len(device_list)
        for i, device in enumerate(device_list):
            device_asserts[i] = all(
                (device.shards[i] == original_state[i][i]).all()
                for i in range(world_size)
            )
        return all(device_asserts)

    for rank in range(world_size):
        src = device_list[rank]
        if src._dir[rank] == -1:  # cw
            dst = src.left
        else:  # ccw
            dst = src.right

        scheduler.schedule(
            time=0,
            latency=hop_latency,
            i=rank,
            src=src,
            dst=dst,  # type: ignore
            payload=src.shards[rank].copy(),
            auto_forward=True,
            reduction=None,
            retain=True,
            direction=None,  # bidirectional optimal
        )

    scheduler.run(start=device_list[0], stopping_condition=stopping_condition)


def reset_state_reduce_scatter(device_list: list[Device], world_size: int, shard_size: int):
    for i, device in enumerate(device_list):
        device.reset()
        for j in range(world_size):
            device.put(j, np.ones(shard_size) * j)


def reduce_scatter(
    reduction: AssociativeReduction,
    scheduler: Scheduler,
    device_list: list[Device],
    world_size: int,
    original_state: np.ndarray,
    hop_latency: int,
    shard_size: int,
):
    def stopping_condition():
        device_asserts = [True] * len(device_list)
        for i, device in enumerate(device_list):
            device_asserts[i] = all(
                (device.shards[i] == original_state.sum(axis=0)[i]).all()
                for i in range(world_size)
            )
        return all(device_asserts)

    for rank in range(world_size):
        src = device_list[rank]
        dst = src.right

        scheduler.schedule(
            time=0,
            latency=hop_latency,
            i=rank,
            src=src,
            dst=dst,  # type: ignore
            payload=src.shards[rank].copy(),
            auto_forward=True,
            reduction=reduction,
            retain=False,
            direction=1,  # always send to right
        )
        # Clear initial slots
        src.shards.clear(rank)

    scheduler.run(start=device_list[0], stopping_condition=stopping_condition)
