import numpy as np

from collectives import (
    AssociativeReduction,
    all_gather,
    create_node,
    reduce_scatter,
    reset_state_allgather,
    reset_state_reduce_scatter,
)

# Global scheduler and device list
WORLD_SIZE = 4
SHARD_SIZE = 8
HOP_LATENCY = 1
GLOBAL_SCHEDULER, GLOBAL_DEVICE_LIST = create_node(WORLD_SIZE, SHARD_SIZE, HOP_LATENCY)


def test_reduce_scatter_sum():
    reset_state_reduce_scatter(GLOBAL_DEVICE_LIST, WORLD_SIZE, SHARD_SIZE)
    original_state = np.array(
        [np.array(device.shards.data) for device in GLOBAL_DEVICE_LIST]
    )

    reduce_scatter(
        reduction=AssociativeReduction.SUM,
        scheduler=GLOBAL_SCHEDULER,
        device_list=GLOBAL_DEVICE_LIST,
        world_size=WORLD_SIZE,
        original_state=original_state,
        hop_latency=HOP_LATENCY,
        shard_size=SHARD_SIZE,
    )

    expected = original_state.sum(axis=0)
    for i, device in enumerate(GLOBAL_DEVICE_LIST):
        np.testing.assert_allclose(device.shards[i], expected[i], rtol=1e-2, atol=1e-2)


def test_reduce_scatter_max():
    reset_state_reduce_scatter(GLOBAL_DEVICE_LIST, WORLD_SIZE, SHARD_SIZE)
    original_state = np.array(
        [np.array(device.shards.data) for device in GLOBAL_DEVICE_LIST]
    )

    reduce_scatter(
        reduction=AssociativeReduction.MAX,
        scheduler=GLOBAL_SCHEDULER,
        device_list=GLOBAL_DEVICE_LIST,
        world_size=WORLD_SIZE,
        original_state=original_state,
        hop_latency=HOP_LATENCY,
        shard_size=SHARD_SIZE,
    )

    expected = original_state.max(axis=0)
    for i, device in enumerate(GLOBAL_DEVICE_LIST):
        np.testing.assert_allclose(device.shards[i], expected[i], rtol=1e-2, atol=1e-2)


def test_reduce_scatter_min():
    reset_state_reduce_scatter(GLOBAL_DEVICE_LIST, WORLD_SIZE, SHARD_SIZE)
    original_state = np.array(
        [np.array(device.shards.data) for device in GLOBAL_DEVICE_LIST]
    )

    reduce_scatter(
        reduction=AssociativeReduction.MIN,
        scheduler=GLOBAL_SCHEDULER,
        device_list=GLOBAL_DEVICE_LIST,
        world_size=WORLD_SIZE,
        original_state=original_state,
        hop_latency=HOP_LATENCY,
        shard_size=SHARD_SIZE,
    )

    expected = original_state.min(axis=0)
    for i, device in enumerate(GLOBAL_DEVICE_LIST):
        np.testing.assert_allclose(device.shards[i], expected[i], rtol=1e-2, atol=1e-2)


def test_all_gather():
    reset_state_allgather(GLOBAL_DEVICE_LIST, WORLD_SIZE, SHARD_SIZE)
    original_state = np.array(
        [np.array(device.shards.data) for device in GLOBAL_DEVICE_LIST]
    )

    all_gather(
        scheduler=GLOBAL_SCHEDULER,
        device_list=GLOBAL_DEVICE_LIST,
        world_size=WORLD_SIZE,
        original_state=original_state,
        hop_latency=HOP_LATENCY,
    )

    for i, device in enumerate(GLOBAL_DEVICE_LIST):
        for j in range(WORLD_SIZE):
            np.testing.assert_allclose(
                device.shards[j], original_state[j][j], rtol=1e-2, atol=1e-2
            )
