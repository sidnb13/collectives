import numpy as np

from collectives import (
    AssociativeReduction,
    all_gather,
    create_node,
    reduce_scatter,
    reset_state_allgather,
    reset_state_reduce_scatter,
)


def test_reduce_scatter_sum():
    world_size = 4
    shard_size = 8
    hop_latency = 1
    scheduler, device_list = create_node(world_size, shard_size, hop_latency)
    reset_state_reduce_scatter(device_list, world_size, shard_size)
    original_state = np.array([np.array(device.shards.data) for device in device_list])

    reduce_scatter(
        reduction=AssociativeReduction.SUM,
        scheduler=scheduler,
        device_list=device_list,
        world_size=world_size,
        original_state=original_state,
        hop_latency=hop_latency,
        shard_size=shard_size,
    )

    # After reduce_scatter, each device's shard should be the sum of all initial shards for its slot
    expected = original_state.sum(axis=0)
    for i, device in enumerate(device_list):
        np.testing.assert_allclose(device.shards[i], expected[i])


def test_all_gather():
    world_size = 4
    shard_size = 8
    hop_latency = 1
    scheduler, device_list = create_node(world_size, shard_size, hop_latency)
    reset_state_allgather(device_list, world_size, shard_size)
    original_state = np.array([np.array(device.shards.data) for device in device_list])

    all_gather(
        scheduler=scheduler,
        device_list=device_list,
        world_size=world_size,
        original_state=original_state,
        hop_latency=hop_latency,
    )

    # After all_gather, each device's i-th shard should equal original_state[i][i]
    for i, device in enumerate(device_list):
        for j in range(world_size):
            np.testing.assert_allclose(device.shards[j], original_state[j][j])
