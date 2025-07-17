import numpy as np

from collectives import (
    AssociativeReduction,
    all_gather,
    reduce_scatter,
    reset_state_allgather,
    reset_state_reduce_scatter,
)
from collectives.helpers import create_node, visualize_topology


def main():
    WORLD_SIZE = 4
    SHARD_SIZE = 8
    HOP_LATENCY = 1

    print("\n=== Demo: Reduce Scatter (SUM) ===")
    scheduler, device_list = create_node(WORLD_SIZE, SHARD_SIZE, HOP_LATENCY)
    reset_state_reduce_scatter(device_list, WORLD_SIZE, SHARD_SIZE)
    print("Initial topology (Reduce Scatter):")
    visualize_topology(device_list[0])
    original_state = np.array([np.array(device.shards.data) for device in device_list])
    reduce_scatter(
        reduction=AssociativeReduction.SUM,
        scheduler=scheduler,
        device_list=device_list,
        world_size=WORLD_SIZE,
        original_state=original_state,
        hop_latency=HOP_LATENCY,
        shard_size=SHARD_SIZE,
    )
    print("Topology after Reduce Scatter (SUM):")
    visualize_topology(device_list[0])

    print("\n=== Demo: All Gather ===")
    scheduler, device_list = create_node(WORLD_SIZE, SHARD_SIZE, HOP_LATENCY)
    reset_state_allgather(device_list, WORLD_SIZE, SHARD_SIZE)
    print("Initial topology (All Gather):")
    visualize_topology(device_list[0])
    original_state = np.array([np.array(device.shards.data) for device in device_list])
    all_gather(
        scheduler=scheduler,
        device_list=device_list,
        world_size=WORLD_SIZE,
        original_state=original_state,
        hop_latency=HOP_LATENCY,
    )
    print("Topology after All Gather:")
    visualize_topology(device_list[0])

    # Noisy demos using new create_node noisy_k argument
    noise_settings = [
        (1, 0.0, 0.01),
        (2, 0.0, 0.1),
        (1, 0.5, 0.2),
    ]
    for noisy_k, mean, std in noise_settings:
        print(f"\n=== Demo: Reduce Scatter (SUM) with {noisy_k} Noisy Device(s) (mean={mean}, std={std}) ===")
        scheduler, device_list = create_node(WORLD_SIZE, SHARD_SIZE, HOP_LATENCY, noisy_k=noisy_k, noise_mean=mean, noise_std=std)
        noisy_indices = [i for i, d in enumerate(device_list) if getattr(d, 'add_communication_noise', False)]
        print(f"Noisy device indices: {noisy_indices}")
        reset_state_reduce_scatter(device_list, WORLD_SIZE, SHARD_SIZE)
        print("Initial topology (Reduce Scatter, noisy):")
        visualize_topology(device_list[0])
        original_state = np.array([np.array(device.shards.data) for device in device_list])
        reduce_scatter(
            reduction=AssociativeReduction.SUM,
            scheduler=scheduler,
            device_list=device_list,
            world_size=WORLD_SIZE,
            original_state=original_state,
            hop_latency=HOP_LATENCY,
            shard_size=SHARD_SIZE,
        )
        print("Topology after Reduce Scatter (SUM, noisy):")
        visualize_topology(device_list[0])

    for noisy_k, mean, std in noise_settings:
        print(f"\n=== Demo: All Gather with {noisy_k} Noisy Device(s) (mean={mean}, std={std}) ===")
        scheduler, device_list = create_node(WORLD_SIZE, SHARD_SIZE, HOP_LATENCY, noisy_k=noisy_k, noise_mean=mean, noise_std=std)
        noisy_indices = [i for i, d in enumerate(device_list) if getattr(d, 'add_communication_noise', False)]
        print(f"Noisy device indices: {noisy_indices}")
        reset_state_allgather(device_list, WORLD_SIZE, SHARD_SIZE)
        print("Initial topology (All Gather, noisy):")
        visualize_topology(device_list[0])
        original_state = np.array([np.array(device.shards.data) for device in device_list])
        all_gather(
            scheduler=scheduler,
            device_list=device_list,
            world_size=WORLD_SIZE,
            original_state=original_state,
            hop_latency=HOP_LATENCY,
        )
        print("Topology after All Gather (noisy):")
        visualize_topology(device_list[0])

if __name__ == "__main__":
    main()
