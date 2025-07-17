from .api import Device, Scheduler
import numpy as np


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


def create_node(
    world_size: int, shard_size: int, hop_latency: int,
    noisy_k: int = 0, noise_mean: float = 0.0, noise_std: float = 0.01
) -> tuple["Scheduler", list["Device"]]:
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

    # Randomly select k devices to be noisy
    if noisy_k > 0:
        noisy_indices = np.random.choice(world_size, size=noisy_k, replace=False)
        for idx in noisy_indices:
            device_list[idx].add_communication_noise = True
            device_list[idx].communication_noise_mean = noise_mean
            device_list[idx].communication_noise_std = noise_std

    return scheduler, device_list
