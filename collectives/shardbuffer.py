import hashlib

import numpy as np


class ShardBuffer(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def checksum(self):
        return hashlib.sha1(self.tobytes()).hexdigest()


class ShardedArray:
    def __init__(self, world_size: int, shard_size: int):
        # Each slot is a ShardBuffer initialized with NaNs
        self.data = [
            ShardBuffer(np.full(shard_size, np.nan)) for _ in range(world_size)
        ]

    def reset(self):
        for i in range(len(self.data)):
            self.data[i] = ShardBuffer(np.full(self.data[i].shape, np.nan))

    def put(self, i: int, shard: ShardBuffer):
        self.data[i] = shard.copy()

    def get(self, i: int) -> ShardBuffer:
        return self.data[i]

    def clear(self, i: int):
        self.data[i] = ShardBuffer(np.full(self.data[i].shape, np.nan))

    def __getitem__(self, i: int) -> ShardBuffer:
        return self.get(i)

    def __setitem__(self, i: int, value: ShardBuffer):
        self.put(i, value)
