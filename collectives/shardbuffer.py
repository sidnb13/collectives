import numpy as np


class ShardBuffer:
    def __init__(self, world_size: int, shard_size: int):
        self.data = np.full((world_size, shard_size), np.nan)

    def reset(self):
        self.data.fill(np.nan)

    def put(self, i: int, shard: np.ndarray):
        self.data[i] = shard

    def get(self, i: int) -> np.ndarray:
        return self.data[i]

    def clear(self, i: int):
        self.data[i] = np.full(self.data.shape[1], np.nan)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.get(i)

    def __setitem__(self, i: int, value: np.ndarray):
        self.put(i, value)
