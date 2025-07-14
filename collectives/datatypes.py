from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

class AssociativeReduction(str, Enum):
    SUM = "sum"
    MAX = "max"
    MIN = "min"

@dataclass(order=True)
class ShardRequest:
    arrival: int = field(compare=True)
    i: int = field(compare=False)
    src: "Device" = field(compare=False)  # Forward reference
    dst: "Device" = field(compare=False)
    payload: Optional[np.ndarray] = field(compare=False)
    auto_forward: bool = field(compare=False)
    reduction: Optional[AssociativeReduction] = field(compare=False)
    retain: bool = field(compare=False)
    direction: Optional[int] = field(compare=False) 