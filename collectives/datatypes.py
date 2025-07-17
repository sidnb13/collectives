from ast import Dict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .shardbuffer import ShardBuffer

if TYPE_CHECKING:
    from .api import Device


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
    payload: Optional[ShardBuffer] = field(compare=False)
    metadata: Optional[Dict] = field(compare=False)
    auto_forward: bool = field(compare=False)
    reduction: Optional[AssociativeReduction] = field(compare=False)
    retain: bool = field(compare=False)
    direction: Optional[int] = field(compare=False)
