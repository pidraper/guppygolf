
from dataclasses import dataclass
from enum import Enum, auto


class Phase(Enum):
    AIMING = auto()
    COMPUTING = auto()
    SIMULATING = auto()

    WIN = auto()
    LOSE = auto()
    ERROR = auto()


@dataclass
class GameState:
    lie: tuple = (
        13,
        13,
    )
    hole_idx: int = 0
    strokes: int = 0
    score: int = 0
    phase: Phase = Phase.AIMING

    kx: float = 0.0
    ky: float = 0.0
    s: float = 1.0
    n_steps: int = 12
    message: str = ""
    detector_cells: tuple = ()
