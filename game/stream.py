
from dataclasses import dataclass, field

from quantum.circuit import StepFrame, TurnEnd


@dataclass
class StreamingResult:
    snapshots: list = field(default_factory=list)
    phase_snapshots: list = field(default_factory=list)
    hole_trace: list = field(default_factory=list)
    detect_at_step: list = field(default_factory=list)
    detections: list = field(default_factory=list)
    landing: tuple = None
    final_hole_idx: int = None


def consume(gen, out, abort):
    for item in gen:
        if abort.is_set():
            break
        if isinstance(item, TurnEnd):
            out.landing = item.landing
            out.final_hole_idx = item.final_hole_idx
            break


        if item.det is not None:
            out.detect_at_step.append(len(out.snapshots))
            out.detections.append(item.det)
        out.hole_trace.append(item.hole_idx)
        out.phase_snapshots.append(item.phase)
        out.snapshots.append(item.prob)
