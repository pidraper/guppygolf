
import time
from dataclasses import dataclass

from config import DEFAULT
from game import course
from quantum.circuit import run_turn, TurnParams
import numpy as np


@dataclass
class State:
    lie: tuple
    strokes: int
    hole_idx: int


def won(cfg, landing, hole_idx):
    cj, ck, r = course.hole_region(cfg, hole_idx)
    j, k = landing
    return ((j - cj) ** 2 + (k - ck) ** 2) ** 0.5 <= r


def play_turn(cfg, state, kx, ky, s, n_steps, detector_cells=()):
    dx = cfg.L / cfg.grid_n
    x0, y0 = state.lie[0] * dx, state.lie[1] * dx
    p = TurnParams(
        x0=x0,
        y0=y0,
        kx=kx,
        ky=ky,
        s=s,
        n_steps=n_steps,
        hole_idx=state.hole_idx,
        detector_cells=detector_cells,
    )
    t0 = time.perf_counter()
    r = run_turn(cfg, p)
    wall = time.perf_counter() - t0
    print(f"  (turn wall time: {wall:.2f}s  -- compile + 1 Selene shot)")
    new = State(lie=r.landing, strokes=state.strokes + 1, hole_idx=r.final_hole_idx)
    return r, new


def main():
    cfg = DEFAULT
    st = State(lie=(8, 8), strokes=0, hole_idx=0)
    cells = course.pick_detector_cells(cfg, np.random.default_rng(0), st.lie)
    while st.strokes < cfg.max_strokes:
        r, st = play_turn(
            cfg,
            st,
            kx=20.0,
            ky=20.0,
            s=1.0,
            n_steps=cfg.n_steps_default,
            detector_cells=cells,
        )
        print(
            f"stroke {st.strokes}: landed {st.lie}, hole {st.hole_idx}, "
            f"detections {r.detections}"
        )
        if won(cfg, st.lie, st.hole_idx):
            print("IN THE HOLE!")
            return
    print("Out of strokes.")


if __name__ == "__main__":
    main()
