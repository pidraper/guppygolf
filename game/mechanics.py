import math
import numpy as np
from game.state import Phase
from game.headless import won


def n_steps_for(cfg, kx, ky):
    k = math.hypot(kx, ky)
    frac = min(k, cfg.k_max) / cfg.k_max
    n = round(cfg.n_steps_floor + frac * (cfg.n_steps_max - cfg.n_steps_floor))
    return int(min(max(n, cfg.n_steps_floor), cfg.n_steps_max))


def sample_landing(prob, rng):
    N = prob.shape[0]
    p = prob.flatten().astype(float)
    p = p / p.sum()
    idx = int(rng.choice(N * N, p=p))
    j, k = divmod(idx, N)
    return (int(j), int(k))


def resolve(cfg, st, landing, hole_idx):
    st.lie = landing
    st.hole_idx = hole_idx
    st.strokes += 1
    st.score += 1
    if won(cfg, landing, hole_idx):
        st.phase = Phase.WIN
        st.message = f"In the hole! {st.strokes} strokes."
    elif st.strokes >= cfg.max_strokes:
        st.phase = Phase.LOSE
        st.message = "Out of strokes -- press R."
    else:
        st.phase = Phase.AIMING
        st.message = ""
    return st








_GATES_PER_WALSH_TERM = 6.5
_KINETIC_GATES = 135
_PREP_GATES = 40
_KICK_GATES = 20
_DETECTOR_GATES = 30


def gate_count(cfg, n_steps, walsh_counts):
    mean_terms = sum(walsh_counts) / len(walsh_counts)
    per_step = 2 * mean_terms * _GATES_PER_WALSH_TERM + _KINETIC_GATES
    n_checks = n_steps // cfg.detector_period
    total = _PREP_GATES + _KICK_GATES + n_steps * per_step + n_checks * _DETECTOR_GATES
    return int(total)


def tick_counter(shown, target, rate=0.12):
    return shown + (target - shown) * rate
