import numpy as np


def n_qubits_per_axis(grid_n: int) -> int:
    n = grid_n.bit_length() - 1
    assert 2 ** n == grid_n, "grid_n must be a power of 2"
    return n


def grid_points(N: int, L: float) -> np.ndarray:
    return np.arange(N) * (L / N)


def momentum_values(N: int, L: float) -> np.ndarray:
    return 2 * np.pi * np.fft.fftfreq(N, d=L / N)
