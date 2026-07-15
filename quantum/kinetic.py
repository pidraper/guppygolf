
import math
import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import qubit, h, rz, crz, discard_array
from guppylang.std.angles import pi
from guppylang.std.debug import state_result
from quantum.qft import cphase


def kinetic_coeffs(N, dt, mass, L):
    n = N.bit_length() - 1
    c = (2 * np.pi / L) ** 2 * dt / (2 * mass)
    v = [2 ** (n - 1 - b) for b in range(n)]
    sign = 0
    alpha = [0.0] * n
    beta = [[0.0] * n for _ in range(n)]
    for b in range(n):
        alpha[b] += -c * v[b] ** 2
    for b in range(n):
        for cc in range(b + 1, n):
            beta[b][cc] += -c * 2 * v[b] * v[cc]
    for b in range(n):
        coef = -c * (-2 * N * v[b])
        if b == sign:
            alpha[b] += coef
        else:
            lo, hi = min(b, sign), max(b, sign)
            beta[lo][hi] += coef
    alpha[sign] += -c * (N * N)
    return alpha, beta


def apply_kinetic(reg, alpha, beta, n):
    for b in range(n):
        rz(reg[b], pi * comptime(alpha[b] / math.pi))
    for b in range(n):
        for c in range(b + 1, n):
            cphase(reg[b], reg[c], pi * comptime(beta[b][c] / math.pi))


def make_kinetic_test_circuit(N, dt, mass, L):
    n = N.bit_length() - 1
    alpha, beta = kinetic_coeffs(N, dt, mass, L)

    @guppy.comptime
    def stage(reg: array[qubit, comptime(n)]) -> None:
        for i in range(n):
            h(reg[i])
        apply_kinetic(reg, alpha, beta, n)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(n)))
        stage(qs)
        state_result("psi", qs)
        discard_array(qs)

    return circuit
