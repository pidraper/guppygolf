
import math
import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import qubit, h, rz, cx, discard_array
from guppylang.std.angles import pi
from guppylang.std.debug import state_result


def walsh_terms(phi_by_position, n, tol):
    N = 2**n
    phi = np.asarray(phi_by_position, dtype=float)

    def signfor(mask):
        s = np.ones(N)
        for a in range(n):
            if mask & (1 << a):
                s = s * np.array([(-1.0) ** ((j >> (n - 1 - a)) & 1) for j in range(N)])
        return s

    coeffs = sorted(
        ((m, np.mean(phi * signfor(m))) for m in range(N)), key=lambda t: -abs(t[1])
    )
    recon = np.zeros(N)
    kept = []
    for mask, c in coeffs:
        if c == 0.0:
            continue
        recon = recon + c * signfor(mask)
        kept.append((mask, float(c)))
        if np.sqrt(np.mean((recon - phi) ** 2)) < tol:
            break
    return kept


def apply_diagonal_phase(reg, terms, n):

    for mask, c in terms:
        bits = [a for a in range(n) if (mask >> a) & 1]
        if not bits:
            continue
        for i in range(len(bits) - 1):
            cx(reg[bits[i]], reg[bits[i + 1]])
        rz(reg[bits[-1]], pi * comptime((-2 * c) / math.pi))
        for i in reversed(range(len(bits) - 1)):
            cx(reg[bits[i]], reg[bits[i + 1]])


def make_potential_test_circuit(phi_by_position, n, tol):
    terms = walsh_terms(phi_by_position, n, tol)

    @guppy.comptime
    def stage(reg: array[qubit, comptime(n)]) -> None:
        for i in range(n):
            h(reg[i])
        apply_diagonal_phase(reg, terms, n)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(n)))
        stage(qs)
        state_result("psi", qs)
        discard_array(qs)

    return circuit
