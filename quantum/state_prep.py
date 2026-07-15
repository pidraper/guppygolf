import numpy as np
import quantum.grid as grid
import math
from guppylang import guppy
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import qubit, ry, cx, discard_array, h, rz
from guppylang.std.angles import pi
from guppylang.std.debug import state_result


def gaussian_amplitudes(N, x0, sigma, L) -> list[float]:
    x = grid.grid_points(N, L)
    gauss = np.exp(-((x - x0) ** 2) / (4 * sigma**2))
    norm = np.sum(gauss**2)
    assert norm > 1e-10, "Normalization factor is too small, check your parameters."
    gaussnorm = gauss / np.sqrt(norm)
    assert gaussnorm.max() ** 2 < 0.999, (
        "The Gaussian is too narrow, consider increasing sigma or adjusting x0."
    )
    return (gaussnorm).tolist()


def mottonen_ry_angles(amps) -> list[float]:
    angles = []
    amps = np.asarray(amps, dtype=float)
    recurs(amps, angles)
    return angles


def recurs(amps, angles):
    N = len(amps)
    if N == 1:
        return

    left = amps[: N // 2]
    right = amps[N // 2 :]
    ln = np.linalg.norm(left)
    rn = np.linalg.norm(right)
    theta = 2 * np.arctan2(rn, ln) if (ln or rn) else 0.0
    angles.append(theta)
    recurs(left / ln if ln else np.full(N // 2, 1 / np.sqrt(N // 2)), angles)
    recurs(right / rn if rn else np.full(N // 2, 1 / np.sqrt(N // 2)), angles)


def angles_by_level(amps) -> list[list[float]]:
    amps = np.asarray(amps, dtype=float)
    depth = 0
    levels = [[] for _ in range(int(math.log2(len(amps))))]
    recurs_with_levels(amps, levels, depth)
    return levels


def recurs_with_levels(amps, levels, depth):
    N = len(amps)
    if N == 1:
        return

    left = amps[: N // 2]
    right = amps[N // 2 :]
    ln = np.linalg.norm(left)
    rn = np.linalg.norm(right)
    theta = 2 * np.arctan2(rn, ln) if (ln or rn) else 0.0
    levels[depth].append(theta)
    recurs_with_levels(
        left / ln if ln else np.full(N // 2, 1 / np.sqrt(N // 2)), levels, depth + 1
    )
    recurs_with_levels(
        right / rn if rn else np.full(N // 2, 1 / np.sqrt(N // 2)), levels, depth + 1
    )


def ucry_gates(controls, target, alphas, gates):
    lenalph = len(alphas)
    if lenalph == 1:
        gates.append(("ry", target, alphas[0]))
    else:
        half = lenalph // 2
        a0 = alphas[:half]
        a1 = alphas[half:]
        ta = [(x + y) / 2 for x, y in zip(a0, a1)]
        tb = [(x - y) / 2 for x, y in zip(a0, a1)]
        ucry_gates(controls[1:], target, ta, gates)
        gates.append(("cx", controls[0], target))
        ucry_gates(controls[1:], target, tb, gates)
        gates.append(("cx", controls[0], target))


def prepare_amplitudes_gates(amps) -> list:
    levels = angles_by_level(amps)
    gates = []
    numlev = len(levels)
    for b in range(numlev):
        ucry_gates(list(range(b)), b, levels[b], gates)
    return gates


def apply_prep_gates(reg, gates):
    for op in gates:
        if op[0] == "ry":
            ry(reg[op[1]], pi * comptime(op[2] / math.pi))
        else:
            cx(reg[op[1]], reg[op[2]])


def apply_kick(reg, k, L, n):
    N = 2**n
    for b in range(n):
        rz(reg[b], pi * comptime(k * (L / N) * 2 ** (n - 1 - b) / math.pi))


def make_prep_circuit(amps):
    n = int(round(math.log2(len(amps))))
    gates = prepare_amplitudes_gates(amps)

    @guppy.comptime
    def prep(reg: array[qubit, comptime(n)]) -> None:
        apply_prep_gates(reg, gates)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(n)))
        prep(qs)
        state_result("psi", qs)
        discard_array(qs)

    return circuit


def make_kick_only_circuit(k, L, n):
    @guppy.comptime
    def kick(reg: array[qubit, comptime(n)]) -> None:
        apply_kick(reg, k, L, n)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(n)))
        for i in range(comptime(n)):
            h(qs[i])
        kick(qs)
        state_result("psi", qs)
        discard_array(qs)

    return circuit
