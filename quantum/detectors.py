
from guppylang import guppy
from guppylang.std.builtins import array, comptime, result
from guppylang.std.quantum import (
    qubit,
    x,
    cx,
    toffoli,
    measure,
    measure_array,
    discard_array,
)
from quantum.grid import n_qubits_per_axis


def mcx(reg, controls, target, anc):
    k = len(controls)
    if k == 1:
        cx(reg[controls[0]], reg[target])
        return
    if k == 2:
        toffoli(reg[controls[0]], reg[controls[1]], reg[target])
        return
    toffoli(reg[controls[0]], reg[controls[1]], reg[anc[0]])
    for i in range(2, k - 1):
        toffoli(reg[controls[i]], reg[anc[i - 2]], reg[anc[i - 1]])
    toffoli(reg[controls[k - 1]], reg[anc[k - 3]], reg[target])
    for i in reversed(range(2, k - 1)):
        toffoli(reg[controls[i]], reg[anc[i - 2]], reg[anc[i - 1]])
    toffoli(reg[controls[0]], reg[controls[1]], reg[anc[0]])


def _kg24_ladder(k):
    n = k + 1
    ops = []
    for i in range(2, n - 2, 2):
        ops.append(("ccx", i + 1, i + 2, i))
        ops.append(("x", i))
    if n % 2 != 0:
        a, b, target = n - 3, n - 5, n - 6
    else:
        a, b, target = n - 1, n - 4, n - 5
    if target > 0:
        ops.append(("ccx", a, b, target))
        ops.append(("x", target))
    for i in range(target, 2, -2):
        ops.append(("ccx", i, i - 1, i - 2))
        ops.append(("x", i - 2))
    return ops, max(0, 6 - n)


def mcx_into_ancillas(k):
    return 1 if k >= 3 else 0


def mcx_into(reg, controls, target, anc):
    k = len(controls)
    if k == 0:
        x(target)
        return
    if k == 1:
        cx(reg[controls[0]], target)
        return
    if k == 2:
        toffoli(reg[controls[0]], reg[controls[1]], target)
        return

    ladder, final_ctrl = _kg24_ladder(k)
    toffoli(
        reg[controls[0]], reg[controls[1]], anc[0]
    )
    for op in ladder:
        if op[0] == "x":
            x(reg[controls[op[1] - 1]])
        else:
            toffoli(
                reg[controls[op[1] - 1]],
                reg[controls[op[2] - 1]],
                reg[controls[op[3] - 1]],
            )
    toffoli(anc[0], reg[controls[final_ctrl]], target)
    for op in reversed(ladder):
        if op[0] == "x":
            x(reg[controls[op[1] - 1]])
        else:
            toffoli(
                reg[controls[op[1] - 1]],
                reg[controls[op[2] - 1]],
                reg[controls[op[3] - 1]],
            )
    toffoli(reg[controls[0]], reg[controls[1]], anc[0])


def detector_check(reg, conditions, flag, anc):
    detector_flip(reg, conditions, flag, anc)
    return measure(reg[flag])


def detector_flip(reg, conditions, flag, anc):
    controls = [q for (q, _v) in conditions]
    zeros = [q for (q, v) in conditions if v == 0]
    for q in zeros:
        x(reg[q])
    mcx(reg, controls, flag, anc)
    for q in zeros:
        x(reg[q])


def detector_flip_q(reg, conditions, flag, anc):
    zeros = [idx for (idx, val) in conditions if val == 0]
    controls = [idx for (idx, _val) in conditions]
    for idx in zeros:
        x(reg[idx])
    mcx_into(reg, controls, flag, anc)
    for idx in zeros:
        x(reg[idx])


def point_conditions(cells, n):
    return [dyadic_conditions(j, j, k, k, n) for (j, k) in cells]


def detector_flip_box(reg, tiles, flag, anc):
    for tile in tiles:
        detector_flip_q(reg, tile, flag, anc)


def make_detector_probe_circuit(cfg, j, k, hole_idx, conditions):
    n = n_qubits_per_axis(cfg.grid_n)
    kc = len(conditions)
    n_anc = max(0, kc - 2)
    nb = len(cfg.burrows)
    h0 = hole_idx

    @guppy.comptime
    def seed_and_flip(
        reg: array[qubit, comptime(2 * n)],
        flag: qubit,
        anc: array[qubit, comptime(n_anc)],
    ) -> None:
        for i in range(n):
            if (j >> (n - 1 - i)) & 1:
                x(reg[i])
            if (k >> (n - 1 - i)) & 1:
                x(reg[n + i])
        detector_flip_q(reg, conditions, flag, anc)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(2 * n)))
        flag = qubit()
        anc = array(qubit() for _ in range(comptime(n_anc)))
        seed_and_flip(qs, flag, anc)
        detected = measure(flag)
        hidx: int = comptime(h0)
        if detected:
            hidx = (hidx + 1) % comptime(nb)
        result("detected", detected)
        result("hole_idx", hidx)
        discard_array(qs)
        discard_array(anc)

    return circuit





def make_detector_test_circuit(jx, jy, conditions):
    n = 4
    k = len(conditions)
    flag = 2 * n
    anc = [2 * n + 1 + j for j in range(max(0, k - 2))]
    nq = 2 * n + 1 + max(0, k - 2)

    @guppy.comptime
    def setup(reg: array[qubit, comptime(nq)]) -> None:
        for i in range(n):
            if (jx >> (n - 1 - i)) & 1:
                x(reg[i])
            if (jy >> (n - 1 - i)) & 1:
                x(reg[n + i])
        detector_flip(reg, conditions, flag, anc)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(nq)))
        setup(qs)
        ms = measure_array(qs)
        result("detected", ms[comptime(flag)])

    return circuit





def dyadic_conditions(j_lo, j_hi, k_lo, k_hi, n):

    def fixed_bits(lo, hi, offset):
        size = hi - lo + 1
        assert size & (size - 1) == 0, f"[{lo},{hi}] length not a power of two"
        assert lo % size == 0, f"[{lo},{hi}] not aligned to its size"
        n_fixed = n - size.bit_length() + 1
        return [(offset + i, (lo >> (n - 1 - i)) & 1) for i in range(n_fixed)]

    return fixed_bits(j_lo, j_hi, 0) + fixed_bits(k_lo, k_hi, n)


def make_mcx_test_circuit(pattern):
    controls = [0, 1, 2, 3]
    target = 4
    anc = [5, 6]

    @guppy.comptime
    def setup(reg: array[qubit, comptime(7)]) -> None:
        for i in range(4):
            if pattern[i] == 1:
                x(reg[i])
        mcx(reg, controls, target, anc)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(7)))
        setup(qs)
        ms = measure_array(qs)
        result("target", ms[4])
        result("anc0", ms[5])
        result("anc1", ms[6])

    return circuit





def make_detector_test_circuit(jx, jy, conditions):
    n = 4
    k = len(conditions)
    flag = 2 * n
    anc = [2 * n + 1 + j for j in range(max(0, k - 2))]
    nq = 2 * n + 1 + max(0, k - 2)

    @guppy.comptime
    def setup(reg: array[qubit, comptime(nq)]) -> None:
        for i in range(n):
            if (jx >> (n - 1 - i)) & 1:
                x(reg[i])
            if (jy >> (n - 1 - i)) & 1:
                x(reg[n + i])
        detector_flip(reg, conditions, flag, anc)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(nq)))
        setup(qs)
        ms = measure_array(qs)
        result("detected", ms[comptime(flag)])

    return circuit


def make_detector_probe_circuit(cfg, j, k, hole_idx, conditions):
    n = n_qubits_per_axis(cfg.grid_n)
    kc = len(conditions)
    n_anc = max(0, kc - 2)
    nb = len(cfg.burrows)
    h0 = hole_idx

    @guppy.comptime
    def seed_and_flip(
        reg: array[qubit, comptime(2 * n)],
        flag: qubit,
        anc: array[qubit, comptime(n_anc)],
    ) -> None:
        for i in range(n):
            if (j >> (n - 1 - i)) & 1:
                x(reg[i])
            if (k >> (n - 1 - i)) & 1:
                x(reg[n + i])
        detector_flip_q(reg, conditions, flag, anc)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(2 * n)))
        flag = qubit()
        anc = array(qubit() for _ in range(comptime(n_anc)))
        seed_and_flip(qs, flag, anc)
        detected = measure(flag)
        hidx: int = comptime(h0)
        if detected:
            hidx = (hidx + 1) % comptime(nb)
        result("detected", detected)
        result("hole_idx", hidx)
        discard_array(qs)
        discard_array(anc)

    return circuit
