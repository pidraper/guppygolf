
from guppylang import guppy
from guppylang.std.builtins import array, comptime
from guppylang.std.quantum import qubit, h, rz, crz, cx, x, discard_array
from guppylang.std.angles import pi
from guppylang.std.debug import state_result
from guppylang.std.mem import mem_swap


def cphase(c, t, phi):
    rz(c, phi / 2)
    crz(c, t, phi)








def qft(reg, n):
    for i in range(n):
        h(reg[i])
        for j in range(i + 1, n):
            cphase(reg[j], reg[i], pi / 2 ** (j - i))
    for i in range(n // 2):
        mem_swap(reg[i], reg[n - 1 - i])


def iqft(reg, n):
    for i in range(n // 2):
        mem_swap(reg[i], reg[n - 1 - i])
    for i in reversed(range(n)):
        for j in range(i + 1, n):
            cphase(reg[j], reg[i], -pi / 2 ** (j - i))
        h(reg[i])


def make_qft_on_basis_state(j0, n):

    @guppy.comptime
    def body(reg: array[qubit, comptime(n)]) -> None:
        for i in range(n):
            if (j0 >> (n - 1 - i)) & 1:
                x(reg[i])
        qft(reg, n)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(n)))
        body(qs)
        state_result("psi", qs)
        discard_array(qs)

    return circuit


def make_qft_iqft_roundtrip(j0, n):

    @guppy.comptime
    def body(reg: array[qubit, comptime(n)]) -> None:
        for i in range(n):
            if (j0 >> (n - 1 - i)) & 1:
                x(reg[i])
        qft(reg, n)
        iqft(reg, n)

    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(n)))
        body(qs)
        state_result("psi", qs)
        discard_array(qs)

    return circuit
