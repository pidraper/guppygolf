from dataclasses import dataclass
import numpy as np


@dataclass
class TurnParams:
    x0: float
    y0: float
    kx: float
    ky: float
    s: float
    n_steps: int
    hole_idx: int
    detector_cells: tuple = ()


@dataclass
class TurnResult:
    snapshots: list
    landing: tuple
    final_hole_idx: int
    detections: list
    detect_at_step: list
    hole_trace: list
    phase_snapshots: list = (
        None
    )


def stabilized_phase(amp2d):
    ph = np.angle(amp2d)
    ref = ph.flat[int(np.argmax(np.abs(amp2d)))]
    return ph - ref


import numpy as np
from guppylang import guppy
from guppylang.std.builtins import array, comptime, result
from guppylang.std.quantum import qubit, measure, measure_array, discard_array
from guppylang.std.debug import state_result
from guppylang.emulator.state import NotSingleStateError

from quantum.grid import n_qubits_per_axis
from quantum.state_prep import (
    gaussian_amplitudes,
    prepare_amplitudes_gates,
    apply_prep_gates,
    apply_kick,
)
from quantum.qft import qft, iqft
from quantum.kinetic import kinetic_coeffs, apply_kinetic
from quantum.potential import walsh_terms, apply_diagonal_phase
from quantum.detectors import (
    detector_flip_q,
    dyadic_conditions,
    detector_flip_box,
    mcx_into_ancillas,
    point_conditions,
)
from game.course import build_potential

WALSH_TOL = 1e-2


def walsh_term_counts(cfg):
    n = n_qubits_per_axis(cfg.grid_n)
    counts = []
    for b in range(len(cfg.burrows)):
        Vb = build_potential(cfg, b)
        phi = (-np.asarray(Vb) * cfg.dt / 2.0).reshape(-1).tolist()
        counts.append(len(walsh_terms(phi, 2 * n, WALSH_TOL)))
    return counts


def build_turn(
    cfg,
    x0,
    y0,
    kx,
    ky,
    sigma_x,
    sigma_y,
    conditions,
    n_steps,
    detector_period,
    detector_offset,
):
    n = n_qubits_per_axis(cfg.grid_n)
    N = cfg.grid_n
    nb = len(cfg.burrows)
    assert nb == 3, "this reference body is unrolled for exactly 3 burrows"
    period = detector_period
    offset = detector_offset


    px = prepare_amplitudes_gates(gaussian_amplitudes(N, x0, sigma_x, cfg.L))
    py = prepare_amplitudes_gates(gaussian_amplitudes(N, y0, sigma_y, cfg.L))
    alpha, beta = kinetic_coeffs(N, cfg.dt, cfg.mass, cfg.L)

    def vhalf_terms(b):
        Vb = build_potential(cfg, b)
        phi = (-np.asarray(Vb) * cfg.dt / 2.0).reshape(-1).tolist()
        return walsh_terms(phi, 2 * n, WALSH_TOL)

    vhalf0_terms = vhalf_terms(0)
    vhalf1_terms = vhalf_terms(1)
    vhalf2_terms = vhalf_terms(2)


    n_anc = max((mcx_into_ancillas(len(tile)) for tile in conditions), default=0)
    n_qubits = 2 * n + 1 + n_anc


    @guppy.comptime
    def prep_and_kick(reg: array[qubit, comptime(2 * n)]) -> None:
        apply_prep_gates(reg[0:n], px)
        apply_prep_gates(reg[n : 2 * n], py)
        apply_kick(reg[0:n], kx, cfg.L, n)
        apply_kick(reg[n : 2 * n], ky, cfg.L, n)

    @guppy.comptime
    def vhalf0(reg: array[qubit, comptime(2 * n)]) -> None:
        apply_diagonal_phase(reg, vhalf0_terms, 2 * n)

    @guppy.comptime
    def vhalf1(reg: array[qubit, comptime(2 * n)]) -> None:
        apply_diagonal_phase(reg, vhalf1_terms, 2 * n)

    @guppy.comptime
    def vhalf2(reg: array[qubit, comptime(2 * n)]) -> None:
        apply_diagonal_phase(reg, vhalf2_terms, 2 * n)

    @guppy.comptime
    def kinetic_step(
        reg: array[qubit, comptime(2 * n)],
    ) -> None:
        qft(reg[0:n], n)
        apply_kinetic(reg[0:n], alpha, beta, n)
        iqft(reg[0:n], n)
        qft(reg[n : 2 * n], n)
        apply_kinetic(reg[n : 2 * n], alpha, beta, n)
        iqft(reg[n : 2 * n], n)

    @guppy.comptime
    def detector_emit(
        reg: array[qubit, comptime(2 * n)],
        flag: qubit,
        anc: array[qubit, comptime(n_anc)],
    ) -> None:
        detector_flip_box(reg, conditions, flag, anc)


    @guppy
    def circuit() -> None:
        qs = array(qubit() for _ in range(comptime(2 * n)))
        prep_and_kick(qs)
        hole_idx: int = 0
        for t in range(comptime(n_steps)):

            if hole_idx == 0:
                vhalf0(qs)
            elif hole_idx == 1:
                vhalf1(qs)
            elif hole_idx == 2:
                vhalf2(qs)

            kinetic_step(qs)

            if hole_idx == 0:
                vhalf0(qs)
            elif hole_idx == 1:
                vhalf1(qs)
            elif hole_idx == 2:
                vhalf2(qs)

            if (t % comptime(period)) == comptime(offset):
                flag = qubit()
                anc = array(qubit() for _ in range(comptime(n_anc)))
                detector_emit(qs, flag, anc)
                detected = measure(flag)
                if detected:
                    hole_idx = (hole_idx + 1) % comptime(nb)
                result("det", detected)
                discard_array(anc)
            result("hole", hole_idx)
            state_result("snap", qs)
        result("hole_final", hole_idx)
        result("land", measure_array(qs))

    meta = dict(
        n=n,
        N=N,
        nb=nb,
        n_steps=n_steps,
        period=period,
        n_qubits=n_qubits,
        detector_steps=[t for t in range(n_steps) if t % period == 0],
    )
    return circuit, meta


def _snapshot_amp(vec, N):
    try:
        amp = np.asarray(vec.as_single_state(), dtype=complex)
    except NotSingleStateError:
        dist = vec.state_distribution()
        amp = np.asarray(max(dist, key=lambda d: d.probability).state, dtype=complex)
    return amp.reshape(N, N)


def _run_turn_raw(circuit, meta, seed=1):
    n, N = meta["n"], meta["N"]
    res = (
        circuit.emulator(n_qubits=meta["n_qubits"]).with_seed(seed).with_shots(1).run()
    )
    shot = res.collated_shots()[0]


    snaps = []
    phases = []
    for _tag, vec in res.partial_states()[0]:
        amp = _snapshot_amp(
            vec, N
        )
        snaps.append(np.abs(amp) ** 2)
        phases.append(stabilized_phase(amp))

    det_trace = [int(d) for d in shot["det"]]
    detections = [bool(d) for d in det_trace]
    detect_at_step = [meta["detector_steps"][i] for i, d in enumerate(det_trace) if d]
    hole_trace = [int(h) for h in shot["hole"]]
    final_hole_idx = int(shot["hole_final"][0])

    bits = [int(b) for b in shot["land"][0]]
    jx = int("".join(str(b) for b in bits[:n]), 2)
    jy = int("".join(str(b) for b in bits[n:]), 2)

    return dict(
        snapshots=snaps,
        phase_snapshots=phases,
        landing=(jx, jy),
        detections=detections,
        detect_at_step=detect_at_step,
        hole_trace=hole_trace,
        final_hole_idx=final_hole_idx,
    )


def run_turn(cfg, params, seed=1):
    assert params.s > 0, "squash factor s must be positive"
    n = n_qubits_per_axis(cfg.grid_n)
    conditions = point_conditions(params.detector_cells, n)
    root_s = params.s**0.5
    sigma_x = cfg.sigma_0 / root_s
    sigma_y = cfg.sigma_0 * root_s
    circuit, meta = build_turn(
        cfg,
        params.x0,
        params.y0,
        params.kx,
        params.ky,
        sigma_x,
        sigma_y,
        conditions,
        params.n_steps,
        cfg.detector_period,
        cfg.detector_offset,
    )
    d = _run_turn_raw(circuit, meta, seed)
    return TurnResult(
        snapshots=d["snapshots"],
        phase_snapshots=d["phase_snapshots"],
        landing=d["landing"],
        final_hole_idx=d["final_hole_idx"],
        detections=d["detections"],


        detect_at_step=meta["detector_steps"],
        hole_trace=d["hole_trace"],
    )
