from dataclasses import dataclass
import numpy as np
from pathlib import Path
from guppylang.emulator.state import NotSingleStateError, PartialVector
from selene_quest_plugin.state import SeleneQuestState
from guppylang import guppy
from guppylang.std.builtins import array, comptime, output
from guppylang.std.quantum import (
    qubit,
    measure,
    measure_array,
    discard_array,
    collect_measurements,
)
from guppylang.std.debug import state_output
from guppylang.emulator._args import validate_record
from guppylang.emulator.instance import _to_provider_args
from selene_argreader_plugin import ArgProvider

from quantum.grid import n_qubits_per_axis
from quantum.state_prep import (
    gaussian_amplitudes,
    prep_structure,
    prep_angle_units,
    kick_units,
    apply_prep_angles,
    apply_kick_angles,
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


WALSH_TOL = 1e-2


def walsh_term_counts(cfg):
    n = n_qubits_per_axis(cfg.grid_n)
    counts = []
    for b in range(len(cfg.burrows)):
        Vb = build_potential(cfg, b)
        phi = (-np.asarray(Vb) * cfg.dt / 2.0).reshape(-1).tolist()
        counts.append(len(walsh_terms(phi, 2 * n, WALSH_TOL)))
    return counts


@dataclass
class Course:

    cfg: object
    inst: object
    meta: dict


def build_course(cfg, detector_cells, n_steps=None, warmup=False):
    n = n_qubits_per_axis(cfg.grid_n)
    N = cfg.grid_n
    nb = len(cfg.burrows)
    assert nb == 3, "this reference body is unrolled for exactly 3 burrows"
    n_steps = cfg.n_steps_default if n_steps is None else n_steps
    period = cfg.detector_period
    offset = cfg.detector_offset
    detector_cells = tuple(tuple(c) for c in detector_cells)
    conditions = point_conditions(detector_cells, n)
    K = 2**n - 1
    structure = prep_structure(n)


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
    def prep_and_kick(
        reg: array[qubit, comptime(2 * n)],
        px_angles: array[float, comptime(K)],
        py_angles: array[float, comptime(K)],
        kickx: array[float, comptime(n)],
        kicky: array[float, comptime(n)],
    ) -> None:
        apply_prep_angles(reg[0:n], structure, px_angles)
        apply_prep_angles(reg[n : 2 * n], structure, py_angles)
        apply_kick_angles(reg[0:n], kickx, n)
        apply_kick_angles(reg[n : 2 * n], kicky, n)

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
    def circuit(
        px_angles: array[float, comptime(K)],
        py_angles: array[float, comptime(K)],
        kickx: array[float, comptime(n)],
        kicky: array[float, comptime(n)],
        hole0: int,
    ) -> None:
        qs = array(qubit() for _ in range(comptime(2 * n)))
        prep_and_kick(qs, px_angles, py_angles, kickx, kicky)
        hole_idx: int = hole0
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
                detected = measure(flag).read()
                if detected:
                    hole_idx = (hole_idx + 1) % comptime(nb)
                output("det", detected)
                discard_array(anc)
            output("hole", hole_idx)
            state_output("snap", qs)
        output("hole_final", hole_idx)
        output("land", collect_measurements(measure_array(qs)))

    meta = dict(
        n=n,
        N=N,
        nb=nb,
        n_steps=n_steps,
        period=period,
        n_qubits=n_qubits,
        detector_steps=[t for t in range(n_steps) if t % period == offset],
        detector_cells=detector_cells,
    )
    inst = circuit.emulator(n_qubits=n_qubits).with_shots(1)
    course = Course(cfg=cfg, inst=inst, meta=meta)
    if warmup:
        zeros = dict(
            px_angles=[0.0] * K,
            py_angles=[0.0] * K,
            kickx=[0.0] * n,
            kicky=[0.0] * n,
            hole0=0,
        )
        inst.with_seed(0).run(**zeros)
    return course


def _snapshot_amp(vec, N):
    try:
        amp = np.asarray(vec.as_single_state(), dtype=complex)
    except NotSingleStateError:
        dist = vec.state_distribution()
        amp = np.asarray(max(dist, key=lambda d: d.probability).state, dtype=complex)
    return amp.reshape(N, N)














































@dataclass
class StepFrame:

    prob: np.ndarray
    phase: np.ndarray
    hole_idx: int
    det: "bool | None"


@dataclass
class TurnEnd:

    landing: tuple
    final_hole_idx: int


def turn_args(cfg, params):
    assert params.s > 0, "squash factor s must be positive"
    n = n_qubits_per_axis(cfg.grid_n)
    root_s = params.s**0.5
    return dict(
        px_angles=prep_angle_units(
            gaussian_amplitudes(cfg.grid_n, params.x0, cfg.sigma_0 / root_s, cfg.L)
        ),
        py_angles=prep_angle_units(
            gaussian_amplitudes(cfg.grid_n, params.y0, cfg.sigma_0 * root_s, cfg.L)
        ),
        kickx=kick_units(params.kx, cfg.L, n),
        kicky=kick_units(params.ky, cfg.L, n),
        hole0=int(params.hole_idx),
    )


def stream_turn(course, params, seed=1):
    meta = course.meta
    assert params.n_steps == meta["n_steps"], (
        f"course compiled for n_steps={meta['n_steps']}, stroke asks {params.n_steps}"
    )
    assert tuple(tuple(c) for c in params.detector_cells) == meta["detector_cells"], (
        "course compiled for a different detector layout"
    )
    args = turn_args(course.cfg, params)
    n, N = meta["n"], meta["N"]
    inst = course.inst.with_seed(seed)



    validate_record(inst._arg_specs, args)
    provider = ArgProvider()
    provider.set_constant_args(**_to_provider_args(args))



    with provider:


        stream = inst._run_instance()
        shot = next(iter(stream))
        det = None
        hole = None
        final_hole = None
        try:
            for tag, value in shot:
                if tag == "det":
                    det = bool(value)
                elif tag == "hole":
                    hole = int(value)
                elif tag == "STATE:snap":
                    state = SeleneQuestState.parse_from_file(Path(value), cleanup=True)
                    amp = _snapshot_amp(PartialVector._from_inner(state), N)
                    yield StepFrame(
                        prob=np.abs(amp) ** 2,
                        phase=stabilized_phase(amp),
                        hole_idx=hole,
                        det=det,
                    )
                    det = None
                elif tag == "hole_final":
                    final_hole = int(value)
                elif tag == "land":
                    bits = [int(b) for b in value]
                    jx = int("".join(str(b) for b in bits[:n]), 2)
                    jy = int("".join(str(b) for b in bits[n:]), 2)
                    yield TurnEnd(landing=(jx, jy), final_hole_idx=final_hole)
        finally:


            shot.close()
            stream.close()













































def run_turn(cfg, params, seed=1):
    course = build_course(cfg, params.detector_cells, n_steps=params.n_steps)
    items = list(stream_turn(course, params, seed))
    frames = [f for f in items if isinstance(f, StepFrame)]
    tail = items[-1]
    assert isinstance(tail, TurnEnd), "complete flight must end in TurnEnd"
    return TurnResult(
        snapshots=[f.prob for f in frames],
        phase_snapshots=[f.phase for f in frames],
        landing=tail.landing,
        final_hole_idx=tail.final_hole_idx,
        detections=[f.det for f in frames if f.det is not None],


        detect_at_step=[i for i, f in enumerate(frames) if f.det is not None],
        hole_trace=[f.hole_idx for f in frames],
    )
