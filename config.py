
from dataclasses import dataclass


@dataclass
class Config:
    grid_n: int = 16
    L: float = 1.0
    mass: float = (
        10.0
    )
    dt: float = 0.005
    n_steps_default: int = 180
    sigma_0: float = (
        0.07
    )


    k_max: float = 32.0
    R_ref_px: float = 300.0
    s_min: float = 0.65
    s_max: float = 1.5
    s_step: float = 1.1
    potential_preset: str = "bowl_barrier"
    boundary_wall: bool = True
    detector_period: int = 10
    detector_offset: int = 5
    detector_n_points: int = 10
    detector_rect: tuple = (
        8,
        11,
        12,
        15,
    )




    burrows: tuple = (
        (2, 2),
        (2, 6),
        (6, 2),
    )

    r_hole: float = 1.5
    max_strokes: int = 10
    frame_ms: int = 60
    streaming: bool = True
    walsh_rms_tol: float = 0.01
    field_gamma: float = (
        1.4
    )
    field_gain: float = 1.0
    bloom_threshold: float = 0.6
    bloom_intensity: float = 1.1


DEFAULT = Config()
