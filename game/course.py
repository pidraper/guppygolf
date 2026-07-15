
import numpy as np
from quantum.grid import grid_points


def _coords(cfg):
    x = grid_points(cfg.grid_n, cfg.L)
    return np.meshgrid(x, x, indexing="ij")


def _flat(cfg, X, Y):
    return np.zeros_like(X)


def _bowl(cfg, X, Y):
    c = cfg.L / 2

    return 120.0 * ((X - c) ** 2 + (Y - c) ** 2)


def _barrier(cfg, X, Y):
    c = cfg.L / 2


    return 52.0 * np.exp(-((X - c) ** 2) / (2 * (cfg.L / 30) ** 2))


def _double_well(cfg, X, Y):
    c = cfg.L / 2
    return 300.0 * (((X - c) ** 2 - (cfg.L / 6) ** 2) ** 2) / (cfg.L**2)


def _bowl_barrier(cfg, X, Y):

    return _bowl(cfg, X, Y) + _barrier(cfg, X, Y)


PRESETS = {
    "flat": _flat,
    "bowl": _bowl,
    "barrier": _barrier,
    "double_well": _double_well,
    "bowl_barrier": _bowl_barrier,
}


def _boundary_wall(cfg, X, Y):
    edge = cfg.L / cfg.grid_n
    d = np.minimum.reduce([X, Y, cfg.L - X - edge, cfg.L - Y - edge])
    return 600.0 * np.exp(
        -d / (cfg.L / 20)
    )


def hole_region(cfg, hole_idx):
    cj, ck = cfg.burrows[hole_idx % len(cfg.burrows)]
    return cj, ck, cfg.r_hole


def _hole_dip(cfg, hole_idx, X, Y):
    cj, ck, _ = hole_region(cfg, hole_idx)
    dx = cfg.L / cfg.grid_n
    cx, cy = cj * dx, ck * dx
    width = 2.0 * dx

    return -150.0 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * width**2))


def build_potential(cfg, hole_idx):
    X, Y = _coords(cfg)
    V = PRESETS[cfg.potential_preset](cfg, X, Y)
    if cfg.boundary_wall:
        V = V + _boundary_wall(cfg, X, Y)
    V = V + _hole_dip(cfg, hole_idx, X, Y)
    return V


def assert_no_overlap(cfg):
    j_lo, j_hi, k_lo, k_hi = cfg.detector_rect
    for cj, ck in cfg.burrows:
        nj = min(max(cj, j_lo), j_hi)
        nk = min(max(ck, k_lo), k_hi)
        dist = ((cj - nj) ** 2 + (ck - nk) ** 2) ** 0.5
        assert dist > cfg.r_hole, f"burrow {(cj, ck)} overlaps detector rect"


def pick_detector_cells(cfg, rng, lie, n_points=None):
    n = cfg.detector_n_points if n_points is None else n_points
    N = cfg.grid_n
    r2 = cfg.r_hole ** 2
    centers = list(cfg.burrows) + [tuple(lie)]

    def ok(j, k):
        return all((j - cj) ** 2 + (k - ck) ** 2 > r2 for (cj, ck) in centers)

    cells = [(j, k) for j in range(N) for k in range(N) if ok(j, k)]
    n = min(n, len(cells))
    picks = rng.choice(len(cells), size=n, replace=False)
    return tuple(cells[int(i)] for i in picks)
