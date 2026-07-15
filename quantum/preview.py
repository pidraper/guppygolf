import numpy as np
from quantum.state_prep import gaussian_amplitudes
from quantum.grid import grid_points


def preview_prob(cfg, x0, y0, s):

    N = cfg.grid_n
    root_s = s**0.5
    ax = np.asarray(gaussian_amplitudes(N, x0, cfg.sigma_0 / root_s, cfg.L))
    ay = np.asarray(gaussian_amplitudes(N, y0, cfg.sigma_0 * root_s, cfg.L))
    return np.outer(ax**2, ay**2)


def preview_field(cfg, x0, y0, s, kx, ky):
    mag = preview_prob(cfg, x0, y0, s)
    x = grid_points(cfg.grid_n, cfg.L)
    phase = kx * x[:, None] + ky * x[None, :]
    return mag, phase
