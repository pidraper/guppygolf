
def grid_to_px(cfg, size, j, k):
    W, H = size
    hud = 52
    side = min(W, H - hud)
    cell = side / cfg.grid_n
    px = (j + 0.5) * cell
    py = hud + (k + 0.5) * cell
    return px, py, cell


def px_to_grid(cfg, size, px, py):
    W, H = size
    hud = 52
    side = min(W, H - hud)
    cell = side / cfg.grid_n
    return (px / cell - 0.5), ((py - hud) / cell - 0.5)
