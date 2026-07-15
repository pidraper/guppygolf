import math

import numpy as np
import pyglet
from contourpy import contour_generator
from pyglet import shapes
from game import course
from render.coords import grid_to_px


def hud_text(cfg, st, gate_shown, gate_target, speed, computing):
    if computing:
        gates_line = f"gates {int(gate_shown):,}"
    elif gate_target > 0:
        gates_line = f"circuit {gate_target:,} gates"
    else:
        gates_line = "circuit  --"
    speed_str = f"   speed {speed:g}x" if speed != 1.0 else ""
    return (
        f"stroke {st.strokes}/{cfg.max_strokes}\n"
        f"steps {st.n_steps}   squash {st.s:.2f}{speed_str}\n"
        f"{gates_line}\n"
        f"{st.message}"
    )


class Overlay:
    def __init__(self, cfg, size):
        self.cfg, self.size = cfg, size
        self.batch = pyglet.graphics.Batch()
        self.panel = shapes.Rectangle(6, 6, 360, 104, color=(0, 0, 0))
        self.panel.opacity = 140
        self.hud = pyglet.text.Label(
            "",
            x=14,
            y=100,
            width=340,
            multiline=True,
            font_name="Menlo",
            font_size=16,
            color=(230, 230, 235, 255),
            anchor_x="left",
            anchor_y="top",
            batch=self.batch,
        )
        self._pulse = 0
        self._contour_batch = self._build_contours(cfg, size)

    def _build_contours(self, cfg, size):
        self._contour_lines = {}
        batches = {}
        for h in range(len(cfg.burrows)):
            V = np.asarray(course.build_potential(cfg, h))
            gen = contour_generator(z=V.T)
            batch = pyglet.graphics.Batch()
            refs = []
            for lvl in np.percentile(V, np.linspace(12, 88, 7)):
                for line in gen.lines(float(lvl)):
                    px = [grid_to_px(cfg, size, v[0], v[1])[:2] for v in line]
                    for (ax, ay), (bx2, by2) in zip(px[:-1], px[1:]):
                        ln = shapes.Line(
                            ax, ay, bx2, by2, thickness=1, color=(150, 160, 190), batch=batch
                        )
                        ln.opacity = 90
                        refs.append(ln)
            batches[h] = batch
            self._contour_lines[h] = refs
        return batches

    def draw(
        self,
        st,
        flash=False,
        hole_idx=None,
        computing=False,
        gate_shown=0,
        gate_target=0,
        speed=1.0,
    ):
        cfg, size = self.cfg, self.size
        hole_idx = st.hole_idx if hole_idx is None else hole_idx
        cb = self._contour_batch.get(hole_idx)
        if cb is not None:
            cb.draw()
        prims = []

        col = (255, 60, 60) if flash else (40, 200, 220)
        for (dj, dk) in getattr(st, "detector_cells", ()):
            dx0, dy0, _cell = grid_to_px(cfg, size, dj - 0.5, dk - 0.5)
            dx1, dy1, _ = grid_to_px(cfg, size, dj + 0.5, dk + 0.5)
            prims.append(shapes.Box(dx0, dy0, dx1 - dx0, dy1 - dy0, thickness=2, color=col))

        cj, ck, r = course.hole_region(cfg, hole_idx)
        hx, hy, cell = grid_to_px(cfg, size, cj, ck)
        prims.append(shapes.Arc(hx, hy, r * cell, thickness=3, color=(80, 255, 120)))

        bx, by, _ = grid_to_px(cfg, size, *st.lie)
        prims.append(shapes.Circle(bx, by, 4, color=(255, 255, 255)))

        if st.phase.name in ("WIN", "LOSE"):
            burst = (120, 255, 150) if st.phase.name == "WIN" else (255, 90, 90)
            prims.append(shapes.Circle(bx, by, 9, color=burst))
            prims.append(shapes.Arc(bx, by, 18, thickness=3, color=burst))

        if computing:
            self._pulse += 1
            r = 12 + 7 * (0.5 + 0.5 * math.sin(self._pulse * 0.3))
            prims.append(shapes.Arc(bx, by, r, thickness=2, color=(255, 220, 120)))

        if st.phase.name == "AIMING" and (st.kx or st.ky):
            ax, ay = bx + st.kx * 4, by + st.ky * 4
            prims.append(
                shapes.Line(bx, by, ax, ay, thickness=2, color=(255, 230, 120))
            )
        for p in prims:
            p.draw()
        self.hud.text = hud_text(
            self.cfg, st, gate_shown, gate_target, speed, computing
        )
        self.panel.draw()
        self.batch.draw()
