import pyglet
from game.state import GameState, Phase
from game import course
import math
import concurrent.futures
import numpy as np
import pyglet
from render.coords import px_to_grid
from game.mechanics import (
    sample_landing,
    resolve,
    gate_count,
    tick_counter,
)
from quantum.preview import preview_field
from quantum.circuit import run_turn, TurnParams, walsh_term_counts, stream_turn
import threading

from game.stream import StreamingResult, consume


















def build_params(cfg, st):
    dx = cfg.L / cfg.grid_n
    return TurnParams(
        x0=st.lie[0] * dx,
        y0=st.lie[1] * dx,
        kx=st.kx,
        ky=st.ky,
        s=st.s,
        n_steps=st.n_steps,
        hole_idx=st.hole_idx,
        detector_cells=st.detector_cells,
    )


class Loop:
    def __init__(self, cfg, size=(768, 820)):
        self.cfg = cfg
        self.st = GameState(n_steps=cfg.n_steps_default)
        self._detector_rng = np.random.default_rng()
        self.result = None
        self._frame = 0
        self._size = size
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._abort = None
        self._rng = np.random.default_rng(1)
        self._walsh_counts = walsh_term_counts(cfg)
        self._gate_target = 0
        self._gate_shown = 0.0
        self._speed = 1.0
        self.st.detector_cells = course.pick_detector_cells(
            cfg, self._detector_rng, self.st.lie
        )


    def on_aim(self, px, py):
        if self.st.phase != Phase.AIMING:
            return
        gj, gk = px_to_grid(self.cfg, self._size, px, py)
        dx, dy = gj - self.st.lie[0], gk - self.st.lie[1]
        d = math.hypot(dx, dy) or 1.0
        mag = min(d / 6.0, 1.0) * self.cfg.k_max
        self.st.kx, self.st.ky = mag * dx / d, mag * dy / d

    def on_squash(self, sy):
        if self.st.phase != Phase.AIMING:
            return



        self.st.s *= self.cfg.s_step**sy
        self.st.s = min(max(self.st.s, self.cfg.s_min), self.cfg.s_max)

    def on_reset(self):
        if self._abort is not None:
            self._abort.set()
        pyglet.clock.unschedule(self._poll_compute)
        pyglet.clock.unschedule(self._tick)
        self.st = GameState(
            n_steps=self.cfg.n_steps_default
        )
        self.result = None
        self._future = None
        self._gate_target = 0
        self._gate_shown = 0.0
        self.st.detector_cells = course.pick_detector_cells(
            self.cfg, self._detector_rng, self.st.lie
        )

    def on_speed(self):
        speeds = (0.5, 1.0, 2.0)
        i = speeds.index(self._speed) if self._speed in speeds else 1
        self._speed = speeds[(i + 1) % len(speeds)]
        if self.st.phase == Phase.SIMULATING:
            pyglet.clock.unschedule(self._tick)
            pyglet.clock.schedule_interval(
                self._tick, self.cfg.frame_ms / 1000.0 / self._speed
            )

    def on_putt(self):
        if self.st.phase == Phase.AIMING:
            p = build_params(self.cfg, self.st)
            if self.cfg.streaming:
                self._abort = threading.Event()
                self.result = StreamingResult()
                self._future = self._pool.submit(
                    consume, stream_turn(self.cfg, p), self.result, self._abort
                )
            else:
                self._abort = None
                self._future = self._pool.submit(run_turn, self.cfg, p)
            self._gate_target = gate_count(
                self.cfg, self.st.n_steps, self._walsh_counts
            )
            self._gate_shown = 0.0
            self.st.phase = Phase.COMPUTING
            self.st.message = "charging..."
            pyglet.clock.schedule_interval(
                self._poll_compute, self.cfg.frame_ms / 1000.0
            )
        elif self.st.phase == Phase.SIMULATING:
            self._stop_at(self._frame)


    def _poll_compute(self, dt):
        if self._future.done() and self._future.exception() is not None:
            pyglet.clock.unschedule(self._poll_compute)
            self.st.phase = Phase.ERROR
            self.st.message = str(self._future.exception())
            return
        if self.cfg.streaming:
            ready = len(self.result.snapshots) > 0
        else:
            ready = self._future.done()
            if ready:
                self.result = self._future.result()
        if ready:
            pyglet.clock.unschedule(self._poll_compute)
            self._frame = 0
            self.st.phase = Phase.SIMULATING
            self.st.message = ""
            pyglet.clock.schedule_interval(
                self._tick, self.cfg.frame_ms / 1000.0 / self._speed
            )
        elif self._future.done():

            pyglet.clock.unschedule(self._poll_compute)
            self.st.phase = Phase.ERROR
            self.st.message = "no frames received"
        else:
            self._gate_shown = tick_counter(self._gate_shown, self._gate_target)




    def _tick(self, dt):
        if self._frame < len(self.result.snapshots) - 1:
            self._frame += 1
            self._gate_shown = (
                self._gate_target * len(self.result.snapshots) / max(self.st.n_steps, 1)
            )
        elif self._future.done():
            if self._future.exception() is not None:
                pyglet.clock.unschedule(self._tick)
                self.st.phase = Phase.ERROR
                self.st.message = str(self._future.exception())
            else:
                self._stop_at(self._frame)


    def _stop_at(self, frame):
        if self._abort is not None:
            self._abort.set()
        if self.st.phase != Phase.SIMULATING:
            return
        pyglet.clock.unschedule(self._tick)
        landing = sample_landing(self.result.snapshots[frame], self._rng)
        resolve(self.cfg, self.st, landing, self.result.hole_trace[frame])


    def render(self, field_renderer, overlay, window, bloom):
        self._size = window.size
        st = self.st
        if st.phase == Phase.SIMULATING:
            frame = min(self._frame, len(self.result.snapshots) - 1)
            field_renderer.upload(
                self.result.snapshots[frame], self.result.phase_snapshots[frame]
            )
            flash = frame in self.result.detect_at_step
            hole_idx = self.result.hole_trace[frame]
        elif st.phase in (Phase.WIN, Phase.LOSE) and self.result is not None:
            N = self.cfg.grid_n
            zeros = np.zeros((N, N))
            field_renderer.upload(
                zeros, zeros
            )
            flash = False
            hole_idx = st.hole_idx
        else:
            dx = self.cfg.L / self.cfg.grid_n
            mag, phase = preview_field(
                self.cfg, st.lie[0] * dx, st.lie[1] * dx, st.s, st.kx, st.ky
            )
            field_renderer.upload(mag, phase)
            flash = False
            hole_idx = st.hole_idx
        bloom.run(field_renderer.draw)
        overlay.draw(
            st,
            flash=flash,
            hole_idx=hole_idx,
            computing=(st.phase == Phase.COMPUTING),
            gate_shown=int(self._gate_shown),
            gate_target=self._gate_target,
            speed=self._speed,
        )
