import pyglet
from pyglet.gl import Config as GLConfig


class GolfWindow(pyglet.window.Window):
    def __init__(self, cfg, loop, width=768, height=820):
        gl = GLConfig(
            major_version=3,
            minor_version=3,
            forward_compatible=True,
            double_buffer=True,
            depth_size=0,
        )
        super().__init__(
            width=width,
            height=height,
            config=gl,
            caption="Quantum Golf",
            resizable=False,
        )
        self.cfg, self.loop = cfg, loop
        self.renderer = None

    def on_draw(self):
        if self.renderer is None:
            import moderngl
            from render.field import FieldRenderer
            from render.overlay import Overlay
            from render.bloom import Bloom

            self.ctx = moderngl.create_context()
            self.renderer = FieldRenderer(self.ctx, self.cfg, self.size)
            self.overlay = Overlay(self.cfg, self.size)
            self.bloom = Bloom(
                self.ctx,
                self.size,
                threshold=self.cfg.bloom_threshold,
                intensity=self.cfg.bloom_intensity,
            )
        self.clear()
        self.loop.render(self.renderer, self.overlay, self, self.bloom)


    def on_mouse_motion(self, x, y, dx, dy):
        self.loop.on_aim(x, y)

    def on_mouse_scroll(self, x, y, sx, sy):
        self.loop.on_squash(sy)

    def on_mouse_press(self, x, y, b, m):
        self.loop.on_putt()

    def on_key_press(self, sym, mod):
        from pyglet.window import key

        if sym == key.R:
            self.loop.on_reset()
