import moderngl
import numpy as np

VERT_SRC = """
#version 330 core
in vec2 in_pos;          // corners in clip space [-1,1]
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;            // [-1,1] -> [0,1]
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

BRIGHT_SRC = """
#version 330 core
in vec2 uv; out vec4 frag;
uniform sampler2D src; uniform float threshold;   // ~0.6
void main() {
    vec3 c = texture(src, uv).rgb;
    float l = dot(c, vec3(0.299, 0.587, 0.114));   // luminance
    frag = vec4(c * smoothstep(threshold, 1.0, l), 1.0);
}
"""

BLUR_SRC = """
#version 330 core
in vec2 uv; out vec4 frag;
uniform sampler2D src; uniform vec2 dir;          // (1/W,0) then (0,1/H)
const float w[5] = float[](0.2270, 0.1946, 0.1216, 0.0540, 0.0162);
void main() {
    vec3 acc = texture(src, uv).rgb * w[0];
    for (int i = 1; i < 5; ++i) {
        acc += texture(src, uv + dir * float(i)).rgb * w[i];
        acc += texture(src, uv - dir * float(i)).rgb * w[i];
    }
    frag = vec4(acc, 1.0);
}
"""

COMP_SRC = """
#version 330 core
in vec2 uv; out vec4 frag;
uniform sampler2D sharp; uniform sampler2D glow; uniform float intensity;  // ~1.1
void main() {
    frag = vec4(texture(sharp, uv).rgb + intensity * texture(glow, uv).rgb, 1.0);
}
"""


def _fbo(ctx, size):
    tex = ctx.texture(size, 3, dtype="f1")
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = tex.repeat_y = False
    return ctx.framebuffer(color_attachments=[tex]), tex


class Bloom:
    def __init__(self, ctx, size, threshold=0.6, intensity=1.1):
        self.ctx, self.size = ctx, size
        self.sharp_fbo, self.sharp_tex = _fbo(ctx, size)
        self.ping, self.ping_tex = _fbo(ctx, size)
        self.pong, self.pong_tex = _fbo(ctx, size)
        self.bright = ctx.program(vertex_shader=VERT_SRC, fragment_shader=BRIGHT_SRC)
        self.blur = ctx.program(vertex_shader=VERT_SRC, fragment_shader=BLUR_SRC)
        self.comp = ctx.program(vertex_shader=VERT_SRC, fragment_shader=COMP_SRC)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vbo = ctx.buffer(quad.tobytes())
        self.bright["threshold"].value = threshold
        self.comp["intensity"].value = intensity


        def vao(prog):
            return ctx.vertex_array(prog, [(self.vbo, "2f", "in_pos")])

        self.bright_vao, self.blur_vao, self.comp_vao = (
            vao(self.bright),
            vao(self.blur),
            vao(self.comp),
        )

    def run(self, draw_field):
        W, H = self.size

        self.sharp_fbo.use()
        self.ctx.clear()
        draw_field()

        self.ping.use()
        self.sharp_tex.use(0)
        self.bright["src"].value = 0
        self.bright_vao.render(moderngl.TRIANGLE_STRIP)

        self.pong.use()
        self.ping_tex.use(0)
        self.blur["src"].value = 0
        self.blur["dir"].value = (1.0 / W, 0.0)
        self.blur_vao.render(moderngl.TRIANGLE_STRIP)
        self.ping.use()
        self.pong_tex.use(0)
        self.blur["dir"].value = (0.0, 1.0 / H)
        self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        self.ctx.screen.use()
        self.ctx.clear()
        self.sharp_tex.use(0)
        self.ping_tex.use(1)
        self.comp["sharp"].value = 0
        self.comp["glow"].value = 1
        self.comp_vao.render(moderngl.TRIANGLE_STRIP)
