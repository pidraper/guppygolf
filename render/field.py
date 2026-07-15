



import numpy as np
import moderngl







VERT_SRC = """           // corners of triangle(s).
#version 330 core
in vec2 in_pos;          // corners in clip space [-1,1]
out vec2 uv;             // this will be the in for the fragment shader
void main() {
    uv = in_pos * 0.5 + 0.5;            // [-1,1] -> [0,1]
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""


FRAG_SRC = """               // domain coloring (HSV): hue = phase, brightness = |psi|^2
#version 330 core
in vec2 uv;
out vec4 frag;
uniform sampler2D field;     // RGB: r=|psi|^2 (peak-norm), g=cos(phase), b=sin(phase)
uniform float gain;          // brightness gain
uniform float gamma;         // brightness tone-curve exponent (>1 suppresses faint tails)
vec3 hsv2rgb(vec3 c) {       // c = (hue, sat, val), all in [0,1]
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
void main() {
    vec3 f = texture(field, uv).rgb;
    float v = clamp(pow(f.r, gamma) * gain, 0.0, 1.0);   // brightness = magnitude ONLY
    float h = atan(f.b, f.g) / 6.2831853 + 0.5;          // phase -> hue [0,1); interp'd via cos/sin -> no wrap seam
    frag = vec4(hsv2rgb(vec3(h, 1.0, v)), 1.0);          // low |psi|^2 -> black regardless of hue
}
"""


def field_bytes(mag, phase):
    m = mag / max(mag.max(), 1e-12)
    rgb = np.stack(
        [m.T, np.cos(phase).T, np.sin(phase).T], axis=-1
    )
    return np.ascontiguousarray(rgb, dtype="f4").tobytes()


class FieldRenderer:
    def __init__(self, ctx, cfg, size):
        self.ctx, self.cfg, self.size = ctx, cfg, size
        N = cfg.grid_n
        self.prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
        quad = np.array(
            [-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4"
        )
        self.vao = ctx.vertex_array(
            self.prog, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")]
        )

        self.field = ctx.texture(
            (N, N), 3, dtype="f4"
        )
        self.field.filter = (
            moderngl.LINEAR,
            moderngl.LINEAR,
        )
        self.field.repeat_x = self.field.repeat_y = False
        self.prog["gain"].value = cfg.field_gain
        self.prog["gamma"].value = cfg.field_gamma

    def upload(self, mag, phase):
        self.field.write(field_bytes(mag, phase))

    def draw(self):
        self.field.use(0)
        self.prog["field"].value = 0
        self.vao.render(moderngl.TRIANGLE_STRIP)
