from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import RenderContext, Rectangle
from kivy.clock import Clock
from kivy.core.window import Window

VERTEX_SHADER = '''
attribute vec2 vPosition;
varying vec2 vUv;

void main() {
    vUv = vPosition * 0.5 + 0.5;
    gl_Position = vec4(vPosition, 0.0, 1.0);
}
'''

FRAGMENT_SHADER = '''
#ifdef GL_ES
    precision mediump float;
#endif

varying vec2 vUv;
uniform float uTime;
uniform vec3 uColor;
uniform float uSpeed;
uniform float uScale;
uniform float uRotation;
uniform float uNoiseIntensity;

float noise(vec2 texCoord) {
    const float G = 2.71828182845904523536;
    vec2 r = (G * sin(G * texCoord));
    return fract(r.x * r.y * (1.0 + texCoord.x));
}

vec2 rotateUvs(vec2 uv, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    mat2 rot = mat2(c, -s, s, c);
    return rot * uv;
}

void main() {
    float rnd = noise(gl_FragCoord.xy);
    vec2 uv = rotateUvs(vUv * uScale, uRotation);
    vec2 tex = uv * uScale;
    float tOffset = uSpeed * uTime;

    tex.y += 0.03 * sin(8.0 * tex.x - tOffset);

    float pattern = 0.6 +
                    0.4 * sin(5.0 * (tex.x + tex.y +
                            cos(3.0 * tex.x + 5.0 * tex.y) +
                            0.02 * tOffset) +
                            sin(20.0 * (tex.x + tex.y - 0.1 * tOffset)));

    vec4 col = vec4(uColor, 1.0) * vec4(pattern) - rnd / 15.0 * uNoiseIntensity;
    col.a = 1.0;
    gl_FragColor = col;
}
'''

class SilkWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.canvas = RenderContext(compute_normal_mat=False)
        self.canvas.shader.source = None
        self.canvas.shader.vs = VERTEX_SHADER
        self.canvas.shader.fs = FRAGMENT_SHADER

        with self.canvas:
            self.rect = Rectangle(pos=self.pos, size=self.size)

        self.time = 0.0
        self.uColor = (0.4, 0.0, 0.4)  # Purple color
        self.uSpeed = 5.0
        self.uScale = 1.0
        self.uRotation = 0.3
        self.uNoiseIntensity = 1.5

        Clock.schedule_interval(self.update_glsl, 1/60)
        self.bind(pos=self._update_rect, size=self._update_rect)

    def _update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def update_glsl(self, dt):
        self.time += dt
        self.canvas['uTime'] = self.time
        self.canvas['uColor'] = self.uColor
        self.canvas['uSpeed'] = self.uSpeed
        self.canvas['uScale'] = self.uScale
        self.canvas['uRotation'] = self.uRotation
        self.canvas['uNoiseIntensity'] = self.uNoiseIntensity
        self.canvas.ask_update()

class SilkApp(App):
    def build(self):
        self.root = SilkWidget()
        self.root.pos = (-1, -1)
        self.root.size = Window.size
        Window.bind(on_resize=self._update_size)
        return self.root

        # noinspection PyUnusedLocal
    def _update_size(self, window, width, height):
        self.root.pos = (-1, -1)
        self.root.size = (width, height)

        # noinspection PyUnusedLocal
    def on_window_resize(self, window, width, height):
        # If you want to manually control size, update here
        # self.root.size = (width, height)
        pass

if __name__ == '__main__':
    SilkApp().run()
