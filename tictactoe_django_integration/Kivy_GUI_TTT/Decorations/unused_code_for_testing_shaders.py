import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QOpenGLWidget
from PyQt5.QtCore import QTimer
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

VERTEX_SHADER = """
attribute vec2 position;
varying vec2 vUv;
void main() {
    vUv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
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
"""


class SilkWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.time = 0.0
        self.initialized = False  # <-- Guard flag
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(16)


    def initializeGL(self):
        try:
            glClearColor(0.1, 0.1, 0.1, 1.0)

            self.shader = compileProgram(
                compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            )

            self.vertices = np.array([
                -1.0, -1.0,
                 1.0, -1.0,
                -1.0,  1.0,
                 1.0,  1.0,
            ], dtype=np.float32)

            self.vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

            self.pos = glGetAttribLocation(self.shader, "position")
            glEnableVertexAttribArray(self.pos)
            glVertexAttribPointer(self.pos, 2, GL_FLOAT, GL_FALSE, 0, None)

            self.initialized = True  # <-- Safe to draw now

        except Exception as e:
            print("initializeGL failed:", e)

    def update_time(self):
        self.time += 0.016
        self.update()

    def paintGL(self):
        if not self.initialized:
            return  # ⛔ Don't draw before ready

        try:
            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(self.shader)

            glUniform1f(glGetUniformLocation(self.shader, "uTime"), self.time)
            glUniform3f(glGetUniformLocation(self.shader, "uColor"), 0.2, 0.0, 0.2)
            glUniform1f(glGetUniformLocation(self.shader, "uSpeed"), 5.0)
            glUniform1f(glGetUniformLocation(self.shader, "uScale"), 1.0)
            glUniform1f(glGetUniformLocation(self.shader, "uRotation"), 0.3)
            glUniform1f(glGetUniformLocation(self.shader, "uNoiseIntensity"), 1.5)

            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glEnableVertexAttribArray(self.pos)
            glVertexAttribPointer(self.pos, 2, GL_FLOAT, GL_FALSE, 0, None)

            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)

        except Exception as e:
            print("paintGL error:", e)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SilkWidget()
    win.setWindowTitle("Silk Shader (Safe + Guarded)")
    win.resize(800, 600)
    win.show()
    sys.exit(app.exec())
