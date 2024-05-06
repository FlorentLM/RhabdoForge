import sys
from pathlib import Path

from PIL import Image
import numpy as np

import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

##

# World unit vectors
WORLD_RIGHT = WORLD_X = np.array([1.0, 0.0, 0.0], dtype=np.float32)
WORLD_UP = WORLD_Y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
WORLD_FORWARD = WORLD_Z = np.array([0.0, 0.0, -1.0], dtype=np.float32)

WORLD_LEFT = - WORLD_RIGHT
WORLD_DOWN = - WORLD_UP
WORLD_BACKWARD = - WORLD_FORWARD


def load_shaders(path_vert, path_frag, path_geom=None):

    vertex_code = Path(path_vert).read_text()
    fragment_code = Path(path_frag).read_text()

    vertex_compiled = compileShader(vertex_code, GL_VERTEX_SHADER)
    frag_compiled = compileShader(fragment_code, GL_FRAGMENT_SHADER)

    comp = [vertex_compiled, frag_compiled]

    if path_geom is not None:
        geometry_code = Path(path_geom).read_text()
        geometry_compiled = compileShader(geometry_code, GL_GEOMETRY_SHADER)
        comp.append(geometry_compiled)

    return compileProgram(*comp)


def load_texture(file_path):
    bitmap_path = Path(file_path)
    if not bitmap_path.exists():
        raise IOError(f"Failed to open texture file {bitmap_path}")

    with Image.open(bitmap_path) as im:
        w, h = im.width, im.height
        im_data = im.transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA").tobytes()

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

    glTexImage2D(GL_TEXTURE_2D,
                 0,
                 GL_RGBA,
                 w,
                 h,
                 0,
                 GL_RGBA,
                 GL_UNSIGNED_BYTE,
                 im_data)

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0)

    # Unbind texture
    glBindTexture(GL_TEXTURE_2D, 0)

    return texture_id


def load_cubemap(folder_path):

    faces = [GL_TEXTURE_CUBE_MAP_POSITIVE_X,
             GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
             GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
             GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
             GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
             GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]

    files = list(Path(folder_path).glob('*'))
    assert len(files) == 6

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

    for i in range(6):
        with Image.open(files[i]) as im:
            w, h = im.width, im.height
            im_data = im.transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA").tobytes()

        glTexImage2D(faces[i],
                     0,
                     GL_RGBA8,
                     w,
                     h,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     im_data)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    # Unbind texture
    glBindTexture(GL_TEXTURE_2D, 0)

    return texture_id


def init_glfw(width=800, height=600, name='Antworlds'):

    if not glfw.init():
        print("Could not initialize OpenGL context.")
        sys.exit(1)

    # macOS supports only forward-compatible core profiles from 3.2+
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), name, None, None)
    glfw.make_context_current(window)

    # Set the wait time for glfwSwapBuffers to 0 (this unlocks FPS)
    glfw.swap_interval(0)
    # The above may not work on all platforms. Another solution is to use single buffer instead of double
    # (add the hint ```glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)``` before creating the window)

    if not window:
        glfw.terminate()
        print("Could not initialize window...")
        sys.exit(1)

    return window


class Input:
    forward: bool = False
    backward: bool = False
    left: bool = False
    right: bool = False
    up: bool = False
    down: bool = False

    mouse_x: float = 0.0
    mouse_y: float = 0.0
    mouse_wh: float = 0.0

    mouse_lb: bool = False
    mouse_rb: bool = False

    _last_mouse_x: float = 0.0
    _last_mouse_y: float = 0.0

    quit: bool = False

    @staticmethod
    def get_keys(window: glfw._GLFWwindow, key: int, scancode: int, action: int, mods: int) -> None:

        if action in (glfw.PRESS, glfw.RELEASE):
            if key == glfw.KEY_ESCAPE:
                Input.quit = True

            if key == glfw.KEY_W:
                Input.forward = not Input.forward

            if key == glfw.KEY_S:
                Input.backward = not Input.backward

            if key == glfw.KEY_A:
                Input.left = not Input.left

            if key == glfw.KEY_D:
                Input.right = not Input.right

            if key == glfw.KEY_Z:
                Input.up = not Input.up

            if key == glfw.KEY_X:
                Input.down = not Input.down

    @staticmethod
    def get_mouse(window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:

        xoffset = xpos - Input._last_mouse_x
        yoffset = ypos - Input._last_mouse_y

        if abs(xoffset) > 1:
            Input.mouse_x = xoffset
        else:
            Input.mouse_x = 0.0
        if abs(yoffset) > 1:
            Input.mouse_y = yoffset
        else:
            Input.mouse_y = 0.0

        Input._last_mouse_x = xpos
        Input._last_mouse_y = ypos

    @staticmethod
    def get_mousebuttons(window: glfw._GLFWwindow, button: int, action: int, mods: int) -> None:
        if action in (glfw.PRESS, glfw.RELEASE):
            if button == glfw.MOUSE_BUTTON_LEFT:
                Input.mouse_lb = not Input.mouse_lb
            if button == glfw.MOUSE_BUTTON_RIGHT:
                Input.mouse_rb = not Input.mouse_rb

    @staticmethod
    def get_scroll(window: glfw._GLFWwindow, xoffset: float, yoffset: float) -> None:
        if yoffset > 0.1:
            Input.mouse_wh = 1
        elif yoffset < -0.1:
            Input.mouse_wh = -1
        else:
            Input.mouse_wh = 0