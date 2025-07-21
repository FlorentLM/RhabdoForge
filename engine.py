import sys
import hashlib
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

#
# def load_shaders(path_vert, path_frag, path_geom=None):
#
#     def _compile_shader(path, shader_type):
#         """ Wrapper to compile a shader with clearer errors """
#         code = Path(path).read_text()
#         shader = compileShader(code, shader_type)
#
#         if not glGetShaderiv(shader, GL_COMPILE_STATUS):
#             error = glGetShaderInfoLog(shader).decode()
#             raise RuntimeError(f"Shader compilation error in {path}:\n{error}")
#         return shader
#
#     vertex_shader = _compile_shader(path_vert, GL_VERTEX_SHADER)
#     fragment_shader = _compile_shader(path_frag, GL_FRAGMENT_SHADER)
#
#     shaders = [vertex_shader, fragment_shader]
#     if path_geom:
#         shaders.append(_compile_shader(path_geom, GL_GEOMETRY_SHADER))
#
#     # Link program
#     program = compileProgram(*shaders)
#     if not glGetProgramiv(program, GL_LINK_STATUS):
#         error = glGetProgramInfoLog(program).decode()
#         raise RuntimeError(f"Shader linking error:\n{error}")
#
#     # # Validate program
#     # glValidateProgram(program)
#     # if not glGetProgramiv(program, GL_VALIDATE_STATUS):
#     #     error = glGetProgramInfoLog(program).decode()
#     #     # This might still be empty on some drivers tho
#     #     raise RuntimeError(f"Shader validation error:\n{error or 'No details provided by driver.'}")
#
#     return program

def load_shaders(path_vert, path_frag, path_geom=None):
    def _compile_single_shader(path, shader_type):
        """Compiles a single shader from a file path."""
        code = Path(path).read_text()
        shader = glCreateShader(shader_type)
        glShaderSource(shader, code)
        glCompileShader(shader)

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            error = glGetShaderInfoLog(shader).decode()
            glDeleteShader(shader)  # Don't leak the shader.
            raise RuntimeError(f"Shader compilation error in {path}:\n{error}")
        return shader

    # Compile all shaders
    vertex_shader = _compile_single_shader(path_vert, GL_VERTEX_SHADER)
    fragment_shader = _compile_single_shader(path_frag, GL_FRAGMENT_SHADER)

    shaders_to_link = [vertex_shader, fragment_shader]
    if path_geom:
        geometry_shader = _compile_single_shader(path_geom, GL_GEOMETRY_SHADER)
        shaders_to_link.append(geometry_shader)

    # Create and link the program
    program = glCreateProgram()
    for shader in shaders_to_link:
        glAttachShader(program, shader)

    glLinkProgram(program)

    # Check for linking errors (this is the crucial and correct check)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        # Clean up shaders and program if linking fails
        for shader in shaders_to_link:
            glDetachShader(program, shader)
            glDeleteShader(shader)
        glDeleteProgram(program)
        raise RuntimeError(f"Shader linking error:\n{error}")

    # Detach and delete shaders after a successful link, they are no longer needed.
    for shader in shaders_to_link:
        glDetachShader(program, shader)
        glDeleteShader(shader)

    return program


class Texture:

    __loaded_textures = {}

    @staticmethod
    def __make_hash(file, mode):
        if mode.upper() not in ('RGB', 'RGBA', 'L'):
            raise AttributeError('Unknown texture mode')

        file = Path(file)
        assert file.is_file() and file.suffix in ('.bmp', '.png', '.jpg', '.jpeg', '.tiff', '.tif')
        to_hash = (file.as_posix() + mode.upper()).encode("utf-8")
        hash = int(hashlib.sha1(to_hash).hexdigest(), 16) % (10 ** 8), file, mode

        return hash

    def __new__(cls, file, mode='RGBA'):
        hash, file, mode = cls.__make_hash(file, mode)
        if hash in Texture.__loaded_textures.keys():
            return Texture.__loaded_textures[hash]
        else:
            instance = super().__new__(cls)
            return instance

    def __init__(self, file, mode='RGBA'):

        hash, file, mode = self.__make_hash(file, mode)
        if hash in Texture.__loaded_textures.keys():
            self._hash = Texture.__loaded_textures[hash]._hash
            self._file_path = Texture.__loaded_textures[hash]._file_path
            self._mode = Texture.__loaded_textures[hash]._mode
            self._tex_unit = Texture.__loaded_textures[hash]._tex_unit
            self._GL_texID = Texture.__loaded_textures[hash]._GL_texID

        else:
            self._hash = hash
            self._file_path = file
            self._mode = mode
            self._tex_unit = len(Texture.__loaded_textures.keys())

            self._load_bitmap()
            Texture.__loaded_textures[self._hash] = self

    def _load_bitmap(self):

        with Image.open(self._file_path) as im:
            w, h = im.width, im.height
            im_data = im.transpose(Image.FLIP_TOP_BOTTOM).convert(self._mode).tobytes()

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        if self._mode == 'RGBA':
            gl_color_mode = GL_RGBA
        elif self._mode == 'RGB':
            gl_color_mode = GL_RGB
        else:
            gl_color_mode = GL_LUMINANCE       # TODO - why doesn't GL_LUMINANCE work??

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     gl_color_mode,
                     w,
                     h,
                     0,
                     gl_color_mode,
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

        self._GL_texID = texture_id

    @property
    def file(self):
        return self._file_path

    @property
    def mode(self):
        return self._mode

    @property
    def hash(self):
        return self._hash

    @property
    def unit(self):
        return self._tex_unit

    @property
    def idx(self):
        return self._GL_texID

# class Cubemap(Texture):
#
#     def __init__(self, file, mode):
#         super().__init__(file, mode)
#
#     def _load_bitmap(self):
#
#         faces = [GL_TEXTURE_CUBE_MAP_POSITIVE_X,
#                  GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
#                  GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
#                  GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
#                  GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
#                  GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]
#
#         files = list(Path(folder_path).glob('*'))
#         assert len(files) == 6
#
#         texture_id = glGenTextures(1)
#         glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)
#
#         for i in range(6):
#             with Image.open(files[i]) as im:
#                 w, h = im.width, im.height
#                 im_data = im.transpose(Image.FLIP_TOP_BOTTOM).convert(self._mode).tobytes()
#
#             glTexImage2D(faces[i],
#                          0,
#                          GL_RGBA8,
#                          w,
#                          h,
#                          0,
#                          GL_RGBA,
#                          GL_UNSIGNED_BYTE,
#                          im_data)
#
#         glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
#         glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
#         glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
#         glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
#         glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
#
#         # Unbind texture
#         glBindTexture(GL_TEXTURE_2D, 0)
#
#         self._GL_texID = texture_id



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