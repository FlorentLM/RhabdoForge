from pathlib import Path
from PIL import Image
import numpy as np
from OpenGL.GL import *
import re


# Precision
VEC_DTYPE = np.float32

# World unit vectors
WORLD_RIGHT = WORLD_X = np.array([1.0, 0.0, 0.0], dtype=VEC_DTYPE)
WORLD_UP = WORLD_Y = np.array([0.0, 1.0, 0.0], dtype=VEC_DTYPE)
WORLD_FORWARD = WORLD_Z = np.array([0.0, 0.0, -1.0], dtype=VEC_DTYPE)

WORLD_LEFT = - WORLD_RIGHT
WORLD_DOWN = - WORLD_UP
WORLD_BACKWARD = - WORLD_FORWARD


# Loader functions

def compile_single_shader(path, shader_type):
    """ Compiles a single shader from a file path """

    path = Path(path)

    # Regex to find #include "some/path.glsl"
    include_pattern = re.compile(r'#include\s+"(.*?)"')

    # Start with the code from the main shader file
    code = path.read_text()

    # Find all '#include' directives
    for match in include_pattern.finditer(code):
        include_path_str = match.group(1)
        # The path in the '#include' is relative to the current shader file
        include_path = path.parent / include_path_str

        if include_path.exists():
            include_content = include_path.read_text()
            # replace the '#include' directive with the content of the included file
            code = code.replace(match.group(0), include_content)
        else:
            raise FileNotFoundError(f"Cannot find include file: {include_path}")

    shader = glCreateShader(shader_type)
    glShaderSource(shader, code)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        glDeleteShader(shader)  # Don't leak the shader
        raise RuntimeError(f"Shader compilation error in {path}:\n{error}")
    return shader


def load_shaders(path_vert, path_frag, path_geom=None):

    # Compile all shaders
    vertex_shader = compile_single_shader(path_vert, GL_VERTEX_SHADER)
    fragment_shader = compile_single_shader(path_frag, GL_FRAGMENT_SHADER)

    shaders_to_link = [vertex_shader, fragment_shader]
    if path_geom:
        geometry_shader = compile_single_shader(path_geom, GL_GEOMETRY_SHADER)
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


def load_compute_shader(path_comp):
    """ Loads, compiles, and links a single compute shader into a program """

    shader = compile_single_shader(path_comp, GL_COMPUTE_SHADER)

    program = glCreateProgram()
    glAttachShader(program, shader)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        glDetachShader(program, shader)
        glDeleteShader(shader)
        glDeleteProgram(program)
        raise RuntimeError(f"Shader linking error for {path_comp}:\n{error}")

    glDetachShader(program, shader)
    glDeleteShader(shader)

    return program


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

    # OpenGL cubemap face order
    faces_gl = [GL_TEXTURE_CUBE_MAP_POSITIVE_X,
                GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
                GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
                GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
                GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
                GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]

    face_files = ['right.jpg', 'left.jpg', 'top.jpg', 'bottom.jpg', 'front.jpg', 'back.jpg']

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)

    for i in range(6):
        filepath = Path(folder_path) / face_files[i]
        if not filepath.exists():
            # Try png
            filepath = filepath.with_suffix('.png')
            if not filepath.exists():
                raise FileNotFoundError(f"Could not find cubemap face: {filepath.with_suffix('.jpg')}")

        with Image.open(filepath) as im:
            w, h = im.width, im.height
            im_data = im.convert("RGBA").tobytes()

        glTexImage2D(faces_gl[i], 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, im_data)

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    return texture_id


def check_MetalGL_context():
    import ctypes

    # Useful to test if MGL is loaded on macOS

    # Load PyGame's SDL2 library
    sdl_path = Path().cwd() / ".venv/lib/python3.13/site-packages/pygame/.dylibs/libSDL2-2.0.0.dylib"
    sdl = ctypes.CDLL(sdl_path.as_posix())
    sdl.SDL_GL_GetCurrentContext.restype = ctypes.c_void_p

    gl_ctx_ptr = sdl.SDL_GL_GetCurrentContext()
    print("SDL_GL_GetCurrentContext returned:", gl_ctx_ptr)
    print("GL version:", glGetString(GL_VERSION).decode())
    print("Renderer:", glGetString(GL_RENDERER).decode())
    print("---------------------------------")
