from pathlib import Path
from PIL import Image
import numpy as np

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
