from pathlib import Path
from typing import Any, Union
from enum import IntEnum
from PIL import Image
import numpy as np
from numpy.typing import ArrayLike
from pyglm import glm
from OpenGL.GL import *
import re

# Precision
VEC_DTYPE = np.float32

# World unit vectors
WORLD_RIGHT = WORLD_X = glm.vec3(1.0, 0.0, 0.0)
WORLD_UP = WORLD_Y = glm.vec3(0.0, 1.0, 0.0)
WORLD_FORWARD = WORLD_Z = glm.vec3(0.0, 0.0, -1.0)

WORLD_LEFT = -WORLD_RIGHT
WORLD_DOWN = -WORLD_UP
WORLD_BACKWARD = -WORLD_FORWARD


class ViewMode(IntEnum):
    compound_eye = 0
    panoramic = 1
    third_person = 2
    perspective = 3


class ProjectionMode(IntEnum):
    Physical = 0
    Acceptance = 1


class DeltaTimeTransformer:
    """
    A temporary proxy object for applying framerate-independent transforms
    It wraps a target object (like an Agent or Instance) and scales all
    subsequent chained transformation calls by a delta_time value
    """
    def __init__(self, target: Any, delta_time: float):
        self._target = target
        self._delta_time = delta_time

    def translate(self, translation: Union[glm.vec3, ArrayLike]):
        scaled_translation = glm.vec3(translation) * self._delta_time
        self._target.translate(scaled_translation)
        return self

    def rotate_axis(self, angle: float, axis: Union[str, glm.vec3, ArrayLike], degrees: bool = True):
        scaled_angle = angle * self._delta_time
        self._target.rotate_axis(scaled_angle, axis, degrees=degrees)
        return self

    def rotate(self, yaw_delta: float = 0.0, pitch_delta: float = 0.0, roll_delta: float = 0.0, degrees: bool = True):
        self._target.rotate(
            yaw_delta * self._delta_time,
            pitch_delta * self._delta_time,
            roll_delta * self._delta_time,
            degrees=degrees
        )
        return self

    def scale(self, scale_factors: Union[glm.vec3, ArrayLike]):
        """
        Applies scaling over time. A scale factor of 1.1 with dt will scale
        towards 10% larger, not instantly become 1.1x as large
        """
        scale_vec = glm.vec3(scale_factors)
        # Interpolate between no-scale (1, 1, 1) and target scale
        interpolated_scale = glm.mix(glm.vec3(1.0), scale_vec, self._delta_time)
        self._target.scale(interpolated_scale)
        return self



## Shader compiling functions

def _process_shader_includes_recursive(path: Path, include_stack: set):
    """
    Recursively processes #include directives in a shader file
    """
    if path in include_stack:
        raise RuntimeError(f"Circular #include detected: {path} is already in the include stack.")

    if not path.exists():
        raise FileNotFoundError(f"Cannot find include file: {path}")

    # Add the current file to the stack for the duration of this call
    include_stack.add(path)

    code = ""
    include_pattern = re.compile(r'#include\s+"(.*?)"')

    with open(path, 'r') as f:
        for line in f:
            match = include_pattern.match(line.strip())
            if match:
                include_path_str = match.group(1)
                # Included paths are relative to the file they are in
                include_path = path.parent / include_path_str
                # Recursively process the included file
                code += _process_shader_includes_recursive(include_path, include_stack) + "\n"
            else:
                code += line

    # Remove the file from the stack after processing is complete
    include_stack.remove(path)
    return code


def compile_single_shader(path, shader_type):
    """ Compiles a single shader from a file path, processing all nested includes """

    shader_path = Path(path)

    # Recursively process all includes to get a single string of code
    combined_code = _process_shader_includes_recursive(shader_path, set())

    # Find and manage all #version directives in the combined code
    version_pattern = re.compile(r'^\s*#version\s+.*$', re.MULTILINE)
    matches = version_pattern.findall(combined_code)

    if not matches:
        raise RuntimeError(f"Shader '{shader_path}' and its includes contain no #version directive.")

    if len(matches) > 1:
        # Create a formatted error message showing the conflicting directives
        conflicts = "\n".join(f"  - {match.strip()}" for match in matches)
        raise RuntimeError(f"Shader '{shader_path}' contains multiple conflicting #version directives:\n{conflicts}")

    # Remove the #version directive from its original position
    version_directive = matches[0]
    code_without_version = version_pattern.sub('', combined_code)

    # SPrepend the single, valid #version directive to the top of the final code
    final_code = version_directive + '\n' + code_without_version

    # Compile the final, correctly formatted shader code
    shader = glCreateShader(shader_type)
    glShaderSource(shader, final_code)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        glDeleteShader(shader)
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
                 GL_SRGB_ALPHA,
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

        glTexImage2D(faces_gl[i], 0, GL_SRGB8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, im_data)

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


class ShaderProgram:
    """ A wrapper for a GLSL shader program and its uniform locations """

    def __init__(self, vert_path=None, frag_path=None, comp_path=None):
        if comp_path:
            self.program_id = load_compute_shader(comp_path)
        else:
            self.program_id = load_shaders(vert_path, frag_path)
        self.locations = {}
        self.use()
        self._cache_all_uniforms()
        self.stop()

    def _cache_all_uniforms(self):
        """ Automatically queries and caches all active uniform locations """

        num_uniforms = glGetProgramiv(self.program_id, GL_ACTIVE_UNIFORMS)

        for i in range(num_uniforms):
            name, size, type = glGetActiveUniform(self.program_id, i)

            # Handle the case where drivers return the name as a numpy array
            if isinstance(name, np.ndarray):
                # Convert numpy array to bytes, then decode to a string, stripping any null terminators
                name = name.tobytes().decode('utf-8').rstrip('\x00')
            else:
                # decode a bytes object to a string
                name = name.decode('utf-8')

            # Handle arrays by removing the '[0]' suffix if present
            if name.endswith('[0]'):
                name = name[:-3]

            self.locations[name] = glGetUniformLocation(self.program_id, name)

    def use(self):
        """ Binds the shader program """
        glUseProgram(self.program_id)

    def stop(self):
        """ Unbinds the shader program """
        glUseProgram(0)

    def get_loc(self, name):
        """ Gets a cached uniform location """
        return self.locations.get(name)

    def free(self):
        """ Deletes the shader program """
        glDeleteProgram(self.program_id)


def write_pytinybvh_preamble(preamble: str):
    """ Writes PyTinyBVH #defines to a shader include that GLSL can #include """

    try:
        out_dir = Path('shaders')
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'pytinybvh_preamble.glsl').write_text(
                '// Auto-generated by pytinybvh.get_SSBO_bundle()\n' + preamble)
    except Exception as e:
        print('[Warn] Could not write pytinybvh_preamble.glsl:', e)


##

def generate_and_save_atlas(font_name=None, font_size=22, output_dir='interactive/fonts', color=(255, 255, 255, 255)):
    """
    Generates a fonts atlas texture and its corresponding metadata file
    """
    from os import environ
    environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    import pygame
    import string
    import json

    pygame.init()

    if font_name is None:
        font_name = pygame.font.get_default_font()

    print(f"Generating fonts atlas for '{font_name}' (size {font_size})...")

    font = pygame.font.SysFont(font_name, font_size)

    # All printable ASCII characters
    chars_to_render = string.printable
    atlas_cols = 16
    atlas_rows = (len(chars_to_render) + atlas_cols - 1) // atlas_cols
    char_data = {}

    # Determine atlas dimensions
    max_w, max_h = 0, 0
    for char in chars_to_render:
        w, h = font.size(char)
        if w > max_w: max_w = w
        if h > max_h: max_h = h

    cell_w, cell_h = max_w, max_h
    atlas_width = atlas_cols * cell_w
    atlas_height = atlas_rows * cell_h

    # Render characters to a Pygame surface
    atlas_surface = pygame.Surface((atlas_width, atlas_height), pygame.SRCALPHA)
    atlas_surface.fill((0, 0, 0, 0))  # Transparent background

    for i, char in enumerate(chars_to_render):
        char_surface = font.render(char, True, color)  # White text
        metrics = font.metrics(char)[0]
        advance = metrics[4]

        col = i % atlas_cols
        row = i // atlas_cols
        x, y = col * cell_w, row * cell_h

        atlas_surface.blit(char_surface, (x, y))

        # Store character metadata
        uv_x0 = x / atlas_width
        uv_y0 = y / atlas_height
        uv_x1 = (x + char_surface.get_width()) / atlas_width
        uv_y1 = (y + char_surface.get_height()) / atlas_height

        char_data[char] = {
            'w': char_surface.get_width(),
            'h': char_surface.get_height(),
            'uv_rect': (uv_x0, uv_y0, uv_x1, uv_y1),
            'advance': advance
        }

    # Save the files
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / f'{font_name}.png'
    json_path = output_dir / f'{font_name}.json'

    pygame.image.save(atlas_surface, image_path)

    with open(json_path, 'w') as f:
        json.dump({
            'font_name': font_name,
            'font_size': font_size,
            'atlas_image': f'{font_name}.png',
            'char_data': char_data
        }, f, indent=4)

    print(f"Saved fonts atlas for '{font_name}' (size {font_size}).")

    pygame.quit()


##

import open3d as o3d

def estimate_radii(pointcloud, k_neighbors=2):
    """
    Estimates the radius for each point in a point cloud such that
    spheres centered at these points would "touch" their nearest neighbors.

    Args:
        pointcloud (np.ndarray or open3d.geometry.PointCloud): The input point cloud
        k_neighbors (int): The number of nearest neighbors to consider.
                           If k_neighbors=2, it finds the single closest neighbor
                           (excluding the point itself).

    Returns:
        numpy.ndarray: An array of estimated radii for each point.
    """

    if isinstance(pointcloud, o3d.geometry.PointCloud):
        pcd_o3d = pointcloud
        points = np.asarray(pointcloud.points)

    elif isinstance(pointcloud, np.ndarray):
        points = pointcloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pointcloud)

    num_points = points.shape[0]
    radii = np.zeros(num_points, dtype=np.float32)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd_o3d)

    for i in range(num_points):
        # Find k_neighbors for the current point
        # k_neighbors + 1 because the point itself will be included in the search result
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd_o3d.points[i], k_neighbors + 1)

        if k > 1:
            # The first index (idx[0]) will be the point itself, so we take the second one
            # If k_neighbors = 1, we need to take idx[1]
            # If k_neighbors > 1, we might average or take the min of multiple distances
            # For "spheres touching" the distance to the single closest neighbor is most direct

            closest_neighbor_idx = idx[1]
            distance = np.linalg.norm(points[i] - points[closest_neighbor_idx])
            radii[i] = distance / 2.0
        else:
            # Handle isolated points or point clouds with less than k_neighbors + 1 points
            radii[i] = 0.1

    return radii