from pathlib import Path
from typing import Any, Union, Sequence, Dict, Optional, Set
from enum import IntEnum
from PIL import Image
import numpy as np
from numpy.typing import ArrayLike
from pyglm import glm
from OpenGL.GL import *
import re

from trimesh import Trimesh, PointCloud
from trimesh.visual import TextureVisuals
from trimesh.visual.material import SimpleMaterial

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
    A proxy object for applying framerate-independent transforms.

    It wraps a target object (like an Agent or Instance) and scales all
    subsequent chained transformation calls by a delta_time value.
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
        Applies scaling over time (i.e. a factor of 1.1 will scale *towards* 10% larger, not instantly become 1.1x as large).
        """
        scale_vec = glm.vec3(scale_factors)

        # interpolate between no-scale (1, 1, 1) and target scale
        interpolated_scale = glm.mix(glm.vec3(1.0), scale_vec, self._delta_time)
        self._target.scale(interpolated_scale)
        return self

    def follow(self, trajectory, align_orientation: bool = True):
        """
        Updates the target's position based on the Trajectory state.

        Args:
            trajectory: An instance of geometry.paths.Trajectory
            align_orientation: If True, calls lookat() (or equivalent) to face movement direction.
        """
        new_pos, tangent = trajectory.advance(self._delta_time)

        self._target.position = new_pos

        if align_orientation:
            # look at a point slightly ahead in tangent direction
            target_look = new_pos - tangent

            if hasattr(self._target, 'lookat'):
                self._target.lookat(target_look)

            elif hasattr(self._target, 'direction'):
                self._target.direction = tangent

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

    # Add the current file to the stack (for the duration of this call)
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

    # Remove file from the stack after processing is complete
    include_stack.remove(path)
    return code


def compile_shader(path, shader_type, defines: Optional[Set[str]] = None):
    """Compiles a single shader from a file path with support for #include and #define directives."""

    shader_path = Path(path)

    combined_code = _process_shader_includes_recursive(shader_path, set())

    version_pattern = re.compile(r'^\s*#version\s+.*$', re.MULTILINE)
    matches = version_pattern.findall(combined_code)

    if not matches:
        raise RuntimeError(f"Shader '{shader_path}' and its includes contain no #version directive.")

    if len(matches) > 1:
        conflicts = "\n".join(f"  - {match.strip()}" for match in matches)
        raise RuntimeError(f"Shader '{shader_path}' contains multiple conflicting #version directives:\n{conflicts}")

    # remove the #version directive from its original position
    version_directive = matches[0]
    code_without_version = version_pattern.sub('', combined_code)

    define_block = ''
    if defines:
        define_block = '\n'.join(f'#define {d}' for d in sorted(defines)) + '\n'

    final_code = version_directive + '\n' + define_block + code_without_version

    shader = glCreateShader(shader_type)
    glShaderSource(shader, final_code)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        glDeleteShader(shader)
        raise RuntimeError(f"Shader compilation error in {path}:\n{error}")

    return shader


def load_shaders(path_vert, path_frag, path_geom=None, defines: Optional[Set[str]] = None):

    vertex_shader = compile_shader(path_vert, GL_VERTEX_SHADER, defines)
    fragment_shader = compile_shader(path_frag, GL_FRAGMENT_SHADER, defines)

    shaders_to_link = [vertex_shader, fragment_shader]
    if path_geom:
        geometry_shader = compile_shader(path_geom, GL_GEOMETRY_SHADER, defines)
        shaders_to_link.append(geometry_shader)

    program = glCreateProgram()
    for shader in shaders_to_link:
        glAttachShader(program, shader)

    glLinkProgram(program)

    # Check for linking errors
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        for shader in shaders_to_link:
            glDetachShader(program, shader)
            glDeleteShader(shader)
        glDeleteProgram(program)

        raise RuntimeError(f"Shader linking error:\n{error}")

    for shader in shaders_to_link:
        glDetachShader(program, shader)
        glDeleteShader(shader)

    return program


def load_compute_shader(path_comp, defines: Optional[Set[str]] = None):
    """Loads, compiles, and links a single compute shader into a program."""

    shader = compile_shader(path_comp, GL_COMPUTE_SHADER, defines)

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


class ShaderProgram:
    """A wrapper for a GLSL shader program and its uniform locations."""

    def __init__(self, vert_path=None, frag_path=None, comp_path=None, defines: Optional[Set[str]] = None):
        if comp_path:
            self.program_id = load_compute_shader(comp_path, defines=defines)
        else:
            self.program_id = load_shaders(vert_path, frag_path, defines=defines)
        self.locations = {}
        self.use()
        self._cache_all_uniforms()
        self.stop()

    def _cache_all_uniforms(self):
        """Automatically queries and caches all active uniform locations."""

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
        """Binds the shader program."""
        glUseProgram(self.program_id)

    def stop(self):
        """Unbinds the shader program."""
        glUseProgram(0)

    def get_loc(self, name):
        """Gets a cached uniform location."""
        return self.locations.get(name, -1)

    def free(self):
        """Deletes the shader program."""
        glDeleteProgram(self.program_id)


def write_pytinybvh_preamble(preamble: str):
    """Writes PyTinyBVH #defines to a shader include that GLSL can #include."""

    try:
        out_dir = Path('shaders')
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'pytinybvh_preamble.glsl').write_text(
                '// Auto-generated by pytinybvh.get_SSBO_bundle()\n' + preamble)
    except Exception as e:
        print('[Warn] Could not write pytinybvh_preamble.glsl:', e)


##

def generate_font_atlas(font_name=None, font_size=22, output_dir='interactive/fonts', color=(255, 255, 255, 255)):
    """
    Generates a fonts atlas texture and its corresponding metadata file.
    """
    from os import environ
    environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    try:
        import pygame
    except ImportError:
        raise ImportError("'pygame' package required for pygame texture generation")
    import string
    import json

    pygame.init()

    if font_name is None:
        font_name = pygame.font.get_default_font()

    print(f"Generating fonts atlas for '{font_name}' (size {font_size})...")

    font = pygame.font.SysFont(font_name, font_size)

    chars_to_render = string.printable
    atlas_cols = 16
    atlas_rows = (len(chars_to_render) + atlas_cols - 1) // atlas_cols
    char_data = {}

    max_w, max_h = 0, 0
    for char in chars_to_render:
        w, h = font.size(char)
        if w > max_w: max_w = w
        if h > max_h: max_h = h

    cell_w, cell_h = max_w, max_h
    atlas_width = atlas_cols * cell_w
    atlas_height = atlas_rows * cell_h

    atlas_surface = pygame.Surface((atlas_width, atlas_height), pygame.SRCALPHA)
    atlas_surface.fill((0, 0, 0, 0))

    for i, char in enumerate(chars_to_render):
        char_surface = font.render(char, True, color)
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

def estimate_radii(pointcloud, k=2):
    """
    Estimates the radius for each point in a point cloud such that
    spheres centered at these points would "touch" their nearest neighbours.

    Args:
        pointcloud (np.ndarray or trimesh.PointCloud): The input point cloud data.
        k (int): The number of nearest neighbours to consider (excluding the point itself).
                           Defaults to 2, which means it looks at the closest distinct point.

    Returns:
        numpy.ndarray: An array of estimated radii for each point.
    """

    import trimesh
    from scipy.spatial import cKDTree

    if isinstance(pointcloud, trimesh.PointCloud):
        points = pointcloud.vertices
    elif isinstance(pointcloud, np.ndarray):
        points = pointcloud
    else:
        raise TypeError("Input 'pointcloud' must be a numpy array or trimesh.PointCloud.")

    num_points = points.shape[0]
    radii = np.zeros(num_points, dtype=np.float32)

    if num_points == 0:
        return radii

    if k < 1:
        raise ValueError("k must be at least 1.")

    kdtree = cKDTree(points)

    distances, _ = kdtree.query(points, k=k + 1)

    for i in range(num_points):
        # check if at least 1 neighbour
        if len(distances[i]) > 1:
            closest_neighbour_dist = distances[i][1] # (index 0 is to the point itself, so 0)
            radii[i] = closest_neighbour_dist / 2.0
        else:
            # point has no neighbours (isolated point or very sparse cloud)
            # or fewer than k + 1 points in total
            radii[i] = 0.1  # default radius
    return radii


def trimesh_from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray = None,
    normals: np.ndarray = None,
    vertex_colors: np.ndarray = None,
    uv_coords: np.ndarray = None,
    texture_image: Image.Image = None
) -> Union[Trimesh, PointCloud, None]:
    """
    Creates a trimesh.Trimesh object from numpy arrays

    Args:
        vertices (np.ndarray): N x 3 array of vertex positions.
        faces (np.ndarray, optional): M x 3 (or M x N_verts_per_face) array of face indices.
                                      If None, a point cloud is created or faces are inferred by trimesh.
        normals (np.ndarray, optional): N x 3 array of vertex normals. Will be calculated by trimesh
                                        if not provided or invalid.
        vertex_colors (np.ndarray, optional): N x 3 (RGB) or N x 4 (RGBA) array of vertex colors.
                                              Values can be 0-255 (uint8) or 0.0-1.0 (float).
        uv_coords (np.ndarray, optional): N x 2 array of UV texture coordinates.
        texture_image (PIL.Image.Image, optional): A PIL Image object to be used as a texture.
                                                   Requires uv_coords to be applied.

    Returns:
        trimesh.Trimesh: A trimesh object created from the provided data.
                         (or None if an error occurs)
    """
    if vertices is None or vertices.ndim != 2 or vertices.shape[1] != 3:
        print("Error: 'vertices' must be an N x 3 numpy array.")
        return None

    try:
        is_mesh = faces is not None and faces.ndim == 2 and faces.shape[1] >= 3

        mesh_kwargs = {'vertices': vertices}
        if is_mesh:
            mesh_kwargs['faces'] = faces
        else:
            print("Info: No valid faces provided. Creating a trimesh.PointCloud object.")

        if normals is not None:
            if normals.shape != vertices.shape:
                print("Warning: 'normals' shape does not match 'vertices' shape. Ignoring provided normals.")
            else:
                mesh_kwargs['vertex_normals'] = normals

        # visual attributes (colors, UVs, texture)
        visual_args = {}

        if uv_coords is not None:
            if uv_coords.shape[0] == vertices.shape[0] and uv_coords.shape[1] == 2:
                visual_args['uv'] = uv_coords
                if texture_image is not None:
                    material = SimpleMaterial(image=texture_image)
                    visual_args['material'] = material
                    print("Info: Texture image and UV coordinates provided.")
                else:
                    print(
                        "Info: UV coordinates provided but no texture image. Model will have UVs but no visual texture.")
            else:
                print("Warning: 'uv_coords' count or shape does not match 'vertices'. Ignoring provided UVs.")
        elif texture_image is not None and uv_coords is None:
            print("Warning: Texture image provided but no UV coordinates. Texture will not be applied visually.")

        if vertex_colors is not None:
            if vertex_colors.shape[0] == vertices.shape[0] and vertex_colors.shape[1] in [3, 4]:
                visual_args['vertex_colors'] = vertex_colors
                print("Info: Vertex colors provided.")
            else:
                print("Warning: 'vertex_colors' count or shape does not match 'vertices'. Ignoring provided colors.")

        if 'uv' in visual_args or 'material' in visual_args:
            if is_mesh:
                mesh_kwargs['visual'] = TextureVisuals(**visual_args)
            else:
                print("Warning: Texture/UVs provided for a PointCloud. Trimesh.PointCloud does not directly "
                    "support TextureVisuals. Visual information will be limited to vertex colors if provided.")

        elif 'vertex_colors' in visual_args:
            mesh_kwargs['vertex_colors'] = visual_args['vertex_colors']

        if is_mesh:
            model = Trimesh(**mesh_kwargs)
        else:
            model = PointCloud(**mesh_kwargs)

        return model

    except Exception as e:
        print(f"Error creating model from arrays: {e}")
        return None


def extract_obj_curves(
        file_path,
        object_filter: Optional[Union[str, Sequence[str]]] = None,
        resample: int = None
    ) -> Dict[str, np.ndarray]:
    """
    Extracts curve coordinates from an .obj file.

    Args:
        file_path: Path to .obj file
        object_filter: Optional name(s) of objects to extract
        resample: Optionally resamples the curve to have this many evenly spaced points.
    """
    file_path = Path(file_path)
    vertices = []
    temp_indices = {}
    current_object = "Default"

    if object_filter is not None and isinstance(object_filter, str):
        target_objects = {object_filter}
    elif object_filter is not None:
        target_objects = set(object_filter)
    else:
        target_objects = object_filter

    with file_path.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            type_code = parts[0]

            if type_code == 'v':
                vertices.append([float(x) for x in parts[1:4]])

            elif type_code == 'o':
                current_object = parts[1]
                if (not target_objects or current_object in target_objects) and current_object not in temp_indices:
                    temp_indices[current_object] = []

            elif type_code == 'l':
                if not target_objects or current_object in target_objects:
                    indices = []
                    for idx in parts[1:]:
                        idx = int(idx)
                        real_idx = idx - 1 if idx > 0 else len(vertices) + idx
                        indices.append(real_idx)

                    current_list = temp_indices[current_object]

                    if not current_list:
                        current_list.extend(indices)
                    else:
                        start_idx = 1 if current_list[-1] == indices[0] else 0
                        current_list.extend(indices[start_idx:])
    final_curves = {}

    for obj_name, indices in temp_indices.items():
        if not indices:
            continue

        coords = np.array([vertices[i] for i in indices])

        if resample and resample > 1:
            coords = resample_path(coords, resample)

        final_curves[obj_name] = coords

    return final_curves


def resample_path(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Takes a path of points and returns a new path with 'num_samples' evenly spaced along the total arc length.
    """
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_length = cum_dist[-1]
    target_dists = np.linspace(0, total_length, num_samples)
    new_points = np.zeros((num_samples, 3))
    for i in range(3):
        new_points[:, i] = np.interp(target_dists, cum_dist, points[:, i])
    return new_points