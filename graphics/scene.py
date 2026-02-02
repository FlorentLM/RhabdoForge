import OpenGL
import trimesh

OpenGL.ERROR_CHECKING = False

from typing import Dict, List, Optional, Union, Sequence
from numpy.typing import ArrayLike
from enum import Enum, auto

from pathlib import Path
import numpy as np
from PIL import Image

from trimesh import Trimesh, PointCloud, Scene as TrimeshScene

from OpenGL.GL import (ctypes,
                       glGenVertexArrays, glGenBuffers,
                       glEnableVertexAttribArray, glVertexAttribPointer, glGetUniformLocation,
                       glBindVertexArray,  glBindBuffer,  glBindTexture,
                       glDepthFunc, glDrawElements,
                       glUseProgram, glEnable, glDisable,
                       glUniformMatrix4fv, glUniform1i,
                       glActiveTexture,  glBufferData,
                       GL_TEXTURE0, GL_TEXTURE_CUBE_MAP, GL_ELEMENT_ARRAY_BUFFER, GL_UNSIGNED_INT, GL_CULL_FACE,
                       GL_TRIANGLES, GL_LESS, GL_LEQUAL, GL_FLOAT, GL_ARRAY_BUFFER, GL_STATIC_DRAW)

from pyglm import glm
from geometry.primitives import CUBE_VERTICES, CUBE_INDICES
from graphics.utils import trimesh_from_arrays, load_shaders, load_cubemap, WORLD_UP, WORLD_RIGHT, WORLD_FORWARD, DeltaTimeTransformer



class MaterialData:
    """Material properties for rendering (colors, specular, etc.)"""

    def __init__(self):
        self.base_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.specular = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w = shininess
        self.emission = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class AssetType(Enum):
    """ Distinguishes between different types of geometry assets """
    Mesh = auto()
    Points = auto()


class Asset:
    """
    A container for a renderable asset (mesh or point cloud).
    """

    def __init__(self, name: str):
        self.id = id(self)
        self.name = name
        self.asset_type: Optional[AssetType] = None

        # Mesh data
        self.vertices: Optional[np.ndarray] = None
        self.indices: Optional[np.ndarray] = None

        # Point cloud data
        self.points: Optional[np.ndarray] = None
        self.colors: Optional[np.ndarray] = None
        self.normals: Optional[np.ndarray] = None
        self.radii: Optional[np.ndarray] = None

        self.material = MaterialData()

        self._texture_path: Optional[Path] = None  # source path for lazy loading
        self._texture_image: Optional[Image.Image] = None  # cached image
        self.texture_id: Optional[int] = None  # OpenGL texture ID (set by renderer)

    @property
    def texture_path(self) -> Optional[Path]:
        return self._texture_path

    @property
    def texture_image(self) -> Optional[Image.Image]:
        """
        Returns the texture as a PIL Image lazily.
        Or None if no texture available.
        """
        if self._texture_image is not None:
            return self._texture_image

        if self._texture_path is not None:
            try:
                self._texture_image = Image.open(self._texture_path).convert("RGBA")
                return self._texture_image
            except Exception as e:
                print(f"Warning: Failed to load texture '{self._texture_path}' for asset '{self.name}': {e}")
                return None

        return None

    @property
    def has_texture(self) -> bool:
        """Returns True if this asset has a texture (path or image)."""
        return self._texture_image is not None or self._texture_path is not None

    def set_texture(self, source: Union[Path, str, Image.Image, np.ndarray, None]):
        """
        Sets the texture from various sources. None to clear.

        source:
            - Path or str: Path to texture file (will be lazy-loaded)
            - PIL Image: Used directly
            - numpy array: Converted to PIL Image
            - None: Clears any existing texture
        """

        # Clear existing
        self._texture_path = None
        self._texture_image = None

        if source is None:
            return

        if isinstance(source, (Path, str)):
            self._texture_path = Path(source)

        elif isinstance(source, Image.Image):
            self._texture_image = source.convert("RGBA")

        elif isinstance(source, np.ndarray):
            try:
                self._texture_image = Image.fromarray(source).convert("RGBA")
            except Exception as e:
                print(f"Warning: Failed to convert numpy array to image for asset '{self.name}': {e}")

        else:
            print(f"Warning: Unrecognized texture source type {type(source).__name__} for asset '{self.name}'")

    @classmethod
    def from_file(cls,
            name: str, file_path: Union[Path, str],
            texture: Optional[Union[Path, str, Image.Image, np.ndarray]] = None,
            radii: Optional[Union[float, ArrayLike]] = None
        ):
        """Creates an Asset by loading a 3D model from a file."""

        instance = cls(name)

        # texture override will be used instead of embedded texture if provided
        texture_override = texture is not None
        if texture_override:
            instance.set_texture(texture)

        trimesh_model = trimesh.load(file_path)

        if trimesh_model is None:
            raise ValueError(f"Failed to load 3D model from {file_path}")

        if isinstance(trimesh_model, TrimeshScene):
            print(f"Info: File '{file_path}' contains multiple meshes. Merging into single Asset '{name}'.")
            trimesh_model = trimesh_model.dump(concatenate=True)
            if not isinstance(trimesh_model, (Trimesh, PointCloud)):
                raise ValueError(f"Failed to extract geometry from scene {file_path}")

        instance._process_trimesh_object(trimesh_model, radii, extract_texture=not texture_override)

        print(f"Created Asset '{instance.name}' ({instance.asset_type.name}) from {file_path}")
        return instance

    @classmethod
    def from_arrays(cls, name: str,
                    vertices: np.ndarray,
                    faces: Optional[np.ndarray] = None,
                    normals: Optional[np.ndarray] = None,
                    vertex_colors: Optional[np.ndarray] = None,
                    uv_coords: Optional[np.ndarray] = None,
                    texture: Optional[Union[Path, str, Image.Image, np.ndarray]] = None,
                    radii: Optional[Union[float, ArrayLike]] = None):
        """Creates an Asset from numpy arrays."""

        instance = cls(name)

        if texture is not None:
            instance.set_texture(texture)

        # For trimesh texture mapping, we need the image if UVs are provided
        texture_for_trimesh = instance._texture_image if uv_coords is not None else None

        trimesh_model = trimesh_from_arrays(
            vertices=vertices, faces=faces, normals=normals,
            vertex_colors=vertex_colors, uv_coords=uv_coords,
            texture_image=texture_for_trimesh
        )

        if trimesh_model is None:
            raise ValueError("Failed to create geometry from arrays.")

        instance._process_trimesh_object(trimesh_model, radii, extract_texture=False)

        print(f"Created Asset '{instance.name}' ({instance.asset_type.name}) from arrays")
        return instance

    def _process_trimesh_object(self,
            trimesh_obj: Union[Trimesh, PointCloud],
            radii: Optional[Union[float, ArrayLike]],
            extract_texture: bool = True
        ):
        """Populates Asset data from a trimesh object."""

        if trimesh_obj.is_empty:
            raise ValueError(f"Geometry is empty for asset '{self.name}'.")

        if isinstance(trimesh_obj, Trimesh) and trimesh_obj.faces is not None and len(trimesh_obj.faces) > 0:
            self.asset_type = AssetType.Mesh
            self._setup_mesh_data(trimesh_obj, extract_texture)

        elif isinstance(trimesh_obj, PointCloud) and trimesh_obj.vertices is not None:
            self.asset_type = AssetType.Points
            self._setup_point_cloud_data(trimesh_obj, radii)

        else:
            raise ValueError(f"No valid geometry found for asset '{self.name}'.")

    def _setup_mesh_data(self, trimesh_obj: Trimesh, extract_texture: bool):
        """Populates mesh-specific data from a trimesh object."""

        vertices_3d = trimesh_obj.vertices.astype(np.float32)
        indices = trimesh_obj.faces.astype(np.uint32)

        # UVs
        uvs = np.zeros((len(vertices_3d), 2), dtype=np.float32)
        if hasattr(trimesh_obj.visual, 'uv') and trimesh_obj.visual.uv is not None:
            if trimesh_obj.visual.uv.shape[0] == vertices_3d.shape[0]:
                uvs = trimesh_obj.visual.uv.astype(np.float32)
            else:
                print(f"Warning: UV count mismatch in '{self.name}', zeroing UVs.")

        self.vertices = np.concatenate((vertices_3d, uvs), axis=1)
        self.indices = indices

        # Extract material properties from trimesh
        if hasattr(trimesh_obj.visual, 'material') and trimesh_obj.visual.material is not None:
            mat = trimesh_obj.visual.material

            # Base colour
            if hasattr(mat, 'main_color') and mat.main_color is not None:
                self.material.base_color = (mat.main_color / 255.0).astype(np.float32)

            # Specular
            if hasattr(mat, 'specular') and mat.specular is not None:
                spec = np.array(mat.specular, dtype=np.float32)
                if spec.max() > 1.0:
                    spec /= 255.0
                shininess = getattr(mat, 'shininess', 0.0)
                self.material.specular = np.array([spec[0], spec[1], spec[2], shininess], dtype=np.float32)

            # Embedded texture (only if not already set)
            if extract_texture and not self.has_texture:
                if hasattr(mat, 'image') and mat.image is not None:
                    self._texture_image = mat.image.convert("RGBA")

        if self.has_texture:
            source = f"path '{self._texture_path}'" if self._texture_path else "embedded/provided image"
            print(f"Info: Asset '{self.name}' has texture from {source}")
        else:
            print(f"Info: Asset '{self.name}' has no texture (will use base_color)")

    def _setup_point_cloud_data(self, trimesh_obj: PointCloud, radii: Optional[Union[float, ArrayLike]]):
        """Populates point cloud-specific data."""

        self.points = trimesh_obj.vertices.astype(np.float32)
        self._nb_points = len(self.points)

        # Colors
        if hasattr(trimesh_obj.visual, 'vertex_colors') and trimesh_obj.visual.vertex_colors is not None:
            vc = trimesh_obj.visual.vertex_colors
            if vc.shape[0] == self._nb_points:
                if vc.dtype == np.uint8:
                    self.colors = vc[:, :3].astype(np.float32) / 255.0
                else:
                    self.colors = vc[:, :3].astype(np.float32)
            else:
                self.colors = np.ones((self._nb_points, 3), dtype=np.float32)
        else:
            self.colors = np.ones((self._nb_points, 3), dtype=np.float32)

        # Normals
        if hasattr(trimesh_obj, 'vertex_normals') and trimesh_obj.vertex_normals is not None:
            if trimesh_obj.vertex_normals.shape[0] == self._nb_points:
                self.normals = trimesh_obj.vertex_normals.astype(np.float32)
            else:
                self.normals = np.zeros((self._nb_points, 3), dtype=np.float32)
        else:
            self.normals = np.zeros((self._nb_points, 3), dtype=np.float32)

        # Radii
        if isinstance(radii, (float, int)):
            self.radii = np.full(self._nb_points, radii, dtype=np.float32)
        elif radii is not None:
            self.radii = np.asarray(radii, dtype=np.float32)
            if len(self.radii) != self._nb_points:
                raise ValueError("Radii count must match point count.")
        else:
            self.radii = np.full(self._nb_points, 0.05, dtype=np.float32)

    @property
    def nb_triangles(self) -> int:
        if self.asset_type == AssetType.Mesh and self.indices is not None:
            return len(self.indices)
        return 0

    @property
    def nb_points(self) -> int:
        if self.asset_type == AssetType.Points and self.points is not None:
            return self._nb_points
        return 0


class Instance:
    """
    Logical instance of an Asset in the scene. Renderer-agnostic.
    This class is the single source of truth for an instance's transform.
    """
    def __init__(self,
                 asset: Asset,
                 transform: Optional[Union[glm.mat4, ArrayLike]] = None,
                 dynamic: bool = False,
                 **kwargs):

        self.id = id(self)
        self.asset = asset
        self.dynamic = dynamic
        self.properties = kwargs

        if transform is None:
            self.transform = glm.mat4(1.0)
        else:
            transform_np = np.asarray(transform, dtype=np.float32)

            if transform_np.shape == (4, 4):
                self.transform = glm.mat4(transform_np)

            elif transform_np.shape == (3,):
                self.transform = glm.translate(glm.mat4(1.0), glm.vec3(transform_np))

            else:
                raise ValueError(
                    f"Unsupported shape for transform: {transform_np.shape}. "
                    "Expected a (4, 4) matrix or a (3,) position vector."
                )

    def dt(self, delta_time: float) -> DeltaTimeTransformer:
        """
        Enables framerate-independent transformations for a chain of method calls

        Example:
            # Rotates at 90 degrees per second
            my_instance.dt(delta_time).rotate_axis(90, 'y')
        """
        return DeltaTimeTransformer(self, delta_time)

    @property
    def position(self):
        return glm.vec3(self.transform[3])

    @position.setter
    def position(self, value: Union[glm.vec3, ArrayLike]):
        self.transform[3] = glm.vec4(glm.vec3(value), 1.0)

    def translate(self, translation: Union[glm.vec3, ArrayLike]):
        self.transform = glm.translate(self.transform, glm.vec3(translation))
        return self

    def rotate_axis(self, angle: float, axis: Union[str, glm.vec3, ArrayLike], degrees: bool = True):
        """
        Rotates the instance around a given axis
        """

        if isinstance(axis, str):
            axis_str = axis.lower()
            axis_map = {
                'x': WORLD_RIGHT,
                'y': WORLD_UP,
                'z': WORLD_FORWARD,
                'right': WORLD_RIGHT,
                'left': -WORLD_RIGHT,
                'up': WORLD_UP,
                'down': -WORLD_UP,
                'forward': WORLD_FORWARD,
                'backward': -WORLD_FORWARD,
                'yaw': WORLD_UP,
                'pitch': WORLD_RIGHT,
                'roll': WORLD_FORWARD,
            }
            try:
                rotation_axis = axis_map[axis_str]
            except KeyError:
                raise ValueError(f"Unknown axis identifier: '{axis}'. Valid options are: {list(axis_map.keys())}")
        else:
            rotation_axis = glm.vec3(axis)

        angle_rad = glm.radians(angle) if degrees else angle
        self.transform = glm.rotate(self.transform, angle_rad, rotation_axis)
        return self

    def rotate(self, yaw_delta: float = 0.0, pitch_delta: float = 0.0, roll_delta: float = 0.0, degrees: bool = True):
        """ Rotates the instance by given Euler angle deltas """

        if yaw_delta != 0.0:
            self.transform = glm.rotate(self.transform, glm.radians(yaw_delta) if degrees else yaw_delta, WORLD_UP)

        if pitch_delta != 0.0:
            self.transform = glm.rotate(self.transform, glm.radians(pitch_delta) if degrees else pitch_delta, WORLD_RIGHT)

        if roll_delta != 0.0:
            self.transform = glm.rotate(self.transform, glm.radians(roll_delta) if degrees else roll_delta, WORLD_FORWARD)

        return self

    def scale(self, scale_factors: Union[glm.vec3, ArrayLike]):
        self.transform = glm.scale(self.transform, glm.vec3(scale_factors))
        return self


class Skybox:
    def __init__(self, texture_path: Path | str = 'textures/bright_day'):

        self.texture_id = load_cubemap(texture_path)

        self.program = load_shaders('shaders/skybox.vert', 'shaders/skybox.frag')

        skybox_vertices = CUBE_VERTICES.reshape(-1, 5)[:, :3]

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, skybox_vertices.nbytes, skybox_vertices, GL_STATIC_DRAW)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, CUBE_INDICES.nbytes, CUBE_INDICES, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

        glBindVertexArray(0)

    def draw(self, projection_matrix, view_matrix):

        glDepthFunc(GL_LEQUAL)  # changing depth function to LEQUAL is needed so the 1.0 depth passes
        glUseProgram(self.program)

        # We are inside the skybox so we need to see the back-faces
        glDisable(GL_CULL_FACE)

        glUniformMatrix4fv(glGetUniformLocation(self.program, "projection"), 1, False, glm.value_ptr(projection_matrix))
        glUniformMatrix4fv(glGetUniformLocation(self.program, "view"), 1, False, glm.value_ptr(view_matrix))

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture_id)
        glUniform1i(glGetUniformLocation(self.program, "skybox"), 0)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glEnable(GL_CULL_FACE) # Re-enable culling for the rest of the scene
        glDepthFunc(GL_LESS) # restore default depth function


class Scene:
    """
    The logical scene representation. A simple container for assets and instances.
    """
    def __init__(self, background_color: Sequence[float] = (0.0, 0.0, 0.0)):
        self.assets: Dict[str, Asset] = {}
        self.instances: List[Instance] = []
        self.skybox: Optional[Skybox] = None
        self.background_color = background_color

    def add_instance(self, asset: Union[Asset, str], transform: Optional[Union[glm.mat4, ArrayLike]] = None, **kwargs) -> Instance:

        if isinstance(asset, Asset):
            asset_obj = asset

            if asset_obj.name not in self.assets:
                print(f"New {asset_obj.asset_type.name} asset '{asset_obj.name}' registered with the scene.")
                self.assets[asset_obj.name] = asset_obj

            elif self.assets[asset_obj.name].id != asset_obj.id:
                raise ValueError(
                    f"An asset named '{asset_obj.name}' already exists but is a different object. "
                    "Asset names must be unique.")

        elif isinstance(asset, str):
            if asset not in self.assets:
                raise ValueError(
                    f"Asset with name '{asset}' not found. "
                    "You must add an instance of the asset object itself first to register it."
                )
            asset_obj = self.assets[asset]

        else:
            raise TypeError(
                f"Invalid type for asset_or_name. Expected Asset or str, but got {type(asset).__name__}.")

        instance = Instance(asset_obj, transform, **kwargs)
        self.instances.append(instance)
        return instance

    def add_skybox(self, texture_path: str):
        """ Creates and loads a skybox from a directory of textures """
        self.skybox = Skybox(texture_path)

    def load(self,
            file_path: Union[str, Path],
            transform: Optional[Union[glm.mat4, ArrayLike]] = None,
            **kwargs
        ) -> List[Instance]:
        """
        Loads a file (obj, gltf, etc.) and creates Assets and Instances.
        """

        file_path = Path(file_path)
        name_prefix = file_path.stem

        data = trimesh.load(file_path)

        if data is None:
            raise ValueError(f"Could not load model: {file_path}")

        new_instances = []

        user_transform = glm.mat4(1.0)
        if transform is not None:
            transform_np = np.asarray(transform, dtype=np.float32)

            if transform_np.shape == (4, 4):
                user_transform = glm.mat4(transform_np)
            elif transform_np.shape == (3,):
                user_transform = glm.translate(glm.mat4(1.0), glm.vec3(transform_np))

        if isinstance(data, trimesh.Scene):
            # Multi-geometry file
            print(f"Loading Scene '{file_path}' ({len(data.geometry)} geometries)...")

            for geom_name, geom_obj in data.geometry.items():

                transform_in_file, _ = data.graph.get(geom_name)
                node_transform = glm.mat4(transform_in_file)
                final_transform = user_transform * node_transform

                asset_name = f"{name_prefix}_{geom_name}"

                if asset_name not in self.assets:
                    asset = Asset(asset_name)
                    asset._process_trimesh_object(geom_obj, radii=kwargs.get('radii'), extract_texture=True)
                    self.assets[asset_name] = asset

                inst = self.add_instance(self.assets[asset_name], transform=final_transform,
                                         **{k: v for k, v in kwargs.items() if k != 'radii'})
                new_instances.append(inst)

        elif isinstance(data, (Trimesh, PointCloud)):
            # Single-geometry file
            print(f"Loading single geometry '{file_path}'...")

            asset_name = name_prefix

            if asset_name not in self.assets:
                asset = Asset(asset_name)
                asset._process_trimesh_object(data, radii=kwargs.get('radii'), extract_texture=True)
                self.assets[asset_name] = asset

            inst = self.add_instance(self.assets[asset_name], transform=user_transform,
                                     **{k: v for k, v in kwargs.items() if k != 'radii'})
            new_instances.append(inst)

        else:
            print(f"Warning: Unsupported data type from '{file_path}': {type(data)}")

        print(f"  Created {len(new_instances)} instance(s)")
        return new_instances

    @property
    def total_triangles(self) -> int:
        count = 0
        for instance in self.instances:
            if instance.asset.asset_type == AssetType.Mesh:
                count += instance.asset.nb_triangles
        return count

    @property
    def total_points(self) -> int:
        count = 0
        for instance in self.instances:
            if instance.asset.asset_type == AssetType.Points:
                count += instance.asset.nb_points
        return count

    def free(self):
        self.assets.clear()
        self.instances.clear()
        # Note: GPU resources tied to skybox/assets are freed by the bakers/renderers