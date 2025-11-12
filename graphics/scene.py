import OpenGL

OpenGL.ERROR_CHECKING = False

from typing import Dict, List, Optional, Union, Sequence
from numpy.typing import ArrayLike
from enum import Enum, auto

from pathlib import Path
import numpy as np
from PIL import Image

from graphics.models_loaders import trimesh_from_file, trimesh_from_arrays
from trimesh import Trimesh, PointCloud, Scene as TrimeshScene

from OpenGL.GL import *
from pyglm import glm
from geometry.primitives import CUBE_VERTICES, CUBE_INDICES
from graphics.utils import load_shaders, load_cubemap, WORLD_UP, WORLD_RIGHT, WORLD_FORWARD, DeltaTimeTransformer


class AssetType(Enum):
    """ Distinguishes between different types of geometry assets """
    Mesh = auto()
    Points = auto()


class Asset:
    """
    A container for a renderable asset which can be either a mesh or a point cloud.

    To create a Mesh:
    - From file: Asset(name="my_mesh", file_path="path/to/mesh.obj", texture_path="path/to/tex.jpg")
    - From data: Asset(name="my_mesh", vertices=verts_array, indices=indices_array)

    To create a Point Cloud:
    - From file: Asset(name="my_pcd", file_path="path/to/points.ply", radii=0.01)
    - From data: Asset(name="my_pcd", points=points_array, colors=colors_array, radii=0.01)
    """

    def __init__(self, name: str):
        self.id = id(self)
        self.name = name
        self.asset_type: Optional[AssetType] = None

        # Mesh data
        self.vertices: Optional[np.ndarray] = None  # (N, 5) array for meshes
        self.indices: Optional[np.ndarray] = None  # (M, 3) array for meshes

        # Point cloud data
        self.points: Optional[np.ndarray] = None  # (N, 3) for point clouds
        self.colors: Optional[np.ndarray] = None  # (N, 3) or (N, 4) for point clouds
        self.normals: Optional[np.ndarray] = None  # (N, 3) for point clouds
        self.radii: Optional[np.ndarray] = None  # (N,) for point clouds

        self.texture_id: Optional[int] = None  # OpenGL texture ID
        self.texture_path: Optional[Path] = None
        self._texture_image_data: Optional[Image.Image] = None

    @classmethod
    def from_file(cls, name: str, file_path: Union[Path, str],
                  texture: Optional[Union[Path, str, Image.Image, np.ndarray]] = None,
                  radii: Optional[Union[float, ArrayLike]] = None):
        """
        Creates an Asset by loading a 3D model from a file using trimesh
        """

        instance = cls(name)
        trimesh_model = trimesh_from_file(file_path)

        if trimesh_model is None:
            raise ValueError(f"Failed to load 3D model from {file_path}")

        if isinstance(trimesh_model, TrimeshScene):
            print(
                f"Info: File '{file_path}' contains multiple meshes (a TrimeshScene). Merging all geometries into a single Asset '{name}'.")

            # TODO: Instead of merging everything into a single asset, this should create multiple Asset objects

            trimesh_model = trimesh_model.dump(concatenate=True)
            if not isinstance(trimesh_model, (Trimesh, PointCloud)):
                raise ValueError(f"Failed to extract a single Trimesh from scene {file_path} after concatenation.")

        instance._process_trimesh_object(trimesh_model, radii)

        print(f"Created Asset '{instance.name}' of type {instance.asset_type.name} from file {file_path}.")
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
        """
        Creates an Asset from numpy arrays using trimesh
        """

        instance = cls(name)

        instance._process_texture_source(texture)

        trimesh_model = trimesh_from_arrays(
            vertices=vertices, faces=faces, normals=normals,
            vertex_colors=vertex_colors, uv_coords=uv_coords, texture_image=instance._texture_image_data
        )
        if trimesh_model is None:
            raise ValueError("Failed to create Trimesh object from arrays.")

        instance._process_trimesh_object(trimesh_model, radii=radii)
        print(f"Created Asset '{instance.name}' of type {instance.asset_type.name} from arrays.")
        return instance

    def _process_texture_source(self, texture_source: Optional[Union[Path, str, Image.Image, np.ndarray]]):
        """Internal helper to process the texture argument."""

        if isinstance(texture_source, Image.Image):
            self._texture_image_data = texture_source

        elif isinstance(texture_source, np.ndarray):
            try:
                self._texture_image_data = Image.fromarray(texture_source)
            except Exception as e:
                print(f"Warning: Failed to convert numpy array to PIL Image for asset '{self.name}': {e}")

        elif isinstance(texture_source, (Path, str)):
            self.texture_path = Path(texture_source)

        elif texture_source is not None:
            print(f"Warning: Unrecognized texture_source type for asset '{self.name}'. Ignoring texture.")

    def _process_trimesh_object(self, trimesh_obj: Union[Trimesh, PointCloud], radii: Optional[Union[float, ArrayLike]]):
        """
        Internal method to populate Asset's data from a trimesh.Trimesh object
        """
        if trimesh_obj.is_empty:
            raise ValueError(f"Trimesh object is empty for asset '{self.name}'.")

        # Decide if mesh or point cloud based on presence of faces attribute and its content
        if isinstance(trimesh_obj, Trimesh) and trimesh_obj.faces is not None and len(trimesh_obj.faces) > 0:
            self.asset_type = AssetType.Mesh
            self._setup_mesh_data(trimesh_obj)

        elif isinstance(trimesh_obj, PointCloud) and trimesh_obj.vertices is not None and len(trimesh_obj.vertices) > 0:
            self.asset_type = AssetType.Points
            self._setup_point_cloud_data(trimesh_obj, radii)

        else:
            raise ValueError(f"Trimesh object for asset '{self.name}' has no discernible geometry (vertices/faces).")

    def _setup_mesh_data(self, trimesh_obj: Trimesh):
        """ Populates mesh-specific data from a trimesh.Trimesh object. """

        vertices_3d = trimesh_obj.vertices.astype(np.float32)
        indices = trimesh_obj.faces.astype(np.uint32)

        uvs = np.zeros((len(vertices_3d), 2), dtype=np.float32)
        if hasattr(trimesh_obj.visual, 'uv') and trimesh_obj.visual.uv is not None and trimesh_obj.visual.uv.shape[0] == \
                vertices_3d.shape[0]:
            uvs = trimesh_obj.visual.uv.astype(np.float32)
        else:
            print(f"Warning: Mesh '{self.name}' has no valid UVs or UVs count mismatch. Generating dummy UVs.")

        self.vertices = np.concatenate((vertices_3d, uvs), axis=1)
        self.indices = indices

        if self.texture_path is not None:
            print(f"Info: Asset '{self.name}' using external texture from path: {self.texture_path}")
        elif self._texture_image_data is not None:
            print(f"Info: Asset '{self.name}' using provided in-memory texture.")
        elif hasattr(trimesh_obj.visual, 'material') and hasattr(trimesh_obj.visual.material, 'image') and trimesh_obj.visual.material.image is not None:
            self._texture_image_data = trimesh_obj.visual.material.image
            print(f"Info: Asset '{self.name}' picked up an embedded texture from the loaded model.")
        else:
            print(f"Info: Asset '{self.name}' has no texture source (path or embedded image).")

        # TODO: if no texture_path and no embedded image this should use vertex colors if possible

    def _setup_point_cloud_data(self, trimesh_obj: Trimesh, radii: Optional[Union[float, ArrayLike]]):
        """ Populates point cloud-specific data from a trimesh.Trimesh object. """

        self.points = trimesh_obj.vertices.astype(np.float32)
        self._nb_points = len(self.points)

        # Extract colors and normals from trimesh visual or compute/default
        if hasattr(trimesh_obj.visual, 'vertex_colors') and trimesh_obj.visual.vertex_colors is not None and \
                trimesh_obj.visual.vertex_colors.shape[0] == self._nb_points:
            # Convert 0-255 uint8 to 0.0-1.0 float if necessary
            if trimesh_obj.visual.vertex_colors.dtype == np.uint8:
                self.colors = trimesh_obj.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            else:
                self.colors = trimesh_obj.visual.vertex_colors[:, :3].astype(np.float32)
        else:
            self.colors = np.ones_like(self.points, dtype=np.float32)  # Default white

        if hasattr(trimesh_obj, 'vertex_normals') and trimesh_obj.vertex_normals is not None and \
                trimesh_obj.vertex_normals.shape[0] == self._nb_points:
            self.normals = trimesh_obj.vertex_normals.astype(np.float32)
        else:
            self.normals = np.zeros_like(self.points, dtype=np.float32)  # Default (0,0,0) if no normals

        # Handle radii
        if isinstance(radii, (float, int)):
            self.radii = np.full(self._nb_points, radii, dtype=np.float32)
        elif radii is not None:
            self.radii = np.asarray(radii, dtype=np.float32)
            if len(self.radii) != self._nb_points:
                raise ValueError("Number of radii must match the number of points.")
        else:
            # Fallback to a default only if radii is explicitly None
            self.radii = np.full(self._nb_points, 0.05, dtype=np.float32)

    @property
    def nb_triangles(self):
        if self.asset_type == AssetType.Mesh and self.indices is not None:
            return len(self.indices)
        return 0

    @property
    def nb_points(self):
        if self.asset_type == AssetType.Points and self.points is not None:
            return self._nb_points
        return 0

    # TODO: more properties


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