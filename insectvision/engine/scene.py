from dataclasses import dataclass

import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from typing import Dict, List, Optional, Union, Sequence, Set
from numpy.typing import ArrayLike
from enum import Enum, auto
from pathlib import Path
import numpy as np
from PIL import Image
import trimesh
from pyglm import glm

from insectvision.engine.lights import Sun, Light, DirectionalLight, PointLight, AreaLight
from insectvision.engine.movement import TransformMixin
from insectvision.engine.materials_utils import load_exr_equirect, sh_irradiance, get_exr_sun


def trimesh_from_arrays(
    vertices: np.ndarray,
    faces: np.ndarray = None,
    normals: np.ndarray = None,
    vertex_colors: np.ndarray = None,
    uv_coords: np.ndarray = None,
    texture_image: Image.Image = None
) -> Union[trimesh.Trimesh, trimesh.PointCloud, None]:
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

                    material = trimesh.visual.material.SimpleMaterial(image=texture_image)

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
                mesh_kwargs['visual'] = trimesh.visual.TextureVisuals(**visual_args)

            else:
                print("Warning: Texture/UVs provided for a PointCloud. trimesh.PointCloud does not directly "
                    "support TextureVisuals. Visual information will be limited to vertex colors if provided.")

        elif 'vertex_colors' in visual_args:
            mesh_kwargs['vertex_colors'] = visual_args['vertex_colors']

        if is_mesh:
            model = trimesh.Trimesh(**mesh_kwargs)
        else:
            model = trimesh.PointCloud(**mesh_kwargs)

        return model

    except Exception as e:
        print(f"Error creating model from arrays: {e}")
        return None

##

@dataclass
class MaterialData:
    """
    Material properties for rendering (colours, specular, etc.)
    """
    base_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    specular = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # w = shininess
    emission = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class AssetType(Enum):
    """
    Distinguishes between different types of geometry assets.
    """
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
        self.is_srgb = True
        self.texture_id: Optional[int] = None  # OpenGL texture ID (set by renderer)

        self._material_rev: int = 0
        self._texture_rev: int = 0
        self._geometry_rev: int = 0

    def __repr__(self):
        kind = self.asset_type.name if self.asset_type else "Uninitialised"
        prim = f"{self.nb_triangles} tris" if self.asset_type == AssetType.Mesh else f"{self.nb_points} pts"
        return f"<Asset '{self.name}' | {kind} | {prim} | {'textured' if self.has_texture else 'untextured'}>"

    def touch_material(self) -> None:
        """Mark material (base colour/specular/emission) as changed."""
        self._material_rev += 1

    def touch_texture(self) -> None:
        """Mark the texture image as changed."""
        self._texture_rev += 1

    def touch_geometry(self) -> None:
        """Mark vertices/points as changed (triggers a BLAS refit or rebuild)."""
        self._geometry_rev += 1

    @property
    def texture_path(self) -> Optional[Path]:
        return self._texture_path

    @property
    def texture_image(self) -> Optional[Image.Image]:
        """
        Returns the texture as a PIL Image (lazily).
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

    def set_material(self, base_color=None, specular=None, emission=None) -> None:
        """Update material fields and flag the change."""
        if base_color is not None:
            self.material.base_color = np.asarray(base_color, dtype=np.float32)
        if specular is not None:
            self.material.specular = np.asarray(specular, dtype=np.float32)
        if emission is not None:
            self.material.emission = np.asarray(emission, dtype=np.float32)
        self.touch_material()

    def set_texture(self, source: Union[Path, str, Image.Image, np.ndarray, None], sRGB: bool = True):
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
        self.touch_texture()

        if source is None:
            return

        if isinstance(source, (Path, str)):
            self._texture_path = Path(source)

        elif isinstance(source, Image.Image):
            self._texture_image = source.convert('RGBA')

        elif isinstance(source, np.ndarray):
            try:
                self._texture_image = Image.fromarray(source).convert('RGBA')
            except Exception as e:
                print(f"Warning: Failed to convert numpy array to image for asset '{self.name}': {e}")
        else:
            print(f"Warning: Unrecognized texture source type {type(source).__name__} for asset '{self.name}'")

        self.is_srgb = sRGB

    @classmethod
    def from_file(cls,
            name: str,
            file_path: Union[Path, str],
            texture: Optional[Union[Path, str, Image.Image, np.ndarray]] = None,
            radii: Optional[Union[float, ArrayLike]] = None,
            sRGB: bool = True
        ):
        """
        Creates an Asset by loading a 3D model from a file.
        """

        asset = cls(name)

        # texture override will be used instead of embedded texture if provided
        texture_override = texture is not None
        if texture_override:
            asset.set_texture(texture)

        trimesh_model = trimesh.load(file_path)

        if trimesh_model is None:
            raise ValueError(f"Failed to load 3D model from {file_path}")

        if isinstance(trimesh_model, trimesh.Scene):
            print(f"Info: File '{file_path}' contains multiple meshes. Merging into single Asset '{name}'.")

            trimesh_model = trimesh_model.dump(concatenate=True)

            if not isinstance(trimesh_model, (trimesh.Trimesh, trimesh.PointCloud)):
                raise ValueError(f"Failed to extract geometry from scene {file_path}")

        asset.process_trimesh(trimesh_model, radii, extract_texture=not texture_override)

        asset.is_srgb = sRGB

        print(f"Created Asset '{asset.name}' ({asset.asset_type.name}) from {file_path}")
        return asset

    @classmethod
    def from_arrays(cls,
            name: str,
            vertices: np.ndarray,
            faces: Optional[np.ndarray] = None,
            normals: Optional[np.ndarray] = None,
            vertex_colors: Optional[np.ndarray] = None,
            uv_coords: Optional[np.ndarray] = None,
            texture: Optional[Union[Path, str, Image.Image, np.ndarray]] = None,
            sRGB: bool = True,
            radii: Optional[Union[float, ArrayLike]] = None
        ):
        """
        Creates an Asset from numpy arrays.
        """

        asset = cls(name)

        if texture is not None:
            asset.set_texture(texture)

        # For trimesh texture mapping, we need the image if UVs are provided
        texture_for_trimesh = asset._texture_image if uv_coords is not None else None

        trimesh_model = trimesh_from_arrays(
            vertices=vertices, faces=faces, normals=normals,
            vertex_colors=vertex_colors, uv_coords=uv_coords,
            texture_image=texture_for_trimesh
        )

        if trimesh_model is None:
            raise ValueError("Failed to create geometry from arrays.")

        asset.process_trimesh(trimesh_model, radii, extract_texture=False)
        asset.is_srgb = sRGB

        print(f"Created Asset '{asset.name}' ({asset.asset_type.name}) from arrays")
        return asset

    def process_trimesh(self,
                        trimesh_obj: Union[trimesh.Trimesh, trimesh.PointCloud],
                        radii: Optional[Union[float, ArrayLike]],
                        extract_texture: bool = True
                        ):
        """
        Populates Asset data from a trimesh object.
        """

        if trimesh_obj.is_empty:
            raise ValueError(f"Geometry is empty for asset '{self.name}'.")

        if isinstance(trimesh_obj, trimesh.Trimesh) and trimesh_obj.faces is not None and len(trimesh_obj.faces) > 0:
            self.asset_type = AssetType.Mesh
            self._setup_mesh_data(trimesh_obj, extract_texture)

        elif isinstance(trimesh_obj, trimesh.PointCloud) and trimesh_obj.vertices is not None:
            self.asset_type = AssetType.Points
            self._setup_point_cloud_data(trimesh_obj, radii)

        else:
            raise ValueError(f"No valid geometry found for asset '{self.name}'.")

    def _setup_mesh_data(self, trimesh_obj: trimesh.Trimesh, extract_texture: bool):
        """
        Populates mesh-specific data from a trimesh object.
        """

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

    def _setup_point_cloud_data(self, trimesh_obj: trimesh.PointCloud, radii: Optional[Union[float, ArrayLike]]):
        """
        Populates point cloud-specific data.
        """

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


class Instance(TransformMixin):
    """
    Logical instance of an Asset in the scene (renderer-agnostic).
    """

    def __init__(self,
                 asset: Asset,
                 transform: Optional[Union[glm.mat4, ArrayLike]] = None,
                 dynamic: bool = False,
                 visible: bool = True,
                 **kwargs):

        self.id = id(self)
        self.asset = asset
        self.dynamic = dynamic
        self._visible = visible
        self.properties = kwargs

        if transform is None:
            self.transform = glm.mat4(1.0)
        else:
            t_np = np.asarray(transform, dtype=np.float32)
            if t_np.shape == (4, 4):
                self.transform = glm.mat4(t_np)
            elif t_np.shape == (3,):
                self.transform = glm.translate(glm.mat4(1.0), glm.vec3(t_np))
            else:
                raise ValueError(
                    f"Unsupported shape for transform: {t_np.shape}. "
                    "Expected a (4, 4) matrix or a (3,) position vector."
                )

    @property
    def visible(self) -> bool:
        """
        Whether this instance is included in ray intersection tests.
        (toggling at runtime requires `dynamic=True` so TLAS can refit)
        """
        # TODO: This will eventually be replaced by the visibility masks
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = bool(value)
        self.touch()

    is_visible = visible


class Skybox:
    def __init__(self, texture_path: str | Path = 'textures/sky.exr', max_height: int = 2048):

        self._texture_path = Path(texture_path)
        data = load_exr_equirect(self._texture_path, max_height=max_height)
        self.sh_coeffs = sh_irradiance(data)

        h, w = data.shape[:2]

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, w, h, 0, GL_RGB, GL_FLOAT, data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)  # wrap azimuth
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)  # clamp poles
        glBindTexture(GL_TEXTURE_2D, 0)

        self.texture_id = texture_id


class Scene:
    """
    The logical scene representation. A simple container for assets and instances.
    """

    def __init__(self, background_color: Sequence[float] = (0.0, 0.0, 0.0), skybox_path: Optional[str | Path] = None):
        self.assets: Dict[str, 'Asset'] = {}
        self.background_color = background_color
        self._skybox: Optional['Skybox'] = None

        self._directional_lights: Set['DirectionalLight'] = set()
        self._point_lights: Set['PointLight'] = set()
        self._area_lights: Set['AreaLight'] = set()

        self._dynamic_instances: Set['Instance'] = set()
        self._mesh_instances: Set['Instance'] = set()
        self._point_instances: Set['Instance'] = set()

        self._topology_rev: int = 0
        self._lights_rev: int = 0

        if skybox_path is not None:
            self.add_skybox(skybox_path)
        else:
            default_sun = Sun(intensity=1.0, angular_size=0.05)
            default_sun.azimuth = 4.84
            default_sun.elevation = 39.75

            self._sun_ref = default_sun
            self.add_light(self._sun_ref)

    def __repr__(self):
        return (f"<Scene | {len(self._mesh_instances)} mesh + {len(self._point_instances)} point instances "
                f"| {len(self.assets)} assets | {len(self.lights)} lights"
                f"{' | skybox' if self._skybox else ''}>")

    def touch_topology(self) -> None:
        self._topology_rev += 1

    def touch_lights(self) -> None:
        self._lights_rev += 1

    def add_instance(self,
        asset: Union['Asset', str],
        transform: Optional[Union[glm.mat4, ArrayLike]] = None,
        **kwargs
    ) -> 'Instance':

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

        if asset_obj.asset_type == AssetType.Mesh:
            self._mesh_instances.add(instance)

        elif asset_obj.asset_type == AssetType.Points:
            self._point_instances.add(instance)

        if instance.dynamic:
            self._dynamic_instances.add(instance)

        self._topology_rev += 1

        return instance

    def add_light(self, light: 'Light'):
        if isinstance(light, DirectionalLight):
            self._directional_lights.add(light)
        elif isinstance(light, PointLight):
            self._point_lights.add(light)
        elif isinstance(light, AreaLight):
            self._area_lights.add(light)

        self._lights_rev += 1

    def add_skybox(self, texture_path: str):
        """Creates and loads a skybox from a directory of textures."""

        self._skybox = Skybox(texture_path)

        if self._sun_ref:

            self.remove_light(self._sun_ref)    # TODO: using _sun_ref is kinda crappy

            azim, elev, col, intensity, ang_radius = get_exr_sun(self._skybox._texture_path)
            # TODO: intensity is whack, must fix it
            sun = Sun(azimuth=azim, elevation=elev, intensity=1.0, angular_size=ang_radius, color=col)
            self._sun_ref = sun

            self.add_light(self._sun_ref)

    def remove_instance(self, instance: 'Instance', prune_asset: bool = False):
        self._mesh_instances.discard(instance)
        self._point_instances.discard(instance)
        self._dynamic_instances.discard(instance)

        if prune_asset:
            # Remove the asset from the registry if no remaining instances reference it
            asset = instance.asset
            still_used = any(
                inst.asset.id == asset.id
                for inst in self._mesh_instances | self._point_instances
            )
            if not still_used and asset.name in self.assets:
                del self.assets[asset.name]

        self._topology_rev += 1

    def remove_asset(self, asset: Union['Asset', str]):
        """
        Removes an asset and all of its instances from the scene.
        """
        if isinstance(asset, str):
            if asset not in self.assets:
                return
            asset_obj = self.assets[asset]
        elif isinstance(asset, Asset):
            asset_obj = asset
        else:
            raise TypeError(
                f"Expected Asset or str, got {type(asset).__name__}."
            )

        # Remove every instance that references this asset
        for pool in (self._mesh_instances, self._point_instances, self._dynamic_instances):
            pool -= {inst for inst in pool if inst.asset.id == asset_obj.id}

        self.assets.pop(asset_obj.name, None)
        self._topology_rev += 1

    def remove_light(self, light: 'Light'):
        if isinstance(light, DirectionalLight):
            self._directional_lights.discard(light)
        elif isinstance(light, PointLight):
            self._point_lights.discard(light)
        elif isinstance(light, AreaLight):
            self._area_lights.discard(light)

        self._lights_rev += 1

    def clear_skybox(self):
        self._skybox = None

    def clear_instances(self, prune_assets: bool = False):
        self._mesh_instances.clear()
        self._point_instances.clear()
        self._dynamic_instances.clear()

        if prune_assets:
            self.assets.clear()

        self._topology_rev += 1

    def clear_lights(self):
        self._directional_lights.clear()
        self._point_lights.clear()
        self._area_lights.clear()

        self._lights_rev += 1

    def load(self,
            file_path: Path | str,
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

            for geom_name, geom_obj in data.geometry.items():

                transform_in_file, _ = data.graph.get(geom_name)
                node_transform = glm.mat4(transform_in_file)
                final_transform = user_transform * node_transform

                asset_name = f"{name_prefix}_{geom_name}"

                if asset_name not in self.assets:
                    asset = Asset(asset_name)
                    asset.process_trimesh(geom_obj, radii=kwargs.get('radii'), extract_texture=True)
                    self.assets[asset_name] = asset

                inst = self.add_instance(self.assets[asset_name], transform=final_transform,
                                         **{k: v for k, v in kwargs.items() if k != 'radii'})
                new_instances.append(inst)

        elif isinstance(data, (trimesh.Trimesh, trimesh.PointCloud)):
            # Single-geometry file

            asset_name = name_prefix

            if asset_name not in self.assets:
                asset = Asset(asset_name)
                asset.process_trimesh(data, radii=kwargs.get('radii'), extract_texture=True)
                self.assets[asset_name] = asset

            inst = self.add_instance(self.assets[asset_name], transform=user_transform,
                                     **{k: v for k, v in kwargs.items() if k != 'radii'})
            new_instances.append(inst)

        else:
            print(f"Warning: Unsupported data type from '{file_path}': {type(data)}")

        return new_instances

    @property
    def instances(self) -> List['Instance']:
        """
        Returns a combined list of all instances.
        """
        return list(self._mesh_instances | self._point_instances)

    @property
    def mesh_instances(self) -> List['Instance']:
        return list(self._mesh_instances)

    @property
    def point_instances(self) -> List['Instance']:
        return list(self._point_instances)

    @property
    def lights(self) -> List['Light']:
        return list(self._directional_lights | self._point_lights | self._area_lights)

    @property
    def directional_lights(self) -> List['DirectionalLight']:
        return list(self._directional_lights)

    @property
    def point_lights(self) -> List['PointLight']:
        return list(self._point_lights)

    @property
    def area_lights(self) -> List['AreaLight']:
        return list(self._area_lights)

    @property
    def skybox(self):
        return self._skybox

    @property
    def total_triangles(self) -> int:
        return sum(inst.asset.nb_triangles for inst in self._mesh_instances)

    @property
    def total_points(self) -> int:
        return sum(inst.asset.nb_points for inst in self._point_instances)

    @property
    def sun(self) -> Optional['Sun']:
        """
        Returns the first Sun found in the directional lights set.
        """
        return next((l for l in self._directional_lights if isinstance(l, Sun)), None)

    @sun.setter
    def sun(self, value: Optional['Sun']):
        """
        Replaces all current Sun instances in the directional lights set.
        """
        suns = [l for l in self._directional_lights if isinstance(l, Sun)]
        for s in suns:
            self._directional_lights.remove(s)
        if value is not None:
            self._directional_lights.add(value)

    def free(self):
        self.assets.clear()
        self.clear_instances()
        self.clear_lights()
        # Note: GPU resources tied to skybox/assets are freed by the bakers/renderers
