from typing import Union
import trimesh
from trimesh import Trimesh, PointCloud, Scene as TrimeshScene
from trimesh.visual.texture import SimpleMaterial, TextureVisuals
import numpy as np
from PIL import Image
from pathlib import Path


def trimesh_from_file(filepath: Union[Path, str]) -> Union[Trimesh, TrimeshScene, PointCloud, None]:
    """
    Loads a 3D model from a specified file path using trimesh.

    Supports various formats like OBJ, GLTF, PLY, STL, etc.
    Automatically handles associated data like normals, vertex colors,
    and texture UV coordinates if present in the file.

    Args:
        filepath (Path, str): The path to the 3D model file.

    Returns:
        trimesh.Trimesh or trimesh.Scene: A trimesh object representing the loaded model.
                                          Returns trimesh.Trimesh for single meshes,
                                          trimesh.Scene for files containing multiple meshes
                                          or scenes (e.g., glTF).
                                          Returns None if loading fails or file not found.
    """

    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: File not found at {filepath}")
        return None

    try:
        model = trimesh.load(filepath)
        print(f"Successfully loaded model from {filepath}")

        # Provide information about the loaded model's attributes
        if isinstance(model, trimesh.Trimesh):
            _print_trimesh_info(model, indent="  ")
            
        elif isinstance(model, trimesh.Scene):
            print(f"  Loaded a Scene with {len(model.geometry)} geometries.")
            
            for name, geo in model.geometry.items():
                print(f"    - Geometry '{name}':")
                _print_trimesh_info(geo, indent="      ")

        elif isinstance(model, trimesh.PointCloud):
            print(f"  Loaded a PointCloud with {len(model.vertices)} vertices.")
            _print_trimesh_info(model, indent="  ")

        return model
    
    except Exception as e:
        print(f"Error loading model from {filepath}: {e}")
        return None


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
                         Or None if an error occurs.
    """
    if vertices is None or vertices.ndim != 2 or vertices.shape[1] != 3:
        print("Error: 'vertices' must be an N x 3 numpy array.")
        return None

    try:
        # Determine if we're creating a Trimesh or PointCloud
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

        # Apply visuals to mesh_kwargs
        if 'uv' in visual_args or 'material' in visual_args:
            if is_mesh:  # only Trimesh can directly handle TextureVisuals in constructor
                mesh_kwargs['visual'] = TextureVisuals(**visual_args)
            else:
                print("Warning: Texture/UVs provided for a PointCloud. Trimesh.PointCloud does not directly "
                    "support TextureVisuals. Visual information will be limited to vertex colors if provided.")

        elif 'vertex_colors' in visual_args:
            # for both Trimesh and PointCloud, vertex_colors can be passed directly
            mesh_kwargs['vertex_colors'] = visual_args['vertex_colors']

        if is_mesh:
            model = Trimesh(**mesh_kwargs)  # use Trimesh if faces are valid
        else:
            model = PointCloud(**mesh_kwargs)  # use PointCloud otherwise

        print(
            f"Successfully created Trimesh object from arrays.")  # This is a bit inaccurate for PointCloud butokay
        _print_trimesh_info(model)
        return model

    except Exception as e:
        print(f"Error creating model from arrays: {e}")
        return None


def _print_trimesh_info(model: Union[Trimesh, PointCloud], indent: str = ""):
    
    if model.vertices is not None:
        print(f"{indent}Vertices: {model.vertices.shape}")

    if hasattr(model, 'faces') and model.faces is not None:
        print(f"{indent}Faces: {model.faces.shape}")
    else:
        print(f"{indent}No faces (point cloud or not triangulated).")

    if hasattr(model, 'vertex_normals') and model.vertex_normals is not None and model.vertex_normals.shape[0] > 0:
        print(f"{indent}Vertex Normals present (either provided or calculated).")
    else:
        print(f"{indent}No explicit vertex normals present.")

    if hasattr(model, 'visual') and model.visual is not None:
        
        if hasattr(model.visual, 'vertex_colors') and model.visual.vertex_colors is not None and model.visual.vertex_colors.shape[0] > 0:
            print(f"{indent}Vertex Colors present.")
            
        if hasattr(model.visual, 'uv') and model.visual.uv is not None and model.visual.uv.shape[0] > 0:
            print(f"{indent}UV coordinates present.")
            
        if hasattr(model.visual, 'material') and model.visual.material is not None:
            
            if hasattr(model.visual.material, 'image') and model.visual.material.image is not None:
                print(f"{indent}Material with texture image present.")
                
            elif hasattr(model.visual.material, 'to_color') and model.visual.material.to_color() is not None:
                print(f"{indent}Material with solid color present.")
    else:
        print(f"{indent}No explicit visual attributes (e.g., colors, texture).")