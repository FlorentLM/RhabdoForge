import numpy as np
from svg.path import parse_path
import xml.etree.ElementTree as ET
import re

def parse_drosophila_svg(svg_file: str) -> dict:
    """
    Parses the digitized SVG file to extract ommatidia, stars, axes, and boundary info
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    data = {'ommatidia': [], 'stars': [], 'axes': {}, 'boundary': {}}

    for circle in root.findall('svg:circle', ns):
        if circle.get('id') != 'boundary':
            data['ommatidia'].append((float(circle.get('cx')), float(circle.get('cy'))))

    boundary_circle = root.find(".//*[@id='boundary']", ns)
    if boundary_circle is not None:
        data['boundary'] = {
            'cx': float(boundary_circle.get('cx')),
            'cy': float(boundary_circle.get('cy')),
            'r': float(boundary_circle.get('r'))
        }
    else:
        raise ValueError("Boundary circle with id='boundary' not found in SVG.")

    for i in range(1, 6):
        star_path_element = root.find(f".//*[@id='star-{i}']", ns)
        if star_path_element is not None:
            path_d = star_path_element.get('d')
            cleaned_path = re.sub(r"[A-Za-z,]", " ", path_d)
            coords = [float(c) for c in cleaned_path.split()]
            points = np.array(coords).reshape(-1, 2)
            data['stars'].append(tuple(np.mean(points, axis=0)))

    for axis_id in ['axis-x', 'axis-y', 'axis-v']:
        axis_path_element = root.find(f".//*[@id='{axis_id}']", ns)
        if axis_path_element is not None:
            data['axes'][axis_id] = parse_path(axis_path_element.get('d'))

    print(f"Parsed SVG: Found {len(data['ommatidia'])} ommatidia, {len(data['stars'])} stars, and {len(data['axes'])} axes.")
    return data

def stereographic_to_spherical(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    rho = np.clip(rho, 0, 1e6)
    theta = np.arctan2(y, x)
    colatitude = 2 * np.arctan(rho / 2.0)
    latitude = np.pi / 2.0 - colatitude
    longitude = theta
    return longitude, latitude

def spherical_to_cartesian(longitude, latitude):
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)
    return np.vstack([x, y, z]).T

# =============================================================================

# TODO: Fix orientation in this


svg_data = parse_drosophila_svg('stuff/biological_data/drosophila/drosophila_Buchner_1971_redigitized.svg')
bc = svg_data['boundary']

def px_to_stereo(coords_px):
    coords_px = np.atleast_2d(coords_px)
    x_stereo = (coords_px[:, 0] - bc['cx']) / bc['r']
    y_stereo = -(coords_px[:, 1] - bc['cy']) / bc['r']
    return np.stack([x_stereo, y_stereo], axis=-1)

def stereo_to_3d(stereo_coords):
    long, lat = stereographic_to_spherical(stereo_coords[:, 0], stereo_coords[:, 1])
    return spherical_to_cartesian(long, lat)

# Convert to initial 3D system
initial_ommatidia_dirs = stereo_to_3d(px_to_stereo(svg_data['ommatidia']))
initial_star_dirs = stereo_to_3d(px_to_stereo(svg_data['stars']))

# Determine Orientation into internal Z-up system
# +X = Forward, +Y = Right, +Z = Up
forward_target = np.array([1., 0., 0.])
up_target = np.array([0., 0., 1.])
right_target = np.cross(forward_target, up_target)

forward_initial = np.mean(initial_star_dirs, axis=0)
forward_initial /= np.linalg.norm(forward_initial)

axis_v = svg_data['axes']['axis-v']
p1_complex, p2_complex = axis_v.point(0.5), axis_v.point(0.51)
p1_3d, p2_3d = stereo_to_3d(px_to_stereo((p1_complex.real, p1_complex.imag))), stereo_to_3d(px_to_stereo((p2_complex.real, p2_complex.imag)))
up_initial = (p2_3d[0] - p1_3d[0])
up_initial /= np.linalg.norm(up_initial)

right_initial = np.cross(forward_initial, up_initial)
right_initial /= np.linalg.norm(right_initial)
up_initial = np.cross(right_initial, forward_initial)

initial_basis = np.array([forward_initial, right_initial, up_initial]).T
target_basis = np.array([forward_target, right_target, up_target]).T
transform_matrix = target_basis @ np.linalg.inv(initial_basis)

left_eye_dirs = initial_ommatidia_dirs @ transform_matrix.T
right_eye_dirs = left_eye_dirs.copy()
right_eye_dirs[:, 1] *= -1 # Mirror across XZ plane for the right eye

num_ommatidia = len(left_eye_dirs)
eye_radius = 0.000420

blind_band = 40     # in degrees
eye_separation = 2 * eye_radius * np.tan(np.radians(blind_band / 2))

left_eye_origins = left_eye_dirs * eye_radius - np.array([0, eye_separation / 2, 0])
right_eye_origins = right_eye_dirs * eye_radius + np.array([0, eye_separation / 2, 0])

# This data is internally consistent in our Z-up system
internal_dirs = np.concatenate((left_eye_dirs, right_eye_dirs))
internal_origins = np.concatenate((left_eye_origins, right_eye_origins))
internal_ids = np.concatenate((np.zeros(num_ommatidia, dtype=int), np.ones(num_ommatidia, dtype=int)))

# ==============================================================================
# Coord system adjustment
# ==============================================================================

final_dirs = np.zeros_like(internal_dirs)
final_origins = np.zeros_like(internal_origins)

# Remap the axes to match the renderer's convention
# Renderer Convention: +X = Right, +Y = Up, -Z = Forward

final_dirs[:, 0] = internal_dirs[:, 1]
final_origins[:, 0] = internal_origins[:, 1]
final_dirs[:, 1] = internal_dirs[:, 2]
final_origins[:, 1] = internal_origins[:, 2]
final_dirs[:, 2] = -internal_dirs[:, 0]
final_origins[:, 2] = -internal_origins[:, 0]

np.savez_compressed(
    "drosophila_eye.npz",
    directions=final_dirs,
    origins=final_origins,
    eye_id=internal_ids
)

print("\nDrosophila eyes model generated successfully!")