from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
from pathlib import Path
from svg.path import parse_path
from sklearn.decomposition import PCA


def get_path_centroid(path):
    """ Calculates the geometric center of a parsed SVG path """
    points = [path[0].start]
    for segment in path:
        points.append(segment.end)
    return np.mean(np.array([(p.real, p.imag) for p in points]), axis=0)


def sample_path(path, num_points=100):
    """ Generates a set of evenly spaced points along an SVG path """
    points = []
    for i in range(num_points):
        pos = i / (num_points - 1)
        point = path.point(pos)
        points.append((point.real, point.imag))
    return np.array(points)


def parse_drosophila_svg(svg_file: Union[Path, str]) -> dict:
    """ Parses the digitized SVG file """

    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    data = {'ommatidia': [], 'stars': [], 'axes': {}, 'boundary': {}}

    for circle in root.findall('svg:circle', ns):
        if circle.get('id') != ['boundary']:
            data['ommatidia'].append((float(circle.get('cx')), float(circle.get('cy'))))

    boundary_circle = root.find(".//*[@id='boundary']", ns)
    if boundary_circle is not None:
        data['boundary'] = {'cx': float(boundary_circle.get('cx')), 'cy': float(boundary_circle.get('cy')),
                            'r': float(boundary_circle.get('r'))}
    else:
        raise ValueError("Boundary circle with id='boundary' not found in SVG.")

    for i in range(1, 6):
        star_path_element = root.find(f".//*[@id='star-{i}']", ns)
        if star_path_element is not None:
            path_d = star_path_element.get('d')
            parsed_path = parse_path(path_d)
            centroid = get_path_centroid(parsed_path)
            data['stars'].append(tuple(centroid))

    lattice_origin = root.find(f".//*[@id='lattice-origin']", ns)
    if lattice_origin is not None:
        path_d = lattice_origin.get('d')
        parsed_path = parse_path(path_d)
        centroid = get_path_centroid(parsed_path)
        data['lattice-origin'] = tuple(centroid)

    for axis_id in ['axis-x', 'axis-y', 'axis-v']:
        axis_path_element = root.find(f".//*[@id='{axis_id}']", ns)
        if axis_path_element is not None:
            data['axes'][axis_id] = parse_path(axis_path_element.get('d'))

    print(
        f"Parsed SVG: Found {len(data['ommatidia'])} ommatidia, {len(data['stars'])} stars, and {len(data['axes'])} axes.")
    return data


def inverse_equatorial_stereographic(x, y, center_lon_deg):
    lon0_rad = np.deg2rad(center_lon_deg)
    rho = np.sqrt(x ** 2 + y ** 2)
    rho[rho == 0] = 1e-9
    c = 2 * np.arctan(rho / 2.0)
    lat_rad = np.arcsin((y * np.sin(c)) / rho)
    lon_rad = lon0_rad + np.arctan2(x * np.sin(c), rho * np.cos(c))
    return np.rad2deg(lon_rad), np.rad2deg(lat_rad)


def spherical_to_opengl(lon_deg, lat_deg):
    lon_rad, lat_rad = np.deg2rad(lon_deg), np.deg2rad(lat_deg)
    x = np.cos(lat_rad) * np.sin(lon_rad)
    y = np.sin(lat_rad)
    z = -np.cos(lat_rad) * np.cos(lon_rad)
    return np.array([x, y, z])


##


PLOT_BOTH_EYES = False

svg_file = Path('biological_data/drosophila/drosophila_Buchner_1971_redigitized.svg')
eye_data_2d = parse_drosophila_svg(svg_file)


def transform_points_2d_to_3d(points_2d, boundary):
    translated = points_2d - np.array([boundary['cx'], boundary['cy']])

    translated[:, 0] *= -1  # flip X for correct anterior-posterior orientation
    translated[:, 1] *= -1  # also flip Y because SVG Y is downwards

    scaled = translated * (2.0 / boundary['r'])
    lon, lat = inverse_equatorial_stereographic(scaled[:, 0], scaled[:, 1], -90.0)
    return spherical_to_opengl(lon, lat).T


boundary = eye_data_2d['boundary']
ommatidia_3d = transform_points_2d_to_3d(np.array(eye_data_2d['ommatidia']), boundary)
stars_3d = transform_points_2d_to_3d(np.array(eye_data_2d['stars']), boundary)
lattice_centre_3d = transform_points_2d_to_3d(np.array([eye_data_2d['lattice-origin']]), boundary)

axes_3d = {}
for axis_id, path_data in eye_data_2d['axes'].items():
    if path_data:
        axis_points_2d = sample_path(path_data, 200)
        axes_3d[axis_id] = transform_points_2d_to_3d(axis_points_2d, boundary)

# Data regularisation
print("\nRegularising eye model based on reference stars...")
curve_coeffs = np.polyfit(stars_3d[:, 1], stars_3d[:, 0], 2)
deviation_curve = np.poly1d(curve_coeffs)


def regularise_points_3d(points_3d, curve_func):
    corrected = points_3d.copy()
    shifts = curve_func(corrected[:, 1])
    corrected[:, 0] -= shifts
    norms = np.linalg.norm(corrected, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return corrected / norms


ommatidia_final = regularise_points_3d(ommatidia_3d, deviation_curve)
stars_final = regularise_points_3d(stars_3d, deviation_curve)
lattice_centre_final = regularise_points_3d(lattice_centre_3d, deviation_curve)

axes_final = {}
normal_vectors = []
for axis_id, points_3d in axes_3d.items():
    axes_final[axis_id] = regularise_points_3d(points_3d, deviation_curve)

    # Use PCA to find the normal of the best-fit plane
    pca = PCA(n_components=3)
    pca.fit(axes_final['axis-x'])
    normal_vectors.append(pca.components_[2])

print("Regularization complete.")

# Plot
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(ommatidia_final[:, 0], ommatidia_final[:, 1], ommatidia_final[:, 2], c='grey', s=10, alpha=0.6,
           label='Ommatidia')

# Plot colored lattice axes
axis_colors = {'axis-x': '#ff60b3', 'axis-y': '#00FF9B', 'axis-v': '#FFC400'}
for axis_id, points in axes_final.items():
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=axis_colors[axis_id], linewidth=2.5,
            label=f'Axis {axis_id[-1].upper()}')

# Plot markers on top
ax.scatter(stars_final[:, 0], stars_final[:, 1], stars_final[:, 2], c='red', s=150, marker='*',
           label='Forward direction', depthshade=False)
ax.scatter(lattice_centre_final[:, 0], lattice_centre_final[:, 1], lattice_centre_final[:, 2], c='black', s=20,
           marker='X', label='Lattice origin', depthshade=False)

if PLOT_BOTH_EYES:
    # Create the right eye by reflecting the final left eye
    ommatidia_right = ommatidia_final.copy()
    ommatidia_right[:, 0] *= -1

    # Plot the right eye ommatidia
    ax.scatter(ommatidia_right[:, 0], ommatidia_right[:, 1], ommatidia_right[:, 2], c='#555555',
               s=10, alpha=0.6)

# OpenGL axes gizmo for clarity
ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='Right (+X)')
ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Up (+Y)')
ax.quiver(0, 0, 0, 0, 0, -0.5, color='b', label='Forward (-Z)')

ax.set_title("Drosophila Eye from Buchner 1971", fontsize=16)

ax.set_box_aspect([1, 1, 1])

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.view_init(elev=50, azim=25, roll=110)
ax.legend()
plt.show()