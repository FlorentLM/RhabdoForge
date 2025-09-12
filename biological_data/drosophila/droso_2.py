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


def generate_small_circle_points(points_on_sphere, num_points=200):
    """
    Fits a plane to the given points and generates the points for the
    resulting small circle of intersection with the unit sphere.
    """

    # Fit a plane to the data using PCA. This plane does NOT have to pass through the origin.
    pca = PCA(n_components=3)
    pca.fit(points_on_sphere)
    normal = pca.components_[2]
    point_on_plane = pca.mean_

    # Ensure the normal points towards the origin for consistency
    if np.dot(normal, point_on_plane) > 0:
        normal *= -1

    dist_from_origin = np.dot(normal, point_on_plane)
    circle_center = dist_from_origin * normal

    circle_radius = np.sqrt(1 - dist_from_origin ** 2)

    # Create an orthonormal basis for the plane
    v1 = np.cross(normal, [0, 0, 1])
    if np.linalg.norm(v1) < 1e-6:
        v1 = np.cross(normal, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    # Generate points around the circle
    t = np.linspace(0, 2 * np.pi, num_points)
    circle_points = circle_center + circle_radius * (np.outer(np.cos(t), v1) + np.outer(np.sin(t), v2))
    return circle_points


def rotate_vector(vector, axis, angle):
    """ Rotates a vector around an axis by a given angle (in radians). """
    axis = axis / np.linalg.norm(axis)
    return (vector * np.cos(angle) +
            np.cross(axis, vector) * np.sin(angle) +
            axis * np.dot(axis, vector) * (1 - np.cos(angle)))


##

PLOT_BOTH_EYES = False
NUDGE_OMMATIDIA = False
REGULARIZE_AXIS_ANGLES = True

svg_file = Path('biological_data/drosophila/drosophila_Buchner_1971_redigitized.svg')
eye_data_2d = parse_drosophila_svg(svg_file)

def transform_points_2d_to_3d(points_2d, boundary):
    translated = points_2d - np.array([boundary['cx'], boundary['cy']])
    translated[:, 0] *= -1
    translated[:, 1] *= -1
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
for axis_id, points_3d in axes_3d.items():
    axes_final[axis_id] = regularise_points_3d(points_3d, deviation_curve)

print("Regularization complete.")

print("\nFitting ideal small circles to lattice axes...")

# Fit initial planes to get original normals and tilts
initial_planes = {}
for axis_id, points in axes_final.items():
    pca = PCA(n_components=3)
    pca.fit(points)
    initial_planes[axis_id] = {'n': pca.components_[2], 'mean': pca.mean_}

C = lattice_centre_final.flatten()
C_norm = C / np.linalg.norm(C)

final_planes = {}

if REGULARIZE_AXIS_ANGLES:
    print("Enforcing 60/120-degree separation between axes in the tangent plane...")
    # Get reference tangent from Axis X
    n_x, p_mean_x = initial_planes['axis-x']['n'], initial_planes['axis-x']['mean']
    t_x = np.cross(n_x, C_norm)
    t_x /= np.linalg.norm(t_x)
    final_planes['axis-x'] = {'n': n_x, 'p': p_mean_x}

    # Create ideal tangents for Y and V by rotating in the tangent plane
    t_y_ideal = rotate_vector(t_x, C_norm, np.deg2rad(60))
    t_v_ideal = rotate_vector(t_x, C_norm, np.deg2rad(120))

    # Reconstruct the planes for Y and V
    for axis_id, t_ideal in [('axis-y', t_y_ideal), ('axis-v', t_v_ideal)]:
        # Preserve the original plane's tilt by preserving its distance from the origin
        n_orig, p_mean_orig = initial_planes[axis_id]['n'], initial_planes[axis_id]['mean']
        dist_preserved = np.dot(n_orig, p_mean_orig)

        # The new normal must be perpendicular to the ideal tangent
        # We solve for the new normal `n_ideal = a*C + b*ortho_vec`
        a = dist_preserved
        b_sq = 1 - a ** 2
        b = np.sqrt(b_sq) if b_sq > 0 else 0

        ortho_vec = np.cross(t_ideal, C_norm)
        ortho_vec /= np.linalg.norm(ortho_vec)

        n_ideal = a * C_norm + b * ortho_vec
        final_planes[axis_id] = {'n': n_ideal, 'p': C}  # the new plane must pass through C
else:
    for axis_id, plane_data in initial_planes.items():
        final_planes[axis_id] = {'n': plane_data['n'], 'p': plane_data['mean']}

# Fitting ideal circles to lattice axes
ideal_axes_points = {}
for axis_id, points in axes_final.items():
    ideal_axes_points[axis_id] = generate_small_circle_points(points)

# Plotting
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(ommatidia_final[:, 0], ommatidia_final[:, 1], ommatidia_final[:, 2], c='grey', s=10, alpha=0.6,
           label='Ommatidia')

axis_colors = {'axis-x': '#ff60b3', 'axis-y': '#00FF9B', 'axis-v': '#FFC400'}
for axis_id, points in axes_final.items():
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=axis_colors[axis_id], linewidth=2.5,
            label=f'Axis {axis_id[-1].upper()}')

# Plot the ideal circles as dotted lines
for axis_id, points in ideal_axes_points.items():
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color=axis_colors[axis_id],
            linestyle='--', linewidth=1.5, label=f'Ideal Axis {axis_id[-1].upper()}')

# if NUDGE_OMMATIDIA:
#
#     ax.scatter(ommatidia_nudged[:, 0], ommatidia_nudged[:, 1], ommatidia_nudged[:, 2],
#                c='green', s=15, alpha=0.5, label='Nudged Ommatidia')
#     for p_orig, p_nudge in zip(ommatidia_final, ommatidia_nudged):
#         ax.plot([p_orig[0], p_nudge[0]], [p_orig[1], p_nudge[1]], [p_orig[2], p_nudge[2]],
#                 color='black', linewidth=0.3)

ax.scatter(stars_final[:, 0], stars_final[:, 1], stars_final[:, 2], c='red', s=150, marker='*',
           label='Forward direction', depthshade=False)
ax.scatter(lattice_centre_final[:, 0], lattice_centre_final[:, 1], lattice_centre_final[:, 2], c='black', s=20,
           marker='X', label='Lattice origin', depthshade=False)

if PLOT_BOTH_EYES:
    ommatidia_right = ommatidia_final.copy()
    ommatidia_right[:, 0] *= -1
    ax.scatter(ommatidia_right[:, 0], ommatidia_right[:, 1], ommatidia_right[:, 2], c='#555555',
               s=10, alpha=0.6)

ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='Right (+X)')
ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Up (+Y)')
ax.quiver(0, 0, 0, 0, 0, -0.5, color='b', label='Forward (-Z)')

ax.set_title("Drosophila Eye from Buchner 1971 (Small Circle Fit)", fontsize=16)
ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.view_init(elev=50, azim=25, roll=110)
ax.legend()
plt.show()