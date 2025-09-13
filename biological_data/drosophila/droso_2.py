from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
from pathlib import Path
from svg.path import parse_path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.signal import find_peaks


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
    if len(points_on_sphere) < 3:
        return np.array([]), None, None

    pca = PCA(n_components=3)
    pca.fit(points_on_sphere)
    normal = pca.components_[2]
    point_on_plane = pca.mean_

    if np.dot(normal, point_on_plane) > 0:
        normal *= -1

    dist_from_origin = np.dot(normal, point_on_plane)

    if abs(dist_from_origin) >= 1.0:
        return np.array([]), normal, point_on_plane

    circle_center = dist_from_origin * normal
    circle_radius = np.sqrt(1 - dist_from_origin ** 2)

    v1 = np.cross(normal, [0, 0, 1])
    if np.linalg.norm(v1) < 1e-6:
        v1 = np.cross(normal, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)

    t = np.linspace(0, 2 * np.pi, num_points)
    circle_points = circle_center + circle_radius * (np.outer(np.cos(t), v1) + np.outer(np.sin(t), v2))
    return circle_points, normal, point_on_plane


def create_small_circle_from_normal(normal, point_on_plane, num_points=200):
    dist_from_origin = np.dot(normal, point_on_plane)
    if abs(dist_from_origin) >= 1.0: return np.array([])
    circle_center = dist_from_origin * normal
    circle_radius = np.sqrt(1 - dist_from_origin ** 2)
    v1 = np.cross(normal, [0, 0, 1])
    if np.linalg.norm(v1) < 1e-6: v1 = np.cross(normal, [0, 1, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    t = np.linspace(0, 2 * np.pi, num_points)
    return circle_center + circle_radius * (np.outer(np.cos(t), v1) + np.outer(np.sin(t), v2))


def fit_quasi_parallel_grid_lines(ommatidia, main_axis_plane_normal, n_lines=15):
    dists = np.dot(ommatidia, main_axis_plane_normal).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_lines, random_state=0, n_init='auto').fit(dists)
    labels = kmeans.labels_

    fitted_circles = []
    for i in range(n_lines):
        cluster_ommatidia = ommatidia[labels == i]
        if len(cluster_ommatidia) > 3:
            circle_points, normal, point_on_plane = generate_small_circle_points(cluster_ommatidia)
            if circle_points.size > 0:
                fitted_circles.append({'points': circle_points, 'normal': normal, 'mean': point_on_plane})
    return fitted_circles


def rotate_vector(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return (vector * np.cos(angle) +
            np.cross(axis, vector) * np.sin(angle) +
            axis * np.dot(axis, vector) * (1 - np.cos(angle)))


def clip_lines_to_data_boundary(lines, data_points, padding_factor=1.05):
    if not lines: return []
    center_vec = np.mean(data_points, axis=0)
    center_vec /= np.linalg.norm(center_vec)
    dot_products = np.clip(np.dot(data_points, center_vec), -1.0, 1.0)
    max_angle = np.max(np.arccos(dot_products))
    min_cos_angle = np.cos(max_angle * padding_factor)

    clipped_lines = []
    for line_points in lines:
        line_dots = np.dot(line_points, center_vec)
        outside_mask = line_dots < min_cos_angle
        clipped_line = line_points.copy()
        clipped_line[outside_mask] = np.nan
        clipped_lines.append(clipped_line)
    return clipped_lines


def debug_plot_histogram(dists, n_bins, peaks, hist, axis_id):
    """ A helper function to visualize the histogram and detected peaks for debugging """
    bin_centers = np.linspace(np.min(dists), np.max(dists), n_bins)
    plt.figure(figsize=(10, 4))
    plt.title(f"Histogram of Projected Ommatidia for {axis_id}")
    plt.bar(bin_centers, hist, width=(bin_centers[1] - bin_centers[0]) * 0.9, label='Ommatidia Count')
    plt.plot(bin_centers[peaks], hist[peaks], "x", color='r', markersize=10, label='Detected Peaks (Rows)')
    plt.xlabel("Distance along Axis Normal")
    plt.ylabel("Number of Ommatidia")
    plt.legend()
    plt.show()


def determine_n_lines(ommatidia, plane_normal, axis_id, n_bins=100, debug_plot=False):
    """ Automatically determines the number of ommatidia rows along a given axis normal """
    dists = np.dot(ommatidia, plane_normal)
    hist, bin_edges = np.histogram(dists, bins=n_bins)
    peaks, _ = find_peaks(hist, height=4, distance=3)

    if debug_plot:
        debug_plot_histogram(dists, n_bins, peaks, hist, axis_id)

    n_lines = len(peaks)
    print(f"    - Found {n_lines} lines for {axis_id}.")
    return n_lines if 5 < n_lines < 30 else 17


## Controls

PLOT_BOTH_EYES = False
N_GRID_LINES_PER_AXIS = {'axis-x': 'auto', 'axis-y': 'auto', 'axis-v': 'auto'}
PARALLELIZE_GRID_LINES = False
PARALLELIZATION_MODE = 'main_axis'  # 'main_axis' or 'average'
REGULARIZE_GRID_ANGLES = False
DEBUG_AUTO_LINE_DETECTION = False

## Data Loading and Initial Projection

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
axes_3d = {id: transform_points_2d_to_3d(sample_path(path, 200), boundary) for id, path in eye_data_2d['axes'].items()
           if path}

## Regularization of eye orientation (based on forward-facing markers in Buchner 1971)

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
axes_final = {id: regularise_points_3d(pts, deviation_curve) for id, pts in axes_3d.items()}

print("Regularization complete.")

## Fitting initial main axes and biological grid lines

print("\nFitting initial main axes and grid lines...")

main_axes_data = {}
biological_grid_lines = {}

for axis_id, points in axes_final.items():
    main_axis_points, normal, mean = generate_small_circle_points(points)
    main_axes_data[axis_id] = {'points': main_axis_points, 'normal': normal, 'mean': mean}

    n_lines = N_GRID_LINES_PER_AXIS.get(axis_id)
    if n_lines == 'auto':
        n_lines = determine_n_lines(ommatidia_final, normal, axis_id, debug_plot=DEBUG_AUTO_LINE_DETECTION)

    biological_grid_lines[axis_id] = fit_quasi_parallel_grid_lines(ommatidia_final, normal, n_lines=n_lines)

print("Initial grid fitting complete.")

## (Optional) Parallelize grid lines

if PARALLELIZE_GRID_LINES:
    print(f"\nParallelizing grid lines using '{PARALLELIZATION_MODE}' method...")

    for axis_id, lines in biological_grid_lines.items():
        if not lines: continue

        target_normal = None
        if PARALLELIZATION_MODE == 'average':
            all_normals = np.array([line['normal'] for line in lines])
            reference_normal = all_normals[0]
            for i in range(1, len(all_normals)):
                if np.dot(all_normals[i], reference_normal) < 0:
                    all_normals[i] *= -1
            avg_normal = np.mean(all_normals, axis=0)
            target_normal = avg_normal / np.linalg.norm(avg_normal)

        else:  # 'main_axis' mode
            target_normal = main_axes_data[axis_id]['normal']

        if target_normal is not None:
            for line in lines:
                line['points'] = create_small_circle_from_normal(target_normal, line['mean'])
                line['normal'] = target_normal

    print("Parallelization complete.")

## (Optional) Final 60-degree regularization of the entire eye

if REGULARIZE_GRID_ANGLES:
    print("\nEnforcing 60/120-degree separation via rigid rotation of the entire grid...")
    C = lattice_centre_final.flatten()
    C_norm = C / np.linalg.norm(C)

    def get_axis_normal(axis_id):
        if PARALLELIZE_GRID_LINES and biological_grid_lines[axis_id]:
            return biological_grid_lines[axis_id][0]['normal']
        return main_axes_data[axis_id]['normal']

    n_x = get_axis_normal('axis-x')
    n_y = get_axis_normal('axis-y')
    n_v = get_axis_normal('axis-v')

    t_x = np.cross(n_x, C_norm)
    t_x /= np.linalg.norm(t_x)
    t_v_ideal = rotate_vector(t_x, C_norm, np.deg2rad(60))
    t_y_ideal = rotate_vector(t_x, C_norm, np.deg2rad(120))

    final_plane_normals = {'axis-x': n_x}
    for axis_id, n_orig, t_ideal in [('axis-y', n_y, t_y_ideal), ('axis-v', n_v, t_v_ideal)]:
        dist_preserved = np.dot(n_orig, C)
        a = dist_preserved
        b_sq = 1 - a ** 2
        b = np.sqrt(b_sq) if b_sq > 0 else 0
        ortho_vec = np.cross(t_ideal, C_norm)
        ortho_vec /= np.linalg.norm(ortho_vec)
        final_plane_normals[axis_id] = a * C_norm + b * ortho_vec

    # rotation: find rotation from old Y/V to new Y/V and average
    rot_axis_y = np.cross(n_y, final_plane_normals['axis-y'])
    rot_angle_y = np.arcsin(np.linalg.norm(rot_axis_y))
    rot_axis_v = np.cross(n_v, final_plane_normals['axis-v'])
    rot_angle_v = np.arcsin(np.linalg.norm(rot_axis_v))

    avg_rot_axis = (rot_axis_y + rot_axis_v) / 2
    avg_rot_angle = (rot_angle_y + rot_angle_v) / 2

    def apply_rigid_rotation(points):
        return np.array([rotate_vector(p, avg_rot_axis, avg_rot_angle) for p in points])

    ommatidia_final = apply_rigid_rotation(ommatidia_final)
    stars_final = apply_rigid_rotation(stars_final)
    lattice_centre_final = apply_rigid_rotation(lattice_centre_final)
    for axis_id in biological_grid_lines:
        for line in biological_grid_lines[axis_id]: line['points'] = apply_rigid_rotation(line['points'])
    for axis_id in main_axes_data: main_axes_data[axis_id]['points'] = apply_rigid_rotation(
        main_axes_data[axis_id]['points'])
    print("Rigid rotation applied.")

## Plotting

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(ommatidia_final[:, 0], ommatidia_final[:, 1], ommatidia_final[:, 2], c='grey', s=10, alpha=0.2,
           label='Ommatidia')

axis_colors = {'axis-x': '#ff60b3', 'axis-y': '#00FF9B', 'axis-v': '#FFC400'}

for axis_id, circles in biological_grid_lines.items():
    lines_to_plot = [c['points'] for c in circles]
    lines_to_plot = clip_lines_to_data_boundary(lines_to_plot, ommatidia_final)
    for line_points in lines_to_plot:
        ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color=axis_colors[axis_id], linewidth=0.7,
                alpha=0.8)

main_axes_to_plot = [d['points'] for d in main_axes_data.values()]
main_axes_to_plot = clip_lines_to_data_boundary(main_axes_to_plot, ommatidia_final)
main_axes_clipped_dict = dict(zip(main_axes_data.keys(), main_axes_to_plot))
for axis_id, points in main_axes_clipped_dict.items():
    if points.size > 0:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=axis_colors[axis_id], linewidth=2.5,
                label=f'Axis {axis_id[-1].upper()}')

ax.scatter(stars_final[:, 0], stars_final[:, 1], stars_final[:, 2], c='red', s=150, marker='*',
           label='Forward direction', depthshade=False)
ax.scatter(lattice_centre_final[:, 0], lattice_centre_final[:, 1], lattice_centre_final[:, 2], c='black', s=20,
           marker='X', label='Lattice origin', depthshade=False)

if PLOT_BOTH_EYES:
    ommatidia_right = ommatidia_final.copy()
    ommatidia_right[:, 0] *= -1
    ax.scatter(ommatidia_right[:, 0], ommatidia_right[:, 1], ommatidia_right[:, 2], c='#555555', s=10, alpha=0.2)

ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='Right (+X)')
ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Up (+Y)')
ax.quiver(0, 0, 0, 0, 0, -0.5, color='b', label='Forward (-Z)')

ax.set_title("Drosophila Eye from Buchner, 1971", fontsize=16)

ax.set_box_aspect([1, 1, 1])

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax.view_init(elev=50, azim=25, roll=110)

ax.legend()
plt.show()