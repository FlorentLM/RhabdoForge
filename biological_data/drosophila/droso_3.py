from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
from pathlib import Path
from svg.path import parse_path, Line, Close
from sklearn.decomposition import PCA


def get_path_centroid(path):
    """ Calculates the geometric center of a parsed SVG path """
    points = [path[0].start]
    for segment in path:
        points.append(segment.end)
    return np.mean(np.array([(p.real, p.imag) for p in points]), axis=0)


def sample_path(path, pts_per_segment=20):
    """
    Generates points along an SVG path by sampling its constituent segments
    """
    points = []

    for segment in path:
        # it is only needed to sample curves (Arc, CubicBezier, QuadraticBezier), for straight lines the endpoints are enough
        if isinstance(segment, (Line, Close)):
            points.append((segment.end.real, segment.end.imag))
        else:
            for i in range(pts_per_segment):
                pos = i / (pts_per_segment - 1)
                p = segment.point(pos)
                points.append((p.real, p.imag))
    if path:
        points.insert(0, (path[0].start.real, path[0].start.imag))
    return np.array(points)


def parse_drosophila_svg(svg_file: Union[Path, str]) -> dict:
    """ Parses the digitized SVG file """

    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    data = {'ommatidia': [], 'stars': [], 'axes': {}, 'hemisphere': {}, 'grid_lines': {}}

    for circle in root.findall('svg:circle', ns):
        if circle.get('id') != 'hemisphere':
            data['ommatidia'].append((float(circle.get('cx')), float(circle.get('cy'))))

    hemisphere_circle = root.find(".//*[@id='hemisphere']", ns)
    if hemisphere_circle is not None:
        data['hemisphere'] = {'cx': float(hemisphere_circle.get('cx')), 'cy': float(hemisphere_circle.get('cy')),
                              'r': float(hemisphere_circle.get('r'))}
    else:
        raise ValueError("Hemisphere circle with id='hemisphere' not found in SVG.")

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

    for grid_id in ['grid-x', 'grid-y', 'grid-v']:
        data['grid_lines'][grid_id] = []
        for path_element in root.findall(f".//svg:g[@id='{grid_id}']/svg:path", ns):
            path_d = path_element.get('d')
            if path_d:
                data['grid_lines'][grid_id].append(parse_path(path_d))

    print(
        f"Parsed SVG: Found {len(data['ommatidia'])} ommatidia, {len(data['stars'])} stars, "
        f"{len(data['axes'])} main axes, and {sum(len(v) for v in data['grid_lines'].values())} grid lines.")
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


def small_circle_points(points_on_sphere, nb_points=200):
    if len(points_on_sphere) < 3:
        return np.array([]), None, None

    pca = PCA(n_components=3)
    pca.fit(points_on_sphere)
    normal = pca.components_[2]
    point_on_plane = pca.mean_

    # Ensure normal points towards the origin for consistency
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

    t = np.linspace(0, 2 * np.pi, nb_points)
    circle_points = circle_center + circle_radius * (np.outer(np.cos(t), v1) + np.outer(np.sin(t), v2))
    return circle_points, normal, point_on_plane


def small_circle_from_normal(normal, point_on_plane, nb_points=200):
    dist_from_origin = np.dot(normal, point_on_plane)

    if abs(dist_from_origin) >= 1.0:
        return np.array([])

    circle_center = dist_from_origin * normal
    circle_radius = np.sqrt(1 - dist_from_origin ** 2)
    v1 = np.cross(normal, [0, 0, 1])

    if np.linalg.norm(v1) < 1e-6:
        v1 = np.cross(normal, [0, 1, 0])

    v1 /= np.linalg.norm(v1)
    v2 = np.cross(normal, v1)
    t = np.linspace(0, 2 * np.pi, nb_points)
    return circle_center + circle_radius * (np.outer(np.cos(t), v1) + np.outer(np.sin(t), v2))


def rotate_vector(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return (vector * np.cos(angle) +
            np.cross(axis, vector) * np.sin(angle) +
            axis * np.dot(axis, vector) * (1 - np.cos(angle)))


def clip_lines(lines, data_points, padding_factor=1.05):
    if not lines:
        return []

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

## Controls

REGULARIZE_WITH_MARKERS = False
REGULARIZE_COPLANAR_GRID = True
REGULARIZE_GRID_ANGLES = False

PLOT_MANUAL_LINES = False
PLOT_GRID_PLANES_NORMALS = False
CLIP_GRID = True
PLOT_BOTH_EYES = False


## Initial projection

svg_file = Path('biological_data/drosophila/drosophila_Buchner_1971_redigitized.svg')
eye_data_2d = parse_drosophila_svg(svg_file)


def unproject(points_2d, hemisphere_boundary):
    translated = points_2d - np.array([hemisphere_boundary['cx'], hemisphere_boundary['cy']])
    translated[:, 0] *= -1
    translated[:, 1] *= -1
    scaled = translated * (2.0 / hemisphere_boundary['r'])
    lon, lat = inverse_equatorial_stereographic(scaled[:, 0], scaled[:, 1], -90.0)
    return spherical_to_opengl(lon, lat).T


hemisphere_boundary = eye_data_2d['hemisphere']

ommatidia_3d = unproject(np.array(eye_data_2d['ommatidia']), hemisphere_boundary)
stars_3d = unproject(np.array(eye_data_2d['stars']), hemisphere_boundary)
lattice_centre_3d = unproject(np.array([eye_data_2d['lattice-origin']]), hemisphere_boundary)
axes_3d = {ax_id: unproject(sample_path(path, 100), hemisphere_boundary)
           for ax_id, path in eye_data_2d['axes'].items() if path}

print("\nSampling the grid lines (this takes a while...)")
grid_lines_3d = {}
for grid_id, paths_2d in eye_data_2d['grid_lines'].items():
    grid_lines_3d[grid_id] = []
    for p in paths_2d:
        sampled_points = sample_path(p, 100)
        projected_points = unproject(sampled_points, hemisphere_boundary)
        grid_lines_3d[grid_id].append(projected_points)

## Regularization of eye orientation

if REGULARIZE_WITH_MARKERS:
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
    grid_lines_final = {grid_id: [regularise_points_3d(pts, deviation_curve) for pts in points_list]
                        for grid_id, points_list in grid_lines_3d.items()}
    print("Regularization complete.")

else:
    ommatidia_final = ommatidia_3d
    stars_final = stars_3d
    lattice_centre_final = lattice_centre_3d
    axes_final = axes_3d
    grid_lines_final = grid_lines_3d

## Fitting main axes and biological grid lines

print("\nFitting circles to main axes and manually traced grid lines...")

main_axes_data = {}
for axis_id, points in axes_final.items():
    main_axis_points, normal, mean = small_circle_points(points)
    main_axes_data[axis_id] = {'points': main_axis_points, 'normal': normal, 'mean': mean, 'source_points': points}

biological_grid_lines = {}
for grid_id, lines_3d in grid_lines_final.items():
    axis_id = grid_id.replace('grid-', 'axis-')
    biological_grid_lines[axis_id] = []
    for line_points_3d in lines_3d:
        circle_points, normal, mean = small_circle_points(line_points_3d)
        if circle_points.size > 0:
            biological_grid_lines[axis_id].append(
                {'points': circle_points, 'normal': normal, 'mean': mean, 'original_data': line_points_3d, 'source_points': line_points_3d})

print("Grid fitting complete.")

## (Optional) Regularize grid lines to have co-planar normals

if REGULARIZE_COPLANAR_GRID:
    print("\nRegularizing grid lines to have co-planar normals based on main axes...")
    for axis_id, lines in biological_grid_lines.items():
        if not lines or axis_id not in main_axes_data:
            continue

        main_axis_normal = main_axes_data[axis_id]['normal']
        common_axis = np.cross(main_axis_normal, lattice_centre_final.flatten())
        common_axis /= np.linalg.norm(common_axis)

        for line in lines:
            n_old = line['normal']
            if np.dot(n_old, main_axis_normal) < 0:
                n_old *= -1

            n_new = n_old - np.dot(n_old, common_axis) * common_axis
            n_new /= np.linalg.norm(n_new)

            line['normal'] = n_new
            line['points'] = small_circle_from_normal(n_new, line['mean'])
    print("Co-planar regularization complete.")


## (Optional) Final 60-degree regularization

if REGULARIZE_GRID_ANGLES:
    print("\nEnforcing 60/120-degree separation via rigid rotation of the entire grid...")
    C = lattice_centre_final.flatten()
    C_norm = C / np.linalg.norm(C)

    n_x = main_axes_data['axis-x']['normal']
    n_y = main_axes_data['axis-y']['normal']
    n_v = main_axes_data['axis-v']['normal']

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
    if PLOT_MANUAL_LINES:
        lines_to_plot = [c['source_points'] for c in circles]
    else:
        lines_to_plot = [c['points'] for c in circles]
    if CLIP_GRID:
        lines_to_plot = clip_lines(lines_to_plot, ommatidia_final)
    for line_points in lines_to_plot:
        if line_points.size > 0:
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color=axis_colors[axis_id], linewidth=0.7,
                    alpha=0.8)
if PLOT_MANUAL_LINES:
    main_axes_to_plot = [d['source_points'] for d in main_axes_data.values()]
else:
    main_axes_to_plot = [d['points'] for d in main_axes_data.values()]
if CLIP_GRID:
    main_axes_to_plot = clip_lines(main_axes_to_plot, ommatidia_final)
main_axes_clipped_dict = dict(zip(main_axes_data.keys(), main_axes_to_plot))
for axis_id, points in main_axes_clipped_dict.items():
    if points.size > 0:
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=axis_colors[axis_id], linewidth=2.0,
                label=f'Axis {axis_id[-1].upper()}')

ax.scatter(stars_final[:, 0], stars_final[:, 1], stars_final[:, 2], c='red', s=150, marker='*',
           label='Forward direction', depthshade=False)
ax.scatter(lattice_centre_final[:, 0], lattice_centre_final[:, 1], lattice_centre_final[:, 2], c='black', s=20,
           marker='X', label='Lattice origin', depthshade=False)

if PLOT_BOTH_EYES:
    ommatidia_right = ommatidia_final.copy()
    ommatidia_right[:, 0] *= -1
    ax.scatter(ommatidia_right[:, 0], ommatidia_right[:, 1], ommatidia_right[:, 2], c='#555555', s=10, alpha=0.2)

if PLOT_GRID_PLANES_NORMALS:
    print("\nPlotting normal vectors for verification...")
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 0.5 * np.outer(np.cos(u), np.sin(v))
    y = 0.5 * np.outer(np.sin(u), np.sin(v))
    z = 0.5 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='grey', alpha=0.1, zorder=-100)

    for axis_id, circles in biological_grid_lines.items():
        normals = np.array([c['normal'] for c in circles])
        ax.quiver(
            np.zeros(len(normals)), np.zeros(len(normals)), np.zeros(len(normals)),
            normals[:, 0], normals[:, 1], normals[:, 2],
            color=axis_colors[axis_id],
            length=0.8,
            label=f'{axis_id[-1].upper()} Normals' if axis_id else ''
        )

ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='Right (+X)')
ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Up (+Y)')
ax.quiver(0, 0, 0, 0, 0, -0.5, color='b', label='Forward (-Z)')

ax.set_title("Drosophila Eye from Buchner, 1971", fontsize=16)

ax.set_box_aspect([1, 1, 1])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.view_init(elev=50, azim=25, roll=110)

handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys())

plt.show()