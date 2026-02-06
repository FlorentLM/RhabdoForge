from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.ElementTree as ET
from svg.path import parse_path, Line, Close
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import RBFInterpolator


GENERATE_PARAMETRIC_MODEL = False
REGULARIZE_WITH_MARKERS = False
PARAMETRIC_DENSITY_MULTIPLIER = 1.0
PARAMETRIC_GRID_ANGLE_DEG = 60.0
USE_2D_DENSITY = True


# ----------------------------------
# SVG parsing
# ----------------------------------

@dataclass
class EyeData:
    ommatidia: np.ndarray
    stars: np.ndarray
    lattice_origin: np.ndarray
    axes: Dict[str, np.ndarray]
    hemisphere_center: np.ndarray
    hemisphere_radius: float


def get_path_centroid(path) -> np.ndarray:
    """Calculate centroid of a path."""
    if not path:
        return np.zeros(2)
    points = [(path[0].start.real, path[0].start.imag)]
    for segment in path:
        points.append((segment.end.real, segment.end.imag))
    return np.mean(points, axis=0)


def sample_path(path, pts_per_segment: int = 20) -> np.ndarray:
    """Sample points along a path."""
    if not path:
        return np.array([])

    points = [(path[0].start.real, path[0].start.imag)]
    for segment in path:
        if isinstance(segment, (Line, Close)):
            points.append((segment.end.real, segment.end.imag))
        else:
            for i in range(pts_per_segment):
                p = segment.point(i / (pts_per_segment - 1))
                points.append((p.real, p.imag))
    return np.array(points)


def parse_drosophila_svg(svg_file: Path) -> EyeData:
    """Parse SVG file and extract eye data."""
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # Extract ommatidia
    ommatidia = []
    for circle in root.findall('svg:circle', ns):
        if circle.get('id') != 'hemisphere':
            ommatidia.append([float(circle.get('cx')), float(circle.get('cy'))])

    # Extract hemisphere
    hemisphere = root.find(".//*[@id='hemisphere']", ns)
    if hemisphere is None:
        raise ValueError("Hemisphere circle not found in SVG")
    hem_center = np.array([float(hemisphere.get('cx')), float(hemisphere.get('cy'))])
    hem_radius = float(hemisphere.get('r'))

    # Extract stars
    stars = []
    for i in range(1, 6):
        star_elem = root.find(f".//*[@id='star-{i}']", ns)
        if star_elem is not None:
            path = parse_path(star_elem.get('d'))
            stars.append(get_path_centroid(path))

    # Extract lattice origin
    lattice_elem = root.find(".//*[@id='lattice-origin']", ns)
    if lattice_elem is not None:
        path = parse_path(lattice_elem.get('d'))
        lattice_origin = get_path_centroid(path)
    else:
        lattice_origin = np.zeros(2)

    # Extract axes
    axes = {}
    for axis_id in ['axis-x', 'axis-y', 'axis-v']:
        axis_elem = root.find(f".//*[@id='{axis_id}']", ns)
        if axis_elem is not None:
            path = parse_path(axis_elem.get('d'))
            axes[axis_id] = sample_path(path, 50)

    print(f"Parsed: {len(ommatidia)} ommatidia, {len(stars)} stars, {len(axes)} axes")

    return EyeData(
        ommatidia=np.array(ommatidia),
        stars=np.array(stars),
        lattice_origin=lattice_origin,
        axes=axes,
        hemisphere_center=hem_center,
        hemisphere_radius=hem_radius
    )

# ----------------------------------
# Projection / unprojection
# ----------------------------------

def inverse_stereographic(points_2d: np.ndarray, center_lon_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse equatorial stereographic projection."""
    x, y = points_2d[:, 0], points_2d[:, 1]
    lon0_rad = np.deg2rad(center_lon_deg)

    rho = np.sqrt(x ** 2 + y ** 2)
    rho = np.where(rho == 0, 1e-9, rho)
    c = 2 * np.arctan(rho / 2.0)

    lat_rad = np.arcsin(np.clip((y * np.sin(c)) / rho, -1.0, 1.0))
    lon_rad = lon0_rad + np.arctan2(x * np.sin(c), rho * np.cos(c))

    return np.rad2deg(lon_rad), np.rad2deg(lat_rad)


def spherical_to_cartesian(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Convert spherical to 3D Cartesian coordinates."""
    lon_rad, lat_rad = np.deg2rad(lon_deg), np.deg2rad(lat_deg)

    x = np.cos(lat_rad) * np.sin(lon_rad)
    y = np.sin(lat_rad)
    z = -np.cos(lat_rad) * np.cos(lon_rad)

    return np.column_stack([x, y, z])


def unproject(points_2d: np.ndarray, hem_center: np.ndarray, hem_radius: float) -> np.ndarray:
    """Unproject 2D points to 3D sphere."""

    # Translate and flip
    translated = points_2d - hem_center
    translated *= -1

    # Scale to [-2, 2] range
    scaled = translated * (2.0 / hem_radius)

    # Apply inverse stereographic projection
    lon, lat = inverse_stereographic(scaled, -90.0)

    return spherical_to_cartesian(lon, lat)


def regularize_with_stars(points_3d: np.ndarray, stars_3d: np.ndarray) -> np.ndarray:
    """Warp points based on star markers."""
    # Fit polynomial to star positions
    coeffs = np.polyfit(stars_3d[:, 1], stars_3d[:, 0], 2)
    deviation_func = np.poly1d(coeffs)

    # Apply correction
    corrected = points_3d.copy()
    corrected[:, 0] -= deviation_func(corrected[:, 1])

    # Normalize to unit sphere
    norms = np.linalg.norm(corrected, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)

    return corrected / norms


# ----------------------------------
# Parametric model
# ----------------------------------

def fit_ellipsoid(points: np.ndarray) -> dict:
    """Fit ellipsoid to points."""

    def loss(p, pts):
        center, radii = p[:3], p[3:6]
        normalized = (pts - center) / radii
        return np.sum((np.sum(normalized ** 2, axis=1) - 1) ** 2)

    # Initial guess
    p0 = np.concatenate([points.mean(axis=0), points.std(axis=0)])

    # Optimize
    bounds = [(-2, 2)] * 3 + [(0.1, 3)] * 3
    result = minimize(loss, p0, args=(points,), method='L-BFGS-B', bounds=bounds)

    ellipsoid = {'center': result.x[:3], 'radii': result.x[3:6]}
    print(f"Ellipsoid: center={ellipsoid['center'].round(3)}, radii={ellipsoid['radii'].round(3)}")

    return ellipsoid


def project_to_ellipsoid(points: np.ndarray, ellipsoid: dict) -> np.ndarray:
    """Project points onto ellipsoid surface."""
    center, radii = ellipsoid['center'], ellipsoid['radii']
    rays = points - center
    if rays.ndim == 1:
        rays = rays[np.newaxis, :]

    t = 1.0 / np.sqrt(np.sum((rays / radii) ** 2, axis=1, keepdims=True))
    return (center + rays * t).squeeze()


def project_to_stereo_plane(points_3d: np.ndarray, center_point: np.ndarray) -> Tuple:
    """Stereographic projection to 2D plane."""

    # Get projection axes
    forward = center_point / np.linalg.norm(center_point)
    right = np.cross([0, 1, 0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    # Project
    x_local = points_3d @ right
    y_local = points_3d @ up
    z_local = points_3d @ forward

    scaling = 1 / (1 + z_local)
    points_2d = np.column_stack([x_local * scaling, y_local * scaling])

    return points_2d, forward, right, up


def stereo_to_sphere(points_2d: np.ndarray, forward: np.ndarray, right: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Map from stereographic plane back to sphere."""
    x2d, y2d = points_2d[:, 0], points_2d[:, 1]
    rho_sq = x2d ** 2 + y2d ** 2

    x3d = 2 * x2d / (1 + rho_sq)
    y3d = 2 * y2d / (1 + rho_sq)
    z3d = (1 - rho_sq) / (1 + rho_sq)

    return x3d[:, None] * right + y3d[:, None] * up + z3d[:, None] * forward


def generate_hexagonal_template(grid_range: int = 50, angle_deg: float = 60.0) -> np.ndarray:
    """Generate hexagonal grid template."""
    angle_rad = np.deg2rad(angle_deg)
    v1 = np.array([1.0, 0])
    v2 = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    points = []
    for i in range(-grid_range, grid_range + 1):
        for j in range(-grid_range, grid_range + 1):
            points.append(i * v1 + j * v2)

    return np.array(points)


def extract_axis_directions(axes_3d: dict) -> dict:
    """Extract principal directions from the axes data."""

    directions = {}
    for axis_id, points in axes_3d.items():
        if len(points) < 2:
            continue

        # Center the points
        centered = points - points.mean(axis=0)

        # SVD of centered points gives principal directions
        _, _, vh = np.linalg.svd(centered)
        principal_dir = vh[0]  # First row of V^T is the first principal component

        # Compare projections to determine direction
        start_proj = points[0] @ principal_dir
        end_proj = points[-1] @ principal_dir
        if end_proj < start_proj:
            principal_dir = -principal_dir

        directions[axis_id] = principal_dir

    return directions


def fit_2d_density_field(points_2d: np.ndarray, method='rbf') -> Tuple[Callable, float]:
    """
    Returns a function that maps 2D positions to local spacing multipliers.
    """

    # Calculate local density using k-nearest neighbors
    tree = cKDTree(points_2d)
    distances, _ = tree.query(points_2d, k=7)
    local_spacings = distances[:, 1:].mean(axis=1)

    # Normalize by mean spacing
    mean_spacing = local_spacings.mean()
    relative_spacings = local_spacings / mean_spacing

    if method == 'rbf':
        interpolator = RBFInterpolator(
            points_2d,
            relative_spacings,
            kernel='thin_plate_spline',
            smoothing=0.2
        )

        def density_func(pos):
            if pos.ndim == 1:
                pos = pos[np.newaxis, :]
            return interpolator(pos).flatten()

    elif method == 'angular_radial':
        # Fit a model with radial and angular components

        # Convert to polar coordinates
        x, y = points_2d[:, 0], points_2d[:, 1]
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)

        # Fit radial profile
        radial_coeffs = np.polyfit(r, relative_spacings, 3)
        radial_poly = np.poly1d(radial_coeffs)

        # Fit angular modulation (Fourier series)
        n_harmonics = 4
        angular_coeffs = []
        residuals = relative_spacings - radial_poly(r)

        for n in range(1, n_harmonics + 1):
            a_n = 2.0 * np.mean(residuals * np.cos(n * theta))
            b_n = 2.0 * np.mean(residuals * np.sin(n * theta))
            angular_coeffs.append((a_n, b_n))

        def density_func(pos):
            """Evaluate density with radial and angular components."""
            if pos.ndim == 1:
                pos = pos[np.newaxis, :]

            x, y = pos[:, 0], pos[:, 1]
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan2(y, x)

            # Radial component
            result = radial_poly(r)

            # Angular modulation
            for n, (a_n, b_n) in enumerate(angular_coeffs, 1):
                result += a_n * np.cos(n * theta) + b_n * np.sin(n * theta)

            return result

    else:
        # Fallback to simple radial
        r = np.sqrt(points_2d[:, 0] ** 2 + points_2d[:, 1] ** 2)
        coeffs = np.polyfit(r, relative_spacings, 3)
        poly = np.poly1d(coeffs)

        def density_func(pos):
            if pos.ndim == 1:
                pos = pos[np.newaxis, :]
            r = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            return poly(r)

    return density_func, mean_spacing


def optimize_lattice(ommatidia_2d: np.ndarray,
                     template: np.ndarray,
                     density_mult: float = 1.0,
                     use_2d_density: bool = True) -> Tuple[np.ndarray, Callable]:

    # Fit the density field
    if use_2d_density:
        print("Fitting 2D density field (RBF)...")
        density_func, mean_spacing = fit_2d_density_field(ommatidia_2d, method='rbf')
    else:
        print("Using angular-radial density model...")
        density_func, mean_spacing = fit_2d_density_field(ommatidia_2d, method='angular_radial')

    def transform_with_density(origin, rotation, base_spacing, template, density_func, density_mult):
        """Transform template using the 2D density field."""

        correction = np.sqrt(2.0)
        scaled = template * base_spacing * density_mult * correction

        # Rotate
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        rotated = scaled @ rot_matrix.T

        # Translate
        transformed = origin + rotated

        # Apply spatially varying density
        density_values = density_func(transformed)
        final = origin + (transformed - origin) * density_values[:, None]

        return final

    def loss(params):
        origin = params[:2]
        rotation = params[2]
        base_spacing = params[3]

        generated = transform_with_density(
            origin, rotation, base_spacing,
            template, density_func, density_mult
        )

        # Find nearest neighbors
        tree = cKDTree(generated)
        dist, _ = tree.query(ommatidia_2d)
        return np.mean(dist ** 2)

    # Initial parameters
    initial = [0.0, 0.0, 0.0, mean_spacing]

    # Bounds
    bounds = [
        (-0.1, 0.1), (-0.1, 0.1),  # origin
        (-np.pi / 6, np.pi / 6),  # rotation
        (mean_spacing * 0.7, mean_spacing * 1.3),  # spacing
    ]

    print("Optimizing lattice...")
    result = minimize(loss, initial, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-10})

    print(f"Optimization complete. Loss: {result.fun:.6f}")
    print(f"  Origin: ({result.x[0]:.4f}, {result.x[1]:.4f})")
    print(f"  Rotation: {np.rad2deg(result.x[2]):.2f}°")
    print(f"  Base spacing: {result.x[3]:.4f}")

    # Generate final lattice
    final_lattice = transform_with_density(
        result.x[:2], result.x[2], result.x[3],
        template, density_func, density_mult
    )

    return final_lattice, density_func


##


# Load and parse SVG
svg_file = Path('biological_data/drosophila_Buchner1971/drosophila_Buchner_1971_redigitized.svg')
eye_data = parse_drosophila_svg(svg_file)

# Unproject to 3D
ommatidia_3d = unproject(eye_data.ommatidia, eye_data.hemisphere_center, eye_data.hemisphere_radius)
stars_3d = unproject(eye_data.stars, eye_data.hemisphere_center, eye_data.hemisphere_radius)
origin_3d = unproject(eye_data.lattice_origin[np.newaxis, :],
                      eye_data.hemisphere_center, eye_data.hemisphere_radius)[0]

axes_3d = {}
for axis_id, points in eye_data.axes.items():
    axes_3d[axis_id] = unproject(points, eye_data.hemisphere_center, eye_data.hemisphere_radius)

if REGULARIZE_WITH_MARKERS:

    ommatidia_3d = regularize_with_stars(ommatidia_3d, stars_3d)
    stars_3d = regularize_with_stars(stars_3d, stars_3d)
    origin_3d = regularize_with_stars(origin_3d[np.newaxis, :], stars_3d).flatten()

    for axis_id in axes_3d:
        axes_3d[axis_id] = regularize_with_stars(axes_3d[axis_id], stars_3d)

if GENERATE_PARAMETRIC_MODEL:

    ellipsoid = fit_ellipsoid(ommatidia_3d)
    ommatidia_ellipsoid = project_to_ellipsoid(ommatidia_3d, ellipsoid)

    # Project to stereographic plane
    points_sphere = (ommatidia_ellipsoid - ellipsoid['center'])
    points_sphere /= np.linalg.norm(points_sphere, axis=1, keepdims=True)

    projection_center = origin_3d - ellipsoid['center']
    projection_center /= np.linalg.norm(projection_center)

    ommatidia_2d, fwd, rgt, up = project_to_stereo_plane(points_sphere, projection_center)

    template = generate_hexagonal_template(angle_deg=PARAMETRIC_GRID_ANGLE_DEG)
    lattice_2d, density_func = optimize_lattice(
        ommatidia_2d, template, PARAMETRIC_DENSITY_MULTIPLIER, use_2d_density=USE_2D_DENSITY
    )

    # Prune to data extent
    hull = Delaunay(ommatidia_2d)
    keep = hull.find_simplex(lattice_2d) >= 0
    lattice_2d = lattice_2d[keep]

    nb_ommatidia = lattice_2d.shape[0]
    print(f"Generated {nb_ommatidia} ommatidia")

    # Back-project to 3D
    lattice_sphere = stereo_to_sphere(lattice_2d, fwd, rgt, up)
    lattice_3d = project_to_ellipsoid(lattice_sphere + ellipsoid['center'], ellipsoid)

    # Compute viewing directions from origin (head center), not from ellipsoid center
    # This ensures directions have proper forward (-Z) components
    lattice_directions = lattice_3d / np.linalg.norm(lattice_3d, axis=1, keepdims=True)

    fig = plt.figure(figsize=(16, 12))

    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(*ommatidia_ellipsoid.T, c='grey', s=20, alpha=0.3, label='Real ommatidia')
    ax1.scatter(*lattice_3d.T, c='green', s=15, alpha=0.5, label='Generated ommatidia')
    ax1.scatter(*stars_3d.T, c='red', s=150, marker='*', label='Forward', depthshade=False)
    ax1.scatter(*origin_3d, c='black', s=50, marker='X', label='Lattice origin', depthshade=False)

    colors = {'axis-x': '#ff60b3', 'axis-y': '#00FF9B', 'axis-v': '#FFC400'}
    for axis_id, points in axes_3d.items():
        ax1.plot(*points.T, color=colors[axis_id], linewidth=2.5,
                 label=f'Axis {axis_id[-1].upper()}')

    ax1.set_title("Parametric Eye model", fontsize=14)
    ax1.legend(loc='upper right', fontsize=8)

    all_points = np.vstack([ommatidia_3d, stars_3d])
    min_coords, max_coords = all_points.min(axis=0), all_points.max(axis=0)
    center, max_range = (max_coords + min_coords) / 2, (max_coords - min_coords).max() / 2

    ax1.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='Right (+X)')
    ax1.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Up (+Y)')
    ax1.quiver(0, 0, 0, 0, 0, -0.5, color='b', label='Forward (-Z)')

    ax1.set_xlim(center[0] - max_range, center[0] + max_range)
    ax1.set_ylim(center[1] - max_range, center[1] + max_range)
    ax1.set_zlim(center[2] - max_range, center[2] + max_range)
    ax1.set_box_aspect([1, 1, 1])
    ax1.view_init(elev=30, azim=45)

    # Density field visualization
    ax2 = fig.add_subplot(222)

    x_range = np.linspace(ommatidia_2d[:, 0].min() - 0.1,
                          ommatidia_2d[:, 0].max() + 0.1, 50)
    y_range = np.linspace(ommatidia_2d[:, 1].min() - 0.1,
                          ommatidia_2d[:, 1].max() + 0.1, 50)
    X, Y = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    Z = density_func(grid_points).reshape(X.shape)

    # Plot density field
    im = ax2.contourf(X, Y, Z, levels=20, cmap='viridis_r', alpha=0.7)
    ax2.scatter(*ommatidia_2d.T, c='red', s=2, alpha=0.3, label='Real data')
    ax2.set_title('Density Field', fontsize=14)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (stereo plane)')
    ax2.set_ylabel('Y (stereo plane)')
    cbar = plt.colorbar(im, ax=ax2, label='Relative spacing')
    cbar.ax.tick_params(labelsize=8)

    # 2D lattice comparison
    ax3 = fig.add_subplot(223)
    ax3.scatter(*ommatidia_2d.T, c='grey', s=8, alpha=0.5, label='Real ommatidia')
    ax3.scatter(*lattice_2d.T, c='green', s=5, alpha=0.6, label='Generated lattice')
    ax3.set_title('Stereographic plane', fontsize=14)
    ax3.set_aspect('equal')
    ax3.set_xlabel('X (stereo plane)')
    ax3.set_ylabel('Y (stereo plane)')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Density profile plot
    ax4 = fig.add_subplot(224)

    # radial distances and local densities
    radii = np.sqrt(ommatidia_2d[:, 0] ** 2 + ommatidia_2d[:, 1] ** 2)
    tree = cKDTree(ommatidia_2d)
    distances, _ = tree.query(ommatidia_2d, k=7)
    local_spacings = distances[:, 1:].mean(axis=1)

    # Plot actual vs fitted density
    ax4.scatter(radii, local_spacings, c='blue', s=10, alpha=0.3, label='Actual spacing')

    # Plot fitted profile at various angles
    test_radii = np.linspace(0, radii.max(), 100)
    for angle in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]:
        test_points = np.column_stack([
            test_radii * np.cos(angle),
            test_radii * np.sin(angle)
        ])
        fitted_density = density_func(test_points) * local_spacings.mean()
        ax4.plot(test_radii, fitted_density, alpha=0.7,
                 label=f'Fitted at θ={np.rad2deg(angle):.0f}°')

    ax4.set_xlabel('Distance from origin')
    ax4.set_ylabel('Local spacing')
    ax4.set_title('Spacing vs Distance', fontsize=14)
    ax4.legend(loc='best', fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Drosophila Eye - Parametric model, {nb_ommatidia} ommatidia\n" +
                 f"Method: {'2D RBF' if USE_2D_DENSITY else 'Angular-Radial'}, " +
                 f"Angle: {PARAMETRIC_GRID_ANGLE_DEG}°, Density mult: {PARAMETRIC_DENSITY_MULTIPLIER}",
                 fontsize=16, y=1.02)
    plt.tight_layout()

else:
    lattice_3d = ommatidia_3d

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*ommatidia_3d.T, c='grey', s=10, alpha=0.5, label='Ommatidia')
    ax.scatter(*stars_3d.T, c='red', s=150, marker='*', label='Forward', depthshade=False)
    ax.scatter(*origin_3d, c='black', s=50, marker='X', label='Origin', depthshade=False)

    # Plot axes
    colors = {'axis-x': '#ff60b3', 'axis-y': '#00FF9B', 'axis-v': '#FFC400'}
    for axis_id, points in axes_3d.items():
        ax.plot(*points.T, color=colors[axis_id], linewidth=2.5,
                label=f'Axis {axis_id[-1].upper()}')

    ax.set_title("Drosophila Eye - Buchner 1971", fontsize=16)

    # Set up axes properly
    all_points = np.vstack([ommatidia_3d, stars_3d])
    min_coords, max_coords = all_points.min(axis=0), all_points.max(axis=0)
    center, max_range = (max_coords + min_coords) / 2, (max_coords - min_coords).max() / 2

    ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='Right (+X)')
    ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Up (+Y)')
    ax.quiver(0, 0, 0, 0, 0, -0.5, color='b', label='Forward (-Z)')

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=45)
    ax.legend()

plt.show()


# ---------
# Generate second eye and save
# ---------

# Eye dimensions in mm (converted from micrometers)
UM_TO_MM = 0.001
eye_dims_mm = np.array([215.0, 510.0, 430.0]) * UM_TO_MM    # Eye dims in mm
inter_eye_dist_mm = 390.0 * UM_TO_MM                         # Distance between eyes in mm

# Determine the scaling factor
scale_factors = eye_dims_mm / (lattice_3d.max(axis=0) - lattice_3d.min(axis=0))

# Apply scaling to the origins (from the center of the eye)
eye_center = lattice_3d.mean(axis=0)
left_eye_origins_scaled = (lattice_3d - eye_center) * scale_factors + eye_center

# The medial edge (rightmost point) of the left eye should be at x = -inter_eye_distance / 2
medial_edge_target_x = -inter_eye_dist_mm / 2.0
current_medial_edge_x = left_eye_origins_scaled[:, 0].max()

# Calculate the necessary translation and apply it
translation = np.array([medial_edge_target_x - current_medial_edge_x, 0, 0])
left_eye_origins = left_eye_origins_scaled + translation

# Mirror the final positioned left eye to create the right eye
right_eye_origins = left_eye_origins.copy()
right_eye_origins[:, 0] *= -1

left_eye_dirs = lattice_3d / np.linalg.norm(lattice_3d, axis=1, keepdims=True)
right_eye_dirs = left_eye_dirs.copy()
right_eye_dirs[:, 0] *= -1
final_dirs = np.concatenate((left_eye_dirs, right_eye_dirs))
final_origins = np.concatenate((left_eye_origins, right_eye_origins))

nb_ommatidia = lattice_directions.shape[0]
eye_ids = np.concatenate([
    np.zeros(nb_ommatidia, dtype=int), # Left eye ID = 0
    np.ones(nb_ommatidia, dtype=int)   # Right eye ID = 1
])

# output_filename = "drosophila_Buchner.npz"
# np.savez_compressed(
#     output_filename,
#     directions=final_dirs,
#     origins=final_origins,
#     eye_id=eye_ids
# )
