from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from svg.path import parse_path, Line, Close
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import RBFInterpolator

from species_models.drosophila_custom.plots_droso_custom import plot_buchner_3d, plot_parametric_model
from species_models.utils import position_eyes
from species_models.plots import plot_eyes_3d


# Custom parametric model, fitted on measurements by Buchner, 1971 (PhD thesis)
# "Dunkelanregung des stationaeren flugs der fruchtfliege Drosophila"
#
# Plots from Heisenberg and Wolff, 1984 (10.1007/978-3-642-69936-8) were manually redigitised,
# and are used as ground truth to fit the model to.
#
PARAMETRIC_DENSITY_MULTIPLIER = 1.0
PARAMETRIC_GRID_ANGLE_DEG = 60.0
USE_2D_DENSITY = True


@dataclass
class EyeData:
    # small class for storing the parsed svg data
    ommatidia: np.ndarray
    stars: np.ndarray
    lattice_origin: np.ndarray
    axes: Dict[str, np.ndarray]
    hemisphere_center: np.ndarray
    hemisphere_radius: float


def get_path_centroid(path) -> np.ndarray:
    if not path:
        return np.zeros(2)
    points = [(path[0].start.real, path[0].start.imag)]
    for segment in path:
        points.append((segment.end.real, segment.end.imag))
    return np.mean(points, axis=0)


def sample_path(path, pts_per_segment: int = 20) -> np.ndarray:
    """
    Sample points along a path.
    """
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
    """
    Parse SVG file and extract eye data.
    """
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


def inverse_stereographic(points_2d: np.ndarray, center_lon_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse equatorial stereographic projection.
    """
    x, y = points_2d[:, 0], points_2d[:, 1]
    lon0_rad = np.deg2rad(center_lon_deg)

    rho = np.sqrt(x ** 2 + y ** 2)
    rho = np.where(rho == 0, 1e-9, rho)
    c = 2 * np.arctan(rho / 2.0)

    lat_rad = np.arcsin(np.clip((y * np.sin(c)) / rho, -1.0, 1.0))
    lon_rad = lon0_rad + np.arctan2(x * np.sin(c), rho * np.cos(c))

    return np.rad2deg(lon_rad), np.rad2deg(lat_rad)


def spherical_to_cartesian(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """
    Convert spherical to 3D Cartesian coordinates.
    """
    lon_rad, lat_rad = np.deg2rad(lon_deg), np.deg2rad(lat_deg)

    x = np.cos(lat_rad) * np.sin(lon_rad)
    y = np.sin(lat_rad)
    z = -np.cos(lat_rad) * np.cos(lon_rad)

    return np.column_stack([x, y, z])


def unproject(points_2d: np.ndarray, hem_center: np.ndarray, hem_radius: float) -> np.ndarray:
    """
    Unproject 2D points to 3D sphere.
    """

    # Translate, flip and scale
    translated = points_2d - hem_center
    translated *= -1
    scaled = translated * (2.0 / hem_radius)

    # Apply inverse stereographic projection
    lon, lat = inverse_stereographic(scaled, -90.0)

    return spherical_to_cartesian(lon, lat)


def regularize_buchner_data(points_3d: np.ndarray, stars_markers_3d: np.ndarray) -> np.ndarray:
    """
    Warp points based on the fiducial star markers in the plot.
    """

    # Fit polynomial to star positions
    coeffs = np.polyfit(stars_markers_3d[:, 1], stars_markers_3d[:, 0], 2)
    deviation_func = np.poly1d(coeffs)

    # Apply correction
    corrected = points_3d.copy()
    corrected[:, 0] -= deviation_func(corrected[:, 1])

    # Normalise to unit sphere
    norms = np.linalg.norm(corrected, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)

    return corrected / norms


def fit_ellipsoid(points: np.ndarray) -> dict:
    """
    Fit ellipsoid to points.
    """

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

    return ellipsoid


def project_to_ellipsoid(points: np.ndarray, ellipsoid: dict) -> np.ndarray:
    """
    Project points onto ellipsoid surface.
    """
    center, radii = ellipsoid['center'], ellipsoid['radii']
    rays = points - center
    if rays.ndim == 1:
        rays = rays[np.newaxis, :]

    t = 1.0 / np.sqrt(np.sum((rays / radii) ** 2, axis=1, keepdims=True))
    return (center + rays * t).squeeze()


def project_to_stereo(points_3d: np.ndarray, center_point: np.ndarray) -> Tuple:
    """
    Project 3D points onto stereographic plane.
    """

    # Orthonormal basis
    fwd = center_point / np.linalg.norm(center_point)
    tmp = np.array([0, 0, 1]) if abs(fwd[2]) < 0.9 else np.array([1, 0, 0])
    rgt = np.cross(fwd, tmp)
    rgt /= np.linalg.norm(rgt)
    up = np.cross(fwd, rgt)

    # Project each point
    projections = []
    for p in points_3d:
        denom = 1 + np.dot(p, fwd)
        denom = max(denom, 1e-6) # avoid div/0
        x = np.dot(p, rgt) / denom
        y = np.dot(p, up) / denom
        projections.append([x, y])

    return np.array(projections), fwd, rgt, up


def stereo_to_sphere(points_2d: np.ndarray, fwd: np.ndarray, rgt: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Back-project from stereographic plane to sphere.
    """
    x, y = points_2d[:, 0], points_2d[:, 1]
    r2 = x ** 2 + y ** 2
    denom = 1 + r2

    # avoid div/0
    denom = np.where(denom < 1e-9, 1e-9, denom)
    scale = 2.0 / denom
    comp_fwd = (1 - r2) / denom

    return scale[:, None] * (x[:, None] * rgt + y[:, None] * up) + comp_fwd[:, None] * fwd


def generate_hexagonal_template(angle_deg: float = 60.0) -> np.ndarray:
    """
    Generate hexagonal lattice template.
    """
    angle_rad = np.deg2rad(angle_deg)
    basis1 = np.array([1, 0])
    basis2 = np.array([np.cos(angle_rad), np.sin(angle_rad)])

    grid_range = 50
    points = []
    for i in range(-grid_range, grid_range + 1):
        for j in range(-grid_range, grid_range + 1):
            points.append(i * basis1 + j * basis2)

    return np.array(points)


def compute_density_field(data_points: np.ndarray, use_2d: bool = True) -> callable:
    """
    Compute density field from data points.
    """

    tree = cKDTree(data_points)
    distances, _ = tree.query(data_points, k=7)
    local_spacing = distances[:, 1:].mean(axis=1)

    if use_2d:
        rbf = RBFInterpolator(data_points, local_spacing, kernel='thin_plate_spline', smoothing=0.01)
        return lambda x: rbf(x)
    else:
        # Radial density
        radii = np.linalg.norm(data_points, axis=1)
        coeffs = np.polyfit(radii, local_spacing, 3)
        poly = np.poly1d(coeffs)
        return lambda x: poly(np.linalg.norm(x, axis=1))


def optimize_lattice(
        data_points: np.ndarray,
        template: np.ndarray,
        density_mult: float = 1.0,
        use_2d_density: bool = True
    ) -> Tuple:
    """
    Optimize lattice to match data density.
    """
    density_func = compute_density_field(data_points, use_2d=use_2d_density)

    # Get reference spacing from data
    tree = cKDTree(data_points)
    distances, _ = tree.query(data_points, k=7)
    ref_spacing = distances[:, 1:].mean()

    scaled_template = template * ref_spacing * density_mult

    return scaled_template, density_func


def build_eye(svg_path, return_raw_data=False, skip_source_regularisation=False, show_plots=False):

    data = parse_drosophila_svg(svg_path)

    # Unproject Buchner data from 2D to 3D
    directions_3d = unproject(data.ommatidia, data.hemisphere_center, data.hemisphere_radius)
    stars_3d = unproject(data.stars, data.hemisphere_center, data.hemisphere_radius)
    origin_3d = unproject(data.lattice_origin[np.newaxis, :], data.hemisphere_center, data.hemisphere_radius)[0]

    axes_3d = {
        axis_id: unproject(points, data.hemisphere_center, data.hemisphere_radius)
        for axis_id, points in data.axes.items()
    }

    if show_plots:
        plot_buchner_3d(directions_3d, stars_3d, origin_3d, axes_3d, title="Drosophila ommatidia viewing directions (Buchner, 1971)")

    if not skip_source_regularisation:
        directions_3d = regularize_buchner_data(directions_3d, stars_3d)

    if return_raw_data:
        return directions_3d

    # Fit ellipsoid
    ellipsoid = fit_ellipsoid(directions_3d)
    ommatidia_ellipsoid = project_to_ellipsoid(directions_3d, ellipsoid)

    # Project to stereographic plane
    points_sphere = ommatidia_ellipsoid - ellipsoid['center']
    points_sphere /= np.linalg.norm(points_sphere, axis=1, keepdims=True)

    projection_center = origin_3d - ellipsoid['center']
    projection_center /= np.linalg.norm(projection_center)

    ommatidia_2d, fwd, rgt, up = project_to_stereo(points_sphere, projection_center)

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

    if show_plots:
        plot_parametric_model(ommatidia_ellipsoid, lattice_3d, stars_3d, origin_3d,
                              axes_3d, ommatidia_2d, lattice_2d, density_func,
                              nb_ommatidia, USE_2D_DENSITY, PARAMETRIC_GRID_ANGLE_DEG,
                              PARAMETRIC_DENSITY_MULTIPLIER)
    return lattice_3d


##

if __name__ == "__main__":

    PLOT_EYES = True

    svg_file = Path("species_models/drosophila_custom/drosophila_Buchner_1971_redigitized.svg")

    positions = build_eye(svg_file, show_plots=PLOT_EYES)

    # positions are on the unit sphere, so positions == directions anyway, but to be explicit
    directions = positions / np.linalg.norm(positions, axis=1, keepdims=True)

    # Values from "Evolution of Eye Morphology and Rhodopsin Expression in the Drosophila melanogaster Species Subgroup"
    HW_um = 830.0  # Head width (µm)
    FW_um = 400.0  # Frons width (µm) (distance between the most medial points of each eye)
    EL_um = 530.0  # Eye length (µm) (vertical)
    ED_um = 420.0  # Eye depth (µm) (anterior-posterior)

    # Scale and position both eyes
    final_origins, final_directions, eye_ids = position_eyes(
        positions, directions,
        HW_um * 0.001,
        FW_um * 0.001,
        EL_um * 0.001,
        ED_um * 0.001
    )

    output_filename = "species_models/drosophila_custom.npz"
    np.savez_compressed(
        output_filename,
        directions=final_directions,
        origins=final_origins,
        eye_id=eye_ids
    )

    if PLOT_EYES:
        plot_eyes_3d(
            final_origins,
            final_directions,
            eye_ids,
            title='Drosophila eyes\n(custom parametric model fitted on data by Buchner, 1971)',
            show_sphere_projection=True
        )