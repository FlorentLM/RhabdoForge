from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
from svg.path import parse_path, Line, Close
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import RBFInterpolator

from species_models.drosophila_custom.plots_droso_custom import plot_density_3d, plot_density_2d, plot_buchner_3d
from species_models.plots import plot_eyes_3d


# Custom parametric model, fitted on measurements by Buchner, 1971 (PhD thesis)
# "Dunkelanregung des stationaeren flugs der fruchtfliege Drosophila"
#
# Plots from Heisenberg and Wolff, 1984 (10.1007/978-3-642-69936-8) were manually redigitised,
# and are used as ground truth to fit the model to.

DENSITY_SCALE = 1.0         # Relative density (1.0 = match data, 2.0 = twice as dense)
LATTICE_ANGLE_DEG = 60.0    # Angle between lattice basis vectors (60° = hexagonal)
REGULARIZATION = True       # Use Buchner's 'forward' markers to correct the raw data points
SHOW_DEBUG_PLOTS = True     # Show intermediate fitting plots


@dataclass
class BuchnerSVGContent:
    ommatidia: np.ndarray
    stars: np.ndarray
    lattice_origin: np.ndarray
    axes: Dict[str, np.ndarray]
    hemisphere_center: np.ndarray
    hemisphere_radius: float


def get_path_centroid(path) -> np.ndarray:
    """Calculate centroid of an SVG path"""
    if not path:
        return np.zeros(2)
    points = [(path[0].start.real, path[0].start.imag)]
    for segment in path:
        points.append((segment.end.real, segment.end.imag))
    return np.mean(points, axis=0)


def sample_path(path, pts_per_segment: int = 20) -> np.ndarray:
    """Sample points along an SVG path"""
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


def parse_drosophila_svg(svg_file: Path) -> BuchnerSVGContent:
    """Parse SVG file and extract eye data"""
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

    return BuchnerSVGContent(
        ommatidia=np.array(ommatidia),
        stars=np.array(stars),
        lattice_origin=lattice_origin,
        axes=axes,
        hemisphere_center=hem_center,
        hemisphere_radius=hem_radius
    )


def unproject(points_2d: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Inverse stereographic projection.

    Coordinate system:
    -Z = Anterior, +Y = Dorsal, +X = Right (for left eye)
    """
    # Center, flip, normalise
    p = (points_2d - center) * -1
    p = p * (2.0 / radius)

    # Inverse stereographic projection
    x, y = p[:, 0], p[:, 1]
    rho = np.sqrt(x ** 2 + y ** 2)
    c = 2 * np.arctan(rho / 2.0)

    # Center is left pole (-90 deg longitude)
    lon0 = -np.pi / 2
    lat = np.arcsin(np.clip((y * np.sin(c)) / (rho + 1e-9), -1, 1))
    lon = lon0 + np.arctan2(x * np.sin(c), rho * np.cos(c))

    # Spherical to Cartesian
    X = np.cos(lat) * np.sin(lon)
    Y = np.sin(lat)
    Z = -np.cos(lat) * np.cos(lon)

    return np.column_stack([X, Y, Z])


def project_to_stereo(dirs_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D directions to stereographic plane centered on the data.
    """
    center_dir = np.mean(dirs_3d, axis=0)
    fwd = center_dir / np.linalg.norm(center_dir)
    rgt = np.cross([0, 1, 0], fwd)
    rgt /= np.linalg.norm(rgt)
    up = np.cross(fwd, rgt)

    denom = 1 + np.dot(dirs_3d, fwd)
    points_2d = np.column_stack([
        np.dot(dirs_3d, rgt) / denom,
        np.dot(dirs_3d, up) / denom
    ])

    return points_2d, fwd, rgt, up


def stereo_to_sphere(points_2d: np.ndarray, fwd: np.ndarray,
                     rgt: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Back-project from stereographic plane to unit sphere.
    """
    x, y = points_2d[:, 0], points_2d[:, 1]
    r2 = x ** 2 + y ** 2
    denom = 1 + r2

    return (2.0 / denom)[:, None] * (x[:, None] * rgt + y[:, None] * up) + \
        ((1 - r2) / denom)[:, None] * fwd


def buchner_regularization(raw_dirs: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    Warp points based on the regularisation markers in the plot.

    raw_dirs: Raw viewing directions from digitized data
    markers: 3D positions of star markers
    """

    # Fit polynomial to star positions
    coeffs = np.polyfit(markers[:, 1], markers[:, 0], 2)
    deviation_func = np.poly1d(coeffs)

    regularized = raw_dirs.copy()
    regularized[:, 0] -= deviation_func(regularized[:, 1])

    # Renormalize to unit sphere
    norms = np.linalg.norm(regularized, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)

    return regularized / norms


def fit_lattice(raw_dirs: np.ndarray,
                density_scale: float = 1.0,
                lattice_angle: float = np.pi / 3,
                show_debug_plots: bool = True) -> np.ndarray:
    """
    Fit hexagonal lattice to match ommatidial density distribution.

    raw_dirs: 3D viewing directions (already unprojected from SVG)
    density_scale: Relative density (1.0 = original, 2.0 = twice as dense)
    lattice_angle: Angle between lattice basis vectors (π/3 = hexagonal)
    show_debug_plots: Whether to display intermediate plots
    """
    # Project to stereographic plane for density analysis
    pts_2d, fwd, rgt, up = project_to_stereo(raw_dirs)

    # Estimate local spacing in the raw data
    tree_raw = cKDTree(pts_2d)
    dist, _ = tree_raw.query(pts_2d, k=7)
    spacing = dist[:, 1:].mean(axis=1)
    mean_s = spacing.mean()

    # Build RBF interpolator for density field
    # (normalized spacing - higher values = lower density)
    rbf = RBFInterpolator(pts_2d, spacing / mean_s,
                          kernel='thin_plate_spline', smoothing=0.1)

    # Generate hexagonal lattice template
    b1 = np.array([1, 0])
    b2 = np.array([np.cos(lattice_angle), np.sin(lattice_angle)])
    grid = np.array([i * b1 + j * b2 for i in range(-55, 55) for j in range(-55, 55)])

    # Create convex hull for pruning
    hull = Delaunay(pts_2d)

    def is_inside(points, buffer=0.03):
        """Check if points are inside data extent."""
        inside = hull.find_simplex(points) >= 0
        if buffer == 0:
            return inside

        outside_indices = np.where(~inside)[0]
        if len(outside_indices) > 0:
            d_to_hull, _ = tree_raw.query(points[outside_indices])
            inside[outside_indices] = d_to_hull < buffer

        return inside

    def loss(p):
        origin, rot, scale = p[:2], p[2], p[3]
        mat = np.array([[np.cos(rot), -np.sin(rot)],
                        [np.sin(rot), np.cos(rot)]])
        transformed = (grid * scale) @ mat.T + origin

        # Forward error: how far are data points from lattice?
        t_tree = cKDTree(transformed)
        d_fwd, _ = t_tree.query(pts_2d)

        # Backward error: how far are lattice points from data?
        mask = is_inside(transformed, buffer=0)
        d_bwd, _ = tree_raw.query(transformed[mask]) if np.any(mask) else (np.array([1e3]), 0)

        return np.mean(d_fwd ** 2) * 2.0 + np.mean(d_bwd ** 2)

    # Adjust base spacing for density scaling
    adjusted_spacing = mean_s / np.sqrt(density_scale)

    # Optimize lattice
    res = minimize(loss, [0, 0, 0, adjusted_spacing],
                   bounds=[(-0.1, 0.1), (-0.1, 0.1),
                           (-np.pi / 4, np.pi / 4),
                           (adjusted_spacing * 0.8, adjusted_spacing * 1.05)])

    o, r, s = res.x[:2], res.x[2], res.x[3]
    mat = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    lattice = (grid * s) @ mat.T + o

    # density-based warping
    density_scales = rbf(lattice)
    lattice = o + (lattice - o) * density_scales[:, None]

    # Prune to eye shape
    lattice = lattice[is_inside(lattice, buffer=mean_s * 0.6)]

    print(f"Lattice optimization complete. Generated {len(lattice)} ommatidia")
    print(f"  Density scale: {density_scale:.2f}x")
    print(f"  Lattice angle: {np.degrees(lattice_angle):.1f}°")

    if show_debug_plots:
        plot_density_2d(pts_2d, lattice, rbf, mean_s)

    return stereo_to_sphere(lattice, fwd, rgt, up)


def get_ellipsoid_points(directions: np.ndarray, rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Calculate intersection points of direction rays with an ellipsoid.
    """
    val = (directions[:, 0] / rx) ** 2 + \
          (directions[:, 1] / ry) ** 2 + \
          (directions[:, 2] / rz) ** 2
    t = 1.0 / np.sqrt(val)
    return directions * t[:, np.newaxis]


def build_eye(svg_path: Path,
              density_scale: float = DENSITY_SCALE,
              lattice_angle_deg: float = LATTICE_ANGLE_DEG,
              regularize: bool = REGULARIZATION,
              show_plots: bool = SHOW_DEBUG_PLOTS,
              return_raw_data: bool = False) -> np.ndarray:
    """
    svg_path : Path to SVG file containing digitized ommatidial positions
    density_scale: Relative density (1.0 = match data, 2.0 = twice as dense)
    lattice_angle_deg: Angle between lattice basis vectors (60° = hexagonal)
    regularize: Regularization (using Buchner's 'forward' markers)
    show_plots: Whether to display debug plots
    return_raw_data: If True, return raw unprojected directions instead of fitted lattice
    """

    data = parse_drosophila_svg(svg_path)

    raw_dirs = unproject(data.ommatidia, data.hemisphere_center, data.hemisphere_radius)

    if len(data.stars) > 0:
        stars_3d = unproject(data.stars, data.hemisphere_center, data.hemisphere_radius)
    else:
        stars_3d = np.array([])

    if regularize and len(stars_3d) > 0:
        raw_dirs = buchner_regularization(raw_dirs, stars_3d)

    if show_plots:
        stars_3d = unproject(data.stars, data.hemisphere_center, data.hemisphere_radius)
        origin_3d = unproject(data.lattice_origin[np.newaxis, :], data.hemisphere_center, data.hemisphere_radius)[0]

        axes_3d = {
            axis_id: unproject(points, data.hemisphere_center, data.hemisphere_radius)
            for axis_id, points in data.axes.items()
        }

        plot_buchner_3d(raw_dirs, stars_3d, origin_3d, axes_3d,
                        title="Drosophila ommatidia viewing directions (Buchner, 1971)")

    if return_raw_data:
        return raw_dirs

    lattice_dirs = fit_lattice(
        raw_dirs,
        density_scale=density_scale,
        lattice_angle=np.radians(lattice_angle_deg),
        show_debug_plots=show_plots
    )

    lattice_dirs = lattice_dirs / np.linalg.norm(lattice_dirs, axis=1, keepdims=True)

    return lattice_dirs


if __name__ == "__main__":

    PLOT = True

    # Head dimensions adapted from Posnien et al., 2012, "Evolution of Eye Morphology and Rhodopsin Expression",
    # 10.1371/journal.pone.0037346
    HW = 830.0      # Head width (µm)
    FW = 400.0      # Frons width, medial gap (µm)
    EL = 530.0      # Eye length, vertical (µm)
    ED = 420.0      # Eye depth, anterior-posterior (µm)

    svg_file = Path("species_models/drosophila_custom/drosophila_Buchner_1971_redigitized.svg")

    L_dirs = build_eye(
        svg_file,
        density_scale=DENSITY_SCALE,
        lattice_angle_deg=LATTICE_ANGLE_DEG,
        regularize=REGULARIZATION,
        show_plots=PLOT
    )

    target_width = (HW - FW) / 2.0
    ry = EL / 2.0
    rz = ED / 2.0


    def error_func(rx_guess):
        pts = get_ellipsoid_points(L_dirs, rx_guess, ry, rz)
        width = np.max(pts[:, 0]) - np.min(pts[:, 0])
        return (width - target_width) ** 2


    res = minimize_scalar(error_func, bounds=(10, 500), method='bounded')
    best_rx = res.x
    print(f"Ellipsoid fit: Rx = {best_rx:.2f} µm")

    # Generate left eye positions
    L_origins_local = get_ellipsoid_points(L_dirs, best_rx, ry, rz)

    # Align medial edge to -FW/2
    current_medial_x = np.max(L_origins_local[:, 0])
    target_medial_x = -FW / 2.0
    shift_x = target_medial_x - current_medial_x
    L_origins = L_origins_local + np.array([shift_x, 0, 0])

    R_origins = L_origins.copy()
    R_origins[:, 0] *= -1
    R_dirs = L_dirs.copy()
    R_dirs[:, 0] *= -1

    all_origins = np.vstack([L_origins, R_origins])
    all_directions = np.vstack([L_dirs, R_dirs])
    eye_ids = np.concatenate([np.zeros(len(L_origins)), np.ones(len(R_origins))])

    print(f"\nFinal eye model:")
    print(f"  Total ommatidia: {len(all_origins)}")
    print(f"  Left eye: {len(L_origins)}")
    print(f"  Right eye: {len(R_origins)}")

    all_origins *= 0.001

    np.savez_compressed("species_models/drosophila_custom.npz",
                        directions=all_directions,
                        origins=all_origins,
                        eye_id=eye_ids)

    if PLOT:
        plot_eyes_3d(
            all_origins,
            all_directions,
            eye_ids,
            title='Drosophila eyes\n(parametric model fitted to Buchner, 1971)',
            show_sphere_projection=True
        )

        plot_density_3d(all_origins, all_directions, title="Ommatidia density")