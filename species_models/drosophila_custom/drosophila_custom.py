"""
Drosophila compound eye model fitted to measurements by Buchner, 1971.

"Dunkelanregung des stationaeren flugs der fruchtfliege Drosophila" (PhD thesis)

Plots from Heisenberg and Wolff, 1984 (10.1007/978-3-642-69936-8) were manually
redigitised and are used as ground truth.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import minimize_scalar
from svg.path import parse_path, Line, Close

from insectvision.utils.lattice_topology import fit_lattice, facet_diameters

from species_models.drosophila_custom.plots_droso_custom import plot_buchner_3d
from species_models.plots import plot_eyes_3d, plot_lattice_3d, plot_density_3d

# TODO: The SVG parsing needs to be made more generic



# Defaults
DENSITY_SCALE = 0.95
LATTICE_ANGLE_DEG = 60.0
REGULARISATION = True
SHOW_DEBUG_PLOTS = True
PHYS_SCALE = 1.0

@dataclass
class SVGContent:
    ommatidia: np.ndarray
    forward_markers: np.ndarray
    lattice_position: np.ndarray
    axes: Dict[str, np.ndarray]
    hemisphere_center: np.ndarray
    hemisphere_radius: float


def _path_centroid(path) -> np.ndarray:
    """Calculate centroid of an SVG path"""

    if not path:
        return np.zeros(2)
    points = [(path[0].start.real, path[0].start.imag)]
    for seg in path:
        points.append((seg.end.real, seg.end.imag))
    return np.mean(points, axis=0)


def _sample_path(path, pts_per_segment: int = 20) -> np.ndarray:
    """Sample points along an SVG path"""

    if not path:
        return np.array([])

    points = [(path[0].start.real, path[0].start.imag)]
    for seg in path:
        if isinstance(seg, (Line, Close)):
            points.append((seg.end.real, seg.end.imag))
        else:
            for i in range(pts_per_segment):
                p = seg.point(i / (pts_per_segment - 1))
                points.append((p.real, p.imag))
    return np.array(points)


def parse_drosophila_svg(svg_file: Path) -> SVGContent:
    """
    Parse SVG file containing redigitised Buchner data.
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

    # Extract regularisation markers
    forward_markers = []
    for i in range(1, 6):
        elem = root.find(f".//*[@id='star-{i}']", ns)
        if elem is not None:
            forward_markers.append(_path_centroid(parse_path(elem.get('d'))))

    # Extract lattice position
    lattice_elem = root.find(".//*[@id='lattice-position']", ns)
    lattice_position = (
        _path_centroid(parse_path(lattice_elem.get('d')))
        if lattice_elem is not None
        else np.zeros(2)
    )

    # Extract axes
    axes = {}
    for axis_id in ('axis-x', 'axis-y', 'axis-v'):
        elem = root.find(f".//*[@id='{axis_id}']", ns)
        if elem is not None:
            axes[axis_id] = _sample_path(parse_path(elem.get('d')), 50)

    return SVGContent(
        ommatidia=np.array(ommatidia),
        forward_markers=np.array(forward_markers),
        lattice_position=lattice_position,
        axes=axes,
        hemisphere_center=hem_center,
        hemisphere_radius=hem_radius,
    )


def unproject(points_2d: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Inverse stereographic projection using Buchner's convention.

    Coordinate system: -Z = Anterior, +Y = Dorsal, +X = Right (left eye)
    """
    # TODO: make this a more generic math util

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


def regularise(raw_dirs: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """
    Correct distortion using Buchner's 'forward' markers.
    """
    coeffs = np.polyfit(markers[:, 1], markers[:, 0], 2)
    deviation = np.poly1d(coeffs)

    corrected = raw_dirs.copy()
    corrected[:, 0] -= deviation(corrected[:, 1])

    norms = np.linalg.norm(corrected, axis=1, keepdims=True)
    return corrected / np.where(norms == 0, 1, norms)


def normals_to_ellipsoid(directions: np.ndarray, rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Map viewing directions to positions on an ellipsoid such that the
    directions are the surface normals.

    This yields larger facet diameters in flatter regions.
    """
    # TODO: make this a more generic math util
    nx = directions[:, 0]
    ny = directions[:, 1]
    nz = directions[:, 2]

    # Calculate the scale factor for the normal mapping
    K = np.sqrt((nx * rx) ** 2 + (ny * ry) ** 2 + (nz * rz) ** 2)

    x = (rx ** 2 * nx) / K
    y = (ry ** 2 * ny) / K
    z = (rz ** 2 * nz) / K

    return np.column_stack([x, y, z])




def build_eye(
        svg_path: Path,
        density_scale: float = DENSITY_SCALE,
        lattice_angle_deg: float = LATTICE_ANGLE_DEG,
        regularize: bool = REGULARISATION,
        n_lloyd_iterations: int = 20,
        show_plots: bool = SHOW_DEBUG_PLOTS,
        return_raw_data: bool = False,
) -> np.ndarray:
    """
    Build Drosophila ommatidial directions from a Buchner SVG file.

    Args:
        svg_path : Path to digitised SVG
        density_scale : 1.0 = match Buchner data, 2.0 = twice as dense
        lattice_angle_deg : hex basis angle (60° = regular)
        regularize : apply Buchner's forward-marker correction
        n_lloyd_iterations : Lloyd's relaxation iterations
        show_plots : debug visualisations
        return_raw_data : return raw digitised directions (skip lattice generation)
    """

    data = parse_drosophila_svg(svg_path)

    raw_dirs = unproject(data.ommatidia, data.hemisphere_center, data.hemisphere_radius)

    if regularize and len(data.forward_markers) > 0:
        stars_3d = unproject(data.forward_markers, data.hemisphere_center, data.hemisphere_radius)
        raw_dirs = regularise(raw_dirs, stars_3d)

    if show_plots:
        stars_3d = unproject(data.forward_markers, data.hemisphere_center, data.hemisphere_radius)
        position_3d = unproject(
            data.lattice_position[np.newaxis, :],
            data.hemisphere_center, data.hemisphere_radius,
        )[0]
        axes_3d = {
            aid: unproject(pts, data.hemisphere_center, data.hemisphere_radius)
            for aid, pts in data.axes.items()
        }
        plot_buchner_3d(
            raw_dirs, stars_3d, position_3d, axes_3d,
            title="Drosophila ommatidia viewing directions (Buchner, 1971)",
        )

    if return_raw_data:
        return raw_dirs

    lattice_dirs = fit_lattice(
        raw_dirs,
        density_scale=density_scale,
        lattice_angle=np.radians(lattice_angle_deg),
        relaxation_max_iter=n_lloyd_iterations,
        verbose=show_plots,
        show_plots=show_plots,
    )

    lattice_dirs /= np.linalg.norm(lattice_dirs, axis=1, keepdims=True)

    print(f"Generated {len(lattice_dirs)} ommatidia "
          f"(density_scale={density_scale:.2f}x, "
          f"lattice_angle={lattice_angle_deg:.1f}°)")

    return lattice_dirs


if __name__ == "__main__":

    PLOT = True

    # Head dimensions from Posnien et al. 2012 (10.1371/journal.pone.0037346)
    HW = 830.0     # head width (µm)
    FW = 390.0     # frons width (µm)
    EL = 460.0     # eye length, vertical (µm)

    # inferred from D. simulans eye shape (ap axis is ~0.8 dv axis)
    ED = 370.0     # eye depth, anterior-posterior (µm)

    svg_file = Path("species_models/drosophila_custom/drosophila_Buchner_1971_redigitized.svg")

    HW *= PHYS_SCALE
    FW *= PHYS_SCALE
    EL *= PHYS_SCALE
    ED *= PHYS_SCALE

    L_dirs = build_eye(
        svg_file,
        density_scale=DENSITY_SCALE,
        lattice_angle_deg=LATTICE_ANGLE_DEG,
        regularize=REGULARISATION,
        show_plots=PLOT,
    )

    target_width = (HW - FW) / 2.0
    ry = EL / 2.0
    rz = ED / 2.0

    res = minimize_scalar(
        lambda rx: (np.ptp(normals_to_ellipsoid(L_dirs, rx, ry, rz)[:, 0]) - target_width) ** 2,
        bounds=(10, 500), method='bounded',
    )
    best_rx = res.x
    print(f"Ellipsoid fit: Rx = {best_rx:.2f} µm")

    L_positions = normals_to_ellipsoid(L_dirs, best_rx, ry, rz)

    # Align medial edge
    shift_x = -FW / 2.0 - np.max(L_positions[:, 0])
    L_positions += np.array([shift_x, 0, 0])

    # Mirror for right eye
    R_positions = L_positions.copy()
    R_positions[:, 0] *= -1
    R_dirs = L_dirs.copy()
    R_dirs[:, 0] *= -1

    all_positions = np.vstack([L_positions, R_positions])
    all_directions = np.vstack([L_dirs, R_dirs])
    eye_ids = np.concatenate([np.zeros(len(L_positions)), np.ones(len(R_positions))])

    # Facet diameter per lens, from the Voronoi cell (the dual of the lattice)
    L_diam = facet_diameters(L_positions, L_dirs)
    R_diam = facet_diameters(R_positions, R_dirs)
    all_diameters = np.concatenate([L_diam, R_diam]).astype(np.float32)
    print(f"Facet diameter: median {np.median(all_diameters):.2f} µm  "
          f"IQR {np.percentile(all_diameters, 25):.2f}-{np.percentile(all_diameters, 75):.2f}  "
          f"range {all_diameters.min():.2f}-{all_diameters.max():.2f}")

    print(f"\nFinal model:  L={len(L_positions)}  R={len(R_positions)}")

    np.savez_compressed(
        "species_models/drosophila_custom.npz",
        directions=all_directions,
        positions=all_positions,
        eye_id=eye_ids,
        lens_diameter_um=all_diameters,
    )

    if PLOT:
        plot_eyes_3d(
            all_positions, all_directions, eye_ids,
            title='Drosophila eyes\n(parametric model fitted to Buchner, 1971)',
            show_sphere_projection=True,
        )
        plot_density_3d(all_positions, all_directions, title="Ommatidia density")

        plot_lattice_3d(L_dirs, wireframe=True, color_by='psi6', title=f"Hexatic order")