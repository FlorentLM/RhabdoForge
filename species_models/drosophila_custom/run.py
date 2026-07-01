"""
Drosophila compound eye model fitted to measurements by Buchner
    "Dunkelanregung des stationaeren flugs der fruchtfliege Drosophila", 1971, PhD thesis

Plots were taken from Heisenberg and Wolff, 1984 (10.1007/978-3-642-69936-8) were manually
redigitised and are used as ground truth.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import numpy as np
import xml.etree.ElementTree as ET
from svg.path import parse_path, Line, Close

from insectvision.geometry.spherical import sphere_to_stereo, stereo_to_sphere
from insectvision.lattice_fitting.generator import FittingParameters, LatticeGenerator
from insectvision.lattice_fitting.plots import plot_lattice
from insectvision.lattice_fitting.profile import EyeMeasurements
from species_models.plots import plot_eyes_3d, plot_lattice_3d, plot_density_3d

# TODO: a GUI that replaces the svg + svg parsing


@dataclass
class SVGContent:
    ommatidia: np.ndarray
    forward_markers: np.ndarray
    lattice_position: np.ndarray
    axes: Dict[str, np.ndarray]
    hemisphere_center: np.ndarray
    hemisphere_radius: float


def _path_centroid(path: np.ndarray) -> np.ndarray:
    """Calculate centroid of an SVG path"""
    if not path:
        return np.zeros(2)

    points = [(path[0].start.real, path[0].start.imag)]
    for seg in path:
        points.append((seg.end.real, seg.end.imag))
    return np.mean(points, axis=0)


def _sample_path(path: np.ndarray, pts_per_segment: int = 20) -> np.ndarray:
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


def parse_drosophila_svg(svg_file: str | Path) -> 'SVGContent':
    """
    Parse svg file containing redigitised Buchner data.
    """

    tree = ET.parse(Path(svg_file))
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
        raise ValueError('Hemisphere circle not found in avg')
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


def plot_buchner_3d(
        ommatidia: np.ndarray,
        fwd_markers: np.ndarray,
        origin: np.ndarray,
        axes: Dict[str, np.ndarray],
        title: str='Drosophila eye (Buchner, 1971)'
    ):
    """3D version of the Buchner data svg."""

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*ommatidia.T, c='grey', s=10, alpha=0.5, label='Ommatidia')
    ax.scatter(*fwd_markers.T, c='red', s=150, marker='*', label='Forward', depthshade=False)
    ax.scatter(*origin, c='black', s=50, marker='X', label='Origin', depthshade=False)

    colors = {'axis-x': '#ff60b3', 'axis-y': '#00FF9B', 'axis-v': '#FFC400'}

    for axis_id, points in axes.items():
        ax.plot(*points.T, color=colors[axis_id], linewidth=2.5, label=f'Axis {axis_id[-1].upper()}')

    ax.set_title(title, fontsize=16)

    ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='Right (+X)')
    ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Up (+Y)')
    ax.quiver(0, 0, 0, 0, 0, -0.5, color='b', label='Forward (-Z)')

    all_points = np.vstack([ommatidia, fwd_markers])

    center = (all_points.max(axis=0) + all_points.min(axis=0)) / 2
    x_range = np.ptp(all_points[:, 0])
    y_range = np.ptp(all_points[:, 1])
    z_range = np.ptp(all_points[:, 2])

    ax.set_box_aspect((x_range, y_range, z_range))

    ax.set_xlim(center[0] - x_range, center[0] + x_range)
    ax.set_ylim(center[1] - y_range, center[1] + y_range)
    ax.set_zlim(center[2] - z_range, center[2] + z_range)

    ax.view_init(elev=30, azim=45)
    ax.legend()
    plt.tight_layout()
    plt.show()


def reconstruct_buchner_data(
        svg_path: str | Path,
        apply_regularisation: bool = True,
        show_plots: bool = False
    ) -> np.ndarray:
    """
    Build Drosophila ommatidial directions from the svg file.

    Args:
        - svg_path: Path to the svg
        - apply_regularisation: apply Buchner's forward-marker correction
        - show_plots: debug visualisations
    """

    data = parse_drosophila_svg(svg_path)

    raw_dirs = unproject(data.ommatidia, data.hemisphere_center, data.hemisphere_radius)

    if apply_regularisation and len(data.forward_markers) > 0:
        stars_3d = unproject(data.forward_markers, data.hemisphere_center, data.hemisphere_radius)

        coeffs = np.polyfit(stars_3d[:, 1], stars_3d[:, 0], 2)
        deviation = np.poly1d(coeffs)

        corrected = raw_dirs.copy()
        corrected[:, 0] -= deviation(corrected[:, 1])

        norms = np.linalg.norm(corrected, axis=1, keepdims=True)
        raw_dirs = corrected / np.where(norms == 0, 1, norms)


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

    return raw_dirs


if __name__ == "__main__":

    DENSITY_SCALE = 1.0
    PHYS_SCALE = 1.0
    SHOW_PLOT = True

    # Head dimensions from Posnien et al. 2012 (10.1371/journal.pone.0037346)
    HW = 830.0     # head width (µm)
    FW = 390.0     # frons width (µm)
    EL = 460.0     # eye length, vertical (µm)
    ED = 370.0     # eye depth, anterior-posterior (µm)  (inferred from D. simulans)

    svg_file = 'species_models/drosophila_custom/drosophila_Buchner_1971_redigitized.svg'

    HW *= PHYS_SCALE
    FW *= PHYS_SCALE
    EL *= PHYS_SCALE
    ED *= PHYS_SCALE

    L_dirs = reconstruct_buchner_data(svg_file, show_plots=SHOW_PLOT)

    pts2d, forward, right, up = sphere_to_stereo(L_dirs)

    # Measure source
    profile = EyeMeasurements.from_points(pts2d)

    # Generate
    gen = LatticeGenerator(profile, FittingParameters(density_scale=DENSITY_SCALE))
    lattice2d = gen.run(align=True, verbose=True)

    if SHOW_PLOT:
        plot_lattice(lattice2d, profile, density_scale=DENSITY_SCALE)

    # Back to the sphere
    lattice_dirs = stereo_to_sphere(lattice2d, forward, right, up)
    print(f'Generated {len(lattice_dirs)} ommatidia')

    # Map directions to head ellipsoid
    target_width = (HW - FW) / 2.0
    ry = EL / 2.0
    rz = ED / 2.0
    best_rx = target_width
    print(f'Ellipsoid fit: Rx = {best_rx:.2f} µm')

    L_positions = normals_to_ellipsoid(lattice_dirs, best_rx, ry, rz)

    # Align medial edge
    shift_x = -FW / 2.0 - np.max(L_positions[:, 0])
    L_positions += np.array([shift_x, 0, 0])

    # Mirror for right eye
    R_positions = L_positions.copy()
    R_positions[:, 0] *= -1
    R_dirs = lattice_dirs.copy()
    R_dirs[:, 0] *= -1

    all_positions = np.vstack([L_positions, R_positions])
    all_directions = np.vstack([lattice_dirs, R_dirs])
    eye_ids = np.concatenate([np.zeros(len(L_positions)), np.ones(len(R_positions))])

    print(f"\nFinal model:  L={len(L_positions)}  R={len(R_positions)}")

    np.savez_compressed(
        'species_models/drosophila_custom.npz',
        directions=all_directions,
        positions=all_positions,
        eye_id=eye_ids,
    )

    if SHOW_PLOT:
        plot_eyes_3d(
            all_positions, all_directions, eye_ids,
            title='Drosophila eyes\n(parametric model fitted to Buchner, 1971)',
            sphere_projection=True,
        )
        plot_density_3d(all_positions, all_directions)
        plot_lattice_3d(lattice_dirs, wireframe=True, color_by='psi6', title='Hexatic order')
