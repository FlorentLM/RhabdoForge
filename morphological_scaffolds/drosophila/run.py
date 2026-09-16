"""
Drosophila compound eye model fitted to measurements by Erich Büchner
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

from rhabdoforge.geometry.spherical import (
    sphere_to_stereo, stereo_to_sphere, EllipsoidCurvature, LinearGradient, position_from_curvature
)
from rhabdoforge.compound_eyes.model import Model
from rhabdoforge.lattice_fitting.relaxation import mirror_bilateral
from rhabdoforge.lattice_fitting.generator import FittingParameters, LatticeGenerator, EyeMeasurements
from rhabdoforge.lattice_fitting.plots import plot_lattice, plot_scalar_field_3d, set_3d_equal, draw_gizmo
from rhabdoforge.lattice_fitting.plots import plot_eye_scaffold_3d, plot_lattice_3d


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


def _stereo_to_sphere_buchner(points_2d: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Inverse stereographic projection using Buchner's convention.
    Coordinate system: -Z = Anterior, +Y = Dorsal, +X = Right (left eye)
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

    # Spherical to cartesian
    X = np.cos(lat) * np.sin(lon)
    Y = np.sin(lat)
    Z = -np.cos(lat) * np.cos(lon)

    return np.column_stack([X, Y, Z])


def _fig_buchner_3d(
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

    draw_gizmo(ax, length=0.5)

    all_points = np.vstack([ommatidia, fwd_markers])
    set_3d_equal(ax, all_points)

    ax.view_init(elev=30, azim=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

# TODO: svg parsing module that is more generic

def parse_buchner_svg(svg_file: str | Path) -> 'SVGContent':
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

    data = parse_buchner_svg(svg_path)

    raw_dirs = _stereo_to_sphere_buchner(data.ommatidia, data.hemisphere_center, data.hemisphere_radius)

    if apply_regularisation and len(data.forward_markers) > 0:
        stars_3d = _stereo_to_sphere_buchner(data.forward_markers, data.hemisphere_center, data.hemisphere_radius)

        coeffs = np.polyfit(stars_3d[:, 1], stars_3d[:, 0], 2)
        deviation = np.poly1d(coeffs)

        corrected = raw_dirs.copy()
        corrected[:, 0] -= deviation(corrected[:, 1])

        norms = np.linalg.norm(corrected, axis=1, keepdims=True)
        raw_dirs = corrected / np.where(norms == 0, 1, norms)

    if show_plots:
        stars_3d = _stereo_to_sphere_buchner(data.forward_markers, data.hemisphere_center, data.hemisphere_radius)
        position_3d = _stereo_to_sphere_buchner(
            data.lattice_position[np.newaxis, :],
            data.hemisphere_center, data.hemisphere_radius,
        )[0]
        axes_3d = {
            aid: _stereo_to_sphere_buchner(pts, data.hemisphere_center, data.hemisphere_radius)
            for aid, pts in data.axes.items()
        }
        _fig_buchner_3d(
            raw_dirs, stars_3d, position_3d, axes_3d,
            title="Drosophila ommatidia viewing directions (Büchner, 1971)",
        )

    return raw_dirs


if __name__ == "__main__":
    from pathlib import Path

    DENSITY_SCALE = 1.0
    PHYS_SCALE = 1.0
    SHOW_PLOTS = True

    # Head dimensions from Posnien et al. 2012 (10.1371/journal.pone.0037346)
    HW = 830.0 * PHYS_SCALE   # head width (µm)
    FW = 390.0 * PHYS_SCALE   # frons width (µm)
    EL = 460.0 * PHYS_SCALE   # eye length, vertical (µm)
    ED = 370.0 * PHYS_SCALE   # eye depth, anterior-posterior (µm)  (inferred from Drosophila simulans)

    # Shape from the head ellipsoid above, scale fitted so median facet spacing
    # matches Gonzalez-Bellido et al. (2011): 16.85 µm
    EYE_SCALE = 0.8512

    # Buffry et al. 2024: anterior-ventral larger, dorsal-posterior smaller
    GRADIENT_AXIS = (0.0, -1.0, -1.0)
    GRADIENT_SLOPE = 0.08   # I dont have a real number for this :(

    svg_file = 'morphological_scaffolds/drosophila/data/buchner1971_redigitised.svg'

    L_dirs = reconstruct_buchner_data(svg_file, show_plots=SHOW_PLOTS)

    pts2d, forward, right, up = sphere_to_stereo(L_dirs)

    # Measure source
    profile = EyeMeasurements.from_points(
        points2d=pts2d,
    )

    # Generate
    gen = LatticeGenerator(profile, FittingParameters(density_scale=DENSITY_SCALE))
    lattice2d = gen.run(align=True, verbose=True)

    if SHOW_PLOTS:
        plot_lattice(lattice2d, profile, density_scale=DENSITY_SCALE)

    # Back to sphere
    lattice_dirs = stereo_to_sphere(lattice2d, forward, right, up)

    # Place each direction at its local eye radius (µm)
    rx, ry, rz = (HW - FW) / 2.0 * EYE_SCALE, EL / 2.0 * EYE_SCALE, ED / 2.0 * EYE_SCALE
    curvature = LinearGradient(base=EllipsoidCurvature(rx, ry, rz), axis=GRADIENT_AXIS, slope=GRADIENT_SLOPE)
    L_positions = position_from_curvature(lattice_dirs, curvature)

    # Align medial edge, then mirror across X=0 to build the right eye
    shift_x = -FW / 2.0 - np.max(L_positions[:, 0])

    positions_both, directions_both, eye_ids_both = mirror_bilateral(
        positions=L_positions,
        directions=lattice_dirs,
        shift=shift_x,
        source_side='left'
    )

    n_right = int(eye_ids_both.sum())
    print(f"\nFinal model:  L={len(positions_both) - n_right}  R={n_right}")

    save_path = Path('assets') / 'drosophila_scaffold.npz'

    np.savez_compressed(save_path,
        positions=positions_both,
        directions=directions_both,
        eye_id=eye_ids_both,
    )

    if SHOW_PLOTS:
        plot_eye_scaffold_3d(
            positions=positions_both,
            directions=directions_both,
            eye_ids=eye_ids_both,
            title='Drosophila eyes\n(fitted to Büchner, 1971)',
            sphere_projection=True
        )

        # DEBUG # TODO: remove
        diag = Model(positions=positions_both, directions=directions_both, eye_indices=eye_ids_both)
        aperture_um = diag._buf['aperture_um']
        ioa_deg = np.rad2deg(diag._buf['ioa_angles', np.arange(len(positions_both))].mean(axis=1))

        plot_scalar_field_3d(positions_both, aperture_um,
            title='Drosophila facet size', label='Facet spacing (µm)')

        plot_scalar_field_3d(positions_both, ioa_deg,
            title='Drosophila interommatidial angle', label='IOA (degrees)', cmap='plasma_r')

        plot_lattice_3d(lattice_dirs, wireframe=True, color_by='psi6', title='Hexatic order')