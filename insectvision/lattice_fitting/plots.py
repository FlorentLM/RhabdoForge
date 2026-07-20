from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

from insectvision.geometry.hexatic import compute_psi6, hexatic_order
from insectvision.geometry.spherical import sphere_to_stereo, chord_to_angle
from insectvision.geometry.neighbours import delaunay_edges, delaunay_neighbours, topological_spacing, metric_spacing
from insectvision.lattice_fitting.generator import EyeMeasurements


# TODO: Move these two to shared plot utils

def set_3d_equal(ax: Axes3D, points: np.ndarray) -> None:
    """
    Equal-aspect 3D box with symmetric limits about the data centre.
    """
    points = np.asarray(points, dtype=float)
    center = 0.5 * (points.max(axis=0) + points.min(axis=0))
    ranges = np.maximum(np.ptp(points, axis=0), 1e-9)

    ax.set_box_aspect(tuple(ranges))
    ax.set_xlim(center[0] - ranges[0], center[0] + ranges[0])
    ax.set_ylim(center[1] - ranges[1], center[1] + ranges[1])
    ax.set_zlim(center[2] - ranges[2], center[2] + ranges[2])


def draw_gizmo(ax: Axes3D, length: float, arrow_ratio: float = 0.1) -> None:
    ax.quiver(0, 0, 0, length, 0, 0, color='r', arrow_length_ratio=arrow_ratio, linewidth=2, label='right (+X)')
    ax.quiver(0, 0, 0, 0, length, 0, color='g', arrow_length_ratio=arrow_ratio, linewidth=2, label='up (+Y)')
    ax.quiver(0, 0, 0, 0, 0, -length, color='b', arrow_length_ratio=arrow_ratio, linewidth=2, label='forward (-Z)')


# Quick look (2D)

def plot_lattice(
        lattice: np.ndarray,
        measurements: 'EyeMeasurements',
        density_scale: float = 1.0,
        save_path: str | Path = 'procedural_lattice.png'
    ):

    spacing = measurements.spacing_fn(lattice).ravel()
    edges = delaunay_edges(lattice, max_length_factor=1.8)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    if measurements.source_points is not None:
        ax.scatter(measurements.source_points[:, 0], measurements.source_points[:, 1],
                   c='k', s=2, alpha=0.15, zorder=1)
    ax.add_collection(LineCollection(lattice[edges], color='0.6', lw=0.5, alpha=0.6, zorder=2))
    sc = ax.scatter(lattice[:, 0], lattice[:, 1], c=spacing, cmap='viridis_r', s=10, zorder=3)

    bnd = np.vstack([measurements.domain.boundary, measurements.domain.boundary[0]])
    ax.plot(bnd[:, 0], bnd[:, 1], 'k--', lw=1.0, alpha=0.5)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='local spacing')
    ax.set_title(f'Procedural lattice (density_scale={density_scale}, N={len(lattice)})')

    fig.tight_layout()
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f'Saved {save_path}')

    plt.show()


# Lattice on the unit sphere (3D)

def plot_lattice_3d(
        dirs_3d: np.ndarray,
        wireframe: bool = True,
        color_by: Optional[str] = 'spacing',
        title: Optional[str] = None,
        point_size: float = 8.0,
        edge_color: str = '0.4',
        edge_alpha: float = 0.3,
        edge_linewidth: float = 0.5,
        max_edge_factor: float = 2.0,
        cmap: str = 'plasma_r',
    ):
    """
    3D scatter / wireframe of a lattice on the unit sphere.

    The Delaunay graph, spacing and hexatic order are all measured on the
    stereographic projection of the directions (same convention as the fitter),
    then drawn back on the 3D directions.

    Args:
        - dirs_3d: unit directions, (N, 3)
        - wireframe: draw pruned Delaunay edges
        - color_by: 'spacing' | 'psi6' | None
        - max_edge_factor: prune edges longer than this * local spacing (hull artefacts)
    """

    dirs_3d = np.asarray(dirs_3d, dtype=float)
    pts_2d, *_ = sphere_to_stereo(dirs_3d)
    have_graph = len(dirs_3d) > 3

    # One Delaunay adjacency serves both colouring and the wireframe.
    adj = delaunay_neighbours(pts_2d, max_length_factor=max_edge_factor) if have_graph else None

    c_values, c_label = None, None
    if adj is not None and color_by == 'spacing':
        c_values = topological_spacing(pts_2d, adj)
        c_label = 'Local spacing (stereo)'
    elif adj is not None and color_by == 'psi6':
        c_values = hexatic_order(compute_psi6(pts_2d, adj))
        c_label, cmap = 'Hexatic order ($\\psi_6$)', 'RdYlGn'

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    if c_values is not None:
        sc = ax.scatter(*dirs_3d.T, s=point_size, c=c_values, cmap=cmap, alpha=0.8, depthshade=False)
        fig.colorbar(sc, ax=ax, label=c_label, shrink=0.6, pad=0.1)
    else:
        ax.scatter(*dirs_3d.T, s=point_size, c='0.35', alpha=0.6, depthshade=False)

    if wireframe and adj is not None:
        edges = delaunay_edges(pts_2d, max_length_factor=max_edge_factor)
        if len(edges):
            ax.add_collection3d(Line3DCollection(
                dirs_3d[edges], colors=edge_color, linewidths=edge_linewidth, alpha=edge_alpha))

    if title:
        ax.set_title(title, fontsize=12)

    set_3d_equal(ax, dirs_3d)

    plt.tight_layout()
    plt.show()


def plot_eye_scaffold_3d(
        positions: np.ndarray,
        directions: np.ndarray,
        eye_ids: np.ndarray,
        title: Optional[str] = None,
        arrow_length: float = 0.1,
        sphere_projection: bool = False
    ):
    """
    Plot an eye model scaffold (positions, directions, eye ID) in 3D with direction arrows or spherical projections.

    Args:
        - positions: ommatidium positions, (N, 3)
        - directions: unit direction vectors, (N, 3)
        - eye_id: 0 for left eye, 1 for right eye, (N,)
        - arrow_length: length of direction arrows (used if sphere_projection=False)
        - sphere_projection: if True, extend direction vectors to a reference sphere
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    right_mask = eye_ids == 1
    left_mask = eye_ids == 0

    ax.scatter(positions[right_mask, 0], positions[right_mask, 1], positions[right_mask, 2],
               c='red', s=20, alpha=0.8, label='Right eye')
    ax.scatter(positions[left_mask, 0], positions[left_mask, 1], positions[left_mask, 2],
               c='blue', s=20, alpha=0.8, label='Left eye')

    if sphere_projection:

        max_origin_dist = np.linalg.norm(positions, axis=1).max()
        sphere_radius = max_origin_dist * 2.5
        plot_scale = sphere_radius

        dot_od = np.einsum('ij,ij->i', positions, directions)
        dot_oo = np.einsum('ij,ij->i', positions, positions)

        b = 2 * dot_od
        c = dot_oo - sphere_radius ** 2

        discriminant = b ** 2 - 4 * c
        t = (-b + np.sqrt(np.maximum(0, discriminant))) / 2.0

        intersections = positions + directions * t[:, np.newaxis]

        ax.scatter(intersections[right_mask, 0], intersections[right_mask, 1], intersections[right_mask, 2],
                   c='red', s=5, alpha=0.3, marker='.', label='Right projections')
        ax.scatter(intersections[left_mask, 0], intersections[left_mask, 1], intersections[left_mask, 2],
                   c='blue', s=5, alpha=0.3, marker='.', label='Left projections')

        # connecting lines (subsampled)
        step = max(1, len(positions) // 100)
        for i in range(0, len(positions), step):
            pts = np.vstack((positions[i], intersections[i]))
            color = 'red' if eye_ids[i] == 1 else 'blue'
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.1, linewidth=0.5)

        # Wireframe reference sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.05, linewidth=0.5)

    else:
        tips = positions + directions * arrow_length

        def plot_sticks(mask, color, label_prefix):
            if not np.any(mask):
                return

            p_start = positions[mask]
            p_end = tips[mask]

            ax.scatter(p_end[:, 0], p_end[:, 1], p_end[:, 2],
                       c=color, s=5, alpha=0.5, marker='.', label=f'{label_prefix} directions')

            nan_separator = np.full((p_start.shape[0], 3), np.nan)
            segments = np.stack((p_start, p_end, nan_separator), axis=1).reshape(-1, 3)

            ax.plot(segments[:, 0], segments[:, 1], segments[:, 2],
                    color=color, alpha=0.3, linewidth=1)

        plot_sticks(right_mask, 'red', 'Right')
        plot_sticks(left_mask, 'blue', 'Left')

        max_range = np.ptp(positions, axis=0).max() / 2.0
        plot_scale = max_range

    if title:
        ax.set_title(title)

    set_3d_equal(ax, positions)
    draw_gizmo(ax, length=plot_scale * 0.5)

    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


# Angular density (3D)

def plot_density_3d(
        positions: np.ndarray,
        directions: np.ndarray,
        title: Optional[str] = 'Ommatidia density',
        k: int = 6,
    ):
    """
    Colour ommatidia by local inter-ommatidial angle (mean angular spacing to the
    k nearest neighbours in direction space).
    """

    # TODO: this plot is rubbish, edges are wrong, should use the robust estimators

    positions = np.asarray(positions, dtype=float)
    directions = np.asarray(directions, dtype=float)

    # Mean chord to k nearest neighbours on the direction sphere -> great-circle angle
    chord = metric_spacing(cKDTree(directions), k=k)
    ioa_deg = np.rad2deg(chord_to_angle(chord))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                    c=ioa_deg, cmap='plasma_r', s=20, alpha=0.8)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Inter-ommatidial angle (degrees)')

    if title:
        ax.set_title(title)

    set_3d_equal(ax, positions)
    draw_gizmo(ax, length=0.5 * np.ptp(positions, axis=0).max())

    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()