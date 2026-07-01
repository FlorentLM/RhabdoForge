from typing import Optional
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

from insectvision.geometry.neighbours import delaunay_edges
from insectvision.geometry.hexatic import hexatic_order, phasor_from_points
from insectvision.geometry.spherical import sphere_to_stereo


# TODO: These functions should be cleaned up and moced to the lattice_fitting.plot submodule


def plot_lattice_3d(
        dirs_3d: np.ndarray,
        wireframe: bool = True,
        color_by: str = 'spacing',
        title: Optional[str] = None,
        point_size: float = 8.0,
        edge_color: str = '0.4',
        edge_alpha: float = 0.3,
        edge_linewidth: float = 0.5,
        max_edge_factor: float = 2.0,
        cmap: str = 'plasma_r',
    ):
    """
    3D scatter/wireframe plot of a lattice on the unit sphere.

    Args:
        - dirs_3d: Unit directions, (N, 3)
        - wireframe: bool, if True, draw Delaunay edges between neighbours
        - color_by: Optional str,
            'spacing': colour points by local inter-ommatidial spacing
            'psi6': colour by local hexatic order parameter
        - title: Optional str, plot title
        - point_size: float
        - edge_color (str or colour): Colour for wireframe edges
        - edge_alpha: float
        - edge_linewidth: float
        - max_edge_factor: float, Edges longer than this times local spacing are pruned (hull artifacts)
        - cmap: str, Colormap name for scalar colouring
    """

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    pts_2d, _, _, _ = sphere_to_stereo(dirs_3d)

    edges = None
    adj = None
    if (wireframe or color_by in ('spacing', 'psi6')) and len(dirs_3d) > 3:
        edges = delaunay_edges(pts_2d, max_length_factor=max_edge_factor)
        adj = [[] for _ in range(len(pts_2d))]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)

    c_values = None
    c_label = None

    if color_by == 'spacing' and adj is not None:
        N = len(pts_2d)
        c_values = np.zeros(N)
        for i in range(N):
            nb = adj[i]
            if not nb:
                continue
            c_values[i] = np.linalg.norm(pts_2d[nb] - pts_2d[i], axis=1).mean()
        c_label = 'Local spacing (stereo)'

    elif color_by == 'psi6' and adj is not None:
        # Psi6 using topological (Delaunay) neighbours
        c_values = hexatic_order(phasor_from_points(pts_2d, adj)).astype(np.float64)
        c_label = 'Hexatic order (ψ₆)'
        cmap = 'RdYlGn'

    if c_values is not None:
        sc = ax.scatter(*dirs_3d.T, s=point_size, c=c_values, cmap=cmap, alpha=0.8, depthshade=False)
        fig.colorbar(sc, ax=ax, label=c_label, shrink=0.6, pad=0.1)

    else:
        ax.scatter(*dirs_3d.T, s=point_size, c='0.35', alpha=0.6, depthshade=False)

    if wireframe and edges is not None:
        segments = []
        for a, b in edges:
            segments.append([dirs_3d[a], dirs_3d[b]])

        lc = Line3DCollection(segments, colors=edge_color, linewidths=edge_linewidth, alpha=edge_alpha)

        ax.add_collection3d(lc)

    center = (dirs_3d.max(axis=0) + dirs_3d.min(axis=0)) / 2
    x_range = np.ptp(dirs_3d[:, 0])
    y_range = np.ptp(dirs_3d[:, 1])
    z_range = np.ptp(dirs_3d[:, 2])

    ax.set_box_aspect((x_range, y_range, z_range))

    ax.set_xlim(center[0] - x_range, center[0] + x_range)
    ax.set_ylim(center[1] - y_range, center[1] + y_range)
    ax.set_zlim(center[2] - z_range, center[2] + z_range)

    if title:
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_eyes_3d(
        origins: np.ndarray,
        directions: np.ndarray,
        eye_id: np.ndarray,
        title: Optional[str] = None,
        arrow_length: float = 0.1,
        sphere_projection: bool = False
    ):
    """
    Plot eye model in 3D with direction arrows or spherical projections.

    Args:
        - origins: ommatidium positions, (N, 3)
        - directions: unit direction vectors, (N, 3)
        - eye_id: array with 0 for left eye, 1 for right eye, (N,)
        - title: optional plot title
        - arrow_length: Length of direction arrows (used if show_sphere_projection=False)
        - sphere_projection: If True, extend direction vectors to sphere surface
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    right_mask = eye_id == 1
    left_mask = eye_id == 0

    # Origins
    ax.scatter(origins[right_mask, 0], origins[right_mask, 1], origins[right_mask, 2],
               c='red', s=20, alpha=0.8, label='Right eye')
    ax.scatter(origins[left_mask, 0], origins[left_mask, 1], origins[left_mask, 2],
               c='blue', s=20, alpha=0.8, label='Left eye')

    if sphere_projection:

        max_origin_dist = np.linalg.norm(origins, axis=1).max()
        sphere_radius = max_origin_dist * 2.5
        plot_scale = sphere_radius

        dot_od = np.einsum('ij,ij->i', origins, directions)
        dot_oo = np.einsum('ij,ij->i', origins, origins)

        a = 1.0
        b = 2 * dot_od
        c = dot_oo - sphere_radius ** 2

        discriminant = b ** 2 - 4 * a * c
        t = (-b + np.sqrt(np.maximum(0, discriminant))) / (2 * a)

        intersections = origins + directions * t[:, np.newaxis]

        # intersection points on sphere
        ax.scatter(intersections[right_mask, 0], intersections[right_mask, 1], intersections[right_mask, 2],
                   c='red', s=5, alpha=0.3, marker='.', label='Right projections')
        ax.scatter(intersections[left_mask, 0], intersections[left_mask, 1], intersections[left_mask, 2],
                   c='blue', s=5, alpha=0.3, marker='.', label='Left projections')

        # connecting lines (subsampled)
        step = max(1, len(origins) // 100)

        for i in range(0, len(origins), step):
            pts = np.vstack((origins[i], intersections[i]))
            color = 'red' if eye_id[i] == 1 else 'blue'
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.1, linewidth=0.5)

        # Wireframe sphere for reference
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.05, linewidth=0.5)

    else:
        tips = origins + directions * arrow_length

        def plot_sticks(mask, color, label_prefix):
            if not np.any(mask):
                return

            p_start = origins[mask]
            p_end = tips[mask]

            ax.scatter(p_end[:, 0], p_end[:, 1], p_end[:, 2],
                       c=color, s=5, alpha=0.5, marker='.', label=f'{label_prefix} directions')

            nan_separator = np.full((p_start.shape[0], 3), np.nan)
            segments = np.stack((p_start, p_end, nan_separator), axis=1).reshape(-1, 3)

            ax.plot(segments[:, 0], segments[:, 1], segments[:, 2],
                    color=color, alpha=0.3, linewidth=1)

        plot_sticks(right_mask, 'red', 'Right')
        plot_sticks(left_mask, 'blue', 'Left')

        max_range = np.array([
            origins[:, 0].max() - origins[:, 0].min(),
            origins[:, 1].max() - origins[:, 1].min(),
            origins[:, 2].max() - origins[:, 2].min()
        ]).max() / 2.0
        plot_scale = max_range

    # Coordinate gizmo
    gizmo_len = plot_scale * 0.5 if sphere_projection else 0.3
    ax.quiver(0, 0, 0, gizmo_len, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='X axis')
    ax.quiver(0, 0, 0, 0, gizmo_len, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='Y axis')
    ax.quiver(0, 0, 0, 0, 0, -gizmo_len, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z axis')

    # TODO: this is wrong?
    ax.set_xlabel('← Insect\'s left | Insect\'s right →')
    ax.set_ylabel('← Insect\'s down | Insect\'s up →')
    ax.set_zlabel('← Insect\'s back | Insect\'s front →')

    if title:
        ax.set_title(title)

    ax.legend(loc='upper right', fontsize=8)

    center = (origins.max(axis=0) + origins.min(axis=0)) / 2
    x_range = np.ptp(origins[:, 0])
    y_range = np.ptp(origins[:, 1])
    z_range = np.ptp(origins[:, 2])

    ax.set_box_aspect((x_range, y_range, z_range))

    ax.set_xlim(center[0] - x_range, center[0] + x_range)
    ax.set_ylim(center[1] - y_range, center[1] + y_range)
    ax.set_zlim(center[2] - z_range, center[2] + z_range)

    plt.tight_layout()
    plt.show()


def plot_density_3d(
        positions: np.ndarray,
        directions: np.ndarray,
        title: Optional[str] = 'Ommatidia density'
    ):

    tree = cKDTree(directions)
    dists, _ = tree.query(directions, k=7)
    angular_spacing_deg = np.degrees(np.mean(dists[:, 1:], axis=1))

    # IOA
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(positions[:, 0], positions[:, 2], positions[:, 1],
                    c=angular_spacing_deg, cmap='plasma_r', s=20, alpha=0.8)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Inter-ommatidial angle (degrees)')

    ax.set_xlabel('Lateral (X)')
    ax.set_ylabel('Anterior-Posterior (Z)')
    ax.set_zlabel('Dorsal-Ventral (Y)')
    if title:
        ax.set_title(title)

    center = (positions.max(axis=0) + positions.min(axis=0)) / 2
    x_range = np.ptp(positions[:, 0])
    y_range = np.ptp(positions[:, 1])
    z_range = np.ptp(positions[:, 2])

    ax.set_box_aspect((x_range, y_range, z_range))

    ax.set_xlim(center[0] - x_range, center[0] + x_range)
    ax.set_ylim(center[1] - y_range, center[1] + y_range)
    ax.set_zlim(center[2] - z_range, center[2] + z_range)

    plt.tight_layout()
    plt.show()