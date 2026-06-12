from typing import Callable
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree

from insectvision.geometry.neighbours import delaunay_edges
from insectvision.geometry.hexatic import hexatic_order, phasor_from_points
from insectvision.geometry.spherical import stereo_proj


def plot_fitted_comparison(
        pts_2d_raw: np.ndarray,
        pts_2d_lattice: np.ndarray,
        density_fn: Callable,
        title: str = "Fitted ommatidia vs. raw data",
):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel 1: density field
    pad = 0.1
    x_lo, x_hi = pts_2d_raw[:, 0].min() - pad, pts_2d_raw[:, 0].max() + pad
    y_lo, y_hi = pts_2d_raw[:, 1].min() - pad, pts_2d_raw[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x_lo, x_hi, 100),
                          np.linspace(y_lo, y_hi, 100))
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    Z = density_fn(grid_pts).reshape(gx.shape)

    im = ax1.contourf(gx, gy, Z, levels=20, cmap='viridis')
    ax1.scatter(*pts_2d_raw.T, c='red', s=2, alpha=0.5, label='Raw data')
    ax1.set_title("Density field")
    ax1.set_aspect('equal')
    ax1.legend(fontsize=8)
    fig.colorbar(im, ax=ax1, label='Relative density')

    # Panel 2: Lattice vs. raw
    ax2.scatter(*pts_2d_raw.T, c='grey', s=6, alpha=0.5, marker='x', label='Raw data')
    ax2.scatter(*pts_2d_lattice.T, c='green', s=4, alpha=0.8, label='Lloyd lattice')
    ax2.set_title(f"Lattice overlay ({len(pts_2d_lattice)} pts)")
    ax2.set_aspect('equal')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # Panel 3: local spacing comparison
    if len(pts_2d_lattice) > 7:
        tree_lat = cKDTree(pts_2d_lattice)
        d_lat, _ = tree_lat.query(pts_2d_lattice, k=7)
        spacing_lat = d_lat[:, 1:].mean(axis=1)

        tree_raw = cKDTree(pts_2d_raw)
        d_raw, _ = tree_raw.query(pts_2d_raw, k=7)
        spacing_raw = d_raw[:, 1:].mean(axis=1)

        r_raw = np.linalg.norm(pts_2d_raw - pts_2d_raw.mean(axis=0), axis=1)
        r_lat = np.linalg.norm(pts_2d_lattice - pts_2d_lattice.mean(axis=0), axis=1)

        ax3.scatter(r_raw, spacing_raw, c='grey', s=4, alpha=0.4, marker='x', label='Raw')
        ax3.scatter(r_lat, spacing_lat, c='green', s=4, alpha=0.8, label='Lloyd')
        ax3.set_xlabel('Distance from centre')
        ax3.set_ylabel('Local spacing')
        ax3.set_title('Spacing profile')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()

    return fig


def plot_lattice_3d(
        dirs_3d: np.ndarray,
        wireframe: bool = True,
        color_by: str = 'spacing',
        title: str = None,
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
        dirs_3d: (N, 3) Unit directions
        wireframe (bool): If True, draw Delaunay edges between neighbours
        color_by (str): Optional
            'spacing': colour points by local inter-ommatidial spacing
               'psi6': colour by local hexatic order parameter
        title (str): Optional title
        point_size: float
        edge_color (str or color): Colour for wireframe edges
        edge_alpha: float
        edge_linewidth: float
        max_edge_factor (float): Edges longer than this times local spacing are pruned (hull artifacts)
        cmap (str): Colormap name for scalar colouring
    """

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    pts_2d, _, _, _ = stereo_proj(dirs_3d)

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

    # Set up axes properly
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

    return fig, ax


def plot_eyes_3d(origins, directions, eye_id, title, arrow_length=0.1, show_sphere_projection=False):
    """
    Plot eye model in 3D with direction arrows or spherical projections.

    Args:
        origins: (N, 3) ommatidium positions
        directions: (N, 3) unit direction vectors
        eye_id: (N,) array with 0 for left eye, 1 for right eye
        title: Plot title
        arrow_length: Length of direction arrows (used if show_sphere_projection=False)
        show_sphere_projection: If True, extend direction vectors to sphere surface
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

    if show_sphere_projection:

        max_origin_dist = np.linalg.norm(origins, axis=1).max()
        sphere_radius = max_origin_dist * 2.5  # multiplier for sphere size
        plot_scale = sphere_radius

        # Ray: P(t) = O + tD
        # Sphere: |P|^2 = R^2  => t^2(D.D) + 2t(O.D) + (O.O - R^2) = 0
        # Since D is unit vector, a = 1

        # Dot products for b and c
        dot_od = np.einsum('ij,ij->i', origins, directions)  # (N,)
        dot_oo = np.einsum('ij,ij->i', origins, origins)  # (N,)

        a = 1.0
        b = 2 * dot_od
        c = dot_oo - sphere_radius ** 2

        discriminant = b ** 2 - 4 * a * c
        t = (-b + np.sqrt(np.maximum(0, discriminant))) / (2 * a)

        intersections = origins + directions * t[:, np.newaxis]

        # Plot intersection points on sphere
        ax.scatter(intersections[right_mask, 0], intersections[right_mask, 1], intersections[right_mask, 2],
                   c='red', s=5, alpha=0.3, marker='.', label='Right projections')
        ax.scatter(intersections[left_mask, 0], intersections[left_mask, 1], intersections[left_mask, 2],
                   c='blue', s=5, alpha=0.3, marker='.', label='Left projections')

        # Draw connecting lines (subsampled)
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
    gizmo_len = plot_scale * 0.5 if show_sphere_projection else 0.3
    ax.quiver(0, 0, 0, gizmo_len, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='X axis')
    ax.quiver(0, 0, 0, 0, gizmo_len, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='Y axis')
    ax.quiver(0, 0, 0, 0, 0, -gizmo_len, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z axis')

    ax.set_xlabel('← Insect\'s left | Insect\'s right →')
    ax.set_ylabel('← Insect\'s down | Insect\'s up →')
    ax.set_zlabel('← Insect\'s back | Insect\'s front →')
    ax.set_title(title)

    ax.legend(loc='upper right', fontsize=8)

    # Set up axes properly
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


def plot_density_2d(raw_pts_2d, lattice_pts_2d, rbf_func, mean_spacing):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    padding = 0.1
    x_min, x_max = raw_pts_2d[:, 0].min() - padding, raw_pts_2d[:, 0].max() + padding
    y_min, y_max = raw_pts_2d[:, 1].min() - padding, raw_pts_2d[:, 1].max() + padding
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])

    spacing_vals = rbf_func(grid_pts).reshape(gx.shape) * mean_spacing

    im = ax1.contourf(gx, gy, spacing_vals, levels=20, cmap='viridis_r')
    ax1.scatter(raw_pts_2d[:, 0], raw_pts_2d[:, 1], c='red', s=2, alpha=0.5, label='Raw data')
    ax1.set_title("Inter-ommatidial spacing (stereographic)")
    fig.colorbar(im, ax=ax1, label='Relative spacing')
    ax1.legend()

    ax2.scatter(lattice_pts_2d[:, 0], lattice_pts_2d[:, 1], s=5, c='black', edgecolors='none')
    ax2.set_title(f"Procedural lattice")
    ax2.set_aspect('equal')

    ax3.scatter(raw_pts_2d[:, 0], raw_pts_2d[:, 1], c='black', s=2, alpha=0.5, label='Real ommatidia')
    ax3.scatter(lattice_pts_2d[:, 0], lattice_pts_2d[:, 1], c='green', s=2, alpha=0.5, label='Procedural lattice')
    ax3.legend()

    plt.tight_layout()
    plt.show()


def plot_density_3d(positions, directions, title="Ommatidia density"):

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
    ax.set_title(title)

    # Set up axes properly
    center = (positions.max(axis=0) + positions.min(axis=0)) / 2
    x_range = np.ptp(positions[:, 0])
    y_range = np.ptp(positions[:, 1])
    z_range = np.ptp(positions[:, 2])

    ax.set_box_aspect((x_range, y_range, z_range))

    ax.set_xlim(center[0] - x_range, center[0] + x_range)
    ax.set_ylim(center[1] - y_range, center[1] + y_range)
    ax.set_zlim(center[2] - z_range, center[2] + z_range)

    plt.show()


def plot_lens_diameters_3d(positions, diameters, title="Lens diameters"):

    # Lens diameters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        positions[:, 0], positions[:, 1], positions[:, 2],
        c=diameters, cmap='plasma', s=15, alpha=0.9
    )

    fig.colorbar(sc, ax=ax, label="Facet Diameter (µm)", fraction=0.03)

    ax.set_xlabel('Lateral (X)')
    ax.set_ylabel('Anterior-Posterior (Z)')
    ax.set_zlabel('Dorsal-Ventral (Y)')
    ax.set_title(title)

    # Set up axes properly
    center = (positions.max(axis=0) + positions.min(axis=0)) / 2
    x_range = np.ptp(positions[:, 0])
    y_range = np.ptp(positions[:, 1])
    z_range = np.ptp(positions[:, 2])

    ax.set_box_aspect((x_range, y_range, z_range))

    ax.set_xlim(center[0] - x_range, center[0] + x_range)
    ax.set_ylim(center[1] - y_range, center[1] + y_range)
    ax.set_zlim(center[2] - z_range, center[2] + z_range)

    plt.show()
