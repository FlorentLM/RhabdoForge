from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from insectvision.geometry.hexatic import compute_psi6, hexatic_order
from insectvision.geometry.linalg import principal_axes
from insectvision.geometry.spherical import sphere_to_stereo, chord_to_angle
from insectvision.geometry.neighbours import delaunay_edges, delaunay_neighbours, graph_spacing, ball_spacing
from insectvision.geometry.lattice import trace_lattice_rows
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


# Six-panel diagnostic (2D)

def plot_lattice_diagnostics(stages: dict, save_path: str | Path = 'lattice_diagnostics.png'):
    """
    A  Source spacing field             : target density, for context
    B  Density residual map             : achieved vs. target spacing (interior)
    C  Achieved-vs-target scatter       : per-point spacing, init (grey) vs. final
    D  Lattice rows: fitted vs. source  : lattice axes traced from the centre
    E  Order + emergent defects         : lattice coloured by psi6, 5-7 disclinations circled
    F  psi6 distribution: gen vs. source: statistical-match plot
    """

    hide_margin = 1.5

    final = stages['final']
    init = stages['init']
    target_spacing_fn = stages['target_spacing_fn']
    gen_ms = stages['gen_spacing']
    density_scale = stages['parameters'].density_scale

    measurements = stages['measurements']

    raw_points = measurements.source_points
    domain = measurements.domain
    spacing_fn = measurements.spacing_fn
    theta_fn = measurements.theta_fn
    src_ms = measurements.mean_spacing

    raw_nb = delaunay_neighbours(raw_points, max_length_factor=1.8)
    gen_nb = delaunay_neighbours(final, max_length_factor=1.8)

    fig, axs = plt.subplots(2, 3, figsize=(19, 12))
    for ax in axs.ravel():
        ax.set_aspect('equal')
        ax.set_facecolor('#fcfcfc')
    bnd = np.vstack([domain.boundary, domain.boundary[0]])

    def grid_over(points, n=140, pad=0.05):
        lo = points.min(0) - pad
        hi = points.max(0) + pad
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
        return gx, gy, np.column_stack([gx.ravel(), gy.ravel()])

    # A. Source spacing field
    ax = axs[0, 0]

    gx, gy, gpts = grid_over(raw_points)
    Z = np.where(domain.inside(gpts).reshape(gx.shape), spacing_fn(gpts).reshape(gx.shape), np.nan)

    cf = ax.contourf(gx, gy, Z, levels=30, cmap='viridis_r', alpha=0.9)
    ax.scatter(raw_points[:, 0], raw_points[:, 1], c='white', s=1.5, alpha=0.5)
    ax.plot(bnd[:, 0], bnd[:, 1], 'k--', lw=1.0, alpha=0.6)
    ax.set_title('Source spacing field', fontsize=13)

    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label='spacing')

    # B. Density residual map
    ax = axs[1, 0]

    ach_pts = ball_spacing(query_points=final, k=6)
    tgt_pts = target_spacing_fn(final).ravel()

    inside = domain.signed_distance(final) < -hide_margin * gen_ms

    rel_pts = np.where(tgt_pts > 1e-9, (ach_pts - tgt_pts) / tgt_pts, np.nan)

    gx, gy, gpts = grid_over(final)
    ach_grid = griddata(final[inside], ach_pts[inside], (gx, gy), method='linear')

    tgt_grid = target_spacing_fn(gpts).reshape(gx.shape)
    gsd = domain.signed_distance(gpts).reshape(gx.shape)

    valid = (gsd < -1.5 * gen_ms) & np.isfinite(ach_grid) & (tgt_grid > 1e-9)
    resid = np.where(valid, (ach_grid - tgt_grid) / tgt_grid, np.nan)

    vmax = 0.15

    rc = ax.contourf(gx, gy, resid,
                     levels=np.linspace(-vmax, vmax, 31),
                     cmap='RdBu_r', extend='both', alpha=0.8)

    ax.contour(gx, gy, resid, levels=[0], colors='k', linewidths=1.0, alpha=0.6)
    ax.scatter(final[inside, 0], final[inside, 1], c=rel_pts[inside], cmap='RdBu_r',
               vmin=-vmax, vmax=vmax, s=12, edgecolors='k', linewidths=0.2, zorder=3)

    ax.scatter(final[~inside, 0], final[~inside, 1], c='lightgrey', s=4, alpha=0.5)
    ax.plot(bnd[:, 0], bnd[:, 1], 'k--', lw=1.0, alpha=0.6)
    ax.set_title('Density residual (achieved vs target, interior)', fontsize=13)

    fig.colorbar(rc, ax=ax, fraction=0.046, pad=0.04, format=lambda x, _: f'{x:+.0%}')

    # C. Achieved vs. target scatter
    ax = axs[0, 1]

    ax.set_aspect('auto')
    for lat, color, label in [(init, '0.6', 'after warp'), (final, '#1f77b4', 'final')]:
        ach = ball_spacing(query_points=lat, k=6)
        tgt = target_spacing_fn(lat).ravel()

        inside = domain.signed_distance(lat) < -hide_margin * gen_ms
        mask = inside & np.isfinite(ach) & (tgt > 0)

        m = mask & (ach > 0) & np.isfinite(tgt)
        slope = float(np.polyfit(np.log(tgt[m]), np.log(ach[m]), 1)[0]) if m.sum() >= 5 else np.nan

        ax.scatter(tgt[mask], ach[mask], s=6, c=color, alpha=0.5, label=f"{label}: slope={slope:.2f}")

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', lw=1.0, alpha=0.7, label='ideal (slope 1)')

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('target spacing')
    ax.set_ylabel('achieved spacing')

    ax.legend(fontsize=10, loc='upper left')

    ax.set_title('Contrast gain (interior)', fontsize=13)

    # D. Lattice rows fitted vs. source
    ax = axs[1, 1]

    # Trace source from geometric center
    src_seed, src_rows, bearings = trace_lattice_rows(raw_points, raw_nb, theta_fn=theta_fn)

    # Trace generated lattice using the same spatial seed and the source bearings
    gen_seed, gen_rows, _ = trace_lattice_rows(final, gen_nb, seed=src_seed, bearings=bearings)

    edges = delaunay_edges(final, max_length_factor=1.8)
    ax.add_collection(LineCollection(final[edges], color='0.8', lw=0.4, alpha=0.5))

    colors = ['#ff60b3', '#00c074', '#f0a000']

    for col, gr, sr, brg, lab in zip(colors, gen_rows, src_rows, bearings, 'XYV'):

        # PCA to get the angle
        major = principal_axes(gr)[0][:, 0]
        angle = np.arctan2(major[1], major[0])
        d = np.rad2deg(np.abs(angle) - brg)  % 180
        d = min(d, 180 - d)

        ax.plot(gr[:, 0], gr[:, 1], color=col, lw=2.2, zorder=5, label=f'Row {lab}  \u0394={d:.1f}\u00b0')
        ax.plot(sr[:, 0], sr[:, 1], color=col, lw=1.4, ls=':', alpha=0.85, zorder=4)

    ax.scatter(gen_seed[0], gen_seed[1], c='k', s=45, marker='*', zorder=6)
    ax.plot(bnd[:, 0], bnd[:, 1], 'k--', lw=1.0, alpha=0.4)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    ax.set_title('Lattice rows: generated (solid) vs. source (dotted)', fontsize=13)

    # E. Hexatic order + defects
    ax = axs[0, 2]

    z6 = compute_psi6(final, gen_nb)
    psi = hexatic_order(z6)
    coord = np.array([len(n) for n in gen_nb])

    inside = domain.signed_distance(final) < -hide_margin * gen_ms

    ax.scatter(final[~inside, 0], final[~inside, 1], c='lightgrey', s=8, alpha=0.6)

    im = ax.scatter(final[inside, 0], final[inside, 1], c=psi[inside], cmap='RdYlGn',
                    s=22, vmin=0.2, vmax=1.0, edgecolors='none')

    defect = inside & (coord != 6)
    ax.scatter(final[defect, 0], final[defect, 1], facecolors='none', edgecolors='k',
               s=46, linewidths=0.7, zorder=4)

    ax.plot(bnd[:, 0], bnd[:, 1], 'k--', lw=1.0, alpha=0.4)

    frac = defect.sum() / max(inside.sum(), 1)
    ax.set_title(f'$\\psi_6$ + defects (interior 5/7: {frac:.0%}, {defect.sum()})', fontsize=13)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$\psi_6$')

    # F. psi6 distribution, generated vs. source
    ax = axs[1, 2]

    ax.set_aspect('auto')
    psi_src = hexatic_order(compute_psi6(raw_points, raw_nb))
    in_src = domain.signed_distance(raw_points) < -hide_margin * src_ms

    cg = coord[inside]
    cs = np.array([len(n) for n in raw_nb])[in_src]

    ax.hist(psi[inside], bins=30, range=(0, 1), density=True, alpha=0.55,
            color='#1f77b4', label=f'generated (6-fold {np.mean(cg == 6):.0%})')
    ax.hist(psi_src[in_src], bins=30, range=(0, 1), density=True, alpha=0.55,
            color='#d62728', label=f'source (6-fold {np.mean(cs == 6):.0%})')

    ax.set_xlabel(r'$\psi_6$ (interior)')
    ax.set_ylabel('density')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_title('Hexatic order vs. source eye', fontsize=13)

    fig.suptitle(f'Procedural lattice diagnostics (density_scale={density_scale}, '
                 f'N={len(final)} vs. source {len(raw_points)})', fontsize=15)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

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
        c_values = graph_spacing(pts_2d, adj)
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

    positions = np.asarray(positions, dtype=float)
    directions = np.asarray(directions, dtype=float)

    # Mean chord to k nearest neighbours on the direction sphere -> great-circle angle
    chord = ball_spacing(cKDTree(directions), k=k)
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