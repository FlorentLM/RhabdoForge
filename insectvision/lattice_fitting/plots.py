from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata

from insectvision.geometry.hexatic import phasor_from_points, hexatic_order
from insectvision.geometry.neighbours import delaunay_edges, delaunay_neighbours, mean_neighbour_distance, walk_rows
from insectvision.geometry.linalg import principal_axis_angle
from insectvision.lattice_fitting.profile import trace_lattice_rows, EyeMeasurements


# Quick look

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


# Six-panel diagnostic

def lattice_diagnostics(stages: dict, save_path: str | Path = 'lattice_diagnostics.png'):
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

    ach_pts = mean_neighbour_distance(query_points=final, k=6)
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
        ach = mean_neighbour_distance(query_points=lat, k=6)
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

    si, src_rows, bearings = trace_lattice_rows(raw_points, raw_nb, theta_fn)

    li = int(np.argmin(np.linalg.norm(final - raw_points[si], axis=1)))
    gen_rows = walk_rows(final, gen_nb, li, bearings)

    edges = delaunay_edges(final, max_length_factor=1.8)
    ax.add_collection(LineCollection(final[edges], color='0.8', lw=0.4, alpha=0.5))

    colors = ['#ff60b3', '#00c074', '#f0a000']

    for col, gr, sr, brg, lab in zip(colors, gen_rows, src_rows, bearings, 'XYV'):

        d = np.rad2deg(abs(principal_axis_angle(gr) - brg)) % 180
        d = min(d, 180 - d)

        ax.plot(gr[:, 0], gr[:, 1], color=col, lw=2.2, zorder=5, label=f'Row {lab}  Δ={d:.1f}°')
        ax.plot(sr[:, 0], sr[:, 1], color=col, lw=1.4, ls=':', alpha=0.85, zorder=4)

    ax.scatter([final[li, 0]], [final[li, 1]], c='k', s=45, marker='*', zorder=6)
    ax.plot(bnd[:, 0], bnd[:, 1], 'k--', lw=1.0, alpha=0.4)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)

    ax.set_title('Lattice rows: generated (solid) vs. source (dotted)', fontsize=13)

    # E. Hexatic order + defects
    ax = axs[0, 2]

    psi = hexatic_order(phasor_from_points(final, gen_nb))
    coord = np.array([len(n) for n in gen_nb])

    inside = domain.signed_distance(final) < -hide_margin * gen_ms

    ax.scatter(final[~inside, 0], final[~inside, 1], c='lightgrey', s=8, alpha=0.6)

    im = ax.scatter(final[inside, 0], final[inside, 1], c=psi[inside], cmap='RdYlGn',
                    s=22, vmin=0.5, vmax=1.0, edgecolors='none')

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
    psi_src = hexatic_order(phasor_from_points(raw_points, raw_nb))
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
