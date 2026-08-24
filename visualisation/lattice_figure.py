import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import griddata

from rhabdoforge.geometry.spherical import sphere_to_stereo
from rhabdoforge.geometry.linalg import principal_axes
from rhabdoforge.geometry.hexatic import compute_psi6, hexatic_order
from rhabdoforge.geometry.neighbours import delaunay_edges, delaunay_neighbours, metric_spacing
from rhabdoforge.geometry.lattice import trace_lattice_rows
from rhabdoforge.lattice_fitting.generator import LatticeGenerator, FittingParameters, EyeMeasurements
from morphological_scaffolds.drosophila.run import reconstruct_buchner_data

from visualisation.plot_settings import PlotSettings, despine, Z_TEXT


SVG = 'morphological_scaffolds/drosophila/data/buchner1971_redigitised.svg'

DENSITY_SCALE     = 1.0         # > 1 packs more ommatidia
DENSITY_SMOOTHING = 0.1         # RBF smoothing of the spacing field
AXES_SMOOTHING    = 0.25        # RBF smoothing of the hexatic-axis field
MIN_HEX_ORDER     = 0.2         # |psi6| below which a point is dropped from the axis fit
GHOST_SOURCE      = 'lattice'   # 'lattice' | 'hull' | 'edge' | 'none'
ALIGN             = True        # align init grid to the source cloud (False = purely parametric)
BYPASS_INIT       = False       # If True, starts from the source data directly

HIDE_MARGIN = 1.5               # Interior mask = points > HIDE_MARGIN * spacing inside the boundary

# Points sizes
PT_SOURCE   = 1.0   # Source cloud dots
PT_LATTICE  = 3.0   # Lattice nodes (B)
PT_RESID    = 6.0   # Residual scatter
PT_PSI      = 8.0   # Psi6 scatter
PT_DEFECT   = 20.0  # defects ring

ROW_COLORS  = ('#ff60b3', '#00c074', '#f0a000')

# Panel letters
LETTER_DX          = -0.012     # right edge this far left of the axes (figure fraction)
LETTER_DX_TICKS    = -0.006     # extra gap left of a panel's y-tick labels (for D)
LETTERS_CLEAR_TICKS = {'D'}     # panels whose letter must clear y-tick labels


## -----------------------------------------------------------------------


def _grid_over(points, n=140, pad=0.05):
    lo = points.min(0) - pad
    hi = points.max(0) + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n))
    return gx, gy, np.column_stack([gx.ravel(), gy.ravel()])


def _map_axes(ax, ylabel=None, s=None):
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=(s.small if s else None))


def _boundary(ax, domain, s, alpha=0.6):
    bnd = np.vstack([domain.boundary, domain.boundary[0]])
    ax.plot(bnd[:, 0], bnd[:, 1], '--', color=s.frame, lw=s.axis_lw, alpha=alpha)


def _cbar(fig, mappable, ax, s, label=None, **kw):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04, **kw)
    if label:
        cb.set_label(label, fontsize=s.small)
    cb.ax.tick_params(labelsize=s.tiny)
    cb.outline.set_linewidth(s.axis_lw)
    return cb


def _spacing_cbar(fig, mappable, ax, s, vlim, label, extend=None):

    kw = {}
    if vlim is not None:
        kw['ticks'] = np.linspace(vlim[0], vlim[1], 4)
    if extend is not None:
        kw['extend'] = extend
    cb = _cbar(fig, mappable, ax, s, label=label, **kw)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2g}'))
    return cb


def _panel_letter(fig, s, letter, ax, renderer, dx=LETTER_DX, clear_ticks=False):

    pos = ax.get_position()
    inv = fig.transFigure.inverted()

    # Vertical: top of the (left) title
    y_top = pos.y1
    title = getattr(ax, '_left_title', None)
    if title is not None and title.get_text():
        try:
            y_top = title.get_window_extent(renderer).transformed(inv).y1
        except Exception:
            pass

    # Horizontal: just left of the axes, clear y-tick labels where present
    x_right = pos.x0 + dx
    if clear_ticks:
        try:
            x_tb = ax.get_tightbbox(renderer).transformed(inv).x0
            x_right = min(x_right, x_tb + LETTER_DX_TICKS)
        except Exception:
            pass

    fig.text(x_right, y_top, letter, ha='right', va='top',
             fontsize=s.initials, fontweight='bold', color='black', zorder=Z_TEXT)


def _spacing_vlim(stages):
    m = stages['measurements']
    _, _, gpts = _grid_over(m.source_points)
    field = m.spacing_fn(gpts)[m.domain.inside(gpts)]
    nodes = m.spacing_fn(stages['final']).ravel()
    allv = np.concatenate([np.asarray(field).ravel(), nodes])
    lo, hi = np.nanpercentile(allv, [1, 99])
    return float(lo), float(hi)


# Panels

def panel_spacing_field(ax, fig, stages, s, vlim=None, ylabel=None):
    """Source spacing field (target density)."""
    m = stages['measurements']
    raw, domain = m.source_points, m.domain

    gx, gy, gpts = _grid_over(raw)
    Z = np.where(domain.inside(gpts).reshape(gx.shape),
                 m.spacing_fn(gpts).reshape(gx.shape), np.nan)

    if vlim is not None:
        levels = np.linspace(vlim[0], vlim[1], 30)
        cf = ax.contourf(gx, gy, Z, levels=levels, cmap='viridis_r', alpha=0.9,
                         extend='both')
    else:
        cf = ax.contourf(gx, gy, Z, levels=30, cmap='viridis_r', alpha=0.9)

    ax.scatter(raw[:, 0], raw[:, 1], c='white', s=PT_SOURCE, alpha=0.5)
    _boundary(ax, domain, s)
    _map_axes(ax, ylabel=ylabel, s=s)

    _spacing_cbar(fig, cf, ax, s, vlim, label='Spacing')   # pointy caps come from extend='both'
    ax.set_title('Source spacing field', fontsize=s.title, loc='left', pad=4)


def panel_lattice(ax, fig, stages, s, vlim=None, ylabel=None):
    """The generated lattice coloured by local spacing."""
    m = stages['measurements']
    final = stages['final']

    spacing = m.spacing_fn(final).ravel()
    edges = delaunay_edges(final, max_length_factor=1.8)
    vmin, vmax = (vlim if vlim is not None else (None, None))

    if m.source_points is not None:
        ax.scatter(m.source_points[:, 0], m.source_points[:, 1],
                   c='k', s=PT_SOURCE, alpha=0.15, zorder=1)
    ax.add_collection(LineCollection(final[edges], color='0.6', lw=0.3, alpha=0.6,
                                     zorder=2))
    sc = ax.scatter(final[:, 0], final[:, 1], c=spacing, cmap='viridis_r',
                    s=PT_LATTICE, vmin=vmin, vmax=vmax, zorder=3)
    _boundary(ax, m.domain, s, alpha=0.5)
    _map_axes(ax, ylabel=ylabel, s=s)

    # extend='both' so this scatter colourbar gets the same pointy caps as A / E
    _spacing_cbar(fig, sc, ax, s, vlim, label='Spacing', extend='both')
    ax.set_title('Generated lattice', fontsize=s.title, loc='left', pad=4)


def panel_density_residual(ax, fig, stages, s, vmax=0.15):
    """Generated vs. target spacing residual."""
    m = stages['measurements']
    domain = m.domain
    final = stages['final']
    target_spacing_fn = stages['target_spacing_fn']
    gen_ms = stages['gen_spacing']

    ach_pts = metric_spacing(query_points=final, k=6)
    tgt_pts = target_spacing_fn(final).ravel()
    inside = domain.signed_distance(final) < -HIDE_MARGIN * gen_ms
    rel_pts = np.where(tgt_pts > 1e-9, (ach_pts - tgt_pts) / tgt_pts, np.nan)

    gx, gy, gpts = _grid_over(final)
    ach_grid = griddata(final[inside], ach_pts[inside], (gx, gy), method='linear')
    tgt_grid = target_spacing_fn(gpts).reshape(gx.shape)
    gsd = domain.signed_distance(gpts).reshape(gx.shape)
    valid = (gsd < -1.5 * gen_ms) & np.isfinite(ach_grid) & (tgt_grid > 1e-9)
    resid = np.where(valid, (ach_grid - tgt_grid) / tgt_grid, np.nan)

    rc = ax.contourf(gx, gy, resid, levels=np.linspace(-vmax, vmax, 31),
                     cmap='RdBu_r', extend='both', alpha=0.8)
    ax.contour(gx, gy, resid, levels=[0], colors='k', linewidths=s.axis_lw, alpha=0.6)
    ax.scatter(final[inside, 0], final[inside, 1], c=rel_pts[inside], cmap='RdBu_r',
               vmin=-vmax, vmax=vmax, s=PT_RESID, edgecolors='k', linewidths=0.15,
               zorder=3)
    ax.scatter(final[~inside, 0], final[~inside, 1], c='lightgrey', s=PT_RESID * 0.35,
               alpha=0.5)
    _boundary(ax, domain, s)
    _map_axes(ax)

    _cbar(fig, rc, ax, s, format=lambda x, _: f'{x:+.0%}')
    ax.set_title('Density residual', fontsize=s.title, loc='left', pad=4)


def panel_radial_profile(ax, stages, s, n_bins=12):
    """
    Local spacing vs. eccentricity, source vs. generated (median + IQR band).
    """
    m = stages['measurements']
    domain = m.domain
    raw, final = m.source_points, stages['final']
    centre = raw.mean(axis=0)

    def profile(pts, ms):
        keep = domain.signed_distance(pts) < -HIDE_MARGIN * ms
        p = pts[keep]
        ecc = np.linalg.norm(p - centre, axis=1)
        spac = metric_spacing(query_points=p, k=6)
        good = np.isfinite(spac)
        return ecc[good], spac[good]

    edges = None
    for pts, ms, col, lab in [(raw, m.mean_spacing, s.red, 'source'),
                              (final, stages['gen_spacing'], s.blue, 'generated')]:
        ecc, spac = profile(pts, ms)
        if edges is None:                       # shared bins (from source extent)
            edges = np.linspace(0, np.percentile(ecc, 99), n_bins + 1)
        idx = np.clip(np.digitize(ecc, edges) - 1, 0, n_bins - 1)
        cen = 0.5 * (edges[:-1] + edges[1:])

        def per_bin(fn):
            return np.array([fn(spac[idx == b]) if np.any(idx == b) else np.nan
                             for b in range(n_bins)])

        med = per_bin(np.median)
        q1 = per_bin(lambda v: np.percentile(v, 25))
        q3 = per_bin(lambda v: np.percentile(v, 75))

        ax.fill_between(cen, q1, q3, color=col, alpha=0.18, lw=0)
        ax.plot(cen, med, color=col, lw=s.curve_lw, label=lab)

    ax.set_xlabel('Eccentricity', fontsize=s.small)
    ax.set_ylabel('Spacing', fontsize=s.small)
    ax.tick_params(labelsize=s.tiny)
    despine(ax)

    leg = ax.legend(fontsize=s.tiny, loc='best')
    leg.get_frame().set_edgecolor(s.frame)
    leg.get_frame().set_linewidth(0.3)
    ax.set_title('Spacing profiles', fontsize=s.title, loc='left', pad=4)


def panel_order_defects(ax, fig, stages, s):
    """Lattice coloured by psi6 with 5-/7-fold disclinations circled."""
    m = stages['measurements']
    domain = m.domain
    final = stages['final']
    gen_ms = stages['gen_spacing']

    gen_nb = delaunay_neighbours(final, max_length_factor=1.8)
    psi = hexatic_order(compute_psi6(final, gen_nb))
    coord = np.array([len(n) for n in gen_nb])
    inside = domain.signed_distance(final) < -HIDE_MARGIN * gen_ms

    ax.scatter(final[~inside, 0], final[~inside, 1], c='lightgrey',
               s=PT_PSI * 0.4, alpha=0.6)
    im = ax.scatter(final[inside, 0], final[inside, 1], c=psi[inside], cmap='RdYlGn',
                    s=PT_PSI, vmin=0.2, vmax=1.0, edgecolors='none')

    defect = inside & (coord != 6)
    ax.scatter(final[defect, 0], final[defect, 1], facecolors='none', edgecolors='k',
               s=PT_DEFECT, linewidths=0.5, zorder=4)
    _boundary(ax, domain, s, alpha=0.4)
    _map_axes(ax)

    frac = defect.sum() / max(inside.sum(), 1)
    _cbar(fig, im, ax, s, label=r'$\psi_6$')
    ax.set_title(f'$\\psi_6$ + disclinations (5-/7-fold: {frac:.0%})',
                 fontsize=s.title, loc='left', pad=4)


def panel_rows(ax, stages, s):
    """Lattice axes, generated (solid) vs. source (dotted)."""
    m = stages['measurements']
    domain = m.domain
    raw, final = m.source_points, stages['final']

    raw_nb = delaunay_neighbours(raw, max_length_factor=1.8)
    gen_nb = delaunay_neighbours(final, max_length_factor=1.8)

    src_seed, src_rows, bearings = trace_lattice_rows(raw, raw_nb, theta_fn=m.theta_fn)
    gen_seed, gen_rows, _ = trace_lattice_rows(final, gen_nb, seed=src_seed, bearings=bearings)

    edges = delaunay_edges(final, max_length_factor=1.8)
    ax.add_collection(LineCollection(final[edges], color='0.8', lw=0.3, alpha=0.5))

    for col, gr, sr, brg, lab in zip(ROW_COLORS, gen_rows, src_rows, bearings, 'XYV'):
        major = principal_axes(gr)[0][:, 0]
        angle = np.arctan2(major[1], major[0])
        d = np.rad2deg(np.abs(angle) - brg) % 180
        d = min(d, 180 - d)
        ax.plot(gr[:, 0], gr[:, 1], color=col, lw=1.4, zorder=5,
                label=f'Row {lab}  \u0394={d:.1f}\u00b0')
        ax.plot(sr[:, 0], sr[:, 1], color=col, lw=0.9, ls=':', alpha=0.85, zorder=4)

    ax.scatter(gen_seed[0], gen_seed[1], c='k', s=20, marker='*', zorder=6)
    _boundary(ax, domain, s, alpha=0.4)
    _map_axes(ax)

    leg = ax.legend(fontsize=s.tiny, loc='upper right', framealpha=0.9)
    leg.get_frame().set_edgecolor(s.frame)
    leg.get_frame().set_linewidth(0.3)
    ax.set_title('Lattice rows:\nGenerated (solid) vs. Source (dotted)',
                 fontsize=s.title, loc='left', pad=4)


def make_figure(stages, s=None):
    s = (s or PlotSettings.nature_double(height_mm=125.0)).apply()
    fig = s.new_figure()

    gs = GridSpec(2, 3, figure=fig, hspace=0.24, wspace=0.28)

    vlim = _spacing_vlim(stages)

    # Row 0: Source (A), Generated lattice (B), Hex order + defects (C)
    axA = fig.add_subplot(gs[0, 0])
    panel_spacing_field(axA, fig, stages, s, vlim=vlim)
    axB = fig.add_subplot(gs[0, 1])
    panel_lattice(axB, fig, stages, s, vlim=vlim)
    axC = fig.add_subplot(gs[0, 2])
    panel_order_defects(axC, fig, stages, s)

    # Row 1: Radial profile (D), density residual (E), Rrows (F)
    axD = fig.add_subplot(gs[1, 0])
    panel_radial_profile(axD, stages, s)
    axE = fig.add_subplot(gs[1, 1])
    panel_density_residual(axE, fig, stages, s)
    axF = fig.add_subplot(gs[1, 2])
    panel_rows(axF, stages, s)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    n_final = len(stages['final'])
    n_src = len(stages['measurements'].source_points)
    fig.suptitle(f'Procedural ommatidial lattice '
                 f'(N={n_final} vs. source {n_src})',
                 fontsize=s.header + 2, y=0.965)

    # Draw once so title / tick-label extents are available for letter placement
    fig.canvas.draw()

    renderer = getattr(fig.canvas, 'get_renderer', lambda: None)()

    axes = (axA, axB, axC, axD, axE, axF)
    for letter, ax in zip('ABCDEF', axes):
        _panel_letter(fig, s, letter, ax, renderer,
                      clear_ticks=(letter in LETTERS_CLEAR_TICKS))

    return fig


## -----------------------------------------------------------------------

if __name__ == '__main__':


    raw_dirs = reconstruct_buchner_data(SVG)
    pts2d, *_ = sphere_to_stereo(raw_dirs)

    measurements = EyeMeasurements.from_points(
        points2d=pts2d,
        density_smoothing=DENSITY_SMOOTHING,
        axes_smoothing=AXES_SMOOTHING,
        min_hex_order=MIN_HEX_ORDER,
    )

    params = FittingParameters(density_scale=DENSITY_SCALE,
                               ghost_source=GHOST_SOURCE,
                               bypass_init=BYPASS_INIT)
    gen = LatticeGenerator(measurements, params)
    gen.run(align=ALIGN, verbose=True)

    m = gen.stages['measurements']
    n = len(m.source_points)

    print(f'Source: N={n} '
          f'mean_spacing={m.mean_spacing:.4f}  '
          f'lattice_angles={np.rad2deg(m.lattice_angles_rad).round(1)}deg')

    print(f'Generated: N={len(gen.stages["final"])} '
          f'(target {int(round(n * DENSITY_SCALE))})')

    settings = PlotSettings.nature_double(height_mm=125.0)

    fig = make_figure(gen.stages, settings)
    settings.savefig(fig, 'lattice_fitting_figure')

    plt.show()