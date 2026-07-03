from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from insectvision.utils import airy_sensitivity_lut


MM_PER_IN = 25.4


# TODO: Move this dataclass somewhere else and use it for other figures

@dataclass
class PlotSettings:

    width_mm: float = 183.0            # Nature double column (PLOS max 190.5)
    height_mm: Optional[float] = None  # if None, derived from 'aspect'
    aspect: float = 0.95               # height / width when height_mm is None
    dpi: int = 600                     # raster image export (PNG/TIFF)
    raster_dpi: int = 300              # resolution of rasterised layers inside EPS/PDF (halftone min)

    font_family: Sequence[str] = ('Arial', 'Helvetica', 'DejaVu Sans')
    base: float = 7.0                  # body text + tick labels
    small: float = 6.5                 # secondary annotations, legends
    tiny: float = 6.0                  # densest labels
    title: float = 8.0                 # panel titles
    header: float = 8.5                # column / row-group headers
    initials: float = 14               # Big letters for subfigures (A, B, C ...)

    axis_lw = 0.6
    curve_lw = 0.8
    grid_lw = 0.5

    bg: str = 'white'
    yellorange: str = '#FFC32F'
    green: str = '#27AE60'
    blue: str = '#2980B9'
    red: str = '#C0392B'
    dark: str = '#34495E'
    frame: str = '#7F8C8D'
    grid: str = '#EAECEE'

    formats: Sequence[str] = ('eps', 'pdf', 'png')
    rasterize: bool = True

    @property
    def width_in(self) -> float:
        return self.width_mm / MM_PER_IN

    @property
    def height_in(self) -> float:
        if self.height_mm is not None:
            return self.height_mm / MM_PER_IN
        return self.width_in * self.aspect

    @property
    def figsize(self):
        return (self.width_in, self.height_in)

    def apply(self) -> 'PlotSettings':
        import shutil
        import platform
        if 'Windows' in platform.system():
            gs_bin = 'gswin64c'
        else:
            gs_bin = 'gs'
        if shutil.which(gs_bin):
            plt.rcParams['ps.usedistiller'] = 'ghostscript'

        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': list(self.font_family),
            'font.size': self.base,
            'axes.titlesize': self.title,
            'axes.titleweight': 'normal',
            'axes.labelsize': self.base,
            'xtick.labelsize': self.base,
            'ytick.labelsize': self.base,
            'legend.fontsize': self.small,
            'axes.linewidth': self.axis_lw,
            'xtick.major.width': self.axis_lw,
            'ytick.major.width': self.axis_lw,
            'lines.linewidth': self.curve_lw,
            'mathtext.fontset': 'dejavusans',
            'axes.unicode_minus': True,
            'svg.fonttype': 'none',
            'pdf.fonttype': 42,   # embed TrueType in PDF
            'ps.fonttype': 42,    # embed TrueType in EPS/PS
            'ps.useafm': False,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            'figure.facecolor': self.bg,
            'savefig.facecolor': self.bg,
        })
        return self

    def new_figure(self) -> plt.Figure:
        return plt.figure(figsize=self.figsize, dpi=self.dpi, facecolor=self.bg)

    def savefig(self, fig: plt.Figure, name: str, formats: Optional[Sequence[str]] = None):
        raster_exts = {'png', 'tif', 'tiff', 'jpg', 'jpeg'}
        for ext in (formats or self.formats):
            dpi = self.dpi if ext.lower() in raster_exts else self.raster_dpi
            fig.savefig(f'{name}.{ext}', format=ext, dpi=dpi, facecolor=self.bg)

    @classmethod
    def nature_double(cls, **kw) -> 'PlotSettings':
        """Nature double column: 183 mm, 5-7 pt type."""
        return cls(width_mm=183.0, base=7.0, small=6.5, tiny=6.0,
                   title=8.0, header=8.5, dpi=600, **kw)

    @classmethod
    def nature_single(cls, **kw) -> 'PlotSettings':
        """Nature single column: 89 mm (dense, this figure prefers double)."""
        return cls(width_mm=89.0, base=6.0, small=5.5, tiny=5.0,
                   title=7.0, header=7.0, dpi=600, **kw)

    @classmethod
    def plos(cls, **kw) -> 'PlotSettings':
        """PLOS: up to 190.5 mm, 8-12 pt type, >=300 dpi."""
        return cls(width_mm=183.0, base=8.0, small=7.0, tiny=7.0,
                   title=9.0, header=9.5, dpi=600, **kw)

    def variant(self, **kw) -> 'PlotSettings':
        """A copy with fields overridden (e.g. a wider or taller sibling figure)."""
        return replace(self, **kw)


# ===========================================================================

# Model / content config

NUM_RAYS = 55
RAY_HEIGHT = 1.35
X_LIMIT = 2.5
UNDER_FLOOR = -0.2
CONVERGENCE_DIST = 4.5

UNIFORM_RADIUS = 1.8       # naive-baseline disk radius (FWHM units)
UNIFORM_PROP_LEVEL = 0.9   # height of the flat "uniform p(theta)" schematic line

# Constants pulled verbatim from commons.glsl / utils.py
GAUSS_K = 2.77258872224    # GAUSS_CONSTANT_K = 4 * log(2)
AIRY_SCALE = 3.232         # makes the Airy FWHM = 1.0
SPREAD_MULT = 2.0          # proposal = spread_mult * acceptance
AIRY_LUT_SIZE = 256

AIRY_LUT = airy_sensitivity_lut(AIRY_LUT_SIZE)

# zorder threshold: everything below this goes into one rasterised layer per ax (EPS-safe translucency)
Z_RASTER = 8.0
Z_TEXT = 12.0   # text remains vector

# Line widths

target_lw = 1.5      # Target curve
proposal_lw = 1.0    # Proposal curve
ray_lw = 0.8         # Rays
guide_lw = 0.9       # FWHM markers, circles


def gaussian(radial_dist):
    r = np.asarray(radial_dist, dtype=float)
    return np.exp(-GAUSS_K * r * r)


def airy(radial_dist):
    idx = np.clip(np.asarray(radial_dist, dtype=float) * (255.0 / 4.0), 0.0, 255.0).astype(np.int64)
    return AIRY_LUT[idx]


def halton(n, base):
    res = []
    for i in range(1, n + 1):
        f, r = 1.0, 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        res.append(r)
    return np.array(res)


# ---------------------------------------------------------------------------
# Ray sampling, all in FWHM units (acc = 1)

def sample_rays(mode, target_func, n, rng):
    u1 = rng.uniform(1e-6, 1.0, n)
    u2 = rng.uniform(0.0, 1.0, n)
    phi = 2.0 * np.pi * u2

    if mode == 'Importance':
        r = np.sqrt(-np.log(u1) / GAUSS_K)          # Gaussian iCDF, acc = 1
        rays_x, rays_y = r * np.cos(phi), r * np.sin(phi)
        weights = np.ones(n)                         # weight hard-coded to 1 in the shader

    elif mode == 'Hybrid':
        sample_sigma = SPREAD_MULT                  # acc * spread_mult, acc = 1
        r = sample_sigma * np.sqrt(-np.log(u1) / GAUSS_K)
        rays_x, rays_y = r * np.cos(phi), r * np.sin(phi)
        rdist = np.sqrt(rays_x ** 2 + rays_y ** 2)
        pdf = np.exp(-GAUSS_K * (rdist / sample_sigma) ** 2)
        weights = target_func(rdist) / np.maximum(pdf, 1e-6)

    else:  # 'Uniform'
        r = UNIFORM_RADIUS * np.sqrt(u1)
        rays_x, rays_y = r * np.cos(phi), r * np.sin(phi)
        weights = target_func(np.sqrt(rays_x ** 2 + rays_y ** 2))

    return rays_x, rays_y, weights


# ---------------------------------------------------------------------------
# Efficiency vs. coverage for the scatter
#
#   efficiency = ESS/N = (int S)^2 / (Z * int S^2/g) under proposal density g
#   coverage = overlapping coefficient (1 - total variation) of proposal, target

_U1_MIN = 1e-6                                   # proposal clamp: r_max = sigma*sqrt(-log(u1)/K)
_COV_R = np.linspace(1e-4, 4.0, 12000)           # radial grid for the overlap metric


def _rmax(sigma):
    return sigma * np.sqrt(-np.log(_U1_MIN) / GAUSS_K)


def _radial_integral(f, R, n=12000):
    r = np.linspace(0.0, R, n)
    return np.trapezoid(f(r) * 2 * np.pi * r, r)


def hybrid_efficiency(sigma, target_func):
    """ESS/N for the hybrid estimator with a Gaussian proposal of width sigma."""
    g = lambda r: np.exp(-GAUSS_K * (r / sigma) ** 2)
    Rm = _rmax(sigma)
    A = _radial_integral(target_func, Rm)
    Z = _radial_integral(g, Rm)
    B = _radial_integral(lambda r: target_func(r) ** 2 / np.maximum(g(r), 1e-300), Rm)
    return A * A / (Z * B)


def uniform_efficiency(target_func):
    A = _radial_integral(target_func, UNIFORM_RADIUS)
    Z = _radial_integral(lambda r: np.ones_like(r), UNIFORM_RADIUS)
    B = _radial_integral(lambda r: target_func(r) ** 2, UNIFORM_RADIUS)
    return A * A / (Z * B)


def _coverage(proposal_density, target_func):
    """Overlapping coefficient of the normalised proposal and target densities, in [0, 1]."""
    dA = 2 * np.pi * _COV_R
    q = proposal_density(_COV_R) * dA
    s = target_func(_COV_R) * dA
    q = q / np.trapezoid(q, _COV_R)
    s = s / np.trapezoid(s, _COV_R)
    return np.trapezoid(np.minimum(q, s), _COV_R)


def _gauss_proposal(sigma):
    def q(r):
        g = np.exp(-GAUSS_K * (r / sigma) ** 2)
        return np.where(r > _rmax(sigma), 0.0, g)
    return q


def _uniform_proposal(r):
    return np.where(r <= UNIFORM_RADIUS, 1.0, 0.0)


def airy_ring_stats():
    """
    Description of what pure importance misses on an Airy target.
    Returns (first_dark_ring, ring_sensitivity_mass, gaussian_ray_reach), all in
    FWHM-normalised units.

    ring_mass = 2D area-weighted sensitivity beyond the first dark ring
    ray_reach = P(Gaussian-iCDF radius > first ring) = exp(-K z^2)
    """
    fz = 3.8317 / AIRY_SCALE                      # first zero of J1 -> first Airy dark ring
    r = np.linspace(0.0, 4.0, 20000)
    w = airy(r) * r                               # 2D area element
    ring_mass = float(np.trapezoid(w[r > fz], r[r > fz]) / np.trapezoid(w, r))
    ray_reach = float(np.exp(-GAUSS_K * fz * fz))
    return fz, ring_mass, ray_reach


# ----------------------------------------------------------------------------------------------------------------------

def sampling_curves(ax, s: PlotSettings, target_func, target_name, mode, target_color, seed):

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)      # translucent -> raster layer (prevents eps from being massive)

    rng = np.random.default_rng(seed)
    x = np.linspace(-X_LIMIT, X_LIMIT, 1000)


    # Target sensitivity
    target_y = target_func(np.abs(x))
    ax.plot(x, target_y, color=target_color, lw=target_lw, zorder=5)
    ax.fill_between(x, target_y, color=target_color, alpha=0.10, zorder=1)


    # Samples distribution
    if mode == 'Importance':
        samp_y = np.exp(-GAUSS_K * x ** 2)
        samp_label = 'Gaussian\n$p(\\theta)$'
    elif mode == 'Hybrid':
        samp_y = np.exp(-GAUSS_K * (x / SPREAD_MULT) ** 2)
        samp_label = f'Proposal\n$p(\\theta)$, {SPREAD_MULT:g}$\\times$'
    else:
        samp_y = np.full_like(x, UNIFORM_PROP_LEVEL)
        samp_label = 'Uniform\n$p(\\theta)$'


    # Proposal distribution
    ax.plot(x, samp_y, color=s.green, lw=proposal_lw, alpha=0.8, linestyle=(0, (1, 2)), zorder=4)
    if mode != 'Importance':
        ax.fill_between(x, samp_y, color=s.green, alpha=0.05, zorder=1)

    ax.text(1.72, 0.60, samp_label, color=s.green, fontsize=s.small, fontweight='bold', ha='center', zorder=Z_TEXT)


    # FWHM marker
    fwhm_half = 0.5
    ax.hlines(0.5, -fwhm_half, fwhm_half, color=s.dark, linestyle='--', lw=guide_lw, zorder=10)
    ax.text(0, 0.40, 'FWHM', color=s.dark, fontsize=s.small, fontweight='bold', ha='center', zorder=Z_TEXT)
    # TODO: Move FWHM text a little bit lower

    ax.text(-1.8, 0.13, f'{target_name} target\n$S(\\theta)$', color=target_color,
            fontsize=s.base, fontweight='bold', ha='center', zorder=Z_TEXT)

    if mode == 'Importance' and target_name == 'Airy':

        fz, ring_mass, ray_reach = airy_ring_stats()
        ring_mask = (np.abs(x) > fz) & (target_y > samp_y)

        ax.fill_between(x, samp_y, target_y, where=ring_mask, interpolate=True,
                        facecolor=s.red, alpha=0.16, hatch='////',
                        edgecolor=s.red, linewidth=0.0, zorder=6)

        ax.annotate(f'Diffraction rings:\n{ring_mass:.0%} of $S(\\theta)$, '
                    f'{ray_reach:.0%} of rays',
                    xy=(1.5, max(target_func(1.5), 0.02)), xytext=(0.0, 1.17),
                    color=s.red, fontsize=s.small, ha='center', va='center', zorder=Z_TEXT,
                    arrowprops=dict(arrowstyle='-', color=s.red, lw=0.6))
        # TODO: Position of this is lame, should be lower right (above the rightmost ripple)


    # Rays (alpha = contribution weight)
    rays_x, rays_y, weights = sample_rays(mode, target_func, NUM_RAYS, rng)

    wmax = weights.max() if weights.max() > 0 else 1.0
    w_norm = weights / wmax
    for rx, wn in zip(rays_x, w_norm):
        if -X_LIMIT < rx < X_LIMIT:
            x_bottom = rx * ((CONVERGENCE_DIST + UNDER_FLOOR) / (RAY_HEIGHT + CONVERGENCE_DIST))
            ax.plot([x_bottom, rx], [UNDER_FLOOR, RAY_HEIGHT],
                    color=s.yellorange, alpha=max(0.03, wn * 0.7), lw=ray_lw, zorder=2)

    # Insets: 2D sample cloud over the target heatmap
    axins = ax.inset_axes([0.02, 0.63, 0.3, 0.3])
    axins.set_zorder(Z_RASTER + 1)                 # inset manages its own raster layer
    if s.rasterize:
        axins.set_rasterization_zorder(Z_RASTER)

    zoom_lim = 1.8
    side = np.linspace(-zoom_lim, zoom_lim, 100)
    X, Y = np.meshgrid(side, side)

    cmap = LinearSegmentedColormap.from_list('c', ['white', target_color])

    axins.imshow(target_func(np.sqrt(X ** 2 + Y ** 2)),
        extent=[-zoom_lim, zoom_lim, -zoom_lim, zoom_lim],
        origin='lower', cmap=cmap, alpha=0.3, zorder=1)

    axins.add_patch(
        Circle((0, 0), fwhm_half, color=s.dark, fill=False, linestyle='--', lw=guide_lw, zorder=6))
    # TODO: Add a thinner, larger ring at the bottom of the curve, and another one at the first ripple for Airy

    axins.scatter(rays_x, rays_y,
                  s=20, color=s.yellorange, alpha=np.clip(w_norm, 0.06, 0.65), edgecolors='white', lw=0.4, zorder=5)

    axins.set_xlim(-zoom_lim, zoom_lim)
    axins.set_ylim(-zoom_lim, zoom_lim)

    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor(s.frame)
        spine.set_linewidth(0.8)

    ax.set_ylim(UNDER_FLOOR - 0.1, RAY_HEIGHT + 0.3)
    ax.set_xlim(-X_LIMIT, X_LIMIT)
    ax.axis('off')


def randomness_mode_scatter(ax, s: PlotSettings, mode_type):

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)

    rng = np.random.default_rng(7)
    n = 64
    zoom = 1.1
    fwhm_half = 0.5

    if mode_type == 'Pseudo-random':
        u1 = rng.uniform(1e-6, 1.0, n)
        u2 = rng.uniform(0.0, 1.0, n)
        r = np.sqrt(-np.log(u1) / GAUSS_K)
        rx, ry = r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)

    elif mode_type == 'Quasi-random (Halton)':
        u1 = np.clip(halton(n, 2), 1e-6, 1.0)
        u2 = halton(n, 3)
        r = np.sqrt(-np.log(u1) / GAUSS_K)
        rx, ry = r * np.cos(2 * np.pi * u2), r * np.sin(2 * np.pi * u2)

    else:  # Stratified
        grid = int(np.ceil(np.sqrt(n)))
        rx, ry = [], []
        for i in range(n):
            u1 = (i % grid + rng.random()) / grid
            u2 = (i // grid + rng.random()) / grid
            r = np.sqrt(-np.log(max(u1, 1e-6)) / GAUSS_K)
            rx.append(r * np.cos(2 * np.pi * u2))
            ry.append(r * np.sin(2 * np.pi * u2))
        rx, ry = np.array(rx), np.array(ry)

    radii = np.sqrt(rx ** 2 + ry ** 2)

    c_map = LinearSegmentedColormap.from_list('rd', ['#1B2631', s.yellorange])

    ax.add_patch(Circle((0, 0), fwhm_half, color=s.dark, fill=False,
                        linestyle='--', lw=guide_lw, zorder=3))

    ax.scatter(rx, ry, s=22, c=c_map(np.clip(radii / 1.5, 0, 1)),
               alpha=0.85, edgecolors='white', lw=0.4, zorder=4)

    ax.set_title(mode_type, fontsize=s.title, pad=5)

    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)

    ax.set_aspect('equal')

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(s.frame)
        spine.set_linewidth(1.0)


# Efficiency-vs-coverage scatter

HYBRID_SIGMAS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

def efficiency_coverage_scatter(ax, s: PlotSettings):

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)    # data marks into raster layer

    targets = [(gaussian, 'Gaussian', s.blue), (airy, 'Airy', s.red)]
    s_lo, s_hi = min(HYBRID_SIGMAS), max(HYBRID_SIGMAS)
    alpha_of = lambda v: 0.30 + 0.70 * (v - s_lo) / (s_hi - s_lo)

    for tf, tname, col in targets:
        cov = [100 * _coverage(_gauss_proposal(v), tf) for v in HYBRID_SIGMAS]
        eff = [100 * hybrid_efficiency(v, tf) for v in HYBRID_SIGMAS]
        ax.plot(cov, eff, color=col, lw=s.curve_lw, alpha=0.3, zorder=2)           # sigma trajectory

        for v, x, y in zip(HYBRID_SIGMAS, cov, eff):
            ax.scatter(x, y, marker='o', s=26, facecolor=col, edgecolor='white',
                       linewidth=0.5, alpha=alpha_of(v), zorder=4)

        i2 = HYBRID_SIGMAS.index(2.0)                                             # default spread
        ax.scatter(cov[i2], eff[i2], marker='o', s=78, facecolor='none',
                   edgecolor=col, linewidth=1.0, zorder=5)

        ax.scatter(100 * _coverage(_uniform_proposal, tf), 100 * uniform_efficiency(tf),
                   marker='s', s=32, facecolor=col, edgecolor='white', linewidth=0.5, zorder=4)

        ax.scatter(100 * _coverage(_gauss_proposal(1.0), tf), 100.0,
                   marker='^', s=40, facecolor=col, edgecolor='white', linewidth=0.5, zorder=4)

    # endpoint sigma labels on the Airy hook (the informative trajectory)
    cov_a = [100 * _coverage(_gauss_proposal(v), airy) for v in (HYBRID_SIGMAS[0], HYBRID_SIGMAS[-1])]
    eff_a = [100 * hybrid_efficiency(v, airy) for v in (HYBRID_SIGMAS[0], HYBRID_SIGMAS[-1])]

    ax.annotate(r'$\sigma$=1', (cov_a[0], eff_a[0]), xytext=(4, -8), textcoords='offset points',
                fontsize=s.tiny, color=s.red, zorder=Z_TEXT)

    ax.annotate(r'$\sigma$=3', (cov_a[1], eff_a[1]), xytext=(-16, -8), textcoords='offset points',
                fontsize=s.tiny, color=s.red, zorder=Z_TEXT)

    ax.text(70, 85, 'Gaussian', color=s.blue, fontsize=s.base, ha='center', va='center', zorder=Z_TEXT)
    ax.text(65, 40, 'Airy', color=s.red, fontsize=s.base, ha='center', va='center', zorder=Z_TEXT)

    strat = [('Uniform', 's'), ('Importance', '^'), ('Hybrid', 'o')]
    handles = [Line2D([0], [0], marker=m, linestyle='none', markerfacecolor=s.frame,
                      markeredgecolor='white', markersize=5.5, label=lab) for lab, m in strat]

    leg = ax.legend(handles=handles, loc='upper left', fontsize=s.small,
                    handletextpad=0.3, borderpad=0.2, labelspacing=0.25,
                    title='Strategy', title_fontsize=s.small)
    # TODO: Thin black legend border

    leg.set_zorder(Z_TEXT)
    ax.add_artist(leg)

    ax.text(0.99, 0.03, r'opacity $\propto\ \sigma$' + '\n' + r'ring: $\sigma{=}2$',
            transform=ax.transAxes, fontsize=s.tiny, color=s.frame, ha='right', va='bottom',
            zorder=Z_TEXT)

    ax.set_xlim(25, 106)
    ax.set_ylim(-19, 109)

    ax.set_xlabel(r'Coverage: proposal $\cap$ target (%)', fontsize=s.base)
    ax.set_ylabel('Sampling efficiency ESS/N (%)', fontsize=s.base)

    ax.set_title('Efficiency vs. coverage', fontsize=s.title, loc='left')

    ax.tick_params(labelsize=s.base)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    ax.grid(color=s.grid, lw=s.grid_lw, zorder=0)
    ax.set_axisbelow(True)


# ===========================================================================

STRATEGIES = ['Uniform', 'Importance', 'Hybrid']
TARGETS_KEY = ['gaussian', 'airy']


def build_figure(s: PlotSettings) -> plt.Figure:
    targets = [(gaussian, 'Gaussian', s.blue), (airy, 'Airy', s.red)]

    fig = s.new_figure()

    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.0, 0.92], hspace=0.30)

    # Rows 0 and 1: target x strategy, each row is 3-column wide
    sampling_axes = []
    for row, (tfunc, tname, tcolor) in enumerate(targets):
        row_gs = outer[row].subgridspec(1, 3, wspace=0.12)
        row_axes = []

        for col, strat in enumerate(STRATEGIES):
            ax = fig.add_subplot(row_gs[0, col])
            sampling_curves(ax, s, tfunc, tname, strat, tcolor, seed=100 * row + col)
            row_axes.append(ax)

        sampling_axes.append(row_axes)

    # Row 2: Randomness modes clusters (left), efficiency scatter (right)
    bottom = outer[2].subgridspec(1, 2, width_ratios=[1.0, 1.18], wspace=0.16)
    rand_gs = bottom[0].subgridspec(2, 2, wspace=0.10, hspace=0.32)

    ax_pseudo = fig.add_subplot(rand_gs[0, 0])
    ax_halton = fig.add_subplot(rand_gs[0, 1])
    ax_strat = fig.add_subplot(rand_gs[1, :])       # centred under the two

    randomness_mode_scatter(ax_pseudo, s, 'Pseudo-random')
    randomness_mode_scatter(ax_halton, s, 'Quasi-random (Halton)')
    randomness_mode_scatter(ax_strat, s, 'Stratified')

    ax_eff = fig.add_subplot(bottom[1])
    efficiency_coverage_scatter(ax_eff, s)

    # Margins fixed explicitly  # TODO: Adjust these
    fig.subplots_adjust(left=0.055, right=0.985, top=0.925, bottom=0.045)

    # Headers and group labels
    for ax, strat in zip(sampling_axes[0], STRATEGIES):     # column headers
        p = ax.get_position()
        fig.text((p.x0 + p.x1) / 2, p.y1 + 0.014, strat,
                 ha='center', va='bottom', fontsize=s.header)

    for row_axes, label, col in ((sampling_axes[0], 'Gaussian target', s.blue),
                                 (sampling_axes[1], 'Airy target', s.red)):
        p = row_axes[0].get_position()
        fig.text(p.x0 - 0.018, (p.y0 + p.y1) / 2, label, rotation=90,
                 va='center', ha='center', fontsize=s.header, color=col)

    rand_reg = bottom[0].get_position(fig)   # randomness modes region
    fig.text(x=rand_reg.x0, y=rand_reg.y1, s='B', ha='right', va='bottom', fontsize=s.initials, fontweight='bold', color='black')

    return fig


# TODO: Add A and B initials

if __name__ == '__main__':
    settings = PlotSettings.nature_double().apply()

    fig = build_figure(settings)
    # TODO: interactive viewer is borked, 600 dpi is too big for it, and point-based fonts cant be resized in it

    settings.savefig(fig, 'sampling', formats=['png'])
    # TODO: Margins are crap, need adjusted