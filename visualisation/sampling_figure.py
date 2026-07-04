from dataclasses import dataclass
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
    screen_dpi: int = 100              # on-screen dpi for the interactive viewer (NOT the export dpi)

    font_family: Sequence[str] = ('Arial', 'Helvetica', 'DejaVu Sans')
    base: float = 7.0                  # body text + tick labels
    small: float = 6.5                 # secondary annotations, legends
    tiny: float = 6.0                  # densest labels
    title: float = 8.0                 # panel titles
    header: float = 8.5                # column / row-group headers
    initials: float = 14               # Big letters for subfigures (A, B, C ...)

    axis_lw: float = 0.6
    curve_lw: float = 0.8
    grid_lw: float = 0.5

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
            'figure.dpi': self.screen_dpi,   # only interactive window, export dpi is set in savefig()
            'savefig.dpi': self.dpi,
            'figure.facecolor': self.bg,
            'savefig.facecolor': self.bg,
        })
        return self

    def new_figure(self) -> plt.Figure:
        return plt.figure(figsize=self.figsize, dpi=self.screen_dpi, facecolor=self.bg)

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
        """Nature single column: 89 mm."""
        return cls(width_mm=89.0, base=6.0, small=5.5, tiny=5.0,
                   title=7.0, header=7.0, dpi=600, **kw)

    @classmethod
    def plos(cls, **kw) -> 'PlotSettings':
        """PLOS: up to 190.5 mm, 8-12 pt type."""
        return cls(width_mm=183.0, base=8.0, small=7.0, tiny=7.0,
                   title=9.0, header=9.5, dpi=600, **kw)


# ===========================================================================

# Model / content config

NUM_RAYS_ROWS12 = 32
NUM_RAYS_ROW3 = 32
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

AIRY_RANGE = 6.0
AIRY_LUT = airy_sensitivity_lut(256, range=AIRY_RANGE)

# zorder threshold: everything below this goes into one rasterised layer per ax (EPS-safe translucency)
Z_RASTER = 8.0
Z_TEXT = 12.0   # text remains vector

# Line widths

target_lw = 1.0      # Target curve
proposal_lw = 1.5    # Proposal curve
ray_lw = 0.8         # Rays
guide_lw = 0.9       # FWHM markers, circles


def gaussian(radial_dist):
    r = np.asarray(radial_dist, dtype=float)
    return np.exp(-GAUSS_K * r * r)


def lookup_sensitivity_LUT(radial_dist):
    lut_x = np.linspace(0.0, 4.0, len(AIRY_LUT))
    return np.interp(np.asarray(radial_dist, dtype=float), lut_x, AIRY_LUT)


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


def radical_inverse_glsl(n_array):
    """Exact replica of the GLSL radical_inverse_v2 function in commons.glsl"""
    bits = np.asarray(n_array, dtype=np.uint32)
    bits = (bits << np.uint32(16)) | (bits >> np.uint32(16))
    bits = ((bits & np.uint32(0x55555555)) << np.uint32(1)) | ((bits & np.uint32(0xAAAAAAAA)) >> np.uint32(1))
    bits = ((bits & np.uint32(0x33333333)) << np.uint32(2)) | ((bits & np.uint32(0xCCCCCCCC)) >> np.uint32(2))
    bits = ((bits & np.uint32(0x0F0F0F0F)) << np.uint32(4)) | ((bits & np.uint32(0xF0F0F0F0)) >> np.uint32(4))
    bits = ((bits & np.uint32(0x00FF00FF)) << np.uint32(8)) | ((bits & np.uint32(0xFF00FF00)) >> np.uint32(8))
    return bits.astype(float) * 2.3283064365386963e-10


def reverse_bits_np(x):
    x = np.asarray(x, dtype=np.uint32)
    x = (x << np.uint32(16)) | (x >> np.uint32(16))
    x = ((x & np.uint32(0x00ff00ff)) << np.uint32(8)) | ((x & np.uint32(0xff00ff00)) >> np.uint32(8))
    x = ((x & np.uint32(0x0f0f0f0f)) << np.uint32(4)) | ((x & np.uint32(0xf0f0f0f0)) >> np.uint32(4))
    x = ((x & np.uint32(0x33333333)) << np.uint32(2)) | ((x & np.uint32(0xcccccccc)) >> np.uint32(2))
    x = ((x & np.uint32(0x55555555)) << np.uint32(1)) | ((x & np.uint32(0xaaaaaaaa)) >> np.uint32(1))
    return x


def sobol_dim1_np(i):
    i  = np.asarray(i, dtype=np.uint32)
    r  = np.zeros_like(i, dtype=np.uint32)
    ii = i.copy()
    v  = np.uint32(1 << 31)
    while np.any(ii):
        m = (ii & np.uint32(1)).astype(bool)
        r[m] ^= v
        ii >>= np.uint32(1)
        v  ^= v >> np.uint32(1)
    return r


def owen_scramble_np(v, seed):
    v, seed = np.asarray(v, dtype=np.uint32).copy(), np.uint32(seed)
    with np.errstate(over='ignore'):    # uint32 wraparound is intended
        v  = reverse_bits_np(v)
        v ^= v * np.uint32(0x3d20adea)
        v += seed
        v *= ((seed >> np.uint32(16)) | np.uint32(1))
        v ^= v * np.uint32(0x05526c56)
        v ^= v * np.uint32(0x53a22864)
    return reverse_bits_np(v)


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


def _reach(sigma, target_func):
    """How much of the total integrated sensor area is reached?"""
    r_max = sigma * np.sqrt(-np.log(1e-6) / GAUSS_K)
    r_grid = np.linspace(0, 6.0, 5000)

    # Area-weighted sensitivity: S(r) * r
    weighted_s = target_func(r_grid) * r_grid
    total = np.trapezoid(weighted_s, r_grid)
    captured = np.trapezoid(weighted_s[r_grid <= r_max], r_grid[r_grid <= r_max])
    return captured / total


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
    r = np.linspace(0.0, AIRY_RANGE, 20000)
    w = lookup_sensitivity_LUT(r) * r                               # 2D area element
    ring_mass = float(np.trapezoid(w[r > fz], r[r > fz]) / np.trapezoid(w, r))
    ray_reach = float(np.exp(-GAUSS_K * fz * fz))
    return fz, ring_mass, ray_reach


# ----------------------------------------------------------------------------------------------------------------------

def sampling_curves(ax, s: PlotSettings, target_func, target_name, mode, target_color, seed):

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)      # translucent -> raster layer (prevents eps from being massive)

    rng = np.random.default_rng(seed)
    x = np.linspace(-X_LIMIT, X_LIMIT, 1000)


    # Target sensitivity, S(theta)
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


    # Area-weighted sensitivity (= real information density), S(theta) * theta
    contribution = target_y * np.abs(x)
    if np.max(contribution) > 0:
        contribution /= (np.max(contribution) * 1.8)
    ax.plot(x, contribution, color=target_color, lw=s.curve_lw, linestyle='--', alpha=0.5, zorder=4)


    # Proposal distribution
    # ax.plot(x, samp_y, color=s.green, lw=proposal_lw, alpha=0.8, linestyle=':', solid_capstyle='round', zorder=4)
    ax.plot(x, samp_y, color=s.green, lw=proposal_lw, alpha=0.8, linestyle=(0, (1, 2)), solid_capstyle='round', zorder=4)
    if mode != 'Importance':
        ax.fill_between(x, samp_y, color=s.green, alpha=0.05, zorder=1)

    ax.text(1.72, 0.60, samp_label, color=s.green, fontsize=s.small, fontweight='bold', ha='center', zorder=Z_TEXT)


    # FWHM marker
    fwhm_half = 0.5
    ax.hlines(0.5, -fwhm_half, fwhm_half, color=s.dark, linestyle='--', lw=guide_lw, zorder=10)
    ax.text(0, 0.33, 'FWHM', color=s.dark, fontsize=s.small, fontweight='bold', ha='center', zorder=Z_TEXT)
    # TODO: Move FWHM text a little bit lower

    ax.text(-1.8, 0.13, f'Target\n$S(\\theta)$', color=target_color,
            fontsize=s.base, fontweight='bold', ha='center', zorder=Z_TEXT)

    if mode == 'Importance' and target_name == 'Airy':

        fz, ring_mass, ray_reach = airy_ring_stats()
        ring_mask = (np.abs(x) > fz) & (target_y > samp_y)

        ax.fill_between(x, samp_y, target_y, where=ring_mask, interpolate=True,
                        facecolor=s.red, alpha=0.16, hatch='////',
                        edgecolor=s.red, linewidth=0.0, zorder=6)

        peak_r = 5.136 / AIRY_SCALE  # first Airy secondary max, ~1.59 FWHM
        ax.annotate(f'Diffraction rings:\n{ring_mass:.0%} of $S(\\theta)$\n{ray_reach:.0%} of rays',
                    xy=(peak_r, target_func(peak_r)), xytext=(2.0, 0.2),
                    ha='center', va='bottom', color=s.red, style='italic', fontsize=s.small * 0.9, zorder=Z_TEXT,
                    arrowprops=dict(arrowstyle='-', color=s.red, lw=0.6))

    # Rays (alpha = contribution weight)
    rays_x, rays_y, weights = sample_rays(mode, target_func, NUM_RAYS_ROWS12, rng)

    wmax = weights.max() if weights.max() > 0 else 1.0
    w_norm = weights / wmax
    for rx, wn in zip(rays_x, w_norm):
        if -X_LIMIT < rx < X_LIMIT:
            x_bottom = rx * ((CONVERGENCE_DIST + UNDER_FLOOR) / (RAY_HEIGHT + CONVERGENCE_DIST))

            min_alpha = 0.001
            displ_weight = max(min_alpha, wn * 0.7)

            col = s.yellorange if displ_weight > min_alpha else 'grey'
            alpha = displ_weight if displ_weight > min_alpha else 0.2

            ax.plot([x_bottom, rx], [UNDER_FLOOR, RAY_HEIGHT],
                    color=col, alpha=alpha, lw=ray_lw, zorder=2)

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

    # FWHM
    axins.add_patch(
        Circle((0, 0), fwhm_half, color=s.dark, fill=False, linestyle='--', lw=guide_lw, zorder=6))

    # Main lobe extent
    axins.add_patch(
        Circle((0, 0), 1.1, color=s.dark, fill=False, linestyle=':', lw=guide_lw * 0.6, alpha=0.7, zorder=6))

    # Airy ripple
    if target_name == 'Airy':
        fz = 3.8317 / AIRY_SCALE
        axins.add_patch(Circle((0, 0), fz, color=s.red, fill=False, linestyle='--', lw=guide_lw * 0.6, zorder=6))

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

    rng = np.random.default_rng(42)
    zoom = 1.3
    fwhm_half = 0.5

    if mode_type == 'Stratified':
        grid = int(np.ceil(np.sqrt(NUM_RAYS_ROW3)))
        offset = int(rng.integers(0, grid * grid))
        cell = (np.arange(NUM_RAYS_ROW3) + offset) % (grid * grid)
        cell_x = cell % grid
        cell_y = cell // grid
        u1 = (cell_x + rng.random(NUM_RAYS_ROW3)) / grid
        u2 = (cell_y + rng.random(NUM_RAYS_ROW3)) / grid

    elif mode_type == 'Halton':
        u1 = np.clip(halton(NUM_RAYS_ROW3, 2), 1e-6, 1.0)
        u2 = halton(NUM_RAYS_ROW3, 3)

    elif mode_type == 'Hammersley':
        offset1 = rng.random()
        offset2 = rng.random()

        u1_base = (np.arange(NUM_RAYS_ROW3) + 0.5) / NUM_RAYS_ROW3
        u2_base = radical_inverse_glsl(np.arange(NUM_RAYS_ROW3))

        u1 = np.mod(u1_base + offset1, 1.0)
        u2 = np.mod(u2_base + offset2, 1.0)
        u1 = np.clip(u1, 1e-6, 1.0)

    elif mode_type == 'Sobol':
        idx = np.arange(NUM_RAYS_ROW3, dtype=np.uint32)
        INV = np.float64(2.3283064365386963e-10)
        u1 = np.clip(owen_scramble_np(reverse_bits_np(idx), 0x9E3779B9).astype(np.float64) * INV, 1e-6, 1.0)
        u2 = owen_scramble_np(sobol_dim1_np(idx), 0x85EBCA6B).astype(np.float64) * INV

    elif mode_type == 'Fibonacci':
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        rot_off = rng.random()
        rad_off = rng.random()
        u1 = np.mod((np.arange(NUM_RAYS_ROW3) + 0.5) / NUM_RAYS_ROW3 + rad_off, 1.0)
        u2 = np.mod((np.arange(NUM_RAYS_ROW3) * golden_angle / (2 * np.pi)) + rot_off, 1.0)
        u1 = np.clip(1.0 - u1, 1e-6, 1.0)

    else: # 'Pseudo-random'
        u1 = rng.random(NUM_RAYS_ROW3)
        u2 = rng.random(NUM_RAYS_ROW3)

    # Convert to radial coordinates (Gaussian importance sample)
    radii = np.sqrt(-np.log(np.clip(u1, 1e-6, 1.0)) / GAUSS_K)
    rx = radii * np.cos(2 * np.pi * u2)
    ry = radii * np.sin(2 * np.pi * u2)

    # Calculate sensitivity value at that point
    # This is S(theta). higher = more useful sample
    utility = np.exp(-GAUSS_K * radii ** 2)
    c_map = LinearSegmentedColormap.from_list('utility', [s.dark, s.yellorange])

    ax.add_patch(
        Circle((0, 0), fwhm_half, color=s.dark, fill=False, linestyle='--', lw=guide_lw, zorder=6))

    ax.add_patch(
        Circle((0, 0), 1.1, color=s.dark, fill=False, linestyle=':', lw=guide_lw * 0.6, alpha=0.7, zorder=6))

    sc = ax.scatter(rx, ry, s=22, c=utility, cmap=c_map, vmin=0, vmax=1,
                    alpha=0.9, edgecolors='white', lw=0.4, zorder=4)

    ax.set_title(mode_type, fontsize=s.title, pad=5)

    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)

    ax.set_aspect('equal')

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(s.frame)
        spine.set_linewidth(1.0)

    return sc

# Efficiency-vs-coverage scatter

HYBRID_SIGMAS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

def efficiency_coverage_scatter(ax, s: PlotSettings):

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)    # data marks into raster layer

    targets = [(gaussian, 'Gaussian', s.blue), (lookup_sensitivity_LUT, 'Airy', s.red)]
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
    cov_a = [100 * _coverage(_gauss_proposal(v), lookup_sensitivity_LUT) for v in (HYBRID_SIGMAS[0], HYBRID_SIGMAS[-1])]
    eff_a = [100 * hybrid_efficiency(v, lookup_sensitivity_LUT) for v in (HYBRID_SIGMAS[0], HYBRID_SIGMAS[-1])]

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
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.15)
    leg.get_frame().set_facecolor('white')

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
    targets = [(gaussian, 'Gaussian', s.blue), (lookup_sensitivity_LUT, 'Airy', s.red)]

    fig = s.new_figure()

    outer = GridSpec(2, 1, figure=fig, height_ratios=[2.0, 0.92], hspace=0.135)
    # both sampling rows in one 2x3 grid with hspace=0.0)
    top = outer[0].subgridspec(2, 3, wspace=0.05, hspace=0.0)

    # Rows 0 and 1: target x strategy, each row is 3-column wide
    sampling_axes = []
    for row, (tfunc, tname, tcolor) in enumerate(targets):
        row_axes = []
        for col, strat in enumerate(STRATEGIES):
            ax = fig.add_subplot(top[row, col])
            sampling_curves(ax, s, tfunc, tname, strat, tcolor, seed=100 * row + col)
            row_axes.append(ax)
        sampling_axes.append(row_axes)

    # Row 2: Randomness modes cluster (left, 2x3), efficiency scatter (right)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.40)
    rand_gs = bottom[0].subgridspec(2, 3, wspace=-0.15, hspace=0.35)

    ax_pseudo = fig.add_subplot(rand_gs[0, 0])
    ax_strat = fig.add_subplot(rand_gs[0, 1])
    ax_fib = fig.add_subplot(rand_gs[0, 2])
    ax_halton = fig.add_subplot(rand_gs[1, 0])
    ax_hammer = fig.add_subplot(rand_gs[1, 1])
    ax_sobol = fig.add_subplot(rand_gs[1, 2])

    sc_obj = randomness_mode_scatter(ax_pseudo, s, 'Pseudo-random')
    randomness_mode_scatter(ax_strat, s, 'Stratified')
    randomness_mode_scatter(ax_fib, s, 'Fibonacci')
    randomness_mode_scatter(ax_halton, s, 'Halton')
    randomness_mode_scatter(ax_hammer, s, 'Hammersley')
    randomness_mode_scatter(ax_sobol, s, 'Sobol')

    p_top_right = ax_fib.get_position(fig)
    p_bot_right = ax_sobol.get_position(fig)

    cb_ax = fig.add_axes([p_top_right.x1 - 0.01, p_bot_right.y0 - 0.02, 0.008, p_top_right.y1 - p_bot_right.y0])
    cb = fig.colorbar(sc_obj, cax=cb_ax, orientation='vertical')
    cb.set_label('Relative Sensitivity $S(\\theta)$', fontsize=s.tiny, labelpad=4)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=s.tiny, width=0.5, length=2)

    p_mid_bot = ax_hammer.get_position(fig)

    fig.text((p_mid_bot.x0 + p_mid_bot.x1) / 2.0 - 0.03, p_mid_bot.y0 - 0.05,  # TODO: why - 0.03??
             'Stochastic distributions', fontsize=s.title, ha='center', va='top', fontweight='bold')

    ax_eff = fig.add_subplot(bottom[1])
    efficiency_coverage_scatter(ax_eff, s)

    # Margins fixed explicitly  # TODO: Adjust these
    fig.subplots_adjust(left=0.06, right=0.97, top=0.905, bottom=0.07)

    # Headers and group labels
    for ax, strat in zip(sampling_axes[0], STRATEGIES):     # column headers
        p = ax.get_position()
        fig.text((p.x0 + p.x1) / 2, p.y1 + 0.014, strat,
                 ha='center', va='bottom', fontsize=s.header)

    for row_axes, label, col in ((sampling_axes[0], 'Gaussian target', s.blue),
                                 (sampling_axes[1], 'Airy target', s.red)):
        p = row_axes[0].get_position()
        fig.text(p.x0 - 0.018, (p.y0 + p.y1) / 2, label, rotation=90,
                 va='center', ha='center', fontweight='bold', fontsize=s.header, color=col)

    a_pos = sampling_axes[0][0].get_position()
    fig.text(a_pos.x0, a_pos.y1 + 0.02, 'A', ha='right', va='bottom',
             fontsize=s.initials, fontweight='bold', color='black')

    b_pos = bottom[0].get_position(fig)
    fig.text(b_pos.x0, b_pos.y1, 'B', ha='right', va='bottom',
             fontsize=s.initials, fontweight='bold', color='black')

    c_pos = bottom[1].get_position(fig)
    fig.text(c_pos.x0 - 0.055, c_pos.y1, 'C', ha='right', va='bottom',
             fontsize=s.initials, fontweight='bold', color='black')

    return fig


if __name__ == '__main__':
    settings = PlotSettings.nature_double().apply()

    fig = build_figure(settings)

    settings.savefig(fig, 'sampling', formats=['svg', 'eps', 'pdf'])
