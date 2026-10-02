from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from rhabdoforge.compound_eyes.rhabdomeres import drosophila_bundle
from rhabdoforge.compound_eyes.helpers.acceptance import RhabdomereOptics
from rhabdoforge.compound_eyes.helpers.waveguide import solve_modes
from rhabdoforge.LUTs import LUT_RANGE, LUT_SIZE, N_PUPIL_STEPS, N_SACCADE_STEPS

from visualisation.plot_settings import PlotSettings, Z_RASTER, Z_TEXT, panel_letter, column_header, row_header


# Config (all angles in FWHM units)

NUM_RAYS = 32
RAY_HEIGHT = 1.35
X_LIMIT = 2.5
UNDER_FLOOR = -0.2
U1_MIN = 1e-6                # proposal clamp (as in the shader)

UNIFORM_RADIUS = 1.8         # naive-baseline disk radius
UNIFORM_LEVEL = 0.9          # height of the flat uniform p(theta) line

# From commons.glsl
GAUSS_K = 4 * np.log(2)
PROPOSAL_WIDEN = 1.7
MAX_LP_MODES = 2

WAVE_F_NUMBER = 2.0
TYPE_NAMES = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7/8']
MARGINS = np.linspace(0.7, 3.0, 61)
SWEEP_MARGINS = [1.2, 1.4, 1.7, 2.0, 2.4, 3.0]   # below ~1.2 ESS/N is non-monotonic

bundle = drosophila_bundle()
CONVERGENCE_DIST = bundle.focal_um / bundle.diameters_um[0] * (X_LIMIT / 2.0)   # fits the cone to the plot

target_lw, proposal_lw, ray_lw, guide_lw = 1.0, 1.5, 0.8, 0.9


# Mode mixtures

class Mix(NamedTuple):
    w0: float          # weight of mode 0
    a0: float          # mode widths
    a1: float
    px: np.ndarray     # profile x-axis
    p0: np.ndarray     # peak-normalised mode profiles
    p1: np.ndarray


def mix_from_modes(m) -> Mix:
    nb = min(m.nb_modes, MAX_LP_MODES)
    i0, i1 = 0, (1 if nb > 1 else 0)
    k0, k1 = m.get_mode_peak(i0), m.get_mode_peak(i1)
    return Mix(float(k0 / (k0 + k1)), m.get_mode_hwhm(i0) / m.d_half, m.get_mode_hwhm(i1) / m.d_half,
               0.5 * m.d_sweep / m.d_half, m.mode_sensitivity[i0] / k0, m.mode_sensitivity[i1] / k1)


optics = RhabdomereOptics(
    diameter_um=bundle.diameters_um,
    wavelength_um=bundle.wavelengths_nm * 1e-3,
    n_rhabdomere=bundle.n_rhabdomere,
    n_surround=bundle.n_surround,
)


def solve(t, drive=0.0, frac=0.0):
    """Modes of type t at a (pupil drive, saccade fraction) state, drive=0 (dark) -> h=inf."""
    return solve_modes(
        v_number=float(optics.v_number[t]), f_number=WAVE_F_NUMBER,
        diameter_um=float(optics.diameter_um[t]), wavelength_um=float(optics.wavelength_um[t]),
        h_um=np.inf if drive <= 0.0 else bundle.pupil_distance_um / drive,
        defocus_um=bundle.tip_defocus_um + frac * bundle.ampl_ax_um, focal_um=bundle.focal_um,
    )


# R1 at rest (V = 3.23, 2 modes): waveguide target and mixture sampler illustration
R1_MODES = solve(0)
R1 = mix_from_modes(R1_MODES)
WAVE_LUT = np.interp(np.linspace(0.0, LUT_RANGE, LUT_SIZE), R1.px, R1_MODES.sensitivity)


def mix_target(x, mix=R1):
    x = np.abs(x)
    f = lambda p: np.interp(x, mix.px, p, left=0.0, right=0.0)
    return mix.w0 * f(mix.p0) + (1.0 - mix.w0) * f(mix.p1)


def mix_proposal(x, margin=PROPOSAL_WIDEN, mix=R1):
    a0, a1 = margin * mix.a0, margin * mix.a1
    return (mix.w0 * np.exp(-GAUSS_K * x ** 2 / a0 ** 2) / a0 ** 2
            + (1.0 - mix.w0) * np.exp(-GAUSS_K * x ** 2 / a1 ** 2) / a1 ** 2)


def gaussian(r):
    return np.exp(-GAUSS_K * np.asarray(r, dtype=float) ** 2)


def waveguide(r):
    return np.interp(np.asarray(r, dtype=float), np.linspace(0.0, LUT_RANGE, LUT_SIZE), WAVE_LUT)


def uniform_proposal(r):
    return np.where(r <= UNIFORM_RADIUS, 1.0, 0.0)


# Worst-case ESS/N over every (pupil, saccade) state, per type
GRID_MIXES = {
    name: [mix_from_modes(solve(t, d, f))
           for d in np.linspace(0.0, 1.0, N_PUPIL_STEPS) for f in np.linspace(0.0, 1.0, N_SACCADE_STEPS)]
    for t, name in enumerate(TYPE_NAMES)
}


# Metrics (radial, so the 2pi factors cancel in the ratios)

def ess(target, proposal, R, n=2500):
    """ESS/N = (int S)^2 / (int g * int S^2/g)."""
    r = np.linspace(0.0, R, n)
    s, g = target(r), np.maximum(proposal(r), 1e-300)
    return float(np.trapezoid(s * r, r) ** 2 / (np.trapezoid(g * r, r) * np.trapezoid(s * s / g * r, r)))


def mix_ess(margin, mix=R1):
    return ess(lambda r: mix_target(r, mix), lambda r: mix_proposal(r, margin, mix),
               3.0 * margin * max(mix.a0, mix.a1))


def mix_reach(margin, R=LUT_RANGE):
    """Fraction of the target's area-weighted mass within the proposal's reach."""
    r_max = margin * max(R1.a0, R1.a1) * np.sqrt(-np.log(U1_MIN) / GAUSS_K)
    r = np.linspace(0.0, R, 20000)
    w = mix_target(r) * r
    return np.trapezoid(np.where(r <= r_max, w, 0.0), r) / np.trapezoid(w, r)


def coverage(proposal, target):
    """Overlap coefficient of the normalised proposal and target (radial densities)."""
    r = np.linspace(1e-4, 4.0, 12000)
    q, s = proposal(r) * r, target(r) * r
    q, s = q / np.trapezoid(q, r), s / np.trapezoid(s, r)
    return np.trapezoid(np.minimum(q, s), r)


GRID_WORST_EFF = {name: np.array([min(mix_ess(m, mx) for mx in mixes) for m in MARGINS]) for name, mixes in GRID_MIXES.items()}


# Low-discrepancy sequences, bit-for-bit with the shaders

def reverse_bits(x):
    x = np.asarray(x, dtype=np.uint32)
    x = (x << np.uint32(16)) | (x >> np.uint32(16))
    for sh, m in ((8, 0x00ff00ff), (4, 0x0f0f0f0f), (2, 0x33333333), (1, 0x55555555)):
        x = ((x & np.uint32(m)) << np.uint32(sh)) | ((x & np.uint32(~m & 0xffffffff)) >> np.uint32(sh))
    return x


def radical_inverse(n):
    return reverse_bits(n).astype(float) * 2.3283064365386963e-10


def halton(n, base):
    out = []
    for i in range(1, n + 1):
        f, r = 1.0, 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        out.append(r)
    return np.array(out)


def sobol_dim1(i):
    ii = np.asarray(i, dtype=np.uint32).copy()
    r = np.zeros_like(ii)
    v = np.uint32(1 << 31)
    while np.any(ii):
        r[(ii & np.uint32(1)).astype(bool)] ^= v
        ii >>= np.uint32(1)
        v ^= v >> np.uint32(1)
    return r


def owen_scramble(v, seed):
    v, seed = np.asarray(v, dtype=np.uint32).copy(), np.uint32(seed)
    with np.errstate(over='ignore'):    # uint32 wraparound is intended
        v = reverse_bits(v)
        v ^= v * np.uint32(0x3d20adea)
        v += seed
        v *= ((seed >> np.uint32(16)) | np.uint32(1))
        v ^= v * np.uint32(0x05526c56)
        v ^= v * np.uint32(0x53a22864)
    return reverse_bits(v)


# Ray sampling

def sample_rays(mode, target, n, rng):
    u1 = rng.uniform(U1_MIN, 1.0, n)
    phi = 2.0 * np.pi * rng.uniform(0.0, 1.0, n)
    weights = np.ones(n)

    if mode == 'Uniform':
        r = UNIFORM_RADIUS * np.sqrt(u1)
        weights = target(r)
    elif mode == 'Mixture-of-modes iCDF' and target is waveguide:
        # sampledir_mixture(): pick a mode by weight, widen its Gaussian, reweight to the true target
        a = PROPOSAL_WIDEN * np.where(rng.random(n) < R1.w0, R1.a0, R1.a1)
        r = a * np.sqrt(-np.log(u1) / GAUSS_K)
        weights = mix_target(r) / np.maximum(mix_proposal(r), 1e-300)
    else:   # Importance and Mixture-of-modes iCDF on a Gaussian target takes the same exact analytic path
        r = np.sqrt(-np.log(u1) / GAUSS_K)

    return r * np.cos(phi), r * np.sin(phi), weights


# Panels

def sampling_curves(ax, s: PlotSettings, target, mode, color, seed):

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)

    rng = np.random.default_rng(seed)
    x = np.linspace(-X_LIMIT, X_LIMIT, 1000)

    # Target S(theta), and area-weighted S(theta) * theta
    ty = target(np.abs(x))

    ax.plot(x, ty, color=color, lw=target_lw, zorder=5)
    ax.fill_between(x, ty, color=color, alpha=0.10, zorder=1)

    contrib = ty * np.abs(x)
    if contrib.max() > 0:
        contrib /= contrib.max() * 1.8

    ax.plot(x, contrib, color=color, lw=s.curve_lw, linestyle='--', alpha=0.5, zorder=4)

    # Proposal p(theta)
    if mode == 'Uniform':
        py, label = np.full_like(x, UNIFORM_LEVEL), 'Uniform\n$p(\\theta)$'
    elif mode == 'Mixture-of-modes iCDF' and target is waveguide:
        py = mix_proposal(x)
        py, label = py / py.max(), f'Mixture\n(reweighted, {PROPOSAL_WIDEN:g}$\\times$)'
    else:
        py, label = gaussian(x), ('Gaussian\n$p(\\theta)$' if mode == 'Importance' else 'Gaussian iCDF\n= Importance')

    ax.plot(x, py, color=s.green, lw=proposal_lw, alpha=0.8, linestyle=(0, (0.1, 2)), dash_capstyle='round', zorder=4)

    if mode != 'Importance':
        ax.fill_between(x, py, color=s.green, alpha=0.05, zorder=1)

    ax.text(1.72, 0.60, label, color=s.green, fontsize=s.small, fontweight='bold', ha='center', zorder=Z_TEXT)

    ax.hlines(0.5, -0.5, 0.5, color=s.dark, linestyle='--', lw=guide_lw, zorder=10)
    ax.text(0, 0.33, 'FWHM', color=s.dark, fontsize=s.small, fontweight='bold', ha='center', zorder=Z_TEXT)
    ax.text(-1.8, 0.13, 'Target\n$S(\\theta)$', color=color, fontsize=s.base, fontweight='bold', ha='center', zorder=Z_TEXT)

    # Rays (alpha = importance weight)
    rx, ry, w = sample_rays(mode, target, NUM_RAYS, rng)
    wn = w / (w.max() if w.max() > 0 else 1.0)

    for x_top, wi in zip(rx, wn):
        if abs(x_top) < X_LIMIT:
            x_bot = x_top * (CONVERGENCE_DIST + UNDER_FLOOR) / (RAY_HEIGHT + CONVERGENCE_DIST)
            a = max(0.001, wi * 0.7)

            ax.plot([x_bot, x_top], [UNDER_FLOOR, RAY_HEIGHT], color=s.yellorange if a > 0.001 else 'grey',
                    alpha=a if a > 0.001 else 0.1, lw=ray_lw, zorder=2)

    # Inset: 2D sample cloud over the target
    ins = ax.inset_axes([0.02, 0.63, 0.3, 0.3])
    ins.set_zorder(Z_RASTER + 1)

    if s.rasterize:
        ins.set_rasterization_zorder(Z_RASTER)

    lim = 1.8
    side = np.linspace(-lim, lim, 100)

    X, Y = np.meshgrid(side, side)

    ins.imshow(target(np.hypot(X, Y)), extent=[-lim, lim, -lim, lim], origin='lower', alpha=0.3, zorder=1,
               cmap=LinearSegmentedColormap.from_list('c', ['white', color]))

    circles(ins, s)
    ins.scatter(rx, ry, s=20, color=s.yellorange, alpha=np.clip(wn, 0.06, 0.65), edgecolors='white', lw=0.4, zorder=5)
    ins.set(xlim=(-lim, lim), ylim=(-lim, lim), xticks=[], yticks=[])

    for sp in ins.spines.values():
        sp.set_edgecolor(s.frame)
        sp.set_linewidth(0.8)

    ax.set_xlim(-X_LIMIT, X_LIMIT)
    ax.set_ylim(UNDER_FLOOR - 0.1, RAY_HEIGHT + 0.3)
    ax.axis('off')


def circles(ax, s):
    """FWHM and main lobe guides."""
    ax.add_patch(Circle((0, 0), 0.5, color=s.dark, fill=False, linestyle='--', lw=guide_lw, zorder=6))
    ax.add_patch(Circle((0, 0), 1.1, color=s.dark, fill=False, linestyle=':', lw=guide_lw * 0.6, alpha=0.7, zorder=6))


def uniforms(kind, n, rng):
    """(u1, u2) for each randomness mode."""

    i = np.arange(n)
    inv = 2.3283064365386963e-10

    if kind == 'Stratified':
        g = int(np.ceil(np.sqrt(n)))
        cell = (i + int(rng.integers(0, g * g))) % (g * g)
        return (cell % g + rng.random(n)) / g, (cell // g + rng.random(n)) / g
    if kind == 'Halton':
        return halton(n, 2), halton(n, 3)
    if kind == 'Hammersley':
        o1, o2 = rng.random(), rng.random()
        return np.mod((i + 0.5) / n + o1, 1.0), np.mod(radical_inverse(i) + o2, 1.0)
    if kind == 'Sobol':
        i = i.astype(np.uint32)
        return (owen_scramble(reverse_bits(i), 0x9E3779B9) * inv, owen_scramble(sobol_dim1(i), 0x85EBCA6B) * inv)
    if kind == 'Fibonacci':
        golden = np.pi * (3.0 - np.sqrt(5.0))
        rot, rad = rng.random(), rng.random()
        return 1.0 - np.mod((i + 0.5) / n + rad, 1.0), np.mod(i * golden / (2 * np.pi) + rot, 1.0)

    return rng.random(n), rng.random(n)    # Pseudo-random


def randomness_scatter(ax, s: PlotSettings, kind):

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)

    u1, u2 = uniforms(kind, NUM_RAYS, np.random.default_rng(42))
    radii = np.sqrt(-np.log(np.clip(u1, U1_MIN, 1.0)) / GAUSS_K)    # Gaussian importance sample

    circles(ax, s)

    sc = ax.scatter(radii * np.cos(2 * np.pi * u2), radii * np.sin(2 * np.pi * u2), s=22, c=gaussian(radii),
                    cmap=LinearSegmentedColormap.from_list('utility', [s.dark, s.yellorange]),
                    vmin=0, vmax=1, alpha=0.9, edgecolors='white', lw=0.4, zorder=4)

    ax.set_title(kind, fontsize=s.title, pad=5)
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), xticks=[], yticks=[], aspect='equal')
    ax.spines[:].set_visible(False)

    return sc


def style_legend(leg):
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.15)
    leg.get_frame().set_facecolor('white')
    leg.set_zorder(Z_TEXT)


def efficiency_coverage_scatter(ax, s: PlotSettings):

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)

    lo, hi = min(SWEEP_MARGINS), max(SWEEP_MARGINS)
    cov = [100 * coverage(lambda r, m=m: mix_proposal(r, m), mix_target) for m in SWEEP_MARGINS]
    eff = [100 * mix_ess(m) for m in SWEEP_MARGINS]

    ax.plot(cov, eff, color=s.red, lw=s.curve_lw, alpha=0.3, zorder=2)

    for m, c, e in zip(SWEEP_MARGINS, cov, eff):
        ax.scatter(c, e, marker='o', s=26, facecolor=s.red, edgecolor='white', linewidth=0.5,
                   alpha=0.30 + 0.70 * (m - lo) / (hi - lo), zorder=4)

    i = SWEEP_MARGINS.index(PROPOSAL_WIDEN)

    ax.scatter(cov[i], eff[i], marker='o', s=78, facecolor='none', edgecolor=s.red, linewidth=1.0, zorder=5)

    for j in (0, -1):
        ax.annotate(rf'$\times${SWEEP_MARGINS[j]:g}', (cov[j], eff[j]), xytext=(0, 3), textcoords='offset points',
                    fontsize=s.tiny, color=s.red, zorder=Z_TEXT)

    for tf, col in [(gaussian, s.blue), (waveguide, s.red)]:
        ax.scatter(100 * coverage(uniform_proposal, tf), 100 * ess(tf, uniform_proposal, UNIFORM_RADIUS),
                   marker='s', s=32, facecolor=col, edgecolor='white', linewidth=0.5, zorder=4)
        ax.scatter(100 * coverage(gaussian, tf), 100 * ess(tf, gaussian, np.sqrt(-np.log(U1_MIN) / GAUSS_K)),
                   marker='^', s=40, facecolor=col, edgecolor='white', linewidth=0.5, zorder=4)

    gc = 100 * coverage(gaussian, gaussian)

    ax.scatter(gc, 100.0, marker='o', s=78, facecolor='none', edgecolor=s.blue, linewidth=1.0, zorder=5)

    ax.annotate('Gaussian', (gc, 100.0), xytext=(-10, -13), textcoords='offset points',
                fontsize=s.tiny, color=s.blue, zorder=Z_TEXT)

    ax.text(70, 35, 'Waveguide', color=s.red, fontsize=s.base, ha='center', va='center', zorder=Z_TEXT)
    ax.text(0.99, 0.03, rf'circled: $\times{PROPOSAL_WIDEN:g}$', transform=ax.transAxes, fontsize=s.tiny,
            color=s.dark, ha='right', va='bottom', zorder=Z_TEXT)

    handles = [Line2D([0], [0], marker=m, linestyle='none', markerfacecolor=s.frame, markeredgecolor='white',
                      markersize=5.5, label=lab) for lab, m in [('Uniform', 's'), ('Importance', '^'), ('Mixture-of-modes iCDF', 'o')]]
    leg = ax.legend(handles=handles, loc='upper left', fontsize=s.tiny, handletextpad=0.25, borderpad=0.2,
                    labelspacing=0.2)

    style_legend(leg)
    ax.add_artist(leg)

    ax.set_xlim(15, 106)
    ax.set_ylim(-19, 112)

    ax.set_xlabel(r'Coverage: proposal $\cap$ target (%)', fontsize=s.base)
    ax.set_ylabel('Efficiency\nESS/N (%)', fontsize=s.base)

    ax.set_title('(i) Strategies comparison', fontsize=s.title, loc='left', pad=3)

    ax.tick_params(labelsize=s.base)

    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color=s.grid, lw=s.grid_lw, zorder=0)
    ax.set_axisbelow(True)


def efficiency_reach_tradeoff(ax, s: PlotSettings):

    ax2 = ax.twinx()

    for curve in GRID_WORST_EFF.values():
        ax.plot(MARGINS, 100 * curve, color=s.red, lw=s.curve_lw * 0.7, alpha=0.35, zorder=3)

    ax.plot(MARGINS, 100 * np.min(list(GRID_WORST_EFF.values()), axis=0), color=s.dark, lw=s.curve_lw * 1.6, zorder=5)
    ax2.plot(MARGINS, [100 * mix_reach(m) for m in MARGINS], color=s.red, lw=s.curve_lw * 1.3,
             ls=(0, (4, 1.5)), alpha=0.9, zorder=3)

    # Gaussian target is exact at any margin: flat references
    ax.axhline(100.0, color=s.blue, lw=s.curve_lw * 1.6, zorder=4)
    ax2.axhline(100.0, color=s.blue, lw=s.curve_lw * 1.3, ls=(0, (4, 1.5)), alpha=0.9, zorder=3)

    ax.axvline(PROPOSAL_WIDEN, color=s.dark, lw=0.6, ls=':', zorder=2)
    ax.text(PROPOSAL_WIDEN + 0.04, 4, rf'$\times{PROPOSAL_WIDEN:g}$', color=s.dark, fontsize=s.tiny,
            ha='left', va='bottom', zorder=Z_TEXT)
    ax.text(1.4, 82, 'Gaussian', color=s.blue, fontsize=s.tiny, ha='center', zorder=Z_TEXT)
    ax.text(2.6, 12, 'Waveguide', color=s.dark, fontsize=s.tiny, ha='center', zorder=Z_TEXT)

    ax.set_xlim(0.7, 3.0)
    ax.set_ylim(-4, 112)
    ax2.set_ylim(90.0, 101.5)

    ax.set_xlabel(r'Widening factor', fontsize=s.base, labelpad=1)
    ax.set_ylabel('Worst-state\nESS/N (%)', fontsize=s.base)
    ax2.set_ylabel('Reach (%)', fontsize=s.base, color=s.red)

    ax2.tick_params(labelsize=s.tiny, colors=s.red, pad=1)
    ax2.spines['right'].set_color(s.frame)

    ax.set_title('(ii) Proposal widening', fontsize=s.title, loc='left', pad=3)

    ax.tick_params(labelsize=s.tiny, pad=1)

    for a in (ax, ax2):
        a.spines['top'].set_visible(False)

    ax.grid(color=s.grid, lw=s.grid_lw)
    ax.set_axisbelow(True)

    h = [Line2D([0], [0], color=s.dark, lw=s.curve_lw * 1.6, label='Lowest of R1–R7/8'),
         Line2D([0], [0], color=s.red, lw=s.curve_lw * 0.7, alpha=0.5, label='Each of R1–R7/8'),
         Line2D([0], [0], color=s.dark, lw=s.curve_lw, ls=(0, (4, 1.5)), label='Reach')]

    style_legend(ax.legend(handles=h, loc='upper right', bbox_to_anchor=(1.0, 0.9), fontsize=s.tiny,
                           handletextpad=0.4, borderpad=0.25))


# Figure

STRATEGIES = ['Uniform', 'Importance', 'Mixture-of-modes iCDF']
RAND_KINDS = ['Pseudo-random', 'Stratified', 'Fibonacci', 'Halton', 'Hammersley', 'Sobol']


def build_figure(s: PlotSettings) -> plt.Figure:

    fig = s.new_figure()

    outer = GridSpec(2, 1, figure=fig, height_ratios=[2.0, 1.3], hspace=0.14)
    top = outer[0].subgridspec(2, 3, wspace=0.05, hspace=0.0)

    # Rows: target x strategy
    rows = []
    for r, (target, color) in enumerate([(gaussian, s.blue), (waveguide, s.red)]):
        rows.append([fig.add_subplot(top[r, c]) for c in range(3)])
        for c, (ax, strat) in enumerate(zip(rows[r], STRATEGIES)):
            sampling_curves(ax, s, target, strat, color, seed=100 * r + c)

    # Bottom: randomness modes (left), efficiency plots (right)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.40)
    rand_gs = bottom[0].subgridspec(2, 3, wspace=-0.15, hspace=0.35)
    rand_axes = [fig.add_subplot(rand_gs[i // 3, i % 3]) for i in range(6)]

    sc = [randomness_scatter(ax, s, k) for ax, k in zip(rand_axes, RAND_KINDS)][0]

    p_tr, p_br = rand_axes[2].get_position(fig), rand_axes[5].get_position(fig)

    cb_ax = fig.add_axes((p_tr.x1 - 0.01, p_br.y1 / 2.0, 0.015, p_tr.height * 2 - 0.02))
    cb = fig.colorbar(sc, cax=cb_ax, orientation='vertical')

    cb.set_label('Relative sensitivity $S(\\theta)$', fontsize=s.tiny, labelpad=3)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=s.tiny, width=0.5, length=2)

    p_mid = rand_axes[4].get_position(fig)

    fig.text((p_mid.x0 + p_mid.x1) / 2.0 - 0.03, p_mid.y0 - 0.05, 'Stochastic distributions',
             fontsize=s.title, ha='center', va='top', fontweight='bold')

    eff_area = bottom[1].subgridspec(1, 2, width_ratios=[1.0, 0.15], wspace=0.0)   # right strip left empty
    eff_gs = eff_area[0].subgridspec(2, 1, hspace=0.75)

    efficiency_coverage_scatter(fig.add_subplot(eff_gs[0]), s)
    efficiency_reach_tradeoff(fig.add_subplot(eff_gs[1]), s)

    fig.subplots_adjust(left=0.06, right=0.97, top=0.905, bottom=0.07)

    # Headers and panel letters (after the final layout)
    for ax, strat in zip(rows[0], STRATEGIES):
        column_header(fig, s, ax, strat, dy=0.014)

    for row, label, col in ((rows[0], 'Gaussian target', s.blue), (rows[1], 'Waveguide target', s.red)):
        row_header(fig, s, row[0], label, dx=0.018, colour=col)

    panel_letter(fig, s, 'A', rows[0][0], dy=0.02)
    panel_letter(fig, s, 'B', bottom[0].get_position(fig), dy=0.0)

    c_pos = bottom[1].get_position(fig)
    panel_letter(fig, s, 'C', c_pos, dx=-0.055, dy=0.0)

    return fig


if __name__ == '__main__':

    settings = PlotSettings.nature_double().apply()

    settings.savefig(build_figure(settings), 'sampling', formats=['pdf'])
