"""
Microsaccadic hyperacuity proof-of-concept.

Drosophila photomechanical microsaccades sweep ~vertically in visual space.
So: thin horizontal bars, agent oscillates vertically, readout is a
single forward-pointing cartridge.

Four per-rhabdomere actuation conditions on the same cartridge:
    none     - no microsaccade (static optical RF)
    axial    - axial move only   -> RF narrowing, no lateral shift
    lateral  - lateral move only -> RF shift, no narrowing
    full     - both

The rise/relax asymmetry means the bar-up and bar-down profiles differ, and that direction-dependent signature is
itself part of the hyperacuity code. In the figure, a symmetric butterfly
means no direction dependence, and any visible asymmetry is the effect.

Two narrowing regimes:
    optical (r_xtra = 1.0): pure optical (Kemppainen 2022, ~2 um axial move).
                            Narrowing ~0.4 deg, so 'axial'/'full' do NOT resolve
                            the two bars below the optical Sparrow limit.
    phenomenological (r_xtra ~ 0.535): exaggerated dynamic RF contraction standing in for
                                       the pupil / transduction narrowing. 'axial'/'full'
                                       then resolve bars that 'none'/'lateral' cannot.

Figure layout:
    A: Schematic placeholder (stimulus, sweep, cartridge)
    B: Regime (rows) x condition (cols), mirrored UP/DOWN pooled R1-R6 profiles
    C: Per-rhabdomere decomposition for one exemplar condition
    D: Scalar summaries: dip depth and UP/DOWN shape divergence
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.signal import savgol_filter

from insectvision.compound_eyes import Model
from insectvision.compound_eyes.helpers.acceptance import SnyderAcceptance
from insectvision.compound_eyes.rhabdomeres import drosophila_bundle
from insectvision.types import WORLD_FORWARD, RHAB_COLOURS
from insectvision.compound_eyes.helpers.alignment import BundlesAligner
from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.engine.meshes import plane_geom
from insectvision.renderers import Renderer

from visualisation.plot_settings import (
    PlotSettings, Z_RASTER, Z_TEXT, panel_letter, column_header, row_header, despine, placeholder
)


# Config

SEP_TEST_DEG    = 4.0               # deg, centre-to-centre of the 2-bar stimulus
BAR_SEPARATIONS = [0.0, SEP_TEST_DEG]   # 0 = single-bar control
BAR_WIDTH_DEG   = 1.0               # deg
DISTANCE        = 2.0               # m (bar plane at z = -DISTANCE)
BAR_LENGTH      = 10.0              # m

SWEEP_SPEED_DEG = 40.0              # deg/s (angular sweep speed at the centre of the field)
SWEEP_AMPLITUDE = 1.0               # m (travel is +/- this)

REGIMES = {'optical': 1.0, 'phenomenological': 0.535}

AMP_LAT  = 2.0                      # um, lateral microsaccade amplitude at full drive
AMP_AX   = 2.0                      # um, axial microsaccade amplitude at full drive

TAU_MEMBRANE = 0.012    # (10-15 ms / 0.010–0.015 s)
# TAU_MEMBRANE = 0.003    # super short just to denoise, and keep almost no blur

TAU_FAST     = 0.005    # (5 ms / 0.005 s)
# TAU_ADAPT    = 0.050    # (50-100 ms / 0.050–0.100 s)
TAU_ADAPT    = 0.100    # (50-100 ms / 0.050–0.100 s)

TAU_RISE     = 0.015    # (10-15 ms / 0.010–0.015 s)
TAU_RELAX    = 0.060    # (60-100 ms / 0.060–0.100 s)
# TAU_RELAX    = 0.150    # (60-100 ms / 0.060–0.100 s)

VIEW_HALFWIDTH = 0.30   # m, x-axis half-range around the bars

SWEEP_SPEED    = DISTANCE * np.radians(SWEEP_SPEED_DEG)     # m/s
SWEEP_DURATION = (2 * SWEEP_AMPLITUDE) / SWEEP_SPEED        # one-way sweep (s)
CYCLE_DURATION = 2 * SWEEP_DURATION                         # up + down (s)
COND_DURATION  = CYCLE_DURATION                             # one full cycle per condition

CONDITIONS = [
    # name,     actuation,   ampl_lat_um,   ampl_ax_um
    ('none',       False,       0.0,          0.0),
    ('axial',      True,        0.0,       AMP_AX),
    ('lateral',    True,    AMP_LAT,          0.0),
    ('full',       True,    AMP_LAT,       AMP_AX),
]
RUN_DURATION   = COND_DURATION * len(CONDITIONS)           # per separation

COND_COLOR = {'none': '#4477AA', 'axial': '#228833', 'lateral': '#EE7733', 'full': '#AA3377'}
COND_LABEL = {'none': 'none', 'axial': 'axial', 'lateral': 'lateral', 'full': 'full'}

EXEMPLAR = ('phenomenological', 'full')     # condition shown broken down per rhabdomere in panel C

DIP_MIN_DEPTH = 0.02                # Below this the two bars count as fused (unresolved)
DIP_CENTRE_ON_PROFILE = True        # Centre the dip search on the profile, not on agent-y = 0

R, CENTER = 7, 6
PERIPH = list(range(6))


_CONTEXT = None
def get_context() -> Context:
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = Context()
        _CONTEXT.time_step = 1 / 1000.0     # 1 ms biological resolution
        _CONTEXT.mouse_captured = False
    return _CONTEXT


def build_model(extra_narrowing):
    """Build the Drosophila eyes model for a given phenomenological narrowing ratio."""

    bundle = drosophila_bundle()
    bundle.extra_narrowing_ratio = extra_narrowing

    droso_head_ptich = np.deg2rad(10.1)  # drosophila head pitch in flight

    aligner = BundlesAligner(
        ref_direction=np.array([0.0, np.sin(droso_head_ptich), np.cos(droso_head_ptich)]),  # optic flow in flight
        combing_strength=1.0,
        combing_angle_deg=45.0,
        combing_falloff=0.7,
        alignment_smoothing_iter=5,
        saccade_smoothing_iter=5,
        flip_polarity=False,
        flip_saccade_polarity=True,
        equatorial_discontinuity=True,
    )

    model = Model.from_file(
        'assets/drosophila_scaffold.npz',
        bundle=bundle,
        acceptance=SnyderAcceptance(),
        orientation=aligner,
        neural_superposition=True,  # superposition eyes
    )

    model.refine_superposition(smooth_iters=5, relax=0.8, adjust_scale=True, adjust_anisotropy=True, rewire=True)
    model.refine_superposition(smooth_iters=5, relax=0.8, adjust_scale=True, adjust_anisotropy=True, rewire=False)

    model.scale(1e-6)

    model.tau_membrane = TAU_MEMBRANE
    model.tau_adapt_fast = TAU_FAST
    model.tau_adapt_slow = TAU_ADAPT
    model.tau_rise = TAU_RISE
    model.tau_relax = TAU_RELAX

    return model


# Scene / sweep helpers

def make_bars(sep_deg):
    """Build the white horizontal bar(s) for a given angular separation (deg)."""

    thickness = 2.0 * DISTANCE * np.tan(np.radians(BAR_WIDTH_DEG) / 2.0)
    sep_m = 2.0 * DISTANCE * np.tan(np.radians(sep_deg) / 2.0)
    centres = ([-sep_m / 2, +sep_m / 2] if sep_deg > 0 else [0.0])
    bars_lum = np.ones((32, 32), dtype=np.uint8) * 255

    bars = []
    for i, cy in enumerate(centres):
        v0 = [-BAR_LENGTH / 2, cy - thickness / 2, -DISTANCE]
        v1 = [-BAR_LENGTH / 2, cy + thickness / 2, -DISTANCE]
        v2 = [+BAR_LENGTH / 2, cy + thickness / 2, -DISTANCE]
        v3 = [+BAR_LENGTH / 2, cy - thickness / 2, -DISTANCE]
        verts, uv, faces = plane_geom(v0, v1, v2, v3)
        bars.append(Asset.from_arrays(name=f'bar{i}', vertices=verts, faces=faces, uv_coords=uv, texture=bars_lum))

    return bars


def apply_condition(renderer, model, actuation, ampl_lat, ampl_ax):
    """Set actuation + microsaccade amplitudes and reset the per-lens dynamic state."""

    model.ommatidia.lateral_amplitude = ampl_lat
    model.ommatidia.axial_amplitude = ampl_ax

    renderer.microsaccades_enabled = actuation

    model.buffer.ommatidia_dynamic['curr_lateral_disp'] = 0.0
    model.buffer.ommatidia_dynamic['curr_axial_disp'] = 0.0
    model.buffer.ommatidia_dynamic['curr_lum_slow'] = 0.0
    model.buffer.ommatidia_dynamic['curr_lum_fast'] = 0.0

    model.buffer.ommatidia_stale = True


def sweep_position(elapsed):
    """Agent Y (m) and apparent-bar-motion direction for the time within one up/down cycle."""

    t = elapsed % CYCLE_DURATION
    if t < SWEEP_DURATION:                 # agent up (+y) -> bar appears to move down
        return -SWEEP_AMPLITUDE + (t / SWEEP_DURATION) * 2 * SWEEP_AMPLITUDE, -1

    return SWEEP_AMPLITUDE - ((t - SWEEP_DURATION) / SWEEP_DURATION) * 2 * SWEEP_AMPLITUDE, +1


def simulate(model, sep_deg):
    """Render the four actuation conditions for one bar separation, return per-timestep arrays."""

    context = get_context()

    bars = make_bars(sep_deg)

    scene = Scene(background_color=(0.0, 0.0, 0.0))
    scene.sun.elevation, scene.sun.azimuth, scene.sun.color = 1.0, 0.0, (1.0, 1.0, 1.0)
    for b in bars:
        scene.add_instance(b)

    agent = Agent(position=(0.0, 0.0, 0.0))

    renderer = Renderer(
        model=model, scene=scene, agent=agent,
        nb_samples=512, time_dithering=True, randomness_mode='Halton',
        enable_microsaccades=True, enable_ambient=True, enable_direct=True, enable_shadows=False)

    renderer.ambient_intensity = 1.5
    renderer.photon_concentration = 0.0    # isolate RF geometry from the photon-concentration gain

    # Single forward-pointing cartridge (closest optical axis to straight ahead)
    cone = model.query_cone(WORLD_FORWARD, angle=10.0, degrees=True, avoid_conflicts=True)
    if len(cone) == 0:
        raise RuntimeError('No forward-facing ommatidia found.')

    az = np.rad2deg(cone.azimuth)
    el = np.rad2deg(cone.elevation)
    selected = int(cone.indices[int(np.argmin(az ** 2 + el ** 2))])
    renderer.selected_ommatidia = [selected]

    rec = {'agent_y': [], 'cond': [], 'mdir': [], 'cart': []}

    t_start, phase = None, -1
    while context.run_interactive():

        context.input()
        if t_start is None:
            t_start = context.total_time

        elapsed = context.total_time - t_start

        new_phase = min(int(elapsed // COND_DURATION), len(CONDITIONS) - 1)
        if new_phase != phase:
            phase = new_phase
            apply_condition(renderer, model, *CONDITIONS[phase][1:])

        name = CONDITIONS[phase][0]

        ay, mdir = sweep_position(elapsed)
        agent.position = (0.0, ay, 0.0)

        out = renderer.step()

        rec['agent_y'].append(ay)
        rec['cond'].append(name)
        rec['mdir'].append(mdir)
        rec['cart'].append(out.per_cartridge[selected, :, :3].mean(axis=-1))

        context.display()

        if elapsed >= RUN_DURATION:
            break

    for obj in (renderer, scene):
        free = getattr(obj, 'free', None)
        if callable(free):
            try:
                free()
            except Exception:
                pass

    return {k: np.array(v) for k, v in rec.items()}


# Analysis

def to_deg(y_m):
    """
    Agent Y (m) -> visual angle (deg) subtended at the bar plane.
    """
    return np.degrees(np.arctan2(np.asarray(y_m, dtype=float), DISTANCE))


def smooth(a, k=5):
    """Light boxcar smoothing, to denoise binned profiles before dip detection."""
    if k <= 1:
        return a
    return np.convolve(a, np.ones(k) / k, mode='same')


def binned(y, sig, bins=160):
    """Spatial binning over the sweep, to align runs onto a common Y grid (then smoothed)."""

    edges = np.linspace(-SWEEP_AMPLITUDE, SWEEP_AMPLITUDE, bins + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(y, edges) - 1, 0, bins - 1)
    out, cnt = np.zeros(bins), np.zeros(bins)
    np.add.at(out, idx, sig)
    np.add.at(cnt, idx, 1)
    nz = cnt > 0
    out[nz] /= cnt[nz]

    return grid, smooth(out)


def trace(res, cond, direction, channel):
    """Binned signal for one condition / sweep direction. 'channel' = receptor index or 'pool'."""
    m = (res['cond'] == cond) & (res['mdir'] == direction)
    sig = res['cart'][m][:, PERIPH].mean(axis=1) if channel == 'pool' else res['cart'][m][:, channel]
    return binned(res['agent_y'][m], sig)


def profile_centre(grid, prof):
    """Centre of mass of the in-view profile, used to centre the dip search."""
    m = np.abs(grid) <= VIEW_HALFWIDTH
    w = np.clip(prof[m], 0.0, None)
    tot = w.sum()

    return float(np.sum(grid[m] * w) / tot) if tot > 0 else 0.0


def get_centered_data(grid, prof):
    """Aligns profile so CoM is at 0 (to remove temporal lag)."""
    c = profile_centre(grid, prof)
    return grid - c, prof


def find_dip(grid, prof, view=0.30, min_peak=0.1, deriv_win=9):
    """
    Dip detection using zero-crossings and inflection points.
    """
    m = np.abs(grid) <= view
    g, p = np.asarray(grid)[m], np.asarray(prof)[m]
    if p.size < 11 or p.max() <= 0:
        return None

    win = max(7, deriv_win | 1)
    win = min(win, p.size - (p.size + 1) % 2)

    # d1 = slope (first derivative), d2 = curvature (second derivative)
    d1 = savgol_filter(p, win, 2, deriv=1, delta=float(g[1] - g[0]))
    d2 = savgol_filter(p, win, 2, deriv=2, delta=float(g[1] - g[0]))

    dmax = np.abs(d1).max()
    base = float(np.percentile(p, 5))
    floor = base + min_peak * (p.max() - base)

    # Dip detection
    # A dip is a local minimum: slope (d1) crosses from negative to positive
    for i in range(2, len(d1) - 2):
        if d1[i - 1] < 0 and d1[i] >= 0:
            idx = i
            # Measure prominence to ensure it's a real dip between two peaks
            p_left = p[:idx].max()
            p_right = p[idx:].max()
            if p_left > floor and p_right > floor:
                depth = (min(p_left, p_right) - p[idx]) / (p.max() - base + 1e-6)
                if depth > 0.005:  # Real but potentially shallow
                    return (float(g[idx]), float(p[idx]), float(depth), 'dip')

    # Shoulder detection
    for i in range(2, len(d2) - 2):
        # We look for where curvature changes sign
        if d2[i - 1] * d2[i] < 0:
            idx = i
            # slope significantly slowed down?
            if abs(d1[idx]) < 0.25 * dmax and p[idx] > floor:
                # Check it's actually a shoulder (slope doesn't flip sign nearby)
                return (float(g[idx]), float(p[idx]), 0.05, 'shoulder')

    return None


def _get_signal_x_limit(results, cond_names):
    max_ext_deg = 1.0
    for res in results.values():
        for cond in cond_names:
            for mdir in [-1, 1]:
                g, p = trace(res, cond, mdir, 'pool')
                if p.max() <= 0: continue
                gc, pc = get_centered_data(g, p)
                active = np.where(pc > 0.01 * pc.max())[0]
                if len(active) > 0:
                    ext = np.abs(to_deg(gc[active])).max()
                    max_ext_deg = max(max_ext_deg, ext)
    return max_ext_deg + 0.5


def updown_shape_divergence(grid, up, down):
    """UP vs DOWN difference in profile shape."""

    mask = np.abs(grid) <= VIEW_HALFWIDTH
    u = up[mask] - up[mask].min()
    d = down[mask] - down[mask].min()
    n = u.size
    max_lag = n // 3

    def overlap(k):     # u against d shifted by k samples
        a = u[k:] if k >= 0 else u[:n + k]
        b = d[:n - k] if k >= 0 else d[-k:]
        return a, b

    k_best = min(range(-max_lag, max_lag + 1),
                 key=lambda k: np.mean((np.subtract(*overlap(k))) ** 2))

    a, b = overlap(k_best)
    span = max(u.max(), d.max()) + 1e-9
    return float(np.mean(np.abs(a - b)) / span)


# Figure

def _cell_stats(results, regime, cond):
    """Everything panel D needs for one (regime, condition) cell."""

    res2 = results[(regime, SEP_TEST_DEG)]

    gu, up = get_centered_data(*trace(res2, cond, +1, 'pool'))
    gd, dn = get_centered_data(*trace(res2, cond, -1, 'pool'))

    dip_u = find_dip(gu, up, view=VIEW_HALFWIDTH)
    dip_d = find_dip(gd, dn, view=VIEW_HALFWIDTH)

    d_up = dip_u[2] if dip_u else 0.0
    d_dn = dip_d[2] if dip_d else 0.0

    return {
        'asym': updown_shape_divergence(gu, up, dn),
        'dip_up': d_up,
        'dip_dn': d_dn,
        'dip': 0.5 * (d_up + d_dn),
        'peak': float(max(up.max(), dn.max())),
        'kind_up': dip_u[3] if dip_u else 'fused',
        'kind_dn': dip_d[3] if dip_d else 'fused',
    }


def _mirror_axes(ax, s: PlotSettings, sep_deg):

    for xb in (-sep_deg / 2, +sep_deg / 2):
        ax.axvline(xb, color=s.red, ls=(0, (1, 1.6)), lw=0.7, alpha=0.75, zorder=2)

    ax.axhline(0.0, color=s.dark, lw=s.axis_lw, zorder=4)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f'{abs(v):g}'))
    ax.tick_params(labelsize=s.tiny, pad=1.5)
    despine(ax, keep=('left', 'bottom'))
    ax.grid(axis='x', color=s.grid, lw=s.grid_lw, zorder=0)
    ax.set_axisbelow(True)


def _profile_panel(ax, s: PlotSettings, results, regime, cond, sep_m, sep_deg, x_lim):
    """Mirrored pooled R1-R6 profile (UP is up / DOWN is down)."""

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)

    res2 = results[(regime, SEP_TEST_DEG)]
    res1 = results[(regime, 0.0)]
    colour = COND_COLOR[cond]

    for direction, sign in ((+1, +1.0), (-1, -1.0)):

        # Center data to remove lag that comes from tau_membrane
        grid, pooled = get_centered_data(*trace(res2, cond, direction, 'pool'))
        gc, ctrl = get_centered_data(*trace(res1, cond, direction, 'pool'))

        # Single-bar control: filled silhouette + dashed outline
        ax.fill_between(to_deg(gc), 0.0, sign * ctrl,
                        facecolor=s.frame, alpha=0.14, lw=0.0, zorder=1)
        ax.plot(to_deg(gc), sign * ctrl, color=s.frame, lw=0.55,
                ls=(0, (2.5, 1.5)), alpha=0.9, zorder=3)

        # 2 bars pooled response
        ax.plot(to_deg(grid), sign * pooled, color=colour, lw=s.curve_lw * 1.5,
                solid_capstyle='round', zorder=6)

        dip = find_dip(grid, pooled)
        if dip is not None and dip[2] >= DIP_MIN_DEPTH:
            xd, yd, _, kind = dip
            ax.plot(to_deg(xd), sign * yd,
                    marker=('v' if sign > 0 else '^') if kind == 'dip' else 's',
                    color=colour, ms=3.0, mec='white', mew=0.4, zorder=7)

    _mirror_axes(ax, s, sep_deg)
    ax.set_xlim(-x_lim, x_lim)
    return ax


def _rhabdomere_panel(ax, s: PlotSettings, results, regime, cond, sep_m, sep_deg, x_lim):
    """Mirrored per-rhabdomere traces for the exemplar condition (panel C)."""

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)

    res2 = results[(regime, SEP_TEST_DEG)]

    for direction, sign in ((+1, +1.0), (-1, -1.0)):
        # Calculate pooled center to align all cells to the same frame
        g_ref, p_ref = trace(res2, cond, direction, 'pool')
        center_shift = profile_centre(g_ref, p_ref)

        for i in range(R):
            gi, ri = trace(res2, cond, direction, i)
            ax.plot(to_deg(gi - center_shift), sign * ri, color=RHAB_COLOURS[i],
                    lw=0.7, alpha=0.85, zorder=5)

        ax.plot(to_deg(g_ref - center_shift), sign * p_ref, color='k', lw=s.curve_lw * 1.6, zorder=6)

    _mirror_axes(ax, s, sep_deg)
    ax.set_xlim(-x_lim, x_lim)

    ymax = max(abs(v) for v in ax.get_ylim())
    ax.set_ylim(-ymax * 1.60, ymax * 1.08)

    handles = [Line2D([0], [0], color=RHAB_COLOURS[i], lw=1.0,
                      label=(f'R{i + 1}' if i < CENTER else 'R7/8')) for i in range(R)]
    handles.append(Line2D([0], [0], color='k', lw=1.4, label='pooled'))

    leg = ax.legend(handles=handles, loc='lower left', fontsize=s.tiny, ncol=4,
                    handlelength=0.9, handletextpad=0.35, columnspacing=0.7,
                    borderpad=0.25, labelspacing=0.2)
    leg.get_frame().set_edgecolor(s.frame)
    leg.get_frame().set_linewidth(0.3)
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(0.9)
    leg.set_zorder(Z_TEXT)
    return ax


def _summary_bars(ax, s: PlotSettings, stats, cond_names, key, title, ylabel,
                  dot_keys=None, threshold=None):
    """
    Grouped bars: one group per condition, one bar per regime.
    """
    regimes = list(REGIMES)
    width = 0.34
    offs = {regimes[0]: -width / 2, regimes[1]: +width / 2}

    for j, cond in enumerate(cond_names):
        for regime in regimes:
            v = stats[(regime, cond)][key]
            x = j + offs[regime]
            solid = (regime == regimes[1])
            ax.bar(x, v, width=width * 0.92,
                   facecolor=COND_COLOR[cond] if solid else 'white',
                   edgecolor=COND_COLOR[cond], linewidth=0.7, zorder=3)

            if dot_keys:
                for dk, mk in dot_keys:
                    ax.plot(x, stats[(regime, cond)][dk], marker=mk, ms=2.6,
                            mfc=s.dark, mec='white', mew=0.3, ls='none', zorder=5)

    if threshold is not None:
        ax.axhline(threshold, color=s.dark, lw=0.6, ls=(0, (3, 2)), zorder=2)
        # ax.text(3.45, threshold, 'fused', fontsize=s.tiny, color=s.dark,
        #         ha='right', va='bottom', zorder=Z_TEXT)

    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels([COND_LABEL[c] for c in cond_names], fontsize=s.tiny)

    ax.set_ylabel(ylabel, fontsize=s.base)

    ax.set_title(title, fontsize=s.title, loc='left', pad=3)

    ax.tick_params(labelsize=s.tiny, pad=1.5)

    ax.set_xlim(-0.6, len(cond_names) - 0.4)
    despine(ax, keep=('left', 'bottom'))

    ax.grid(axis='y', color=s.grid, lw=s.grid_lw, zorder=0)

    ax.set_axisbelow(True)


def make_figure(results, s: PlotSettings) -> plt.Figure:

    sep_deg = SEP_TEST_DEG
    sep_m = 2.0 * DISTANCE * np.tan(np.radians(sep_deg) / 2.0)
    cond_names = [c[0] for c in CONDITIONS]
    regimes = list(REGIMES)

    stats = {(rg, c): _cell_stats(results, rg, c)
             for rg in regimes for c in cond_names}
    x_limit = _get_signal_x_limit(results, cond_names)

    fig = s.new_figure()
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.90, 0.95], hspace=0.46)

    # A: Schematic placeholder
    axA = fig.add_subplot(outer[0])
    placeholder(axA, s, '(placeholder)')

    # B: Regime x condition, mirrored UP/DOWN
    gsB = outer[1].subgridspec(2, len(cond_names), wspace=0.12, hspace=0.26)
    axB = np.empty((2, len(cond_names)), dtype=object)

    for i, regime in enumerate(regimes):
        for j, cond in enumerate(cond_names):
            ax = fig.add_subplot(gsB[i, j])
            _profile_panel(ax, s, results, regime, cond, sep_m, sep_deg, x_limit)
            axB[i, j] = ax

        ymax = max(abs(v) for ax in axB[i] for v in ax.get_ylim())
        for j, ax in enumerate(axB[i]):
            ax.set_ylim(-ymax, ymax)
            if j:
                ax.tick_params(labelleft=False)
            if i == 0:
                ax.tick_params(labelbottom=False)

        axB[i, 0].set_ylabel('DOWN  $\\leftarrow$  R1-R6 (a.u.)  $\\rightarrow$  UP',
                             fontsize=s.small, labelpad=2)

    for ax in axB.ravel():
        ax.set_xticks([-8, -4, 0, 4, 8])

    # C and D Grid
    gsCD = outer[2].subgridspec(1, 2, width_ratios=[1.35, 1.2], wspace=0.35)

    ex_regime, ex_cond = EXEMPLAR
    axC = fig.add_subplot(gsCD[0])
    _rhabdomere_panel(axC, s, results, ex_regime, ex_cond, sep_m, sep_deg, x_limit)
    axC.set_xticks([-8, -4, 0, 4, 8])
    axC.set_xlabel('Visual angle (deg)', fontsize=s.base, labelpad=1)
    axC.set_ylabel('DOWN  $\\leftarrow$  signal (a.u.)  $\\rightarrow$  UP',
                   fontsize=s.small, labelpad=2)
    axC.set_title(f'{ex_cond} / {ex_regime}', fontsize=s.title, loc='left', pad=3)

    # D: Direction dependence scalar summary
    axD = fig.add_subplot(gsCD[1])
    _summary_bars(axD, s, stats, cond_names, 'asym',
                  'Direction dependence', 'UP/DOWN shape $\\Delta$')
    axD.set_ylim(0, None)
    axD.yaxis.set_major_locator(MaxNLocator(nbins=4))

    reg_handles = [
        Patch(facecolor='white', edgecolor=s.dark, lw=0.7, label=regimes[0].title()),
        Patch(facecolor=s.dark, edgecolor=s.dark, lw=0.7, label=regimes[1].title()),
    ]
    leg = axD.legend(handles=reg_handles, loc='upper left', fontsize=s.tiny,
                     handlelength=1.0, handletextpad=0.4, borderpad=0.25,
                     labelspacing=0.22)
    leg.get_frame().set_edgecolor(s.frame)
    leg.get_frame().set_linewidth(0.3)
    leg.set_zorder(Z_TEXT)

    # Margins, headers
    fig.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.062)

    for j, cond in enumerate(cond_names):
        column_header(fig, s, axB[0, j], COND_LABEL[cond], dy=0.006,
                      color=COND_COLOR[cond], fontweight='bold')

    for i, regime in enumerate(regimes):
        row_header(fig, s, axB[i, 0],
                   f'{regime.title()}\n(r = {REGIMES[regime]:g})',
                   dx=0.062, colour=s.dark)

    pl, pr = axB[-1, 0].get_position(), axB[-1, -1].get_position()

    fig.text((pl.x0 + pr.x1) / 2, pl.y0 - 0.030, 'Visual angle (deg)',
             ha='center', va='top', fontsize=s.base, zorder=Z_TEXT)

    fig.text(pr.x1, pl.y0 - 0.030, f'bar centres {sep_deg:.0f}$\\degree$ apart',
             ha='right', va='top', fontsize=s.tiny, color=s.red, zorder=Z_TEXT)

    # Panels initials
    DX = -0.065
    panel_letter(fig, s, 'A', axA, dx=DX, dy=0.0)
    panel_letter(fig, s, 'B', axB[0, 0], dx=DX, dy=0.020)
    panel_letter(fig, s, 'C', axC, dx=DX, dy=0.010)
    panel_letter(fig, s, 'D', axD, dx=DX, dy=0.010)

    return fig


## --------------------------------------------------------------------------

if __name__ == '__main__':

    settings = PlotSettings.nature_double(height_mm=185.0).apply()

    results = {}
    for regime, r in REGIMES.items():
        print(f'\nRegime: {regime} (extra narrowing r = {r})')

        model = build_model(r)

        for sep in BAR_SEPARATIONS:
            label = 'one bar' if sep == 0 else f'{sep:.1f} deg'
            print(f'  simulating {label}...')
            results[(regime, sep)] = simulate(model, sep)

    fig = make_figure(results, settings)
    settings.savefig(fig, 'hyperacuity', formats=['png', 'pdf', 'svg'])

    plt.show()