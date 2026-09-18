"""
Photoreceptor dynamics proof-of-concept: pupil adaptation x microsaccadic actuation.

Drosophila photomechanical microsaccades sweep ~vertically in visual space.
So: thin horizontal bars, agent oscillates vertically, readout is a
single forward-pointing cartridge.

Two dynamics mechanisms, crossed on the same cartridge:
    - pupil adaptation: three forced states (dark/halfway/light-adapted, via
      Renderer.pupil_drive), narrowing and dimming the RF by absorbing higher-order
      waveguide modes
    - microsaccade actuation, four conditions:
        none     - no microsaccade (static optical RF)
        axial    - axial move only   -> RF narrowing, no lateral shift
        lateral  - lateral move only -> RF shift + clipping, no axial narrowing
        full     - both

Run against two bar separations (single control bar + the test separation)

Move/return durations differ, so bar-up and bar-down sweeps give different response
profiles, panel D's UP/DOWN divergence shows that asymmetry.

Figure layout:
    A: Schematic placeholder (stimulus, sweep, cartridge)
    B: Pupil state (rows) x condition (cols), mirrored UP/DOWN pooled R1-R6 profiles
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

from rhabdoforge.compound_eyes import Model
from rhabdoforge.compound_eyes.helpers.waveguide import WaveguideAcceptance
from rhabdoforge.compound_eyes.rhabdomeres import drosophila_bundle
from rhabdoforge.types import WORLD_FORWARD, RHAB_COLOURS
from rhabdoforge.compound_eyes.helpers.alignment import BundlesAligner
from rhabdoforge.engine import Context, Agent, Scene, Asset
from rhabdoforge.engine.meshes import plane_geom
from rhabdoforge.renderers import Renderer

from visualisation.plot_settings import (
    PlotSettings, Z_RASTER, Z_TEXT, panel_letter, column_header, row_header, despine, placeholder
)


# Config

SEP_TEST_DEG    = 4.0                   # deg, centre-to-centre of the 2-bar stimulus
BAR_SEPARATIONS = [0.0, SEP_TEST_DEG]   # 0 = single-bar control
BAR_WIDTH_DEG   = 1.0                   # deg
DISTANCE        = 2.0                   # m (bar plane at z = -DISTANCE)
BAR_LENGTH      = 10.0                  # m

SWEEP_SPEED_DEG = 40.0              # deg/s (angular sweep speed at the centre of the field)
SWEEP_AMPLITUDE = 1.0               # m (travel is +/- this)

# Forced steady_state_drive per pupil adaptation state
PUPIL_STATES = {'dark-adapted': 0.0, 'halfway': 0.5, 'light-adapted': 1.0}

CLIP_RATIO = 0.55

bundle = drosophila_bundle()

# All values from the drosophila bundle
AMP_LAT      = bundle.ampl_lat_um
AMP_AX       = bundle.ampl_ax_um
TAU_MEMBRANE = bundle.tau_membrane
TAU_FAST     = bundle.tau_fast
TAU_ADAPT    = bundle.tau_adapt

# Microsaccade implemented as a reflex arc: a threshold crossing launches a fixed, committed
# sequence (latency -> ballistic move -> interruptible return)

NOISE_THRESHOLD = 0.02      # contrast deadzone below which a trigger doesn't fire

VIEW_HALFWIDTH = 0.30       # m, x-axis half-range around the bars

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

EXEMPLAR = ('dark-adapted', 'full')     # condition shown broken down per rhabdomere in panel C

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


def build_model():
    """Build the Drosophila eyes model used for every run; pupil state is forced per-run at render time."""

    bundle = drosophila_bundle()
    bundle.clip_ratio = CLIP_RATIO

    bundle.ampl_ax_um = AMP_AX  # axial_scale_floor is baked from this at build time, so it must match the runtime axial amplitude

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
        acceptance=WaveguideAcceptance(),
        orientation=aligner,
        neural_superposition=True,
    )

    model.refine_superposition(smooth_iters=5, relax=0.8, adjust_scale=True, adjust_anisotropy=True, rewire=True)
    model.refine_superposition(smooth_iters=5, relax=0.8, adjust_scale=True, adjust_anisotropy=True, rewire=False)

    model.scale(1e-6)

    model.tau_membrane = TAU_MEMBRANE
    model.tau_adapt_fast = TAU_FAST
    model.tau_adapt_slow = TAU_ADAPT

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
    model.buffer.ommatidia_dynamic['mech_phase'] = 0.0
    model.buffer.ommatidia_dynamic['mech_t'] = 0.0
    model.buffer.ommatidia_dynamic['mech_frac0'] = 0.0

    model.buffer.ommatidia_stale = True


def sweep_position(elapsed):
    """Agent Y (m) and apparent-bar-motion direction for the time within one up/down cycle."""

    t = elapsed % CYCLE_DURATION
    if t < SWEEP_DURATION:                 # agent up (+y) -> bar appears to move down
        return -SWEEP_AMPLITUDE + (t / SWEEP_DURATION) * 2 * SWEEP_AMPLITUDE, -1

    return SWEEP_AMPLITUDE - ((t - SWEEP_DURATION) / SWEEP_DURATION) * 2 * SWEEP_AMPLITUDE, +1


def simulate(model, sep_deg, pupil_drive):
    """
    Render the four actuation conditions for one bar separation, at a forced pupil adaptation
    state (steady_state_drive, in [0, 1]). Returns per-timestep arrays.
    """

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
        enable_microsaccades=True,
        enable_ambient=True, enable_direct=True, enable_shadows=False)

    renderer.ambient_intensity = 1.5
    renderer.photon_concentration = 0.0     # isolate RF geometry from the photon-concentration gain
    renderer.pupil_drive = pupil_drive      # fixed light adaptation state
    renderer.noise_threshold = NOISE_THRESHOLD

    # Single forward-pointing cartridge (closest optical axis to straight ahead)
    cone = model.query_cone(WORLD_FORWARD, angle=10.0, degrees=True, avoid_conflicts=True)
    if len(cone) == 0:
        raise RuntimeError('No forward-facing ommatidia found.')

    az = np.rad2deg(cone.azimuth)
    el = np.rad2deg(cone.elevation)
    selected = int(cone.indices[int(np.argmin(az ** 2 + el ** 2))])
    renderer.selected_ommatidia = [selected]

    rec = {'agent_y': [], 'cond': [], 'mdir': [], 'cart': [], 'axial_disp': [], 'lateral_disp': [], 'drho_deg': []}

    rest_acc = np.rad2deg(model.buffer['rest_acc_angles'][selected, :, 0])

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

        # Diagnostic: read this ommatidium's actuation state
        dyn = renderer.eye_buffers['omm_dynamic'].read(start=selected, count=1)[0]

        rec['agent_y'].append(ay)
        rec['cond'].append(name)
        rec['mdir'].append(mdir)
        rec['cart'].append(out.per_cartridge[selected, :, :3].mean(axis=-1))
        rec['axial_disp'].append(float(dyn['curr_axial_disp']))
        rec['lateral_disp'].append(float(dyn['curr_lateral_disp']))

        rhab = renderer.eye_buffers['rhab_dynamic'].read(start=selected * R, count=R)
        rec['drho_deg'].append(np.rad2deg(rhab['curr_acc_angles'][PERIPH, 0]))

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

    out_rec = {k: np.array(v) for k, v in rec.items()}
    out_rec['rest_drho_deg'] = rest_acc
    return out_rec


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


def print_actuation_diagnostics(res, label):
    """Fraction of AMP_AX/AMP_LAT actually reached per condition, measured from the run."""

    print(f'  [{label}] peak actuation reached (fraction of amplitude):')
    for cond in [c[0] for c in CONDITIONS]:
        m = res['cond'] == cond
        if not m.any():
            continue
        ax_frac = np.max(res['axial_disp'][m]) / AMP_AX if AMP_AX > 0 else 0.0
        lat_frac = np.max(res['lateral_disp'][m]) / AMP_LAT if AMP_LAT > 0 else 0.0
        print(f'    {cond:8s}: axial {ax_frac:.2f}   lateral {lat_frac:.2f}')

    print_drho_diagn(res, label)


def print_drho_diagn(res, label):
    """Acceptance angle reached per condition per receptor."""
    rest = res.get('rest_drho_deg')
    if 'drho_deg' not in res or res['drho_deg'].size == 0:
        return

    DIP_THRESH = 0.95 * SEP_TEST_DEG

    print(f'  [{label}] acceptance angle reached, R1-R6 (deg):')
    if rest is not None:
        print(f'    {"rest":8s}: {rest[PERIPH].min():.2f} - {rest[PERIPH].max():.2f}')

    for cond in [c[0] for c in CONDITIONS]:

        m = res['cond'] == cond
        if not m.any():
            continue

        d = res['drho_deg'][m]
        lo, hi = d.min(axis=0), d.max(axis=0)   # per receptor

        if rest is not None:
            rr = lo / rest[PERIPH]
            ratio_txt = f'x{rr.min():.2f}-{rr.max():.2f} of rest'
        else:
            ratio_txt = ''

        n45, n37 = int((lo <= 4.5).sum()), int((lo <= 3.7).sum())
        sharp = d.min(axis=1)      # sharpest receptor

        pct = 100.0 * float(np.mean(sharp <= DIP_THRESH))

        print(f'    {cond:8s}: narrowest {lo.min():.2f} - {lo.max():.2f}   '
              f'widest {hi.max():.2f}   ({ratio_txt})   '
              f'{n45}/6 <=4.5, {n37}/6 <=3.7')

        print(f'    {"":8s}  mean {d.mean():.2f}   median {np.median(d):.2f}   '
              f'{pct:.0f}% of frames <= {DIP_THRESH} (dip threshold)')


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


def _curvature_peak(d2, i, sign):

    L = i
    while L > 0 and np.sign(d2[L - 1]) == sign:
        L -= 1
    if L == i:
        return i
    seg = d2[L:i]
    return L + (int(np.argmax(seg)) if sign > 0 else int(np.argmin(seg)))


def find_dip(grid, prof, view=0.30, min_peak=0.1, deriv_win=9, shoulder_prominence=0.10, shoulder_noise_floor=0.03):

    m = np.abs(grid) <= view
    g, p = np.asarray(grid)[m], np.asarray(prof)[m]
    if p.size < 11 or p.max() <= 0:
        return None

    win = max(7, deriv_win | 1)
    win = min(win, p.size - (p.size + 1) % 2)
    dx = g[1] - g[0]

    d1 = savgol_filter(p, win, 2, deriv=1, delta=dx)
    d2 = savgol_filter(p, win, 3, deriv=2, delta=dx)

    base = float(np.percentile(p, 5))
    scale = p.max() - base + 1e-9
    floor = base + min_peak * scale
    d1_scale = np.ptp(d1) + 1e-9

    edge = win // 2 + 1
    dips, shoulders = [], []
    for i in range(edge, len(d1) - edge):

        if d1[i - 1] < 0 <= d1[i]:
            p_left, p_right = p[:i].max(), p[i:].max()
            depth = (min(p_left, p_right) - p[i]) / scale
            if depth > 1e-9:
                (dips if (p_left > floor and p_right > floor) else shoulders).append((depth, i))
            continue

        if d1[i] > 0 and d1[i - 1] >= d1[i] and d1[i + 1] > d1[i]:      # rising flank stalls
            bound = min(d1[:i].max(), d1[i:].max())
            raw = bound - d1[i]
            curve_sign = -1.0
        elif d1[i] < 0 and d1[i - 1] <= d1[i] and d1[i + 1] < d1[i]:    # falling flank stalls
            bound = max(d1[:i].min(), d1[i:].min())
            raw = d1[i] - bound
            curve_sign = 1.0
        else:
            continue

        if raw < shoulder_noise_floor * d1_scale:
            continue
        rel = raw / (abs(bound) + 1e-9)
        if rel > shoulder_prominence:
            shoulders.append((rel, _curvature_peak(d2, i, curve_sign)))

    for candidates, kind in ((dips, 'dip'), (shoulders, 'shoulder')):
        if not candidates:
            continue
        depth, i = max(candidates, key=lambda c: c[0])

        if kind == 'dip':
            y0, y1, y2 = p[i - 1], p[i], p[i + 1]
            denom = y0 - 2.0 * y1 + y2
            frac = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0)) if abs(denom) > 1e-12 else 0.0
            g_dip, p_dip = g[i] + frac * dx, y1 - 0.25 * (y0 - y2) * frac
        else:
            g_dip, p_dip = g[i], p[i]

        return (float(g_dip), float(p_dip), float(depth), kind)

    return None


def _get_signal_x_limit(results, cond_names):

    max_ext_deg = 1.0
    for res in results.values():
        for cond in cond_names:
            for mdir in [-1, 1]:
                g, p = trace(res, cond, mdir, 'pool')
                if p.max() <= 0: continue
                gc, pc = get_centered_data(g, p)
                in_view = np.abs(gc) <= VIEW_HALFWIDTH
                active = np.where(in_view & (pc > 0.01 * pc.max()))[0]
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

def _cell_stats(results, pupil_state, cond):
    """Everything panel D needs for one (pupil_state, condition) cell."""

    res2 = results[(pupil_state, SEP_TEST_DEG)]

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


def _profile_panel(ax, s: PlotSettings, results, pupil_state, cond, sep_m, sep_deg, x_lim):
    """Mirrored pooled R1-R6 profile (UP is up / DOWN is down)."""

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)

    res2 = results[(pupil_state, SEP_TEST_DEG)]
    res1 = results[(pupil_state, 0.0)]
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
        if dip is not None:
            xd, yd, depth, kind = dip

            if kind == 'shoulder' or depth >= DIP_MIN_DEPTH:
                ax.plot(to_deg(xd), sign * yd,
                        marker=('v' if sign > 0 else '^') if kind == 'dip' else 's',
                        color=colour, ms=3.0, mec='white', mew=0.4, zorder=7)

    _mirror_axes(ax, s, sep_deg)
    ax.set_xlim(-x_lim, x_lim)
    return ax


def _rhabdomere_panel(ax, s: PlotSettings, results, pupil_state, cond, sep_m, sep_deg, x_lim):
    """Mirrored per-rhabdomere traces for the exemplar condition (panel C)."""

    if s.rasterize:
        ax.set_rasterization_zorder(Z_RASTER)

    res2 = results[(pupil_state, SEP_TEST_DEG)]

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
    Grouped bars: one group per condition, one bar per pupil state (dark -> light adapted,
    shown as increasing fill alpha, same edge colour per condition).
    """
    pupil_states = list(PUPIL_STATES)
    n = len(pupil_states)
    width = 0.68 / n
    offs = {pst: (i - (n - 1) / 2) * width for i, pst in enumerate(pupil_states)}

    for j, cond in enumerate(cond_names):
        for i, pst in enumerate(pupil_states):
            v = stats[(pst, cond)][key]
            x = j + offs[pst]
            alpha = 0.15 + 0.85 * (i / max(n - 1, 1))     # dark-adapted -> near-white, light -> full colour
            ax.bar(x, v, width=width * 0.92,
                   facecolor=COND_COLOR[cond], alpha=alpha,
                   edgecolor=COND_COLOR[cond], linewidth=0.7, zorder=3)

            if dot_keys:
                for dk, mk in dot_keys:
                    ax.plot(x, stats[(pst, cond)][dk], marker=mk, ms=2.6,
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
    pupil_states = list(PUPIL_STATES)

    stats = {(pst, c): _cell_stats(results, pst, c)
             for pst in pupil_states for c in cond_names}
    x_limit = _get_signal_x_limit(results, cond_names)

    fig = s.new_figure()
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.90, 0.95], hspace=0.46)

    # A: Schematic placeholder
    axA = fig.add_subplot(outer[0])
    placeholder(axA, s, '(placeholder)')

    # B: Pupil state x condition, mirrored UP/DOWN
    n_states = len(pupil_states)
    gsB = outer[1].subgridspec(n_states, len(cond_names), wspace=0.12, hspace=0.26)
    axB = np.empty((n_states, len(cond_names)), dtype=object)

    for i, pupil_state in enumerate(pupil_states):
        for j, cond in enumerate(cond_names):
            ax = fig.add_subplot(gsB[i, j])
            _profile_panel(ax, s, results, pupil_state, cond, sep_m, sep_deg, x_limit)
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

    ex_state, ex_cond = EXEMPLAR
    axC = fig.add_subplot(gsCD[0])
    _rhabdomere_panel(axC, s, results, ex_state, ex_cond, sep_m, sep_deg, x_limit)
    axC.set_xticks([-8, -4, 0, 4, 8])
    axC.set_xlabel('Visual angle (deg)', fontsize=s.base, labelpad=1)
    axC.set_ylabel('DOWN  $\\leftarrow$  signal (a.u.)  $\\rightarrow$  UP',
                   fontsize=s.small, labelpad=2)
    axC.set_title(f'{ex_cond} / {ex_state}', fontsize=s.title, loc='left', pad=3)

    # D: Direction dependence scalar summary
    axD = fig.add_subplot(gsCD[1])
    _summary_bars(axD, s, stats, cond_names, 'asym',
                  'Direction dependence', 'UP/DOWN shape $\\Delta$')
    axD.set_ylim(0, None)
    axD.yaxis.set_major_locator(MaxNLocator(nbins=4))

    n_states = len(pupil_states)
    state_handles = [
        Patch(facecolor=s.dark, edgecolor=s.dark,
              alpha=0.15 + 0.85 * (i / max(n_states - 1, 1)), lw=0.7, label=pst.title())
        for i, pst in enumerate(pupil_states)
    ]
    leg = axD.legend(handles=state_handles, loc='upper left', fontsize=s.tiny,
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

    for i, pupil_state in enumerate(pupil_states):
        row_header(fig, s, axB[i, 0],
                   f'{pupil_state.title()}',
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

    model = build_model()

    results = {}
    for pupil_state, drive in PUPIL_STATES.items():
        print(f'\nPupil state: {pupil_state} (forced steady_state_drive = {drive})')

        for sep in BAR_SEPARATIONS:
            label = 'one bar' if sep == 0 else f'{sep:.1f} deg'
            print(f'  simulating {label}...')
            res = simulate(model, sep, pupil_drive=drive)
            results[(pupil_state, sep)] = res
            if sep == SEP_TEST_DEG:
                print_actuation_diagnostics(res, f'{pupil_state}, {label}')

    fig = make_figure(results, settings)
    settings.savefig(fig, 'dynamics', formats=['png', 'pdf', 'svg'])

    plt.show()