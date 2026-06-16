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

The two sweep directions are kept separate (columns: Bar UP / Bar DOWN): the
rise/relax asymmetry means the bar-up and bar-down profiles differ, and that
direction-dependent signature is itself part of the hyperacuity code.

Two narrowing regimes, each rebuilt and plotted in its own figure:
    optical (r_xtra = 1.0): pure optical (Kemppainen 2022, ~2 um axial move).
                            Narrowing ~0.4 deg, so 'axial'/'full' do NOT resolve
                            the two bars below the optical Sparrow limit.
    phenomenological (r_xtra ~ 0.535): exaggerated dynamic RF contraction standing in for
                                       the pupil / transduction narrowing. 'axial'/'full'
                                       then resolve bars that 'none'/'lateral' cannot.
"""
import numpy as np
import matplotlib.pyplot as plt

from insectvision.compound_eyes import Model
from insectvision.compound_eyes.helpers.acceptance import SnyderAcceptance
from insectvision.compound_eyes.rhabdomeres import drosophila_bundle, RHAB_COLOURS
from insectvision.compound_eyes.helpers.alignment import BundlesAligner
from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.engine.meshes import plane_geom
from insectvision.engine.world_utils import WORLD_FORWARD
from insectvision.renderers import Raytracer

# Config ----------------------------------------------------------

SEP_TEST_DEG    = 4.0               # deg, centre-to-centre of the two-bar stimulus
BAR_SEPARATIONS = [0.0, SEP_TEST_DEG]   # 0 = single-bar control
BAR_WIDTH_DEG   = 1.0               # deg
DISTANCE        = 2.0               # m (bar plane at z = -DISTANCE)
BAR_LENGTH      = 10.0              # m

SWEEP_SPEED_DEG = 40.0              # deg/s (angular sweep speed at the centre of the field)
SWEEP_AMPLITUDE = 1.0               # m (travel is +/- this)

# Narrowing regimes, each rebuilt and given its own figure
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
COND_DURATION  = CYCLE_DURATION                            # one full cycle per condition
CONDITIONS = [
    # name,     actuation,   ampl_lat_um,   ampl_ax_um
    ('none',       False,       0.0,          0.0),
    ('axial',      True,        0.0,       AMP_AX),
    ('lateral',    True,    AMP_LAT,          0.0),
    ('full',       True,    AMP_LAT,       AMP_AX),
]
RUN_DURATION   = COND_DURATION * len(CONDITIONS)           # per separation

COND_COLOR = {'none': '#4477aa', 'axial': '#228833', 'lateral': '#ee7733', 'full': '#cc3311'}

R, CENTER = 7, 6                   # Drosophila: 7 rhabdomeres, R7/8 is the central (index 6)
PERIPH = list(range(6))            # R1-R6, the motion / acuity cells

context = Context()
context.time_step = 1 / 1000.0  # 1 ms biological resolution
context.mouse_captured = False


# Build the eye ----------------------------------------------------------

def build_eye(extra_narrowing):
    """Build the Drosophila eye for a given phenomenological narrowing ratio."""

    bundle = drosophila_bundle()
    bundle.extra_narrowing_ratio = extra_narrowing

    head_pitch = np.deg2rad(10.1)
    optic_flow = np.array([0.0, np.sin(head_pitch), np.cos(head_pitch)])

    aligner = BundlesAligner(
        flow_direction=optic_flow,
        diagonal_strength=1.0,
        diagonal_angle_deg=45.0,
        alignment_smoothing_iterations=4,
        saccade_smoothing_iterations=15,
    )

    model = Model.from_file(
        'species_models/drosophila_custom.npz',
        bundle=bundle, orientation=aligner, acceptance=SnyderAcceptance(),
        neural_superposition=True
    )

    model.refine_superposition(smooth_iters=2, relax=0.5, adjust_scale=True)

    model.scale(1e-6)

    model.rhabdomeres.tau_membrane = TAU_MEMBRANE
    model.ommatidia.tau_adapt_fast = TAU_FAST
    model.ommatidia.tau_adapt_slow = TAU_ADAPT
    model.ommatidia.tau_rise = TAU_RISE
    model.ommatidia.tau_relax = TAU_RELAX

    return model


# Scene / sweep helpers ----------------------------------------------------

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

    renderer.microsaccades_enabled = actuation

    model.ommatidia.lateral_amplitude = ampl_lat
    model.ommatidia.axial_amplitude = ampl_ax

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
    """Render the four actuation conditions for one bar separation; return per-timestep arrays."""

    bars = make_bars(sep_deg)

    scene = Scene(background_color=(0.0, 0.0, 0.0))
    scene.sun.elevation, scene.sun.azimuth, scene.sun.color = 1.0, 0.0, (1.0, 1.0, 1.0)
    for b in bars:
        scene.add_instance(b)

    agent = Agent(position=(0.0, 0.0, 0.0))

    renderer = Raytracer(
        model=model, scene=scene, agent=agent, context=context,
        nb_samples=512, time_dithering=True, randomness_mode='Halton',
        enable_microsaccades=True, enable_ambient=True, enable_direct=True, enable_shadows=False)

    renderer.ambient_intensity = 1.5
    renderer.photon_concentration = 0.0    # c = 0: isolate RF geometry from the photon-concentration gain

    # Single forward-pointing cartridge (closest optical axis to straight ahead)
    cone = model.query_cone(WORLD_FORWARD, angle=10.0, degrees=True, avoid_conflicts=True)
    if len(cone) == 0:
        raise RuntimeError("No forward-facing ommatidia found.")

    az = np.rad2deg(cone.azimuth)
    el = np.rad2deg(cone.elevation)
    selected = int(cone.indices[int(np.argmin(az ** 2 + el ** 2))])
    renderer.selected_ommatidia = [selected]

    rec = {'agent_y': [], 'cond': [], 'mdir': [], 'cart': []}

    t_start, phase = None, -1
    while context.run_interactive(agent=agent, scene=scene, renderer=renderer, use_dashboard=False):

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

        context.draw(out)
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


# Analysis ------------------------------------------------------------------

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
    """Binned signal for one condition / sweep direction. channel = receptor index or 'pool'."""

    m = (res['cond'] == cond) & (res['mdir'] == direction)
    sig = res['cart'][m][:, PERIPH].mean(axis=1) if channel == 'pool' else res['cart'][m][:, channel]
    return binned(res['agent_y'][m], sig)


def find_dip(grid, prof, sep_m, min_depth=0.02):
    """Locate the central notch of a two-bar profile.

    Finds the flanking maxima on either side of centre and the minimum between them.
    Returns (x_dip, y_dip, depth) with depth in [0, 1], or None if no real dip
    (i.e. the two bars fuse into a single peak).
    """

    half = sep_m / 2
    left = np.where((grid >= -1.6 * half) & (grid <= 0.0))[0]
    right = np.where((grid >= 0.0) & (grid <= 1.6 * half))[0]
    if left.size < 2 or right.size < 2:
        return None

    il = left[int(np.argmax(prof[left]))]
    ir = right[int(np.argmax(prof[right]))]
    if ir <= il:
        return None

    iv = il + int(np.argmin(prof[il:ir + 1]))
    peak = 0.5 * (prof[il] + prof[ir])
    depth = (peak - prof[iv]) / (peak + 1e-9)

    if depth < min_depth or iv == il or iv == ir:
        return None

    return grid[iv], prof[iv], float(depth)


def updown_shape_divergence(grid, up, down):
    """UP vs DOWN difference in profile SHAPE, after removing a pure spatial shift.

    Neural superposition makes the pooled cartridge peak at a different agent-Y for
    the two sweep directions (the leading neighbour is above the cartridge on the way
    down, below it on the way up). That is an uninteresting translation along X. We
    slide one profile against the other to the lag that best matches them, then report
    the residual mismatch (range-normalised, in [0, 1]) the shift could not explain --
    i.e. genuine changes in the shape of the response (as in 'lateral' / 'full')."""

    mask = np.abs(grid) <= VIEW_HALFWIDTH
    u = up[mask] - up[mask].min()
    d = down[mask] - down[mask].min()
    n = u.size
    max_lag = n // 3

    def overlap(k):                          # u against d shifted by k samples
        a = u[k:] if k >= 0 else u[:n + k]
        b = d[:n - k] if k >= 0 else d[-k:]
        return a, b

    k_best = min(range(-max_lag, max_lag + 1),
                 key=lambda k: np.mean((np.subtract(*overlap(k))) ** 2))

    a, b = overlap(k_best)
    span = max(u.max(), d.max()) + 1e-9
    return float(np.mean(np.abs(a - b)) / span)


# Figure --------------------------------------------------------------------

def make_figure(results, regime):
    """One figure per regime: conditions (rows) x sweep direction (cols)."""

    res2 = results[(regime, SEP_TEST_DEG)]    # two-bar
    res1 = results[(regime, 0.0)]             # single-bar control
    sep_m = 2.0 * DISTANCE * np.tan(np.radians(SEP_TEST_DEG) / 2.0)
    cond_names = [c[0] for c in CONDITIONS]
    cols = [(+1, 'Bar UP'), (-1, 'Bar DOWN')]

    fig, axs = plt.subplots(len(cond_names), 2, figsize=(12, 14), sharex=True, sharey='row')

    for row, cond in enumerate(cond_names):

        # UP/DOWN divergence of the pooled two-bar response (one number per condition)
        gu, up = trace(res2, cond, +1, 'pool')
        gd, dn = trace(res2, cond, -1, 'pool')
        asym = updown_shape_divergence(gu, up, dn)
        pooled_dir = {+1: (gu, up), -1: (gd, dn)}

        for col, (direction, dlabel) in enumerate(cols):
            ax = axs[row, col]
            grid, pooled = pooled_dir[direction]

            # individual rhabdomeres (two-bar)
            for i in range(R):
                gi, ri = trace(res2, cond, direction, i)
                label = (f'R{i + 1}' if i < CENTER else 'R7/8')
                ax.plot(gi, ri, color=RHAB_COLOURS[i], lw=1.0, alpha=0.6, label=label)

            # single-bar control (dashed) and opposite sweep direction (dotted) for reference
            gc, ctrl = trace(res1, cond, direction, 'pool')
            ax.plot(gc, ctrl, color='#7f8fb0', lw=1.3, ls='--', alpha=0.8, label='1 bar (control)')

            other = pooled_dir[-direction][1]
            ax.plot(grid, other, color=COND_COLOR[cond], lw=1.2, ls=':', alpha=0.55, label='other dir')

            # pooled two-bar (thick) + dip marker
            ax.plot(grid, pooled, color='k', lw=2.2, label='R1-R6 (2 bars)')
            dip = find_dip(grid, pooled, sep_m)
            if dip is not None:
                xd, yd, depth = dip
                ax.plot(xd, yd, marker='v', color='k', ms=9, zorder=7)
                ax.annotate(f'dip {depth:.2f}', (xd, yd), textcoords='offset points',
                            xytext=(5, 6), fontsize=8, fontweight='bold')

            for xb in (-sep_m / 2, +sep_m / 2):
                ax.axvline(xb, color='r', ls=':', alpha=0.4)

            ax.text(0.02, 0.96, f'UP/DOWN shape \u0394 = {asym:.2f}', transform=ax.transAxes,
                    fontsize=8, va='top', color='grey')
            ax.set_title(f'{cond}  |  {dlabel}')
            ax.set_xlim(-VIEW_HALFWIDTH, VIEW_HALFWIDTH)
            # ax.set_ylim(bottom=-0.025)
            ax.grid(True, alpha=0.3)

        axs[row, 0].set_ylabel('signal intensity')

    axs[-1, 0].set_xlabel('agent Y position (m)')
    axs[-1, 1].set_xlabel('agent Y position (m)')
    axs[0, 1].legend(fontsize=7, loc='upper right', ncol=2)

    fig.suptitle(f'Microsaccadic hyperacuity \u2014 {regime} regime (r = {REGIMES[regime]})  '
                 f'|  two bars {SEP_TEST_DEG:.1f}\u00b0 apart (red dotted)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# Run -----------------------------------------------------------------------

if __name__ == '__main__':

    for regime, r in REGIMES.items():
        print(f'\n=== regime: {regime} (extra narrowing r = {r}) ===')

        model = build_eye(r)

        results = {}
        for sep in BAR_SEPARATIONS:
            label = 'one bar' if sep == 0 else f'{sep:.1f} deg'
            print(f'  simulating {label}...')
            results[(regime, sep)] = simulate(model, sep)

        free = getattr(model, 'free', None)
        if callable(free):
            try:
                free()
            except Exception:
                pass

        fig = make_figure(results, regime)
        fig.savefig(f'hyperacuity_{regime}.png', dpi=200)
        # fig.savefig(f'hyperacuity_{regime}.pdf')

    plt.show()