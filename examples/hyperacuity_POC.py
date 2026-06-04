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

Several bar separations run sequentially.

Two regimes, switched by EXTRA_NARROWING:
    1.0: pure optical (Kemppainen 2022, Table S6, 2 um axial move). This produces a narrowing of ~0.4 deg,
        so 'axial'/'full' do NOT resolve below the optical Sparrow limit: pure optics + saccades are not hyperacute.
    ~0.535: phenomenological. An exaggerated dynamic RF contraction standing in for the pupil /
          transduction narrowing. 'axial'/'full' then resolve bars that 'none'/'lateral' cannot.
"""
import numpy as np
import matplotlib.pyplot as plt

from insectvision.compound_eyes.acceptance import SnyderAcceptance
from insectvision.compound_eyes.rhabdomeres import drosophila_bundle, RHAB_COLOURS
from insectvision.compound_eyes.orientation import BundlesAligner
from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.compound_eyes import CompoundEyeModel
from insectvision.engine.world_utils import WORLD_FORWARD
from insectvision.renderers import Raytracer
from insectvision.geometry import plane_geom


# Config ----------------------------------------------------------

BAR_SEPARATIONS = [0.0, 4.0]        # deg, centre-to-centre
BAR_WIDTH_DEG   = 1.0               # deg
DISTANCE        = 2.0               # m (bar plane at z = -DISTANCE)
BAR_LENGTH      = 10.0              # m

SWEEP_SPEED_DEG = 40.0              # deg/s (angular sweep speed at the centre of the field)
SWEEP_AMPLITUDE = 1.0               # m (travel is +/- this)

SELECTED_LENS   = None              # the single analysed cartridge (forward-pointing)

EXTRA_NARROWING = 1.0               # 1.0 = pure optical
# EXTRA_NARROWING = 0.535             # ~0.535 = phenomenological (see header)

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

FOCUS_R_TYPE = 5    # Single receptor type to plot

SWEEP_SPEED    = DISTANCE * np.radians(SWEEP_SPEED_DEG)     # m/s
SWEEP_DURATION = (2 * SWEEP_AMPLITUDE) / SWEEP_SPEED        # one-way sweep (s)
CYCLE_DURATION = 2 * SWEEP_DURATION                        # up + down (s)
COND_DURATION  = CYCLE_DURATION                            # one full cycle per condition
CONDITIONS = [
    # name,     actuation,   ampl_lat_um,   ampl_ax_um
    ('none',       False,       0.0,          0.0),
    ('axial',      True,        0.0,       AMP_AX),
    ('lateral',    True,    AMP_LAT,          0.0),
    ('full',       True,    AMP_LAT,       AMP_AX),
]
RUN_DURATION   = COND_DURATION * len(CONDITIONS)           # per separation

# Build the eye ----------------------------------------------------------

context = Context()
context.time_step = 1 / 1000.0  # 1 ms biological resolution
context.mouse_captured = False

bundle = drosophila_bundle()
bundle.extra_narrowing_ratio = EXTRA_NARROWING

head_ptich = np.deg2rad(10.1)
optic_flow = np.array([0.0, np.sin(head_ptich), np.cos(head_ptich)])

aligner = BundlesAligner(
    flow_direction=optic_flow,
    diagonal_strength=1.0,
    diagonal_angle_deg=45.0,
    alignment_smoothing_iterations=4,
    saccade_smoothing_iterations=15,
)

model = CompoundEyeModel.from_file(
    'species_models/drosophila_custom.npz',
    bundle=bundle, orientation=aligner, acceptance=SnyderAcceptance(),
)
model.calibrate_superposition_alignment()
model.scale(1e-6)


with model.unlock(lenses=True, receptors=True):
    model.receptors.tau_membrane = TAU_MEMBRANE
    model.lenses.tau_fast = TAU_FAST
    model.lenses.tau_adapt = TAU_ADAPT
    model.lenses.tau_rise = TAU_RISE
    model.lenses.tau_relax = TAU_RELAX

# Helpers ------------------------------------------------------------------

COND_COLOR = {'none': '#4477aa', 'axial': '#228833', 'lateral': '#ee7733', 'full': '#cc3311'}

R, CENTER = 7, 6                   # Drosophila: 7 rhabdomeres, R7/8 is the central (index 6)
PERIPH = list(range(6))            # R1-R6, the motion / acuity cells

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


def apply_condition(renderer, actuation, ampl_lat, ampl_ax):
    """Set actuation + microsaccade amplitudes and reset the per-lens dynamic state."""

    renderer.actuation = actuation

    with model.unlock(lenses=True):
        model.lenses.ampl_lat_um = float(ampl_lat)
        model.lenses.ampl_ax_um = float(ampl_ax)
        model.lens_dynamic_data['lateral_um'] = 0.0
        model.lens_dynamic_data['axial_um'] = 0.0
        model.lens_dynamic_data['adapted_lum'] = 0.0
        model.lens_dynamic_data['fast_lum'] = 0.0

    model.buffer.lens_dirty = True


def sweep_position(elapsed):
    """Agent Y (m) and apparent-bar-motion direction for the time within one up/down cycle."""

    t = elapsed % CYCLE_DURATION
    if t < SWEEP_DURATION:                 # agent up (+y) -> bar appears to move down
        return -SWEEP_AMPLITUDE + (t / SWEEP_DURATION) * 2 * SWEEP_AMPLITUDE, -1

    return SWEEP_AMPLITUDE - ((t - SWEEP_DURATION) / SWEEP_DURATION) * 2 * SWEEP_AMPLITUDE, +1


def simulate(sep_deg):
    """Render the four conditions for one bar separation; return per-timestep arrays."""

    global SELECTED_LENS
    bars = make_bars(sep_deg)

    scene = Scene(background_color=(0.0, 0.0, 0.0))
    scene.sun.elevation, scene.sun.azimuth, scene.sun.color = 1.0, 0.0, (1.0, 1.0, 1.0)

    for b in bars:
        scene.add_instance(b)

    agent = Agent(position=(0.0, 0.0, 0.0))

    renderer = Raytracer(
        model=model, scene=scene, agent=agent, context=context,
        nb_samples=512, time_dithering=True, randomness_mode='Halton',
        enable_actuation=True, enable_ambient=True, enable_direct=True, enable_shadows=False)

    renderer.ambient_intensity = 1.5
    renderer.photon_concentration = 0.0

    if SELECTED_LENS is None:
        cone = model.query_cone(WORLD_FORWARD, angle=10.0, degrees=True, avoid_conflicts=True)
        if len(cone) == 0:
            raise RuntimeError("No forward-facing ommatidia found.")
        SELECTED_LENS = int(cone.indices[int(np.argmin(cone.azimuth_deg ** 2 + cone.elevation_deg ** 2))])

    drho = np.degrees(model.rcpt_dynamic_data['acc_axes'][SELECTED_LENS * R + PERIPH[0], 0])
    print(f"Cartridge {SELECTED_LENS}: R1 rest acceptance Δρ = {drho:.2f} deg, narrowed = {drho * EXTRA_NARROWING:.2f} deg (x{EXTRA_NARROWING})")

    renderer.selected_lenses = [SELECTED_LENS]

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
            apply_condition(renderer, *CONDITIONS[phase][1:])

        name = CONDITIONS[phase][0]

        ay, mdir = sweep_position(elapsed)
        agent.position = (0.0, ay, 0.0)

        out = renderer.step()

        rec['agent_y'].append(ay)
        rec['cond'].append(name)
        rec['mdir'].append(mdir)
        rec['cart'].append(out.per_cartridge[SELECTED_LENS, :, :3].mean(axis=-1))

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


# Plotting ------------------------------------------------------------------

def scatter_receptor(ax, agent_y, cond, mdir, signal, direction, bar_m, title):
    """One receptor trace as a scatter"""

    for name, color in COND_COLOR.items():
        m = (cond == name) & (mdir == direction)
        ax.scatter(agent_y[m], signal[m], s=3, alpha=0.5, color=color, label=name)

    for xb in ([-bar_m / 2, +bar_m / 2] if bar_m > 0 else [0.0]):
        ax.axvline(xb, color='k', ls=':', alpha=0.35)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.set_xlim([-0.75, 0.75])


def line_individual_receptors(ax, agent_y, cond, mdir, cart, direction, bar_m, title):
    """Individual rhabdomere traces as lines"""

    m = (cond == 'full') & (mdir == direction)
    order = np.argsort(agent_y[m])
    ys = agent_y[m][order]

    for i in range(R):
        label = f'R{i + 1}' if i < CENTER else 'R7/8'
        ax.plot(ys, cart[m][order, i], color=RHAB_COLOURS[i], lw=1.3, alpha=0.85, label=label)

    for xb in ([-bar_m / 2, +bar_m / 2] if bar_m > 0 else [0.0]):
        ax.axvline(xb, color='k', ls=':', alpha=0.35)

    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax.set_xlim([-0.75, 0.75])


def plot_single_run(res, sep_deg, bars_m):
    """Per-run figure: R7/8, R1-R6 pool, individual cells. Bar UP / Bar DOWN."""

    ay, cond, mdir, cart = res['agent_y'], res['cond'], res['mdir'], res['cart']

    # Rows
    signals_to_plot = {
        'R7/8 (central): ':         cart[:, CENTER],
        f'R{FOCUS_R_TYPE + 1}: ':   cart[:, FOCUS_R_TYPE],
        # 'R1-R6 (pool): ':           cart[:, PERIPH].sum(axis=1),
    }
    nrows = len(signals_to_plot) + 1
    lastrow = nrows - 1

    fig, axs = plt.subplots(nrows, 2, figsize=(11, 11), sharex=True)
    for col, (direction, dlabel) in enumerate(((+1, 'Bar UP'), (-1, 'Bar DOWN'))):

        for r, (row_label, row_data) in enumerate(signals_to_plot.items()):
            scatter_receptor(axs[r, col], ay, cond, mdir, row_data, direction, bars_m, row_label + dlabel)

        # The per-receptor line plot is always there in the last row
        line_individual_receptors(axs[lastrow, col], ay, cond, mdir, cart, direction, bars_m, f"Individual cells, full: {dlabel}")

    for r in range(nrows):
        axs[r, 0].set_ylabel("signal intensity")

    axs[lastrow, 0].set_xlabel("agent Y position (m)")
    axs[lastrow, 1].set_xlabel("agent Y position (m)")
    axs[1, 1].legend(loc='upper right', markerscale=4, fontsize=8)
    axs[lastrow - 1, 1].legend(loc='upper right', fontsize=8)

    title = ("one bar (control)" if sep_deg == 0 else f"two bars {sep_deg:.1f} deg apart")
    fig.suptitle(f"{title}   |   narrowing = {EXTRA_NARROWING}   "
                 f"(dotted = true bar positions)", y=1.0)

    plt.tight_layout()


def norm(v):
    """Normalise a vector to [0, 1] for visual overlays."""
    v = np.asarray(v, float)
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    return (v - vmin) / (vmax - vmin + 1e-9)


def binned(y, sig, bins=200):
    """Clean spatial binning for aligning different runs"""

    edges = np.linspace(-SWEEP_AMPLITUDE, SWEEP_AMPLITUDE, bins + 1)
    grid = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(y, edges) - 1, 0, bins - 1)
    out, cnt = np.zeros(bins), np.zeros(bins)
    np.add.at(out, idx, sig)
    np.add.at(cnt, idx, 1)
    nz = cnt > 0
    out[nz] /= cnt[nz]

    return grid, out


def plot_both_runs(res_1bar, res_2bar, sep_2bar):
    """Both runs figure: 1-bar vs. 2-bar aligned across all conditions."""

    fig, axs = plt.subplots(4, 2, figsize=(11, 12), sharex=True)

    for row, (cname, _, _, _) in enumerate(CONDITIONS):
        for col, (direction, dlabel) in enumerate(((+1, 'Bar UP'), (-1, 'Bar DOWN'))):
            ax = axs[row, col]

            for res, is_2bar in [(res_1bar, False), (res_2bar, True)]:
                m = (res['cond'] == cname) & (res['mdir'] == direction)
                if not np.any(m):
                    continue

                ay, cart = res['agent_y'][m], res['cart'][m]

                # Rows
                signals_to_plot = {
                    # 'R7/8 (central): ':         (cart[:, CENTER],               '#000000'),
                    f'R{FOCUS_R_TYPE + 1}: ':   (cart[:, FOCUS_R_TYPE],         '#993A56'),
                    'R1-R6 (pool): ':           (cart[:, PERIPH].sum(axis=1),   '#999933')
                }

                for r, (row_label, row_data_and_col) in enumerate(signals_to_plot.items()):
                    row_data, color = row_data_and_col

                    grid, row_data_binned = binned(ay, row_data)
                    ls, lw = ('-', 1.8) if is_2bar else ('--', 1.4)

                    ax.plot(grid, norm(row_data_binned), color=color, lw=lw, ls=ls,
                            label=row_label + f'{"2-bar" if is_2bar else "1-bar"}')

            bar_m = 2.0 * DISTANCE * np.tan(np.radians(sep_2bar) / 2.0)
            for xb in [-bar_m / 2, +bar_m / 2]:
                ax.axvline(xb, color='r', ls=':', alpha=0.5)

            ax.set_title(f"{cname} | {dlabel}")
            ax.grid(True, alpha=0.3)

            ax.set_xlim([-0.75, 0.75])

    axs[0, 1].legend(fontsize=8, loc='upper right')
    fig.suptitle(f"Single-bar vs. Two-bar ({sep_2bar:.1f} deg)", y=1.0)

    plt.tight_layout()

# Run every separation ------------------------------------------------------------------

all_results = {}

for sep in BAR_SEPARATIONS:
    label = "one bar" if sep == 0 else f"{sep:.1f} deg"

    print(f"\nSimulating {label}...")

    bars_m = 2.0 * DISTANCE * np.tan(np.radians(sep) / 2.0)

    res = simulate(sep)
    all_results[sep] = res

    # Figure for current condition
    plot_single_run(res, sep, bars_m)


# Synthesis figure: 1-bar vs. 2-bar overlay (if both were run)
controls = [s for s in BAR_SEPARATIONS if s == 0.0]
tests = [s for s in BAR_SEPARATIONS if s > 0.0]
if controls and tests:
    plot_both_runs(all_results[controls[0]], all_results[tests[0]], tests[0])


plt.show()