import math
import numpy as np
import matplotlib.pyplot as plt

from insectvision.compound_eyes.kernel import drosophila_kernel, RECEPTOR_PALETTE
from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.compound_eyes import CompoundEyeModel
from insectvision.renderers import Raytracer
from insectvision.geometry import plane_geom

# Hyperacuity demo
#
# Drosophila microsaccades are roughly vertically in visual space (aligned with optic flow during forward flight)
# So:
#  - Stimulus: thin horizontal bars
#  - Agent oscillates vertically
#  - Readout is a single (or a horizontal band of) forward-pointing ommatidia

# Stimulus
BAR_THICKNESS = 0.04    # metres
BAR_LENGTH = 10.0       # metres
DISTANCE= 2.0           # metres

# Two bars:
# BAR_SEPARATION = 0.10
BAR_SEPARATION = 0.14      # 4.01° centre-to-centre → 2.86° gap
# BAR_SEPARATION = 0.15      # 4.30° centre-to-centre → 3.15° gap
# BAR_SEPARATION = 0.16      # 4.58° centre-to-centre → 3.43° gap
# BAR_SEPARATION = 0.18      # 5.15° centre-to-centre → 4.00° gap
# BAR_SEPARATION = 0.19      # 5.44° centre-to-centre → 4.29° gap
# BAR_SEPARATION = 0.21      # 6.01° centre-to-centre → 4.86° gap
# BAR_SEPARATION = 0.25      # 7.16° centre-to-centre → 6.00° gap

# One bar:
# BAR_SEPARATION = 0.0       # single bar at origin


# Motion - Linear Sweeps
SWEEP_SPEED = 1.0        # m/s
SWEEP_AMPLITUDE = 0.5    # m
NUM_CYCLES_PER_PHASE = 3 # Complete 3 Up/Down sweeps OFF, then 3 Up/Down sweeps ON

# One-way sweep time
SWEEP_DURATION = (2 * SWEEP_AMPLITUDE) / SWEEP_SPEED
# Full cycle (Up + Down)
CYCLE_DURATION = 2 * SWEEP_DURATION
MAX_TIME = CYCLE_DURATION * NUM_CYCLES_PER_PHASE * 2

# Readout ommatidium (or band of ommatidia)
# Horizontal strip: broad in azimuth, narrow in elevation
# BAND_AZ_HALFWIDTH = 20.0  # deg
# BAND_EL_HALFWIDTH = 1.5  # deg
# BAND_CONE_DEG = 15.0

# This should select just one ommatidium
BAND_AZ_HALFWIDTH = 1.0  # deg
BAND_EL_HALFWIDTH = 1.0  # deg
BAND_CONE_DEG = 15.0

EYE_MODEL_PATH = 'species_models/drosophila_custom.npz'


# ---------------------------------------------------------------------------

def create_bar(name, cx, cy, width_x, width_y, distance, texture):
    """Bar at (cx, cy) in the plane z = -distance, extents width_x × width_y."""
    v_0 = [cx - width_x / 2, cy - width_y / 2, -distance]
    v_1 = [cx - width_x / 2, cy + width_y / 2, -distance]
    v_2 = [cx + width_x / 2, cy + width_y / 2, -distance]
    v_3 = [cx + width_x / 2, cy - width_y / 2, -distance]
    vertices, uv_coords, faces = plane_geom(v_0, v_1, v_2, v_3)
    return Asset.from_arrays(
        name=name, vertices=vertices, faces=faces,
        uv_coords=uv_coords, texture=texture,
    )


def pick_ommatidia(model, agent, az_halfwidth_deg=BAND_AZ_HALFWIDTH, el_halfwidth_deg=BAND_EL_HALFWIDTH, cone_deg=BAND_CONE_DEG):

    cone = model.query_cone(agent.forward, angle=cone_deg)
    mask = (np.abs(cone.azimuth_deg) < az_halfwidth_deg) & (np.abs(cone.elevation_deg) < el_halfwidth_deg)
    band = cone[mask]

    chir = band.chirality
    n_pos = int(np.sum(chir > 0))
    n_neg = int(np.sum(chir < 0))
    print(f"Forward band: {len(band)} lenses "
          f"(chirality +1:{n_pos}, -1:{n_neg})  "
          f"az ±{az_halfwidth_deg}°, el ±{el_halfwidth_deg}°")
    return band.indices


def print_stim_geometry():

    bar_t_deg = math.degrees(BAR_THICKNESS  / DISTANCE)
    bar_s_deg = math.degrees(BAR_SEPARATION / DISTANCE)
    sweep_w = math.degrees(SWEEP_SPEED / DISTANCE)

    print(f"Horizontal bars: {bar_t_deg:.2f}° thick (elevation), {bar_s_deg:.2f}° apart")
    print(f"Linear vertical sweep speed: {sweep_w:.1f}°/s")


# ---------------------------------------------------------------------------

context = Context()

context.mouse_captured = False
context.time_step = 1 / 1000.0   # 1 ms of biological simulation resolution

scene = Scene(background_color=(0.0, 0.0, 0.0))

# Put sun behind and make it full white
scene.sun.elevation = 1.0
scene.sun.azimuth = 0.0
scene.sun.color = (1.0, 1.0, 1.0)

white_tex = np.ones((32, 32), dtype=np.uint8) * 255

low_y  = -BAR_SEPARATION / 2.0
high_y = +BAR_SEPARATION / 2.0

bar_low  = create_bar('bar_low',  0.0, low_y,  BAR_LENGTH, BAR_THICKNESS, DISTANCE, white_tex)
bar_high = create_bar('bar_high', 0.0, high_y, BAR_LENGTH, BAR_THICKNESS, DISTANCE, white_tex)
scene.add_instance(bar_low)
scene.add_instance(bar_high)


model = CompoundEyeModel.from_file(EYE_MODEL_PATH, kernel=drosophila_kernel())
model.scale(1e-6)
with model.unlock(receptors=True):
    # model.receptors.tau_membrane = 0.012
    model.receptors.tau_membrane = 0.0

agent = Agent(position=(0.0, 0.0, 0.0))

renderer = Raytracer(
    model=model,
    scene=scene,
    agent=agent,
    context=context,
    nb_samples=256,
    time_dithering=True,
    randomness_mode='Halton',
    enable_actuation=True,
    enable_ambient=True,
    enable_direct=True,
    enable_shadows=False,
)
renderer.ambient_intensity = 1.5

# Tune / disable luminance boost on RF narrowing
renderer.photon_concentration = 0.2
# renderer.photon_concentration = 0.0


with model.unlock(lenses=True):
    # Lateral gain: 1.5 um is roughly 1-2 degrees of angular shift?
    model.lenses.gain_lat_um = 1.5

    # Axial gain: RF narrowing
    model.lenses.gain_ax_um = 8.0

    # Kemppainen says 5 fast and 80 slow (check this) ?
    model.lenses.tau_rise = 0.005
    model.lenses.tau_relax = 0.080

# --------------------------------------------------------------------------

print_stim_geometry()
selected_lenses = pick_ommatidia(model, agent)

if len(selected_lenses) == 0:
    raise RuntimeError("Forward band is empty. Widen the cone / strip.")
renderer.selected_lenses = selected_lenses[:min(10, len(selected_lenses))].tolist()

print(f"R7/8 acceptance: {np.degrees(model.rcpt_dynamic_data['acc_axes'][selected_lenses[0] * 7 + 6])}°")

results = {
    'time':             [],
    'agent_y':          [],
    'actuation':        [],
    'L2_cart':          [],
    'R78_cart':         [],
    'apposition_pool':  [],
    'indiv_lens':       [],
    'motion_dir':       [],
}

print(f"\nRunning for {MAX_TIME:.1f}s: {NUM_CYCLES_PER_PHASE} cycles OFF, then {NUM_CYCLES_PER_PHASE} cycles ON ...")

while context.run_interactive(agent=agent, scene=scene, renderer=renderer, use_dashboard=True):

    context.input()
    if not context.hud.show:
        context.hud.show = True

    sim_time = context.total_time

    cycle_count = int(sim_time // CYCLE_DURATION)
    intra_cycle_time = sim_time % CYCLE_DURATION

    renderer.actuation = cycle_count >= NUM_CYCLES_PER_PHASE

    if intra_cycle_time < SWEEP_DURATION:
        # Agent moving up (+y) -> apparent bar motion down
        ay = -SWEEP_AMPLITUDE + (intra_cycle_time / SWEEP_DURATION) * (2 * SWEEP_AMPLITUDE)
        motion_dir = -1
    else:
        # Agent moving down (-y) -> apparent bar motion up
        ay = SWEEP_AMPLITUDE - ((intra_cycle_time - SWEEP_DURATION) / SWEEP_DURATION) * (2 * SWEEP_AMPLITUDE)
        motion_dir = 1

    agent.position = (0.0, ay, 0.0)
    visual_output = renderer.step()

    radiance_cart = visual_output.per_cartridge[selected_lenses, ..., :3].mean(axis=-1)
    radiance_lens = visual_output.per_lens[selected_lenses, ..., :3].mean(axis=-1)

    # Calculate LMC (Cartridge Pool) and Central Cell
    L2_cart = radiance_cart[:, :6].mean(axis=1).mean() * 6.0
    R78_cart = radiance_cart[:, 6].mean()

    # Calculate evolutionary controls
    apposition_pool = radiance_lens[:, :6].mean(axis=1).mean() * 6.0
    indiv_lens = radiance_lens.mean(axis=0)

    results['time'].append(sim_time)
    results['agent_y'].append(ay)
    results['actuation'].append(bool(renderer.actuation))
    results['L2_cart'].append(L2_cart)
    results['R78_cart'].append(R78_cart)
    results['apposition_pool'].append(apposition_pool)
    results['indiv_lens'].append(indiv_lens)
    results['motion_dir'].append(motion_dir)

    context.draw(visual_output)

    if sim_time >= MAX_TIME:
        break

# --------------------------------------------------------------------------

times = np.array(results['time'])
agent_y = np.array(results['agent_y'])
act = np.array(results['actuation'])
L2_cart = np.array(results['L2_cart'])
R78_cart = np.array(results['R78_cart'])
apposition_pool = np.array(results['apposition_pool'])
indiv_lens = np.array(results['indiv_lens'])
mdir = np.array(results['motion_dir'])

fig, axs = plt.subplots(4, 2, figsize=(14, 16), sharex=True)


def plot_sweep(ax, data, direction, title):
    mask = (mdir == direction)
    ax.scatter(agent_y[mask & ~act], data[mask & ~act], c='blue', s=1, alpha=0.5, label='OFF')
    ax.scatter(agent_y[mask & act], data[mask & act], c='red', s=1, alpha=0.5, label='ON')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_individual_receptors(ax, direction, title):
    """Plots lines for individual receptors to show their spatial staggering."""
    mask = (mdir == direction) & ~act  # Plotting OFF only to avoid clutter
    y_vals = agent_y[mask]
    sort_idx = np.argsort(y_vals)
    y_sorted = y_vals[sort_idx]

    for i in range(7):
        label = f'R{i + 1}' if i < 6 else 'R7/8'
        r_vals = indiv_lens[mask, i][sort_idx]
        ax.plot(y_sorted, r_vals, color=RECEPTOR_PALETTE[i], label=label, alpha=0.8, lw=1.5)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# Column 0: Bar moving up (agent moving down)
plot_sweep(axs[0, 0], R78_cart, 1, "Row 1 | R7/8 (Central Cell): Bar UP")
plot_sweep(axs[1, 0], L2_cart, 1, "Row 2 | Neural Superposition (Cartridge Pool): Bar UP")
plot_sweep(axs[2, 0], apposition_pool, 1, "Row 3 | Apposition Proxy (Ommatidium Pool): Bar UP")
plot_individual_receptors(axs[3, 0], 1, "Row 4 | Individual Receptors (Physical Lens) OFF-only: Bar UP")

axs[0, 0].set_ylabel("Signal Intensity")
axs[1, 0].set_ylabel("Signal Intensity")
axs[2, 0].set_ylabel("Signal Intensity")
axs[3, 0].set_ylabel("Signal Intensity")
axs[3, 0].set_xlabel("Agent Y position (m)")

# Column 1: Bar moving down (agent moving up)
plot_sweep(axs[0, 1], R78_cart, -1, "Row 1 | R7/8 (Central Cell): Bar DOWN")
plot_sweep(axs[1, 1], L2_cart, -1, "Row 2 | Neural Superposition (Cartridge Pool): Bar DOWN")
plot_sweep(axs[2, 1], apposition_pool, -1, "Row 3 | Apposition Proxy (Ommatidium Pool): Bar DOWN")
plot_individual_receptors(axs[3, 1], -1, "Row 4 | Individual Receptors (Physical Lens) OFF-only: Bar DOWN")

axs[3, 1].set_xlabel("Agent Y position (m)")
axs[0, 1].legend(loc='upper right', markerscale=5)
axs[3, 1].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

# renderer.free()
# scene.free()
# context.free()