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
#  - Readout is a single forward-pointing ommatidium

# ---------------------------------------------------------------------------
# Stimulus

BAR_WIDTH_DEG  = 1.0      # degrees

# Two bars:

# BAR_SEP_DEG    = 1.0
# BAR_SEP_DEG    = 2.0
# BAR_SEP_DEG    = 3.0
BAR_SEP_DEG    = 4.0    # degrees centre-to-centre
# BAR_SEP_DEG    = 5.0
# BAR_SEP_DEG    = 6.0
# BAR_SEP_DEG    = 7.0

# One bar:
# BAR_SEP_DEG    = 0.0

DISTANCE       = 2.0      # m
BAR_LENGTH     = 10.0     # m

BAR_THICKNESS  = 2.0 * DISTANCE * np.tan(np.radians(BAR_WIDTH_DEG) / 2.0)
BAR_SEPARATION = 2.0 * DISTANCE * np.tan(np.radians(BAR_SEP_DEG) / 2.0)

# Motion: Linear sweeps
SWEEP_SPEED_DEG  = 30.0    # deg/s (angular speed at the centre of the field)
SWEEP_AMPLITUDE  = 1.0     # m (total travel is +/- this value)

SWEEP_SPEED = DISTANCE * np.radians(SWEEP_SPEED_DEG)

NUM_CYCLES_PER_PHASE = 3 # Complete 3 Up/Down sweeps OFF, then 3 Up/Down sweeps ON

# One-way sweep time
SWEEP_DURATION = (2 * SWEEP_AMPLITUDE) / SWEEP_SPEED
# Full cycle (Up + Down)
CYCLE_DURATION = 2 * SWEEP_DURATION
MAX_TIME = CYCLE_DURATION * NUM_CYCLES_PER_PHASE * 2

print(f"\nRunning for {MAX_TIME:.1f}s: {NUM_CYCLES_PER_PHASE} cycles OFF, then {NUM_CYCLES_PER_PHASE} cycles ON ...")

# ---------------------------------------------------------------------------

# This should select just one ommatidium in each eye
CONE_DEG = 10.0

GAIN_LAT = 1.5
GAIN_AX = 8.0

TAU_MEMBRANE = 0.012

TAU_FAST     = 0.005
TAU_ADAPT    = 0.050
TAU_RISE     = 0.005
TAU_RELAX    = 0.080

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

def pick_ommatidia(model, agent, cone_deg=CONE_DEG):
    cone = model.query_cone(agent.forward, angle=cone_deg, degrees=True, avoid_conflicts=True)
    if len(cone) == 0:
        raise RuntimeError("Forward band is empty. Widen the cone.")

    chir = cone.chirality
    n_pos = int(np.sum(chir > 0))
    n_neg = int(np.sum(chir < 0))
    print(f"Forward cartridges: lenses {cone} "
          f"(chirality +1:{n_pos}, -1:{n_neg})  "
          f"(picked from {len(cone)} candidates within {cone_deg} deg of forward)")
    return cone.indices


def print_stim_geometry():

    bar_t_deg = math.degrees(BAR_THICKNESS  / DISTANCE)
    bar_s_deg = math.degrees(BAR_SEPARATION / DISTANCE)
    sweep_w = math.degrees(SWEEP_SPEED / DISTANCE)

    print(f"Horizontal bars: {bar_t_deg:.2f}° thick (elevation), {bar_s_deg:.2f}° apart")
    print(f"Linear vertical sweep speed: {sweep_w:.1f}°/s")

print_stim_geometry()

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

model = CompoundEyeModel.from_file('species_models/drosophila_custom.npz', kernel=drosophila_kernel())
model.scale(1e-6)

with model.unlock(receptors=True):
    model.receptors.tau_membrane = TAU_MEMBRANE

with model.unlock(lenses=True):
    model.lenses.tau_fast = TAU_FAST
    model.lenses.tau_adapt = TAU_ADAPT
    model.lenses.gain_lat_um = GAIN_LAT
    model.lenses.gain_ax_um = GAIN_AX
    model.lenses.tau_rise = TAU_RISE
    model.lenses.tau_relax = TAU_RELAX

agent = Agent(position=(0.0, 0.0, 0.0))

renderer = Raytracer(
    model=model,
    scene=scene,
    agent=agent,
    context=context,
    nb_samples=512,
    time_dithering=True,
    randomness_mode='Halton',
    enable_actuation=True,
    enable_ambient=True,
    enable_direct=True,
    enable_shadows=False,
)

# Make the bar very bright
renderer.ambient_intensity = 1.5

# Tune / disable luminance boost on RF narrowing
# renderer.photon_concentration = 0.2
renderer.photon_concentration = 0.0

# --------------------------------------------------------------------------

selected_lenses = pick_ommatidia(model, agent)

# Only take one lens / cartridge
selected_lens = selected_lenses[0]

renderer.selected_lenses = [571]

print(f"R7/8 acceptance: {np.degrees(model.rcpt_dynamic_data['acc_axes'][selected_lens * 7 + 6])}°")

results = {
    'time':             [],
    'agent_y':          [],
    'actuation':        [],
    'L2_cart':          [],
    'R78_cart':         [],
    'cartridge':        [],
    'motion_dir':       [],
}

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

    # Radiance = mean of colours
    radiance_cart = visual_output.per_cartridge[selected_lens, :, :3].mean(axis=-1)
    radiance_lens = visual_output.per_lens[selected_lens, :, :3].mean(axis=-1)

    # Calculate LMC (Cartridge pool) and Central Cell
    L2_cart = radiance_cart[:6].sum()
    R78_cart = radiance_cart[6]

    results['time'].append(sim_time)
    results['agent_y'].append(ay)
    results['actuation'].append(bool(renderer.actuation))
    results['cartridge'].append(radiance_cart)
    results['L2_cart'].append(L2_cart)
    results['R78_cart'].append(R78_cart)
    results['motion_dir'].append(motion_dir)

    context.draw(visual_output)

    if sim_time >= MAX_TIME:
        break

# --------------------------------------------------------------------------

times = np.array(results['time'])
agent_y = np.array(results['agent_y'])
act = np.array(results['actuation'])
cartridge = np.array(results['cartridge'])
L2_cart = np.array(results['L2_cart'])
R78_cart = np.array(results['R78_cart'])
mdir = np.array(results['motion_dir'])

fig, axs = plt.subplots(3, 2, figsize=(10, 10), sharex=True)


def plot_sweep(ax, data, direction, title):
    mask = (mdir == direction)
    ax.scatter(agent_y[mask & ~act], data[mask & ~act], c='blue', s=1, alpha=0.5, label='OFF')
    ax.scatter(agent_y[mask & act], data[mask & act], c='red', s=1, alpha=0.5, label='ON')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def plot_individual_receptors(ax, direction, title):
    """Plots lines for individual receptors."""
    mask = (mdir == direction) & act
    y_vals = agent_y[mask]
    sort_idx = np.argsort(y_vals)
    y_sorted = y_vals[sort_idx]

    for i in range(7):
        label = f'R{i + 1}' if i < 6 else 'R7/8'
        r_vals = cartridge[mask, i][sort_idx]
        ax.plot(y_sorted, r_vals, color=RECEPTOR_PALETTE[i], label=label, alpha=0.8, lw=1.5)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)


# Column 0: Bar moving up (agent moving down)
plot_sweep(axs[0, 0], R78_cart, 1, "R7/8 (Central cell): Bar UP")
plot_sweep(axs[1, 0], L2_cart, 1, "R1-R6 (Cartridge pool): Bar UP")
plot_individual_receptors(axs[2, 0], 1, "Individual receptors (Cartridge) ON-only: Bar UP")

axs[0, 0].set_ylabel("Signal Intensity")
axs[1, 0].set_ylabel("Signal Intensity")
axs[2, 0].set_ylabel("Signal Intensity")
axs[2, 0].set_xlabel("Agent Y position (m)")

# Column 1: Bar moving down (agent moving up)
plot_sweep(axs[0, 1], R78_cart, -1, "R7/8 (Central cell): Bar DOWN")
plot_sweep(axs[1, 1], L2_cart, -1, "R1-R6 (Cartridge pool): Bar DOWN")
plot_individual_receptors(axs[2, 1], -1, "Individual Receptors (Cartridge) ON-only: Bar DOWN")

axs[2, 1].set_xlabel("Agent Y position (m)")
axs[0, 1].legend(loc='upper right', markerscale=5)
axs[2, 1].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()

# renderer.free()
# scene.free()
# context.free()