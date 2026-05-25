import math
import numpy as np
import matplotlib.pyplot as plt

from insectvision.compound_eyes.kernel import drosophila_kernel
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
# BAR_SEPARATION = 0.14      # 4.01° centre-to-centre → 2.86° gap
# BAR_SEPARATION = 0.15      # 4.30° centre-to-centre → 3.15° gap
# BAR_SEPARATION = 0.16      # 4.58° centre-to-centre → 3.43° gap
# BAR_SEPARATION = 0.18      # 5.15° centre-to-centre → 4.00° gap
# BAR_SEPARATION = 0.19      # 5.44° centre-to-centre → 4.29° gap
# BAR_SEPARATION = 0.21      # 6.01° centre-to-centre → 4.86° gap
# BAR_SEPARATION = 0.25      # 7.16° centre-to-centre → 6.00° gap

# One bar:
BAR_SEPARATION = 0.0       # single bar at origin


# Motion
# Vertical sinusoidal sweep
# At f=0.5, A=0.5, DISTANCE=2.0 -> ~45 deg/s peak vertical sweep
OSC_FREQ = 0.5
OSC_AMPLITUDE = 0.5
SWITCH_PERIOD = 4.0
MAX_TIME = 2 * SWITCH_PERIOD

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

    cone_idx = model.query_cone(agent.forward, angle=cone_deg)
    dirs = model.lenses[cone_idx].direction

    azim = np.degrees(np.arctan2(dirs[:, 0], -dirs[:, 2]))
    elev = np.degrees(np.arcsin(np.clip(dirs[:, 1], -1.0, 1.0)))

    strip = (np.abs(azim) < az_halfwidth_deg) & (np.abs(elev) < el_halfwidth_deg)
    band = cone_idx[strip]

    chir = model.lenses.chirality[band]
    n_pos = int(np.sum(chir > 0))
    n_neg = int(np.sum(chir < 0))
    print(f"Forward band: {len(band)} lenses "
          f"(chirality +1:{n_pos}, -1:{n_neg})  "
          f"az ±{az_halfwidth_deg}°, el ±{el_halfwidth_deg}°")
    return band


def print_stim_geometry():

    bar_t_deg = math.degrees(BAR_THICKNESS  / DISTANCE)
    bar_s_deg = math.degrees(BAR_SEPARATION / DISTANCE)
    peak_v = 2 * math.pi * OSC_FREQ * OSC_AMPLITUDE
    peak_w = math.degrees(peak_v / DISTANCE)

    print(f"Horizontal bars: {bar_t_deg:.2f}° thick (elevation), "
          f"{bar_s_deg:.2f}° apart (Drosophila IOA ~4.5°, R7/8 acceptance ~4.6°)")
    print(f"Peak angular vertical sweep speed: {peak_w:.1f}°/s")


# ---------------------------------------------------------------------------

context = Context()

context.mouse_captured = False
context.fixed_sim_dt = 1 / 1000.0   # 1 ms of biological simulation resolution

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
model.receptors.tau_membrane = 0.012

agent = Agent(position=(0.0, 0.0, 0.0))

renderer = Raytracer(
    model=model,
    scene=scene,
    agent=agent,
    context=context,
    nb_samples=256,
    time_dithering=True,
    quasi_random=True,
    enable_actuation=True,
    enable_ambient=True,
    enable_direct=True,
    enable_shadows=False,
)
renderer.ambient_intensity = 1.5

model.lenses.gain_lat_um = 2.0
model.lenses.gain_ax_um = 8.0
model.lenses.tau_relax = 0.080

# --------------------------------------------------------------------------

print_stim_geometry()
band_lenses = pick_ommatidia(model, agent)

if len(band_lenses) == 0:
    raise RuntimeError("Forward band is empty. Widen the cone / strip.")
renderer.selected_lenses = band_lenses[:min(10, len(band_lenses))].tolist()

print(f"R7/8 acceptance: {np.degrees(model.rcpt_dynamic_data['acc_axes'][band_lenses[0] * 7 + 6])}°")

results = {
    'time':       [],
    'agent_y':    [],
    'actuation':  [],
    'L2_cart':    [],
    'R78_cart':   [],
    'L2_lens':    [],
}

print(f"\nRunning for {MAX_TIME:.1f}s: first {SWITCH_PERIOD:.1f}s OFF, then ON ...")

while context.run_interactive(agent=agent, scene=scene, renderer=renderer, use_dashboard=True):
    context.input()
    if not context.hud.show:
        context.hud.show = True

    dt = context.tick()
    sim_time = context.total_sim_time

    # Vertical sinusoidal oscillation, aparent bar motion is opposite sign
    ay = math.sin(sim_time * 2 * math.pi * OSC_FREQ) * OSC_AMPLITUDE
    agent.position = (0.0, ay, 0.0)

    renderer.actuation = (int(sim_time / SWITCH_PERIOD) % 2) == 1

    visual_output = renderer.step()

    band_cart = visual_output.cartridges[band_lenses]
    band_lens = visual_output.lenses[band_lenses]

    L2_cart  = band_cart[:, :6, 3].sum(axis=1).mean()
    R78_cart = band_cart[:,  6, 3].mean()
    L2_lens  = band_lens[:, :6, 3].sum(axis=1).mean()

    results['time'].append(sim_time)
    results['agent_y'].append(ay)
    results['actuation'].append(bool(renderer.actuation))
    results['L2_cart'].append(L2_cart)
    results['R78_cart'].append(R78_cart)
    results['L2_lens'].append(L2_lens)

    context.draw(visual_output)

    if sim_time >= MAX_TIME:
        break

# --------------------------------------------------------------------------

times = np.array(results['time'])
agent_y = np.array(results['agent_y'])
act = np.array(results['actuation'])
L2_cart = np.array(results['L2_cart'])
R78_cart = np.array(results['R78_cart'])
L2_lens = np.array(results['L2_lens'])


def phase_fold(t, sig, mask, freq):
    period = 1.0 / freq
    phase = (t[mask] % period) / period
    order = np.argsort(phase)
    return phase[order], sig[mask][order]


fig, axs = plt.subplots(3, 2, figsize=(13, 9),
                        gridspec_kw={'width_ratios': [2, 1]})

axs[0, 0].plot(times, agent_y, 'k', lw=1)
axs[0, 0].axvspan(SWITCH_PERIOD, MAX_TIME, color='red', alpha=0.07, label='Actuation ON')
axs[0, 0].set_ylabel('agent.y (m)')
axs[0, 0].set_title("Stimulus position (vertical)")
axs[0, 0].legend(loc='upper right')

axs[1, 0].plot(times[~act], R78_cart[~act], 'b.', ms=2, label='OFF')
axs[1, 0].plot(times[ act], R78_cart[ act], 'r.', ms=2, label='ON')
axs[1, 0].axvspan(SWITCH_PERIOD, MAX_TIME, color='red', alpha=0.07)
axs[1, 0].set_ylabel('central R7/8\n(finest RF, band-avg)')
axs[1, 0].legend(loc='upper right')
axs[1, 0].grid(True, alpha=0.3)

axs[2, 0].plot(times[~act], L2_cart[~act], 'b.', ms=2, label='L2 proxy OFF')
axs[2, 0].plot(times[ act], L2_cart[ act], 'r.', ms=2, label='L2 proxy ON')
axs[2, 0].plot(times, L2_lens, 'g-', alpha=0.5, lw=0.8,
                label='lens-level R1-R6 (no superposition)')
axs[2, 0].axvspan(SWITCH_PERIOD, MAX_TIME, color='red', alpha=0.07)
axs[2, 0].set_ylabel('L2 proxy\n(cartridge R1-R6 sum)')
axs[2, 0].set_xlabel('time (s)')
axs[2, 0].legend(loc='upper right', fontsize=8)
axs[2, 0].grid(True, alpha=0.3)

period = 1.0 / OSC_FREQ
off_window = (times >= SWITCH_PERIOD - period) & (times < SWITCH_PERIOD) & (~act)
on_window  = (times >= MAX_TIME      - period) & (times < MAX_TIME)        & ( act)

for ax_p, sig, name in [
    (axs[0, 1], agent_y,  'agent.y'),
    (axs[1, 1], R78_cart, 'R7/8'),
    (axs[2, 1], L2_cart,  'L2 proxy'),
]:
    p_off, s_off = phase_fold(times, sig, off_window, OSC_FREQ)
    p_on,  s_on  = phase_fold(times, sig, on_window,  OSC_FREQ)
    ax_p.plot(p_off, s_off, 'b-', lw=1.2, label='OFF')
    ax_p.plot(p_on,  s_on,  'r-', lw=1.2, label='ON')
    ax_p.set_ylabel(name)
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc='upper right', fontsize=8)

axs[0, 1].set_title("Phase-folded\n(last period of each phase)")
axs[2, 1].set_xlabel('phase (0-1)')

plt.tight_layout()
plt.show()


renderer.free()
scene.free()
context.free()