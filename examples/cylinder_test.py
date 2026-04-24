from dataclasses import dataclass, field
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.geometry import cylinder_geom
from insectvision.geometry.materials_utils import grating_texture
from insectvision.renderers import Raytracer
from insectvision.utils import Colormap
from insectvision.interactive.debug import AxesGizmo, DebugGrid, DebugBox
from insectvision.neuromorphic.basic_models import HassensteinReichardtEMD
from insectvision.compound_eyes import ReceptorArray, RhabdomereKernel


@dataclass
class RunLog:
    time: List[float] = field(default_factory=list)
    left_flow: List[float] = field(default_factory=list)
    right_flow: List[float] = field(default_factory=list)
    agent_yaw: List[float] = field(default_factory=list)
    drum_yaw: List[float] = field(default_factory=list)


## Setup test environment

context = Context(window_size=(1280, 720), fps_limit=None, vsync=False)

scene = Scene(background_color=(0.15, 0.15, 0.3))
scene.add_skybox('assets/textures/bright_day_nosun')

# Cylinder props
diameter = 5.0
radius = diameter / 2.0
h = 5.0
texture_res = 4096, 1024

NUM_BANDS = 5

# Create cylinder
v_cyl, uv_cyl, idx_cyl = cylinder_geom(radius=radius, height=h, segments=64, inwards=True)
cyl_pattern = grating_texture(*texture_res, num_bands=NUM_BANDS, orientation='horizontal', wave_type='square')

drum = Asset.from_arrays(
    name='optomotor_drum',
    vertices=v_cyl,
    faces=idx_cyl,
    uv_coords=uv_cyl,
    texture=cyl_pattern
)
cylinder_drum = scene.add_instance(asset=drum, dynamic=True)


## Setup eye model and agent

droso = RhabdomereKernel(
    name='Drosophila',
    offsets_um=np.array([
        [-1.6881,  1.0273],
        [-1.8046, -0.9934],
        [-1.7111, -2.9717],
        [-0.0025, -1.9261],
        [ 1.6690, -0.9493],
        [ 1.6567,  0.9762],
        [ 0.0045, -0.0113],
    ]),
    diameters_um=np.array([1.8627, 1.8627, 1.8627, 1.8627, 1.8627, 1.8627, 1.5743]),
    nodal_distance_um=21.0,
    center_index=6,
    main_axis_indices=(2, 5)
)

pitch_rad = np.deg2rad(10.1)
optic_flow = np.array([0.0, np.sin(pitch_rad), np.cos(pitch_rad)])

ra = ReceptorArray.from_file(file_path='species_models/drosophila_custom.npz',
# ra = ReceptorArray(ommatidia_count=1600,
    kernel=droso,
    flow_direction=optic_flow,
    weight_flow=0.6,
    weight_tissue=0.6
)
# ra.scale(0.01)
ra.receptors.tau = 0.012

left_eye = ra.eyes[0]
right_eye = ra.eyes[1]


##

NB_SAMPLES = 512


ACTUATE = True
ACTUATION_GAIN = 5.0
TAU_ON = 0.03
TAU_OFF = 0.4
LATERAL_CLAMP = 1.7

start_position = (0.0, h/2.0, 0.0)

agent = Agent(position=start_position)
agent.yaw = 0.0
agent.pitch = 0.0
agent.roll = 0.0


renderer = Raytracer(
    receptor_array=ra, scene=scene, agent=agent,
    nb_samples=NB_SAMPLES,
    time_dithering=True,
    quasi_random=True,
    enable_actuation=True,
    enable_shadows=False,
)

# Add some debug stuff, hide HUD
context.debug.add(DebugGrid(size=1000.0, step=5.0))
context.debug.add(AxesGizmo(size=0.4))

for blas in renderer.BLASes:
    context.debug.add(DebugBox(blas))

##

left_emd = HassensteinReichardtEMD(
    eye=left_eye,
    direction=(0.0, -1.0),
    coordinate='spherical'
)
right_emd = HassensteinReichardtEMD(
    eye=right_eye,
    direction=(0.0, -1.0),
    coordinate='spherical'
)

## Run

log = RunLog()

sim_time = 0.0
drum_current_yaw = 0.0

# Sensor calibration
drum_velocity = 45.0
# 45 deg/s rotation produces an EMD output of ~0.035 (from open-loop observation)
EMD_TO_DEG_PER_SEC = 45.0 / 0.035


ra.actuate(lateral_um=0.0, axial_um=0.0)
# Excitation/inhibition per eye
exc_left = np.zeros(left_eye.lens_count)
inh_left = np.zeros(left_eye.lens_count)
exc_right = np.zeros(right_eye.lens_count)
inh_right = np.zeros(right_eye.lens_count)



while context.run_interactive(agent=agent, scene=scene, renderer=renderer, use_dashboard=True):
    context.hud.show = False

    context.input()

    dt = 1 / 120.0
    sim_time += dt

    visual_output = renderer.get_output()

    left_lens_out = visual_output[left_eye].lenses
    right_lens_out = visual_output[right_eye].lenses

    left_motion = left_emd.process(left_lens_out, dt)
    right_motion = right_emd.process(right_lens_out, dt)

    # Per-rhabdomere luminance
    left_rhab_lum = left_lens_out[:, :, :3].mean(axis=2)
    right_rhab_lum = right_lens_out[:, :, :3].mean(axis=2)

    # Per-lens mean for actuation
    left_lens_lum = left_rhab_lum.mean(axis=1)
    right_lens_lum = right_rhab_lum.mean(axis=1)

    # Rotational optic flow
    # Drum spins left (+) -> left eye sees front-to-back (+) -> flow should be positive
    net_rotational_flow = left_lens_lum - right_lens_lum
    estimated_slip_velocity = net_rotational_flow * EMD_TO_DEG_PER_SEC

    if ACTUATE:
        alpha_f = dt / (dt + TAU_ON)
        alpha_s = dt / (dt + TAU_OFF)

        exc_left = alpha_f * left_lens_lum + (1 - alpha_f) * exc_left
        inh_left = alpha_s * left_lens_lum + (1 - alpha_s) * inh_left
        left_lat = np.clip(ACTUATION_GAIN * (exc_left - inh_left), -LATERAL_CLAMP, LATERAL_CLAMP)
        left_eye.actuate(lateral_um=left_lat, axial_um=0.0)

        exc_right = alpha_f * right_lens_lum + (1 - alpha_f) * exc_right
        inh_right = alpha_s * right_lens_lum + (1 - alpha_s) * inh_right
        right_lat = np.clip(ACTUATION_GAIN * (exc_right - inh_right), -LATERAL_CLAMP, LATERAL_CLAMP)
        right_eye.actuate(lateral_um=right_lat, axial_um=0.0)

    # Log
    log.time.append(sim_time)
    log.left_flow.append(left_lens_lum)
    log.right_flow.append(right_lens_lum)
    log.agent_yaw.append(float(agent.yaw))
    log.drum_yaw.append(float(drum_current_yaw))

    renderer.set_overlay(
        {left_eye: left_motion, right_eye: right_motion},
        colormap=Colormap.Thermal, compression=1.0
    )

    context.draw(visual_output)


context.free()


## Plot results

t = np.array(log.time)

fig, (ax_flow, ax_yaw) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
fig.suptitle("Optomotor Response (OMR) in Virtual Cylinder", fontsize=14, fontweight='bold')

# Bilateral optic flow
ax_flow.plot(t, log.left_flow, color='red', alpha=0.7, lw=1.2, label='Left eye (mean EMD)')
ax_flow.plot(t, log.right_flow, color='blue', alpha=0.7, lw=1.2, label='Right eye (mean EMD)')
ax_flow.axhline(0, color='k', ls='--', lw=0.5)
ax_flow.set_xlabel("Time (s)")
ax_flow.set_ylabel("Mean EMD response")
ax_flow.set_title("Wide-field Optic Flow")
ax_flow.legend(fontsize=9, loc='upper right')

# Heading (yaw) tracking
ax_yaw.plot(t, log.drum_yaw, color='gray', lw=2.0, linestyle='--', label='Drum Position (Stimulus)')
ax_yaw.plot(t, log.agent_yaw, color='#3AF3C9', lw=2.0, label='Agent Heading (Response)')
ax_yaw.axhline(0, color='k', ls='--', lw=0.5)
ax_yaw.set_xlabel("Time (s)")
ax_yaw.set_ylabel("Yaw Angle (°)")
ax_yaw.set_title("Gaze Stabilization (Tracking)")
ax_yaw.legend(fontsize=9, loc='upper right')

plt.show()
