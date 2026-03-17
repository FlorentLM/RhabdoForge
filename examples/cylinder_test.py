import random
from dataclasses import dataclass, field
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.geometry import cylinder_geom
from insectvision.geometry.materials_utils import grating_texture
from insectvision.renderers import Raytracer
from insectvision.renderers.commons import Colormap
from insectvision.interactive.debug import AxesGizmo, DebugGrid, DebugBox
from insectvision.neuromorphic.basic_models import HassensteinReichardtEMD
from insectvision.compound_eyes import ReceptorArray


@dataclass
class RunLog:
    time: List[float] = field(default_factory=list)
    left_flow: List[float] = field(default_factory=list)
    right_flow: List[float] = field(default_factory=list)
    agent_yaw: List[float] = field(default_factory=list)
    drum_yaw: List[float] = field(default_factory=list)


## Setup test environment

context = Context(window_size=(1280, 720), fps_limit=None, v_sync=False)

scene = Scene(background_color=(0.15, 0.15, 0.3))
scene.add_skybox('assets/textures/bright_day_nosun')

# Cylinder props
diameter = 5.0
radius = diameter / 2.0
h = 5.0
texture_res = 4096, 1024

# Create cylinder
v_cyl, uv_cyl, idx_cyl = cylinder_geom(radius=radius, height=h, segments=64, inwards=True)
cyl_pattern = grating_texture(*texture_res, num_bands=18, orientation='vertical', wave_type='square')

drum = Asset.from_arrays(
    name='optomotor_drum',
    vertices=v_cyl,
    faces=idx_cyl,
    uv_coords=uv_cyl,
    texture=cyl_pattern
)
cylinder_drum = scene.add_instance(asset=drum, dynamic=True)


## Setup eye model and agent

receptor_array = ReceptorArray.from_file('species_models/drosophila_custom.npz', eye_parameter=1.5)
receptor_array.scale(0.01)
receptor_array.tau = 0.012   # 12 ms time accumulation

left_eye = receptor_array.eye(0)
right_eye = receptor_array.eye(1)

start_position = (0.0, h/2.0, 0.0)
agent = Agent(position=start_position)

renderer = Raytracer(
    receptor_array=receptor_array, scene=scene,
    nb_samples=512,
    time_dithering=True,
    quasi_random=True,
    enable_shadows=False
)
renderer.heatmap_enabled = True

# Add some debug stuff, hide HUD
context.debug.add(DebugGrid(size=1000.0, step=5.0))
context.debug.add(AxesGizmo(size=0.4))

for blas in renderer._scene_baked.BLASes:
    context.debug.add(DebugBox(blas))


test_direction = (0.0, 0.0, 1.0)  # front to back

agent.position = start_position
agent.yaw = 0.0
agent.pitch = 0.0
agent.roll = 0.0

left_emd = HassensteinReichardtEMD(
    eye=left_eye,
    direction=test_direction,
    coordinate='cartesian'
)
right_emd = HassensteinReichardtEMD(
    eye=right_eye,
    direction=test_direction,
    coordinate='cartesian'
)

## Run

log = RunLog()

sim_time = 0.0
drum_current_yaw = 0.0

# Sensor calibration
drum_velocity = 45.0
# 45 deg/s rotation produces an EMD output of ~ 0.035 (from open-loop observation)
EMD_TO_DEG_PER_SEC = 45.0 / 0.035

switch_duration_range = (1.5, 4.0)  # between 1.5 and 4.0 seconds
next_switch_time = random.uniform(*switch_duration_range)

while context.run_interactive(agent=agent, scene=scene, renderer=renderer):
    context.hud.show = False

    context.input()

    dt = 1 / 120.0
    sim_time += dt

    if sim_time >= next_switch_time:
        drum_velocity *= -1.0
        interval = random.uniform(*switch_duration_range)
        next_switch_time = sim_time + interval

    drum_yaw_delta = drum_velocity * dt
    cylinder_drum.rotate_axis(drum_yaw_delta, 'up')
    drum_current_yaw += drum_yaw_delta

    view = renderer.get_visual_output(agent)
    ommatidia_data = view.per_ommatidium

    left_motion = left_emd.process(ommatidia_data, dt)
    right_motion = right_emd.process(ommatidia_data, dt)

    mean_left = float(np.mean(left_motion))
    mean_right = float(np.mean(right_motion))

    # Rotational optic flow
    # Drum spins left (+) -> left eye sees front-to-back (+) -> flow should be positive
    net_rotational_flow = mean_left - mean_right
    estimated_slip_velocity = net_rotational_flow * EMD_TO_DEG_PER_SEC

    # Optomotor controller
    optomotor_gain = 2.0
    turn_rate = estimated_slip_velocity * optomotor_gain

    agent.dt(dt).rotate(yaw_delta=turn_rate)

    # Log
    log.time.append(sim_time)
    log.left_flow.append(mean_left)
    log.right_flow.append(mean_right)
    log.agent_yaw.append(float(agent.yaw))
    log.drum_yaw.append(float(drum_current_yaw))

    renderer.set_heatmap_eyes(
        {left_eye: left_motion, right_eye: right_motion},
        colormap=Colormap.Diverging, compression=0.5
    )

    context.draw()

    if sim_time > 12.0:
        break

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