import random
from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt

from insectvision.engine import Context, Agent, Scene, Asset

from insectvision.geometry.compound_eyes import ReceptorArray, Eye
from insectvision.renderers import Raytracer
from insectvision.renderers.commons import Colormap
from insectvision.interactive.debug import AxesGizmo, DebugGrid, DebugBox


# TODO: These should go to the geometry submodule

def inwards_cylinder(radius, height, segments=64):
    """Generate upright cylinder mesh facing inwards."""
    vertices = []
    uv_coords = []

    for i in range(segments + 1):
        u = i / segments
        theta = 2.0 * np.pi * u

        # circle on the X-Z plane
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)

        vertices.append([x, 0.0, z])
        uv_coords.append([u, 0.0])

        vertices.append([x, height, z])
        uv_coords.append([u, 1.0])

    vertices = np.array(vertices, dtype=np.float32)
    uv_coords = np.array(uv_coords, dtype=np.float32)

    indices = []
    for i in range(segments):
        idx0 = i * 2
        idx1 = i * 2 + 1
        idx2 = (i + 1) * 2
        idx3 = (i + 1) * 2 + 1

        # Winding order for inward-facing normals
        indices.append([idx0, idx2, idx1])
        indices.append([idx1, idx2, idx3])

    indices = np.array(indices, dtype=np.uint32)
    return vertices, uv_coords, indices


def vertical_bands_texture(width, height, num_bands=18):
    """Generate full contrast vertical grating (square wave)."""
    # repeating pattern along X axis
    x = np.linspace(0, num_bands * 2 * np.pi, width, endpoint=False)
    stripe_1d = (np.sin(x) > 0).astype(np.uint8) * 255
    pattern = np.zeros((height, width), dtype=np.uint8)
    pattern[:, :] = stripe_1d[np.newaxis, :]
    return pattern



class HassensteinReichardtEMD:
    """
    Elementary Motion Detector based on Hassenstein-Reichardt correlator.

    - GPU temporal accumulation: Photoreceptors (R1-R6) - low-pass integration
    - Python high-pass: Lamina monopolar cells (L1/L2) - Luminance adaptation
    - Python delay/correlator: Medulla (e.g., Mi1/Tm3 to T4/T5) - Delay and multiplication
    """

    def __init__(self,
            eye: Eye, direction: Tuple[float, float, float],
            delay_hz: float = 8.0,
            highpass_hz: float = 2.0
        ):
        self.eye = eye

        # Convert Hz to time constants
        self.tau_delay = 1.0 / (2.0 * np.pi * delay_hz)
        self.tau_hp = 1.0 / (2.0 * np.pi * highpass_hz)

        self.targets, self.weights = eye.directed_neighbours(
            direction=direction, k=1, coordinate='cartesian', return_weights=True
        )

        self._mean_lum = None   # For high-pass L1/L2 adaptation
        self._delayed_A = None  # LP[A(t)] (correlator)
        self._delayed_B = None  # LP[B(t)] (correlator)

    def process(self, ommatidia_data: np.ndarray, dt: float) -> np.ndarray:
        # GPU output (must be already R1-R6 low-pass filtered)
        eye_data = ommatidia_data[self.eye.global_indices]
        luminance = eye_data[:, :3].mean(axis=1)

        # Lamina L1/L2 high-pass equivalent (luminance adaptation / contrast)
        alpha_hp = dt / (self.tau_hp + dt)
        if self._mean_lum is None:
            self._mean_lum = luminance.copy()
        else:
            self._mean_lum += alpha_hp * (luminance - self._mean_lum)

        # Convert to contrast (removes DC, normalises by local mean)
        contrast = (luminance - self._mean_lum) / (self._mean_lum + 1e-6)

        # Medulla delay lines
        alpha_delay = dt / (self.tau_delay + dt)

        signal_A = contrast  # direct channel
        signal_B = contrast[self.targets]  # neighbour channel

        if self._delayed_A is None:
            self._delayed_A = signal_A.copy()
            self._delayed_B = signal_B.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        # Update delay lines
        self._delayed_A += alpha_delay * (signal_A - self._delayed_A)
        self._delayed_B += alpha_delay * (signal_B - self._delayed_B)

        # Correlator: preferred arm - anti-preferred arm
        motion = signal_B * self._delayed_A - signal_A * self._delayed_B

        return motion * self.weights


@dataclass
class RunLog:
    time: List[float] = field(default_factory=list)
    left_flow: List[float] = field(default_factory=list)
    right_flow: List[float] = field(default_factory=list)
    agent_yaw: List[float] = field(default_factory=list)
    drum_yaw: List[float] = field(default_factory=list)


## Environment

context = Context(window_size=(1280, 720), fps_limit=None, v_sync=False)
scene = Scene(background_color=(0.15, 0.15, 0.3))

diameter = 5.0
radius = diameter / 2.0
h = 5.0

v_cyl, uv_cyl, idx_cyl = inwards_cylinder(radius=radius, height=h, segments=64)

texture_res = (4096, 1024)
cyl_pattern = vertical_bands_texture(*texture_res, num_bands=18)

drum = Asset.from_arrays(
    name='optomotor_drum',
    vertices=v_cyl,
    faces=idx_cyl,
    uv_coords=uv_cyl,
    texture=cyl_pattern
)
cylinder_drum = scene.add_instance(asset=drum, dynamic=True)

##

eye_array = ReceptorArray.from_file('species_models/drosophila_custom.npz', eye_parameter=1.5)
eye_array.scale(0.01)

eye_array.tau = 0.012

left_eye = eye_array.eye(0)
right_eye = eye_array.eye(1)

start_position = (0.0, h/2.0, 0.0)
agent = Agent(position=start_position)

renderer = Raytracer(
    receptor_array=eye_array, scene=scene,
    nb_samples=512,
    time_dithering=True,
    quasi_random=True,
    enable_shadows=False
)
renderer.heatmap_enabled = True

context.debug.add(DebugGrid(size=1000.0, step=5.0))
context.debug.add(AxesGizmo(size=0.4))

for blas in renderer._scene_baked.BLASes:
    context.debug.add(DebugBox(blas))

##

test_direction = (0.0, 0.0, 1.0)  # front to back

agent.position = start_position
agent.yaw = 0.0
agent.pitch = 0.0
agent.roll = 0.0

left_emd = HassensteinReichardtEMD(eye=left_eye, direction=test_direction)
right_emd = HassensteinReichardtEMD(eye=right_eye, direction=test_direction)

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

##

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