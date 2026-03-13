from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from insectvision.engine.scene import Scene, Asset
from insectvision.engine.agent import Agent
from insectvision.engine.context import Context

from insectvision.geometry.compound_eyes.proxies import Eye
from insectvision.geometry.compound_eyes.receptor_array import ReceptorArray
from insectvision.renderers.raytracer import Raytracer
from insectvision.renderers.commons import Colormap
from insectvision.debug import AxesGizmo, DebugGrid, DebugBox


# TODO: These should go to the geometry submodule

def checkerboard_texture(width, height, block_size=1, ratio=0.5):
    low_res_w = width // block_size
    low_res_h = height // block_size
    random_grid = np.random.random((low_res_w, low_res_h))
    small_pattern = (random_grid < ratio).astype(np.uint8) * 255
    pattern = np.repeat(np.repeat(small_pattern, block_size, axis=0), block_size, axis=1)
    return pattern.astype(np.uint8)


def create_plane(v0, v1, v2, v3):
    vertices = np.array([v0, v1, v2, v3], dtype=np.float32)
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    uv_coords = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
    return vertices, uv_coords, indices


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
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    z: List[float] = field(default_factory=list)
    left_flow: List[float] = field(default_factory=list)
    right_flow: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)


## Environment

context = Context(window_size=(1280, 720), fps_limit=None, v_sync=False)
scene = Scene(background_color=(0.15, 0.15, 0.3))

scene.add_skybox('assets/textures/bright_day_nosun')

w, h, l = 5.0, 5.0, 150.0

block_size = 8
checkerboard_ratio = 0.5
texture_res = 512, 15360


v_left, uv_left, idx_left = create_plane(
    [-w/2.0, 0.0, -l], [-w/2.0,  h, -l], [-w/2.0,  h, 0.0], [-w/2.0, 0.0, 0.0]
)
left_pattern = checkerboard_texture(*texture_res, block_size=block_size, ratio=checkerboard_ratio)
left_wall = Asset.from_arrays(
    name='left_wall',
    vertices=v_left,
    faces=idx_left,
    uv_coords=uv_left,
    texture=left_pattern
)
scene.add_instance(left_wall)


v_right, uv_right, idx_right = create_plane(
    [w/2.0, 0.0, 0.0], [w/2.0,  h, 0.0], [w/2.0,  h, -l], [w/2.0, 0.0, -l]
)
right_pattern = checkerboard_texture(*texture_res, block_size=block_size, ratio=checkerboard_ratio)
right_wall = Asset.from_arrays(
    name='right_wall',
    vertices=v_right,
    faces=idx_right,
    uv_coords=uv_right,
    texture=right_pattern
)
scene.add_instance(right_wall)


v_bottom, uv_bottom, idx_bottom = create_plane(
    [-w/2.0, 0.0, 0.0], [w/2.0,  0.0, 0.0], [w/2.0,  0.0, -l], [-w/2.0, 0.0, -l]
)
bottom_pattern = checkerboard_texture(*texture_res, block_size=block_size, ratio=checkerboard_ratio)
bottom_wall = Asset.from_arrays(
    name='bottom_wall',
    vertices=v_bottom,
    faces=idx_bottom,
    uv_coords=uv_bottom,
    texture=bottom_pattern
)
scene.add_instance(bottom_wall)


v_top, uv_top, idx_top = create_plane(
    [-w/2.0, h, 0.0], [w/2.0,  h, 0.0], [w/2.0,  h, -l], [-w/2.0, h, -l]
)
top_pattern = checkerboard_texture(*texture_res, block_size=block_size, ratio=checkerboard_ratio)
top_wall = Asset.from_arrays(
    name='top_wall',
    vertices=v_top,
    faces=idx_top,
    uv_coords=uv_top,
    texture=top_pattern
)
scene.add_instance(top_wall)

##

eye_array = ReceptorArray.from_file('species_models/drosophila_custom.npz', eye_parameter=1.5)
eye_array.scale(0.01)

left_eye = eye_array.eye(0)
right_eye = eye_array.eye(1)


def random_tunnel_start(tunnel_width: float, tunnel_height: float, randomise_height=False):
    start_x = (np.random.rand() - 0.5) * tunnel_width
    start_y = np.random.rand() * tunnel_height if randomise_height else tunnel_height / 2.0
    return start_x, start_y, 0.0


agent = Agent()

renderer = Raytracer(
    receptor_array=eye_array, scene=scene,
    nb_samples=512,
    time_accumulation=0.012,
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
saved_runs: Dict[str, RunLog] = {}
modes = ["Non-holonomic (yaw steering)", "Holonomic (lateral shift)"]


##

for mode in modes:

    print(f"Run: {mode}")

    agent.position = random_tunnel_start(w, h, randomise_height=True)
    agent.yaw = 0.0
    agent.pitch = 0.0
    agent.roll = 0.0

    left_emd = HassensteinReichardtEMD(eye=left_eye, direction=test_direction)
    right_emd = HassensteinReichardtEMD(eye=right_eye, direction=test_direction)

    log = RunLog()
    saved_runs[mode] = log

    sim_time = 0.0
    flight_speed = 3.5 # m/s

    while context.run_interactive(agent=agent, scene=scene, renderer=renderer):

        context.input()

        dt = 1/200.0
        sim_time += dt

        view = renderer.get_visual_output(agent)
        ommatidia_data = view.per_ommatidium

        left_motion = left_emd.process(ommatidia_data, dt)
        right_motion = right_emd.process(ommatidia_data, dt)

        mean_left = float(np.mean(left_motion))
        mean_right = float(np.mean(right_motion))

        # Normalised error (-1.0 to 1.0)
        diff = mean_right - mean_left
        summ = abs(mean_right) + abs(mean_left) + 1e-6
        error = diff / summ

        # Controller
        if mode == "Non-holonomic (yaw steering)":
            # Direct proportional control
            yaw_gain = 25.0
            damping_gain = 5.0

            turn_rate = error * yaw_gain - agent.yaw * damping_gain

            agent.dt(dt).rotate(yaw_delta=turn_rate)
            agent.dt(dt).translate(agent.forward * flight_speed)

        elif mode == "Holonomic (lateral shift)":
            # Negative gain: wall closer (error > 0) -> strafe left (-X)
            strafe_gain = -1.5
            strafe_speed = error * strafe_gain

            agent.dt(dt).translate(agent.forward * flight_speed + agent.right * strafe_speed)

        # Prevent agent from clipping through tunnel walls
        pos = agent.position
        pos.x = np.clip(pos.x, -(w * 0.48), (w * 0.48))
        pos.y = np.clip(pos.y, 0.1, h - 0.1)
        agent.position = pos

        # Log
        log.time.append(sim_time)
        log.x.append(float(agent.position.x))
        log.y.append(float(agent.position.y))
        log.z.append(float(agent.position.z))
        log.left_flow.append(mean_left)
        log.right_flow.append(mean_right)
        log.yaw.append(float(agent.yaw))

        renderer.set_heatmap_eyes(
            {left_eye: left_motion, right_eye: right_motion},
            colormap=Colormap.DIVERGING, compression=1.0
        )

        context.draw()

        if agent.position.z < -l:
            print(f"Finished {mode} (end of tunnel)")
            break

        if sim_time >= 60.0:
            print(f"Finished {mode} (time limit)")
            break

context.free()

##

n_runs = len(saved_runs)

fig = plt.figure(figsize=(14, 4 * n_runs + 2), constrained_layout=True)
fig.suptitle("Centering experiment", fontsize=14, fontweight='bold')

gs = GridSpec(n_runs, 4, figure=fig)

colours = {"Holonomic (lateral shift)": "#2196F3",
           "Non-holonomic (yaw steering)": "#FF5722"}

for row, (run_name, log) in enumerate(saved_runs.items()):
    col = colours.get(run_name, "gray")
    t = np.array(log.time)

    # Trajectory (top-down X vs. Z)
    ax_traj_top = fig.add_subplot(gs[row, 0])
    ax_traj_top.plot(log.z, log.x, color=col, linewidth=1.2)
    ax_traj_top.axhline(0.0, color='k', ls='--', lw=0.5, label='centre')
    ax_traj_top.axhline(-w / 2.0, color='grey', ls='-', lw=0.8)
    ax_traj_top.axhline(w / 2.0, color='grey', ls='-', lw=0.8)
    ax_traj_top.set_xlabel("Z position (m)")
    ax_traj_top.set_ylabel("X position (m)")
    ax_traj_top.set_title(f"{run_name}\nTrajectory (top-down view)")
    ax_traj_top.set_ylim(-w / 2.0 - 0.5, w / 2.0 + 0.5)
    ax_traj_top.invert_xaxis()
    ax_traj_top.legend(fontsize=8)

    # Trajectory (side view Y vs. Z)
    ax_traj_side = fig.add_subplot(gs[row, 1])
    ax_traj_side.plot(log.z, log.y, color=col, linewidth=1.2)
    ax_traj_side.axhline(h / 2.0, color='k', ls='--', lw=0.5, label='centre')
    ax_traj_side.axhline(0.0, color='grey', ls='-', lw=0.8)
    ax_traj_side.axhline(h, color='grey', ls='-', lw=0.8)
    ax_traj_side.set_xlabel("Z position (m)")
    ax_traj_side.set_ylabel("Y position (m)")
    ax_traj_side.set_title(f"{run_name}\nTrajectory (side view)")
    ax_traj_side.set_ylim(- 0.5, h + 0.5)
    ax_traj_side.invert_xaxis()
    ax_traj_side.legend(fontsize=8)

    # Bilateral optic flow
    ax_flow = fig.add_subplot(gs[row, 2])
    ax_flow.plot(t, log.left_flow, color='red', alpha=0.7, lw=0.9, label='Left eye')
    ax_flow.plot(t, log.right_flow, color='blue', alpha=0.7, lw=0.9, label='Right eye')
    ax_flow.axhline(0, color='k', ls='--', lw=0.5)
    ax_flow.set_xlabel("Time (s)")
    ax_flow.set_ylabel("Mean EMD response")
    ax_flow.set_title("Bilateral optic flow")
    ax_flow.legend(fontsize=8)

    # Heading (yaw)
    ax_yaw = fig.add_subplot(gs[row, 3])
    ax_yaw.plot(t, log.yaw, color=col, lw=1.0)
    ax_yaw.axhline(0, color='k', ls='--', lw=0.5)
    ax_yaw.set_xlabel("Time (s)")
    ax_yaw.set_ylabel("Yaw (°)")
    ax_yaw.set_ylim(-90, 90)
    ax_yaw.set_title("Heading angle")

plt.show()