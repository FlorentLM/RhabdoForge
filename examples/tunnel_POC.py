from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from insectvision.engine import Context, Agent, Scene, Asset
from insectvision.compound_eyes import CompoundEyeModel
from insectvision.engine.world_utils import WORLD_BACKWARD
from insectvision.renderers import Raytracer
from insectvision.utils import Colormap
from insectvision.interactive.debug import AxesGizmo, DebugGrid, DebugBox
from insectvision.geometry import plane_geom
from insectvision.geometry.materials_utils import checkerboard_texture
from insectvision.neuromorphic.basic_models import HassensteinReichardtEMD


@dataclass
class RunLog:
    time: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    z: List[float] = field(default_factory=list)
    left_flow: List[float] = field(default_factory=list)
    right_flow: List[float] = field(default_factory=list)
    yaw: List[float] = field(default_factory=list)


def random_tunnel_start(tunnel_width: float, tunnel_height: float, margin_pct: float = 0.25, randomise_height=False):
    margin_pct = 1.0 - np.clip(margin_pct, 0.0, 1.0)
    start_x = (np.random.rand() - 0.5) * (tunnel_width * margin_pct)
    start_y = np.random.rand() * (tunnel_height * margin_pct) if randomise_height else (tunnel_height * margin_pct) / 2.0
    return start_x, start_y, 0.0


## Setup test environment

context = Context(window_size=(1280, 720), fps_limit=None, vsync=False)

scene = Scene(background_color=(0.15, 0.15, 0.3))
scene.add_skybox('assets/textures/bright_day_nosun')

# Tunnel props
w, h, l = 5.0, 5.0, 150.0
block_size = 8
checkerboard_ratio = 0.5
texture_res = 512, 15360


# Left wall
v_left, uv_left, idx_left = plane_geom(
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

# Right wall
v_right, uv_right, idx_right = plane_geom(
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


# Floor
v_bottom, uv_bottom, idx_bottom = plane_geom(
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


# Ceiling
v_top, uv_top, idx_top = plane_geom(
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


## Setup eye model and agent

model = CompoundEyeModel.from_file('species_models/drosophila_custom.npz')
model.scale(1e-6)
with model.unlock(receptors=True):
    model.receptors.tau_membrane = 0.012

left_eye = model.eye(0)
right_eye = model.eye(1)

agent = Agent()

renderer = Raytracer(
    model=model, scene=scene, agent=agent,
    context=context,
    nb_samples=512,
    time_dithering=True,
    quasi_random=True,
    enable_shadows=False
)
renderer.overlay_enabled = True

# Add some debug stuff, hide HUD
context.debug.add(DebugGrid(size=1000.0, step=5.0))
context.debug.add(AxesGizmo(size=0.4))

for blas in renderer.BLASes:
    context.debug.add(DebugBox(blas))


## Run

# Simulation runs at a fixed 200 Hz biological clock
context.fixed_sim_dt = 1.0 / 200.0

saved_runs: Dict[str, RunLog] = {}
modes = ["Non-holonomic (yaw steering)", "Holonomic (lateral shift)"]

for mode in modes:
    print(f"Running {mode} mode...")

    agent.position = random_tunnel_start(w, h, randomise_height=True)
    agent.yaw = 0.0
    agent.pitch = 0.0
    agent.roll = 0.0

    # Left eye looks for 'decreasing azimuth' neighbours
    left_emd = HassensteinReichardtEMD(
        eye=left_eye,
        direction=(-1.0, 0.0),
        coordinate='spherical'
    )
    # Right eye looks for 'increasing azimuth' neighbours
    right_emd = HassensteinReichardtEMD(
        eye=right_eye,
        direction=(1.0, 0.0),
        coordinate='spherical'
    )

    log = RunLog()
    saved_runs[mode] = log

    sim_time = 0.0
    flight_speed = 3.5 # m/s

    while context.run_interactive(agent=agent, scene=scene, renderer=renderer):
        context.hud.show = False

        context.input()

        dt = context.tick()

        visual_output = renderer.step()

        left_motion = left_emd.process(visual_output.per_lens, dt)
        right_motion = right_emd.process(visual_output.per_lens, dt)

        mean_left = float(np.mean(left_motion))
        mean_right = float(np.mean(right_motion))

        # Normalised error (-1.0 to 1.0)
        diff = mean_right - mean_left
        summ = abs(mean_right) + abs(mean_left) + 1e-6
        error = diff / summ

        # Controller
        if mode == "Non-holonomic (yaw steering)":
            # Direct proportional control
            yaw_gain = 30.0
            damping_gain = 5.0

            turn_rate = error * yaw_gain - agent.yaw * damping_gain

            agent.rotate(yaw=turn_rate * dt)
            agent.translate(agent.forward * flight_speed * dt)

        elif mode == "Holonomic (lateral shift)":
            # Negative gain: wall closer (error > 0) -> strafe left (-X)
            strafe_gain = -1.0
            strafe_speed = error * strafe_gain

            agent.translate((agent.forward * flight_speed + agent.right * strafe_speed) * dt)

        # Prevent agent from clipping through tunnel walls
        pos = agent.position
        pos.x = np.clip(pos.x, -(w * 0.48), (w * 0.48))
        pos.y = np.clip(pos.y, 0.1, h - 0.1)
        agent.position = pos

        # Set overlay to display optic flow
        renderer.set_overlay(
            {left_eye: left_motion, right_eye: right_motion},
            colormap=Colormap.Diverging, compression=1.0
        )
        context.draw()

        # Log
        log.time.append(context.total_sim_time)
        log.x.append(float(agent.position.x))
        log.y.append(float(agent.position.y))
        log.z.append(float(agent.position.z))
        log.left_flow.append(mean_left)
        log.right_flow.append(mean_right)
        log.yaw.append(float(agent.yaw))


        if agent.position.z < -l:
            print(f"Finished {mode} (end of tunnel)")
            break

        if context.total_sim_time >= 180.0:
            print(f"Finished {mode} (time limit)")
            break

context.free()


## Plot results

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