from typing import Tuple
import numpy as np

from graphics.scene import Scene, Asset
from graphics.agent import Agent
from graphics.renderers.raytracer import Raytracer
from graphics.renderers.base import Colormap
from graphics.context import Context
from geometry.compound_eyes import OmmatidialArray, Eye
from graphics.debug import AxesGizmo, DebugGrid, DebugBox


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

    For each ommatidium A, a neighbour B is selected along a specified direction on the eye surface.
    The detector computes:
        R = A(t) * LP[B(t)] − B(t) * LP[A(t)]
    where LP is a first-order low-pass filter

    For a static scene both arms are equal, so R = 0
    Motion in the preferred direction: R > 0
    Motion in the anti-preferred direction: R < 0

    All temporal parameters are specified in Hz
    """

    def __init__(self,
            eye: Eye, direction: Tuple[float, float, float],
            delay_hz: float = 8.0,  # TODO: check if ~8 Hz is typical for Drosophila L1/L2 delay filters
            prefilter_hz: float = 5.0
        ):
        """
        Args:
            eye: An Eye view from an OmmatidialArray.
            direction: (delta_azimuth, delta_elevation) in radians.
            delay_hz: Cutoff frequency (Hz) for the correlator delay arm.
                      This controls the temporal frequency tuning of the detector.
                      (lower = tuned to slower motion, higher = tuned to faster motion)
            prefilter_hz: Cutoff frequency (Hz) for the luminance pre-filter.
                          Smooths Monte Carlo sampling noise before it reaches the
                          correlator. Should be above delay_hz to pass the motion
                          signal, but low enough to reject rendering noise.
                          Set to 0 to disable.
        """
        self.eye = eye

        # Convert Hz to time constants
        self.tau = 1.0 / (2.0 * np.pi * delay_hz)
        self.tau_pre = 1.0 / (2.0 * np.pi * prefilter_hz) if prefilter_hz > 0 else 0.0
        self.tau_hp = 1.0 / (2.0 * np.pi * 2.0)  # ~2 Hz high-pass cutoff
        self._mean_lum = None

        self.targets, self.weights = eye.directed_neighbours(
            direction=direction, k=1, coordinate='cartesian', return_weights=True
        )

        self._filtered_lum = None  # pre-filtered luminance
        self._delayed_A = None  # LP[A(t)] (correlator)
        self._delayed_B = None  # LP[B(t)] (correlator)

    def process(self, ommatidia_data: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute per-ommatidium motion signal for one frame.
        """
        eye_data = ommatidia_data[self.eye.global_indices]
        raw_luminance = eye_data[:, :3].mean(axis=1)

        # Pre-filter: smooth rendering noise
        if self.tau_pre > 0:
            alpha_pre = dt / (self.tau_pre + dt)

            if self._filtered_lum is None:
                self._filtered_lum = raw_luminance.copy()
            else:
                self._filtered_lum += alpha_pre * (raw_luminance - self._filtered_lum)

            luminance = self._filtered_lum
        else:
            luminance = raw_luminance

        # Correlator
        alpha = dt / (self.tau + dt)

        signal_A = luminance  # direct channel (source)
        signal_B = luminance[self.targets]  # neighbour channel (target)

        if self._delayed_A is None:
            # First frame: initialise both delay lines
            self._delayed_A = signal_A.copy()
            self._delayed_B = signal_B.copy()
            return np.zeros(len(self.eye), dtype=np.float32)

        # Update both low-pass filters
        self._delayed_A += alpha * (signal_A - self._delayed_A)
        self._delayed_B += alpha * (signal_B - self._delayed_B)

        # Correlator: preferred arm - anti-preferred arm
        motion = signal_B * self._delayed_A - signal_A * self._delayed_B

        return motion * self.weights


##

context = Context(window_size=(1280, 720), fps_limit=None, v_sync=False)
scene = Scene(background_color=(0.15, 0.15, 0.3))

scene.add_skybox('textures/bright_day_nosun')

w, h, l = 5.0, 5.0, 50.0

block_size = 16
checkerboard_ratio = 0.5
texture_res = 512, 5120

##

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

eye_array = OmmatidialArray.from_file('species_models/drosophila_custom.npz', eye_parameter=1.5)

left_eye = eye_array.eye(0)
right_eye = eye_array.eye(1)

agent = Agent(position=(2.0, 0.5, 0.0))

renderer = Raytracer(
    eye_model=eye_array,
    scene=scene,
    nb_samples=512,
    time_dithering=False,
    batch_size=1,
    enable_shadows=False
)

context.debug.add(DebugGrid(size=1000.0, step=5.0))
context.debug.add(AxesGizmo(size=0.4))

for blas in renderer._scene_baked.BLASes:
    context.debug.add(DebugBox(blas))

##

test_direction = (0.0, 0.0, 1.0)  # front to back

left_emd = HassensteinReichardtEMD(eye=left_eye, direction=test_direction)
right_emd = HassensteinReichardtEMD(eye=right_eye, direction=test_direction)

renderer.heatmap_enabled = True

left_vals = []
right_vals = []

while context.run_interactive(agent=agent, scene=scene, renderer=renderer):
    context.input()

    dt = context.dt if hasattr(context, 'dt') else 1.0 / 60.0

    ommatidia_data = renderer.get_ommatidia_data(agent)

    left_motion = left_emd.process(ommatidia_data, dt)
    right_motion = right_emd.process(ommatidia_data, dt)

    left_vals.append(float(np.mean(left_motion)))
    right_vals.append(float(np.mean(right_motion)))

    renderer.set_heatmap_eyes(
        {left_eye: left_motion, right_eye: right_motion},
        colormap=Colormap.DIVERGING,
        compression=0.5
    )

    context.draw()

##

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(left_vals, alpha=0.8, color="red", label="Left eye")
ax.plot(right_vals, alpha=0.8, color="blue", label="Right eye")
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_xlabel("Frame")
ax.set_ylabel("Mean EMD response")
ax.set_title("Per-eye optic flow (Hassenstein-Reichardt)")
ax.legend()
plt.tight_layout()
plt.show()