from typing import Tuple
import numpy as np
from graphics.scene import Scene, Asset
from graphics.agent import Agent
from graphics.renderers.raytracer import Raytracer
from graphics.context import Context
from geometry.compound_eyes import CompoundEye
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


class SimpleEMD:
    def __init__(self, eye_model, temporal_window: int = 3):
        self.temporal_window = temporal_window
        self.frame_history = []

        azimuths = eye_model.ommatidia[:].azimuth_rad
        self.left_mask = azimuths < 0.0
        self.right_mask = azimuths > 0.0

    def process(self, ommatidia_data: np.ndarray) -> Tuple[float, float]:
        luminance = ommatidia_data[:, :3].mean(axis=1)
        self.frame_history.append(luminance)
        if len(self.frame_history) > self.temporal_window:
            self.frame_history.pop(0)
        if len(self.frame_history) < 2:
            return 0.0, 0.0

        prev_frame = self.frame_history[-2]
        curr_frame = self.frame_history[-1]

        # raw temporal difference
        diff = np.abs(curr_frame - prev_frame)

        left_flow = np.mean(diff[self.left_mask])
        right_flow = np.mean(diff[self.right_mask])

        return float(left_flow), float(right_flow)


##

context = Context(window_size=(1280, 720), fps_limit=None, v_sync=False)
scene = Scene(background_color=(0.15, 0.15, 0.3))

scene.add_skybox('textures/bright_day_nosun')

w, h, l = 5.0, 5.0, 50.0

block_size = 8
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

eye_model = CompoundEye.from_file('species_models/drosophila_custom.npz', eye_parameter=1.5)

agent = Agent(position=(-2.0, 0.5, 0.0))

renderer = Raytracer(
    eye_model=eye_model,
    scene=scene,
    nb_samples=256,
    time_dithering=True,
    batch_size=1,
    enable_shadows=False
)

context.debug.add(DebugGrid(size=1000.0, step=5.0))
context.debug.add(AxesGizmo(size=0.4))

for blas in renderer._scene_baked.BLASes:
    context.debug.add(DebugBox(blas))

##

emd = SimpleEMD(eye_model=eye_model)

left_vals = []
right_vals = []

while context.run_interactive(agent=agent, scene=scene, renderer=renderer):
    context.input()

    ommatidia_data = renderer.get_ommatidia_data(agent)
    left, right = emd.process(ommatidia_data)

    left_vals.append(left)
    right_vals.append(right)

    context.draw()

##

import matplotlib.pyplot as plt

plt.plot(left_vals, alpha=0.8, color="red", label="Left")
plt.plot(right_vals, alpha=0.8, color="blue", label="Right")

plt.show()