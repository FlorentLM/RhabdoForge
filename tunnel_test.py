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


##

context = Context(window_size=(1280, 720), fps_limit=None, v_sync=False)
scene = Scene(background_color=(0.15, 0.15, 0.3))

scene.add_skybox('textures/bright_day_nosun')

w, h, l = 1.0, 1.0, 20.0

block_size = 8

##

v_left, uv_left, idx_left = create_plane(
    [-w/2.0, 0.0, 0.0], [-w/2.0,  h, 0.0], [-w/2.0,  h, -l], [-w/2.0, 0.0, -l]
)
left_pattern = checkerboard_texture(256, 2560, block_size=block_size, ratio=0.5)
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
right_pattern = checkerboard_texture(256, 2560, block_size=block_size, ratio=0.5)
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
bottom_pattern = checkerboard_texture(256, 2560, block_size=block_size, ratio=0.5)
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
top_pattern = checkerboard_texture(256, 2560, block_size=block_size, ratio=0.5)
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

agent = Agent(position=(0.0, 0.5, 0.0))

renderer = Raytracer(
    eye_model=eye_model,
    scene=scene,
    nb_samples=2,
    time_dithering=False,
    batch_size=1,
    enable_shadows=False
)

context.debug.add(DebugGrid())
context.debug.add(AxesGizmo(size=0.4))
for blas in renderer._scene_baked.BLASes:
    context.debug.add(DebugBox(blas))


##

while context.run_interactive(agent=agent, scene=scene, renderer=renderer):
    context.input()

    ommatidia_data = renderer.get_ommatidia_data(agent)

    context.draw()

renderer.free()
scene.free()
context.free()
