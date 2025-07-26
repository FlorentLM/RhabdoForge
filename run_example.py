import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
import numpy as np

from OpenGL.GL import *

from graphics.engine import Engine
from graphics.scene import Instance
from graphics.eye_model import EyeModel
from graphics.glm import translation_mat, rotation_mat
from geometry.primitives import CUBE_VERTICES
from graphics.skybox import Skybox
from graphics.utils import load_cubemap, WORLD_UP
from graphics.insect_eye import InsectEyeRaster, InsectEyeRay
from graphics.raster_mode import PanoramicEye


def main():

    USE_RAYTRACER = True
    IS_HEADLESS = False
    SHOW_PANO_VIEW = True
    SIMULATION_STEPS = 1000
    TIME_DITHERING = False
    EYE_RADIUS = 0.5  # eye physical size, only used for RT version

    # Setup
    eng = Engine(width=1280, height=720, headless=IS_HEADLESS)

    crate_mesh = eng.load_mesh("crate", CUBE_VERTICES, 'shaders/base.vert', 'shaders/base.frag', 'textures/wood.jpg')

    eng.skybox = Skybox()
    eng.skybox_texture_id = load_cubemap('textures/bright_day')

    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([0.0, 0.0, 0.0])))
    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([-3.0, 0.0, 0.0])))
    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([3.0, 0.0, 0.0])))

    # Create the eye model
    print("Initializing insect eye model...")
    eye_geom = EyeModel.generate_uniform_eye(num_ommatidia=1962, eye_radius=EYE_RADIUS)

    if USE_RAYTRACER:
        print("Mode: Ray-Tracer")
        insect_eye = InsectEyeRay(eye_model=eye_geom, scene=eng.scene, time_dithering=TIME_DITHERING)
    else:
        print("Mode: Rasterizer")
        insect_eye = InsectEyeRaster(eye_model=eye_geom, time_dithering=TIME_DITHERING)

    pano_debug_view = PanoramicEye()

    # Simulation loop
    SHOW_INSECT_EYE_VIEW = False
    TILED_MODE = True

    # Simulation variables
    rotation_per_step_deg = 0.5
    current_rotation_deg = 0.0

    if not IS_HEADLESS:
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    is_running = True
    frame_count = 0
    while is_running:
        # Event handling
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE): is_running = False
            if event.type == MOUSEWHEEL: eng.camera.fov -= event.y * 1.5
            if event.type == KEYDOWN and event.key == K_p: SHOW_INSECT_EYE_VIEW = not SHOW_INSECT_EYE_VIEW
            if event.type == KEYDOWN and event.key == K_t: TILED_MODE = not TILED_MODE
            if event.type == KEYDOWN and event.key == K_h: insect_eye.samples_per_ommatidium *= 2
            if event.type == KEYDOWN and event.key == K_g: insect_eye.samples_per_ommatidium //= 2
            if event.type == KEYDOWN and event.key == K_v: SHOW_PANO_VIEW = not SHOW_PANO_VIEW
        eng.update_movement()

        # Update scene and re-packing for dynamic elements (optional)
        # current_rotation_deg = (current_rotation_deg + rotation_per_step_deg) % 360.0
        # eng.scene.instances[0].transform = rotation_mat(
        #     np.deg2rad(current_rotation_deg), WORLD_UP
        # )

        if USE_RAYTRACER:
            insect_eye.update_geometry(eng.scene.instances)     # fast update method
            # insect_eye.replace_scene(eng.scene)  # Slower update, to use when elements are added / removed from scene

        # Data Acquisition
        if USE_RAYTRACER:
            ommatidia_values = insect_eye.get_ommatidia_data(eng.camera, eng.skybox_texture_id)
        else:
            scene_cubemap_id = eng.render_to_cubemap(eng.scene, eng.camera)
            ommatidia_values = insect_eye.get_ommatidia_data(scene_cubemap_id)

        # Example of CPU-side use of ommatidia data
        if frame_count % 100 == 0:  # Print a sample every 100 frames
            print(f"Step {frame_count}: Ommatidium 0 value: {ommatidia_values[0]}")
        # np.save(f'data/frame_{frame_count}.npy', ommatidia_values)

        # Drawing
        if not IS_HEADLESS:
            glViewport(0, 0, eng.width, eng.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if SHOW_INSECT_EYE_VIEW:
                insect_eye.draw(tiled_mode=TILED_MODE)

            elif SHOW_PANO_VIEW:
                scene_cubemap_id = eng.render_to_cubemap(eng.scene, eng.camera)
                pano_debug_view.draw(scene_cubemap_id)

            else:
                eng.render_frame()  # default to normal 3D view

            eng.clock.tick()
            eng._draw_fps()
            pygame.display.flip()

        frame_count += 1
        if IS_HEADLESS and frame_count >= SIMULATION_STEPS: is_running = False

    print(f"Simulation finished after {frame_count} steps.")
    insect_eye.free()
    eng.close()


if __name__ == "__main__":
    main()