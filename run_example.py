import os
import time

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
from graphics.compound_eye import CompoundEyeRaster, CompoundEyeRay
from graphics.raster_mode import PanoramicEye


def main():

    USE_RAYTRACER = True

    # TODO: move these flags to the engine
    PANORAMIC_VIEW = True
    RUN_HEADLESS = False
    TIME_DITHERING = False

    EYE_RADIUS = 0.01  # only used for RT version
    NB_OMMATIDIA = 1962
    NB_SAMPLES = 256

    HEADLESS_MAX_STEPS = 1000

    # Setup
    eng = Engine(width=1280, height=720, headless=RUN_HEADLESS)

    crate_mesh = eng.load_mesh("crate", CUBE_VERTICES, 'shaders/base.vert', 'shaders/base.frag', 'textures/wood.jpg')

    eng.skybox = Skybox()
    eng.skybox_texture_id = load_cubemap('textures/bright_day')

    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([0.0, 0.0, 0.0])))
    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([-3.0, 0.0, 0.0])))
    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([3.0, 0.0, 0.0])))

    # Create the eye model
    eye_geom = EyeModel.generate_uniform_eye(num_ommatidia=NB_OMMATIDIA, eye_radius=EYE_RADIUS)

    if USE_RAYTRACER:
        print("Mode: Ray-Tracer")
        compoundeye = CompoundEyeRay(eye_model=eye_geom, scene=eng.scene, time_dithering=TIME_DITHERING, nb_samples=NB_SAMPLES)
    else:
        print("Mode: Rasterizer")
        compoundeye = CompoundEyeRaster(eye_model=eye_geom, time_dithering=TIME_DITHERING, nb_samples=NB_SAMPLES)

    pano_debug_view = PanoramicEye()

    # Simulation loop
    COMPOUND_EYE_VIEW = False
    VORONOI_VIEW = True

    # Simulation variables
    rotation_per_step_deg = 0.5
    current_rotation_deg = 0.0

    if not RUN_HEADLESS:
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        print(f"Interactive mode with {NB_OMMATIDIA} ommatidia.")
    else:
        print(f"Simulation started with {NB_OMMATIDIA} ommatidia...")

    is_running = True
    frame_count = 0
    start = time.time_ns()
    while is_running:
        # Event handling
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE): is_running = False
            if event.type == KEYDOWN and event.key == K_c: COMPOUND_EYE_VIEW = not COMPOUND_EYE_VIEW
            if event.type == KEYDOWN and event.key == K_v: VORONOI_VIEW = not VORONOI_VIEW
            if event.type == KEYDOWN and event.key == K_p: PANORAMIC_VIEW = not PANORAMIC_VIEW
            if event.type == KEYDOWN and event.key == K_t: compoundeye.time_dithering = not compoundeye.time_dithering
            if event.type == KEYDOWN and event.key == K_h: eng.show_hud = not eng.show_hud
            if event.type == KEYDOWN and event.key in (K_KP_PLUS, K_EQUALS): compoundeye.samples_per_ommatidium *= 2
            if event.type == KEYDOWN and event.key in (K_KP_MINUS, K_MINUS): compoundeye.samples_per_ommatidium //= 2
        eng.update_movement()

        # Update scene and re-packing for dynamic elements (optional)
        # current_rotation_deg = (current_rotation_deg + rotation_per_step_deg) % 360.0
        # eng.scene.instances[0].transform = rotation_mat(
        #     np.deg2rad(current_rotation_deg), WORLD_UP
        # )

        if USE_RAYTRACER:
            compoundeye.update_geometry(eng.scene.instances)     # fast update method
            # compoundeye.replace_scene(eng.scene)  # Slower update, to use when elements are added / removed from scene

        # Data Acquisition
        if USE_RAYTRACER:
            ommatidia_values = compoundeye.get_ommatidia_data(eng.camera, eng.skybox_texture_id)
        else:
            scene_cubemap_id = eng.render_to_cubemap(eng.scene, eng.camera)
            ommatidia_values = compoundeye.get_ommatidia_data(scene_cubemap_id)

        # # Example of CPU-side use of ommatidia data
        # if frame_count % 100 == 0:  # Print a sample every 100 frames
        #     print(f"Step {frame_count}: Ommatidium 0 value: {ommatidia_values[0]}")
        # # np.save(f'data/frame_{frame_count}.npy', ommatidia_values)

        # Drawing
        if not RUN_HEADLESS:
            glViewport(0, 0, eng.width, eng.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if COMPOUND_EYE_VIEW:
                compoundeye.draw(tiled_mode=VORONOI_VIEW)

            elif PANORAMIC_VIEW:
                scene_cubemap_id = eng.render_to_cubemap(eng.scene, eng.camera)
                pano_debug_view.draw(scene_cubemap_id)

            else:
                eng.render_frame()  # default to normal 3D view

            eng.clock.tick()
            eng.draw_hud(compoundeye)

            pygame.display.flip()

        frame_count += 1
        if RUN_HEADLESS and frame_count >= HEADLESS_MAX_STEPS: is_running = False

    total_time = (time.time_ns() - start) * 1e-9
    print(f"Simulation finished.")
    print(f"Total: {frame_count} frames in {total_time:.3f} seconds ({int(frame_count / total_time)} avg fps, {(total_time / frame_count) * 1e4:.3f} ms per frame).")
    compoundeye.free()
    eng.close()


if __name__ == "__main__":
    main()