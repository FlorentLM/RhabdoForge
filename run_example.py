import os
import time

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
import numpy as np

from OpenGL.GL import *

from graphics.engine import Engine
from graphics.scene import Instance, PointCloud
from geometry.compound_eyes import CompoundEye
from graphics.glm import translation_mat, rotation_mat
from geometry.primitives import CUBE_VERTICES
from graphics.skybox import Skybox
from graphics.utils import load_cubemap, WORLD_UP
from graphics.eye_rendering import EyeRendererRaster, EyeRendererRay
from graphics.raster_mode import PanoramicEye


def main():

    USE_POINT_CLOUD = True
    USE_RAYTRACER = True

    # TODO: move these flags to the engine
    PANORAMIC_VIEW = False
    RUN_HEADLESS = False
    TIME_DITHERING = False
    COMPOUND_EYE_VIEW = True
    VORONOI_VIEW = False

    NB_OMMATIDIA = 1962
    # NB_OMMATIDIA = 1962
    NB_SAMPLES = 16

    HEADLESS_MAX_STEPS = 1000

    POINT_HIT_RADIUS = 0.01

    # Setup
    eng = Engine(width=1280, height=720, headless=RUN_HEADLESS)

    if USE_POINT_CLOUD:
        point_cloud = PointCloud('assets/seville_filtered.ply', hit_radius=POINT_HIT_RADIUS)
        eng.add_point_cloud(point_cloud)

    else:
        # Load the debug scene
        crate_mesh = eng.load_mesh("crate", CUBE_VERTICES, 'shaders/base.vert', 'shaders/base.frag',
                                   'textures/wood.jpg')
        eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([0.0, 0.0, 0.0])))
        eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([-3.0, 0.0, 0.0])))
        eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([3.0, 0.0, 0.0])))
        print("Default crate scene loaded.")

    eng.skybox = Skybox()
    eng.skybox_texture_id = load_cubemap('textures/bright_day')
    # eng.skybox_texture_id = load_cubemap('textures/black')

    # Create the eye model
    eye = CompoundEye(num_ommatidia=NB_OMMATIDIA, force_isotropic=True)

    if USE_RAYTRACER:
        print("Mode: Ray-Tracer")
        eye_renderer = EyeRendererRay(eye_model=eye, scene=eng.scene, time_dithering=TIME_DITHERING, nb_samples=NB_SAMPLES, point_radius=POINT_HIT_RADIUS)
    else:
        print("Mode: Rasterizer")
        eye_renderer = EyeRendererRaster(eye_model=eye, time_dithering=TIME_DITHERING, nb_samples=NB_SAMPLES)

    pano_debug_view = PanoramicEye()

    # Assign the eye to the engine
    eng.compound_eye = eye_renderer

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
            if event.type == KEYDOWN and event.key == K_t: eye_renderer.time_dithering = not eye_renderer.time_dithering
            if event.type == KEYDOWN and event.key == K_h: eng.show_hud = not eng.show_hud
            if event.type == KEYDOWN and event.key in (K_KP_PLUS, K_EQUALS): eye_renderer.samples_per_ommatidium *= 2
            if event.type == KEYDOWN and event.key in (K_KP_MINUS, K_MINUS): eye_renderer.samples_per_ommatidium //= 2
        eng.update_movement()

        # Update scene and re-packing for dynamic elements (optional)
        if not USE_POINT_CLOUD:
            current_rotation_deg = (current_rotation_deg + rotation_per_step_deg) % 360.0
            eng.scene.instances[0].transform = rotation_mat(
                np.deg2rad(current_rotation_deg), WORLD_UP
            )
            if USE_RAYTRACER:
                eye_renderer.update_geometry(eng.scene.instances)     # fast update method
                # compoundeye.replace_scene(eng.scene)  # Slower update, to use when elements are added / removed from scene

        # Data Acquisition
        if USE_RAYTRACER:
            ommatidia_values = eye_renderer.get_ommatidia_data(eng.camera, eng.skybox_texture_id)
        else:
            scene_cubemap_id = eng.render_to_cubemap(eng.scene, eng.camera)
            ommatidia_values = eye_renderer.get_ommatidia_data(scene_cubemap_id)

        # # Example of CPU-side use of ommatidia data
        # if frame_count % 100 == 0:  # Print a sample every 100 frames
        #     print(f"Step {frame_count}: Ommatidium 0 value: {ommatidia_values[0]}")
        # # np.save(f'data/frame_{frame_count}.npy', ommatidia_values)

        # Drawing
        if not RUN_HEADLESS:
            glViewport(0, 0, eng.width, eng.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if COMPOUND_EYE_VIEW:
                eye_renderer.draw(tiled_mode=VORONOI_VIEW)

            elif PANORAMIC_VIEW:
                scene_cubemap_id = eng.render_to_cubemap(eng.scene, eng.camera)
                pano_debug_view.draw(scene_cubemap_id)

            else:
                if USE_POINT_CLOUD:
                    # The default renderer only draws triangles for now
                    # TODO: raster-based point cloud renderer
                    # For now just draw the skybox
                    if eng.skybox and eng.skybox_texture_id is not None:
                        eng.skybox.draw(eng.camera.projection, eng.camera.view, eng.skybox_texture_id)
                else:
                    eng.render_frame()  # default to normal 3D view for debug crates

            eng.clock.tick()
            eng.draw_hud()

            pygame.display.flip()

        frame_count += 1
        if RUN_HEADLESS and frame_count >= HEADLESS_MAX_STEPS: is_running = False

    total_time = (time.time_ns() - start) * 1e-9
    print(f"Simulation finished.")
    print(f"Total: {frame_count} frames in {total_time:.3f} seconds ({int(frame_count / total_time)} fps (avg.), {(total_time / frame_count) * 1e4:.3f} ms per frame).")
    eye_renderer.free()
    eng.close()


if __name__ == "__main__":
    main()