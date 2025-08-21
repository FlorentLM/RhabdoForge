import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import OpenGL
OpenGL.ERROR_CHECKING = False

import pygame
from pygame.locals import *
import time
from pyglm import glm

from OpenGL.GL import *

from graphics.engine import Engine
from geometry.compound_eyes import CompoundEye
from geometry.primitives import CUBE_VERTICES
from graphics.utils import load_cubemap
from graphics.renderers.rasterizer import EyeRendererRaster
from graphics.renderers.raytracer import EyeRendererRay
from graphics.scene import Scene, Skybox


def main():

    # Configuration flags
    WINDOW_SIZE = (1280, 720)
    USE_POINT_CLOUD = False
    USE_RAYTRACER = False
    RUN_HEADLESS = False

    # View mode management
    view_modes = ['compound_eye', 'panoramic', 'standard_3d']
    current_view_idx = 0
    VORONOI_VIEW = False

    # Simulation Config
    NB_OMMATIDIA = 19362
    NB_SAMPLES = 16
    TIME_DITHERING = False
    POINT_RADIUS = 0.1
    HEADLESS_MAX_STEPS = 1000

    # Setup Engine and Scene
    eng = Engine(width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], headless=RUN_HEADLESS)
    scene = Scene()
    if USE_POINT_CLOUD:
        scene.add_point_cloud("canberra", 'assets/canberra_filtered.ply')
    else:
        crate_asset = scene.load_mesh("crate", CUBE_VERTICES, 'textures/wood.jpg')
        scene.add_instance(asset=crate_asset)
        scene.add_instance(asset=crate_asset, transform=glm.translate(glm.mat4(1.0), glm.vec3(-3.0, 0.0, 0.0)))
        scene.add_instance(asset=crate_asset, transform=glm.translate(glm.mat4(1.0), glm.vec3(3.0, 0.0, 0.0)))

    scene.skybox = Skybox()
    scene.skybox_texture_id = load_cubemap('textures/bright_day')

    eye_model = CompoundEye(num_ommatidia=NB_OMMATIDIA, force_isotropic=True)

    debug_renderer = None  # this is only be used for raytracer debug view

    if USE_RAYTRACER:
        renderer = EyeRendererRay(
            eye_model=eye_model,
            scene=scene,
            time_dithering=TIME_DITHERING,
            nb_samples=NB_SAMPLES,
            point_radius=POINT_RADIUS
        )

        # The debug renderer is only needed when the primary is a raytracer
        if not RUN_HEADLESS:
            debug_renderer = EyeRendererRaster(
                eye_model=eye_model,
                scene=scene,
                window_size=WINDOW_SIZE,
                time_dithering=TIME_DITHERING,
                nb_samples=NB_SAMPLES
            )
    else:
        # If not using raytracer, the rasterizer the one and only renderer
        renderer = EyeRendererRaster(
            eye_model=eye_model,
            scene=scene,
            window_size=WINDOW_SIZE,
            time_dithering=TIME_DITHERING,
            nb_samples=NB_SAMPLES
        )


    if not RUN_HEADLESS:
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    is_running = True

    start = time.time_ns()
    for frame_count in range(HEADLESS_MAX_STEPS if RUN_HEADLESS else 10000):
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                is_running = False

            if event.type == KEYDOWN:
                if event.key == K_c: current_view_idx = (current_view_idx + 1) % len(view_modes)
                if event.key == K_v: VORONOI_VIEW = not VORONOI_VIEW
                # if event.key == K_h: eng.hud.show = not eng.hud.show

                # TODO: Re-implement these controls properly
                # if event.type == KEYDOWN and event.key == K_t: eye_renderer.time_dithering = not eye_renderer.time_dithering
                # if event.type == KEYDOWN and event.key == K_h: eng.show_hud = not eng.show_hud
                # if event.type == KEYDOWN and event.key in (K_KP_PLUS,
                #                                            K_EQUALS): eye_renderer.samples_per_ommatidium *= 2
                # if event.type == KEYDOWN and event.key in (K_KP_MINUS,
                #                                            K_MINUS): eye_renderer.samples_per_ommatidium //= 2
            eng.update_movement()

        if not is_running:
            break

        eng.update_movement()

        # Data acquisition
        # The primary renderer is always responsible for generating the eye data
        ommatidia_values = renderer.get_ommatidia_data(eng.camera, to_cpu=True)

        # Drawing
        if not RUN_HEADLESS:

            glViewport(0, 0, WINDOW_SIZE[0], WINDOW_SIZE[1])
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            view_mode = view_modes[current_view_idx]

            renderer_to_use = renderer

            # if raytracing and in debug view, switch to debug renderer
            if USE_RAYTRACER and view_mode != 'compound_eye':
                renderer_to_use = debug_renderer

            renderer_to_use.draw(view_mode, eng.camera, VORONOI_VIEW)

            # eng.hud.draw()
            eng.clock.tick()
            pygame.display.flip()

    total_time = (time.time_ns() - start) * 1e-9
    print(f"Finished. Total: {frame_count} frames in {total_time:.3f}s ({frame_count / total_time:.2f} FPS).")

    if renderer:
        renderer.free()

    if debug_renderer:
        debug_renderer.free()

    eng.close()
    scene.free()


if __name__ == "__main__":
    main()