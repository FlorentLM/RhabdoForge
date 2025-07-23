import os

from graphics.voronoi_visualiser import VoronoiVisualiser

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
import numpy as np

from OpenGL.GL import *

from graphics.engine import Engine
from graphics.scene import Instance
from graphics.insect_eye import InsectEye
from graphics.panoramic_eye import PanoramicEye
from graphics.glm import translation_mat, rotation_mat
from geometry.primitives import CUBE_VERTICES
from graphics.skybox import Skybox
from graphics.utils import WORLD_UP, load_cubemap


def main():

    IS_HEADLESS = False
    PANORAMIC_DEBUG_MODE = True
    VORONOI_MODE = True
    TILED_VORONOI_MODE = False
    SIMULATION_STEPS = 1000

    eng = Engine(width=1280, height=720, headless=IS_HEADLESS)

    crate_mesh = eng.load_mesh(
        name="crate",
        vertex_data=CUBE_VERTICES,
        vert_shader_path='shaders/base.vert',
        frag_shader_path='shaders/base.frag',
        texture_path='textures/wood.jpg'
    )

    eng.skybox = Skybox()
    eng.skybox_texture_id = load_cubemap('textures/bright_day')

    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([ 0.0, 0.0, 0.0])))
    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([-3.0, 0.0, 0.0])))
    eng.add_instance(Instance(asset=crate_mesh, transform=translation_mat([ 3.0, 0.0, 0.0])))

    print("Initializing insect eye model...")
    insect_eye = InsectEye(num_ommatidia=162, acceptance_angle_deg=30.0)
    pano_debug_view = PanoramicEye()
    voronoi_view = VoronoiVisualiser(insect_eye.num_ommatidia)

    # Simulation variables are defined here for non-interactive mode

    # Define rotation as a fixed amount per frame
    rotation_per_step_deg = 0.5
    current_rotation_deg = 0.0

    if not IS_HEADLESS:
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    is_running = True
    frame_count = 0
    while is_running:

        if not IS_HEADLESS:
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    is_running = False
                if event.type == MOUSEWHEEL:  # Let engine handle this
                    eng.camera.fov -= event.y * 1.5
                if event.type == KEYDOWN and event.key == K_p:
                    PANORAMIC_DEBUG_MODE = not PANORAMIC_DEBUG_MODE
                    print(f"Toggled panoramic debug mode: {'ON' if PANORAMIC_DEBUG_MODE else 'OFF'}")

                if event.type == KEYDOWN and event.key == K_h:
                    insect_eye.samples_per_ommatidium = insect_eye.samples_per_ommatidium * 2
                    print(f"Samples per ommatidium: {insect_eye.samples_per_ommatidium}")

                if event.type == KEYDOWN and event.key == K_g:
                    insect_eye.samples_per_ommatidium = insect_eye.samples_per_ommatidium / 2
                    print(f"Samples per ommatidium: {insect_eye.samples_per_ommatidium}")

                if event.type == KEYDOWN and event.key == K_t:
                    TILED_VORONOI_MODE = not TILED_VORONOI_MODE
                    print(f"Toggled tiled Voronoid debug mode: {'ON' if TILED_VORONOI_MODE else 'OFF'}")

            # Update camera from continuous input (W, A, S, D, mouse)
            eng.update_movement()

        # Update scene state
        current_rotation_deg = (current_rotation_deg + rotation_per_step_deg) % 360.0
        eng.scene.instances[0].transform = rotation_mat(
            np.deg2rad(current_rotation_deg), WORLD_UP
        )

        # ======================== INSECT EYE RENDER PASSES ===================

        # PASS 1: Render the 3D scene into the cubemap FBO
        scene_cubemap_id = eng.render_to_cubemap(eng.scene, eng.camera)

        # PASS 2: Use the generated cubemap to get ommatidia sensory data
        ommatidia_values = insect_eye.get_ommatidia_data(scene_cubemap_id)

        if frame_count % 100 == 0:  # Print a sample every 100 frames
            print(f"Step {frame_count}: Ommatidium 0 value: {ommatidia_values[0]}")
        # Example: np.save(f'data/frame_{frame_count}.npy', ommatidia_values)

        # =====================================================================

        if not IS_HEADLESS:
            glViewport(0, 0, eng.width, eng.height)
            glClearColor(0.05, 0.05, 0.05, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Draw either the panoramic debug view or the insect eye visualization
            if PANORAMIC_DEBUG_MODE:
                pano_debug_view.draw(scene_cubemap_id)
            elif VORONOI_MODE:
                voronoi_view.draw(insect_eye, tiled_mode=TILED_VORONOI_MODE)
            else:
                insect_eye.draw()

            # Draw FPS overlay and update the display
            eng.clock.tick()
            eng._draw_fps()
            pygame.display.flip()

        frame_count += 1
        if IS_HEADLESS and frame_count >= SIMULATION_STEPS:
            is_running = False

    # Need to cleanup when running non-interactive
    print(f"Simulation finished after {frame_count} steps.")
    eng.close()


if __name__ == "__main__":
    main()