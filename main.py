import time
from collections import deque
import numpy as np

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from OpenGL.GL import *

from camera import Camera
from engine import Input
from glm import Matrices
import engine
import things


##

def main(controls=True, use_imgui=True):

    display = 1280, 1024

    # -- Initialise window GLFW and optionally hook it to imgui
    if use_imgui:
        imgui.create_context()

    window = engine.init_glfw(width=display[0], height=display[1], name='Antworlds')

    if use_imgui:
        impl = GlfwRenderer(window)
    # --

    # Enable OpenGL depth testing
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    if controls:
        glfw.set_key_callback(window, Input.get_keys)
        glfw.set_cursor_pos_callback(window, Input.get_mouse)
        glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        glfw.set_cursor_pos(window, 0, 0)
        glfw.set_mouse_button_callback(window, Input.get_mousebuttons)
        glfw.set_scroll_callback(window, Input.get_scroll)

    # --

    # Set up starting camera position and aspect
    cam = Camera()
    cam.pos = 0, 0, 4
    cam.ratio = display[0] / display[1]

    # Load Assets
    c = things.CubeAsset(texture_name='rock')
    # t = things.TerrainAsset(texture_name='grass')

    # Create instances
    # terrain = things.Instance(t)
    # terrain.transform = Matrices.translation([0, -1, 0]) @ Matrices.scaling([10, 1, 10])

    crate_0 = things.Instance(c)
    crate_1 = things.Instance(c)
    crate_1.transform = Matrices.translation([-3, 0, 0])
    crate_2 = things.Instance(c)
    crate_2.transform = Matrices.translation([3, 0, 0])

    instances = [crate_0, crate_1, crate_2]

    # Initialise some global variables to be used during runtime

    # This is for an example object-space rotation
    test_rotation_speed = 45.0      # in degrees per second
    cam_move_speed = 3.0            # in metres per second (?)
    zoom_speed = 0.1                # idk :(
    mouse_speed = 0.01              # idk :(

    cumulative_test_rotation = 0    # i.e. current angle of the rotating test object

    len_fps_rolling = 100
    fps_rolling = deque(maxlen=len_fps_rolling)

    tick = time.time_ns()
    running_t = 0
    while not glfw.window_should_close(window):

        # Clear the render with black
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Per the documentation: GLFW needs to poll the window system for events both to provide input to the
        # application and to prove to the window system that the application hasn't locked up
        glfw.poll_events()

        # Update passed time variables
        tock = time.time_ns()
        t = (tock - tick) * 1e-9
        running_t += t
        fps_rolling.append(1.0 / t)
        fps = sum(fps_rolling) / len_fps_rolling  # This will be wrong for a few frames but it's faster than using len()

        # Move camera
        if controls:
            to_move_this_frame = cam_move_speed * t

            if Input.quit:
                break

            if Input.forward:
                cam.position += cam.forward * to_move_this_frame
            if Input.backward:
                cam.position += cam.backward * to_move_this_frame
            if Input.left:
                cam.position += cam.left * to_move_this_frame
            if Input.right:
                cam.position += cam.right * to_move_this_frame
            if Input.up:
                cam.position += engine.WORLD_UP * to_move_this_frame
            if Input.down:
                cam.position += engine.WORLD_DOWN * to_move_this_frame

            cam.fov += Input.mouse_wh * zoom_speed
            cam.yaw -= Input.mouse_x * mouse_speed
            cam.pitch -= Input.mouse_y * mouse_speed

        # Update object transforms
        to_rotate_this_frame = test_rotation_speed * t

        cumulative_test_rotation += to_rotate_this_frame
        while cumulative_test_rotation > 360.0:
            cumulative_test_rotation -= 360.0

        crate_0.transform = Matrices.rotation(np.deg2rad(cumulative_test_rotation), engine.WORLD_UP)

        # Render all instances
        for instance in instances:
            things.render_instance(instance, cam)  # We render to the cam - TODO - render to texture instead?

        # imgui main loop - needs to be done *after* rendering the scene so it appears in front of things
        if use_imgui:
            # Process inputs from imgui
            impl.process_inputs()

            # Start new frame context
            imgui.new_frame()

            # Draw text label inside of current window
            imgui.text(f'{fps:.2f} fps')

            # Pass all drawing comands to the rendering pipeline and close frame context
            imgui.render()
            impl.render(imgui.get_draw_data())
            imgui.end_frame()

        else:
            if running_t >= 1:
                print(f'{fps:.2f} fps')
                running_t = 0

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # And finally update the time for current frame
        tick = tock

    if use_imgui:
        impl.shutdown()
    glfw.terminate()


##

if __name__ == "__main__":
    main(controls=True, use_imgui=True)

