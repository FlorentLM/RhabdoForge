import glfw
from pyglm import glm
from typing import Optional, Tuple

from insectvision.engine.world_utils import WORLD_UP, WORLD_DOWN
from insectvision.interactive.utils import DisplayMode
from insectvision.interactive.controls_interface import Controls


class KeyboardMouse(Controls):
    """
    Keyboard and mouse controller
    """

    def __init__(self,
                 look_sensitivity: float = 0.5,
                 keyboard_turn_speed: float = 1.0,
                 invert_mouse_x: bool = False,  # for true psychopaths
                 invert_mouse_y: bool = False,
                 ):

        self.mouse_sensitivity = look_sensitivity
        self.keyboard_turn_speed = keyboard_turn_speed

        self._mouse_x_dir: float = 1.0 if invert_mouse_x else -1.0
        self._mouse_y_dir: float = 1.0 if invert_mouse_y else -1.0

        self._last_mouse_pos: Optional[Tuple[float, float]] = None
        self._ctx = None

        self._sun_orbit_speed = 0.2    # TODO: this should be somewhere else probably

    def setup(self, ctx):
        self._ctx = ctx

        glfw.set_key_callback(ctx.window, self._on_key)
        glfw.set_scroll_callback(ctx.window, self._on_scroll)
        if ctx.mouse_captured:
            glfw.set_input_mode(ctx.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        else:
            glfw.set_input_mode(ctx.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

        self._last_mouse_pos = None

    def free(self):
        if self._ctx is not None and self._ctx.window:
            glfw.set_key_callback(self._ctx.window, None)
            glfw.set_scroll_callback(self._ctx.window, None)
            glfw.set_input_mode(self._ctx.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        self._ctx = None

    # GLFW callbacks

    def _on_key(self, window, key, scancode, action, mods):
        ctx = self._ctx

        if action == glfw.PRESS:
            if key == glfw.KEY_TAB:
                ctx.mouse_captured = not ctx.mouse_captured
            if ctx.mouse_captured:
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)
            else:
                glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
                self._last_mouse_pos = None

            if key == glfw.KEY_C:   ctx.cycle_view_mode()
            elif key == glfw.KEY_V: ctx.toggle_voronoi()
            elif key == glfw.KEY_P: ctx.toggle_projection_mode()
            elif key == glfw.KEY_I: ctx.toggle_hud()
            elif key == glfw.KEY_H: ctx.toggle_heatmap()
            elif key == glfw.KEY_L: ctx.toggle_sun_control()
            elif key == glfw.KEY_X: ctx.dither_once()
            elif key == glfw.KEY_T: ctx.toggle_time_dithering()
            elif key == glfw.KEY_G: ctx.toggle_debug()
            elif key == glfw.KEY_R: ctx.toggle_saccades()
            elif key in (glfw.KEY_KP_ADD, glfw.KEY_EQUAL):      ctx.increase_samples()
            elif key in (glfw.KEY_KP_SUBTRACT, glfw.KEY_MINUS): ctx.decrease_samples()
            elif key == glfw.KEY_KP_MULTIPLY:                   ctx.increase_pixel_samples()
            elif key == glfw.KEY_KP_DIVIDE:                     ctx.decrease_pixel_samples()

        # Custom bindings
        binding = (key, action)
        for callback in ctx._key_bindings.get(binding, ()):
            callback()

    def _on_scroll(self, window, xoffset, yoffset):
        ctx = self._ctx

        if ctx.sun_control_mode and ctx.scene and ctx.scene.sun:
            sun = ctx.scene.sun
            intensity_factor = 1.1 ** yoffset
            sun.intensity = max(0.1, min(10.0, sun.intensity * intensity_factor))
        else:
            zoom_factor = 0.9 ** yoffset
            ctx.observer.zoom(zoom_factor)

    # Per-frame poll

    def poll(self, ctx):

        window = ctx.window
        agent = ctx.agent
        dt = ctx.delta_time

        # Keyboard

        move_direction = glm.vec3(0.0)
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:            move_direction += agent.forward
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:            move_direction += agent.backward
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:        move_direction += WORLD_UP
        if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS: move_direction += WORLD_DOWN

        # Roll
        roll_input = 0.0
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS: roll_input += 1.0
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS: roll_input -= 1.0

        # Yaw / strafe
        yaw_input = 0.0
        strafe_mode = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS

        if ctx.view_mode == DisplayMode.Third_person and not strafe_mode:
            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: yaw_input += 1.0
            if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: yaw_input -= 1.0
        else:
            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: move_direction += agent.left
            if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: move_direction += agent.right

        if glm.length(move_direction) > 0:
            agent.dt(dt).translate(glm.normalize(move_direction) * ctx.move_speed)

        # Mouse

        if not ctx.mouse_captured:
            return

        current_mouse_pos = glfw.get_cursor_pos(window)
        if self._last_mouse_pos is None:
            self._last_mouse_pos = current_mouse_pos

        dx = current_mouse_pos[0] - self._last_mouse_pos[0]
        dy = current_mouse_pos[1] - self._last_mouse_pos[1]
        self._last_mouse_pos = current_mouse_pos

        # Sun control: mouse orbits the sun
        if ctx.sun_control_mode and ctx.scene and ctx.scene.sun:
            sun = ctx.scene.sun

            new_azimuth = sun.azimuth + dx * self._sun_orbit_speed * -1
            new_elevation = sun.elevation - dy * self._sun_orbit_speed
            new_elevation = max(1.0, min(89.0, new_elevation))

            if abs(dx) > 0.1 or abs(dy) > 0.1:
                sun.from_angles(new_azimuth, new_elevation, sun.distance)

            agent.rotate(
                yaw_delta=yaw_input * self.keyboard_turn_speed,
                roll_delta=roll_input * self.keyboard_turn_speed,
                degrees=True
            )

        # Normal view control
        else:
            mouse_yaw = dx * self.mouse_sensitivity * self._mouse_y_dir
            mouse_pitch = dy * self.mouse_sensitivity * self._mouse_y_dir

            if ctx.view_mode == DisplayMode.Third_person:
                ctx.observer.pan(
                    azimuth_delta=mouse_yaw * 0.5,
                    elevation_delta=mouse_pitch * 0.5,
                    degrees=True
                )
                agent.rotate(
                    yaw_delta=yaw_input * self.keyboard_turn_speed,
                    roll_delta=roll_input * self.keyboard_turn_speed,
                    degrees=True
                )
            else:
                agent.rotate(
                    yaw_delta=mouse_yaw,
                    pitch_delta=mouse_pitch,
                    roll_delta=roll_input * self.keyboard_turn_speed,
                    degrees=True
                )

        # Resets
        if glfw.get_key(window, glfw.KEY_O) == glfw.PRESS:
            ctx.reset_position()
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            ctx.reset_rotation()
