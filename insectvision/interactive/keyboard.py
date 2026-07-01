import glfw
from pyglm import glm
from typing import Optional, Tuple, Dict

from insectvision.engine.world_utils import WORLD_UP, WORLD_DOWN
from insectvision.utils import DisplayMode
from insectvision.interactive.controls import Controls


class KeyboardMouse(Controls):
    """
    Keyboard and mouse controller.
    """

    def __init__(self,
                 look_sensitivity: float = 0.5,
                 keyboard_turn_speed: float = 1.0,
                 invert_mouse_x: bool = False,
                 invert_mouse_y: bool = False,
                 ):

        self.mouse_sensitivity = look_sensitivity
        self.keyboard_turn_speed = keyboard_turn_speed

        self._mouse_x_dir: float = 1.0 if invert_mouse_x else -1.0
        self._mouse_y_dir: float = 1.0 if invert_mouse_y else -1.0

        self._last_mouse_pos: Optional[Tuple[float, float]] = None
        self.ctx = None

        self._sun_orbit_speed = 0.2

        self.action_map: Dict[int, str] = {
            glfw.KEY_C: 'view_cycle',
            glfw.KEY_V: 'voronoi_toggle',
            glfw.KEY_P: 'proj_toggle',
            glfw.KEY_I: 'hud_toggle',
            glfw.KEY_H: 'heatmap_toggle',
            glfw.KEY_L: 'sun_ctrl_toggle',
            glfw.KEY_T: 'dither_toggle',
            glfw.KEY_R: 'saccade_toggle',
            glfw.KEY_O: 'reset_pos',
            glfw.KEY_BACKSPACE: 'reset_rot',
            glfw.KEY_EQUAL: 'samples_inc',
            glfw.KEY_KP_ADD: 'samples_inc',
            glfw.KEY_MINUS: 'samples_dec',
            glfw.KEY_KP_SUBTRACT: 'samples_dec',
            glfw.KEY_TAB: 'mouse_lock_toggle',
            glfw.KEY_X: 'dither_once',
            glfw.KEY_G: 'debug_toggle',
            glfw.KEY_INSERT: 'take_snapshot',
        }

    def setup(self, ctx):
        self.ctx = ctx

        glfw.set_key_callback(self.ctx.window, self._on_key)
        glfw.set_scroll_callback(self.ctx.window, self._on_scroll)
        glfw.set_mouse_button_callback(self.ctx.window, self._on_mouse_button)

        # Force sync glfw state with context property on boot
        self.ctx.mouse_captured = self.ctx.mouse_captured

        self._last_mouse_pos = None

    def free(self):

        if self.ctx is not None and self.ctx.window:
            glfw.set_key_callback(self.ctx.window, None)
            glfw.set_scroll_callback(self.ctx.window, None)
            glfw.set_mouse_button_callback(self.ctx.window, None)
            glfw.set_input_mode(self.ctx.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

        self.ctx = None

    def _on_key(self, window, key, scancode, action, mods):

        if action != glfw.PRESS:
            return

        action_id = self.action_map.get(key)
        if action_id:
            self.ctx.actions.trigger(action_id)

        # Custom bindings
        binding = (key, action)
        for callback in self.ctx._key_bindings.get(binding, ()):
            callback()

    def _on_mouse_button(self, window, button, action, mods):
        if action == glfw.PRESS and button == glfw.MOUSE_BUTTON_LEFT:
            if not self.ctx.mouse_captured:
                x, y = glfw.get_cursor_pos(window)
                w, h = glfw.get_window_size(window)
                if w > 0 and h > 0:
                    ndc_x = (x / w) * 2.0 - 1.0
                    ndc_y = 1.0 - (y / h) * 2.0

                    picked_lens = self.ctx.pick_ommatidium(ndc_x, ndc_y)
                    if picked_lens is not None:
                        if self.ctx.dashboard:
                            is_shift = (mods & glfw.MOD_SHIFT) != 0
                            self.ctx.dashboard.toggle_omm_selection(picked_lens, multi=is_shift)

    def _on_scroll(self, window, xoffset, yoffset):

        if self.ctx.sun_control_mode and self.ctx.scene and self.ctx.scene.sun:
            sun = self.ctx.scene.sun
            intensity_factor = 1.1 ** yoffset
            sun.intensity = max(0.1, min(10.0, sun.intensity * intensity_factor))

        else:
            zoom_factor = 0.9 ** yoffset
            self.ctx.observer.zoom(zoom_factor)

    def poll(self, ctx):
        """
        Per-frame polling for continuous movement and mouse look.
        """

        window = self.ctx.window
        wall_dt = self.ctx.wall_dt

        agent = self.ctx.renderer.agent

        # Continuous movement (WASD + vertical)
        move_direction = glm.vec3(0.0)
        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:            move_direction += agent.forward
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:            move_direction += agent.backward
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:        move_direction += WORLD_UP
        if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS: move_direction += WORLD_DOWN

        # Yaw / strafe
        strafe_mode = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        yaw_input = 0.0

        if self.ctx.display_mode == DisplayMode.Third_person and not strafe_mode:
            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: yaw_input += 1.0
            if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: yaw_input -= 1.0
        else:
            if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS: move_direction += agent.left
            if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS: move_direction += agent.right

        if glm.length(move_direction) > 0:
            agent.translate(glm.normalize(move_direction) * self.ctx.move_speed * wall_dt)

        # Roll
        roll_input = 0.0
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS: roll_input += 1.0
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS: roll_input -= 1.0

        # Mouse look
        if not self.ctx.mouse_captured:
            self._last_mouse_pos = None

            if yaw_input != 0 or roll_input != 0:
                agent.rotate(
                    yaw=yaw_input * self.keyboard_turn_speed,
                    roll=roll_input * self.keyboard_turn_speed,
                    degrees=True
                )
            return

        current_mouse_pos = glfw.get_cursor_pos(window)
        if self._last_mouse_pos is None:
            self._last_mouse_pos = current_mouse_pos

        dx = current_mouse_pos[0] - self._last_mouse_pos[0]
        dy = current_mouse_pos[1] - self._last_mouse_pos[1]
        self._last_mouse_pos = current_mouse_pos

        # Sun control
        if self.ctx.sun_control_mode and self.ctx.scene and self.ctx.scene.sun:
            sun = self.ctx.scene.sun
            new_azimuth = sun.azimuth + dx * self._sun_orbit_speed * -1
            new_elevation = sun.elevation - dy * self._sun_orbit_speed
            new_elevation = max(0.01, min(89.99, new_elevation))

            if abs(dx) > 0.1 or abs(dy) > 0.1:
                sun.from_angles(new_azimuth, new_elevation)

            agent.rotate(
                yaw=yaw_input * self.keyboard_turn_speed,
                roll=roll_input * self.keyboard_turn_speed,
                degrees=True
            )

        # Standard control
        else:
            mouse_yaw = dx * self.mouse_sensitivity * self._mouse_x_dir
            mouse_pitch = dy * self.mouse_sensitivity * self._mouse_y_dir

            if self.ctx.display_mode == DisplayMode.Third_person:
                self.ctx.observer.pan(mouse_yaw * 0.5, mouse_pitch * 0.5)
                agent.rotate(
                    yaw=yaw_input * self.keyboard_turn_speed,
                    roll=roll_input * self.keyboard_turn_speed,
                    degrees=True
                )
            else:
                agent.rotate(
                    yaw=mouse_yaw,
                    pitch=mouse_pitch,
                    roll=roll_input * self.keyboard_turn_speed,
                    degrees=True
                )