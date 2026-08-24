import glfw
from pyglm import glm
from typing import Optional, Dict

from rhabdoforge.types import WORLD_UP, WORLD_DOWN, DisplayMode
from rhabdoforge.interactive.controls import Controls


class Gamepad(Controls):
    """
    Gamepad controller.
    """

    def __init__(self,
            look_sensitivity: float = 240.0,
            move_sensitivity: float = 1.0,
            invert_joystick_x: bool = False,
            invert_joystick_y: bool = False,
            deadzone: float = 0.30,
        ):

        self.look_sensitivity = look_sensitivity
        self.move_sensitivity = move_sensitivity
        self.deadzone = deadzone

        self._look_x_dir = -1.0 if invert_joystick_x else 1.0
        self._look_y_dir = -1.0 if invert_joystick_y else 1.0

        self._sun_orbit_speed = 60.0
        self._gamepad_id: Optional[int] = None
        self._prev_buttons: Dict[int, bool] = {}
        self.ctx = None

        self.action_map: Dict[int, str] = {
            glfw.GAMEPAD_BUTTON_RIGHT_THUMB: 'view_cycle',
            glfw.GAMEPAD_BUTTON_LEFT_THUMB: 'sun_ctrl_toggle',
            glfw.GAMEPAD_BUTTON_BACK: 'hud_toggle',
            glfw.GAMEPAD_BUTTON_START: 'reset_pos',
            glfw.GAMEPAD_BUTTON_B: 'voronoi_toggle',    # circle
            glfw.GAMEPAD_BUTTON_Y: 'proj_toggle',       # triangle
            glfw.GAMEPAD_BUTTON_DPAD_UP: 'samples_inc',
            glfw.GAMEPAD_BUTTON_DPAD_DOWN: 'samples_dec',
            glfw.GAMEPAD_BUTTON_DPAD_LEFT: 'dither_toggle',
            glfw.GAMEPAD_BUTTON_DPAD_RIGHT: 'saccade_toggle',
            glfw.GAMEPAD_BUTTON_A: 'dither_once',
        }

    def setup(self, ctx):
        self.ctx = ctx
        self._prev_buttons.clear()
        self._detect_gamepad()

    def free(self):
        self.ctx = None
        self._gamepad_id = None
        self._prev_buttons.clear()

    def _detect_gamepad(self):
        for jid in range(glfw.JOYSTICK_LAST + 1):
            if glfw.joystick_is_gamepad(jid):
                self._gamepad_id = jid
                return
        self._gamepad_id = None

    def _set_deadzone(self, value: float) -> float:
        dz = self.deadzone
        if abs(value) < dz:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - dz) / (1.0 - dz)

    def _button_pressed(self, state, button: int) -> bool:
        """Helper to detect the rising edge of a button press."""

        current = bool(state.buttons[button])
        prev = self._prev_buttons.get(button, False)
        self._prev_buttons[button] = current

        return current and not prev

    def poll(self, ctx):
        if self._gamepad_id is None:
            if glfw.get_time() % 2.0 < 0.01:  # Periodically check for plug-in
                self._detect_gamepad()
            return

        if not glfw.joystick_is_gamepad(self._gamepad_id):
            self._gamepad_id = None
            self._detect_gamepad()
            return

        state = glfw.get_gamepad_state(self._gamepad_id)
        if state is None:
            return

        # Discrete actions (buttons)
        self._poll_buttons(ctx, state)

        # Continuous inputs (sticks/triggers)
        self._poll_axes(ctx, state)

    def _poll_buttons(self, ctx, state):

        for btn_const, action_id in self.action_map.items():
            if self._button_pressed(state, btn_const):
                ctx.actions.trigger(action_id)

        # Combos (LB + RB for reset rotation)
        lb = state.buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER]
        rb = state.buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER]

        combo_active = lb and rb
        prev_combo = self._prev_buttons.get(999, False)  # using 999 as virtual id
        self._prev_buttons[999] = combo_active

        if combo_active and not prev_combo:
            ctx.actions.trigger('reset_rot')

        # LB + DPAD UP for Heatmap
        # TODO: Add more combos
        if lb and self._button_pressed(state, glfw.GAMEPAD_BUTTON_DPAD_UP):
            ctx.actions.trigger('heatmap_toggle')

    def _poll_axes(self, ctx, state):

        agent = ctx.renderer.agent
        scene = ctx.renderer.scene
        wall_dt = ctx.wall_dt

        # Normalised stick inputs
        lx = self._set_deadzone(state.axes[glfw.GAMEPAD_AXIS_LEFT_X])
        ly = self._set_deadzone(state.axes[glfw.GAMEPAD_AXIS_LEFT_Y])
        rx = self._set_deadzone(state.axes[glfw.GAMEPAD_AXIS_RIGHT_X])
        ry = self._set_deadzone(state.axes[glfw.GAMEPAD_AXIS_RIGHT_Y])

        # Triggers
        lt = (state.axes[glfw.GAMEPAD_AXIS_LEFT_TRIGGER] + 1.0) * 0.5
        rt = (state.axes[glfw.GAMEPAD_AXIS_RIGHT_TRIGGER] + 1.0) * 0.5
        lt = lt if lt > 0.1 else 0.0
        rt = rt if rt > 0.1 else 0.0

        # Movement
        move_direction = glm.vec3(0.0)
        left_stick_yaw_delta = 0.0

        if abs(ly) > 0:
            move_direction += agent.forward * (-ly)

        if ctx.display_mode == DisplayMode.Third_person:
            left_stick_yaw_delta = -lx * self.look_sensitivity * wall_dt
        else:
            if abs(lx) > 0:
                move_direction += agent.right * lx

        # Vertical movement
        if state.buttons[glfw.GAMEPAD_BUTTON_X]:
            move_direction += WORLD_DOWN
        if state.buttons[glfw.GAMEPAD_BUTTON_A]:
            move_direction += WORLD_UP

        if glm.length(move_direction) > 0:
            speed = ctx.move_speed * self.move_sensitivity
            agent.translate(glm.normalize(move_direction) * speed * wall_dt)

        # Zoom / Sun intensity
        scroll_delta = (rt - lt) * wall_dt * 10.0

        if abs(scroll_delta) > 0:
            if ctx.sun_control_mode and scene and scene.sun:
                sun = scene.sun
                intensity_factor = 1.1 ** scroll_delta
                sun.intensity = max(0.1, min(10.0, sun.intensity * intensity_factor))

            elif ctx.display_mode == DisplayMode.Third_person:
                zoom_factor = 0.9 ** scroll_delta
                ctx.observer.zoom(zoom_factor)

        # Looking
        if abs(rx) > 0 or abs(ry) > 0 or abs(left_stick_yaw_delta) > 0:

            if ctx.sun_control_mode and scene and scene.sun:
                sun = scene.sun
                new_azimuth = sun.azimuth - (rx * self._sun_orbit_speed * wall_dt)
                new_elevation = sun.elevation - (ry * self._sun_orbit_speed * wall_dt)
                new_elevation = max(1.0, min(89.0, new_elevation))

                sun.from_angles(new_azimuth, new_elevation)

                if abs(left_stick_yaw_delta) > 0:
                    agent.rotate(yaw=left_stick_yaw_delta, degrees=True)

            elif ctx.display_mode == DisplayMode.Third_person:
                ctx.observer.pan(
                    -rx * self._look_x_dir * self.look_sensitivity * wall_dt,
                    ry * self._look_y_dir * self.look_sensitivity * wall_dt
                )
                if abs(left_stick_yaw_delta) > 0:
                    agent.rotate(yaw=left_stick_yaw_delta, degrees=True)

            else:
                agent.rotate(
                    yaw=-rx * self._look_x_dir * self.look_sensitivity * wall_dt,
                    pitch=ry * self._look_y_dir * self.look_sensitivity * wall_dt,
                    degrees=True
                )

        # Roll
        roll = 0.0
        if state.buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER] and not state.buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER]:
            roll += 1.0
        elif state.buttons[glfw.GAMEPAD_BUTTON_RIGHT_BUMPER] and not state.buttons[glfw.GAMEPAD_BUTTON_LEFT_BUMPER]:
            roll -= 1.0

        if roll != 0.0:
            agent.rotate(roll=roll * self.look_sensitivity * wall_dt, degrees=True)