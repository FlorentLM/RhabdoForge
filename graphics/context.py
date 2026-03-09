import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

import glfw
from pyglm import glm
import time
from typing import Optional, Tuple

from graphics.renderers.base import BaseInsectEyeRenderer
from graphics.scene import Scene
from graphics.agent import Agent, OrbitCamera
from graphics.utils import WORLD_UP, WORLD_DOWN, ViewMode, ProjectionMode
from graphics.interactive.hud import HUD
from graphics.debug import DebugOverlay


class Context:

    def __init__(self,
                 window_size: tuple = None,
                 debug_overlay: bool = True,
                 fps_limit: int = None,
                 v_sync: bool = False,
                 invert_mouseY: bool = False
         ):

        self._window_size = window_size if window_size is not None else (1280, 720)
        self._fps_limit = fps_limit if fps_limit is not None else 0
        self._interactive_initialised = False
        self._v_sync = v_sync

        if not glfw.init():
            raise Exception("GLFW can't be initialized")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self.window = glfw.create_window(self._window_size[0], self._window_size[1],
                                         title="Interactive mode",
                                         monitor=None,
                                         share=None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")

        glfw.make_context_current(self.window)

        glfw.swap_interval(int(self._v_sync))

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_FRAMEBUFFER_SRGB)  # we want linear (non-gamma corrected)

        self.agent: Optional[Agent] = None
        self.renderer: Optional[BaseInsectEyeRenderer] = None
        self.scene: Optional[Scene] = None
        self.observer: Optional[OrbitCamera] = None
        self.view_mode: Optional[ViewMode] = None
        self.hud: Optional[HUD] = None
        self.last_mouse_pos: Optional[Tuple[float, float]] = None

        self.debug: Optional[DebugOverlay] = DebugOverlay() if debug_overlay else None

        self.last_frame_time: float = 0.0
        self._delta_time: float = 0.0

        self.move_speed: float = 3.0
        self.keyboard_turn_speed: float = 1.0
        self.mouse_sensitivity: float = 0.5
        self.mouse_y_dir: float = 1.0 if invert_mouseY else -1.0

        # Sun control mode
        self.sun_control_mode: bool = False
        self.sun_orbit_sensitivity: float = 0.2

    def run_interactive(self, agent: Agent, scene: Scene, renderer: BaseInsectEyeRenderer,
                        window_size=None, fps_limit=None, v_sync=None, invert_mouseY=None):
        """ On first call, initialises and shows the window. Then checks if the interactive loop should continue. """

        if not self._interactive_initialised:

            if window_size is not None:
                self.window_size = window_size

            if fps_limit is not None:
                self.fps_limit = fps_limit

            if v_sync is not None:
                self._v_sync = bool(v_sync)

            if invert_mouseY is not None:
                self.mouse_y_dir = 1 if invert_mouseY else -1

            glfw.swap_interval(int(self._v_sync))

            glfw.show_window(self.window)

            self.agent = agent
            self.scene = scene
            self.renderer = renderer
            renderer.runs_interactive = True

            self.observer = OrbitCamera(target=agent, distance=1.5, ratio=self._window_size[0] / self._window_size[1])

            glfw.set_key_callback(self.window, self.key_callback)
            # glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
            glfw.set_scroll_callback(self.window, self.scroll_callback)

            self.view_mode = ViewMode.compound_eye

            self.hud = HUD(self)

            # Initialise last_frame_time right before the loop starts to prevent a massive initial delta_time
            self.last_frame_time = self.current_time

            self._interactive_initialised = True

        return not glfw.window_should_close(self.window)

    def scroll_callback(self, window, xoffset, yoffset):
        if self.sun_control_mode and self.scene and self.scene.sun:
            # scroll adjusts sun intensity
            sun = self.scene.sun
            intensity_factor = 1.1 ** yoffset  # yoffset > 0 => brighter
            sun.intensity = max(0.1, min(10.0, sun.intensity * intensity_factor))
        else:
            # Normal mode: scroll zooms camera
            zoom_factor = 0.9 ** yoffset  # yoffset > 0 => 0.9 (zoom in), yoffset < 0 => 1.11 (zoom out)
            self.observer.zoom(zoom_factor)

    def key_callback(self, window, key, scancode, action, mods):

        if action == glfw.PRESS:

            if key == glfw.KEY_C:
                self.view_mode = (self.view_mode + 1) % 3

            if key == glfw.KEY_V:
                self.renderer.tiled_mode = not self.renderer.tiled_mode

            if key == glfw.KEY_P:
                self.renderer.projection_mode = ProjectionMode.Physical if self.renderer.projection_mode == ProjectionMode.Acceptance else ProjectionMode.Acceptance

            if key == glfw.KEY_I:
                if self.hud:
                    self.hud.show = not self.hud.show

            if key == glfw.KEY_H:
                self.renderer.heatmap_enabled = not self.renderer.heatmap_enabled

            if key == glfw.KEY_L:
                self.sun_control_mode = not self.sun_control_mode
                mode_name = "Sun" if self.sun_control_mode else "View"
                print(f"Mouse control: {mode_name}")

            if key == glfw.KEY_X:
                self.renderer.dither()

            if key == glfw.KEY_T:
                if hasattr(self.renderer, 'time_dithering'):
                    self.renderer.time_dithering = not self.renderer.time_dithering

            if key in (glfw.KEY_KP_ADD, glfw.KEY_EQUAL):
                if hasattr(self.renderer, 'samples_per_ommatidium'):
                    self.renderer.samples_per_ommatidium *= 2

            if key in (glfw.KEY_KP_SUBTRACT, glfw.KEY_MINUS):
                if hasattr(self.renderer, 'samples_per_ommatidium'):
                    self.renderer.samples_per_ommatidium = max(1, self.renderer.samples_per_ommatidium // 2)

            if key == glfw.KEY_KP_MULTIPLY:
                if hasattr(self.renderer, 'samples_per_pixel'):
                    self.renderer.samples_per_pixel *= 2

            if key == glfw.KEY_KP_DIVIDE:
                if hasattr(self.renderer, 'samples_per_pixel'):
                    self.renderer.samples_per_pixel = max(1, self.renderer.samples_per_pixel // 2)

            if key == glfw.KEY_G:
                if self.debug is not None:
                    self.debug.enabled = not self.debug.enabled

    def input(self):

        if not self._interactive_initialised:
            return

        current_time = self.current_time
        self._delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        glfw.poll_events()

        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
            return

        # Movement
        move_direction = glm.vec3(0.0)
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS: move_direction += self.agent.forward
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS: move_direction += self.agent.backward
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS: move_direction += WORLD_UP
        if glfw.get_key(self.window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS: move_direction += WORLD_DOWN

        # Rotation inputs
        roll_input = 0.0
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS: roll_input += 1.0
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS: roll_input -= 1.0

        yaw_input = 0.0
        strafe_mode = glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS

        if self.view_mode == ViewMode.third_person and not strafe_mode:
            # in 3rd person, A/D turns the agent
            if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS: yaw_input += 1.0
            if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS: yaw_input -= 1.0
        else:
            # in 1st person or when holding Shift, A/D strafes
            if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS: move_direction += self.agent.left
            if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS: move_direction += self.agent.right

        # Apply translation
        if glm.length(move_direction) > 0:
            self.agent.dt(self._delta_time).translate(glm.normalize(move_direction) * self.move_speed)

        # Mouse input
        current_mouse_pos = glfw.get_cursor_pos(self.window)
        if self.last_mouse_pos is None:
            self.last_mouse_pos = current_mouse_pos

        dx = (current_mouse_pos[0] - self.last_mouse_pos[0])
        dy = (current_mouse_pos[1] - self.last_mouse_pos[1])
        self.last_mouse_pos = current_mouse_pos

        # Sun control: mouse orbits the sun around the scene
        if self.sun_control_mode and self.scene and self.scene.sun:
            sun = self.scene.sun
            current_azimuth = sun.azimuth
            current_elevation = sun.elevation
            current_distance = sun.distance

            # Horizontal mouse = azimuth change
            new_azimuth = current_azimuth + dx * self.sun_orbit_sensitivity * -1

            # Vertical mouse = elevation change (clamped to stay above horizon)
            new_elevation = current_elevation - dy * self.sun_orbit_sensitivity
            new_elevation = max(1.0, min(89.0, new_elevation))

            # Apply if there was any mouse movement
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                sun.from_angles(new_azimuth, new_elevation, current_distance)

            # Keyboard rotates the agent
            self.agent.rotate(
                yaw_delta=yaw_input * self.keyboard_turn_speed,
                roll_delta=roll_input * self.keyboard_turn_speed,
                degrees=True
            )

        # Normal view control
        else:
            mouse_yaw_delta = dx * self.mouse_sensitivity * -1
            mouse_pitch_delta = dy * self.mouse_sensitivity * self.mouse_y_dir

            # Apply rotation
            if self.view_mode == ViewMode.third_person:
                # Mouse pans the camera
                self.observer.pan(azimuth_delta=mouse_yaw_delta * 0.5,
                                  elevation_delta=mouse_pitch_delta * 0.5,
                                  degrees=True)

                # Keyboard rotates the agent
                self.agent.rotate(
                    yaw_delta=yaw_input * self.keyboard_turn_speed,
                    roll_delta=roll_input * self.keyboard_turn_speed,
                    degrees=True
                )
            else:
                # First-person: Mouse controls yaw/pitch, keyboard controls roll
                self.agent.rotate(
                    yaw_delta=mouse_yaw_delta,
                    pitch_delta=mouse_pitch_delta,
                    roll_delta=roll_input * self.keyboard_turn_speed,
                    degrees=True
                )

        # Resets
        if glfw.get_key(self.window, glfw.KEY_O) == glfw.PRESS: self.agent.position = (0.0, 0.0, 0.0)
        if glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS: self.agent.yaw, self.agent.pitch, self.agent.roll = (
            0.0, 0.0, 0.0)

    def draw(self):

        if not self._interactive_initialised:
            return

        if self.scene:
            # convert to linear (non gamma-corrected)
            linear_bg_color = tuple(pow(c, 2.2) for c in self.scene.background_color)
            glClearColor(linear_bg_color[0], linear_bg_color[1], linear_bg_color[2], 1.0)

        glViewport(0, 0, self._window_size[0], self._window_size[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.view_mode == ViewMode.third_person:
            self.observer.ratio = self._window_size[0] / self._window_size[1]
            self.observer.update()
            pov = self.observer
        else:
            pov = self.agent

        self.renderer.draw(self.view_mode, pov, self.agent)

        if self.debug is not None and self.view_mode != ViewMode.panoramic: # TODO: debug projection in panoramic mode? idk if useful
            self.debug.draw(view=pov.view, proj=pov.projection)

        if self.hud:
            self.hud.draw()

        glfw.swap_buffers(self.window)

        if self._fps_limit > 0:
            frame_end_time = self.current_time
            elapsed_time = frame_end_time - self.last_frame_time
            wait_time = (1.0 / self._fps_limit) - elapsed_time

            if wait_time > 0:
                time.sleep(wait_time)

    def free(self):
        if self.debug:
            self.debug.free()
        if self.hud:
            self.hud.free()
        glfw.terminate()

    @property
    def current_time(self) -> float:
        return glfw.get_time()

    @property
    def delta_time(self) -> float:
        return self._delta_time

    @property
    def window_size(self) -> tuple:
        return self._window_size

    @window_size.setter
    def window_size(self, value: tuple):
        self._window_size = value
        if self.window:
            glfw.set_window_size(self.window, value[0], value[1])

    @property
    def fps_limit(self) -> int:
        return self._fps_limit

    @fps_limit.setter
    def fps_limit(self, value: Optional[int] = None):
        self._fps_limit = max(0, value) if value else 0

    @property
    def v_sync(self) -> bool:
        return self._v_sync

    @v_sync.setter
    def v_sync(self, value: bool):
        self._v_sync = bool(value)