import time
from typing import Optional
import OpenGL

OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

import glfw
from pyglm import glm

from graphics.renderers.base import EyeRendererBase
from graphics.scene import Scene
from graphics.agent import Agent
from graphics.utils import WORLD_UP, WORLD_DOWN, ViewMode
from graphics.interactive.hud import HUD


class Context:

    def __init__(self, window_size: tuple = None, fps_limit: int = None, v_sync: bool = False, invert_mouseY = False):

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

        glEnable(GL_FRAMEBUFFER_SRGB)

        self.agent = None
        self.renderer = None
        self.scene = None
        self.camera = None
        self.orbit = None
        self.view_mode = None
        self.hud = None
        self.last_mouse_pos = None
        self.last_frame_time = 0
        self._delta_time = 0

        self.move_speed: float = 3.0
        self.roll_speed: float = 2.0
        self.mouse_sensitivity: float = 0.5
        self.mouse_y_dir = 1 if invert_mouseY else -1

    def run_interactive(self, agent: Agent, scene: Scene, renderer: EyeRendererBase,
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

            self.camera = Agent(fov=60.0, ratio=self._window_size[0] / self._window_size[1])  # orbit camera
            self.orbit = {
                "distance": 1.5,  # meters behind the insect
                "azimuth": 0.0,  # radians, around Y (left/right)
                "elevation": glm.radians(20.0),  # radians, up/down
                "min_elev": glm.radians(-85.0),
                "max_elev": glm.radians(85.0)
            }

            glfw.set_key_callback(self.window, self.key_callback)
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
            glfw.set_scroll_callback(self.window, self.scroll_callback)

            self.view_mode = ViewMode.compound_eye

            self.hud = HUD(self)

            # Initialize last_frame_time right before the loop starts to prevent a massive initial delta_time
            self.last_frame_time = self.current_time

            self._interactive_initialised = True

        return not glfw.window_should_close(self.window)

    def _update_orbit_camera(self):
        """ Positions self.camera on a sphere around the insect agent and looks at it """
        t = self.agent.position
        d = self.orbit["distance"]
        az = self.orbit["azimuth"]
        el = self.orbit["elevation"]

        # Spherical (Y-up): x = d*cos(el)*sin(az), y = d*sin(el), z = d*cos(el)*cos(az)
        offset = glm.vec3(d * glm.cos(el) * glm.sin(az),
                          d * glm.sin(el),
                          d * glm.cos(el) * glm.cos(az))
        self.camera.position = t + offset
        self.camera.lookat(t)  # points at the insect center

    def scroll_callback(self, window, xoffset, yoffset):
        # zoom the orbit with mouse wheel: yoffset > 0 => zoom in
        self.orbit["distance"] = float(max(0.1, self.orbit["distance"] * (0.9 ** yoffset)))

    def key_callback(self, window, key, scancode, action, mods):

        if action == glfw.PRESS:

            # if key == glfw.KEY_C: self.view_mode = (self.view_mode + 1) % len(ViewMode)
            if key == glfw.KEY_C: self.view_mode = (self.view_mode + 1) % 2

            if key == glfw.KEY_V: self.renderer.tiled_mode = not self.renderer.tiled_mode

            if key == glfw.KEY_P: self.renderer.projection_mode = 'physical_layout' if self.renderer.projection_mode == 'visual_field' else  'visual_field'

            if key == glfw.KEY_H:
                if self.hud: self.hud.show = not self.hud.show

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

        # Handle movement
        move_direction = glm.vec3(0.0)
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS: move_direction += self.agent.forward
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS: move_direction += self.agent.backward
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS: move_direction += self.agent.left
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS: move_direction += self.agent.right
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS: move_direction += WORLD_UP
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS: move_direction += WORLD_DOWN

        # Reset position/orientation (instantaneous, so no .dt())
        if glfw.get_key(self.window, glfw.KEY_O) == glfw.PRESS: self.agent.position = (0, 0, 0)
        if glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS: self.agent.yaw, self.agent.pitch, self.agent.roll = (0, 0, 0)

        if glm.length(move_direction) > 0:
            # Use the .dt() proxy for framerate-independent movement
            self.agent.dt(self._delta_time).translate(glm.normalize(move_direction) * self.move_speed)

        # Handle rotation
        roll_input = 0.0
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS: roll_input += 1.0
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS: roll_input -= 1.0

        current_mouse_pos = glfw.get_cursor_pos(self.window)
        if self.last_mouse_pos is None:
            self.last_mouse_pos = current_mouse_pos

        dx = (current_mouse_pos[0] - self.last_mouse_pos[0])
        dy = (current_mouse_pos[1] - self.last_mouse_pos[1])
        self.last_mouse_pos = current_mouse_pos

        # per-second sensitivity
        yaw_delta = dx * self.mouse_sensitivity * -1
        pitch_delta = dy * self.mouse_sensitivity * self.mouse_y_dir

        if self.view_mode == ViewMode.third_person:
            self.orbit["azimuth"] += float(yaw_delta) * self._delta_time
            self.orbit["elevation"] += float(pitch_delta) * self._delta_time
            # clamp elevation to avoid flipping
            self.orbit["elevation"] = float(glm.clamp(self.orbit["elevation"],
                                                      self.orbit["min_elev"],
                                                      self.orbit["max_elev"]))
        else:
            # first-person control of the insect agent
            self.agent.dt(self._delta_time).rotate(
                yaw_delta=yaw_delta,
                pitch_delta=pitch_delta,
                roll_delta=roll_input * self.roll_speed,
                degrees=False
            )

    def draw(self):

        if not self._interactive_initialised:
            return

        if self.scene:
            linear_bg_color = tuple(pow(c, 2.2) for c in self.scene.background_color)
            glClearColor(linear_bg_color[0], linear_bg_color[1], linear_bg_color[2], 1.0)

        glViewport(0, 0, self._window_size[0], self._window_size[1])
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.view_mode == ViewMode.third_person:
            self.camera.ratio = self._window_size[0] / self._window_size[1]
            self._update_orbit_camera()
            pov = self.camera
        else:
            pov = self.agent

        self.renderer.draw(self.view_mode, pov)

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