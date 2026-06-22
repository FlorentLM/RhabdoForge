import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

import numpy as np
from pyglm import glm
import glfw
import time
from typing import TYPE_CHECKING, Optional, Tuple, Callable, Dict, List, Union
from collections import deque

from .scene import Scene
from .agent import Agent, OrbitCamera

from insectvision.utils.shared import DisplayMode
from insectvision.interactive.controls import Controls, ActionRegistry
from insectvision.interactive.hud import HUD
from insectvision.interactive.debug import DebugOverlay
from insectvision.interactive.dashboard import Dashboard
from insectvision.engine.resources import ShaderProgram

if TYPE_CHECKING:
    from insectvision.renderers.base import Renderer
    from insectvision.renderers.helpers import VisualOutput


class HDRTarget:
    """Offscreen linear-HDR render target: RGBA16F colour + depth24. Resizes lazily."""

    def __init__(self):
        self._fbo = self._color = self._depth = None
        self._size = (0, 0)

    def _ensure(self, w, h):

        if self._fbo is not None and self._size == (w, h):
            return

        self.free()
        self._size = (w, h)

        self._color = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._color)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, None)

        for p, v in ((GL_TEXTURE_MIN_FILTER, GL_LINEAR), (GL_TEXTURE_MAG_FILTER, GL_LINEAR),
                     (GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE), (GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)):
            glTexParameteri(GL_TEXTURE_2D, p, v)

        self._depth = glGenRenderbuffers(1)

        glBindRenderbuffer(GL_RENDERBUFFER, self._depth)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w, h)

        self._fbo = glGenFramebuffers(1)

        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._color, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self._depth)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("HDRTarget incomplete")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    @property
    def color(self):
        return self._color

    def bind(self, w, h):
        self._ensure(w, h)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo)
        glViewport(0, 0, w, h)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, *self._size)

    def blit_depth_to_default(self):
        # so 3D overlays (debug.draw) depth-test against the world
        w, h = self._size
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_DEPTH_BUFFER_BIT, GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def free(self):
        if self._color:
            glDeleteTextures([self._color])
            self._color = None
        if self._depth:
            glDeleteRenderbuffers(1, [self._depth])
            self._depth = None
        if self._fbo:
            glDeleteFramebuffers(1, [self._fbo])
            self._fbo = None
        self._size = (0, 0)


class Context:

    def __init__(self,
                 window_size: tuple = None,
                 debug_overlay: bool = True,
                 fps_limit: int = None,
                 vsync: bool = False,
                 controls: Optional['Controls'] = None
                 ):

        self._window_size = window_size if window_size is not None else (1280, 720)
        self._fps_limit = fps_limit if fps_limit is not None else 0
        self._interactive_initialised = False
        self._vsync = vsync

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
        glfw.swap_interval(int(self._vsync))
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_FRAMEBUFFER_SRGB)  # we want linear (non-gamma corrected)

        self._hdr = HDRTarget()
        self._tonemap_shader = ShaderProgram(vert_path='fullscreen.vert', frag_path='tonemap.frag')
        self._tonemap_vao = glGenVertexArrays(1)
        self.exposure = 1.0     # TODO: Wire this in

        self.agent: Optional['Agent'] = None
        self.renderer: Optional['Renderer'] = None
        self.scene: Optional['Scene'] = None
        self.observer: Optional['OrbitCamera'] = None
        self.display_mode: Optional['DisplayMode'] = None
        self.hud: Optional['HUD'] = None
        self.dashboard: Optional['Dashboard'] = None
        self.debug: Optional['DebugOverlay'] = DebugOverlay() if debug_overlay else None

        # Timing states
        self._last_wall_time: float = glfw.get_time()
        self._wall_dt: float = 0.0
        self._total_wall_time: float = 0.0
        self._dt: float = 1e-12  # effective delta for biology/physics
        self._total_time: float = 0.0  # accumulated simulation time
        self._frame_count: int = 0
        self._frame_times = deque(maxlen=60)  # for FPS tracking

        self.time_step: Optional[float] = None  # None = variable (wall-clock), Float = fixed time resolution

        # Shared movement parameter
        self.move_speed: float = 3.0
        self._mouse_captured: bool = True

        # Sun control mode
        self.sun_control_mode: bool = False

        # Key bindings: (glfw_key, glfw_action) -> [callbacks]
        self._key_bindings: Dict[Tuple[int, int], List[Callable]] = {}
        self._key_bindings_desc: Dict[int, str] = {}

        self.actions = ActionRegistry(self)
        self._controls: Optional[Controls] = controls

    def __repr__(self):
        mode = f"Fixed ({self.time_step * 1000:.1f}ms)" if self.time_step else "Variable (wall-clock)"
        return (f"<Context | Mode: {mode} | "
                f"Biol. sim time: {self.total_time:.3f}s | "
                f"Hardware: {self.fps:.1f} FPS>")

    def _tonemap_pass(self):

        shader = self._tonemap_shader

        with shader:
            glDisable(GL_DEPTH_TEST)
            glDepthMask(GL_FALSE)

            glActiveTexture(GL_TEXTURE0)

            glBindTexture(GL_TEXTURE_2D, self._hdr.color)
            glUniform1i(shader.get_loc('hdr_scene'), 0)
            glUniform1f(shader.get_loc('exposure'), float(self.exposure))
            glBindVertexArray(self._tonemap_vao)
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glBindVertexArray(0)

        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

    @property
    def mouse_captured(self) -> bool:
        return self._mouse_captured

    @mouse_captured.setter
    def mouse_captured(self, value: bool):
        self._mouse_captured = value
        if self.window:
            mode = glfw.CURSOR_DISABLED if value else glfw.CURSOR_NORMAL
            glfw.set_input_mode(self.window, glfw.CURSOR, mode)

    def toggle_mouse_capture(self):
        self.mouse_captured = not self.mouse_captured

    @property
    def controls(self) -> Optional[Controls]:
        return self._controls

    @controls.setter
    def controls(self, new_controls: Optional[Controls]):
        """Swap the active controller (safe to call at any time)."""
        if self._controls is not None:
            self._controls.free()

        self._controls = new_controls
        if self._controls is not None and self._interactive_initialised:
            self._controls.setup(self)

    # Timing

    @property
    def dt(self) -> float:
        """The delta-time to use for all biological and agent logic."""
        return self._dt

    @property
    def wall_dt(self) -> float:
        """Actual time passed on hardware (real world)."""
        return self._wall_dt

    @property
    def total_time(self) -> float:
        """Total biological/simulated time elapsed."""
        return self._total_time

    @property
    def wall_time(self) -> float:
        """Total real-world time elapsed since the context started ticking."""
        return self._total_wall_time

    @property
    def frame_count(self) -> int:
        """Total number of ticks/frames processed."""
        return self._frame_count

    def tick(self) -> float:
        """
        Advance both clocks by one step.
        - Wall clock: real elapsed time since previous tick, hardware-dependent
        - Sim clock: advances by 'time_step' if set, otherwise by the wall_dt
        """

        now = glfw.get_time()
        self._wall_dt = now - self._last_wall_time
        self._last_wall_time = now

        # Accumulate real world time
        self._total_wall_time += self._wall_dt
        self._frame_count += 1

        self._dt = self.time_step if self.time_step is not None else self._wall_dt

        # Accumulate simulated time
        self._total_time += self._dt
        self._frame_times.append(now)

        return self._wall_dt

    @property
    def fps(self) -> float:
        """Average hardware frames per second."""
        if len(self._frame_times) < 2:
            return 0.0
        return (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])

    # Custom key bindings

    @staticmethod
    def _resolve_key(key: Union[int, str]) -> int:

        if isinstance(key, int):
            return key

        attr = f"KEY_{key.upper()}"
        code = getattr(glfw, attr, None)

        if code is None:
            raise ValueError(
                f"Unknown key '{key}' (tried glfw.{attr}).  "
                f"Use a GLFW constant like glfw.KEY_M, or a string like 'm', "
                f"'space', 'left_shift', 'f1', 'kp_add', etc."
            )
        return code

    def bind_key(self, key: Union[int, str], callback: Callable, action: int = None, description: str = None):
        """
        Register a callback for a key event.

        Args:
            key: The key to bind (string or GLFW key constant)
            callback: A (no-argument) callable invoked when the key event triggers
            action: The key event type to bind (`glfw.PRESS` (default), `glfw.RELEASE`, or `glfw.REPEAT`)
            description: Optional, The name to display in the help HUD

        Example:
            context.bind_key('m', lambda: print("M pressed"))
            context.bind_key(glfw.KEY_SPACE, on_jump, action=glfw.RELEASE, description='Jump')
        """
        key_code = self._resolve_key(key)
        key_str = key if isinstance(key, str) else None

        if action is None:
            action = glfw.PRESS

        if description is None:
            description = callback.__name__.replace('_', ' ').title()

        display_name = key_str.upper() if key_str else f"Key {key_code}"
        self._key_bindings_desc[key_code] = (display_name, description)

        binding = (key, action)
        if binding not in self._key_bindings:
            self._key_bindings[binding] = []

        if callback not in self._key_bindings[binding]:
            self._key_bindings[binding].append(callback)

    def unbind_key(self, key: Union[int, str], callback: Callable = None, action: int = None):
        """
        Remove one or all callbacks for a key event.

        Args:
            key: The key to unbind (string or GLFW key constant)
            callback: The callable to remove, or None to fully unbind the action on that key
            action: The key event type to unbind (`glfw.PRESS` (default), `glfw.RELEASE`, or `glfw.REPEAT`)
        """
        key_code = self._resolve_key(key)

        if action is None:
            action = glfw.PRESS

        binding = (key_code, action)

        if binding not in self._key_bindings:
            return

        if callback is None:
            del self._key_bindings[binding]
        else:
            try:
                self._key_bindings[binding].remove(callback)
            except ValueError:
                pass
            if not self._key_bindings[binding]:
                del self._key_bindings[binding]

        # A key might be bound to multiple actions, only remove if unbound
        key_still_bound = False
        for (bound_key, bound_action) in self._key_bindings.keys():
            if bound_key == key_code:
                key_still_bound = True
                break

        if not key_still_bound and key_code in self._key_bindings_desc:
            del self._key_bindings_desc[key_code]

    @property
    def bound_keys(self) -> dict:
        return dict(self._key_bindings)

    # Actions

    def cycle_display_mode(self):
        next_mode = (self.display_mode.value + 1) % len(DisplayMode)
        self.display_mode = DisplayMode(next_mode)

    def toggle_tiled_mode(self):
        self.renderer.tiled_mode = not self.renderer.tiled_mode

    def toggle_projection_mode(self):
        from insectvision.utils.shared import OmmatidiaProjection

        self.renderer.projection_mode = (
            OmmatidiaProjection.Position
            if self.renderer.projection_mode == OmmatidiaProjection.OpticalAxis
            else OmmatidiaProjection.OpticalAxis
        )

    def toggle_hud(self):
        if self.hud:
            self.hud.show = not self.hud.show

    def toggle_overlay(self):
        self.renderer.overlay_enabled = not self.renderer.overlay_enabled

    def toggle_sun_control(self):
        self.sun_control_mode = not self.sun_control_mode
        mode_name = "Sun" if self.sun_control_mode else "View"
        print(f"Mouse control: {mode_name}")

    def toggle_time_dithering(self):
        self.renderer.time_dithering = not self.renderer.time_dithering

    def dither_once(self):
        self.renderer.dither()

    def increase_samples(self):
        self.renderer.nb_samples *= 2

    def decrease_samples(self):
        self.renderer.nb_samples = max(1, self.renderer.nb_samples // 2)

    def increase_pixel_samples(self):
        self.renderer.pixel_samples *= 2

    def decrease_pixel_samples(self):
        self.renderer.pixel_samples = max(1, self.renderer.pixel_samples // 2)

    def toggle_debug(self):
        if self.debug is not None:
            self.debug.enabled = not self.debug.enabled

    def toggle_microsaccades(self):
        self.renderer.microsaccades_enabled = not self.renderer.microsaccades_enabled

    def reset_position(self):
        self.agent.position = (0.0, 0.0, 0.0)

    def reset_rotation(self):
        self.agent.yaw, self.agent.pitch, self.agent.roll = (0.0, 0.0, 0.0)

    def pick_ommatidium(self, ndc_x: float, ndc_y: float) -> Optional[int]:
        """Calculates closest ommatidium based on active display projection."""
        if not self.renderer or not getattr(self.renderer, '_model', None):
            return None
        if self.display_mode not in (DisplayMode.Compound, DisplayMode.Third_person):
            return None

        model = self.renderer._model

        p_local = model.positions

        if self.display_mode == DisplayMode.Compound:
            aspect_ratio = self.window_size[0] / self.window_size[1]
            norms = np.linalg.norm(p_local, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            p_vec = p_local / norms

            longi = np.arctan2(p_vec[:, 0], -p_vec[:, 2])
            lati = np.arcsin(p_vec[:, 1])

            x = (longi / np.pi) / aspect_ratio
            y = lati / (np.pi / 2.0)

            dist_sq = (x - ndc_x) ** 2 + (y - ndc_y) ** 2
            best_idx = np.argmin(dist_sq)

            if dist_sq[best_idx] < 0.05:
                return int(best_idx)

        elif self.display_mode == DisplayMode.Third_person:
            eye_to_world = np.array(glm.inverse(self.agent.view))
            p_local_h = np.column_stack((p_local, np.ones(model.N)))
            p_world_h = p_local_h @ eye_to_world

            view_mat = np.array(self.observer.view)
            proj_mat = np.array(self.observer.projection)

            p_view_h = p_world_h @ view_mat
            p_clip_h = p_view_h @ proj_mat

            w = p_clip_h[:, 3]
            valid = w > 0.01

            if not np.any(valid):
                return None

            ndc = p_clip_h[valid, :2] / w[valid, np.newaxis]
            dist_sq = np.sum((ndc - [ndc_x, ndc_y]) ** 2, axis=1)

            best_valid_idx = np.argmin(dist_sq)
            if dist_sq[best_valid_idx] < 0.05:
                return int(np.nonzero(valid)[0][best_valid_idx])

        return None

    # Interactive loop

    def run_interactive(self, agent: 'Agent', scene: 'Scene', renderer: 'Renderer',
                        window_size=None, fps_limit=None, vsync=None, use_dashboard=False):
        """
        On first call, initialises and shows the window. Then checks if the interactive loop should continue.
        """

        if not self._interactive_initialised:

            if window_size is not None:
                self.window_size = window_size

            if fps_limit is not None:
                self.fps_limit = fps_limit

            if vsync is not None:
                self._vsync = bool(vsync)

            glfw.swap_interval(int(self._vsync))
            glfw.show_window(self.window)

            self.observer = OrbitCamera(target=agent, distance=1.5, ratio=self._window_size[0] / self._window_size[1])
            self.display_mode = DisplayMode.Compound
            self.hud = HUD(self, font_size=18)

            if use_dashboard:
                self.dashboard = Dashboard(self)
                self.hud.show = False  # default HUD to false dashboard is active

            # Default to kb + mouse
            if self._controls is None:
                from insectvision.interactive.keyboard import KeyboardMouse
                self._controls = KeyboardMouse()

            self._controls.setup(self)

            # Start the wall-clock anchor right before the first frame
            self._last_wall_time = glfw.get_time()
            self._interactive_initialised = True

        # Check if user wants to quit
        if glfw.window_should_close(self.window):
            return False

        # Advance the clocks
        self.tick()

        # Sync renderer/agent/scene state
        renderer.context = self
        renderer.runs_interactive = True

        if agent is not self.agent:
            self.agent = agent
            self.observer = OrbitCamera(target=agent, distance=1.5, ratio=self._window_size[0] / self._window_size[1])

        if scene is not self.scene:
            self.scene = scene

        return True

    def input(self):

        if not self._interactive_initialised:
            return

        glfw.poll_events()

        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)
            return

        if self._controls is not None:
            self._controls.poll(self)

    def draw(self, view_data: Optional['VisualOutput'] = None):

        if not self._interactive_initialised:
            return

        # World / scene radiance -> HDR target (linear)
        self._hdr.bind(self._window_size[0], self._window_size[1])
        if self.scene:
            linear_bg = tuple(pow(c, 2.2) for c in self.scene.background_color)
            glClearColor(linear_bg[0], linear_bg[1], linear_bg[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.display_mode == DisplayMode.Third_person:
            self.observer.ratio = self._window_size[0] / self._window_size[1]
            self.observer.update()
            pov = self.observer
        else:
            pov = self.agent

        self.renderer.draw(self.display_mode, pov)

        # Tonemap HDR -> default FB (hardware sRGB-encodes), carries depth for overlays
        self._hdr.unbind()
        self._hdr.blit_depth_to_default()
        self._tonemap_pass()

        # Overlays in display space (not tonemapped)
        if self.debug is not None and self.display_mode != DisplayMode.Panoramic:
            self.debug.draw(view=pov.view, proj=pov.projection)
        if self.hud:
            self.hud.draw()

        glfw.swap_buffers(self.window)

        if self.dashboard:
            if not self.dashboard.render(view_data):
                self.dashboard.free()
                self.dashboard = None

        # Update FPS throttling check
        if self._fps_limit > 0:
            now = glfw.get_time()
            elapsed = now - self._last_wall_time  # last_wall_time is set in tick()
            wait_time = (1.0 / self._fps_limit) - elapsed

            if wait_time > 0:
                time.sleep(wait_time)

    def free(self):
        if self._controls:
            self._controls.free()
        if self.debug:
            self.debug.free()
        if self.hud:
            self.hud.free()
        if self.dashboard:
            self.dashboard.free()
        if self.renderer:
            self.renderer.free()
        if self.scene:
            self.scene.free()
        glfw.terminate()

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
    def vsync(self) -> bool:
        return self._vsync

    @vsync.setter
    def vsync(self, value: bool):
        self._vsync = bool(value)