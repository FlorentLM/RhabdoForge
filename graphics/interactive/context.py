import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame
from pygame.locals import *
from pyglm import glm
import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from graphics.agent import Agent
from graphics.utils import WORLD_UP, WORLD_DOWN


class Context:
    """
    Minimal OpenGL context provider for headless mode, and viewer for interactive mode
    """

    def __init__(self, window_size: tuple, headless: bool = False):

        self.window_size = window_size
        self.headless = headless

        # Initialize Pygame and create the OpenGL context

        pygame.init()
        flags = DOUBLEBUF | OPENGL
        if self.headless:
            flags |= pygame.HIDDEN
        pygame.display.set_mode(self.window_size, flags)

        # Interactive-mode specific setup

        if not self.headless:
            self._interactive_initialised = False
            self.agent = None
            self.renderer = None
            self.debug_renderer = None
            self.view_modes = None
            self.current_view_idx = 0
            self.voronoi_view = False

        self._running = True

    def handle_input(self):
        """
        Processes all user input events and updates the agent's state
        """

        if self.headless:
            # no input to handle in headless mode
            return

        # Process discrete events (single key presses, quit)
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                self._running = False
                return
            if event.type == KEYDOWN:
                if event.key == K_c:
                    self.current_view_idx = (self.current_view_idx + 1) % len(self.view_modes)
                if event.key == K_v:
                    self.voronoi_view = not self.voronoi_view

        # Process continuous input
        pressed_keys = pygame.key.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_rel()

        move_direction = glm.vec3(0.0)
        if pressed_keys[K_w]: move_direction += self.agent.camera.forward
        if pressed_keys[K_s]: move_direction += self.agent.camera.backward
        if pressed_keys[K_a]: move_direction += self.agent.camera.left
        if pressed_keys[K_d]: move_direction += self.agent.camera.right
        if pressed_keys[K_SPACE]: move_direction += WORLD_UP
        if pressed_keys[K_LSHIFT]: move_direction += WORLD_DOWN

        self.agent.move(move_direction)
        if mouse_x != 0 or mouse_y != 0:
            self.agent.rotate(mouse_x, mouse_y)

    def interactive(self, agent: Agent, renderer, debug_renderer=None):

        if not self._interactive_initialised:
            self.agent = agent
            self.renderer = renderer
            self.debug_renderer = debug_renderer

            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)

            self.view_modes = ['compound_eye', 'panoramic', 'standard_3d']
            self.current_view_idx = 0
            self.voronoi_view = False

            self._interactive_initialised = True

        return self._running

    def draw(self):
        """ Draws the current view to the active OpenGL context """

        if self.headless:
            return

        w, h = self.window_size
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        renderer_to_use = self.renderer
        if self.debug_renderer and self.view_mode != 'compound_eye':
            renderer_to_use = self.debug_renderer

        renderer_to_use.draw(self.view_mode, self.agent.camera, self.voronoi_view)

        pygame.display.flip()

    def free(self):
        """ Destroys the window and quits Pygame """
        pygame.quit()

    @property
    def view_mode(self):
        return self.view_modes[self.current_view_idx]