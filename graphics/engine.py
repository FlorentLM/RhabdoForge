import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import OpenGL
OpenGL.ERROR_CHECKING = False

import numpy as np
import pygame
from pygame.locals import *
from pyglm import glm
from OpenGL.GL import *

from graphics.camera import Camera
from graphics.utils import WORLD_UP, WORLD_DOWN
from graphics.renderers.panoramic import PanoramicEye


class Engine:
    def __init__(self, width=800, height=600, headless=False):
        self.width = width
        self.height = height
        self.headless = headless

        # References set by the main script
        self.compound_eye = None

        # Pygame OpenGL context
        # TODO: maybe something lighter than pygame? glfw?
        pygame.init()
        flags = DOUBLEBUF | OPENGL
        if self.headless:
            flags |= pygame.HIDDEN
        pygame.display.set_mode((self.width, self.height), flags)

        # OpenGL state
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glClearColor(0.1, 0.1, 0.1, 1.0)

        # Core stuff
        self.camera = Camera(position=(0, 0, 4), ratio=width / height)

        self.pano_debug_view = PanoramicEye()

        # Stuff for interactive mode
        self.clock = pygame.time.Clock()
        # self.hud = HUD(self)

        self.move_sensitivity = 0.01  # units per frame
        self.mouse_sensitivity = 0.25

    # TODO: move the counting logic to the scene bakers

    # def _update_geometry_counts(self):
    #     """ Recalculates total primitives from all geometry in the scene """
    #
    #     self._total_scene_vertices = 0
    #     self._total_scene_triangles = 0
    #
    #     # Count vertices and triangles from standard mesh instances
    #     mesh_vert_count = 0
    #     for instance in self.scene.instances:
    #         mesh_vert_count += instance.asset.draw_count
    #
    #     self._total_scene_vertices += mesh_vert_count
    #     self._total_scene_triangles = mesh_vert_count // 3
    #
    #     # If a point cloud exists add its points to the vertex count
    #     # (for display purposes, treats "points" as a type of "vertex")
    #     if self.scene.point_cloud:
    #         self._total_scene_vertices += self.scene.point_cloud.num_points

    def update_movement(self):
        """ Processes continuous input (keyboard/mouse) to move the camera """

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        cam_displacement = glm.vec3(0.0)

        if keys[K_w]: cam_displacement += self.camera.forward
        if keys[K_s]: cam_displacement += self.camera.backward
        if keys[K_a]: cam_displacement += self.camera.left
        if keys[K_d]: cam_displacement += self.camera.right
        if keys[K_SPACE]: cam_displacement += WORLD_UP
        if keys[K_LSHIFT]: cam_displacement += WORLD_DOWN

        # Normalize (prevents faster diagonal movement) and apply speed
        if glm.length(cam_displacement) > 0:
            cam_displacement = glm.normalize(cam_displacement) * self.move_sensitivity
            self.camera.pos += cam_displacement

        # Handle mouse look
        mouse_x, mouse_y = pygame.mouse.get_rel()
        if mouse_x != 0 or mouse_y != 0:
            self.camera.yaw -= mouse_x * self.mouse_sensitivity
            self.camera.pitch -= mouse_y * self.mouse_sensitivity
            self.camera.pitch = np.clip(self.camera.pitch, -89.0, 89.0)

    def close(self):
        """ Frees all allocated resources """
        pygame.quit()