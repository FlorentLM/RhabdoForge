from typing import Sequence
from numpy.typing import ArrayLike
import numpy as np
from pyglm import glm

from graphics.utils import WORLD_RIGHT, WORLD_UP, WORLD_FORWARD

RIGHT_VEC4 = glm.vec4(WORLD_RIGHT, 0.0)
UP_VEC4 = glm.vec4(WORLD_UP, 0.0)
FORWARD_VEC4 = glm.vec4(WORLD_FORWARD, 0.0)


class Agent:
    """ Represents an agent in the 3D world """

    def __init__(self,
                 position: Sequence[float | int] = (0.0, 0.0, 0.0),
                 yaw: float = 0.0,
                 pitch: float = 0.0,
                 roll: float = 0.0,
                 fov: float = 50.0,
                 near: float = 0.1,
                 far: float = 100.0,
                 ratio: float = 16.0 / 9.0,
                 move_sensitivity: float = 0.01,
                 mouse_sensitivity: float = 0.25):

        self._position = glm.vec3(position)
        self.tilt = self.pitch = self.vertical_angle = pitch
        self.pan = self.yaw = self.horizontal_angle = yaw
        self.roll = self.forward_angle = roll

        self._aspect_ratio = ratio
        self._fov = fov
        self._near = near
        self._far = far

        self.move_sensitivity = move_sensitivity
        self.mouse_sensitivity = mouse_sensitivity

        self._ident = glm.mat4(1.0)

    def move(self, direction: glm.vec3 | ArrayLike):
        """ Moves the agent in a specified direction vector """

        direction = glm.vec3(direction)

        if glm.length(direction) > 0:
            displacement = glm.normalize(direction) * self.move_sensitivity
            self._position += displacement

        return self

    def rotate(self, yaw_delta: float, pitch_delta: float):
        """ Rotates the agent's view """
        self.yaw -= yaw_delta * self.mouse_sensitivity
        self.pitch -= pitch_delta * self.mouse_sensitivity
        self.pitch = np.clip(self.pitch, -89.999, 89.999)
        return self

    def translate(self, translation: glm.vec3 | ArrayLike):
        """ Translates the agent by a given vector """
        self._position += glm.vec3(translation)
        return self

    def lookat(self, target_pos: glm.vec3 | ArrayLike):
        """ Orients the agent to look at a specific target position """

        target_pos = glm.vec3(target_pos)

        # Avoid issues if the target is at the agent's position
        if glm.distance(target_pos, self.position) < 1e-6:
            return

        # direction from the agent to the target
        direction = glm.normalize(target_pos - self.position)

        # Calculate pitch (up/down look) from the Y component
        # Clamp the argument to asin (avoids domain errors from floating point inaccuracies)
        self.pitch = glm.degrees(glm.asin(glm.clamp(direction.y, -1.0, 1.0)))

        # Calculate yaw (left/right look) from the X and Z components
        # (negative Z component for a right-handed coordinate system)
        self.yaw = glm.degrees(glm.atan2(direction.x, -direction.z))

        return self

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = glm.vec3(value)

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, val):
        self._fov = np.clip(val, 0.001, 180.0)

    @property
    def near(self):
        return self._near

    @near.setter
    def near(self, val):
        self._near = max(0.001, val)

    @property
    def far(self):
        return self._far

    @far.setter
    def far(self, val):
        self._far = max(self._near, val)

    @property
    def ratio(self):
        return self._aspect_ratio

    @ratio.setter
    def ratio(self, val):
        self._aspect_ratio = max(0.001, val)

    pos = position
    far_plane = far
    near_plane = near
    zoom = field_of_view = fov
    aspect_ratio = ratio

    @property
    def identity(self):
        return self._ident

    @property
    def orientation(self):
        """ The agent's orientation matrix (rotates from local to world space) """
        orientation = glm.mat4(1.0)
        orientation = glm.rotate(orientation, glm.radians(self.yaw), WORLD_UP)
        orientation = glm.rotate(orientation, glm.radians(self.pitch), WORLD_RIGHT)
        return orientation

    @property
    def forward(self):
        # Transform the world's forward vector by the orientation
        # (negative sign to look 'forward' into a right-handed system)
        return glm.normalize((self.orientation * glm.vec4(0, 0, -1, 0)).xyz)

    @property
    def backward(self):
        return -self.forward

    @property
    def right(self):
        # Transform the world's right vector by the orientation
        return glm.normalize((self.orientation * glm.vec4(1, 0, 0, 0)).xyz)

    @property
    def left(self):
        return -self.right

    @property
    def up(self):
        # The agent's 'up' is the cross product of its right and forward vectors
        return glm.cross(self.right, self.forward)

    @property
    def down(self):
        return -self.up

    @property
    def projection(self):
        return glm.perspective(glm.radians(self.fov), self._aspect_ratio, self.near, self.far)

    @property
    def view(self):
        target = self.position + self.forward
        return glm.lookAt(self.position, target, self.up)

    @property
    def matrix(self):
        # For column-major matrices (like glm), the order is P * V
        return self.projection * self.view