import numpy as np
from pyglm import glm
from graphics.utils import WORLD_RIGHT, WORLD_UP, WORLD_FORWARD

RIGHT_HOMOGENEOUS = glm.vec4(WORLD_RIGHT, 0.0)
UP_HOMOGENEOUS = glm.vec4(WORLD_UP, 0.0)
FORWARD_HOMOGENEOUS = glm.vec4(WORLD_FORWARD, 0.0)


class Camera:

    def __init__(self,
                 position=(0.0, 0.0, 0.0),
                 yaw=0.0,
                 pitch=0.0,
                 forward_roll=0.0,
                 fov=50.0,
                 near=0.1,
                 far=100.0,
                 ratio=4.0/3.0):

        self._position = glm.vec3(position)
        self.tilt = self.pitch = self.vertical_angle = pitch
        self.pan = self.yaw = self.horizontal_angle = yaw
        self.roll = self.forward_angle = forward_roll

        self._aspect_ratio = ratio
        self._fov = fov
        self._near = near
        self._far = far

        self._ident = glm.mat4(1.0)

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
        """ The camera's orientation matrix (rotates from local to world space) """
        orientation = glm.mat4(1.0)
        orientation = glm.rotate(orientation, glm.radians(self.yaw), WORLD_UP)
        orientation = glm.rotate(orientation, glm.radians(self.pitch), WORLD_RIGHT)
        return orientation

    @property
    def forward(self):
        # Transform the world's forward vector by the orientation
        # (negative sign to look "forward" into a right-handed system)
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
        # The camera's 'up' is the cross product of its right and forward vectors
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
        # For column-major matrices (like pyglm), the order is P * V
        return self.projection * self.view

    def lookat(self, target_pos):
        """
        Orients the camera to look at a specific target position
        """
        target_pos = glm.vec3(target_pos)

        # Avoid issues if the target is at the camera's position
        if glm.distance(target_pos, self.position) < 1e-6:
            return

        # direction from the camera to the target
        direction = glm.normalize(target_pos - self.position)

        # Calculate pitch (up/down look) from the Y component
        # Clamp the argument to asin (avoids domain errors from floating point inaccuracies)
        self.pitch = glm.degrees(glm.asin(glm.clamp(direction.y, -1.0, 1.0)))

        # Calculate yaw (left/right look) from the X and Z components
        # (negative Z component for a right-handed coordinate system)
        self.yaw = glm.degrees(glm.atan2(direction.x, -direction.z))