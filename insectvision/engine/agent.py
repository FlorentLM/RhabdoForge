import OpenGL
OpenGL.ERROR_CHECKING = False

from typing import Sequence
import numpy as np
from pyglm import glm

from .utils import WORLD_RIGHT, WORLD_UP, WORLD_FORWARD, DeltaTimeTransformer
from .movement import TransformMixin


RIGHT_VEC4 = glm.vec4(WORLD_RIGHT, 0.0)

UP_VEC4 = glm.vec4(WORLD_UP, 0.0)

FORWARD_VEC4 = glm.vec4(WORLD_FORWARD, 0.0)


class Agent(TransformMixin):
    """Represents an agent in the 3D world."""

    def __init__(self,
                 position: Sequence[float | int] = (0.0, 0.0, 0.0),
                 yaw: float = 0.0,
                 pitch: float = 0.0,
                 roll: float = 0.0,
                 fov: float = 50.0,
                 near: float = 0.001,
                 far: float = 100.0,
                 ratio: float = 16.0 / 9.0,
                 degrees: bool = True):

        if not degrees:
            yaw = glm.degrees(yaw)
            pitch = glm.degrees(pitch)
            roll = glm.degrees(roll)
            fov = glm.degrees(fov)

        self._position = glm.vec3(position)
        self.tilt = self.pitch = self.vertical_angle = pitch
        self.pan = self.yaw = self.horizontal_angle = yaw
        self.roll = self.forward_angle = roll

        self._aspect_ratio = ratio
        self._fov = fov
        self._near = near
        self._far = far

        self._ident = glm.mat4(1.0)

    def dt(self, delta_time: float) -> DeltaTimeTransformer:
        """
        Enables framerate-independent transformations for a chain of method calls.

        Example:
            # Rotates at 90 degrees per second
            my_instance.dt(delta_time).rotate_axis(90, 'y')
        """
        return DeltaTimeTransformer(self, delta_time)


    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, fov_degrees):
        self._fov = np.clip(fov_degrees, 0.001, 180.0)

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

    far_plane = far
    near_plane = near
    field_of_view = fov
    aspect_ratio = ratio

    @property
    def identity(self):
        return self._ident

    @property
    def orientation(self):
        """The agent's orientation matrix (rotates from local to world space)."""

        identity = glm.mat4(1.0)

        yaw = glm.rotate(identity, glm.radians(self.yaw), WORLD_UP)
        pitch = glm.rotate(identity, glm.radians(self.pitch), WORLD_RIGHT)
        roll = glm.rotate(identity, glm.radians(self.roll), WORLD_FORWARD)

        return yaw * pitch * roll

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


class OrbitCamera:
    """
    Wrapper camera that orbits around a target position.
    """

    def __init__(self,
                 target: Agent,
                 distance: float = 1.5,
                 azimuth: float = 0.0,
                 elevation: float = 20.0,
                 fov: float = 60.0,
                 ratio: float = 16.0 / 9.0,
                 near: float = 0.001,
                 far: float = 100.0,
                 degrees: bool = True):

        self.target = target
        self.distance = distance
        self.azimuth = azimuth if degrees else glm.degrees(azimuth)
        self.elevation = elevation if degrees else glm.degrees(elevation)

        # Reasonable limits for camera position
        self.min_elevation = -89.999
        self.max_elevation = 89.999
        self.min_distance = 0.01

        self._observer = Agent(fov=fov, ratio=ratio, near=near, far=far)
        self.update()

    def pan(self, azimuth_delta: float, elevation_delta: float, degrees: bool = True):
        """Pans the camera by changing azimuth and elevation."""

        if not degrees:
            azimuth_delta = glm.degrees(azimuth_delta)
            elevation_delta = glm.degrees(elevation_delta)

        self.azimuth += azimuth_delta
        self.elevation = glm.clamp(self.elevation + elevation_delta, self.min_elevation, self.max_elevation)
        self.update()

    def zoom(self, factor: float):
        """Zooms the camera by adjusting its distance to the target."""
        self.distance = max(self.min_distance, self.distance * factor)
        self.update()

    def update(self):
        """Recalculates the observer's position and orientation based on orbit parameters."""

        az_rad = glm.radians(self.azimuth)
        el_rad = glm.radians(self.elevation)

        # Calculate offset from target using spherical coordinates
        offset = glm.vec3(self.distance * glm.cos(el_rad) * glm.sin(az_rad),
                          self.distance * glm.sin(el_rad),
                          self.distance * glm.cos(el_rad) * glm.cos(az_rad))

        self._observer.position = self.target.position + offset
        self._observer.lookat(self.target.position)

    @property
    def view(self):
        # This prevents the camera from rolling on its own as it orbits the target
        return glm.lookAt(self._observer.position, self.target.position, WORLD_UP)

    @property
    def projection(self):
        return self._observer.projection

    @property
    def position(self):
        return self._observer.position

    @property
    def ratio(self):
        return self._observer.ratio

    @ratio.setter
    def ratio(self, value):
        self._observer.ratio = value