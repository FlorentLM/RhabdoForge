import numpy as np

from engine import WORLD_RIGHT, WORLD_UP
from glm import Matrices, rotate, translate



class Camera:

    def __init__(self,
                 position=(0.0, 0.0, 0.0),
                 h_angle=0.0,
                 v_angle=0.0,
                 forward_roll=0.0,
                 fov=50.0,
                 near=0.1,
                 far=100.0,
                 ratio=4.0/3.0):

        self._position = np.asarray(position).astype(np.float32)
        self.tilt = self.pitch = self.vertical_angle = v_angle
        self.pan = self.yaw = self.horizontal_angle = h_angle
        self.roll = self.forward_angle = forward_roll

        self._aspect_ratio = ratio
        self._fov = fov
        self._near = near
        self._far = far

        self._ident = np.eye(4, dtype=np.float32)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, *args):
        self._position = np.asarray(*args).astype(np.float32)[:3]

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
        val = 0.001 if val <= 0.0 else val
        self._near = val

    @property
    def far(self):
        return self._far

    @far.setter
    def far(self, val):
        val = self._near if val < self._near else val
        self._far = val

    @property
    def ratio(self):
        return self._aspect_ratio

    @ratio.setter
    def ratio(self, val):
        val = 0.001 if val <= 0.0 else val
        self._aspect_ratio = val

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
        orientation = self.identity
        orientation = rotate(orientation, np.deg2rad(self.pitch), WORLD_RIGHT)
        orientation = rotate(orientation, np.deg2rad(self.yaw), WORLD_UP)
        return orientation

    @property
    def forward(self):
        return (np.array([0.0, 0.0, -1.0, 1.0], dtype=np.float32) @ np.linalg.inv(self.orientation))[:3]

    @property
    def backward(self):
        return (np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32) @ np.linalg.inv(self.orientation))[:3]

    @property
    def right(self):
        return (np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32) @ np.linalg.inv(self.orientation))[:3]

    @property
    def left(self):
        return (np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32) @ np.linalg.inv(self.orientation))[:3]

    @property
    def up(self):
        return (np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32) @ np.linalg.inv(self.orientation))[:3]

    @property
    def down(self):
        return (np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32) @ np.linalg.inv(self.orientation))[:3]

    @property
    def projection(self):
        return Matrices.perspective(np.deg2rad(self.fov), self._aspect_ratio, self.near, self.far)

    @property
    def view(self):
        return translate(self.identity, -self.pos) @ self.orientation

    @property
    def matrix(self):
        return self.view @ self.projection

    def lookat(self, *target_pos):
        # TODO - move this in a static function in glm?
        target_pos = np.asarray(*target_pos).astype(np.float32)[:3]

        if np.allclose(target_pos, self.pos):
            return

        direction = target_pos - self.pos
        direction /= np.linalg.norm(direction)

        self.pitch = np.rad2deg(np.arcsin(-direction[1]))
        self.yaw = np.rad2deg(- np.arctan2(-direction[0], -direction[2]))

