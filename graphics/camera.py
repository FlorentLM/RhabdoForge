import numpy as np
from graphics.glm import rotation_mat, perspective_mat, translation_mat
from graphics.utils import DTYPE, WORLD_RIGHT, WORLD_UP, WORLD_FORWARD

RIGHT_HOMOGENEOUS = np.array([*WORLD_RIGHT, 0.0], dtype=DTYPE)
UP_HOMOGENEOUS = np.array([*WORLD_UP, 0.0], dtype=DTYPE)
FORWARD_HOMOGENEOUS = np.array([*WORLD_FORWARD, 0.0], dtype=DTYPE)


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

        self._position = np.asarray(position, dtype=DTYPE)
        self.tilt = self.pitch = self.vertical_angle = pitch
        self.pan = self.yaw = self.horizontal_angle = yaw
        self.roll = self.forward_angle = forward_roll

        self._aspect_ratio = ratio
        self._fov = fov
        self._near = near
        self._far = far

        self._ident = np.eye(4, dtype=DTYPE)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = np.asarray(value, dtype=DTYPE)[:3]

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, val):
        self._fov = np.clip(val, 0.001, 180.0, dtype=DTYPE)

    @property
    def near(self):
        return self._near

    @near.setter
    def near(self, val):
        val = np.maximum(0.001, val, dtype=DTYPE)
        self._near = val

    @property
    def far(self):
        return self._far

    @far.setter
    def far(self, val):
        val = np.maximum(self._near, val, dtype=DTYPE)
        self._far = val

    @property
    def ratio(self):
        return self._aspect_ratio

    @ratio.setter
    def ratio(self, val):
        val = np.maximum(0.001, val, dtype=DTYPE)
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
        """ The camera's orientation matrix (rotates from local to world space) """
        # column-major, so post-multiply: Parent_Transform * Local_Transform
        # Yaw (around world up) is the parent, pitch is the local rotation
        yaw_mat = rotation_mat(np.deg2rad(self.yaw, dtype=DTYPE), WORLD_UP)
        pitch_mat = rotation_mat(np.deg2rad(self.pitch, dtype=DTYPE), WORLD_RIGHT)
        return yaw_mat @ pitch_mat

    @property
    def forward(self):
        # Transform the world's forward vector into the camera's local space
        return (self.orientation @ FORWARD_HOMOGENEOUS)[:3]

    @property
    def backward(self):
        return -self.forward

    @property
    def right(self):
        # Transform the world's right vector into the camera's local space
        return (self.orientation @ RIGHT_HOMOGENEOUS)[:3]

    @property
    def left(self):
        return -self.right

    @property
    def up(self):
        # The camera's 'up' is the cross product of its right and forward vectors
        return np.cross(self.right, self.forward)

    @property
    def down(self):
        return -self.up

    @property
    def projection(self):
        return perspective_mat(np.deg2rad(self.fov, dtype=DTYPE), self._aspect_ratio, self.near, self.far)

    @property
    def view(self):
        # The view matrix is the inverse of the camera's world transform
        # Inverse of (T * R) is (R_inv * T_inv), which is (R_transpose * T_negative)
        return self.orientation.T @ translation_mat(-self.pos)

    @property
    def matrix(self):
        return self.view @ self.projection

    def lookat(self, *target_pos):
        # TODO - move this in a static function in glm?
        target_pos = np.asarray(target_pos, dtype=DTYPE)

        if np.allclose(target_pos, self.position):
            return

        direction = target_pos - self.position
        direction /= np.linalg.norm(direction)

        # Calculate pitch (up/down look) from the Y component
        self.pitch = np.rad2deg(np.arcsin(-direction[1]), dtype=DTYPE)
        # Calculate yaw (left/right look) from the X and Z components
        self.yaw = np.rad2deg(np.arctan2(direction[0], -direction[2]), dtype=DTYPE)

