import numpy as np
from graphics.glm import rotation_mat, perspective_mat, translation_mat
from graphics.utils import VEC_DTYPE, WORLD_RIGHT, WORLD_UP, WORLD_FORWARD

RIGHT_HOMOGENEOUS = np.array([*WORLD_RIGHT, 0.0], dtype=VEC_DTYPE)
UP_HOMOGENEOUS = np.array([*WORLD_UP, 0.0], dtype=VEC_DTYPE)
FORWARD_HOMOGENEOUS = np.array([*WORLD_FORWARD, 0.0], dtype=VEC_DTYPE)


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

        self._position = np.asarray(position, dtype=VEC_DTYPE)
        self.tilt = self.pitch = self.vertical_angle = pitch
        self.pan = self.yaw = self.horizontal_angle = yaw
        self.roll = self.forward_angle = forward_roll

        self._aspect_ratio = ratio
        self._fov = fov
        self._near = near
        self._far = far

        self._ident = np.eye(4, dtype=VEC_DTYPE)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = np.asarray(value, dtype=VEC_DTYPE)[:3]

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, val):
        self._fov = np.clip(val, 0.001, 180.0, dtype=VEC_DTYPE)

    @property
    def near(self):
        return self._near

    @near.setter
    def near(self, val):
        val = np.maximum(0.001, val, dtype=VEC_DTYPE)
        self._near = val

    @property
    def far(self):
        return self._far

    @far.setter
    def far(self, val):
        val = np.maximum(self._near, val, dtype=VEC_DTYPE)
        self._far = val

    @property
    def ratio(self):
        return self._aspect_ratio

    @ratio.setter
    def ratio(self, val):
        val = np.maximum(0.001, val, dtype=VEC_DTYPE)
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
        yaw_mat = rotation_mat(np.deg2rad(self.yaw, dtype=VEC_DTYPE), WORLD_UP)
        pitch_mat = rotation_mat(np.deg2rad(self.pitch, dtype=VEC_DTYPE), WORLD_RIGHT)
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
        return perspective_mat(np.deg2rad(self.fov, dtype=VEC_DTYPE), self._aspect_ratio, self.near, self.far)

    @property
    def view(self):
        # The view matrix is the inverse of the camera's world transform
        # Inverse of (T * R) is (R_inv * T_inv), which is (R_transpose * T_negative)
        return self.orientation.T @ translation_mat(-self.pos)

    @property
    def matrix(self):
        return self.view @ self.projection

    def lookat(self, target_pos):
        """
        Orients the camera to look at a specific target position
        """

        target_pos = np.asarray(target_pos, dtype=VEC_DTYPE)

        # Avoid division by zero if the target is at the camera's position
        if np.allclose(target_pos, self.position):
            return

        # direction from the camera to the target
        direction = target_pos - self.position
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            return
        direction /= direction_norm

        # Calculate pitch (up/down look) from the Y component
        # Clamp the argument to asin (avoids domain errors from floating point inaccuracies)
        self.pitch = np.rad2deg(np.arcsin(np.clip(direction[1], -1.0, 1.0)), dtype=VEC_DTYPE)

        # Calculate yaw (left/right look) from the X and Z components
        self.yaw = np.rad2deg(np.arctan2(direction[0], -direction[2]), dtype=VEC_DTYPE)

