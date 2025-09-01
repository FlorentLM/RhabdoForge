from typing import Sequence, Union
from numpy.typing import ArrayLike
import numpy as np
from pyglm import glm

from graphics.utils import WORLD_RIGHT, WORLD_UP, WORLD_FORWARD, DeltaTimeTransformer

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
        Enables framerate-independent transformations for a chain of method calls

        Example:
            # Rotates at 90 degrees per second
            my_instance.dt(delta_time).rotate_axis(90, 'y')
        """
        return DeltaTimeTransformer(self, delta_time)

    def rotate_axis(self, angle: float, axis: Union[str, glm.vec3, ArrayLike], degrees: bool = True):
        """ Rotates the agent around a given axis """

        if isinstance(axis, str):
            axis_str = axis.lower()
            # Yaw is special: it's almost always around the world's UP vector for intuitive control
            # Pitch and Roll are relative to the agent's current orientation
            axis_map = {
                'x': self.right,
                'y': self.up,
                'z': self.forward,
                'right': self.right,
                'left': self.left,
                'up': self.up,
                'down': self.down,
                'forward': self.forward,
                'backward': self.backward,
                'yaw': WORLD_UP,
                'pitch': self.right,
                'roll': self.forward,
            }
            try:
                rotation_axis = axis_map[axis_str]
            except KeyError:
                raise ValueError(f"Unknown axis identifier: '{axis}'. Valid options are: {list(axis_map.keys())}")
        else:
            rotation_axis = glm.vec3(axis)
        
        angle_rad = glm.radians(angle) if degrees else angle
        new_orientation = glm.rotate(self.orientation, angle_rad, rotation_axis)

        # Decompose the new orientation matrix back into yaw, pitch, and roll
        # The decomposition logic depends on the rotation order (Yaw, Pitch, Roll in our case)

        sin_pitch = -new_orientation[2][1]
        # Clamp the value to avoid domain errors with asin
        if sin_pitch >= 1.0:
            self.pitch = 90.0
        elif sin_pitch <= -1.0:
            self.pitch = -90.0
        else:
            self.pitch = glm.degrees(glm.asin(sin_pitch))

        cos_pitch = glm.cos(glm.radians(self.pitch))

        # Avoid division by zero if pitch is +/- 90 degrees (gimbal lock)
        if abs(cos_pitch) > 1e-6:
            # Yaw
            sin_yaw = new_orientation[2][0] / cos_pitch
            cos_yaw = new_orientation[2][2] / cos_pitch
            self.yaw = glm.degrees(glm.atan2(sin_yaw, cos_yaw))
            # Roll
            sin_roll = new_orientation[0][1] / cos_pitch
            cos_roll = new_orientation[1][1] / cos_pitch
            self.roll = glm.degrees(glm.atan2(sin_roll, cos_roll))
        else:
            # Gimbal Lock: The roll and yaw axes are aligned
            # We can set roll to 0 and calculate yaw
            self.roll = 0.0
            sin_yaw = -new_orientation[0][2]
            cos_yaw = new_orientation[0][0]
            self.yaw = glm.degrees(glm.atan2(sin_yaw, cos_yaw))

        return self

    def rotate(self, yaw_delta: float = 0.0, pitch_delta: float = 0.0, roll_delta: float = 0.0, degrees: bool = True):
        self.yaw += yaw_delta if degrees else glm.degrees(yaw_delta)
        self.pitch += pitch_delta if degrees else glm.degrees(pitch_delta)
        self.roll += roll_delta if degrees else glm.degrees(roll_delta)
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
        # yaw
        orientation = glm.rotate(orientation, glm.radians(self.yaw), WORLD_UP)
        # pitch
        orientation = glm.rotate(orientation, glm.radians(self.pitch), WORLD_RIGHT)
        # Roll
        orientation = glm.rotate(orientation, glm.radians(self.roll), glm.vec3(0, 0, -1))
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