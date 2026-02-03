from typing import Union, Sequence
from numpy.typing import ArrayLike
import numpy as np
from pyglm import glm

from graphics.utils import WORLD_UP, WORLD_RIGHT, WORLD_FORWARD, DeltaTimeTransformer


class _SunDeltaTimeTransformer(DeltaTimeTransformer):

    def __init__(self, target: 'Sun', delta_time: float):
        super().__init__(target, delta_time)

    def orbit(self, angle: float, axis: Union[str, glm.vec3, ArrayLike] = 'y', degrees: bool = True):
        scaled_angle = angle * self._delta_time
        self._target.orbit(scaled_angle, axis, degrees=degrees)
        return self


class Sun:
    """
    Represents the Sun in the scene.
    """
    # TODO: Have this inherit from a Light class

    def __init__(self,
                 position: Sequence[float] = (50.0, 100.0, 30.0),
                 intensity: float = 2.0,
                 angular_radius: float = 0.02,
                 color: Sequence[float] = (1.0, 1.0, 1.0)):
        """
        Args:
            position: World-space position of the sun. Direction toward origin is computed from this
            intensity: Brightness multiplier
            angular_radius: Apparent angular size of the sun disk (in radians) (affects soft shadows)
            color: RGB colour of the sunlight (normalised 0-1)
        """
        self._position = glm.vec3(position)
        self._intensity = intensity
        self._angular_radius = angular_radius
        self._color = glm.vec3(color)

        # Target point the sun orbits around (default: origin)
        self._target = glm.vec3(0.0, 0.0, 0.0)

    def dt(self, delta_time: float) -> DeltaTimeTransformer:
        """
        Enables framerate-independent transformations for a chain of method calls.
        """
        return _SunDeltaTimeTransformer(self, delta_time)

    @property
    def position(self) -> glm.vec3:
        """World-space position of the sun."""
        return self._position

    @position.setter
    def position(self, value: Union[glm.vec3, ArrayLike]):
        self._position = glm.vec3(value)

    def translate(self, translation: Union[glm.vec3, ArrayLike]):
        """Translates the sun by a given vector."""
        self._position += glm.vec3(translation)
        return self

    def orbit(self, angle: float, axis: Union[str, glm.vec3, ArrayLike] = 'y', degrees: bool = True):
        """
        Orbits the sun around the target point (default: origin) along the given axis.

        Args:
            angle: Rotation angle
            axis: Axis to rotate around (can be 'x', 'y', 'z' or a vec3)
            degrees: If True, angle is in degrees
        """
        if isinstance(axis, str):
            axis_map = {
                'x': WORLD_RIGHT,
                'y': WORLD_UP,
                'z': WORLD_FORWARD,
                'right': WORLD_RIGHT,
                'up': WORLD_UP,
                'forward': WORLD_FORWARD,
            }
            rotation_axis = axis_map.get(axis.lower())
            if rotation_axis is None:
                raise ValueError(f"Unknown axis: '{axis}'. Use 'x', 'y', 'z' or a vec3.")
        else:
            rotation_axis = glm.vec3(axis)

        angle_rad = glm.radians(angle) if degrees else angle

        offset = self._position - self._target
        rotation = glm.rotate(glm.mat4(1.0), angle_rad, rotation_axis)
        rotated_offset = glm.vec3(rotation * glm.vec4(offset, 0.0))
        self._position = self._target + rotated_offset

        return self

    def from_angles(self, azimuth: float, elevation: float, distance: float = 100.0, degrees: bool = True):
        """
        Sets the sun position from spherical coordinates.

        Args:
            azimuth: Horizontal angle (0 = +Z, 90 = +X when looking down Y axis)
            elevation: Vertical angle above horizon (0 = horizon, 90 = directly overhead)
            distance: Distance from target point
            degrees: If True, angles are in degrees
        """
        if degrees:
            azimuth = glm.radians(azimuth)
            elevation = glm.radians(elevation)

        cos_el = np.cos(elevation)
        x = distance * cos_el * np.sin(azimuth)
        y = distance * np.sin(elevation)
        z = distance * cos_el * np.cos(azimuth)

        self._position = self._target + glm.vec3(x, y, z)
        return self

    @property
    def azimuth(self) -> float:
        """Current azimuth angle (in degrees)."""
        offset = self._position - self._target
        return np.degrees(np.arctan2(offset.x, offset.z))

    @property
    def elevation(self) -> float:
        """Current elevation angle (in degrees)."""
        offset = self._position - self._target
        horizontal_dist = np.sqrt(offset.x**2 + offset.z**2)
        return np.degrees(np.arctan2(offset.y, horizontal_dist))

    @property
    def distance(self) -> float:
        """Distance from target point."""
        return glm.length(self._position - self._target)

    @property
    def target(self) -> glm.vec3:
        """The point the sun is directed toward."""
        return self._target

    @target.setter
    def target(self, value: Union[glm.vec3, ArrayLike]):
        self._target = glm.vec3(value)

    def lookat(self, target: Union[glm.vec3, ArrayLike]):
        """Sets the target point the sun shines toward."""
        self._target = glm.vec3(target)
        return self

    @property
    def direction(self) -> glm.vec3:
        """
        Normalized direction vector *from* the sun *to* the target.
        """
        dir_vec = self._target - self._position
        length = glm.length(dir_vec)
        if length < 1e-6:
            # if sun is at target
            return glm.normalize(glm.vec3(0.0, -1.0, 0.0))
        return glm.normalize(dir_vec)

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, value)

    @property
    def angular_radius(self) -> float:
        """Angular radius in radians (for soft shadow penumbra)."""
        return self._angular_radius

    @angular_radius.setter
    def angular_radius(self, value: float):
        self._angular_radius = max(0.001, value)

    @property
    def color(self) -> glm.vec3:
        return self._color

    @color.setter
    def color(self, value: Union[glm.vec3, ArrayLike]):
        self._color = glm.vec3(value)

    def set_time_of_day(self, hour: float, latitude: float = 45.0):
        """
        Rough sun position based on time of day.

        Args:
            hour: Time in 24-hour format
            latitude: Observer latitude in degrees (affects max sun elevation)
        """
        azimuth = (hour - 6.0) * 15.0  # 15 degrees/hour, 0 at 6am

        # Elevation: peaks at noon, 0 at sunrise/sunset
        hour_from_noon = abs(hour - 12.0)
        max_elevation = 90.0 - abs(latitude - 23.5)  # rough approximation
        elevation = max_elevation * np.cos(np.radians(hour_from_noon * 15.0))
        elevation = max(0.0, elevation)  # below horizon

        self.set_from_angles(azimuth, elevation)

        # Adjust colour temperature based on elevation :)
        if elevation < 15.0:
            # golden hour
            warmth = 1.0 - (elevation / 15.0)
            self._color = glm.vec3(1.0, 0.9 - 0.2 * warmth, 0.7 - 0.3 * warmth)
        else:
            self._color = glm.vec3(1.0, 1.0, 1.0)

        return self