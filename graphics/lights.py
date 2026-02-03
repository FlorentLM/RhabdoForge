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
    Represents a directional sun light.

    The sun is treated as infinitely distant - only direction matters for lighting.
    Position is stored for visualisation (sun disk) and to make orbit() intuitive.
    """

    def __init__(self,
                 position: Sequence[float] = (50.0, 100.0, 30.0),
                 intensity: float = 2.0,
                 angular_radius: float = 0.02,
                 color: Sequence[float] = None):
        """
        Args:
            position: Direction the sun shines FROM (will be normalised for lighting)
            intensity: Brightness multiplier
            angular_radius: Apparent angular size of the sun disk in radians (affects soft shadows)
            color: RGB colour override (None = automatic elevation-based colour)
        """
        self._position = glm.vec3(position)
        self._intensity = intensity
        self._angular_radius = angular_radius
        self._color_override = glm.vec3(color) if color is not None else None

    def dt(self, delta_time: float) -> DeltaTimeTransformer:
        """Enables framerate-independent transformations."""
        return _SunDeltaTimeTransformer(self, delta_time)

    @property
    def position(self) -> glm.vec3:
        """World-space position (for visualisation / orbit calculations)."""
        return self._position

    @position.setter
    def position(self, value: Union[glm.vec3, ArrayLike]):
        self._position = glm.vec3(value)

    @property
    def direction(self) -> glm.vec3:
        """Normalised direction TO the sun (for shadow rays and lighting calculations)."""
        length = glm.length(self._position)
        if length < 1e-6:
            return glm.vec3(0.0, 1.0, 0.0)
        return glm.normalize(self._position)

    def translate(self, translation: Union[glm.vec3, ArrayLike]):
        """Translates the sun position."""
        self._position += glm.vec3(translation)
        return self

    def orbit(self, angle: float, axis: Union[str, glm.vec3, ArrayLike] = 'y', degrees: bool = True):
        """
        Orbits the sun around the origin.

        Args:
            angle: Rotation angle
            axis: Axis to rotate around ('x', 'y', 'z' or a vec3)
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
        rotation = glm.rotate(glm.mat4(1.0), angle_rad, rotation_axis)
        self._position = glm.vec3(rotation * glm.vec4(self._position, 0.0))

        return self

    def from_angles(self, azimuth: float, elevation: float, distance: float = 100.0, degrees: bool = True):
        """
        Sets the sun position from spherical coordinates.

        Args:
            azimuth: Horizontal angle (0 = +Z, 90 = +X when looking down Y axis)
            elevation: Vertical angle above horizon (0 = horizon, 90 = directly overhead)
            distance: Distance from origin (for visualisation only)
            degrees: If True, angles are in degrees
        """
        if degrees:
            azimuth = glm.radians(azimuth)
            elevation = glm.radians(elevation)

        cos_el = np.cos(elevation)
        x = distance * cos_el * np.sin(azimuth)
        y = distance * np.sin(elevation)
        z = distance * cos_el * np.cos(azimuth)

        self._position = glm.vec3(x, y, z)
        return self

    @property
    def azimuth(self) -> float:
        """Current azimuth angle in degrees."""
        return np.degrees(np.arctan2(self._position.x, self._position.z))

    @property
    def elevation(self) -> float:
        """Current elevation angle in degrees."""
        horizontal_dist = np.sqrt(self._position.x ** 2 + self._position.z ** 2)
        return np.degrees(np.arctan2(self._position.y, horizontal_dist))

    @property
    def distance(self) -> float:
        """Distance from origin."""
        return glm.length(self._position)

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, value)

    @property
    def angular_radius(self) -> float:
        """Angular radius in radians."""
        return self._angular_radius

    @angular_radius.setter
    def angular_radius(self, value: float):
        self._angular_radius = max(0.001, value)

    @property
    def color(self) -> glm.vec3:
        """
        Sun colour (based on elevation).

        Returns warm orange/red near horizon, white at high elevation.
        """

        if self._color_override is not None:
            return self._color_override

        elevation = self.elevation  # in degrees

        if elevation <= 0.0:
            # Below horizon: deep red/orange
            return glm.vec3(1.0, 0.3, 0.1)
        elif elevation < 6.0:
            # Golden hour: orange gold
            t = elevation / 6.0
            return glm.vec3(1.0, 0.3 + 0.4 * t, 0.1 + 0.3 * t)
        elif elevation < 15.0:
            # Golden hour transition: gold / warm white
            t = (elevation - 6.0) / 9.0
            return glm.vec3(1.0, 0.7 + 0.25 * t, 0.4 + 0.5 * t)
        elif elevation < 30.0:
            # Warm daylight
            t = (elevation - 15.0) / 15.0
            return glm.vec3(1.0, 0.95 + 0.05 * t, 0.9 + 0.1 * t)
        else:
            # High sun: neutral white
            return glm.vec3(1.0, 1.0, 1.0)

    @color.setter
    def color(self, value: Union[glm.vec3, ArrayLike, None]):
        """Set a fixed colour override, or None to use automatic elevation-based colour."""
        if value is None:
            self._color_override = None
        else:
            self._color_override = glm.vec3(value)

    def set_time_of_day(self, hour: float, latitude: float = 45.0):
        """
        Approximate sun position based on time of day.

        Args:
            hour: Time in 24-hour format (6 = sunrise, 12 = noon, 18 = sunset)
            latitude: Observer latitude in degrees
        """
        azimuth = (hour - 6.0) * 15.0  # 15 degrees/hour, 0 at 6am

        hour_from_noon = abs(hour - 12.0)
        max_elevation = 90.0 - abs(latitude - 23.5)
        elevation = max_elevation * np.cos(np.radians(hour_from_noon * 15.0))
        elevation = max(0.0, elevation)

        self.from_angles(azimuth, elevation)
        return self