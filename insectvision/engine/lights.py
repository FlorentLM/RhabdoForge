from abc import ABC, abstractmethod
from typing import Union, Sequence, Optional
from enum import Enum, auto
from numpy.typing import ArrayLike
import numpy as np
from pyglm import glm

from .world_utils import WORLD_UP
from .movement import TransformMixin

DIR_LIGHT_DTYPE = np.dtype([
    ('direction', np.float32, 3),
    ('angular_radius', np.float32),
    ('color', np.float32, 3),
    ('intensity', np.float32),
    ('cast_shadows', np.uint32),
    ('_pad', np.uint32, 3),
])  # total 48 bytes

POINT_LIGHT_DTYPE = np.dtype([
    ('position', np.float32, 3),
    ('radius', np.float32),
    ('color', np.float32, 3),
    ('intensity', np.float32),
    ('constant_atten', np.float32),
    ('linear_atten', np.float32),
    ('quadratic_atten', np.float32),
    ('cast_shadows', np.uint32),
])  # total 48 bytes

AREA_LIGHT_DTYPE = np.dtype([
    ('position', np.float32, 3),
    ('width', np.float32),
    ('normal', np.float32, 3),
    ('height', np.float32),
    ('tangent', np.float32, 3),
    ('intensity', np.float32),
    ('bitangent', np.float32, 3),
    ('cast_shadows', np.uint32),
    ('color', np.float32, 3),
    ('two_sided', np.uint32),
])  # total 64 bytes

# TODO: these three datatypes could be optimised a bit


class LightType(Enum):
    Directional = auto()    # Infinitely distant (sun, moon)
    Point = auto()          # Omnidirectional (with falloff)
    Area = auto()           # Rectangular/disk emitter


class Light(ABC, TransformMixin):
    """
    Abstract base class for all light types.
    """

    def __init__(self,
                 color: Sequence[float] = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 cast_shadows: bool = True,
                 active: bool = True):
        """
        Args:
            color: RGB colour of the light
            intensity: Brightness multiplier
            cast_shadows: Whether this light casts shadows
            active: Whether this light contributes to the scene
        """
        self.transform = glm.mat4(1.0)
        self._color = glm.vec3(color)
        self._intensity = max(0.0, intensity)
        self.cast_shadows = cast_shadows
        self._active = active

    @property
    def active(self) -> bool:
        return self._active and self._intensity > 0.0

    @property
    @abstractmethod
    def light_type(self) -> LightType:
        pass

    @property
    def color(self) -> glm.vec3:
        return self._color

    @color.setter
    def color(self, value: Union[glm.vec3, ArrayLike, None]):
        if value is None:
            self._color = glm.vec3(1.0, 1.0, 1.0)
        else:
            self._color = glm.vec3(value)

    @property
    def intensity(self) -> float:
        return self._intensity

    @intensity.setter
    def intensity(self, value: float):
        self._intensity = max(0.0, value)


class DirectionalLight(Light):
    """Infinitely distant light source."""

    _CONVENTIONAL_DISTANCE = 1000.0     # just for lookat which needs a finite value, has no physical meaning

    def __init__(self,
                 direction: Sequence[float] = (0.0, 1.0, 0.0),
                 color: Sequence[float] = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 angular_size: float = 0.0,
                 **kwargs):
        """
        Args:
            direction: direction: Unit vector pointing toward the light source (i.e. where the light arrives *from*). Will be normalised.
            color: RGB colour
            intensity: Brightness multiplier
            angular_size: Angular radius in radians (for soft shadows, 0 = hard shadows)
        """
        super().__init__(color=color, intensity=intensity, **kwargs)
        self._angular_size = max(0.0, angular_size)
        self.direction = direction

    @property
    def light_type(self) -> LightType:
        return LightType.Directional

    @property
    def direction(self) -> glm.vec3:
        """Unit vector pointing toward the light source (i.e. where the light arrives *from*)."""
        return self.backward

    @direction.setter
    def direction(self, value):
        d = glm.normalize(glm.vec3(value))
        self.position = d * self._CONVENTIONAL_DISTANCE
        self.lookat(glm.vec3(0.0, 0.0, 0.0))

    @property
    def angular_size(self) -> float:
        """Angular size in radians (affects soft shadow size)."""
        return self._angular_size

    @angular_size.setter
    def angular_size(self, value: float):
        self._angular_size = max(0.0, value)

    def pack(self) -> np.ndarray:
        data = np.zeros(1, dtype=DIR_LIGHT_DTYPE)
        d = self.direction
        data['direction'] = d.x, d.y, d.z
        data['angular_radius'] = self.angular_size
        data['color'] = self.color.x, self.color.y, self.color.z
        data['intensity'] = self.intensity
        data['cast_shadows'] = 1 if self.cast_shadows else 0
        return data


class Sun(DirectionalLight):
    """
    Directional light with solar convenience properties:

    - Position-based direction
    - Elevation-based colour temperature (warm at horizon, white at zenith)
    - Time-of-day simulation
    """

    def __init__(self,
                 azimuth: float = 0.0,
                 elevation: float = 45.0,
                 intensity: float = 2.0,
                 angular_size: float = 0.02,
                 color: Optional[Sequence[float]] = None,
                 **kwargs):
        """
        Args:
            azimuth: Horizontal angle in degrees
            elevation: Vertical angle above horizon in degrees
            intensity: Brightness multiplier
            angular_size: Apparent angular size of sun disk in radians
            color: RGB colour override (None = automatic elevation-based colour)
        """
        super().__init__(intensity=intensity, angular_size=angular_size, **kwargs)

        self.from_angles(azimuth, elevation)
        self._color_override = glm.vec3(color) if color is not None else None

    @property
    def azimuth(self) -> float:
        """Azimuth from position (degrees)."""
        return glm.degrees(glm.atan2(self.position.x, self.position.z))

    @azimuth.setter
    def azimuth(self, val: float):
        self.from_angles(float(val), self.elevation, self.distance)

    @property
    def elevation(self) -> float:
        """Elevation from position (degrees)."""
        horiz = glm.sqrt(self.position.x * self.position.x + self.position.z * self.position.z)
        return glm.degrees(glm.atan2(self.position.y, horiz))

    @elevation.setter
    def elevation(self, val: float):
        self.from_angles(self.azimuth, float(val), self.distance)

    @property
    def distance(self) -> float:
        """Distance from origin."""
        return glm.length(self.position)

    @distance.setter
    def distance(self, val: float):
        """Sets the distance while maintaining direction."""
        dir_vec = glm.normalize(self.position) if self.distance > 1e-6 else WORLD_UP
        self.position = dir_vec * val

    def orbit(self, angle: float, axis: Union[str, glm.vec3, ArrayLike] = 'y', degrees: bool = True):
        """Orbits the sun by rotating its transform around the world origin."""
        self.rotate_axis(angle, axis, degrees=degrees)
        return self

    def from_angles(self, azimuth: float, elevation: float, distance: float = 1000.0, degrees: bool = True):
        """Sets the sun position and orientation from spherical coordinates."""
        if degrees:
            az_rad = glm.radians(azimuth)
            el_rad = glm.radians(elevation)
        else:
            az_rad, el_rad = azimuth, elevation

        cos_el = np.cos(el_rad)
        x = distance * cos_el * np.sin(az_rad)
        y = distance * np.sin(el_rad)
        z = distance * cos_el * np.cos(az_rad)

        self.position = glm.vec3(x, y, z)
        self.lookat(glm.vec3(0.0, 0.0, 0.0))
        return self

    @property
    def color(self) -> glm.vec3:
        """
        Sun colour (based on elevation if no override set).
        Warm orange/red near horizon, white at zenith.
        """
        if self._color_override is not None:
            return self._color_override

        el = self.elevation
        if el <= 0.0:
            return glm.vec3(1.0, 0.3, 0.1)
        elif el < 6.0:
            t = el / 6.0
            return glm.vec3(1.0, 0.3 + 0.4 * t, 0.1 + 0.3 * t)
        elif el < 30.0:
            t = (el - 6.0) / 24.0
            return glm.vec3(1.0, 0.7 + 0.3 * t, 0.4 + 0.6 * t)
        return glm.vec3(1.0, 1.0, 1.0)

    @color.setter
    def color(self, value: Optional[Union[glm.vec3, ArrayLike]]):
        """Set a fixed colour override, or None to use elevation-based colour."""
        self._color_override = glm.vec3(value) if value is not None else None

    def set_time_of_day(self, hour: float, latitude: float = 45.0):
        """
        Approximate sun position based on time of day.

        Args:
            hour: Time in 24-hour format (6 = sunrise, 12 = noon, 18 = sunset)
            latitude: Observer latitude in degrees
        """
        azimuth = (hour - 6.0) * 15.0  # 15 degrees/hour
        hour_from_noon = abs(hour - 12.0)
        max_elevation = 90.0 - abs(latitude - 23.5)
        elevation = max_elevation * np.cos(np.radians(hour_from_noon * 15.0))

        self.from_angles(azimuth, max(0.0, elevation))
        return self


class PointLight(Light):
    """Omnidirectional point light with distance attenuation."""

    def __init__(self,
                 position: Sequence[float] = (0.0, 1.0, 0.0),
                 color: Sequence[float] = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 radius: float = 0.0,
                 attenuation: Sequence[float] = (1.0, 0.09, 0.032),
                 **kwargs):
        """
        Args:
            position: World-space position
            color: RGB colour
            intensity: Brightness multiplier
            radius: Physical radius for soft shadows (0 = point source)
            attenuation: (constant, linear, quadratic) factors
        """
        super().__init__(color=color, intensity=intensity, **kwargs)

        self.position = position
        self._radius = max(0.0, radius)
        self.constant, self.linear, self.quadratic = attenuation

    @property
    def light_type(self) -> LightType:
        return LightType.Point

    @property
    def radius(self) -> float:
        """Physical radius for soft shadows."""
        return self._radius

    @radius.setter
    def radius(self, value: float):
        self._radius = max(0.0, value)

    def attenuation_at(self, distance: float) -> float:
        """Calculate attenuation factor at a given distance."""
        return 1.0 / (self.constant + self.linear * distance + self.quadratic * distance * distance)

    def pack(self) -> np.ndarray:
        data = np.zeros(1, dtype=POINT_LIGHT_DTYPE)
        p = self.position
        data['position'] = p.x, p.y, p.z
        data['radius'] = self.radius
        data['color'] = self.color.x, self.color.y, self.color.z
        data['intensity'] = self.intensity
        data['constant_atten'] = self.constant
        data['linear_atten'] = self.linear
        data['quadratic_atten'] = self.quadratic
        data['cast_shadows'] = 1 if self.cast_shadows else 0
        return data


class AreaLight(Light):
    """Rectangular area light for soft shadows."""

    def __init__(self,
                 position: Sequence[float] = (0.0, 2.0, 0.0),
                 look_at: Sequence[float] = (0.0, 0.0, 0.0),
                 width: float = 1.0,
                 height: float = 1.0,
                 color: Sequence[float] = (1.0, 1.0, 1.0),
                 intensity: float = 5.0,
                 two_sided: bool = False,
                 **kwargs):
        """
        Args:
            position: Center of the light rectangle
            look_at: Point in space the light faces
            width: Width of the rectangle
            height: Height of the rectangle
            color: RGB colour
            intensity: Brightness multiplier
            two_sided: Whether light emits from both sides
        """
        super().__init__(color=color, intensity=intensity, **kwargs)
        self.position = position
        self.lookat(look_at)
        self._width = max(0.001, width)
        self._height = max(0.001, height)
        self.two_sided = two_sided

    @property
    def light_type(self) -> LightType:
        return LightType.Area

    # Basis vector mapping

    @property
    def normal(self) -> glm.vec3:
        """The emission normal is the local forward (-Z) vector."""
        return self.forward

    @property
    def tangent(self) -> glm.vec3:
        """The tangent is the local right (+X) vector."""
        return self.right

    @property
    def bitangent(self) -> glm.vec3:
        """The bitangent is the local up (+Y) vector."""
        return self.up

    @property
    def width(self) -> float:
        """Width of the rectangular emitter."""
        return self._width

    @width.setter
    def width(self, value: float):
        self._width = max(0.001, value)

    @property
    def height(self) -> float:
        """Height of the rectangular emitter."""
        return self._height

    @height.setter
    def height(self, value: float):
        self._height = max(0.001, value)

    @property
    def area(self) -> float:
        """Surface area of the light."""
        return self._width * self._height

    def sample_point(self, u: float, v: float) -> glm.vec3:
        """
        Sample a point on the light surface.
        u, v: Values in [0, 1]
        """
        local_u = (u - 0.5) * self._width
        local_v = (v - 0.5) * self._height
        return self.position + self.tangent * local_u + self.bitangent * local_v

    def pack(self) -> np.ndarray:
        data = np.zeros(1, dtype=AREA_LIGHT_DTYPE)
        p, n, t, b = self.position, self.normal, self.tangent, self.bitangent
        data['position'] = p.x, p.y, p.z
        data['width'] = self.width
        data['normal'] = n.x, n.y, n.z
        data['height'] = self.height
        data['tangent'] = t.x, t.y, t.z
        data['intensity'] = self.intensity
        data['bitangent'] = b.x, b.y, b.z
        data['cast_shadows'] = 1 if self.cast_shadows else 0
        data['color'] = self.color.x, self.color.y, self.color.z
        data['two_sided'] = 1 if self.two_sided else 0
        return data