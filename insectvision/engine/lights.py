from abc import ABC, abstractmethod
from typing import Union, Sequence
from enum import Enum, auto
from numpy.typing import ArrayLike
import numpy as np
from pyglm import glm

from insectvision.engine.utils import WORLD_UP, WORLD_RIGHT, WORLD_FORWARD, DeltaTimeTransformer


# Custom dtypes for the GPU SSBOs

directional_light_dtype = np.dtype([
    ('direction', np.float32, 3),
    ('angular_radius', np.float32),
    ('color', np.float32, 3),
    ('intensity', np.float32),
    ('cast_shadows', np.uint32),
    ('_pad', np.uint32, 3),
])  # total 48 bytes

point_light_dtype = np.dtype([
    ('position', np.float32, 3),
    ('radius', np.float32),
    ('color', np.float32, 3),
    ('intensity', np.float32),
    ('constant_atten', np.float32),
    ('linear_atten', np.float32),
    ('quadratic_atten', np.float32),
    ('cast_shadows', np.uint32),
])  # total 48 bytes

area_light_dtype = np.dtype([
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
    DIRECTIONAL = auto()    # Infinitely distant (sun, moon)
    POINT = auto()          # Omnidirectional (with falloff)
    AREA = auto()           # Rectangular/disk emitter


class Light(ABC):
    """
    Abstract base class for all light types.
    """

    def __init__(self,
                 color: Sequence[float] = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 cast_shadows: bool = True,
                 enabled: bool = True):
        """
        Args:
            color: RGB colour of the light
            intensity: Brightness multiplier
            cast_shadows: Whether this light casts shadows
            enabled: Whether this light contributes to the scene
        """
        self._color = glm.vec3(color)
        self._intensity = max(0.0, intensity)
        self.cast_shadows = cast_shadows
        self.enabled = enabled

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


class _LightDeltaTimeTransformer(DeltaTimeTransformer):
    """Delta-time transformer for lights with orbital/positional controls."""

    # TODO: Orbit might make its way into the other classes actually

    def __init__(self, target: 'Light', delta_time: float):
        super().__init__(target, delta_time)

    def orbit(self, angle: float, axis: Union[str, glm.vec3, ArrayLike] = 'y', degrees: bool = True):
        """Orbits the light around the origin (for lights that support it)."""
        if hasattr(self._target, 'orbit'):
            scaled_angle = angle * self._delta_time
            self._target.orbit(scaled_angle, axis, degrees=degrees)
        return self


class DirectionalLight(Light):
    """
    Infinitely distant light source.
    """

    def __init__(self,
                 direction: Sequence[float] = (0.0, 1.0, 0.0),
                 color: Sequence[float] = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 angular_size: float = 0.0,
                 **kwargs):
        """
        Args:
            direction: Direction the light shines from (will be normalised)
            color: RGB colour
            intensity: Brightness multiplier
            angular_size: Angular radius in radians (for soft shadows, 0 = hard shadows)
        """
        super().__init__(color=color, intensity=intensity, **kwargs)
        self._direction = glm.normalize(glm.vec3(direction))
        self._angular_radius = max(0.0, angular_size)

    @property
    def light_type(self) -> LightType:
        return LightType.DIRECTIONAL

    @property
    def direction(self) -> glm.vec3:
        """Normalised direction to the light source."""
        return self._direction

    @direction.setter
    def direction(self, value: Union[glm.vec3, ArrayLike]):
        vec = glm.vec3(value)
        length = glm.length(vec)
        if length > 1e-6:
            self._direction = vec / length
        else:
            self._direction = glm.vec3(0.0, 1.0, 0.0)

    @property
    def angular_radius(self) -> float:
        """Angular radius in radians (affects soft shadow size)."""
        return self._angular_radius

    @angular_radius.setter
    def angular_radius(self, value: float):
        self._angular_radius = max(0.0, value)

    def pack(self) -> np.ndarray:
        data = np.zeros(1, dtype=directional_light_dtype)
        data['direction'] = self.direction.x, self.direction.y, self.direction.z
        data['angular_radius'] = self.angular_radius
        data['color'] = self.color.x, self.color.y, self.color.z
        data['intensity'] = self.intensity
        data['cast_shadows'] = 1 if self.cast_shadows else 0
        return data


class Sun(DirectionalLight):
    """
    Directional sun light with orbital controls and automatic colour temperature.

    Extends DirectionalLight with:
    - Position-based direction
    - Elevation-based colour temperature (warm at horizon, white at zenith)
    - Time-of-day simulation
    """

    def __init__(self,
                 position: Sequence[float] = (50.0, 100.0, 30.0),
                 intensity: float = 2.0,
                 angular_size: float = 0.02,
                 color: Sequence[float] = None,
                 **kwargs):
        """
        Args:
            position: Direction the sun shines from (magnitude used for visualisation)
            intensity: Brightness multiplier
            angular_size: Apparent angular size of sun disk in radians
            color: RGB colour override (None = automatic elevation-based colour)
        """
        super().__init__(
            direction=(0.0, 1.0, 0.0),
            color=color if color is not None else (1.0, 1.0, 1.0),
            intensity=intensity,
            angular_size=angular_size,
            **kwargs
        )

        self._position = glm.vec3(position)
        self._color_override = glm.vec3(color) if color is not None else None
        self._update_direction_from_position()

    def _update_direction_from_position(self):
        """Updates the direction vector from the current position."""
        length = glm.length(self._position)
        if length > 1e-6:
            self._direction = self._position / length
        else:
            self._direction = glm.vec3(0.0, 1.0, 0.0)

    def dt(self, delta_time: float) -> _LightDeltaTimeTransformer:
        """Enables framerate-independent transformations."""
        return _LightDeltaTimeTransformer(self, delta_time)

    @property
    def position(self) -> glm.vec3:
        """World-space position (for visualisation / orbit calculations)."""
        return self._position

    @position.setter
    def position(self, value: Union[glm.vec3, ArrayLike]):
        self._position = glm.vec3(value)
        self._update_direction_from_position()

    @property
    def direction(self) -> glm.vec3:
        """Normalised direction to the sun."""
        return self._direction

    def translate(self, translation: Union[glm.vec3, ArrayLike]):
        """Translates the sun position."""
        self._position += glm.vec3(translation)
        self._update_direction_from_position()
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
        self._update_direction_from_position()

        return self

    def from_angles(self, azimuth: float, elevation: float, distance: float = 1000.0, degrees: bool = True):
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
        self._update_direction_from_position()
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
    def color(self) -> glm.vec3:
        """
        Sun colour (based on elevation if no override set).
        Warm orange/red near horizon, white at zenith.
        """
        if self._color_override is not None:
            return self._color_override

        elevation = self.elevation  # in degrees

        if elevation <= 0.0:
            return glm.vec3(1.0, 0.3, 0.1)
        elif elevation < 6.0:
            t = elevation / 6.0
            return glm.vec3(1.0, 0.3 + 0.4 * t, 0.1 + 0.3 * t)
        elif elevation < 15.0:
            t = (elevation - 6.0) / 9.0
            return glm.vec3(1.0, 0.7 + 0.25 * t, 0.4 + 0.5 * t)
        elif elevation < 30.0:
            t = (elevation - 15.0) / 15.0
            return glm.vec3(1.0, 0.95 + 0.05 * t, 0.9 + 0.1 * t)
        else:
            return glm.vec3(1.0, 1.0, 1.0)

    @color.setter
    def color(self, value: Union[glm.vec3, ArrayLike, None]):
        """Set a fixed colour override, or None to use elevation-based colour."""
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
        azimuth = (hour - 6.0) * 15.0  # 15 degrees/hour

        hour_from_noon = abs(hour - 12.0)
        max_elevation = 90.0 - abs(latitude - 23.5)
        elevation = max_elevation * np.cos(np.radians(hour_from_noon * 15.0))
        elevation = max(0.0, elevation)

        self.from_angles(azimuth, elevation)
        return self


class PointLight(Light):
    """
    Omnidirectional point light with distance attenuation.
    """

    def __init__(self,
                 position: Sequence[float] = (0.0, 1.0, 0.0),
                 color: Sequence[float] = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 radius: float = 0.0,
                 constant: float = 1.0,
                 linear: float = 0.09,
                 quadratic: float = 0.032,
                 **kwargs):
        """
        Args:
            position: World-space position
            color: RGB colour
            intensity: Brightness multiplier
            radius: Physical radius for soft shadows (0 = point source)
            constant: Constant attenuation factor
            linear: Linear attenuation factor
            quadratic: Quadratic attenuation factor
        """
        super().__init__(color=color, intensity=intensity, **kwargs)
        self._position = glm.vec3(position)
        self._radius = max(0.0, radius)
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic

    @property
    def light_type(self) -> LightType:
        return LightType.POINT

    @property
    def position(self) -> glm.vec3:
        return self._position

    @position.setter
    def position(self, value: Union[glm.vec3, ArrayLike]):
        self._position = glm.vec3(value)

    @property
    def radius(self) -> float:
        """Physical radius for soft shadows."""
        return self._radius

    @radius.setter
    def radius(self, value: float):
        self._radius = max(0.0, value)

    def translate(self, translation: Union[glm.vec3, ArrayLike]):
        self._position += glm.vec3(translation)
        return self

    def attenuation_at(self, distance: float) -> float:
        """Calculate attenuation factor at a given distance."""
        return 1.0 / (self.constant + self.linear * distance + self.quadratic * distance * distance)

    def pack(self) -> np.ndarray:
        data = np.zeros(1, dtype=point_light_dtype)
        data['position'] = self.position.x, self.position.y, self.position.z
        data['radius'] = self.radius
        data['color'] = self.color.x, self.color.y, self.color.z
        data['intensity'] = self.intensity
        data['constant_atten'] = self.constant
        data['linear_atten'] = self.linear
        data['quadratic_atten'] = self.quadratic
        data['cast_shadows'] = 1 if self.cast_shadows else 0
        return data


class AreaLight(Light):
    """
    Rectangular area light for soft shadows (path tracing only).
    """

    def __init__(self,
                 position: Sequence[float] = (0.0, 2.0, 0.0),
                 normal: Sequence[float] = (0.0, -1.0, 0.0),
                 width: float = 1.0,
                 height: float = 1.0,
                 color: Sequence[float] = (1.0, 1.0, 1.0),
                 intensity: float = 5.0,
                 two_sided: bool = False,
                 **kwargs):
        """
        Args:
            position: Center of the light rectangle
            normal: Direction the light faces (perpendicular to surface)
            width: Width of the rectangle
            height: Height of the rectangle
            color: RGB colour
            intensity: Brightness multiplier
            two_sided: Whether light emits from both sides
        """
        super().__init__(color=color, intensity=intensity, **kwargs)
        self._position = glm.vec3(position)
        self._normal = glm.normalize(glm.vec3(normal))
        self._width = max(0.001, width)
        self._height = max(0.001, height)
        self.two_sided = two_sided

        self._update_tangent_frame()

    def _update_tangent_frame(self):
        """Builds orthonormal tangent and bitangent vectors from the normal."""
        up = glm.vec3(0.0, 1.0, 0.0) if abs(self._normal.y) < 0.999 else glm.vec3(1.0, 0.0, 0.0)
        self._tangent = glm.normalize(glm.cross(up, self._normal))
        self._bitangent = glm.cross(self._normal, self._tangent)

    @property
    def light_type(self) -> LightType:
        return LightType.AREA

    @property
    def position(self) -> glm.vec3:
        """Centre of the light rectangle."""
        return self._position

    @position.setter
    def position(self, value: Union[glm.vec3, ArrayLike]):
        self._position = glm.vec3(value)

    @property
    def normal(self) -> glm.vec3:
        """Direction the light faces."""
        return self._normal

    @normal.setter
    def normal(self, value: Union[glm.vec3, ArrayLike]):
        self._normal = glm.normalize(glm.vec3(value))
        self._update_tangent_frame()

    @property
    def tangent(self) -> glm.vec3:
        """Tangent vector (width direction)."""
        return self._tangent

    @property
    def bitangent(self) -> glm.vec3:
        """Bitangent vector (height direction)."""
        return self._bitangent

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float):
        self._width = max(0.001, value)

    @property
    def height(self) -> float:
        return self._height

    @height.setter
    def height(self, value: float):
        self._height = max(0.001, value)

    @property
    def area(self) -> float:
        """Surface area of the light."""
        return self._width * self._height

    def translate(self, translation: Union[glm.vec3, ArrayLike]):
        """Translates the light position."""
        self._position += glm.vec3(translation)
        return self

    def sample_point(self, u: float, v: float) -> glm.vec3:
        """
        Sample a point on the light surface.

        Args:
            u, v: Random values in [0, 1]
        """
        local_u = (u - 0.5) * self._width
        local_v = (v - 0.5) * self._height
        return self._position + self._tangent * local_u + self._bitangent * local_v

    def pack(self) -> np.ndarray:
        data = np.zeros(1, dtype=area_light_dtype)
        data['position'] = self.position.x, self.position.y, self.position.z
        data['width'] = self.width
        data['normal'] = self.normal.x, self.normal.y, self.normal.z
        data['height'] = self.height
        data['tangent'] = self.tangent.x, self.tangent.y, self.tangent.z
        data['intensity'] = self.intensity
        data['bitangent'] = self.bitangent.x, self.bitangent.y, self.bitangent.z
        data['cast_shadows'] = 1 if self.cast_shadows else 0
        data['color'] = self.color.x, self.color.y, self.color.z
        data['two_sided'] = 1 if self.two_sided else 0
        return data


##

def compute_light_space_matrix(light: DirectionalLight, scene_center=(0.0, 0.0, 0.0), scene_radius: float = 50.0) -> glm.mat4:

    center = glm.vec3(scene_center)
    light_pos = center + light.direction * scene_radius

    up = glm.vec3(0.0, 1.0, 0.0)
    if abs(glm.dot(glm.normalize(light.direction), up)) > 0.999:
        up = glm.vec3(0.0, 0.0, 1.0)

    light_view = glm.lookAt(light_pos, center, up)
    light_proj = glm.ortho(
        -scene_radius, scene_radius,
        -scene_radius, scene_radius,
        0.01, scene_radius * 2.0)
    return light_proj * light_view
