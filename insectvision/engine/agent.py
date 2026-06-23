from pyglm import glm
from .world_utils import WORLD_UP
from .movement import TransformMixin


class Agent(TransformMixin):
    def __init__(self,
                 position=(0.0, 0.0, 0.0),
                 yaw=0.0,
                 pitch=0.0,
                 roll=0.0,
                 fov=50.0,
                 near=0.001,
                 far=100.0,
                 ratio=16.0 / 9.0):

        self.transform = glm.mat4(1.0)

        self.position = position
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

        self._fov = fov
        self._near = near
        self._far = far
        self._ratio = ratio

    def __repr__(self):
        p = self.position
        return (f"<Agent | pos=({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) | "
                f"yaw={self.yaw:.1f}° pitch={self.pitch:.1f}° roll={self.roll:.1f}° | fov={self.fov:.0f}°>")

    # Projection setters

    @property
    def fov(self):
        return self._fov

    @fov.setter
    def fov(self, value):
        self._fov = glm.clamp(value, 0.1, 179.0)

    @property
    def near(self):
        return self._near

    @near.setter
    def near(self, value):
        self._near = max(0.0001, value)

    @property
    def far(self):
        return self._far

    @far.setter
    def far(self, value):
        self._far = max(self._near + 0.1, value)

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        self._ratio = max(0.01, value)

    # Matrices

    @property
    def projection(self):
        return glm.perspective(glm.radians(self.fov), self.ratio, self.near, self.far)

    @property
    def view(self):
        return glm.inverse(self.transform)

    @property
    def matrix(self):
        return self.projection * self.view


class OrbitCamera:
    """
    A helper that orbits an Agent. Uses an internal Agent as the 'observer'.
    """

    def __init__(self, target, distance=0.5, azimuth=0.0, elevation=20.0, **kwargs):

        self.target = target
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation

        self._observer = Agent(**kwargs)
        self.update()

    def __repr__(self):
        t = self.target.position
        return (f"<OrbitCamera | target=({t.x:.2f}, {t.y:.2f}, {t.z:.2f}) | "
                f"dist={self.distance:.2f} | az={self.azimuth:.1f}° el={self.elevation:.1f}°>")

    def pan(self, az_delta, el_delta):
        self.azimuth += az_delta
        self.elevation = glm.clamp(self.elevation + el_delta, -89.9, 89.9)
        self.update()

    def zoom(self, factor):
        self.distance = max(0.001, self.distance * factor)
        self.update()

    def update(self):
        """Recalculates the observer's position and orientation based on orbit parameters."""

        az_rad = glm.radians(self.azimuth)
        el_rad = glm.radians(self.elevation)

        offset = glm.vec3(
            self.distance * glm.cos(el_rad) * glm.sin(az_rad),
            self.distance * glm.sin(el_rad),
            self.distance * glm.cos(el_rad) * glm.cos(az_rad)
        )

        self._observer.position = self.target.position + offset
        self._observer.lookat(self.target.position, up=WORLD_UP)

    @property
    def view(self):
        # Explicit lookAt to ensure no roll (horizon stays level)
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
    def ratio(self, v):
        self._observer.ratio = v