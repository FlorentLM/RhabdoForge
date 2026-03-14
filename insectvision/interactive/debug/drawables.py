from typing import Sequence, Optional, Union
import numpy as np
from pyglm import glm

from . import debug_primitives as dp
from .debug_renderer import DebugRenderer


class DebugDrawable:
    """
    Base for all debug eleemnts.
    """
    def __init__(self, color: Sequence[float] = (1, 1, 1), alpha: float = 1.0):
        self.color = tuple(color)
        self.alpha = alpha
        self.visible = True

    def draw(self, r: DebugRenderer):
        raise NotImplementedError


class LineObject(DebugDrawable):
    """
    Base for debug objects rendered as segments.
    """
    def __init__(self, color=(1, 1, 1), alpha=1.0):
        super().__init__(color=color, alpha=alpha)

    def _build_lines(self) -> Union[np.ndarray, tuple[np.ndarray, glm.mat4]]:
        """Returns line vertex data, optionally with model matrix"""
        return np.zeros(0, dtype=np.float32)

    def draw(self, r: DebugRenderer):
        if not self.visible:
            return

        result = self._build_lines()

        if isinstance(result, tuple):
            data, model = result
        else:
            data, model = result, None
        if data.size > 0:
            r.submit_lines(data, model=model, alpha=self.alpha)


class ArrowObject(DebugDrawable):
    """
    Base for debug objects rendered as an arrow
    """
    def __init__(self,
            color=(1, 1, 1),
            alpha=1.0,
            head_fraction=0.22,
            head_radius=0.02,
            shaft_width=3.0
        ):
        super().__init__(color=color, alpha=alpha)

        self.head_fraction = head_fraction
        self.head_radius = head_radius
        self.shaft_width = shaft_width

    def _resolve(self) -> tuple:
        return (0, 0, 0), (0, 0, -1), 1.0  # (origin, direction, length)

    def draw(self, r: DebugRenderer):

        if not self.visible:
            return
        origin, direction, length = self._resolve()
        shaft, cone = dp.make_arrow(
            length=length,
            color=self.color,
            direction=_as_tuple(direction),
            origin=_as_tuple(origin),
            head_fraction=self.head_fraction,
            head_radius=self.head_radius,
        )
        r.submit_lines(shaft, alpha=self.alpha, line_width=self.shaft_width)
        r.submit_tris(cone, alpha=self.alpha)


class MarkerObject(DebugDrawable):
    """
    Base for debug objects rendered as a small wireframe sphere marker with text label
    """
    def __init__(self,
            color=(1.0, 0.9, 0.2),
            alpha=1.0,
            marker_radius=0.03,
            label_scale=0.0018,
            label_offset=(0, 0.06, 0)
        ):
        super().__init__(color=color, alpha=alpha)
        self.marker_radius = marker_radius
        self.label_scale = label_scale
        self.label_offset = glm.vec3(label_offset)

    def _resolve(self) -> tuple[glm.vec3, str]:
        return glm.vec3(0), ""  # (world_position, label_text)

    def draw(self, r: DebugRenderer):
        if not self.visible:
            return
        pos, label = self._resolve()
        pos = glm.vec3(pos)

        # Wireframe sphere
        model = glm.translate(glm.mat4(1.0), pos)
        sphere_data = dp.make_wire_sphere(radius=self.marker_radius, color=self.color)
        r.submit_lines(sphere_data, model=model, alpha=self.alpha)

        # Billboard label
        if label:
            text_data = dp.make_text_lines(label, color=self.color)
            if text_data.size > 0:
                r.submit_billboard_lines(text_data, pos + self.label_offset, scale=self.label_scale)


def _as_tuple(v) -> tuple:
    if hasattr(v, 'x'):
        return (v.x, v.y, v.z)
    return tuple(v)


def _extract_aabb(source) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract an axis-aligned bounding box from various engine types.

    Accepts:
        (center, half_extents) tuple of two 3-sequences
        Instance (uses .asset geometry + .transform)
        Asset (uses .vertices / .points in local space)
        object with .aabb  -> (min, max) tuple
        object with .bounds -> (min, max) tuple
        object with .vertices + optional .transform
        pytinybvh BVH (has .aabb_min / .aabb_max)
    """

    transform = None
    if hasattr(source, 'asset') and hasattr(source, 'transform'):
        transform = source.transform
        source = source.asset

    if isinstance(source, (tuple, list)) and len(source) == 2:
        a, b = np.asarray(source[0], dtype=np.float32), np.asarray(source[1], dtype=np.float32)
        if a.shape == (3,) and b.shape == (3,):
            return a - b, a + b

    if hasattr(source, 'aabb'):
        mn, mx = source.aabb
        return np.asarray(mn, dtype=np.float32), np.asarray(mx, dtype=np.float32)

    if hasattr(source, 'bounds'):
        mn, mx = source.bounds
        return np.asarray(mn, dtype=np.float32), np.asarray(mx, dtype=np.float32)

    # BVH
    if hasattr(source, 'aabb_min') and hasattr(source, 'aabb_max'):
        return (np.asarray(source.aabb_min, dtype=np.float32),
                np.asarray(source.aabb_max, dtype=np.float32))

    # Asset (mesh or point cloud)
    verts = None
    if hasattr(source, 'vertices') and source.vertices is not None:
        verts = np.asarray(source.vertices, dtype=np.float32)
        if verts.ndim == 1:
            stride = getattr(source, 'vertex_stride', 3)
            verts = verts.reshape(-1, stride)[:, :3]
        elif verts.shape[1] > 3:
            verts = verts[:, :3]

    elif hasattr(source, 'points') and source.points is not None:
        verts = np.asarray(source.points, dtype=np.float32)
        if verts.ndim == 1:
            verts = verts.reshape(-1, 3)

    if verts is not None:
        mn = verts.min(axis=0)
        mx = verts.max(axis=0)

        # Apply transform (from Instance unwrap or from source.transform)
        T = transform if transform is not None else getattr(source, 'transform', None)

        if T is not None:
            corners = _aabb_corners(mn, mx)
            transformed = np.array([
                (T * glm.vec4(float(c[0]), float(c[1]), float(c[2]), 1.0)).xyz
                for c in corners
            ])
            mn = transformed.min(axis=0)
            mx = transformed.max(axis=0)

        return mn, mx

    raise TypeError(
        f"Cannot extract AABB from {type(source).__name__}. "
        "Expected (center, half_extents), an Instance, an Asset, "
        "an object with .aabb/.bounds/.vertices/.points, or a pytinybvh BVH."
    )


def _aabb_corners(mn, mx):
    return np.array([
        [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
    ], dtype=np.float32)


def _extract_points(source) -> np.ndarray:
    """
    Extract a point array from various stuff.

    Accepts:
        Trajectory (extracts .curve.points)
        Curve (extracts .points)
        Instance (extracts .asset.vertices positions)
        Asset (extracts .vertices or .points positions)
        raw array-like of (N, 3) points
    """
    # Trajectory -> Curve
    if hasattr(source, 'curve') and hasattr(source, 'progress'):
        source = source.curve

    # Curve
    if hasattr(source, 'points') and hasattr(source, 'total_length'):
        return np.asarray(source.points, dtype=np.float32)

    # Instance -> Asset
    if hasattr(source, 'asset'):
        source = source.asset

    # Asset with mesh (strips UV columns)
    if hasattr(source, 'vertices') and source.vertices is not None:
        verts = np.asarray(source.vertices, dtype=np.float32)

        if verts.ndim == 2 and verts.shape[1] > 3:
            return verts[:, :3]
        return verts

    # Asset with point cloud
    if hasattr(source, 'points') and source.points is not None:
        return np.asarray(source.points, dtype=np.float32)

    return np.asarray(source, dtype=np.float32)


def _resolve_position(source) -> glm.vec3:
    """Get world position from an object, tuple, or glm.vec3"""
    if hasattr(source, 'position'):
        return glm.vec3(source.position)
    return glm.vec3(source)


def _resolve_direction(source) -> glm.vec3:
    """Get direction from vec3, tuple, or callable"""
    if callable(source):
        return glm.vec3(source())
    return glm.vec3(source)


class DebugLine(LineObject):
    """
    Line segment between two points.
    Accepts raw tuples or any object with ``.position``
    """

    def __init__(self, start=(0, 0, 0), end=(1, 0, 0), **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.end = end

    def _build_lines(self):
        s = _as_tuple(_resolve_position(self.start))
        e = _as_tuple(_resolve_position(self.end))
        return dp.make_line(s, e, self.color)


class DebugPath(LineObject):
    """
    Connected polyline.

    Accepts:
        list/array of (x,y,z) points
        Curve object
        Trajectory object
    """

    def __init__(self,
            source,
            closed: bool = False,
            show_progress: bool = True,
            progress_color: Sequence[float] = (1, 1, 0),
            progress_radius: float = 0.04,
            **kwargs
        ):
        kwargs.setdefault('color', (1, 0.6, 0))
        super().__init__(**kwargs)
        self.source = source
        self.closed = closed
        self.show_progress = show_progress
        self.progress_color = progress_color
        self.progress_radius = progress_radius

    def _build_lines(self):
        pts = _extract_points(self.source)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return np.zeros(0, dtype=np.float32)
        return dp.make_polyline(pts, self.color, self.closed)

    def draw(self, r: DebugRenderer):

        super().draw(r)

        # If it's a Trajectory, add a marker at the current position
        if not self.visible or not self.show_progress:
            return

        src = self.source
        if hasattr(src, 'curve') and hasattr(src, 'progress'):
            pos, tangent = src.curve.get_sample_at(src.progress * src.curve.total_length)
            model = glm.translate(glm.mat4(1.0), pos)
            sphere = dp.make_wire_sphere(radius=self.progress_radius, color=self.progress_color)
            r.submit_lines(sphere, model=model)


class DebugGrid(LineObject):
    """XZ ground-plane grid (Blender-style)."""

    def __init__(self,
            size: float = 10.0,
            step: float = 1.0,
            y: float = 0.0,
            **kwargs
        ):
        kwargs.setdefault('color', (0.30, 0.30, 0.30))
        kwargs.setdefault('alpha', 0.6)
        super().__init__(**kwargs)
        self._data = dp.make_grid(size=size, step=step, color=self.color, y=y)

    def _build_lines(self):
        return self._data


class DebugBox(LineObject):
    """
    Wireframe bounding box.

    Accepts:
       ``DebugBox(center=(x,y,z), half_extents=(hx,hy,hz))``
       ``DebugBox(source=my_instance)``   — applies instance transform
       ``DebugBox(source=my_asset)``      — local-space bounds
       ``DebugBox(source=my_bvh)``        — pytinybvh BVH
       ``DebugBox(source=(center, half_ext))``
    """

    def __init__(self,
            source=None,
            center: Sequence[float] = None,
            half_extents: Sequence[float] = None,
            **kwargs
        ):
        kwargs.setdefault('color', (0, 1, 1))
        super().__init__(**kwargs)
        self.source = source
        self._center = center
        self._half_extents = half_extents

    def _build_lines(self):
        if self.source is not None:
            mn, mx = _extract_aabb(self.source)
            center = (mn + mx) * 0.5
            half = (mx - mn) * 0.5
        elif self._center is not None and self._half_extents is not None:
            center = np.asarray(self._center, dtype=np.float32)
            half = np.asarray(self._half_extents, dtype=np.float32)
        else:
            return np.zeros(0, dtype=np.float32)

        model = glm.translate(glm.mat4(1.0), glm.vec3(float(center[0]), float(center[1]), float(center[2])))
        data = dp.make_wire_box(tuple(half), self.color)
        return data, model


class DebugFrustum(LineObject):
    """
    Wireframe frustum.
    """

    def __init__(self,
            agent=None,
            fov: float = 50.0,
            aspect: float = 16/9,
            near: float = 0.1,
            far: float = 2.0,
            **kwargs
        ):
        kwargs.setdefault('color', (0.8, 0.8, 0.2))
        super().__init__(**kwargs)
        self.agent = agent
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

    def _build_lines(self):
        fov = self.agent.fov if self.agent else self.fov
        aspect = self.agent.ratio if self.agent else self.aspect
        data = dp.make_frustum_lines(fov, aspect, self.near, self.far, self.color)

        if self.agent:
            model = glm.inverse(self.agent.view)
        else:
            model = glm.mat4(1.0)
        return data, model


class GazeArrow(ArrowObject):
    """
    Arrow showing agent.forward. Tracks the agent every frame.
    (works on any object with ``.position`` and ``.forward`` attributes)
    """

    def __init__(self, agent, length=0.5, offset=(0, 0, 0), **kwargs):
        kwargs.setdefault('color', (1.0, 0.85, 0.0))
        super().__init__(**kwargs)
        self.agent = agent
        self.length = length
        self.offset = glm.vec3(offset)

    def _resolve(self):
        pos = glm.vec3(self.agent.position) + self.offset
        fwd = glm.vec3(self.agent.forward)
        return pos, fwd, self.length


class DirectionArrow(ArrowObject):
    """
    Arrow from a point to a direction.
    origin can be a tuple or any object with ``.position``
    direction can be a tuple, glm.vec3, or a callable returning one
    """

    def __init__(self, origin=(0, 0, 0), direction=(0, 0, -1), length=0.5, **kwargs):
        super().__init__(**kwargs)
        self._origin = origin
        self._direction = direction
        self.length = length

    def _resolve(self):
        o = _resolve_position(self._origin)
        d = _resolve_direction(self._direction)
        return o, d, self.length


class DebugRay(ArrowObject):
    """
    Visualise a ray (e.g. from collision queries).
    origin can be a tuple or any object with ``.position``
    """

    def __init__(self, origin=(0, 0, 0), direction=(0, 0, -1), length=5.0, **kwargs):
        kwargs.setdefault('color', (1, 0.4, 0.4))
        kwargs.setdefault('head_fraction', 0.05)
        super().__init__(**kwargs)

        self._origin = origin
        self.direction = np.asarray(direction, dtype=np.float32)
        n = np.linalg.norm(self.direction)
        if n > 1e-9:
            self.direction = self.direction / n
        self.length = length

    def _resolve(self):
        o = _resolve_position(self._origin)
        return o, self.direction, self.length


class LabeledPoint(MarkerObject):
    """
    Wireframe sphere with a billboard coordinate / text label.

    Construct from:
        - position tuple: ``LabeledPoint((1, 2, 3))``
        - position + label: ``LabeledPoint((1, 2, 3), label="spawn")``
        - asset vertex: ``LabeledPoint.from_vertex(asset, 42)``
        - object with .position: ``LabeledPoint(agent, label="agent")``
    """

    def __init__(self,
            source=(0, 0, 0),
            label: Optional[str] = None,
            coord_precision: int = 2,
            **kwargs
        ):
        kwargs.setdefault('color', (1.0, 0.9, 0.2))
        super().__init__(**kwargs)
        self.source = source
        self._label = label
        self.coord_precision = coord_precision

    @classmethod
    def from_vertex(cls,
                    source,
                    vertex_index: int,
                    **kwargs) -> "LabeledPoint":
        """
        Create a labelled point at a specific vertex.

        Accepts:
            Instance: uses .asset geometry and applies .transform to get world position
            Asset: uses geometry directly (local space, no transform applied)
        """
        if hasattr(source, 'asset'):
            # Instance: unwrap to Asset + transform
            asset = source.asset
            transform = getattr(source, 'transform', None)
        else:
            # Asset (or anything with .vertices)
            asset = source
            transform = None

        verts = np.asarray(asset.vertices, dtype=np.float32)
        stride = getattr(asset, 'vertex_stride', 3)

        if verts.ndim == 1:
            base = vertex_index * stride
            local = glm.vec3(float(verts[base]), float(verts[base + 1]), float(verts[base + 2]))
        else:
            local = glm.vec3(*verts[vertex_index, :3])

        if transform is not None:
            world = glm.vec3((transform * glm.vec4(local, 1.0)).xyz)
        else:
            world = local

        label = kwargs.pop('label', f"v{vertex_index}")
        return cls(source=world, label=label, **kwargs)

    def _resolve(self):
        pos = _resolve_position(self.source)

        if self._label is not None:
            text = self._label
        else:
            p = self.coord_precision
            text = f"X:{pos.x:.{p}f} Y:{pos.y:.{p}f} Z:{pos.z:.{p}f}"
        return pos, text


class AxesGizmo(DebugDrawable):

    def __init__(self,
            size: float = 0.3,
            position: Sequence[float] = None,
            track=None,
            offset: Sequence[float] = (0, 0, 0),
            **kwargs
        ):
        super().__init__(**kwargs)

        self.size = size
        self._position = glm.vec3(position) if position is not None else glm.vec3(0)
        self.track = track
        self.offset = glm.vec3(offset)

    def draw(self, r: DebugRenderer):
        if not self.visible:
            return

        origin = _resolve_position(self.track) + self.offset if self.track else self._position + self.offset
        lines, tris = dp.make_gizmo(self.size, origin=_as_tuple(origin))

        r.submit_lines(lines, alpha=self.alpha, line_width=2.5)
        r.submit_tris(tris, alpha=self.alpha)