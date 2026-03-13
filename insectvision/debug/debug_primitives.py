import numpy as np
from typing import Sequence, Tuple


def _arrow_basis(direction):

    d = np.asarray(direction, dtype=np.float32)
    norm = np.linalg.norm(d)

    if norm < 1e-9:
        d = np.array([0, 0, -1], dtype=np.float32)
        norm = 1.0
    d = d / norm

    arbitrary = np.array([0, 1, 0], dtype=np.float32)
    if abs(np.dot(d, arbitrary)) > 0.99:
        arbitrary = np.array([1, 0, 0], dtype=np.float32)
    perp1 = np.cross(d, arbitrary)
    perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(d, perp1)
    return d, perp1, perp2


def make_arrow(
        length: float = 1.0,
        color: Sequence[float] = (1, 1, 1),
        direction: Sequence[float] = (0, 0, -1),
        origin: Sequence[float] = (0, 0, 0),
        head_fraction: float = 0.22,
        head_radius: float = 0.02,
        cone_segments: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:

    d, perp1, perp2 = _arrow_basis(direction)
    o = np.asarray(origin, dtype=np.float32)
    tip = o + d * length
    neck = o + d * length * (1.0 - head_fraction)
    c = list(color)

    # shaft (GL_LINES)
    shaft = np.array([list(o) + c, list(neck) + c], dtype=np.float32).flatten()

    # head (GL_TRIANGLES)
    angles = np.linspace(0, 2 * np.pi, cone_segments + 1)[:-1]
    ring = np.empty((cone_segments, 3), dtype=np.float32)
    for i, a in enumerate(angles):
        ring[i] = neck + (perp1 * np.cos(a) + perp2 * np.sin(a)) * head_radius

    tris = []
    tip_c = list(tip) + c
    neck_c = list(neck) + c
    for i in range(cone_segments):
        j = (i + 1) % cone_segments
        ri = list(ring[i]) + c
        rj = list(ring[j]) + c
        tris += [tip_c, ri, rj]
        tris += [neck_c, rj, ri]

    cone = np.array(tris, dtype=np.float32).flatten()
    return shaft, cone


def make_gizmo(size: float = 0.3, origin: Sequence[float] = (0, 0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    lines, tris = [], []
    for color, direction in [
        ((1, 0.2, 0.2), (1, 0, 0)),
        ((0.2, 1, 0.2), (0, 1, 0)),
        ((0.3, 0.5, 1), (0, 0, -1))
    ]:
        shaft, cone = make_arrow(size, color, direction, origin)
        lines.append(shaft)
        tris.append(cone)
    return np.concatenate(lines), np.concatenate(tris)


def make_grid(
        size: float = 10.0,
        step: float = 1.0,
        color: Sequence[float] = (0.35, 0.35, 0.35),
        x_col: Sequence[float] = (0.55, 0.2, 0.2),
        z_col: Sequence[float] = (0.2, 0.2, 0.55),
        y: float = 0.0
    ) -> np.ndarray:

    half = size / 2.0
    divisions = int(np.ceil(size / step))

    lines = []
    for i in range(divisions + 1):
        t = -half + i * step

        # parallel to Z (varies along X)
        c = list(x_col) if abs(t) < step * 0.01 else list(color)
        lines += [list([t, y, -half]) + c, list([t, y, half]) + c]

        # parallel to X (varies along Z)
        c = list(z_col) if abs(t) < step * 0.01 else list(color)
        lines += [list([-half, y, t]) + c, list([half, y, t]) + c]

    return np.array(lines, dtype=np.float32).flatten()


def make_wire_sphere(
        radius: float = 0.03,
        segments: int = 16,
        color: Sequence[float] = (1, 1, 0)
    ) -> np.ndarray:

    c = list(color)
    lines = []
    angles = np.linspace(0, 2 * np.pi, segments + 1)
    for i in range(segments):
        a0, a1 = angles[i], angles[i + 1]
        cos0, sin0 = np.cos(a0) * radius, np.sin(a0) * radius
        cos1, sin1 = np.cos(a1) * radius, np.sin(a1) * radius

        # XY circle
        lines += [[cos0, sin0, 0] + c, [cos1, sin1, 0] + c]
        # XZ circle
        lines += [[cos0, 0, sin0] + c, [cos1, 0, sin1] + c]
        # YZ circle
        lines += [[0, cos0, sin0] + c, [0, cos1, sin1] + c]

    return np.array(lines, dtype=np.float32).flatten()


def make_wire_box(half_extents: Sequence[float] = (0.5, 0.5, 0.5), color: Sequence[float] = (1, 1, 0)) -> np.ndarray:

    hx, hy, hz = half_extents
    c = list(color)

    corners = [
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # back face
        (4,5), (5,6), (6,7), (7,4),  # front face
        (0,4), (1,5), (2,6), (3,7),  # edges
    ]
    lines = []
    for a, b in edges:
        lines += [corners[a] + c, corners[b] + c]
    return np.array(lines, dtype=np.float32).flatten()


def make_frustum_lines(
        fov_deg: float = 50.0,
        aspect: float = 16/9,
        near: float = 0.1,
        far: float = 1.0,
        color: Sequence[float] = (0.8, 0.8, 0.2)
    ) -> np.ndarray:

    half_v = np.tan(np.radians(fov_deg) / 2)
    half_h = half_v * aspect
    c = list(color)

    def rect(d):
        h, v = half_h * d, half_v * d
        return [[-h, -v, -d], [h, -v, -d], [h, v, -d], [-h, v, -d]]

    n = rect(near)
    f = rect(far)

    lines = []
    # near and far rectangles
    for rect_pts in (n, f):
        for i in range(4):
            lines += [rect_pts[i] + c, rect_pts[(i+1) % 4] + c]
    # edges
    for i in range(4):
        lines += [n[i] + c, f[i] + c]

    return np.array(lines, dtype=np.float32).flatten()


# 7‑segment digits
#
# segments numbered like:
#    ─ 0 ─
#   |     |
#   5     1
#   |     |
#    ─ 6 ─
#   |     |
#   4     2
#   |     |
#    ─ 3 ─
#
# dot = segment 7

_SEGMENT_MAP = {
    '0': (0, 1, 2, 3, 4, 5),
    '1': (1, 2),
    '2': (0, 1, 6, 4, 3),
    '3': (0, 1, 6, 2, 3),
    '4': (5, 6, 1, 2),
    '5': (0, 5, 6, 2, 3),
    '6': (0, 5, 4, 3, 2, 6),
    '7': (0, 1, 2),
    '8': (0, 1, 2, 3, 4, 5, 6),
    '9': (0, 1, 2, 3, 5, 6),
    '-': (6,),
    '.': (7,),
    ' ': (),
    'X': (1, 2, 4, 5, 6),
    'Y': (1, 2, 5, 6, 3),
    'Z': (0, 1, 6, 4, 3),  # same as '2' but that's fine
    ':': (7,),
}

_SEGMENT_COORDS = {
    0: ((0, 2), (1, 2)),  # top
    1: ((1, 2), (1, 1)),  # top right
    2: ((1, 1), (1, 0)),  # bottom right
    3: ((0, 0), (1, 0)),  # bottom
    4: ((0, 0), (0, 1)),  # bottom left
    5: ((0, 1), (0, 2)),  # top left
    6: ((0, 1), (1, 1)),  # middle
    7: ((0.4, -0.3), (0.6, -0.3)),  # dot
}


def make_text_lines(
        text: str,
        char_width: float = 0.012,
        color: Sequence[float] = (0.9, 0.9, 0.9)
    ) -> np.ndarray:

    c = list(color)
    lines = []
    cursor_x = 0.0
    w = char_width
    kerning = 1.4

    for ch in text.upper():
        segs = _SEGMENT_MAP.get(ch, ())
        for seg_id in segs:
            (x0, y0), (x1, y1) = _SEGMENT_COORDS[seg_id]
            lines.append([cursor_x + x0 * w, y0 * w, 0] + c)
            lines.append([cursor_x + x1 * w, y1 * w, 0] + c)
        cursor_x += w * kerning

    if not lines:
        return np.zeros(0, dtype=np.float32)

    return np.array(lines, dtype=np.float32).flatten()


def make_line(
        start: Sequence[float],
        end: Sequence[float],
        color: Sequence[float] = (1, 1, 1)
    ) -> np.ndarray:

    c = list(color)
    return np.array([list(start) + c, list(end) + c], dtype=np.float32).flatten()


def make_polyline(
        points: Sequence[Sequence[float]],
        color: Sequence[float] = (1, 0.6, 0),
        closed: bool = False
    ) -> np.ndarray:

    c = list(color)
    lines = []
    pts = list(points)

    n = len(pts)
    for i in range(n - 1):
        lines += [list(pts[i]) + c, list(pts[i + 1]) + c]

    if closed and n > 2:
        lines += [list(pts[-1]) + c, list(pts[0]) + c]

    return np.array(lines, dtype=np.float32).flatten()