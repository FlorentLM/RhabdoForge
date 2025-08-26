import numpy as np
from graphics.utils import VEC_DTYPE


CUBE_VERTICES = np.array((
    # Position           # UV Coords

    -1.0, -1.0, -1.0,    0.0, 0.0,  # 0
     1.0, -1.0, -1.0,    1.0, 0.0,  # 1
     1.0,  1.0, -1.0,    1.0, 1.0,  # 2
    -1.0,  1.0, -1.0,    0.0, 1.0,  # 3

    -1.0, -1.0,  1.0,    0.0, 0.0,  # 4
     1.0, -1.0,  1.0,    1.0, 0.0,  # 5
     1.0,  1.0,  1.0,    1.0, 1.0,  # 6
    -1.0,  1.0,  1.0,    0.0, 1.0,  # 7

    -1.0, -1.0, -1.0,    1.0, 0.0,  # 8
    -1.0,  1.0, -1.0,    1.0, 1.0,  # 9
    -1.0,  1.0,  1.0,    0.0, 1.0,  # 10
    -1.0, -1.0,  1.0,    0.0, 0.0,  # 11

     1.0, -1.0, -1.0,    0.0, 0.0,  # 12
     1.0,  1.0, -1.0,    0.0, 1.0,  # 13
     1.0,  1.0,  1.0,    1.0, 1.0,  # 14
     1.0, -1.0,  1.0,    1.0, 0.0,  # 15

    -1.0, -1.0, -1.0,    0.0, 1.0,  # 16
     1.0, -1.0, -1.0,    1.0, 1.0,  # 17
     1.0, -1.0,  1.0,    1.0, 0.0,  # 18
    -1.0, -1.0,  1.0,    0.0, 0.0,  # 19

    -1.0,  1.0, -1.0,    0.0, 0.0,  # 20
     1.0,  1.0, -1.0,    1.0, 0.0,  # 21
     1.0,  1.0,  1.0,    1.0, 1.0,  # 22
    -1.0,  1.0,  1.0,    0.0, 1.0,  # 23

), dtype=VEC_DTYPE)

CUBE_INDICES = np.array((
    # Back face (-Z)
    0, 3, 2,   2, 1, 0,

    # Front face (+Z)
    4, 5, 6,   6, 7, 4,

    # Left face (-X)
    11, 10, 9,   9, 8, 11,

    # Right face (+X)
    12, 13, 14,   14, 15, 12,

    # Bottom face (-Y)
    19, 16, 17,   17, 18, 19,

    # Top face (+Y)
    20, 23, 22,   22, 21, 20
), dtype=np.uint32)


def create_cone_data(radius=1.0, height=1.0, segments=16):
    """
    Creates vertex data for a cone suitable for depth-buffer Voronoi diagrams
    The apex is at Z = -height (closest to camera in clip space)
    The base is at Z = 0 (farther from camera)
    """

    # Smaller Z value for apex = 'closer'
    apex = np.array([0.0, 0.0, -height], dtype=VEC_DTYPE)

    # Base vertices should have a LARGER Z value
    base_verts = []
    base_z = 0.0
    for i in range(segments):
        angle = (i / segments) * 2.0 * np.pi
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        base_verts.append(np.array([x, y, base_z], dtype=VEC_DTYPE))

    # Assemble triangles for drawing with glDrawArrays
    # Each triangle is (apex, base_vertex_i, base_vertex_i+1)
    triangle_strip = []
    for i in range(segments):
        triangle_strip.append(apex)
        triangle_strip.append(base_verts[i])
        triangle_strip.append(base_verts[(i + 1) % segments])  # modulo to close the loop

    return np.array(triangle_strip, dtype=VEC_DTYPE).flatten()


CONE_VERTICES = create_cone_data(radius=1.0, height=1.0, segments=32)
