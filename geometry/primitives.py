import numpy as np
from graphics.utils import VEC_DTYPE


CUBE_VERTICES = np.array((
    # X, Y, Z, U, V
    # Position           # UV Coords
    # bottom
    -1.0, -1.0, -1.0,    0.0, 0.0,
     1.0, -1.0, -1.0,    1.0, 0.0,
    -1.0, -1.0,  1.0,    0.0, 1.0,
     1.0, -1.0, -1.0,    1.0, 0.0,
     1.0, -1.0,  1.0,    1.0, 1.0,
    -1.0, -1.0,  1.0,    0.0, 1.0,
    # top
    -1.0,  1.0, -1.0,    0.0, 0.0,
    -1.0,  1.0,  1.0,    0.0, 1.0,
     1.0,  1.0, -1.0,    1.0, 0.0,
     1.0,  1.0, -1.0,    1.0, 0.0,
    -1.0,  1.0,  1.0,    0.0, 1.0,
     1.0,  1.0,  1.0,    1.0, 1.0,
    # front
    -1.0, -1.0,  1.0,    1.0, 0.0,
     1.0, -1.0,  1.0,    0.0, 0.0,
    -1.0,  1.0,  1.0,    1.0, 1.0,
     1.0, -1.0,  1.0,    0.0, 0.0,
     1.0,  1.0,  1.0,    0.0, 1.0,
    -1.0,  1.0,  1.0,    1.0, 1.0,
    # back
    -1.0, -1.0, -1.0,    0.0, 0.0,
    -1.0,  1.0, -1.0,    0.0, 1.0,
     1.0, -1.0, -1.0,    1.0, 0.0,
     1.0, -1.0, -1.0,    1.0, 0.0,
    -1.0,  1.0, -1.0,    0.0, 1.0,
     1.0,  1.0, -1.0,    1.0, 1.0,
    # left
    -1.0, -1.0,  1.0,    0.0, 1.0,
    -1.0,  1.0, -1.0,    1.0, 0.0,
    -1.0, -1.0, -1.0,    0.0, 0.0,
    -1.0, -1.0,  1.0,    0.0, 1.0,
    -1.0,  1.0,  1.0,    1.0, 1.0,
    -1.0,  1.0, -1.0,    1.0, 0.0,
    # right
     1.0, -1.0,  1.0,    1.0, 1.0,
     1.0, -1.0, -1.0,    1.0, 0.0,
     1.0,  1.0, -1.0,    0.0, 0.0,
     1.0, -1.0,  1.0,    1.0, 1.0,
     1.0,  1.0, -1.0,    0.0, 0.0,
     1.0,  1.0,  1.0,    0.0, 1.0
), dtype=VEC_DTYPE)


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
