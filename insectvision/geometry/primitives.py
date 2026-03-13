import numpy as np


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

), dtype=np.float32)

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
    apex = np.array([0.0, 0.0, -height], dtype=np.float32)

    # Base vertices should have a LARGER Z value
    base_verts = []
    base_z = 0.0
    for i in range(segments):
        angle = (i / segments) * 2.0 * np.pi
        x = np.cos(angle) * radius
        y = np.sin(angle) * radius
        base_verts.append(np.array([x, y, base_z], dtype=np.float32))

    # Assemble triangles for drawing with glDrawArrays
    # Each triangle is (apex, base_vertex_i, base_vertex_i+1)
    triangle_strip = []
    for i in range(segments):
        triangle_strip.append(apex)
        triangle_strip.append(base_verts[i])
        triangle_strip.append(base_verts[(i + 1) % segments])  # modulo to close the loop

    return np.array(triangle_strip, dtype=np.float32).flatten()


CONE_VERTICES = create_cone_data(radius=1.0, height=1.0, segments=12)


def create_hemisphere_vertices(stacks=8, sectors=16):
    """
    Generates vertices for a unit hemisphere mesh made of triangles.
    The hemisphere is oriented with its base on the XY plane and capping along the positive Z axis.
    """
    vertices = []

    # Generate vertices
    for i in range(stacks + 1):
        stack_angle = np.pi / 2.0 * (i / stacks)
        z = np.sin(stack_angle)
        radius = np.cos(stack_angle)

        for j in range(sectors + 1):
            sector_angle = 2 * np.pi * (j / sectors)
            x = radius * np.cos(sector_angle)
            y = radius * np.sin(sector_angle)
            vertices.append([x, y, z])

    triangle_indices = []
    for i in range(stacks):
        for j in range(sectors):
            # Define the four vertices of a quad
            v1 = i * (sectors + 1) + j
            v2 = v1 + 1
            v3 = (i + 1) * (sectors + 1) + j
            v4 = v3 + 1

            # First triangle of the quad
            triangle_indices.extend([vertices[v1], vertices[v3], vertices[v2]])
            # Second triangle of the quad
            triangle_indices.extend([vertices[v2], vertices[v3], vertices[v4]])

    return np.array(triangle_indices, dtype=np.float32).flatten()

HEMISPHERE_VERTICES = create_hemisphere_vertices()


def create_sphere_vertices(stacks=16, sectors=32):
    """
    Generates vertices for a unit sphere mesh made of triangles.
    """
    vertices = []

    # Generate vertices
    for i in range(stacks + 1):
        stack_angle = np.pi * (i / stacks)  # From 0 to pi
        z = np.cos(stack_angle)
        radius = np.sin(stack_angle)

        for j in range(sectors + 1):
            sector_angle = 2 * np.pi * (j / sectors)  # From 0 to 2*pi
            x = radius * np.cos(sector_angle)
            y = radius * np.sin(sector_angle)
            vertices.append([x, y, z])

    triangle_indices = []
    for i in range(stacks):
        for j in range(sectors):
            # Define the four vertices of a quad
            v1 = i * (sectors + 1) + j
            v2 = v1 + 1
            v3 = (i + 1) * (sectors + 1) + j
            v4 = v3 + 1

            # First triangle of the quad
            triangle_indices.extend([vertices[v1], vertices[v3], vertices[v2]])
            # Second triangle of the quad
            triangle_indices.extend([vertices[v2], vertices[v3], vertices[v4]])

    return np.array(triangle_indices, dtype=np.float32).flatten()

SPHERE_VERTICES = create_sphere_vertices()