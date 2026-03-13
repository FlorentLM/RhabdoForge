import numpy as np

# TODO: these functions could be cleaned up a bit


def cone_vertices(radius=1.0, height=1.0, segments=16):
    """
    Generates vertices for a cone (default oriented for the depth-buffer Voronoi diagrams)
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

    # Assemble triangles for drawing with glDrawArrays:
    # each triangle is (apex, base_vertex_i, base_vertex_i+1)
    triangle_strip = []
    for i in range(segments):
        triangle_strip.append(apex)
        triangle_strip.append(base_verts[i])
        triangle_strip.append(base_verts[(i + 1) % segments])  # modulo to close the loop

    return np.array(triangle_strip, dtype=np.float32).flatten()


def hemisphere_vertices(stacks=8, sectors=16):
    """
    Generates vertices for a unit hemisphere mesh (triangles).
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

            v1 = i * (sectors + 1) + j
            v2 = v1 + 1
            v3 = (i + 1) * (sectors + 1) + j
            v4 = v3 + 1

            # First triangle of the quad
            triangle_indices.extend([vertices[v1], vertices[v3], vertices[v2]])
            # Second triangle of the quad
            triangle_indices.extend([vertices[v2], vertices[v3], vertices[v4]])

    return np.array(triangle_indices, dtype=np.float32).flatten()


def sphere_vertices(stacks=16, sectors=32):
    """
    Generates vertices for a unit sphere mesh (triangles).
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


def estimate_lod(n_vertices: int) -> int:
    """
    LoD for icosphere subdivision to approximate `nb_vertices`.
    """
    if n_vertices < 12:
        return 1
    return int(np.round(np.sqrt((n_vertices - 2) / 10.0)))


def icosahedron_faces() -> np.ndarray:
    """
    Base z-axis-aligned icosahedron.
    Returns (20, 3, 3) face vertices.
    """

    G = (1 + np.sqrt(5)) / 2.0

    p = np.array([
        [G, -G, -G, G, 1, 1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, G, -G, -G, G, 1, 1, -1, -1],
        [1, 1, -1, -1, 0, 0, 0, 0, G, -G, -G, G]
    ], dtype=np.float32).T

    p /= np.linalg.norm(p[0])
    ang = np.arctan(p[0, 0] / p[0, 2])

    ca, sa = np.cos(ang), np.sin(ang)
    rot = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
    p = np.inner(rot, p).T
    p = p[[0, 3, 4, 8, -1, 5, -2, -3, 7, 1, 6, 2]]

    tri = np.array([
        [1, 2, 3, 4, 5, 6, 2, 7, 2, 8, 3, 9, 10, 10, 6, 6, 7, 8, 9, 10],
        [2, 3, 4, 5, 1, 7, 1, 8, 8, 9, 9, 10, 5, 6, 1, 11, 11, 11, 11, 11],
        [0, 0, 0, 0, 0, 1, 7, 2, 3, 3, 4, 4, 4, 5, 5, 7, 8, 9, 10, 6]
    ]).T
    return p[tri]


def barycentric_coords(n_subdiv: int) -> np.ndarray:
    """
    Barycentric coordinates for subdivided reference triangle.
    """

    vals = np.linspace(0, 1, n_subdiv + 1)
    num = int((n_subdiv + 1) * (n_subdiv + 2) / 2)
    bc = np.zeros((num, 3))

    shifts = np.arange(n_subdiv + 1, 0, -1)
    starts = np.zeros(n_subdiv + 1, dtype=int)
    starts[1:] = np.cumsum(shifts[:-1])
    stops = starts + shifts

    for i, (s, e, sh) in enumerate(zip(starts, stops, shifts)):
        bc[s:e, 0] = vals[sh - 1::-1]
        bc[s:e, 1] = vals[:sh]
        bc[s:e, 2] = vals[i]
    return bc


def subdivide_icosahedron(n_subdiv: int) -> np.ndarray:
    """
    Subdivide icosahedron via barycentric interpolation onto unit sphere.
    """

    verts = icosahedron_faces()
    bary = barycentric_coords(n_subdiv)

    all_v = np.einsum('ij,kjl->kil', bary, verts).reshape(-1, 3)
    all_v /= np.linalg.norm(all_v, axis=1)[:, np.newaxis]
    _, iu = np.unique(np.round(all_v, 6), axis=0, return_index=True)

    return all_v[iu].astype(np.float32)


def fibonacci_sphere(samples: int) -> np.ndarray:
    """
    Uniform points on unit sphere (Fibonacci method).
    """

    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(samples)
    y = 1 - (i / float(samples - 1)) * 2
    r = np.sqrt(1 - y * y)
    theta = phi * i

    return np.column_stack([np.cos(theta) * r, y, np.sin(theta) * r])
