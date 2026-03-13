import numpy as np
from insectvision.geometry.geom_utils import cone_vertices, hemisphere_vertices, sphere_vertices


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

CONE_VERTICES = cone_vertices()

HEMISPHERE_VERTICES = hemisphere_vertices()

SPHERE_VERTICES = sphere_vertices()