import numpy as np
import numba
from graphics.utils import DTYPE, WORLD_RIGHT, WORLD_UP


@numba.jit(nopython=True, cache=True)
def perspective_mat(fov_rad, aspect_ratio, near_plane, far_plane):

    f = 1.0 / np.tan(fov_rad / 2.0)

    return np.array([
        [f / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far_plane + near_plane) / (near_plane - far_plane),
         (2.0 * far_plane * near_plane) / (near_plane - far_plane)],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=DTYPE)


@numba.jit(nopython=True, cache=True)
def lookat_mat(eye_pos, target_pos, up_vector):
    eye = np.asarray(eye_pos, dtype=DTYPE)
    target = np.asarray(target_pos, dtype=DTYPE)
    up = np.asarray(up_vector, dtype=DTYPE)

    # Camera's basis vectors (its local coordinate system)
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return np.eye(4, dtype=DTYPE)   # eye and target are in the same position
    forward /= forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)

    # case where forward and up are collinear (looking straight up or down)
    if right_norm < 1e-6:
        if np.allclose(np.abs(forward), WORLD_UP):
            right = np.cross(forward, WORLD_RIGHT)
        else:
            right = np.cross(forward, WORLD_UP)
        right /= np.linalg.norm(right)
    else:
        right /= right_norm

    # Re cross to ensure perfect orthogonality
    up = np.cross(right, forward)

    rot_part = np.array([
        [right[0],  up[0],  -forward[0],  0.0],
        [right[1],  up[1],  -forward[1],  0.0],
        [right[2],  up[2],  -forward[2],  0.0],
        [     0.0,    0.0,          0.0,  1.0]
    ], dtype=DTYPE)

    trans_part = translation_mat(-eye)

    # view matrix is R_inv * T_inv
    # Note: We premultiply because the rotation should be applied first
    return rot_part @ trans_part


@numba.jit(nopython=True, cache=True)
def translation_mat(vector):
    mat = np.eye(4, dtype=DTYPE)
    # Translation is in the last column's first 3 rows
    mat[:3, 3] = vector[:3]
    return mat


@numba.jit(nopython=True, cache=True)
def scaling_mat(vector):
    return np.array([
        [ vector[0],       0.0,       0.0,      0.0],
        [       0.0, vector[1],       0.0,      0.0],
        [       0.0,       0.0, vector[2],      0.0],
        [       0.0,       0.0,       0.0,      1.0]
    ], dtype=DTYPE)


@numba.jit(nopython=True, cache=True)
def rotation_mat(angle_rad, axis_vector):
    axis_norm = np.linalg.norm(axis_vector)

    if axis_norm < 1e-6:
        return np.eye(4, dtype=DTYPE)
    axis = axis_vector / axis_norm

    x, y, z = axis
    s = np.sin(angle_rad)
    c = np.cos(angle_rad)
    nc = 1 - c

    return np.array([
        [x * x * nc + c,     y * x * nc + z * s, z * x * nc - y * s, 0.0],
        [x * y * nc - z * s, y * y * nc + c,     z * y * nc + x * s, 0.0],
        [x * z * nc + y * s, y * z * nc - x * s, z * z * nc + c,     0.0],
        [0.0,                  0.0,                  0.0,            1.0]
    ], dtype=DTYPE)


@numba.jit(nopython=True, cache=True)
def translate(matrix, vector):
    return matrix @ translation_mat(vector)


@numba.jit(nopython=True, cache=True)
def scale(matrix, vector):
    return matrix @ scaling_mat(vector)


@numba.jit(nopython=True, cache=True)
def rotate(matrix, angle_rad, axis_vector):
    return matrix @ rotation_mat(angle_rad, axis_vector)