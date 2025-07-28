import numpy as np
import numba
from graphics.utils import VEC_DTYPE, WORLD_RIGHT, WORLD_UP


@numba.jit(nopython=True, cache=True)
def perspective_mat(fov_rad, aspect_ratio, near_plane, far_plane):

    f = 1.0 / np.tan(fov_rad / 2.0)

    return np.array([
        [f / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far_plane + near_plane) / (near_plane - far_plane),
         (2.0 * far_plane * near_plane) / (near_plane - far_plane)],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=VEC_DTYPE)


@numba.jit(nopython=True, cache=True)
def lookat_mat(eye_pos, target_pos, up_vector):
    eye = np.asarray(eye_pos, dtype=VEC_DTYPE)
    target = np.asarray(target_pos, dtype=VEC_DTYPE)
    up = np.asarray(up_vector, dtype=VEC_DTYPE)

    # Calculate camera's local coordinate system basis vectors
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return np.eye(4, dtype=VEC_DTYPE)
    forward /= forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)

    if right_norm < 1e-6:
        # 'up' was no good so use different axis to calculate the 'right' vector
        right = np.cross(WORLD_RIGHT, forward)
        # This can still fail if forward is also aligned with WORLD_RIGHT but that's extremely unlikely
        # ultimate fallback is to use WORLD_UP
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(WORLD_UP, forward)
        right /= np.linalg.norm(right)

    # Recalculate the true 'up' vector to ensure orthogonality
    up = np.cross(right, forward)

    # The view matrix is T_inv * R_inv
    rotation_inverse = np.array([
        [right[0],    right[1],    right[2],   0.0],
        [up[0],       up[1],       up[2],      0.0],
        [-forward[0], -forward[1], -forward[2], 0.0],
        [0.0,         0.0,         0.0,        1.0]
    ], dtype=VEC_DTYPE)

    translation_inverse = translation_mat(-eye)

    # Final matrix is Translation_inverse * Rotation_inverse
    return rotation_inverse @ translation_inverse


@numba.jit(nopython=True, cache=True)
def translation_mat(vector):
    mat = np.eye(4, dtype=VEC_DTYPE)
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
    ], dtype=VEC_DTYPE)


@numba.jit(nopython=True, cache=True)
def rotation_mat(angle_rad, axis_vector):
    axis_norm = np.linalg.norm(axis_vector)

    if axis_norm < 1e-6:
        return np.eye(4, dtype=VEC_DTYPE)
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
    ], dtype=VEC_DTYPE)


@numba.jit(nopython=True, cache=True)
def translate(matrix, vector):
    return matrix @ translation_mat(vector)


@numba.jit(nopython=True, cache=True)
def scale(matrix, vector):
    return matrix @ scaling_mat(vector)


@numba.jit(nopython=True, cache=True)
def rotate(matrix, angle_rad, axis_vector):
    return matrix @ rotation_mat(angle_rad, axis_vector)