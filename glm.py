import numpy as np


def perspective_mat(fov, aspect_ratio, near_plane, far_plane):

    num = 1.0 / np.tan(fov / 2.0)
    num9 = num / aspect_ratio

    return np.array([
        [num9, 0.0, 0.0, 0.0],
        [0.0, num, 0.0, 0.0],
        [0.0, 0.0, far_plane / (near_plane - far_plane), -1.0],
        [0.0, 0.0, (near_plane * far_plane) / (near_plane - far_plane), 0.0]
    ], dtype=np.float32)


def lookat_mat(eye_pos, target_pos, up_vector):
    eye_pos = np.asarray(eye_pos, dtype=np.float32)[:3]
    target_pos = np.asarray(target_pos, dtype=np.float32)[:3]
    up_vector = np.asarray(up_vector, dtype=np.float32)[:3]

    vect = target_pos - eye_pos
    vect /= np.linalg.norm(vect)

    vect2 = np.cross(up_vector, vect)
    vect2 /= np.linalg.norm(vect2)

    vect3 = np.cross(vect2, vect)

    mat = np.array([
        [vect2[0], vect3[0], vect[0], 0.0],
        [vect2[1], vect3[1], vect[1], 0.0],
        [vect2[2], vect3[2], vect[2], 0.0],
        [-np.dot(vect2, eye_pos), -np.dot(vect3, eye_pos), np.dot(vect, eye_pos), 1.0]
    ])
    return mat


def translation_mat(vector):
    vector = np.asarray(vector, dtype=np.float32)[:3]
    mat = np.eye(4, dtype=np.float32)
    mat[3, :3] = vector[:3]
    return mat


def translate(matrix, vector):
    return translation_mat(vector) @ matrix

def scaling_mat(vector):
    vector = np.asarray(vector, dtype=np.float32)[:3]
    x, y, z = vector
    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]], dtype=np.float32)


def scale(matrix, vector):
    return scaling_mat(vector) @ matrix

def rotation_mat(angle, axis_vector):
    axis_vector = np.asarray(axis_vector, dtype=np.float32)
    axis_vector /= np.linalg.norm(axis_vector)

    x, y, z = axis_vector
    s = np.sin(angle)
    c = np.cos(angle)

    nc = 1 - c
    return np.array([
        [x*x*nc +   c, x*y*nc - z*s, x*z*nc + y*s, 0],
        [y*x*nc + z*s, y*y*nc +   c, y*z*nc - x*s, 0],
        [x*z*nc - y*s, y*z*nc + x*s, z*z*nc +   c, 0],
        [           0,            0,            0, 1]], dtype=np.float32)


def rotate(matrix, angle, axis_vector):
    return rotation_mat(angle, axis_vector) @ matrix