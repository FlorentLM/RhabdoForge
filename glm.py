import numpy as np

##

# TODO - create datatypes for vect2 vect3 mat3 and mat4


def normalise_vector(vector):
    return vector / np.linalg.norm(vector)


class Matrices:

    @staticmethod
    def perspective(fov, aspect_ratio, near_plane, far_plane):

        num = 1.0 / np.tan(fov / 2.0)
        num9 = num / aspect_ratio

        return np.array([
            [num9, 0.0, 0.0, 0.0],
            [0.0, num, 0.0, 0.0],
            [0.0, 0.0, far_plane / (near_plane - far_plane), -1.0],
            [0.0, 0.0, (near_plane * far_plane) / (near_plane - far_plane), 0.0]], dtype=np.float32)

    @staticmethod
    def lookat(eye_pos, target_pos, up_vector):
        eye_pos = np.asarray(eye_pos, dtype=np.float32)[:3]
        target_pos = np.asarray(target_pos, dtype=np.float32)[:3]
        up_vector = np.asarray(up_vector, dtype=np.float32)[:3]

        v1 = normalise_vector(target_pos - eye_pos)
        v2 = normalise_vector(np.cross(up_vector, v1))
        v3 = normalise_vector(np.cross(v2, v1))

        return np.array([
            [v2[0],                   v3[0],                   v1[0],                  0.0],
            [v2[1],                   v3[1],                   v1[1],                  0.0],
            [v2[2],                   v3[2],                   v1[2],                  0.0],
            [-np.dot(v2, eye_pos),    -np.dot(v3, eye_pos),    np.dot(v1, eye_pos),    1.0]], dtype=np.float32)

    @staticmethod
    def translation(vector):
        vector = np.asarray(vector, dtype=np.float32)[:3]
        mat = np.eye(4, dtype=np.float32)
        mat[3, :3] = vector[:3]
        return mat

    @staticmethod
    def scaling(vector):
        vector = np.asarray(vector, dtype=np.float32)[:3]
        x, y, z = vector
        return np.array([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1]], dtype=np.float32)

    @staticmethod
    def rotation(angle, axis_vector):

        x, y, z = normalise_vector(np.asarray(axis_vector, dtype=np.float32))
        s = np.sin(angle)
        c = np.cos(angle)

        nc = 1 - c
        return np.array([
            [x*x*nc + c,      x*y*nc - z*s,    x*z*nc + y*s,    0],
            [y*x*nc + z*s,    y*y*nc + c,      y*z*nc - x*s,    0],
            [x*z*nc - y*s,    y*z*nc + x*s,    z*z*nc + c,      0],
            [0,               0,               0,               1]], dtype=np.float32)

##


def translate(matrix, vector):
    return Matrices.translation(vector) @ matrix


def scale(matrix, vector):
    return Matrices.scaling(vector) @ matrix


def rotate(matrix, angle, axis_vector):
    return Matrices.rotation(angle, axis_vector) @ matrix