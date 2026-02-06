import numpy as np
from scipy.spatial import cKDTree as KDTree
from species_models.plots import plot_eyes_3d


# Replica of model from Kemppainen et al., 2022 (10.1073/pnas.2109717119).
#
# Parameters from original code:
EYE_RADIUS = 1.1 * 1000 * (400 * (0.8 / 985)) / 2           # R_z ~ 178.68 µm
EYE_HORIZONTAL_R = 1.1 * 1000 * (470 * (0.8 / 985)) / 2     # R_wide ~ 209.95 µm
R_OMMATIDIA = 8.0
EYE_LOWER_ANGLE = np.radians(60)
EYE_HEAD_CP_DISTANCE = 193.4
OMMATIDIA_LIMIT = 800


# TODO: These three should go to the utils module, but need to deal with the various reference frames

def cart_to_sph(pts):
    """
    Cartesian to spherical (internal coordinate system for eye construction).
    """
    r = np.linalg.norm(pts, axis=1)
    theta = np.arccos(np.clip(pts[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    return r, theta, phi


def sph_to_cart(r, theta, phi):
    """
    Spherical to cartesian (internal coordinate system for eye construction).
    """
    return np.stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    ], axis=1)


def local_to_global(local_pts, center_pt):
    """
    Rotates local points into global space.
    """
    _, otheta, ophi = cart_to_sph(center_pt[None, :])
    ot, op = otheta[0], ophi[0]

    # Rotation Z: ophi - pi/2
    # Rotation X: pi/2 - otheta
    rz_angle = op - np.pi / 2
    rx_angle = np.pi / 2 - ot

    cz, sz = np.cos(rz_angle), np.sin(rz_angle)
    cx, sx = np.cos(rx_angle), np.sin(rx_angle)

    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    # Global = Rz @ Rx @ Local
    return (Rz @ Rx @ local_pts.T).T


##


def hex_neighbours(center_pt):
    """
    Generates 6 neighbour candidates for a point.
    """
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    # Local candidates in XZ plane relative to a Y-axis eye-normal
    local_nodes = np.zeros((6, 3))
    local_nodes[:, 0] = 2 * R_OMMATIDIA * np.sin(angles)
    local_nodes[:, 1] = np.linalg.norm(center_pt)  # push out to current radius
    local_nodes[:, 2] = 2 * R_OMMATIDIA * np.cos(angles)

    return local_to_global(local_nodes, center_pt)


def build_eye():
    """
    Build a single eye in internal coordinate system (the right one).
    Returns ommatidia locations in µm.
    """
    start_p = sph_to_cart(np.array([EYE_RADIUS]), np.array([np.pi / 2]), np.array([0.0]))[0]
    points = [start_p]

    # Initiate 'star' (6 main branches)
    for i_dir in range(6):
        curr_p = start_p
        for _ in range(50):
            candidates = hex_neighbours(curr_p)
            next_p = candidates[i_dir]
            if next_p[0] <= 0:
                break
            points.append(next_p)
            curr_p = next_p

    # Main growth (BFS)
    queue = list(points)
    points_arr = np.array(points)

    while queue:
        tree = KDTree(points_arr)
        next_queue = []

        all_candidates = []
        for p in queue:
            all_candidates.append(hex_neighbours(p))

        candidates_flat = np.vstack(all_candidates)
        dists, _ = tree.query(candidates_flat, k=1)

        added = []
        for i, d in enumerate(dists):
            cand = candidates_flat[i]

            # Boundary check
            if cand[0] <= 0:
                continue

            # Distance check
            if d > R_OMMATIDIA * 1.25:
                if len(added) > 0:
                    temp_tree = KDTree(added)
                    d_internal, _ = temp_tree.query(cand, k=1)

                    if d_internal < R_OMMATIDIA * 1.25:
                        continue

                added.append(cand)
                next_queue.append(cand)

        if not added:
            break

        points_arr = np.vstack([points_arr, added])
        queue = next_queue

        if len(points_arr) > OMMATIDIA_LIMIT * 2:  # hard break safety
            break

    # Project onto ellipsoid surface
    r_vals, theta_vals, phi_vals = cart_to_sph(points_arr)
    actual_radii = EYE_HORIZONTAL_R - (EYE_HORIZONTAL_R - EYE_RADIUS) * np.abs(np.sin(theta_vals))
    locations = sph_to_cart(actual_radii, theta_vals, phi_vals)

    # Rotation and filtering
    rot_y = np.radians(-30)
    Ry = np.array([[np.cos(rot_y), 0, np.sin(rot_y)], [0, 1, 0], [-np.sin(rot_y), 0, np.cos(rot_y)]])
    locations = locations @ Ry.T

    # Angular cutoff
    k = -1 / np.tan(EYE_LOWER_ANGLE)
    mask = (k * locations[:, 0] < locations[:, 2]) & (locations[:, 0] > 0)
    locations = locations[mask]

    # Sort and limit
    if len(locations) > OMMATIDIA_LIMIT:
        locations = locations[np.argsort(-locations[:, 0])][:OMMATIDIA_LIMIT]

    # Center Y
    locations[:, 1] -= np.mean(locations[:, 1])
    return locations


def generate_eyes():
    """
    Generate both eyes in OpenGL coordinate system (X=right, Y=up, Z=back/into screen).

    Returns:
        directions: (N, 3) array of unit direction vectors in Cartesian coordinates
        origins: (N, 3) array of ommatidium origins in Cartesian coordinates (in mm if scale_to_mm=True)
        eye_id: (N,) array of eye identifiers (1=right, 0=left)
    """

    # Build right eye in internal coordinate system
    right_eye_internal = build_eye()

    # Mirror for left eye
    left_eye_internal = right_eye_internal.copy()
    left_eye_internal[:, 0] *= -1

    offset = EYE_HEAD_CP_DISTANCE

    right_eye_internal[:, 0] += offset
    left_eye_internal[:, 0] -= offset

    # Convert to OpenGL:
    # Internal: X = lateral, Y = up, Z = front
    # OpenGL: X = lateral(right), Y = up, Z = back
    right_eye = right_eye_internal.copy()
    right_eye[:, 2] = -right_eye_internal[:, 2]

    left_eye = left_eye_internal.copy()
    left_eye[:, 2] = -left_eye_internal[:, 2]

    # Compute directions (pointing from eye centers to ommatidium)
    right_center = np.array([offset, 0, 0])
    left_center = np.array([-offset, 0, 0])

    r_dir = right_eye - right_center
    r_dir /= np.linalg.norm(r_dir, axis=1, keepdims=True)

    l_dir = left_eye - left_center
    l_dir /= np.linalg.norm(l_dir, axis=1, keepdims=True)

    directions = np.vstack([r_dir, l_dir])
    origins = np.vstack([right_eye, left_eye])
    eye_id = np.concatenate([np.ones(len(right_eye)), np.zeros(len(left_eye))])

    origins *= 0.001

    return directions, origins, eye_id


if __name__ == "__main__":

    PLOT_EYES = True

    directions, origins, eye_id = generate_eyes()

    output_filename = "species_models/drosophila_Kemppainen.npz"
    np.savez_compressed(
        output_filename,
        directions=directions,
        origins=origins,
        eye_id=eye_id
    )

    if PLOT_EYES:
        plot_eyes_3d(origins, directions, eye_id, title='Drosophila eyes (from Kemppainen et al., 2022)')