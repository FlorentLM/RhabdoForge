"""
Replica of model from Kemppainen et al., 2022 (10.1073/pnas.2109717119)
"""
import numpy as np
from scipy.spatial import cKDTree as KDTree
from insectvision.lattice_fitting.plots import plot_eyes_3d


# Parameters from original code:
EYE_RADIUS = 1.1 * 1000 * (400 * (0.8 / 985)) / 2           # R_z ~ 178.68 µm
EYE_HORIZONTAL_R = 1.1 * 1000 * (470 * (0.8 / 985)) / 2     # R_wide ~ 209.95 µm
R_OMMATIDIA = 8.0
EYE_LOWER_ANGLE = np.radians(60)
EYE_HEAD_CP_DISTANCE = 193.4
OMMATIDIA_LIMIT = 800


# TODO: These three should go to the utils module, but need to deal with the various reference frames

def cartesian_to_spherical_kemppainen(pts):
    """
    Cartesian to spherical (internal coordinate system for eye construction).
    """
    r = np.linalg.norm(pts, axis=1)
    theta = np.arccos(np.clip(pts[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    return r, theta, phi


def spherical_to_cartesian_kemppainen(r, theta, phi):
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
    _, otheta, ophi = cartesian_to_spherical_kemppainen(center_pt[None, :])
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


def build_kemppainen_data():
    """
    Build a single eye in internal coordinate system (the right one).
    Returns ommatidia locations in µm.
    """
    start_p = spherical_to_cartesian_kemppainen(np.array([EYE_RADIUS]), np.array([np.pi / 2]), np.array([0.0]))[0]
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
    r_vals, theta_vals, phi_vals = cartesian_to_spherical_kemppainen(points_arr)
    actual_radii = EYE_HORIZONTAL_R - (EYE_HORIZONTAL_R - EYE_RADIUS) * np.abs(np.sin(theta_vals))
    locations = spherical_to_cartesian_kemppainen(actual_radii, theta_vals, phi_vals)

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


if __name__ == "__main__":

    SHOW_PLOTS = True

    # Build right eye in internal coordinate system
    positions = build_kemppainen_data()

    # Mirror for left eye
    left_eye_internal = positions.copy()
    left_eye_internal[:, 0] *= -1

    offset = EYE_HEAD_CP_DISTANCE

    positions[:, 0] += offset
    left_eye_internal[:, 0] -= offset

    right_eye = positions.copy()
    right_eye[:, 2] = -positions[:, 2]

    left_eye = left_eye_internal.copy()
    left_eye[:, 2] = -left_eye_internal[:, 2]

    # Compute directions (pointing from eye centers to ommatidium)
    right_center = np.array([offset, 0, 0])
    left_center = np.array([-offset, 0, 0])

    R_dirs = right_eye - right_center
    R_dirs /= np.linalg.norm(R_dirs, axis=1, keepdims=True)

    L_dirs = left_eye - left_center
    L_dirs /= np.linalg.norm(L_dirs, axis=1, keepdims=True)

    all_directions = np.vstack([R_dirs, L_dirs])
    all_positions = np.vstack([right_eye, left_eye])
    eye_ids = np.concatenate([np.ones(len(right_eye)), np.zeros(len(left_eye))])

    np.savez_compressed(
        'species_models/drosophila_Kemppainen.npz',
        directions=all_directions,
        positions=all_positions,
        eye_id=eye_ids
    )

    if SHOW_PLOTS:
        plot_eyes_3d(all_positions, all_directions, eye_ids,
                     title='Drosophila eyes (from Kemppainen et al., 2022)')