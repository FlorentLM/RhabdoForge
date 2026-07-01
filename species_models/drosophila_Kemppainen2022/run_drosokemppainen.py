"""
Replica of model from Kemppainen et al., 2022 (10.1073/pnas.2109717119)
"""
import numpy as np
from scipy.spatial import cKDTree as KDTree

from insectvision.lattice_fitting.algo import mirror_bilateral
from insectvision.lattice_fitting.plots import plot_eye_scaffold_3d


# Parameters from original Kemppainen code:
# https://github.com/JuusolaLab/Hyperacute_Stereopsis_paper/blob/main/CG-Compound-Eye/model_init.py

EYE_RADIUS = 1.1 * 1000 * (400 * (0.8 / 985)) / 2           # Main radius of the eye ~178.68 µm
EYE_HORIZONTAL_R = 1.1 * 1000 * (470 * (0.8 / 985)) / 2     # Eye radius in (x, y) plane ~209.95 µm
# TODO: Actually why these constants again?

R_OMMATIDIA = 8.0
EYE_LOWER_ANGLE = np.deg2rad(60.0)      # Maximum OA angle in coronal plane from top (rad)
EYE_HEAD_CP_DISTANCE = 193.4            # Distance between L and R eye inner corners (µm)
OMMATIDIA_LIMIT = 800                   # maximum number of ommatidia


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


def _bfs_hex_neighbours(center_pt):
    """
    Generates 6 neighbour candidates for a point during the BFS algorithm.
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
    Build a single eye (right eye) in Blender coordinate system (as in Kemppainen's code).
    Returns ommatidia positions (in µm).
    """
    start_p = spherical_to_cartesian_kemppainen(np.array([EYE_RADIUS]), np.array([np.pi / 2]), np.array([0.0]))[0]
    points = [start_p]

    # Initiate 'star' (6 main branches)
    for i_dir in range(6):
        curr_p = start_p
        for _ in range(50):
            candidates = _bfs_hex_neighbours(curr_p)
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
            all_candidates.append(_bfs_hex_neighbours(p))

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

    base_positions = build_kemppainen_data()
    base_positions[:, 2] *= -1

    # Direction = outward radial vector from the (origin-centred) generation sphere
    base_directions = base_positions / np.linalg.norm(base_positions, axis=1, keepdims=True)

    positions_both, directions_both, eye_ids_both = mirror_bilateral(
        positions=base_positions,
        directions=base_directions,
        shift=EYE_HEAD_CP_DISTANCE,
        source_side='right'
    )

    n_right = int(eye_ids_both.sum())
    print(f"Final model:  L={len(positions_both) - n_right}  R={n_right}")

    np.savez_compressed(
        'species_models/drosophila_Kemppainen.npz',
        positions=positions_both,
        directions=directions_both,
        eye_id=eye_ids_both
    )

    if SHOW_PLOTS:
        plot_eye_scaffold_3d(
            positions=positions_both,
            directions=directions_both,
            eye_ids=eye_ids_both,
            title='Drosophila eyes (from Kemppainen et al., 2022)'
        )