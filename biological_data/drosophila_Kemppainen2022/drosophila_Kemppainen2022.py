import numpy as np
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt


# Generate drosophila eyes according to Kemppainen et al. 2022
#
# Parameters from original code:
#
EYE_RADIUS = 1.1 * 1000 * (400 * (0.8 / 985)) / 2  # R_z ≈ 178.68 µm
EYE_HORIZONTAL_R = 1.1 * 1000 * (470 * (0.8 / 985)) / 2  # R_wide ≈ 209.95 µm
R_OMMATIDIA = 8.0
EYE_LOWER_ANGLE = np.radians(60)
EYE_HEAD_CP_DISTANCE = 193.4
OMMATIDIA_LIMIT = 800

##

def cart_to_sph(pts):
    r = np.linalg.norm(pts, axis=1)
    theta = np.arccos(np.clip(pts[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    return r, theta, phi


def sph_to_cart(r, theta, phi):
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


def hex_neighbours(center_pt):
    """Generates 6 neighbour candidates for a point."""
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    # Local candidates in XZ plane relative to a Y-axis eye-normal
    local_nodes = np.zeros((6, 3))
    local_nodes[:, 0] = 2 * R_OMMATIDIA * np.sin(angles)
    local_nodes[:, 1] = np.linalg.norm(center_pt)  # Push out to current radius
    local_nodes[:, 2] = 2 * R_OMMATIDIA * np.cos(angles)

    return local_to_global(local_nodes, center_pt)


def build_eye_geometry():

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

        added_in_batch = []
        for i, d in enumerate(dists):
            cand = candidates_flat[i]

            # Boundary check
            if cand[0] <= 0:
                continue

            # Distance check (original logic: > 1.25 * R_omm)
            if d > R_OMMATIDIA * 1.25:
                if len(added_in_batch) > 0:
                    temp_tree = KDTree(added_in_batch)
                    d_internal, _ = temp_tree.query(cand, k=1)

                    if d_internal < R_OMMATIDIA * 1.25:
                        continue

                added_in_batch.append(cand)
                next_queue.append(cand)

        if not added_in_batch:
            break

        points_arr = np.vstack([points_arr, added_in_batch])
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
    right_eye = build_eye_geometry()

    left_eye = right_eye.copy()
    left_eye[:, 0] *= -1

    def to_gl_mm(pts):
        # Blender (X, Y, Z) -> OpenGL (X, Z, -Y)
        return np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=1) * 0.001

    r_orig = to_gl_mm(right_eye)
    l_orig = to_gl_mm(left_eye)

    offset = EYE_HEAD_CP_DISTANCE * 0.001
    r_orig[:, 0] += offset
    l_orig[:, 0] -= offset

    # Directions
    r_dir = r_orig - [offset, 0, 0]
    r_dir /= np.linalg.norm(r_dir, axis=1, keepdims=True)
    l_dir = l_orig - [-offset, 0, 0]
    l_dir /= np.linalg.norm(l_dir, axis=1, keepdims=True)

    directions = np.vstack([r_dir, l_dir])
    origins = np.vstack([r_orig, l_orig])
    eye_id = np.concatenate([np.ones(len(r_orig)), np.zeros(len(l_orig))])

    return directions, origins, eye_id



def plot_eye_model(origins, eye_id, title=''):

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    left_mask = eye_id == 0
    right_mask = eye_id == 1

    ax.scatter(origins[left_mask, 0], origins[left_mask, 1], origins[left_mask, 2],
               c='#4a90d9', s=8, alpha=0.7, label=f'Left ({left_mask.sum()})')
    ax.scatter(origins[right_mask, 0], origins[right_mask, 1], origins[right_mask, 2],
               c='#d94a4a', s=8, alpha=0.7, label=f'Right ({right_mask.sum()})')

    max_extent = np.abs(origins).max() * 1.3
    ax.quiver(0, 0, 0, max_extent*0.3, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, max_extent*0.3, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, -max_extent*0.3, color='b', arrow_length_ratio=0.1)

    ax.set_xlabel('X (mm) - Right')
    ax.set_ylabel('Y (mm) - Up')
    ax.set_zlabel('Z (mm)')
    ax.set_title(title)

    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)
    ax.set_zlim(-max_extent, max_extent)
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc='upper right')
    ax.view_init(elev=15, azim=-70)

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    import time

    print("Parameters from original code:")
    print(f"  EYE_RADIUS (R_z): {EYE_RADIUS:.2f} µm")
    print(f"  EYE_HORIZONTAL_R (R_wide): {EYE_HORIZONTAL_R:.2f} µm")
    print(f"  R_OMMATIDIA: {R_OMMATIDIA} µm")
    print(f"  EYE_LOWER_ANGLE: {np.degrees(EYE_LOWER_ANGLE):.1f}°")
    print(f"  EYE_HEAD_CP_DISTANCE: {EYE_HEAD_CP_DISTANCE:.2f} µm")
    print("(source: https://github.com/JuusolaLab/Hyperacute_Stereopsis_paper/blob/main/CG-Compound-Eye/model_init.py)")
    print()

    start = time.time()
    directions, origins, eye_id = generate_eyes()

    n_total = len(directions)
    n_left = (eye_id == 0).sum()
    n_right = (eye_id == 1).sum()

    print(f"Generated {n_total} ommatidia in {time.time() - start:.2f} seconds:")
    print(f"  Left eye: {n_left}")
    print(f"  Right eye: {n_right}")

    output_file = "drosophila_eye_Kemppainen.npz"
    np.savez_compressed(
        output_file,
        directions=directions,
        origins=origins,
        eye_id=eye_id
    )
    print(f"\nSaved to '{output_file}'")

    plot_eye_model(origins, eye_id, "Drosophila eyes\n(adapted from Kemppainen et al., 2022)")
    plt.show()
