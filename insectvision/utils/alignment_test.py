import pyvista as pv
import numpy as np

HEAD_PITCH = 10.1
MAIN_AXIS_OFFSET = -81.0
SACCADE_AXIS_OFFSET = -28.6


def smooth_phasor_field(mesh, array_name, iterations=5):
    """Smooths a vector field treating it as a phasor (180-deg agnostic)."""

    vectors = mesh.point_data[array_name].copy()

    # Build adjacency list
    adj = [[] for _ in range(mesh.n_points)]
    for cell_idx in range(mesh.n_cells):
        cell_pts = mesh.get_cell(cell_idx).point_ids
        for p_id in cell_pts:
            adj[p_id].extend([x for x in cell_pts if x != p_id])
    adj = [list(set(neighbors)) for neighbors in adj]

    for _ in range(iterations):
        new_vectors = np.zeros_like(vectors)
        for i in range(mesh.n_points):
            neighbors = adj[i]
            if not neighbors:
                new_vectors[i] = vectors[i]
                continue

            base_v = vectors[i]
            neighbor_vs = vectors[neighbors]

            # Align neighbours to base_v (phasor logic)
            dots = np.einsum('j,ij->i', base_v, neighbor_vs)
            neighbor_vs[dots < 0] *= -1.0

            avg_v = np.mean(np.vstack([base_v, neighbor_vs]), axis=0)
            norm = np.linalg.norm(avg_v)
            new_vectors[i] = avg_v / norm if norm > 1e-8 else base_v
        vectors = new_vectors
    return vectors


def generate_eye(eye_sign, strength, n_sub, cut_angle_deg, flow_dir):
    """
    Generates an eye mesh.
    eye_sign = 1 for Left Eye, -1 for Right Eye.
    """
    sphere = pv.Icosphere(radius=1.0, nsub=n_sub)
    sphere = sphere.compute_normals(point_normals=True, cell_normals=False)

    points = sphere.points
    normals = sphere.point_data['Normals']

    # Calculate local basis vectors
    # Primary axis (forward/flow)
    e_x = flow_dir / np.linalg.norm(flow_dir)

    # Lateral axis (Right/Left)
    # cross Global Up with Flow to get a horizontal vector relative to flow
    world_up = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(e_x, world_up)) > 0.999:
        e_y = np.array([0.0, 1.0, 0.0]) if eye_sign > 0 else np.array([0.0, -1.0, 0.0])
    else:
        e_y = np.cross(world_up, e_x)
        e_y /= np.linalg.norm(e_y)

    # Vertical axis (Dorsal/Ventral)
    # This vector is the normal to the equatorial plane defined by the flow
    e_z = np.cross(e_x, e_y)
    e_z /= np.linalg.norm(e_z)

    # Calculate height relative to the Flow-Equatorial plane
    p_height = np.einsum('ij,j->i', points, e_z)
    hemisphere_sign = np.sign(p_height)
    hemisphere_sign[hemisphere_sign == 0] = 1.0

    # Chirality Logic:
    # A (Brown/ -1): Left-Ventral or Right-Dorsal
    # B (Orange/ 1): Left-Dorsal or Right-Ventral
    chirality = eye_sign * hemisphere_sign
    sphere.point_data['Chirality'] = chirality

    # Optic flow projection
    dot_p = np.dot(normals, flow_dir)
    raw_flow = flow_dir - (dot_p[:, np.newaxis] * normals)
    raw_flow_unit = raw_flow / np.linalg.norm(raw_flow, axis=1, keepdims=True).clip(min=1e-8)

    # Combing / Alignment phasors
    source_point = -1.0 * e_x
    combed_flow = np.zeros_like(raw_flow)

    for i in range(sphere.n_points):
        p, n, v_raw = points[i], normals[i], raw_flow[i]
        dist_from_source = np.linalg.norm(p - source_point)
        local_w = max(0, 1.0 - (dist_from_source * 0.7)) * strength

        # Mirror target ideal by eye_sign
        v_target_ideal = -e_x - (e_y * eye_sign) + (e_z * hemisphere_sign[i])
        v_target_proj = v_target_ideal - np.dot(v_target_ideal, n) * n
        mag = np.linalg.norm(v_raw)
        if np.linalg.norm(v_target_proj) > 1e-6:
            v_target_proj = (v_target_proj / np.linalg.norm(v_target_proj)) * mag
        else:
            v_target_proj = v_raw

        combed_flow[i] = (1.0 - local_w) * v_raw + (local_w * v_target_proj)

    sphere.point_data['RawFlow'] = raw_flow_unit
    sphere.point_data['AlignmentPhasors'] = combed_flow

    # Major axis (-81° offset)
    rot_angle_81 = np.radians(MAIN_AXIS_OFFSET * eye_sign)
    cross_nv_81 = np.cross(normals, combed_flow)
    rotated_81 = combed_flow * np.cos(rot_angle_81) + cross_nv_81 * np.sin(rot_angle_81)

    # Standardize phasor direction
    ref_binormal = np.cross(normals, combed_flow)
    dot_check = np.einsum('ij,ij->i', rotated_81, ref_binormal)
    rotated_81[dot_check < 0] *= -1.0
    sphere.point_data['MajorAxis'] = rotated_81 / np.linalg.norm(rotated_81, axis=1, keepdims=True).clip(min=1e-8)

    # Saccade axis (28.6°)
    # The saccade axis depends explicitly on chirality (mirrors across equator)
    rot_angle_28 = np.radians(SACCADE_AXIS_OFFSET * chirality)
    cross_nv_28 = np.cross(normals, sphere.point_data['MajorAxis'])
    rotated_28 = (sphere.point_data['MajorAxis'] * np.cos(rot_angle_28)[:, None]) + (
                cross_nv_28 * np.sin(rot_angle_28)[:, None])

    sphere.point_data['SaccadeAxis'] = rotated_28 / np.linalg.norm(rotated_28, axis=1, keepdims=True).clip(min=1e-8)
    sphere.point_data['SaccadeAxisSmooth'] = smooth_phasor_field(sphere, 'SaccadeAxis', iterations=10)

    # Heatmaps
    sphere.point_data['Collinearity'] = np.abs(np.einsum('ij,ij->i', raw_flow_unit,
                                                         combed_flow / np.linalg.norm(combed_flow, axis=1,
                                                                                      keepdims=True).clip(min=1e-8)))
    sphere.point_data['SmoothnessComparison'] = np.abs(
        np.einsum('ij,ij->i', sphere.point_data['SaccadeAxis'], sphere.point_data['SaccadeAxisSmooth']))

    # Clip appropriate side for left vs right
    angle_rad = np.radians(cut_angle_deg * eye_sign)
    clip_normal = [np.cos(angle_rad), np.sin(angle_rad), 0.0]
    clipped_mesh = sphere.clip(normal=clip_normal, origin=(0, 0, 0), invert=True)

    # Translate slightly to create separation between eyes
    clipped_mesh.translate([0, -0.5 * eye_sign, 0], inplace=True)

    return clipped_mesh


def alignment_study(strength=1.0, sparsity=0.01, tilt_deg=0.0, pitch_deg=0.0):
    n_sub = 4
    # cut_angle_deg = 90.0  # makes the eyes perfectly hemispherical
    cut_angle_deg = 75.0  # ~15 deg binocular overlap (and ~30 deg blind zone in the back)

    tilt_rad = np.radians(tilt_deg)
    pitch_rad = np.radians(pitch_deg)

    flow_dir = np.array([
        np.cos(pitch_rad) * np.cos(tilt_rad),
        np.cos(pitch_rad) * np.sin(tilt_rad),
        np.sin(pitch_rad)
    ])

    left_eye = generate_eye(1, strength, n_sub, cut_angle_deg, flow_dir)
    right_eye = generate_eye(-1, strength, n_sub, cut_angle_deg, flow_dir)

    both_eyes = left_eye.merge(right_eye)

    # Reference elements
    e_x = flow_dir / np.linalg.norm(flow_dir)
    if np.abs(e_x[2]) > 0.999:
        e_y = np.array([0.0, 1.0, 0.0])
    else:
        e_y = np.cross(np.array([0.0, 0.0, 1.0]), e_x)
        e_y /= np.linalg.norm(e_y)
    e_z = np.cross(e_x, e_y)

    plane_sagittal = pv.Plane(center=(0, 0, 0), direction=(0, 1, 0), i_size=3.0, j_size=3.0)
    plane_equatorial = pv.Plane(center=(0, 0, 0), direction=e_z, i_size=3.0, j_size=3.0)

    # Geometry definitions
    phasor_line = pv.Line(pointa=(-0.5, 0, 0), pointb=(0.5, 0, 0))
    vector_arrow = pv.Arrow(tip_radius=0.08, shaft_radius=0.03, tip_length=0.25)
    flow_arrow = pv.Arrow(start=-2.0 * e_x, direction=e_x, scale=0.5)

    plotter = pv.Plotter(shape=(2, 4), window_size=(2500, 1000))

    def add_standard_setup(p, title):
        p.add_text(title, font_size=10)
        p.add_mesh(both_eyes, color='white', opacity=0.15)
        p.add_mesh(plane_sagittal, color='gray', opacity=0.1)
        p.add_mesh(plane_equatorial, color='blue', opacity=0.1)
        p.add_mesh(flow_arrow, color='purple', lighting=False)
        p.add_axes()

    # Panel 1: Raw flow
    plotter.subplot(0, 0)
    add_standard_setup(plotter, "Optical Flow")
    g = both_eyes.glyph(geom=vector_arrow, orient='RawFlow', tolerance=sparsity, factor=0.08, scale=False)
    plotter.add_mesh(g, color='magenta')

    # Panel 2: Green alignment phasors
    plotter.subplot(0, 1)
    add_standard_setup(plotter, "Alignment Axes (Combed)")
    g = both_eyes.glyph(geom=phasor_line, orient='AlignmentPhasors', tolerance=sparsity, factor=0.08, scale=False)
    plotter.add_mesh(g, color='green', line_width=2)

    # Panel 3: Collinearity heatmap
    plotter.subplot(0, 2)
    add_standard_setup(plotter, "Collinearity Heatmap")
    plotter.add_mesh(both_eyes.copy(), scalars='Collinearity', cmap='inferno', clim=[0, 1])

    # Panel 4: Major axis
    plotter.subplot(0, 3)
    add_standard_setup(plotter, "Major Axes (Brown: Chirality A | Orange: Chirality B)")
    # Filter by chirality for distinct colors
    chir_a = both_eyes.threshold(value=-0.5, scalars='Chirality', preference='point', invert=True)
    chir_b = both_eyes.threshold(value=0.5, scalars='Chirality', preference='point')

    g_a = chir_a.glyph(geom=vector_arrow, orient='MajorAxis', tolerance=sparsity, factor=0.08, scale=False)
    g_b = chir_b.glyph(geom=vector_arrow, orient='MajorAxis', tolerance=sparsity, factor=0.08, scale=False)
    plotter.add_mesh(g_a, color='#5D4037')  # Brown (A)
    plotter.add_mesh(g_b, color='#FF9800')  # Orange (B)

    # Panel 5: Saccade field
    plotter.subplot(1, 0)
    add_standard_setup(plotter, "Saccade Axes (Chirality-Dependent)")
    g = both_eyes.glyph(geom=phasor_line, orient='SaccadeAxis', tolerance=sparsity, factor=0.08, scale=False)
    plotter.add_mesh(g, color='red', line_width=2)

    # Panel 6: Smoothed saccade field
    plotter.subplot(1, 1)
    add_standard_setup(plotter, "Saccade Axes (Smoothed)")
    g = both_eyes.glyph(geom=phasor_line, orient='SaccadeAxisSmooth', tolerance=sparsity, factor=0.08, scale=False)
    plotter.add_mesh(g, color='pink', line_width=2)

    # Panel 7: Smoothing comparison heatmap
    plotter.subplot(1, 2)
    add_standard_setup(plotter, "Smoothing Consistency")
    plotter.add_mesh(both_eyes.copy(), scalars='SmoothnessComparison', cmap='inferno', clim=[0.9, 1])

    # Panel 8: Chirality map
    plotter.subplot(1, 3)
    add_standard_setup(plotter, "Chirality Map (A vs B)")
    plotter.add_mesh(both_eyes.copy(), scalars='Chirality', cmap='copper')

    plotter.link_views()

    plotter.camera_position = [(-3.5, 0.0, 1.0), (0, 0, 0), (0, 0, 1)]
    plotter.show()


if __name__ == "__main__":
    alignment_study(strength=1.0, sparsity=0.01, tilt_deg=0.0, pitch_deg=HEAD_PITCH)