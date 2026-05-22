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
    e_x = flow_dir / np.linalg.norm(flow_dir)
    if np.abs(e_x[2]) > 0.999:
        e_y = np.array([0.0, 1.0, 0.0])
    else:
        e_y = np.cross(np.array([0.0, 0.0, 1.0]), e_x)
        e_y /= np.linalg.norm(e_y)
    e_z = np.cross(e_x, e_y)

    # Base projection
    dot_p = np.dot(normals, flow_dir)
    raw_flow = flow_dir - (dot_p[:, np.newaxis] * normals)
    raw_flow_unit = raw_flow / np.linalg.norm(raw_flow, axis=1, keepdims=True).clip(min=1e-8)

    source_point = -1.0 * e_x
    combed_flow = np.zeros_like(raw_flow)

    for i in range(sphere.n_points):
        p, n, v_raw = points[i], normals[i], raw_flow[i]
        dist_from_source = np.linalg.norm(p - source_point)
        local_w = max(0, 1.0 - (dist_from_source * 0.7)) * strength

        # Invert e_y influence for the right eye to mirror bilateral symmetry
        if np.dot(p, e_z) >= 0:
            v_target_ideal = -e_x - (e_y * eye_sign) + e_z
        else:
            v_target_ideal = -e_x - (e_y * eye_sign) - e_z

        v_target_proj = v_target_ideal - np.dot(v_target_ideal, n) * n
        mag = np.linalg.norm(v_raw)
        if np.linalg.norm(v_target_proj) > 1e-6:
            v_target_proj = (v_target_proj / np.linalg.norm(v_target_proj)) * mag
        else:
            v_target_proj = v_raw

        combed_flow[i] = (1.0 - local_w) * v_raw + (local_w * v_target_proj)

    sphere.point_data['RawFlow'] = raw_flow_unit
    sphere.point_data['AlignmentPhasors'] = combed_flow

    # Main axis -81° offset (orange)
    # Multiply rotation by eye_sign to mirror right-hand rule cross products
    rot_angle_rad_81 = np.radians(MAIN_AXIS_OFFSET * eye_sign)
    cross_nv_81 = np.cross(normals, combed_flow)
    rotated_flow_81 = combed_flow * np.cos(rot_angle_rad_81) + cross_nv_81 * np.sin(rot_angle_rad_81)

    p_height = np.einsum('ij,j->i', points, e_z)
    ref_binormal = np.cross(normals, combed_flow)
    dot_check = np.einsum('ij,ij->i', rotated_flow_81, ref_binormal)
    rotated_flow_81[dot_check < 0] *= -1.0

    rotated_flow_81_unit = rotated_flow_81 / np.linalg.norm(rotated_flow_81, axis=1, keepdims=True).clip(min=1e-8)
    sphere.point_data['MajorAxis'] = rotated_flow_81_unit

    # Saccade offsets
    hemisphere_sign = np.sign(p_height)
    hemisphere_sign[hemisphere_sign == 0] = 1.0

    # Mirror the angle for right eye
    rot_angle_rad_28_base = np.radians(SACCADE_AXIS_OFFSET * eye_sign)
    angles_28 = rot_angle_rad_28_base * hemisphere_sign

    cos_28 = np.cos(angles_28)[:, np.newaxis]
    sin_28 = np.sin(angles_28)[:, np.newaxis]

    cross_nv_28 = np.cross(normals, rotated_flow_81_unit)
    rotated_flow_28 = (rotated_flow_81_unit * cos_28) + (cross_nv_28 * sin_28)

    rotated_flow_28_unit = rotated_flow_28 / np.linalg.norm(rotated_flow_28, axis=1, keepdims=True).clip(min=1e-8)

    sphere.point_data['Collinearity'] = np.abs(np.einsum('ij,ij->i', raw_flow_unit,
                                                         combed_flow / np.linalg.norm(combed_flow, axis=1,
                                                                                      keepdims=True).clip(min=1e-8)))

    sphere.point_data['SaccadeAxis'] = rotated_flow_28_unit
    sphere.point_data['SaccadeAxisSmooth'] = smooth_phasor_field(sphere, 'SaccadeAxis', iterations=10)
    sphere.point_data['SmoothnessComparison'] = np.abs(np.einsum('ij,ij->i',
                                                                 sphere.point_data['SaccadeAxis'],
                                                                 sphere.point_data['SaccadeAxisSmooth']))

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

    both_eyes_mesh = left_eye.merge(right_eye)

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
    vector_arrow_geom = pv.Arrow(tip_radius=0.08, shaft_radius=0.03, tip_length=0.25)
    flow_arrow = pv.Arrow(start=-2.0 * e_x, direction=e_x, scale=0.5)

    # Visualisation
    plotter = pv.Plotter(shape=(2, 4), window_size=(2500, 1000))

    def add_ref(p):
        p.add_mesh(plane_sagittal, color='gray', opacity=0.1)
        p.add_mesh(plane_equatorial, color='blue', opacity=0.1)
        p.add_mesh(flow_arrow, color='purple', lighting=False)
        p.add_axes()

    # Panel 1: Raw flow
    plotter.subplot(0, 0)
    plotter.add_text("Optical flow", font_size=10)
    g1 = both_eyes_mesh.glyph(geom=vector_arrow_geom, orient='RawFlow', tolerance=sparsity, scale=False, factor=0.08)
    plotter.add_mesh(g1, color='magenta', lighting=False)
    plotter.add_mesh(both_eyes_mesh, color='white', opacity=0.3)
    add_ref(plotter)

    # Panel 2: Green alignment phasors
    plotter.subplot(0, 1)
    plotter.add_text("Alignment axes", font_size=10)
    g2 = both_eyes_mesh.glyph(geom=phasor_line, orient='AlignmentPhasors', tolerance=sparsity, scale=False, factor=0.08)
    plotter.add_mesh(g2, color='green', line_width=2, lighting=False)
    plotter.add_mesh(both_eyes_mesh, color='white', opacity=0.3)
    add_ref(plotter)

    # Panel 3: Collinearity heatmap
    plotter.subplot(0, 2)
    plotter.add_text("Collinearity heatmap", font_size=10)
    plotter.add_mesh(both_eyes_mesh.copy(), scalars='Collinearity', cmap='inferno', clim=[0, 1])
    add_ref(plotter)

    # Panel 4: Major axis
    plotter.subplot(0, 3)
    plotter.add_text("Major axes", font_size=10)
    g3 = both_eyes_mesh.glyph(geom=vector_arrow_geom, orient='MajorAxis', tolerance=sparsity, scale=False, factor=0.08)
    plotter.add_mesh(g3, color='orange', lighting=False)
    plotter.add_mesh(both_eyes_mesh, color='white', opacity=0.3)
    add_ref(plotter)

    # Panel 5: Saccade field
    plotter.subplot(1, 0)
    plotter.add_text("Saccade axes", font_size=10)
    g4 = both_eyes_mesh.glyph(geom=phasor_line, orient='SaccadeAxis', tolerance=sparsity, scale=False, factor=0.08)
    plotter.add_mesh(g4, color='red', line_width=2, lighting=False)
    plotter.add_mesh(both_eyes_mesh, color='white', opacity=0.3)
    add_ref(plotter)

    # Panel 6: Smoothed saccade field
    plotter.subplot(1, 1)
    plotter.add_text("Saccade axes (Smoothed)", font_size=10)
    g5 = both_eyes_mesh.glyph(geom=phasor_line, orient='SaccadeAxisSmooth', tolerance=sparsity, scale=False,
                              factor=0.08)
    plotter.add_mesh(g5, color='pink', line_width=2, lighting=False)
    plotter.add_mesh(both_eyes_mesh, color='white', opacity=0.3)
    add_ref(plotter)

    # Panel 7: Smoothing comparison heatmap
    plotter.subplot(1, 2)
    plotter.add_text("Smoothing Consistency", font_size=10)
    plotter.add_mesh(both_eyes_mesh.copy(), scalars='SmoothnessComparison', cmap='inferno', clim=[0.9, 1])
    add_ref(plotter)

    plotter.link_views()

    plotter.camera_position = [(-3.5, 0.0, 1.0), (0, 0, 0), (0, 0, 1)]
    plotter.show()


if __name__ == "__main__":
    alignment_study(strength=1.0, sparsity=0.01, tilt_deg=0.0, pitch_deg=HEAD_PITCH)