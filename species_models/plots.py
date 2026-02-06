import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_eyes_3d(origins, directions, eye_id, title, arrow_length=0.1, show_sphere_projection=False):
    """
    Plot eye model in 3D with direction arrows or spherical projections.

    Args:
        origins: (N, 3) ommatidium positions
        directions: (N, 3) unit direction vectors
        eye_id: (N,) array with 0 for left eye, 1 for right eye
        title: Plot title
        arrow_length: Length of direction arrows (used if show_sphere_projection=False)
        show_sphere_projection: If True, extend direction vectors to sphere surface
    """

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    right_mask = eye_id == 1
    left_mask = eye_id == 0

    # Origins
    ax.scatter(origins[right_mask, 0], origins[right_mask, 1], origins[right_mask, 2],
               c='red', s=20, alpha=0.8, label='Right eye')
    ax.scatter(origins[left_mask, 0], origins[left_mask, 1], origins[left_mask, 2],
               c='blue', s=20, alpha=0.8, label='Left eye')

    if show_sphere_projection:

        max_origin_dist = np.linalg.norm(origins, axis=1).max()
        sphere_radius = max_origin_dist * 2.5  # multiplier for sphere size
        plot_scale = sphere_radius

        # Ray: P(t) = O + tD
        # Sphere: |P|^2 = R^2  => t^2(D.D) + 2t(O.D) + (O.O - R^2) = 0
        # Since D is unit vector, a = 1

        # Dot products for b and c
        dot_od = np.einsum('ij,ij->i', origins, directions)  # (N,)
        dot_oo = np.einsum('ij,ij->i', origins, origins)  # (N,)

        a = 1.0
        b = 2 * dot_od
        c = dot_oo - sphere_radius ** 2

        discriminant = b ** 2 - 4 * a * c
        t = (-b + np.sqrt(np.maximum(0, discriminant))) / (2 * a)

        intersections = origins + directions * t[:, np.newaxis]

        # Plot intersection points on sphere
        ax.scatter(intersections[right_mask, 0], intersections[right_mask, 1], intersections[right_mask, 2],
                   c='red', s=5, alpha=0.3, marker='.', label='Right projections')
        ax.scatter(intersections[left_mask, 0], intersections[left_mask, 1], intersections[left_mask, 2],
                   c='blue', s=5, alpha=0.3, marker='.', label='Left projections')

        # Draw connecting lines (subsampled)
        step = max(1, len(origins) // 100)

        for i in range(0, len(origins), step):
            pts = np.vstack((origins[i], intersections[i]))
            color = 'red' if eye_id[i] == 1 else 'blue'
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.1, linewidth=0.5)

        # Wireframe sphere for reference
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.05, linewidth=0.5)

    else:
        # Direction arrows
        ax.quiver(origins[right_mask, 0], origins[right_mask, 1], origins[right_mask, 2],
                  directions[right_mask, 0], directions[right_mask, 1], directions[right_mask, 2],
                  length=arrow_length, color='red', alpha=0.3, linewidth=1, arrow_length_ratio=0.2)

        ax.quiver(origins[left_mask, 0], origins[left_mask, 1], origins[left_mask, 2],
                  directions[left_mask, 0], directions[left_mask, 1], directions[left_mask, 2],
                  length=arrow_length, color='blue', alpha=0.3, linewidth=1, arrow_length_ratio=0.2)

        max_range = np.array([
            origins[:, 0].max() - origins[:, 0].min(),
            origins[:, 1].max() - origins[:, 1].min(),
            origins[:, 2].max() - origins[:, 2].min()
        ]).max() / 2.0
        plot_scale = max_range

    # Coordinate gizmo
    gizmo_len = plot_scale * 0.5 if show_sphere_projection else 0.3
    ax.quiver(0, 0, 0, gizmo_len, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='X axis')
    ax.quiver(0, 0, 0, 0, gizmo_len, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='Y axis')
    ax.quiver(0, 0, 0, 0, 0, -gizmo_len, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z axis')

    ax.set_xlabel('← Insect\'s left | Insect\'s right →')
    ax.set_ylabel('← Insect\'s down | Insect\'s up →')
    ax.set_zlabel('← Insect\'s back | Insect\'s front →')
    ax.set_title(title)

    # Equal aspect ratio
    if show_sphere_projection:
        limit = sphere_radius * 1.1
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
    else:
        mid_x = (origins[:, 0].max() + origins[:, 0].min()) * 0.5
        mid_y = (origins[:, 1].max() + origins[:, 1].min()) * 0.5
        mid_z = (origins[:, 2].max() + origins[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - plot_scale, mid_x + plot_scale)
        ax.set_ylim(mid_y - plot_scale, mid_y + plot_scale)
        ax.set_zlim(mid_z - plot_scale, mid_z + plot_scale)

    try:
        ax.set_box_aspect([1, 1, 1])
    except AttributeError:
        # older matplotlib versions
        pass

    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.show()