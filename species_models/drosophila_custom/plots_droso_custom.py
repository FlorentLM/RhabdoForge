import numpy as np
import matplotlib.pyplot as plt


def plot_buchner_3d(ommatidia_3d, stars_3d, origin_3d, axes_3d, title="Drosophila eye - Buchner, 1971"):
    """
    3D version of the Buchner data with.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*ommatidia_3d.T, c='grey', s=10, alpha=0.5, label='Ommatidia')
    ax.scatter(*stars_3d.T, c='red', s=150, marker='*', label='Forward', depthshade=False)
    ax.scatter(*origin_3d, c='black', s=50, marker='X', label='Origin', depthshade=False)

    # Plot axes
    colors = {'axis-x': '#ff60b3', 'axis-y': '#00FF9B', 'axis-v': '#FFC400'}
    for axis_id, points in axes_3d.items():
        ax.plot(*points.T, color=colors[axis_id], linewidth=2.5,
                label=f'Axis {axis_id[-1].upper()}')

    ax.set_title(title, fontsize=16)

    # Set up axes properly
    all_points = np.vstack([ommatidia_3d, stars_3d])
    min_coords, max_coords = all_points.min(axis=0), all_points.max(axis=0)
    center, max_range = (max_coords + min_coords) / 2, (max_coords - min_coords).max() / 2

    ax.quiver(0, 0, 0, 0.5, 0, 0, color='r', label='Right (+X)')
    ax.quiver(0, 0, 0, 0, 0.5, 0, color='g', label='Up (+Y)')
    ax.quiver(0, 0, 0, 0, 0, -0.5, color='b', label='Forward (-Z)')

    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=45)
    ax.legend()

    plt.tight_layout()
    plt.show()