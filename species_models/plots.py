import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_eyes_3d(origins, directions, eye_id, title, arrow_length=0.05):
    """Plot eye model in 3D with direction arrows."""

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    right_mask = eye_id == 1
    ax.scatter(origins[right_mask, 0],
               origins[right_mask, 1],
               origins[right_mask, 2],
               c='red', s=5, alpha=0.6, label='Right eye')

    left_mask = eye_id == 0
    ax.scatter(origins[left_mask, 0],
               origins[left_mask, 1],
               origins[left_mask, 2],
               c='blue', s=5, alpha=0.6, label='Left eye')

    # Sample some direction arrows (every 50 th ommatidium)
    sample_idx = np.arange(0, len(origins), 50)
    for idx in sample_idx:
        color = 'red' if eye_id[idx] == 1 else 'blue'
        ax.quiver(origins[idx, 0], origins[idx, 1], origins[idx, 2],
                  directions[idx, 0], directions[idx, 1], directions[idx, 2],
                  length=arrow_length, color=color, alpha=0.3, arrow_length_ratio=0.3)

    # Draw coordinate axes at origin (showing insect's perspective)
    axis_length = 0.3
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.2, linewidth=2, label='X (right)')
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.2, linewidth=2, label='Y (up)')
    ax.quiver(0, 0, 0, 0, 0, -axis_length, color='blue', arrow_length_ratio=0.2, linewidth=2, label='Z (back)')

    ax.set_xlabel('← Insect\'s left | Insect\'s right →')
    ax.set_ylabel('← Insect\'s down | Insect\'s up →')
    ax.set_zlabel('← Insect\'s back | Insect\'s front →')
    ax.set_title(title)

    # Equal aspect ratio
    max_range = np.array([
        origins[:, 0].max() - origins[:, 0].min(),
        origins[:, 1].max() - origins[:, 1].min(),
        origins[:, 2].max() - origins[:, 2].min()
    ]).max() / 2.0

    mid_x = (origins[:, 0].max() + origins[:, 0].min()) * 0.5
    mid_y = (origins[:, 1].max() + origins[:, 1].min()) * 0.5
    mid_z = (origins[:, 2].max() + origins[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend(loc='upper right', fontsize=8)

    plt.show()
