# DEBUG_BVH.py

import pygame
import numpy as np
from pathlib import Path

# --- Import the refactored PointCloud class ---
from graphics.scene import RaytracingPointCloud

# --- CONFIG ---
FILENAME = 'dolphins.ply'
ASSETS_DIR = Path('../assets')
FILE_PATH = ASSETS_DIR / FILENAME

HIT_RADIUS = 3.0

bvh_node_dtype = np.dtype([
    ('aabb_min', 'f4', (3,)),       # 3x 32-bit floats
    ('left_first', 'u4', (1,)),     # 1x 32-bit unsigned int
    ('aabb_max', 'f4', (3,)),       # 3x 32-bit floats
    ('tri_count', 'u4', (1,))       # 1x 32-bit unsigned int
], align=True)

# =========================================================================

def intersect_aabb(ray_origin, ray_direction, aabb_min, aabb_max):
    inv_dir = np.divide(1.0, ray_direction, out=np.full_like(ray_direction, np.inf), where=ray_direction != 0)
    t1 = (aabb_min - ray_origin) * inv_dir
    t2 = (aabb_max - ray_origin) * inv_dir
    tmin = np.maximum(np.minimum(t1, t2), 0.0)  # clamp tmin to be >= 0
    tmax = np.minimum(np.maximum(t1, t2), np.inf)
    t_near = np.max(tmin)
    t_far = np.min(tmax)
    if t_near > t_far:
        return float('inf')
    return t_near

def intersect_ray_sphere(ray_origin, ray_direction, sphere_center, sphere_radius):
    oc = ray_origin - sphere_center
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere_radius ** 2
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return float('inf')

    sqrt_d = np.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2.0 * a)
    t2 = (-b + sqrt_d) / (2.0 * a)

    # check for the smallest non-negative t value
    if t1 >= 0 and t2 >= 0:
        return min(t1, t2)
    if t1 >= 0:
        return t1
    if t2 >= 0:
        return t2
    return float('inf')

# =========================================================================

point_cloud = RaytracingPointCloud(FILE_PATH, hit_radius=HIT_RADIUS)

bvh_nodes = point_cloud.bvh_nodes
reordered_primitives = point_cloud.point_attributes[:, 0:3]

# View the raw bvh_nodes array with the structured dtype for easier access in Python
bvh_nodes_structured = bvh_nodes.view(bvh_node_dtype).reshape(-1)

# Pygame setup
pygame.init()
pygame.mouse.set_visible(False)
screen_width, screen_height = 1280, 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("BVH Ray Traversal Viewer (DEBUG MODE)")
font = pygame.font.Font(None, 24)
clock = pygame.time.Clock()

# Viewport calculation
scene_min = bvh_nodes_structured[0]['aabb_min']
scene_max = bvh_nodes_structured[0]['aabb_max']
scene_size = scene_max - scene_min
view_plane_size = max(scene_size[0], scene_size[2], 1e-6)
scale = min(screen_width / view_plane_size, screen_height / view_plane_size) * 0.9
offset_x = (screen_width - scene_size[0] * scale) / 2
offset_y = (screen_height - scene_size[2] * scale) / 2


def world_to_screen(pos):
    x = (pos[0] - scene_min[0]) * scale + offset_x
    y = (pos[2] - scene_min[2]) * scale + offset_y
    return int(x), int(y)


def screen_to_world_xz(screen_pos):
    world_x = scene_min[0] + (screen_pos[0] - offset_x) / scale
    world_z = scene_min[2] + (screen_pos[1] - offset_y) / scale
    return world_x, world_z


print("Calculating screen positions for background primitives...")
all_prim_screen_pos = [world_to_screen(p) for p in reordered_primitives]
print("Ready.")

# =========================================================================

running = True
frame_count = 0
mouse_pos = (0, 0)

# State variables for the interactive ray
ray_start_point = None  # will be set on mouse click
# start the height in the middle of the scene's Y-axis
ray_height = (scene_min[1] + scene_max[1]) / 2.0
height_change_speed = scene_size[1] / 50.0  # adjust speed based on scene size

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            frame_count = 0  # Reset to print debug info when mouse moves

        # Handle mouse click and key presses for ray control
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse click
                world_x, world_z = screen_to_world_xz(event.pos)
                ray_start_point = np.array([world_x, ray_height, world_z], dtype=np.float32)

        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                ray_height += height_change_speed
                if ray_start_point is not None:
                    ray_start_point[1] = ray_height

            if event.key == pygame.K_MINUS:
                ray_height -= height_change_speed
                if ray_start_point is not None:
                    ray_start_point[1] = ray_height

    # Drawing
    screen.fill((20, 20, 30))

    # Draw all primitives in the background
    for pos in all_prim_screen_pos:
        pygame.draw.circle(screen, (50, 50, 80), pos, 1)

    traversal_path = []
    closest_hit_dist = float('inf')

    if ray_start_point is not None:

        # Ray setup
        mouse_world_x, mouse_world_z = screen_to_world_xz(mouse_pos)

        # target is at the same height as the ray's origin for a 2D-like projection
        ray_target_point = np.array([mouse_world_x, ray_height, mouse_world_z], dtype=np.float32)

        ray_origin = ray_start_point

        # Calculate direction vector and normalize it
        direction_vec = ray_target_point - ray_origin
        norm = np.linalg.norm(direction_vec)
        if norm > 1e-6:  # Avoid division by zero
            ray_direction = direction_vec / norm
        else:  # If start and end are the same, create a zero direction vector
            ray_direction = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Traversal logic
        hit_primitives = []
        closest_hit_point = None
        stack = [0]

        while stack:
            node_idx = stack.pop()
            aabb_min = bvh_nodes_structured[node_idx]['aabb_min']
            aabb_max = bvh_nodes_structured[node_idx]['aabb_max']
            dist_to_aabb = intersect_aabb(ray_origin, ray_direction, aabb_min, aabb_max)
            hit = dist_to_aabb < closest_hit_dist
            traversal_path.append((node_idx, hit))

            if not hit:
                continue

            primitive_count = bvh_nodes_structured[node_idx]['tri_count'][0]  # [0] to get the scalar
            if primitive_count == 0:  # internal node
                left_child = bvh_nodes_structured[node_idx]['left_first'][0]
                stack.append(left_child + 1)
                stack.append(left_child)

            else:  # leaf node
                start_index = bvh_nodes_structured[node_idx]['left_first'][0]
                end_index = start_index + primitive_count
                for i in range(start_index, end_index):
                    if i < len(reordered_primitives):
                        prim_pos = reordered_primitives[i]
                        dist_to_prim = intersect_ray_sphere(ray_origin, ray_direction, prim_pos, HIT_RADIUS)

                        if dist_to_prim < closest_hit_dist:
                            closest_hit_dist = dist_to_prim
                            closest_hit_point = ray_origin + closest_hit_dist * ray_direction

        # Drawing ray
        # Draw the traversal path AABBs
        for node_idx, hit in reversed(traversal_path):
            aabb_min = bvh_nodes[node_idx, 0:3]
            aabb_max = bvh_nodes[node_idx, 4:7]
            p_min = world_to_screen(aabb_min)
            p_max = world_to_screen(aabb_max)
            rect_w = max(1, p_max[0] - p_min[0])
            rect_h = max(1, p_max[1] - p_min[1])
            color = (0, 150, 0) if hit else (150, 0, 0)
            pygame.draw.rect(screen, color, (p_min[0], p_min[1], rect_w, rect_h), 1)

        # Draw the successfully hit point
        if closest_hit_point is not None:
            pygame.draw.circle(screen, (255, 0, 255), world_to_screen(closest_hit_point), int(HIT_RADIUS*scale), 2)


        # Draw the ray and its origin marker
        ray_start_screen = world_to_screen(ray_start_point)
        # line from the start point to the current mouse position
        pygame.draw.line(screen, (255, 255, 0), ray_start_screen, mouse_pos, 2)
        # marker at the ray's origin
        pygame.draw.circle(screen, (0, 255, 255), ray_start_screen, 6, 2)

    info_text = [
        f"Ray Height (Y): {ray_height:.2f} (+/- to change)",
        f"Click to set ray origin",
        f"---",
        f"Traversal Path Length: {len(traversal_path)}",
        f"Hit Nodes: {sum(1 for _, hit in traversal_path if hit)}",
        f"Closest Hit Distance: {closest_hit_dist if closest_hit_dist != float('inf') else 'N/A'}"
    ]
    for i, line in enumerate(info_text):
        text_surf = font.render(line, True, (255, 255, 255))
        screen.blit(text_surf, (10, 10 + i * 25))

    crosshair_inner_radius = 5
    crosshair_outer_radius = 10
    cursor_color = (255, 255, 255)
    mx, my = mouse_pos
    pygame.draw.line(screen, cursor_color, (mx, my - crosshair_inner_radius), (mx, my - crosshair_outer_radius), 1)
    pygame.draw.line(screen, cursor_color, (mx, my + crosshair_inner_radius), (mx, my + crosshair_outer_radius), 1)
    pygame.draw.line(screen, cursor_color, (mx - crosshair_inner_radius, my), (mx - crosshair_outer_radius, my), 1)
    pygame.draw.line(screen, cursor_color, (mx + crosshair_inner_radius, my), (mx + crosshair_outer_radius, my), 1)

    pygame.display.flip()
    clock.tick(60)
    frame_count += 1

pygame.quit()