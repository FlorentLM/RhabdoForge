#version 430 core

#include "commons.glsl"

// Input: Per-vertex attribute for the base cone mesh
layout (location = 0) in vec3 a_cone_vertex_pos;

// Bindings
layout(std430, binding = 0) readonly buffer OmmatidiaInputBlock {
   Ommatidium u_ommatidia_data[];
};

layout(std430, binding = 1) readonly buffer ColorDataBlock {
   vec4 u_ommatidia_colors[];
};

// Uniforms queried by name
uniform bool u_tiled_mode;
uniform float u_cone_scale;
uniform float u_aspect_ratio;

// Output: Varying to fragment shader
layout (location = 0) out vec3 v_color;


void main() {
    // Get the data for the specific instance (ommatidium) we are drawing
    int instance_id = gl_InstanceID;

    Ommatidium om = u_ommatidia_data[instance_id];

    // need to re-calculate screen position from direction vector
    vec4 dir = om.direction;
    float longitude = atan(dir.x, -dir.z); // atan2(x, z)
    float latitude = asin(dir.y);

    // Map longitude/latitude to screen space [-1, 1]
    float screen_x = longitude / PI;
    float screen_y = latitude / (PI / 2.0);
    vec2 instance_screen_pos = vec2(screen_x, screen_y);

    vec2 acceptance_angles = om.acceptance_angles;
    vec3 instance_color = u_ommatidia_colors[instance_id].rgb;

    vec3 scaled_cone_pos = a_cone_vertex_pos;

    if (u_tiled_mode) {
        // To generate a classic Voronoi diagram, all cones must be huge
        // A radius of 5.0 in clip space is more than enough to cover the screen
        scaled_cone_pos.xy *= u_cone_scale;
    } else {
        // For visualizing receptive fields, scale by acceptance angle
        scaled_cone_pos.xy *= acceptance_angles * u_cone_scale;
    }

    // Final position is the scaled cone's vertex position, translated to the ommatidium's unique screen position
    // We are drawing in 2D clip space, so Z is for depth
    vec3 final_pos = scaled_cone_pos + vec3(instance_screen_pos, 0.0);

    // Squash the X coordinate by the aspect ratio so they have the correct proportions when displayed
    final_pos.x /= u_aspect_ratio;

    gl_Position = vec4(final_pos, 1.0);

    // Pass the unique color for this instance to the fragment shader
    v_color = instance_color;
}

