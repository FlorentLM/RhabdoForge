#version 430 core

// Per-vertex attribute for the base cone mesh
layout (location = 0) in vec3 a_cone_vertex_pos;

// Data from SSBOs indexed by gl_InstanceID
struct OmmatidiumInput {
    vec3 direction;
    float acceptance_angle;
};

layout(std430, binding = 0) readonly buffer OmmatidiaInputBlock {
   OmmatidiumInput u_ommatidia_data[];
};

layout(std430, binding = 1) readonly buffer ColorDataBlock {
   vec4 u_ommatidia_colors[];
};

// Whether to fill the screen or represent the actual acceptance angles
uniform bool u_tiled_mode;

// To fragment shader
out vec3 v_color;

const float PI = 3.14159265359;
const float VISUAL_SCALE_MULTIPLIER = 0.167;

void main() {
    // Get the data for the specific instance (ommatidium) we are drawing
    int instance_id = gl_InstanceID;

    // need to re-calculate screen position from direction vector
    vec3 dir = u_ommatidia_data[instance_id].direction;
    float longitude = atan(dir.x, -dir.z); // atan2(x, z)
    float latitude = asin(dir.y);

    // Map longitude/latitude to screen space [-1, 1]
    float screen_x = longitude / PI;
    float screen_y = latitude / (PI / 2.0);
    vec2 instance_screen_pos = vec2(screen_x, screen_y);

    float acceptance_angle = u_ommatidia_data[instance_id].acceptance_angle;
    vec3 instance_color = u_ommatidia_colors[instance_id].rgb;

    vec3 scaled_cone_pos = a_cone_vertex_pos;

    if (u_tiled_mode) {
        // To generate a classic Voronoi diagram, all cones must be huge
        // A radius of 5.0 in clip space is more than enough to cover the screen
        scaled_cone_pos.xy *= 5.0;
    } else {
        // For visualizing receptive fields, scale by acceptance angle
        scaled_cone_pos.xy *= acceptance_angle * VISUAL_SCALE_MULTIPLIER;
    }

    // Final position is the scaled cone's vertex position, translated to the ommatidium's unique screen position
    // We are drawing in 2D clip space, so Z is for depth
    vec3 final_pos = scaled_cone_pos + vec3(instance_screen_pos, 0.0);

    gl_Position = vec4(final_pos, 1.0);

    // Pass the unique color for this instance to the fragment shader
    v_color = instance_color;
}

