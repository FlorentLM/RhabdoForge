#version 430 core

#include "commons.glsl"

// Input: Per-vertex attribute for the base cone mesh
layout (location = 0) in vec3 cone_vertex;

// Bindings
layout(std430, binding = 0) readonly buffer OmmatidiaInputBlock {
   Ommatidium ommatidia_data[];
};

layout(std430, binding = 1) readonly buffer ColorDataBlock {
   vec4 ommatidia_colors[];
};

// Uniforms
uniform float aspect_ratio;
uniform int projection_mode; // 0 = Physical Layout, 1 = Acceptance angle
uniform bool tiled_mode;

uniform float tiled_mode_scale;         // factor for tiled mode
uniform float receptive_field_scale;    // factor for visual fields

// Output: Varying to fragment shader
layout (location = 0) out vec3 v_color;

void main() {
    // Get the data for the specific instance (ommatidium) we are drawing
    int instance_id = gl_InstanceID;
    Ommatidium om = ommatidia_data[instance_id];
    vec2 acceptance_angles = om.acceptance_angles;
    vec3 instance_color = ommatidia_colors[instance_id].rgb;

    // Choose the vector to use for projection based on mode
    vec3 projection_vector;

    if (projection_mode == 1) {
        // Acceptance - use direction vectors
        projection_vector = om.direction.xyz;
    } else {
        // Physical layout projection - use origin vectors
         projection_vector = normalize(om.origin.xyz);
    }

    // Apply spherical projection to screen space
    float longitude = atan(projection_vector.x, -projection_vector.z);
    float latitude = asin(projection_vector.y);

    // Map longitude/latitude to screen space [-1, 1]
    float screen_x = longitude / PI;
    float screen_y = latitude / HPI;

    // Apply scaling for physical layout mode
    vec2 instance_screen_pos = vec2(screen_x, screen_y);

    // Scale the cone vertex based on mode
    vec3 scaled_cone_pos = cone_vertex;

    if (tiled_mode) {
        scaled_cone_pos.xy *= om.interommatidial_angles * receptive_field_scale * 2.5;
    } else {
        // Receptive field mode - scale by acceptance angles
        scaled_cone_pos.xy *= acceptance_angles * receptive_field_scale;
    }

    // Position the cone at the ommatidium's screen position
    vec3 final_pos = scaled_cone_pos + vec3(instance_screen_pos, 0.0);

    // Apply aspect ratio correction
    final_pos.x /= aspect_ratio;

    gl_Position = vec4(final_pos, 1.0);
    v_color = instance_color;
}