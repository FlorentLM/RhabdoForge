#version 430 core

#include "commons.glsl"

// Input: Per-vertex attribute for the base cone mesh
layout (location = 0) in vec3 cone_vertex;

// Bindings
layout(std430, binding = 0) readonly buffer ReceptorsInputBlock {
   Receptor receptors_data[];
};

layout(std430, binding = 1) readonly buffer ColorDataBlock {
   vec4 receptors_colors[];
};

// Uniforms
uniform float aspect_ratio;
uniform int projection_mode; // 0 = Physical, 1 = Acceptance
uniform bool tiled_mode;

uniform float tiled_mode_scale;
uniform float receptive_field_scale;

// Output: Varying to fragment shader
layout (location = 0) out vec3 v_color;

void main() {
    // Get the data for this instance
    int instance_id = gl_InstanceID;
    Receptor rcpt = receptors_data[instance_id];

    float tilt_angle = rcpt.tilt;
    vec3 instance_color = receptors_colors[instance_id].rgb;

    // Determine projection vector based on mode
    vec3 projection_vector = (projection_mode == 1)
        ? rcpt.direction.xyz
        : normalize(rcpt.position.xyz);

    // Apply spherical projection to get screen position
    float longitude = atan(projection_vector.x, -projection_vector.z);
    float latitude = asin(projection_vector.y);
    vec2 instance_screen_pos = vec2(longitude / PI, latitude / HPI);

    // ellipse shape is determined by acceptance angles or interommatidial angles depending on the mode
    vec2 base_ellipse_shape;
    if (tiled_mode) {
        base_ellipse_shape = cone_vertex.xy * rcpt.interommatidial_angles;
    } else {
        base_ellipse_shape = cone_vertex.xy * rcpt.acceptance_angles;
    }

    // Orient the ellipse
    float s = sin(tilt_angle);
    float c = cos(tilt_angle);
    mat2 rotation_matrix = mat2(c, -s, s, c);
    vec2 rotated_ellipse_xy = rotation_matrix * base_ellipse_shape;

    // Screen-space scaling
    vec3 scaled_cone_pos;
    if (tiled_mode) {
        scaled_cone_pos.xy = rotated_ellipse_xy * receptive_field_scale * 2.5;
    } else {
        scaled_cone_pos.xy = rotated_ellipse_xy * receptive_field_scale;
    }
    scaled_cone_pos.z = cone_vertex.z;

    // Position the rotated and scaled cone
    vec3 final_pos = scaled_cone_pos + vec3(instance_screen_pos, 0.0);

    // Apply aspect ratio correction
    final_pos.x /= aspect_ratio;

    gl_Position = vec4(final_pos, 1.0);
    v_color = instance_color;
}