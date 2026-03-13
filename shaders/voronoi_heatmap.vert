#version 430 core

#include "commons.glsl"

layout (location = 0) in vec3 cone_vertex;

layout(std430, binding = 0) readonly buffer ReceptorsInputBlock {
   Receptor receptors_data[];
};

layout(std430, binding = 1) readonly buffer ScalarDataBlock {
   float scalar_data[];
};

uniform float aspect_ratio;
uniform int projection_mode;  // 0 = physical, 1 = Acceptance
uniform bool tiled_mode;
uniform float receptive_field_scale;

uniform float data_min;       // lower bound of colormap range
uniform float data_max;       // upper bound of colormap range
uniform int colormap;         // 0 = diverging (blue white red), 1 = sequential (viridis), 2 = thermal
uniform float compression;    // power exponent: 1.0 = linear, 0.5 = sqrt, lower = very contrasty

layout (location = 0) out float v_scalar;   // normalised [0, 1]

void main() {
    int instance_id = gl_InstanceID;
    Receptor rcpt = receptors_data[instance_id];

    vec3 projection_vector = (projection_mode == 1) ? rcpt.direction.xyz: normalize(rcpt.position.xyz);

    float longitude = atan(projection_vector.x, -projection_vector.z);
    float latitude  = asin(projection_vector.y);
    vec2 instance_screen_pos = vec2(longitude / PI, latitude / HPI);

    vec2 base_ellipse_shape;
    if (tiled_mode) {
        base_ellipse_shape = cone_vertex.xy * rcpt.interommatidial_angles;
    } else {
        base_ellipse_shape = cone_vertex.xy * rcpt.acceptance_angles;
    }

    float s = sin(rcpt.tilt);
    float c = cos(rcpt.tilt);
    mat2 rotation_matrix = mat2(c, -s, s, c);
    vec2 rotated_ellipse_xy = rotation_matrix * base_ellipse_shape;

    vec3 scaled_cone_pos;
    if (tiled_mode) {
        scaled_cone_pos.xy = rotated_ellipse_xy * receptive_field_scale * 2.5;
    } else {
        scaled_cone_pos.xy = rotated_ellipse_xy * receptive_field_scale;
    }
    scaled_cone_pos.z = cone_vertex.z;

    vec3 final_pos = scaled_cone_pos + vec3(instance_screen_pos, 0.0);
    final_pos.x /= aspect_ratio;

    gl_Position = vec4(final_pos, 1.0);

    // normalised value with dynamic range compr
    float raw = scalar_data[instance_id];
    float range = data_max - data_min;
    float t = (range > 1e-8) ? clamp((raw - data_min) / range, 0.0, 1.0) : 0.5;

    if (colormap == 0) {
        // Diverging: compress symmetrically around midpoint
        float centered = t * 2.0 - 1.0;  // [0,1] → [-1,1]
        float compressed = sign(centered) * pow(abs(centered), compression);
        t = compressed * 0.5 + 0.5;       // [-1,1] → [0,1]
    } else {
        // Sequential or thermal: compress from zero
        t = pow(t, compression);
    }

    v_scalar = t;
}
