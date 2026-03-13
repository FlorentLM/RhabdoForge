#version 430 core

#include "commons.glsl"

layout (location = 0) in vec3 cone_vertex;

layout(std430, binding = 0) readonly buffer ReceptorsInputBlock { ReceptorData receptors_data[]; };
layout(std430, binding = 1) readonly buffer LensDataBlock { LensData lenses_data[]; };
layout(std430, binding = 2) readonly buffer ScalarDataBlock { float scalar_data[]; };

// Uniforms
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
    ReceptorData rcpt = receptors_data[instance_id];

    uint lens_id = unpack_lens_id(rcpt);
    LensData lens = lenses_data[lens_id];

    vec3 proj_vec = (projection_mode == 1) ? rcpt.direction : normalize(rcpt.position);

    float longitude = atan(proj_vec.x, -proj_vec.z);
    float latitude  = asin(proj_vec.y);
    vec2 instance_screen_pos = vec2(longitude / PI, latitude / HPI);

    float tilt_angle;
    vec2 base_ellipse_shape;

    if (tiled_mode) {
        tilt_angle = lens.tilt;
        base_ellipse_shape = cone_vertex.xy * lens.ioa_axes;
    } else {
        tilt_angle = rcpt.acc_tilt;
        base_ellipse_shape = cone_vertex.xy * rcpt.acc_axes;
    }

    float s = sin(tilt_angle);
    float c = cos(tilt_angle);
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

    // Normalise scalar value with dynamic range compr
    float raw = scalar_data[instance_id];
    float range = data_max - data_min;
    float t = (range > 1e-8) ? clamp((raw - data_min) / range, 0.0, 1.0) : 0.5;

    if (colormap == 0) {
        // Diverging: compress symmetrically around midpoint (0.5)
        float centered = t * 2.0 - 1.0;  // [0,1] → [-1,1]
        float compressed = sign(centered) * pow(abs(centered), compression);
        t = compressed * 0.5 + 0.5;       // [-1,1] → [0,1]
    } else {
        // Sequential or thermal: compress from zero
        t = pow(t, compression);
    }

    v_scalar = t;
}
