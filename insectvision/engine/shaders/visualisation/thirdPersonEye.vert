#version 430 core

// Third-person 3D eye model visualisation
//
// Without OVERLAY_MODE: physical mode shows simulation colour,
//                       acceptance mode shows per-eye ID colour
// With OVERLAY_MODE: both modes show scalar data through a LUT

#include "commons.glsl"

#ifdef OVERLAY_MODE
#include "colormaps.glsl"
#endif

layout(location = 0) in vec3 model_vertex;

layout(std430, binding = 0)  readonly buffer ReceptorsInputBlock { ReceptorData receptor_data[]; };
layout(std430, binding = 1)  readonly buffer LensDataBlock       { LensData lenses_data[]; };
layout(std430, binding = 17) readonly buffer CartridgeBlock     { uint cartridge_map[]; };

#ifdef OVERLAY_MODE
layout(std430, binding = 2) readonly buffer DataBlock { float scalar_data[]; };
uniform float overlay_data_min;
uniform float overlay_data_max;
uniform int colormap;
uniform float compression;
#else
layout(std430, binding = 2) readonly buffer DataBlock { vec4 color_data[]; };
#endif

// Uniforms
uniform mat4 view;
uniform mat4 projection;
uniform mat4 eye_to_world;

uniform int projection_mode;
uniform float cone_length;
uniform float visualisation_scale;

// Eye output uniforms
uniform int output_mode;    // 0 = Raw, 1 = Ommatidium, 2 = Cartridge
uniform int receptor_count;
uniform int center_index;

flat out uint v_mode;
flat out uint v_eye_id;
out vec3 v_world_normal;

#ifdef OVERLAY_MODE
out float v_scalar;
#else
out vec3 v_color;
#endif

mat3 rmatFromDir(vec3 z) {
    vec3 x = normalize(cross( (abs(z.y) > 0.999) ? vec3(1,0,0) : vec3(0,1,0), z));
    vec3 y = cross(z, x);
    return mat3(x, y, z);
}

void main() {
    int i = gl_InstanceID;

    // Resolve receptor data and value based on output mode
    uint lens_id;
    ReceptorData rcpt;

    #ifdef OVERLAY_MODE
    float value = 0.0;
    #else
    vec3 value = vec3(0.0);
    #endif

    if (output_mode == 0) {
        // Raw: one instance per receptor (N*R)
        rcpt = receptor_data[i];
        lens_id = unpack_lens_id(rcpt);

        #ifdef OVERLAY_MODE
        value = scalar_data[i];
        #else
        value = color_data[i].rgb;
        #endif

    } else {
        // Ommatidium or Cartridge: one instance per lens (N)
        lens_id = uint(i);
        uint central_idx = lens_id * uint(receptor_count) + uint(center_index);
        rcpt = receptor_data[central_idx];

        // Pool across R receptors
        for (int r = 0; r < receptor_count; r++) {
            uint src;
            if (output_mode == 2) {
                uint src_lens = cartridge_map[i * receptor_count + r];
                src = src_lens * uint(receptor_count) + uint(r);
            } else {
                src = lens_id * uint(receptor_count) + uint(r);
            }

            #ifdef OVERLAY_MODE
            value += scalar_data[src];
            #else
            value += color_data[src].rgb;
            #endif
        }
        value /= float(receptor_count);
    }

    LensData lens = lenses_data[lens_id];

    v_mode = projection_mode;
    v_eye_id = unpack_eye_id(rcpt);

    #ifdef OVERLAY_MODE
    v_scalar = normalize_scalar(value, overlay_data_min, overlay_data_max, compression, colormap);
    #else
    v_color = value;
    #endif

    // Geometry (shared)

    vec3 P_world = (eye_to_world * vec4(rcpt.position * visualisation_scale, 1.0)).xyz;
    vec3 D_world = normalize((eye_to_world * vec4(rcpt.direction, 0.0)).xyz);

    mat3 R_world = rmatFromDir(D_world);
    mat3 R_tilt;
    mat3 S;

    if (projection_mode == 1) {
        float tilt = rcpt.acc_tilt;
        R_tilt = mat3(cos(tilt), -sin(tilt), 0,
                      sin(tilt),  cos(tilt), 0,
                      0,          0,         1);

        float half_acc_minor = 0.5 * rcpt.acc_axes.x;
        float half_acc_major = 0.5 * rcpt.acc_axes.y;
        float radius_minor = cone_length * tan(half_acc_minor);
        float radius_major = cone_length * tan(half_acc_major);

        S = mat3(radius_minor, 0, 0, 0, radius_major, 0, 0, 0, cone_length);
    }
    else {
        float tilt = lens.tilt;
        R_tilt = mat3(cos(tilt), -sin(tilt), 0,
                      sin(tilt),  cos(tilt), 0,
                      0,          0,         1);

        float eye_radius_world = length(rcpt.position) * visualisation_scale;
        if (eye_radius_world < 0.001) eye_radius_world = 0.01 * visualisation_scale;

        float half_ioa_minor = 0.5 * lens.ioa_axes.x;
        float half_ioa_major = 0.5 * lens.ioa_axes.y;
        float radius_minor = eye_radius_world * sin(half_ioa_minor);
        float radius_major = eye_radius_world * sin(half_ioa_major);

        float ovoid_height = (radius_minor + radius_major) * 0.15;

        S = mat3(radius_minor, 0, 0, 0, radius_major, 0, 0, 0, ovoid_height);
    }

    mat3 model_transform = R_world * R_tilt * S;

    vec3 v = (projection_mode == 1) ? model_vertex + vec3(0.0, 0.0, 1.0) : model_vertex;
    vec3 pos_world = P_world + (model_transform * v);

    mat3 normal_matrix = transpose(inverse(model_transform));
    v_world_normal = normalize(normal_matrix * v);

    gl_Position = projection * view * vec4(pos_world, 1.0);
}
