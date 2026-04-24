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

layout(std430, binding = BINDING_RCPT_STATIC)  readonly buffer RcptStaticBlock  { ReceptorStatic  rcpt_static[]; };
layout(std430, binding = BINDING_LENS_STATIC)  readonly buffer LensStaticBlock  { LensStatic      lens_static[]; };
layout(std430, binding = BINDING_RCPT_DYNAMIC) readonly buffer RcptDynamicBlock { ReceptorDynamic rcpt_dynamic[]; };

#ifdef OVERLAY_MODE
layout(std430, binding = BINDING_COLORS) readonly buffer DataBlock { float scalar_data[]; };
layout (location = 0) out float v_scalar;
uniform float overlay_data_min;
uniform float overlay_data_max;
uniform int colormap;
uniform float compression;
#else
layout(std430, binding = BINDING_COLORS) readonly buffer DataBlock { vec4 color_data[]; };
layout (location = 0) out vec3 v_color;
#endif

layout (location = 0) in vec3 model_vertex;
layout (location = 1) out vec2 v_local_pos;
layout (location = 2) flat out int  v_instance_id;
layout (location = 3) flat out uint v_mode;
layout (location = 4) flat out uint v_eye_id;
layout (location = 5) out vec3 v_world_normal;

uniform mat4 view;
uniform mat4 projection;
uniform mat4 eye_to_world;
uniform int projection_mode;
uniform float cone_length;
uniform float visualisation_scale;
uniform float saccade_visualisation_gain;
uniform int selected_id;
uniform int frame_offset;

uniform int output_mode;    // 0 = Raw, 1 = Ommatidium, 2 = Cartridge
uniform int receptor_count;
uniform int center_index;

mat3 rmatFromDir(vec3 z) {
    vec3 x = normalize(cross((abs(z.y) > 0.999) ? vec3(1,0,0) : vec3(0,1,0), z));
    vec3 y = cross(z, x);
    return mat3(x, y, z);
}

void main() {
    int i = gl_InstanceID;
    v_instance_id = i;
    v_local_pos = model_vertex.xy;
    uint base = uint(frame_offset) * rcpt_static.length();

    uint l_id;
    ReceptorStatic rs;
    ReceptorDynamic rd;

    #ifdef OVERLAY_MODE
    float val = 0.0;
    #else
    vec3 val = vec3(0.0);
    #endif

    if (output_mode == 0) {
        rs = rcpt_static[i]; rd = rcpt_dynamic[i];
        l_id = unpack_lens_id(rs.metadata);
        #ifdef OVERLAY_MODE
        val = scalar_data[base + i];
        #else
        val = color_data[base + i].rgb;
        #endif

    } else {
        l_id = uint(i);
        uint c_idx = l_id * uint(receptor_count) + uint(center_index);
        rs = rcpt_static[c_idx]; rd = rcpt_dynamic[c_idx];
        for (int r = 0; r < receptor_count; r++) {
            uint src = (output_mode == 2) ? rcpt_static[l_id * receptor_count + r].cartridge_src : (l_id * receptor_count + r);
            #ifdef OVERLAY_MODE
            val += scalar_data[base + src];
            #else
            val += color_data[base + src].rgb;
            #endif
        }
        val /= float(receptor_count);
    }

    v_mode = uint(projection_mode);
    v_eye_id = unpack_eye_id(rs.metadata);

    #ifdef OVERLAY_MODE
    v_scalar = normalize_scalar(val, overlay_data_min, overlay_data_max, compression, colormap);
    #else
    v_color = val;
    #endif

    LensStatic ls = lens_static[l_id];

    // Nudge ommatidia by their actuation values
    vec3 pos_local = rs.position * visualisation_scale;
    if (projection_mode == 0) {
        LensStatic ls_nudge = ls;
        vec3 tangent_shift = rd.direction - ls_nudge.forward * dot(rd.direction, ls_nudge.forward);
        pos_local += tangent_shift * saccade_visualisation_gain;
    }

    vec3 P_world = (eye_to_world * vec4(pos_local, 1.0)).xyz;
    vec3 D_world = normalize((eye_to_world * vec4(rd.direction, 0.0)).xyz);

    mat3 R_world = rmatFromDir(D_world);
    float tilt = (projection_mode == 1) ? rs.acc_tilt : ls.ioa_tilt;
    mat3 R_tilt = mat3(cos(tilt), -sin(tilt), 0, sin(tilt), cos(tilt), 0, 0, 0, 1);

    vec2 axes = (projection_mode == 1) ? rd.acc_axes : ls.ioa_axes;
    vec3 scale;
    if (projection_mode == 1) {
        scale = vec3(cone_length * tan(axes * 0.5), cone_length);

    } else {
        float rad = length(rs.position) * visualisation_scale;
        vec2 radii = rad * sin(axes * 0.5);
        scale = vec3(radii, (radii.x + radii.y) * 0.15);
    }

    vec3 v = (projection_mode == 1) ? model_vertex + vec3(0, 0, 1) : model_vertex;
    v_world_normal = normalize(R_world * R_tilt * vec3(0, 0, 1));

    gl_Position = projection * view * vec4(P_world + (R_world * R_tilt * (v * scale)), 1.0);
}