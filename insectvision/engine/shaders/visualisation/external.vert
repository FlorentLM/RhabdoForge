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

layout(std430, binding = BINDING_RHAB_STATIC)  readonly buffer RcptStaticBlock  { RhabdomereStatic  rhab_static[]; };
layout(std430, binding = BINDING_OMM_STATIC)   readonly buffer OmmatidiumStaticBlock  { OmmatidiumStatic      omm_static[]; };
layout(std430, binding = BINDING_RHAB_DYNAMIC) readonly buffer RcptDynamicBlock { RhabdomereDynamic rhab_dynamic[]; };
layout(std430, binding = BINDING_COLORS)       readonly buffer ColorBlock       { vec4 color_data[]; };

#ifdef OVERLAY_MODE
layout(std430, binding = BINDING_OVERLAY) readonly buffer DataBlock { float scalar_data[]; };
layout (location = 0) out float v_scalar;
uniform float overlay_data_min;
uniform float overlay_data_max;
uniform int overlay_colormap;
uniform float overlay_compression;
uniform bool overlay_fallback;
#else
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
uniform float visualisation_omm_length;
uniform float visualisation_eyes_scale;
uniform float visualisation_saccade_scale;
uniform int selected_ommatidia[10];
uniform int frame_offset;

uniform int output_mode;    // 0 = Raw, 1 = Ommatidium, 2 = Cartridge
uniform int rhab_per_omm;
uniform int bundle_centre_idx;

mat3 rmatFromDir(vec3 z) {
    vec3 x = normalize(cross((abs(z.y) > 0.999) ? vec3(1,0,0) : vec3(0,1,0), z));
    vec3 y = cross(z, x);
    return mat3(x, y, z);
}

void main() {
    int i = gl_InstanceID;
    v_instance_id = i;
    v_local_pos = model_vertex.xy;
    uint base = uint(frame_offset) * rhab_static.length();

    uint omm_idx;
    RhabdomereStatic rs;
    RhabdomereDynamic rd;

    #ifdef OVERLAY_MODE
    float value = 0.0;
    #else
    vec3 value = vec3(0.0);
    #endif

    if (output_mode == 0) {
        rs = rhab_static[i]; rd = rhab_dynamic[i];
        omm_idx = unpack_omm_id(rs.metadata);
        #ifdef OVERLAY_MODE
        if (overlay_fallback) {
//            value = color_data[base + i].w;
//            value = dot(color_data[base + i].rgb, vec3(0.299, 0.587, 0.114));
            value = rhab_dynamic[i].curr_adaptation;
        } else {
            value = scalar_data[base + i];
        }
        #else
        value = color_data[base + i].rgb;
        #endif

    } else {
        omm_idx = uint(i);
        uint c_idx = omm_idx * uint(rhab_per_omm) + uint(bundle_centre_idx);
        rs = rhab_static[c_idx]; rd = rhab_dynamic[c_idx];

        for (int r = 0; r < rhab_per_omm; r++) {

            uint src;
            if (output_mode == 2) {
                src = rhab_static[omm_idx * rhab_per_omm + r].cartridge_src;
            } else {
                src = omm_idx * rhab_per_omm + r;
            }

            #ifdef OVERLAY_MODE
            if (overlay_fallback) {
//                value = color_data[base + i].w;
//                value += dot(color_data[base + src].rgb, vec3(0.299, 0.587, 0.114));
                value += rhab_dynamic[src].curr_adaptation;
            } else {
                value += scalar_data[base + src];
            }
            #else
            value += color_data[base + src].rgb;
            count++;
            #endif
        }
        value /= float(rhab_per_omm);
    }

    v_mode = uint(projection_mode);
    v_eye_id = unpack_eye_id(rs.metadata);

    #ifdef OVERLAY_MODE
    v_scalar = normalize_scalar(value, overlay_data_min, overlay_data_max, overlay_compression, overlay_colormap);
    #else
    v_color = value;
    #endif

    OmmatidiumStatic os = omm_static[omm_idx];

    // Nudge ommatidia by their actuation values
    vec3 pos_local = os.position * visualisation_eyes_scale;
    if (projection_mode == 0) {
        OmmatidiumStatic ls_nudge = os;
        vec3 tangent_shift = rd.curr_direction - ls_nudge.forward * dot(rd.curr_direction, ls_nudge.forward);
        pos_local += tangent_shift * visualisation_saccade_scale;
    }

    vec3 P_world = (eye_to_world * vec4(pos_local, 1.0)).xyz;
    vec3 D_world = normalize((eye_to_world * vec4(rd.curr_direction, 0.0)).xyz);

    mat3 R_world = rmatFromDir(D_world);
    float tilt = os.ioa_tilt;
    mat3 R_tilt = mat3(cos(tilt), -sin(tilt), 0, sin(tilt), cos(tilt), 0, 0, 0, 1);

    vec2 axes = (projection_mode == 1) ? rd.curr_acc_angles : os.ioa_angles;
    vec3 scale;
    if (projection_mode == 1) {
        scale = vec3(visualisation_omm_length * tan(axes * 0.5), visualisation_omm_length);

    } else {
        float rad = length(os.position) * visualisation_eyes_scale;
        vec2 radii = rad * sin(axes * 0.5);
        scale = vec3(radii, (radii.x + radii.y) * 0.15);
    }

    vec3 v = (projection_mode == 1) ? model_vertex + vec3(0, 0, 1) : model_vertex;
    v_world_normal = normalize(R_world * R_tilt * vec3(0, 0, 1));

    gl_Position = projection * view * vec4(P_world + (R_world * R_tilt * (v * scale)), 1.0);
}