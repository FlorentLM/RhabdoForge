#version 430 core
#include "commons.glsl"

#ifdef OVERLAY_MODE
#include "colormaps.glsl"
#endif

layout (location = 0) in vec3 cone_vertex;

layout(std430, binding = BINDING_RCPT_STATIC)  readonly buffer RcptStaticBlock  { ReceptorStatic  rcpt_static[]; };
layout(std430, binding = BINDING_LENS_STATIC)  readonly buffer LensStaticBlock  { LensStatic      lens_static[]; };
layout(std430, binding = BINDING_RCPT_DYNAMIC) readonly buffer RcptDynamicBlock { ReceptorDynamic rcpt_dynamic[]; };
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

layout (location = 1) out vec2 v_local_pos;
layout (location = 2) flat out int v_instance_id;
layout (location = 3) flat out float v_select_f;

uniform int frame_offset;
uniform float aspect_ratio;
uniform int projection_mode;
uniform bool tiled_mode;
uniform float vis_rf_scale;
uniform int output_mode;
uniform int receptors_per_lens;
uniform int bundle_centre_idx;
uniform int selected_lenses[10];

void main() {
    int inst_idx = gl_InstanceID;
    v_instance_id = inst_idx;
    v_local_pos = cone_vertex.xy;

    uint base = uint(frame_offset) * rcpt_static.length();

    uint l_id;
    vec3 p_vec;

    #ifdef OVERLAY_MODE
    float value = 0.0;
    #else
    vec3 value = vec3(0.0);
    #endif

    if (output_mode == 0) {
        l_id = unpack_lens_id(rcpt_static[inst_idx].metadata);
        p_vec = (projection_mode == 1) ? rcpt_dynamic[inst_idx].direction : normalize(rcpt_static[inst_idx].position);
        #ifdef OVERLAY_MODE
        if (overlay_fallback) {
//            value = color_data[base + i].w;
//            value = dot(color_data[base + inst_idx].rgb, vec3(0.299, 0.587, 0.114));
            value = rcpt_dynamic[inst_idx].adaptation_state;
        } else {
            value = scalar_data[base + inst_idx];
        }
        #else
        value = color_data[base + inst_idx].rgb;
        #endif
    } else {
        l_id = uint(inst_idx);
        uint c_idx = l_id * uint(receptors_per_lens) + uint(bundle_centre_idx);
        p_vec = (projection_mode == 1) ? rcpt_dynamic[c_idx].direction : normalize(rcpt_static[c_idx].position);

        for (int r = 0; r < receptors_per_lens; r++) {
            uint src;
            if (output_mode == 2) {
                src = rcpt_static[l_id * receptors_per_lens + r].cartridge_src;
            } else {
                src = l_id * uint(receptors_per_lens) + uint(r);
            }

            #ifdef OVERLAY_MODE
            if (overlay_fallback) {
//                value = color_data[base + i].w;
//                value += dot(color_data[base + src].rgb, vec3(0.299, 0.587, 0.114));
                value = rcpt_dynamic[inst_idx].adaptation_state;
            } else {
                value += scalar_data[base + src];
            }
            #else
            value += color_data[base + src].rgb;
            #endif
        }
        value /= float(receptors_per_lens);
    }

    float tilt = tiled_mode ? lens_static[l_id].ioa_tilt : rcpt_static[output_mode == 0 ? inst_idx : l_id * receptors_per_lens + bundle_centre_idx].acc_tilt;
    vec2 axes = tiled_mode ? lens_static[l_id].ioa_axes : rcpt_dynamic[output_mode == 0 ? inst_idx : l_id * receptors_per_lens + bundle_centre_idx].acc_axes;

    uint rcpt_idx = (output_mode == 0 ? inst_idx : l_id * receptors_per_lens + bundle_centre_idx);
    vec2 dynamic_axes = rcpt_dynamic[rcpt_idx].acc_axes;
    vec2 rest_axes = rcpt_static[rcpt_idx].rest_acc;

    // axial contraction (narrowing)
    vec2 acc_scale_factor = dynamic_axes / max(rest_axes, 1e-6);

    vec2 base_axes = tiled_mode ? lens_static[l_id].ioa_axes : dynamic_axes;
    vec2 final_axes = base_axes * (tiled_mode ? acc_scale_factor : vec2(1.0));

    // Highlight selected lenses
    float select_f = 0.0;
    for (int j = 0; j < 10; j++) {
        if (selected_lenses[j] != -1 && uint(selected_lenses[j]) == uint(inst_idx)) {
            select_f = 1.0;
            break;
        }
    }
    v_select_f = select_f;

    float s = sin(tilt), c = cos(tilt);
    vec2 rot = cone_vertex.xy * final_axes * vis_rf_scale;

    vec2 rotated_offset = mat2(c, -s, s, c) * rot * (1.0 + select_f * 0.1);

    // Z-Voronoi logic
    // flatten slightly so they don't clip the far plane
    float z_voronoi = cone_vertex.z * 0.5;

    // If selected, subtract from Z, if not, add an offset to stay behind
    z_voronoi += (select_f > 0.5) ? -0.8 : 0.2;

    float longi = atan(p_vec.x, -p_vec.z);
    float lati = asin(p_vec.y);
    vec3 pos = vec3(rotated_offset, z_voronoi) + vec3(longi/PI, lati/HPI, 0.0);

    pos.x /= aspect_ratio;

    gl_Position = vec4(pos, 1.0);

    #ifdef OVERLAY_MODE
    v_scalar = normalize_scalar(value, overlay_data_min, overlay_data_max, overlay_compression, overlay_colormap);
    #else
    v_color = value;
    #endif
}