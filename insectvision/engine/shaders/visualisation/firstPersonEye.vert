#version 430 core
#include "commons.glsl"

#ifdef OVERLAY_MODE
#include "colormaps.glsl"
#endif

layout (location = 0) in vec3 cone_vertex;

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

layout (location = 1) out vec2 v_local_pos;
layout (location = 2) flat out int v_instance_id;

uniform int frame_offset;
uniform float aspect_ratio;
uniform int projection_mode;
uniform bool tiled_mode;
uniform float receptive_field_scale;
uniform int output_mode;
uniform int receptor_count;
uniform int center_index;
uniform int selected_id;

void main() {
    int inst_idx = gl_InstanceID;
    v_instance_id = inst_idx;
    v_local_pos = cone_vertex.xy;

    // Calculate buffer offset for batched rendering
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
        value = scalar_data[base + inst_idx];
        #else
        value = color_data[base + inst_idx].rgb;
        #endif
    } else {
        l_id = uint(inst_idx);
        uint c_idx = l_id * uint(receptor_count) + uint(center_index);
        p_vec = (projection_mode == 1) ? rcpt_dynamic[c_idx].direction : normalize(rcpt_static[c_idx].position);

        for (int r = 0; r < receptor_count; r++) {
            uint src;
            if (output_mode == 2) {
                src = rcpt_static[l_id * receptor_count + r].cartridge_src;
            } else {
                src = l_id * uint(receptor_count) + uint(r);
            }

            #ifdef OVERLAY_MODE
            value += scalar_data[base + src];
            #else
            value += color_data[base + src].rgb;
            #endif
        }
        value /= float(receptor_count);
    }

    float tilt = tiled_mode ? lens_static[l_id].ioa_tilt : rcpt_static[output_mode == 0 ? inst_idx : l_id * receptor_count + center_index].acc_tilt;

    // Fetch axes from Dynamic (Binding 4) for contraction, or Static for layout
    vec2 axes = tiled_mode ? lens_static[l_id].ioa_axes : rcpt_dynamic[output_mode == 0 ? inst_idx : l_id * receptor_count + center_index].acc_axes;

    float is_sel = 1.0 - clamp(abs(float(inst_idx - selected_id)), 0.0, 1.0);
    float s = sin(tilt), c = cos(tilt);
    vec2 rot = mat2(c, -s, s, c) * (cone_vertex.xy * axes) * (1.0 + is_sel * 0.1);

    float longi = atan(p_vec.x, -p_vec.z), lati = asin(p_vec.y);
    vec3 pos = vec3(rot * receptive_field_scale * (tiled_mode ? 2.5 : 1.0), cone_vertex.z) + vec3(longi/PI, lati/HPI, 0.0);

    pos.z -= (is_sel * 0.8);
    pos.x /= aspect_ratio;
    gl_Position = vec4(pos, 1.0);

    #ifdef OVERLAY_MODE
    v_scalar = normalize_scalar(value, overlay_data_min, overlay_data_max, compression, colormap);
    #else
    v_color = value;
    #endif
}