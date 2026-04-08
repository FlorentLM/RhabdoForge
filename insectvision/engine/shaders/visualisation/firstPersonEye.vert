#version 430 core

// First-person compound-eye visualisation
//
// Without OVERLAY_MODE: pass-through RGB
// With OVERLAY_MODE: apply colormap LUT to scalar

#include "commons.glsl"

#ifdef OVERLAY_MODE
#include "colormaps.glsl"
#endif

layout (location = 0) in vec3 cone_vertex;

layout(std430, binding = 0)  readonly buffer ReceptorsDataBlock { ReceptorData receptors_data[]; };
layout(std430, binding = 1)  readonly buffer LensDataBlock      { LensData lenses_data[]; };
layout(std430, binding = 17) readonly buffer CartridgeBlock     { uint cartridge_map[]; };

#ifdef OVERLAY_MODE
layout(std430, binding = 2) readonly buffer DataBlock { float scalar_data[]; };
layout (location = 0) out float v_scalar;
uniform float overlay_data_min;
uniform float overlay_data_max;
uniform int colormap;
uniform float compression;
#else
layout(std430, binding = 2) readonly buffer DataBlock { vec4 color_data[]; };
layout (location = 0) out vec3 v_color;
#endif

// View uniforms
uniform float aspect_ratio;
uniform int projection_mode;    // 0 = Physical, 1 = Acceptance
uniform bool tiled_mode;
uniform float receptive_field_scale;

// Eye output uniforms
uniform int output_mode;    // 0 = Raw, 1 = Ommatidium, 2 = Cartridge
uniform int receptor_count;                  // receptors per lens
uniform int center_index;       // kernel center receptor index


void main() {
    int instance_id = gl_InstanceID;

    uint lens_id;
    vec3 projection_vector;

    // Per-instance data value (rgb or scalar depending on mode)
    #ifdef OVERLAY_MODE
    float value = 0.0;
    #else
    vec3 value = vec3(0.0);
    #endif

    if (output_mode == 0) {
        // Raw: one instance per receptor (N*R total)
        ReceptorData rcpt = receptors_data[instance_id];
        lens_id = unpack_lens_id(rcpt);
        projection_vector = (projection_mode == 1) ? rcpt.direction : normalize(rcpt.position);

        #ifdef OVERLAY_MODE
        value = scalar_data[instance_id];
        #else
        value = color_data[instance_id].rgb;
        #endif

    } else {
        // Ommatidium or Cartridge: one instance per lens (N total)
        lens_id = uint(instance_id);
        uint central_idx = lens_id * uint(receptor_count) + uint(center_index);
        ReceptorData central = receptors_data[central_idx];
        projection_vector = (projection_mode == 1) ? central.direction : normalize(central.position);

        // Pool across R receptors
        for (int r = 0; r < receptor_count; r++) {
            uint src;
            if (output_mode == 2) {
                // Cartridge: neural superposition wiring
                uint src_lens = cartridge_map[instance_id * receptor_count + r];
                src = src_lens * uint(receptor_count) + uint(r);
            } else {
                // Ommatidium: colocated receptors
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

    // Geometry (shared)

    LensData lens = lenses_data[lens_id];

    // Spherical projection to screen position
    float longitude = atan(projection_vector.x, -projection_vector.z);
    float latitude  = asin(projection_vector.y);
    vec2 instance_screen_pos = vec2(longitude / PI, latitude / HPI);

    // Tile shape
    float tilt_angle;
    vec2 base_ellipse_shape;

    if (tiled_mode) {
        tilt_angle = lens.tilt;
        base_ellipse_shape = cone_vertex.xy * lens.ioa_axes;
    } else {
        if (output_mode == 0) {
            ReceptorData rcpt = receptors_data[instance_id];
            tilt_angle = rcpt.acc_tilt;
            base_ellipse_shape = cone_vertex.xy * rcpt.acc_axes;
        } else {
            uint central_idx = lens_id * uint(receptor_count) + uint(center_index);
            ReceptorData central = receptors_data[central_idx];
            tilt_angle = central.acc_tilt;
            base_ellipse_shape = cone_vertex.xy * central.acc_axes;
        }
    }

    float s = sin(tilt_angle);
    float c = cos(tilt_angle);
    vec2 rotated_ellipse_xy = mat2(c, -s, s, c) * base_ellipse_shape;

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

    // Output

    #ifdef OVERLAY_MODE
    v_scalar = normalize_scalar(value, overlay_data_min, overlay_data_max, compression, colormap);
    #else
    v_color = value;
    #endif
}
