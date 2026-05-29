#version 430 core
#include "commons.glsl"

// First-person compound-eye visualisation
//
// Without OVERLAY_MODE: pass-through RGB
// With OVERLAY_MODE: apply overlay_colormap LUT to scalar

#ifdef OVERLAY_MODE
#include "colormaps.glsl"

layout (location = 0) in float v_scalar;
uniform int overlay_colormap;
#else
layout (location = 0) in vec3  v_color;
#endif

layout (location = 1) in vec2 v_local_pos;
layout (location = 2) flat in int v_instance_id;
layout (location = 3) flat in float v_select_f;

uniform int selected_lenses[10];
uniform bool false_colors;
uniform bool uv_encoding;

layout (location = 0) out vec4 FragColor;

void main() {
    #ifdef OVERLAY_MODE
    vec3 base_rgb = apply_overlay_colormap(v_scalar, overlay_colormap);
    #else
    vec3 base_rgb = v_color;

    if (uv_encoding) {
        // Channel 0 is UV -> make it look purple
        base_rgb = vec3(v_color.r, v_color.g, min(1.0, v_color.b + v_color.r));
    } else if (false_colors) {
        // Normal textures, but simulating an insect that can't see Red
        base_rgb = vec3(0.0, v_color.g, v_color.b);
    }
    #endif

    // Calculate rim/contour
    float dist = length(v_local_pos);
    float rim_mask = smoothstep(0.75, 0.99, dist);

    vec3 highlight_color = vec3(1.0, 1.0, 0.0); // yellow
    vec3 final_rgb = mix(base_rgb, highlight_color, v_select_f * rim_mask);

    FragColor = vec4(final_rgb, 1.0);
}