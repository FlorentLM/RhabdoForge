#version 430 core
#include "commons.glsl"

// First-person compound-eye visualisation
//
// Without OVERLAY_MODE: pass-through RGB
// With OVERLAY_MODE: apply colormap LUT to scalar

#ifdef OVERLAY_MODE
#include "colormaps.glsl"

layout (location = 0) in float v_scalar;
uniform int colormap;
#else
layout (location = 0) in vec3  v_color;
#endif
layout (location = 1) in vec2 v_local_pos;
layout (location = 2) flat in int v_instance_id;

uniform int selected_id;

layout (location = 0) out vec4 FragColor;

void main() {
    #ifdef OVERLAY_MODE
    vec3 base_rgb = apply_colormap(v_scalar, colormap);
    #else
    vec3 base_rgb = v_color;
    #endif

    float is_selected = 1.0 - clamp(abs(float(v_instance_id - selected_id)), 0.0, 1.0);
    float dist = length(v_local_pos);

    float contour = smoothstep(0.85, 0.90, dist);
    float inner_glow = 0.4;

    vec3 highlight_color = vec3(1.0, 1.0, 0.0); // yellow highlight
    float total_highlight = is_selected * (contour + inner_glow);

    FragColor = vec4(mix(base_rgb, highlight_color, total_highlight), 1.0);
}