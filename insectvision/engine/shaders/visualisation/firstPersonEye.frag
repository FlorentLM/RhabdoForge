#version 430 core

// First-person compound-eye visualisation
//
// Without OVERLAY_MODE: pass-through RGB
// With OVERLAY_MODE: apply colormap LUT to scalar

#ifdef OVERLAY_MODE
#include "colormaps.glsl"

layout (location = 0) in float v_scalar;
uniform int colormap;
#else
layout (location = 0) in vec3 v_color;
#endif

layout (location = 0) out vec4 FragColor;

void main() {
    #ifdef OVERLAY_MODE
    FragColor = vec4(apply_colormap(v_scalar, colormap), 1.0);
    #else
    FragColor = vec4(v_color, 1.0);
    #endif
}
