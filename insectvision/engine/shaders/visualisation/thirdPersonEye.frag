#version 430 core

// Third-person 3D eye model visualisation
//
// Without OVERLAY_MODE: physical mode shows simulation colour,
//                       acceptance mode shows per-eye ID colour
// With OVERLAY_MODE: both modes show scalar data through a LUT

flat in uint v_mode;
flat in uint v_eye_id;
in vec3 v_world_normal;

uniform float albedo_boost = 1.0;

out vec4 FragColor;

#ifdef OVERLAY_MODE

#include "colormaps.glsl"

in float v_scalar;
uniform int colormap;

#else

in vec3 v_color;

const vec4 EYE_COLORS[8] = vec4[](
    vec4(1.0, 0.2, 0.2, 0.25), // 0: Red
    vec4(0.2, 0.5, 1.0, 0.25), // 1: Blue
    vec4(0.2, 1.0, 0.2, 0.25), // 2: Green
    vec4(1.0, 1.0, 0.2, 0.25), // 3: Yellow
    vec4(0.2, 1.0, 1.0, 0.25), // 4: Cyan
    vec4(1.0, 0.2, 1.0, 0.25), // 5: Magenta
    vec4(1.0, 0.6, 0.2, 0.25), // 6: Orange
    vec4(0.6, 0.2, 1.0, 0.25)  // 7: Purple
);

#endif

void main() {
    #ifdef OVERLAY_MODE
    vec3 rgb = apply_colormap(v_scalar, colormap);
    FragColor = vec4(clamp(rgb * albedo_boost, 0.0, 1.0), 1.0);

    #else
    if (v_mode == 1u) {
        // Acceptance mode: eye ID color
        FragColor = EYE_COLORS[v_eye_id];
    } else {
        // Physical mode: simulation color
        vec3 lit_rgb = v_color * albedo_boost;
        FragColor = vec4(clamp(lit_rgb, 0.0, 1.0), 1.0);
    }
    #endif
}
