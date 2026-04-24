#version 430 core

// Third-person 3D eye model visualisation
//
// Without OVERLAY_MODE: physical mode shows simulation colour,
//                       acceptance mode shows per-eye ID colour
// With OVERLAY_MODE: both modes show scalar data through a LUT

#ifdef OVERLAY_MODE
#include "colormaps.glsl"
layout (location = 0) in float v_scalar;
uniform int overlay_colormap;
#else
layout (location = 0) in vec3  v_color;

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
layout (location = 1) in vec2 v_local_pos;
layout (location = 2) flat in int  v_instance_id;
layout (location = 3) flat in uint v_mode;
layout (location = 4) flat in uint v_eye_id;
layout (location = 5) in vec3 v_world_normal;

uniform float visualisation_eye_surface_albedo;
uniform int selected_lens;
uniform bool false_colors;
uniform bool uv_encoding;

out vec4 FragColor;

void main() {
    #ifdef OVERLAY_MODE
    vec3 rgb = apply_overlay_colormap(v_scalar, overlay_colormap);
    FragColor = vec4(clamp(rgb * visualisation_eye_surface_albedo, 0.0, 1.0), 1.0);

    #else
    if (v_mode == 1u) {
        // Acceptance mode: eye ID color
        FragColor = EYE_COLORS[v_eye_id];
    } else {
        // Physical mode: simulation color
        vec3 lit_rgb = v_color * visualisation_eye_surface_albedo;

        if (uv_encoding) {
            // Channel 0 is UV -> make it look purple
            lit_rgb = vec3(lit_rgb.r, lit_rgb.g, min(1.0, lit_rgb.b + lit_rgb.r));
        } else if (false_colors) {
            // Normal textures, but simulating an insect that can't see Red
            lit_rgb = vec3(0.0, lit_rgb.g, lit_rgb.b);
        }

        FragColor = vec4(clamp(lit_rgb, 0.0, 1.0), 1.0);
    }
    #endif

    // Highlight one ommatidia
    float is_selected = 1.0 - clamp(abs(float(v_instance_id - selected_lens)), 0.0, 1.0);
    float dist = length(v_local_pos);

    float contour = smoothstep(0.85, 0.90, dist);

    float inner_glow = 0.4;

    vec3 highlight_color = vec3(1.0, 1.0, 0.0);
    float total_highlight = is_selected * (contour + inner_glow);

    vec3 final_rgb = mix(FragColor.rgb, highlight_color, total_highlight);

    FragColor = vec4(final_rgb, 1.0);
}
