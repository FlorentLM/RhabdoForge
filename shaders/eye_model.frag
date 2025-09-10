#version 430 core

layout(std430, binding = 1) readonly buffer ColorDataBlock {
    vec4 final_rgba[];
};

in flat uint v_index;
flat in uint v_mode;
flat in uint v_eye_id;

out vec4 FragColor;

uniform float albedo_boost = 1.0;

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

void main() {
    if (v_mode == 1u) {
        // Acceptance Mode: Colour according to eye ID
        FragColor = EYE_COLORS[v_eye_id];
    } else {
        // Physical Layout Mode: Use the ray-traced color
        vec4 c = final_rgba[v_index];
        FragColor = vec4(clamp(c.rgb * albedo_boost, 0.0, 1.0), 1.0);
    }
}
