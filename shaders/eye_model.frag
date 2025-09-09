#version 430 core

layout(std430, binding = 1) readonly buffer ColorDataBlock {
    vec4 final_rgba[];
};

in flat int v_index;
flat in int v_mode; // Receive mode from vertex shader

out vec4 FragColor;

uniform float albedo_boost = 1.0;

void main() {
    if (v_mode == 1) {
        // Acceptance Mode: Use a fixed translucent white color
        FragColor = vec4(1.0, 1.0, 1.0, 0.1);
    } else {
        // Physical Layout Mode: Use the ray-traced color
        vec4 c = final_rgba[v_index];
        FragColor = vec4(clamp(c.rgb * albedo_boost, 0.0, 1.0), 1.0);
    }
}
