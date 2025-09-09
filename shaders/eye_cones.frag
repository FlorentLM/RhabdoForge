#version 430 core

layout(std430, binding = 1) readonly buffer ColorDataBlock {
    vec4 final_rgba[];
};

in flat int v_index;

out vec4 FragColor;

uniform float albedo_boost = 1.0;

void main() {
    vec4 c = final_rgba[v_index];
    FragColor = vec4(clamp(c.rgb * albedo_boost, 0.0, 1.0), 1.0);
}
