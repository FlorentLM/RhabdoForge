#version 430 core

// Input: Varying from vertex shader
layout (location = 0) in vec3 v_color;

// Output: Final color to framebuffer
layout (location = 0) out vec4 FragColor;

void main() {
    FragColor = vec4(v_color, 1.0);
}