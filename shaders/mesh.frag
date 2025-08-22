#version 430 core

// Input: Varying from vertex shader
layout (location = 0) in vec2 fragTexCoord;

// Output: Final color to framebuffer
layout (location = 0) out vec4 finalColor;

// Uniforms
layout (binding = 0) uniform sampler2D texture1;

void main() {
    finalColor = texture(texture1, fragTexCoord);
}