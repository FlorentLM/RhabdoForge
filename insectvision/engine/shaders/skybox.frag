#version 430 core

// Input: Varying from vertex shader
layout (location = 0) in vec3 texCoords;

// Output: Final color to framebuffer
layout (location = 0) out vec4 FragColor;

// Uniforms
layout (binding = 0) uniform samplerCube skybox;

uniform bool false_colors;
uniform bool uv_encoding;

void main()
{
    vec4 color = texture(skybox, texCoords);
    if (uv_encoding || false_colors) {
        color.r = 0.0;
    }
    FragColor = color;
}