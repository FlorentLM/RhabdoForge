#version 430 core

// Input: Varying from vertex shader
layout (location = 0) in vec3 texCoords;

// Output: Final color to framebuffer
layout (location = 0) out vec4 FragColor;

// Uniforms
layout (binding = 0) uniform samplerCube skybox;


void main()
{
    FragColor = texture(skybox, texCoords);
}