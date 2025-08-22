#version 430 core

// Input: Vertex attributes from VBO
layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 vertTexCoord;

// Output: Varying to fragment shader
layout (location = 0) out vec2 fragTexCoord;

// Uniforms
uniform mat4 camera;    // pre-combined P * V matrix
uniform mat4 model;     // model-to-world transform matrix

void main()
{
    fragTexCoord = vertTexCoord;

    // Transform for column-major vertex is: P * V * M * v
    gl_Position = camera * model * vec4(pos, 1.0);
}