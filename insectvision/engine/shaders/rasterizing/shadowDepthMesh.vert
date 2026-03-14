#version 430 core

// Shadow depth pass: renders geometry from the light's pov

layout (location = 0) in vec3 position;

uniform mat4 light_space_matrix;   // light projection * light view
uniform mat4 model;                // model-to-world transform

void main()
{
    gl_Position = light_space_matrix * model * vec4(position, 1.0);
}
