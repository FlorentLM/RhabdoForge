#version 430 core

// Shadow depth pass for point cloud splats
// Renders points from the light's pov with appropriate splat sizes

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;      // unused here but keeps VAO layout consistent
layout (location = 2) in float radius;

uniform mat4 light_space_matrix;   // light projection * light view
uniform mat4 model;                // model-to-world transform
uniform float radius_scale;        // TODO: same as visual pass or a separate shadow-specific scale??

void main()
{
    gl_Position = light_space_matrix * model * vec4(position, 1.0);

    // Scale point size for the light's projection
    gl_PointSize = radius * radius_scale;
}
