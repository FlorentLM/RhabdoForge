#version 430 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 vertTexCoord;

out vec2 fragTexCoord;

uniform mat4 camera; // This will be Projection * View
uniform mat4 model;

void main()
{
    fragTexCoord = vertTexCoord;
    // This is column-major multiplication, which is what we want inside GLSL
    // glUniformMatrix with GL_TRUE ensures the matrices arrive in the correct format
    gl_Position = camera * model * vec4(pos, 1.0);
}