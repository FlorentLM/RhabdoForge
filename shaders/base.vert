#version 430 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 vertTexCoord;

out vec2 fragTexCoord;

uniform mat4 camera; // This will be Projection * View
uniform mat4 model;

void main()
{
    fragTexCoord = vertTexCoord;
    // The C-side code provides row-major matrices, but glUniformMatrix4fv with
    // transpose=GL_FALSE causes OpenGL to read them as column-major, which transposes them
    // The C-side multiplication order is thus camera = view * proj, which becomes proj * view here
    gl_Position = camera * model * vec4(pos, 1.0);
}