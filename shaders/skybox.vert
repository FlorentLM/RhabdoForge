#version 430 core

layout (location = 0) in vec3 pos;

out vec3 texCoords;

// Only the projection and the rotational part of the view matrix are needed
uniform mat4 projection;
uniform mat4 view; // This will be the view matrix without translation

void main()
{
    texCoords = pos;
    // Remove translation from the view matrix by converting to mat3 and back
    mat4 view_no_translation = mat4(mat3(view));
    vec4 clip_pos = projection * view_no_translation * vec4(pos, 1.0);

    // Force the depth to be 1.0, so it's always in the background
    gl_Position = clip_pos.xyww;
}