#version 430 core

// Input: Varying from vertex shader
layout (location = 0) in vec2 v_tex_coord;

// Output: Final color to framebuffer
layout (location = 0) out vec4 FragColor;

// Uniforms
layout (binding = 0) uniform samplerCube cubemap;

const float PI = 3.14159265359;
const float HPI = PI * 0.5;

void main()
{
    // Convert incoming UVs [0, 1] to screen position [-1, 1]
     vec2 v_screen_pos = v_tex_coord * 2.0 - 1.0;

    // Convert the incoming screen position [-1, 1] into panoramic coordinates
    // longitude [-PI, PI] and latitude [-PI/2, PI/2]
    float longitude = v_screen_pos.x * PI;
    float latitude = v_screen_pos.y * HPI;

    // Convert spherical coordinates back to 3D direction vector
    vec3 dir;
    dir.y = sin(latitude);
    float cos_lat = cos(latitude);
    dir.x = cos_lat * sin(longitude);
    dir.z = -cos_lat * cos(longitude);

    // Sample the cubemap
    FragColor = texture(cubemap, dir);
}