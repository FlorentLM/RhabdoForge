#version 430 core

// Input: Varying from vertex shader
layout (location = 0) in vec2 v_screen_pos;

// Output: Final color to framebuffer
layout (location = 0) out vec4 FragColor;

// Uniforms
layout (binding = 0) uniform samplerCube u_cubemap;

const float PI = 3.14159265359;

void main()
{
    // Convert the incoming screen position [-1, 1] into panoramic coordinates
    // longitude [-PI, PI] and latitude [-PI/2, PI/2]
    float longitude = v_screen_pos.x * PI;
    float latitude = v_screen_pos.y * (PI / 2.0);

    // Convert spherical coordinates back to a 3D direction vector
    // (note: this is the reverse of the math in William Martin's eul2geo)
    vec3 dir;
    dir.y = sin(latitude);
    float cos_lat = cos(latitude);
    dir.x = cos_lat * sin(longitude);
    dir.z = -cos_lat * cos(longitude);

    // Sample the cubemap
    FragColor = texture(u_cubemap, dir);
}