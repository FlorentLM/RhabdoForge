#version 430 core

in vec3 vertColor;
out vec4 FragColor;

void main() {
    // Convert the sRGB color to linear space
    vec3 linearColor = pow(vertColor, vec3(2.2));

    // GL_FRAMEBUFFER_SRGB is enabled, OpenGL expects a linear color
    // It will handle the final conversion back to sRGB for the monitor
    FragColor = vec4(linearColor, 1.0);
}