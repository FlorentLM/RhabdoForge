#version 430 core
in vec2 v_tex_coord;
out vec4 f_color;

uniform sampler2D hdr_scene;
uniform float exposure;

float aces_scalar(float x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 c = texture(hdr_scene, v_tex_coord).rgb * exposure;

    float L  = dot(c, vec3(0.2126, 0.7152, 0.0722));   // Rec.709 luminance
    float Lt = aces_scalar(L);                 // curve applied to luminance only
    c *= (L > 1e-6) ? (Lt / L) : 0.0;          // ratio scale, so chroma untouched

    f_color = vec4(c, 1.0);   // GL_FRAMEBUFFER_SRGB encodes on write
}