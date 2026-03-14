#version 430 core

layout (location = 0) in float v_scalar;   // normalised [0, 1]
layout (location = 0) out vec4 FragColor;

uniform int colormap;  // 0 = diverging (blue white red), 1 = sequential (viridis), 2 = thermal


vec3 colormap_diverging(float t) {
    vec3 blue = vec3(0.230, 0.299, 0.754);
    vec3 white = vec3(0.970, 0.970, 0.970);
    vec3 red = vec3(0.706, 0.016, 0.150);

    if (t < 0.5) {
        return mix(blue, white, t * 2.0);
    } else {
        return mix(white, red, (t - 0.5) * 2.0);
    }
}

vec3 colormap_sequential(float t) {
    vec3 c0 = vec3(0.267, 0.004, 0.329);
    vec3 c1 = vec3(0.282, 0.141, 0.458);
    vec3 c2 = vec3(0.127, 0.566, 0.551);
    vec3 c3 = vec3(0.741, 0.873, 0.150);
    vec3 c4 = vec3(0.993, 0.906, 0.144);

    if (t < 0.25) {
        return mix(c0, c1, t * 4.0);
    } else if (t < 0.5) {
        return mix(c1, c2, (t - 0.25) * 4.0);
    } else if (t < 0.75) {
        return mix(c2, c3, (t - 0.5) * 4.0);
    } else {
        return mix(c3, c4, (t - 0.75) * 4.0);
    }
}

vec3 colormap_thermal(float t) {
    vec3 c0 = vec3(0.0,   0.0,   0.0);
    vec3 c1 = vec3(0.55,  0.0,   0.0);
    vec3 c2 = vec3(1.0,   0.35,  0.0);
    vec3 c3 = vec3(1.0,   0.85,  0.0);
    vec3 c4 = vec3(1.0,   1.0,   1.0);

    if (t < 0.25) {
        return mix(c0, c1, t * 4.0);
    } else if (t < 0.5) {
        return mix(c1, c2, (t - 0.25) * 4.0);
    } else if (t < 0.75) {
        return mix(c2, c3, (t - 0.5) * 4.0);
    } else {
        return mix(c3, c4, (t - 0.75) * 4.0);
    }
}

void main() {
    vec3 color;

    if (colormap == 0) {
        color = colormap_diverging(v_scalar);
    } else if (colormap == 1) {
        color = colormap_sequential(v_scalar);
    } else {
        color = colormap_thermal(v_scalar);
    }

    FragColor = vec4(color, 1.0);
}
