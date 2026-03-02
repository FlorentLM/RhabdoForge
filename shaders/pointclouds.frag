#version 430 core

in vec3 vertColor;
in vec4 fragLightSpacePos;

out vec4 FragColor;

// Shadow map
layout (binding = 1) uniform sampler2DShadow shadow_map;

// Lighting toggles
uniform bool enable_shadows;
uniform float shadow_darkness;     // how dark shadows are (0.0 = black, 1.0 = no shadow)

// Shadow bias
uniform float shadow_bias;


float calculate_shadow(vec4 light_space_pos)
{
    vec3 proj = light_space_pos.xyz / light_space_pos.w;
    proj = proj * 0.5 + 0.5;

    // Outside frustum = lit
    if (proj.x < 0.0 || proj.x > 1.0 ||
        proj.y < 0.0 || proj.y > 1.0 ||
        proj.z > 1.0) {
        return 0.0;
    }

    // PCF 3x3
    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(shadow_map, 0);

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec3 sample_coord = vec3(proj.xy + vec2(x, y) * texel_size, proj.z - shadow_bias);
            shadow += texture(shadow_map, sample_coord);
        }
    }

    return 1.0 - (shadow / 9.0);
}


void main()
{
    // Convert sRGB to linear space
    vec3 linearColor = pow(vertColor, vec3(2.2));

    if (enable_shadows) {
        float shadow = calculate_shadow(fragLightSpacePos);
        float light_factor = mix(shadow_darkness, 1.0, 1.0 - shadow); // Darken (shadow_darkness controls the floor)
        linearColor *= light_factor;
    }

    // GL_FRAMEBUFFER_SRGB is enabled so OpenGL expects linear color
    FragColor = vec4(linearColor, 1.0);
}
