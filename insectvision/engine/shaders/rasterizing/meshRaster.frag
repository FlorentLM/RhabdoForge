#version 430 core

// Input: Varyings from vertex shader
layout (location = 0) in vec2 fragTexCoord;
layout (location = 1) in vec3 fragWorldPos;
layout (location = 2) in vec3 fragWorldNormal;
layout (location = 3) in vec4 fragLightSpacePos;

// Output: Final color to framebuffer
layout (location = 0) out vec4 FragColor;

// Matrial
layout (binding = 0) uniform sampler2D texture1;
uniform bool has_texture;
uniform vec4 base_color;

uniform bool false_colors;
uniform bool uv_encoding;

// Shadow map
layout (binding = 1) uniform sampler2DShadow shadow_map;

// Lighting
uniform bool enable_direct;
uniform bool enable_shadows;

// Primary directional light
uniform vec3  light_direction;        // normalised direction *to* the light
uniform vec3  light_color;
uniform float light_intensity;

// Ambient
uniform vec3  ambient_color;
uniform float ambient_intensity;

// Shadow bias
uniform float shadow_bias;            // depth bias to reduce acne


float calculate_shadow(vec4 light_space_pos, vec3 normal, vec3 light_dir)
{
    // Perspective divide (no-op for ortho, but correct for both)
    vec3 proj = light_space_pos.xyz / light_space_pos.w;

    // Map from [-1, 1] to [0, 1] for texture lookup
    proj = proj * 0.5 + 0.5;

    // Fragments outside the shadow map frustum are lit
    if (proj.x < 0.0 || proj.x > 1.0 ||
        proj.y < 0.0 || proj.y > 1.0 ||
        proj.z > 1.0) {
        return 0.0;
    }

    // Slope-scaled bias
    float bias = max(shadow_bias * (1.0 - dot(normal, light_dir)), shadow_bias * 0.1);

    // PCF 3x3 for slightly softer shadow edges
    float shadow = 0.0;
    vec2 texel_size = 1.0 / textureSize(shadow_map, 0);

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            vec3 sample_coord = vec3(proj.xy + vec2(x, y) * texel_size, proj.z - bias);
            shadow += texture(shadow_map, sample_coord);  // hardware PCF comparison
        }
    }

    return 1.0 - (shadow / 9.0);
}


void main()
{
    // Base surface color
    vec4 surface_color;
    if (has_texture) {
        surface_color = texture(texture1, fragTexCoord);
    } else {
        surface_color = base_color;
    }

    if (!enable_direct) {
        FragColor = surface_color;
        return;
    }

    vec3 N = normalize(fragWorldNormal);
    vec3 L = normalize(light_direction);

    // Lambertian diffuse
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = light_color * light_intensity * NdotL;

    // Shadow
    float shadow = 0.0;
    if (enable_shadows) {
        shadow = calculate_shadow(fragLightSpacePos, N, L);
    }

    // Ambient
    vec3 ambient = ambient_color * ambient_intensity;

    // Final lit color
    vec3 lit = surface_color.rgb * (ambient + diffuse * (1.0 - shadow));

    vec4 color = vec4(lit, surface_color.a);

    if (uv_encoding || false_colors) {
        // Drop the red channel (UV) for the human screen
        color.r = 0.0;
    }

    FragColor = color;
}
