#version 430 core

out vec4 FragColor;

layout(std430, binding = 0) readonly buffer OmmatidiaBlock {
   vec4 u_ommatidia_directions[]; // std430 alignment needs vec4
};

uniform samplerCube u_scene_cubemap;
uniform float u_acceptance_angle; // This is a radius in radians
uniform int u_num_ommatidia;

const int SAMPLES_PER_OMMATIDIUM = 64;
const float PI = 3.14159265359;

// Simple pseudo-random number generator
float rand(vec2 co){
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// Gaussian falloff function
float gaussian(float angle_zeta, float acceptance_angle_rho) {
    // Snyder 1979
    return exp(-4.0 * log(2.0) * pow(angle_zeta / acceptance_angle_rho, 2.0));
}

void main()
{
    // Determine which ommatidium this fragment corresponds to
    int ommatidium_idx = int(gl_FragCoord.x);

    // Stop if past the last ommatidium
    if (ommatidium_idx >= u_num_ommatidia) {
        discard;
    }

    vec3 om_dir = normalize(u_ommatidia_directions[ommatidium_idx].xyz);
    vec3 total_color = vec3(0.0);
    float total_weight = 0.0;

    // Create a local coordinate system (TBN matrix) for the ommatidium
    vec3 up = abs(om_dir.y) > 0.99 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
    vec3 tangent = normalize(cross(up, om_dir)); // Note the order for a right-handed system
    vec3 bitangent = cross(om_dir, tangent);
    mat3 tbn = mat3(tangent, bitangent, om_dir);

    // Monte Carlo sampling within the cone
    float cos_acceptance_angle = cos(u_acceptance_angle);

    for (int i = 0; i < SAMPLES_PER_OMMATIDIUM; i++) {
        // Generate two random numbers for sampling
        float r1 = rand(gl_FragCoord.xy + vec2(i, i * 2.0));
        float r2 = rand(gl_FragCoord.xy - vec2(i * 3.0, i));

        // Uniformly sample the solid angle of the cone
        // cos_theta is the z-axis in local space
        float cos_theta = mix(cos_acceptance_angle, 1.0, r1);
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        float phi = 2.0 * PI * r2; //

        // Construct sample direction in local space
        vec3 sample_local = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);

        // Transform sample to world space
        vec3 sample_dir = normalize(tbn * sample_local);

        // The angle between the sample and the ommatidium's center is acos(cos_theta)
        float angle_zeta = acos(cos_theta);

        // Get weight from Gaussian function
        float weight = gaussian(angle_zeta, u_acceptance_angle);

        // Sample the cubemap
        vec3 sampled_color = texture(u_scene_cubemap, sample_dir).rgb;

        // Accumulate weighted color
        total_color += sampled_color * weight;
        total_weight += weight;
    }

    if (total_weight > 0.0) {
        total_color /= total_weight;
    }

    FragColor = vec4(total_color, 1.0);
}