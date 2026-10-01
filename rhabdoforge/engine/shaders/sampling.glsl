#ifndef SAMPLING_GLSL
#define SAMPLING_GLSL

// Ray direction sampling for the ommatidia dispatch

#include "commons.glsl"

#ifdef SAMPLER_LUT
// Layout: [gaussian (1 block)][active target per R type (LUT_MODE_SLOTS blocks)]
layout(std430, binding = BINDING_SENSITIVITY_ICDF_LUT) readonly buffer SensitivityIcdfLutBlock { float sensitivity_iCDF_LUT[]; };

// Per-mode profiles: one row per (saccade step, mode, rhab type), LUT_SIZE radii wide
layout(binding = 4) uniform sampler2D mode_profile_lut;
uniform int sampling_target;
const int SAMPLING_TARGET_GAUSSIAN  = 0;
const int SAMPLING_TARGET_WAVEGUIDE = 1;
#endif

// =====================================================================================================================

#ifdef SAMPLER_LUT

// Maps u1 to a radius (FWHM units) via the active target's inverse CDF, so ray weight can stay 1.0
float sample_lut_radius(uint rhab_type, float u1) {
    float float_idx = clamp(u1, 0.0, 1.0) * float(LUT_SIZE - 1);
    int   i0 = clamp(int(floor(float_idx)), 0, LUT_SIZE - 1);
    int   i1 = min(i0 + 1, LUT_SIZE - 1);
    float t  = fract(float_idx);

    int base = (sampling_target != 0) ? (LUT_SIZE + int(rhab_type) * LUT_SIZE) : 0;
    return mix(sensitivity_iCDF_LUT[base + i0], sensitivity_iCDF_LUT[base + i1], t);
}

// Third uniform to pick a mode, quality barely affects variance so hash-derived is bueno
float mode_rand(float u1, float u2) {
    uint h = pcg_hash(floatBitsToUint(u1) ^ (floatBitsToUint(u2) * 0x9E3779B9u) ^ 0x5BD1E995u);
    return float(h) * THE_TINIEST_FLOAT;
}

const float PROPOSAL_WIDEN = 1.7;   // proposal width mult (over curr_mode_angle)

// One mode's profile value at normalized radius x for the current saccade fraction
// (the pupil only reweights modes so lookup doesn't vary with it)
float mode_profile(uint mode, uint rhab_type, float frac, float x) {
    float zf = clamp(frac, 0.0, 1.0) * float(N_SACCADE_STEPS - 1);
    int   z0 = clamp(int(floor(zf)), 0, N_SACCADE_STEPS - 1);
    int   z1 = min(z0 + 1, N_SACCADE_STEPS - 1);
    float tz = fract(zf);

    float xi = clamp(x, 0.0, LUT_RANGE) * LUT_SCALE;
    int   i0 = clamp(int(floor(xi)), 0, LUT_SIZE - 1);
    int   i1 = min(i0 + 1, LUT_SIZE - 1);
    float t  = fract(xi);

    int r0 = (z0 * MAX_LP_MODES + int(mode)) * LUT_MODE_SLOTS + int(rhab_type);
    int r1 = (z1 * MAX_LP_MODES + int(mode)) * LUT_MODE_SLOTS + int(rhab_type);

    float p0 = mix(texelFetch(mode_profile_lut, ivec2(i0, r0), 0).r, texelFetch(mode_profile_lut, ivec2(i1, r0), 0).r, t);
    float p1 = mix(texelFetch(mode_profile_lut, ivec2(i0, r1), 0).r, texelFetch(mode_profile_lut, ivec2(i1, r1), 0).r, t);
    return mix(p0, p1, tz);
}

// Proposal = wide-Gaussian mixture, reweighted against mode profiles
vec3 sampledir_mixture(RhabdomereStatic rs, RhabdomereDynamic rd, OmmatidiumStatic os, vec3 T, vec3 B, vec3 F, float u1, float u2, out float weight) {

    float phi = TWOPI * u2;
    uint rhab_type = unpack_rhab_type(rs.metadata);

    // Pick a mode (by walking cumulative weight)
    float u_pick = mode_rand(u1, u2);
    float cum_w = 0.0;
    float a_pick = rd.curr_mode_angle[0] * PROPOSAL_WIDEN;
    bool picked = false;
    for (int m = 0; m < MAX_LP_MODES; ++m) {
        cum_w += rd.curr_mode_weight[m];
        if (!picked && u_pick < cum_w) {
            a_pick = rd.curr_mode_angle[m] * PROPOSAL_WIDEN;
            picked = true;
        }
    }
    float angle = a_pick * sqrt(-log(u1) / GAUSS_CONSTANT_K);

    // Proposal and target densities over all modes, target normalised by each mode's angle
    float proposal = 0.0;
    float target = 0.0;
    for (int m = 0; m < MAX_LP_MODES; ++m) {
        float w = rd.curr_mode_weight[m];
        float a = rd.curr_mode_angle[m] * PROPOSAL_WIDEN;
        float e = exp(-GAUSS_CONSTANT_K * angle * angle / (a * a)) / (a * a);
        proposal += w * e;
        target += w * mode_profile(uint(m), rhab_type, rd.curr_saccade_frac, angle / max(rd.curr_mode_angle[m], 1e-6));
    }

    weight = (proposal > 1e-20) ? (target / proposal) : 0.0;

    vec2 p = vec2(tan(angle) * cos(phi), tan(angle) * sin(phi));

    float s = sin(os.ioa_tilt), c = cos(os.ioa_tilt);
    vec2 tp = mat2(c, -s, s, c) * p;

    return normalize(mat3(T, B, F) * normalize(vec3(tp, 1.0)));
}

vec3 sampledir_lut_importance(RhabdomereStatic rs, RhabdomereDynamic rd, OmmatidiumStatic os, vec3 T, vec3 B, vec3 F, float u1, float u2, out float weight) {

    if (sampling_target == SAMPLING_TARGET_WAVEGUIDE) {
        return sampledir_mixture(rs, rd, os, T, B, F, u1, u2, weight);
    }

    float phi = TWOPI * u2;
    float r = sample_lut_radius(unpack_rhab_type(rs.metadata), u1);

    float angle_min = rd.curr_acc_angles.x * r;
    float angle_maj = rd.curr_acc_angles.y * r;

    vec2 p = vec2(tan(angle_min) * cos(phi), tan(angle_maj) * sin(phi));

    weight = 1.0;  // sampled from the target's own distribution: no need to reweight

    float s = sin(os.ioa_tilt), c = cos(os.ioa_tilt);
    vec2 tp = mat2(c, -s, s, c) * p;

    return normalize(mat3(T, B, F) * normalize(vec3(tp, 1.0)));
}

#endif // SAMPLER_LUT

#ifdef SAMPLER_PURE_IMPORTANCE

vec3 sampledir_importance(RhabdomereStatic rs, RhabdomereDynamic rd, OmmatidiumStatic os, vec3 T, vec3 B, vec3 F, float u1, float u2, out float weight) {

    float phi = TWOPI * u2;
    float angle_min = rd.curr_acc_angles.x * sqrt(-log(u1) / GAUSS_CONSTANT_K);
    float angle_maj = rd.curr_acc_angles.y * sqrt(-log(u1) / GAUSS_CONSTANT_K);

    vec2 p = vec2(tan(angle_min) * cos(phi), tan(angle_maj) * sin(phi));

    float s = sin(os.ioa_tilt), c = cos(os.ioa_tilt);
    vec2 tp = mat2(c, -s, s, c) * p;

    weight = 1.0; // pure importance sampling: weight is uniform
    return normalize(mat3(T, B, F) * normalize(vec3(tp, 1.0)));
}

#endif // SAMPLER_PURE_IMPORTANCE

#endif // SAMPLING_GLSL
