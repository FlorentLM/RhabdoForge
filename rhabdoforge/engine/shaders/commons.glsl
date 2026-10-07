#ifndef COMMONS_GLSL
#define COMMONS_GLSL

const int RNG_PSEUDO     = 0;
const int RNG_HALTON     = 1;
const int RNG_STRATIFIED = 2;
const int RNG_FIBONACCI  = 3;
const int RNG_HAMMERSLEY = 4;
const int RNG_SOBOL      = 5;

// Sensitivity LUT, should match LUT_RANGE / LUT_SIZE / LUT_MODE_SLOTS in LUTs.py
const int   LUT_SIZE  = 256;
const float LUT_RANGE = 4.0;
const float LUT_SCALE = float(LUT_SIZE - 1) / LUT_RANGE;
const int   LUT_MODE_SLOTS = 16;   // rhabdomere types a per-type LUT can index

#ifndef MAX_LP_MODES    //  injected by the renderer
#define MAX_LP_MODES 2
#endif
const int N_PUPIL_STEPS   = 8;   // grid steps: dark to lit
const int N_SACCADE_STEPS = 8;   // grid steps: rest to full saccade


const float PI = 3.141592653589793;
const float HPI = 1.5707963267948966;
const float TWOPI = 6.283185307179586;
const float GOLDEN_RATIO_ANGL = 2.399963229728653;  // pi * (3.0 - sqrt(5.0))
const float GAUSS_CONSTANT_K = 2.772588722239781;   // 4 * log(2), for a Gaussian with FWHM = acceptance_angle
const float THE_TINIEST_FLOAT = 2.3283064365386963e-10;  // 2^-32


struct Material {
    uint texture_idx;       // 0xFFFFFFFF means no texture (use base_color)
    uint base_color;        // RGBA8 packed into uint32
    float alpha_cutoff;     // <=0 means opaque (no alpha test)
    uint texture_tier;      // which size bucket scene_textures_N arrays texture_idx indexes into
    vec2 uv_scale;          // non-square textures are letterboxed into a (square) tier canvas, this maps mesh UV [0,1] to the used rectangle
};

// Ommatidium static  (read only)
struct OmmatidiumStatic {
    vec3  position;
    float chi;
    vec3  forward;
    float focal_um;
    vec3  right;
    float aperture_um;
    vec3  up;
    float ioa_tilt;
    vec2 saccade_dxdy;
    float ampl_lateral;
    float ampl_axial;
    float move_duration;
    float return_duration;
    float tau_fast;
    float tau_adapt;
    vec2  ioa_angles;
    vec2  retina_dxdy;
    float tau_pupil;
    float tau_return;
    float _pad0;
    float _pad1;
}; // 128 bytes

// Ommatidium dynamic
struct OmmatidiumDynamic {
    float curr_lum_fast;
    float curr_lum_slow;
    float curr_lateral_disp;
    float curr_axial_disp;
    float mech_phase;      // 0=idle, 1=latency, 2=moving, 3=returning
    float mech_t;          // elapsed time in current phase (s)
    float mech_frac0;      // ballistic fraction at the start of the current phase
    float curr_pupil_drive; // pupil engagement, 0 = dark, 1 = closed
}; // 32 bytes

// Rhabdomere static (read only)
struct RhabdomereStatic {
    vec3  sensitivity;
    float wavelength_um;
    vec2  rest_acc_angles;
    vec2  rest_offset;
    float tau_membrane;
    uint  cartridge_src;
    float diameter_um;
    uint  metadata;
    float closed_pupil_ratio;      // D rho (light-adapted) / D rho (dark), <= 1 (narrower)
    float closed_pupil_transmit[3];   // Peak sensitivity ratio per (R, G, B) channel, <= 1 (dimmer)
    float saccade_ratio_dark;   // D rho (full saccade) / D rho (rest), dark-adapted
    float saccade_ratio_lit;    // D rho (full saccade) / D rho (rest), light-adapted
    float lateral_clipping;     // Clipping weight against the lens/aperture, scales with the offset from the bundle centre (0 on axis, 1 farthest)

    // LP modes mixture importance sampling
    float mode_weight[MAX_LP_MODES];        // Each mode's power fraction at rest/dark (sums to 1)
    float mode_pupil_ratio[MAX_LP_MODES];   // Each mode's power ratio: light-adapted / dark-adapted
    float rad_per_hwhm;                     // (unit is dimensionless mode half-width)
#if (MAX_LP_MODES % 2) == 1
    float _pad[2];
#endif
}; // size: depends on MAX_LP_MODES

// Rhabdomere dynamic
struct RhabdomereDynamic {
    vec3  curr_direction;
    float curr_adaptation;
    vec2  curr_acc_angles;
    float saccade_scale;    // RF narrowing from the saccade only (no pupil/clip) for photon concentration
    float pupil_transmit[3];    // per (R, G, B) channel

    float curr_mode_weight[MAX_LP_MODES]; // Current per-mode mixture proba (sum to 1)
    float curr_mode_angle[MAX_LP_MODES];  // Current (actuated) per-mode acceptance angle (rad)
    float curr_saccade_frac;           // Saccade fraction (0 -> 1)
#if (MAX_LP_MODES % 2) == 0
    float _pad;
#else
    float _pad[3];
#endif
}; // size: depends on MAX_LP_MODES


// Metadata Unpacking
uint  unpack_eye_id(uint m)          { return m & 15u; }                          // bits 0-3
uint  unpack_rhab_type(uint m)       { return (m >> 4u)  & 15u; }                 // bits 4-7
uint  unpack_neighbour_count(uint m) { return (m >> 8u)  & 15u; }                 // bits 8-11
uint  unpack_omm_id(uint m)          { return (m >> 12u) & 65535u; }              // bits 12-27
float unpack_chirality(uint m)       { return ((m >> 28u) & 1u) == 1u ? -1.0 : 1.0; }  // bit 28
bool  unpack_binocularity(uint m)    { return ((m >> 29u) & 1u) == 1u; }          // bit 29
bool  unpack_wiring_valid(uint m)    { return ((m >> 30u) & 1u) == 1u; }          // bit 30
bool  unpack_is_edge(uint m)         { return ((m >> 31u) & 1u) == 1u; }          // bit 31


vec4 unpack_color(uint packed_color) {
    return vec4(float(packed_color & 255u) / 255.0,
                float((packed_color >> 8u) & 255u) / 255.0,
                float((packed_color >> 16u) & 255u) / 255.0,
                float((packed_color >> 24u) & 255u) / 255.0);
}

struct Triangle {
    vec4 v0, v1, v2;       // offsets 0, 16 and 32, size 16. w is unused
    vec2 uv0, uv1, uv2;    // offsets 48, 56 and 64, size 8
    uint material_idx;     // offset 72, size 4
}; // 80 bytes

struct Point {
    vec3 position;
    float radius;
    vec3 normal;
    vec3 color;
    float pad0, pad1;
};


// =====================================================================================================================

// Simple RNG with temporal dithering
float rand(vec2 co, float dither){
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233)) + dither) * 43758.5453);
}

// Radical-inverse Halton sequence (low-discrepancy quasi-random)
// (as in https://en.wikipedia.org/wiki/Halton_sequence#Implementation)
float halton_sequence(uint index, uint base) {
    float f = 1.0;
    float r = 0.0;
    uint i = index;
    while (i > 0u) {
        f /= float(base);
        r += f * float(i % base);
        i /= base;
    }
    return r;
}

// Standard 32-bit radical-inverse for base 2 (for Hammersley)
float radical_inverse_v2(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * THE_TINIEST_FLOAT;
}

uint reverse_bits(uint x) {
    x = (x << 16u) | (x >> 16u);
    x = ((x & 0x00ff00ffu) << 8u) | ((x & 0xff00ff00u) >> 8u);
    x = ((x & 0x0f0f0f0fu) << 4u) | ((x & 0xf0f0f0f0u) >> 4u);
    x = ((x & 0x33333333u) << 2u) | ((x & 0xccccccccu) >> 2u);
    x = ((x & 0x55555555u) << 1u) | ((x & 0xaaaaaaaau) >> 1u);
    return x;
}

// Second dimension of the Sobol (0,2)-sequence
uint sobol_dim1(uint i) {
    uint r = 0u;
    for (uint v = 1u << 31u; i != 0u; i >>= 1u, v ^= v >> 1u)
        if ((i & 1u) != 0u) r ^= v;
    return r;
}

// Hash-based nested-uniform (Owen) scramble, Burley 2020 / pbrt-v4
uint owen_scramble(uint v, uint seed) {
    v  = reverse_bits(v);
    v ^= v * 0x3d20adeau;
    v += seed;
    v *= (seed >> 16u) | 1u;
    v ^= v * 0x05526c56u;
    v ^= v * 0x53a22864u;
    return reverse_bits(v);
}

uint pcg_hash(uint seed) {
    uint state = seed * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float random_float(inout uint rng_state) {
    rng_state = pcg_hash(rng_state);
    return float(rng_state) / 4294967295.0;
}

struct Sampler {
    float u1;
    float u2;
};

Sampler get_samples(int mode, uint sample_idx, uint nb_samples, uint rhab_idx, uint dither_counter) {
    Sampler s;
    uint seed = pcg_hash(rhab_idx * 1973u + dither_counter * 26699u + sample_idx * 749u);

    if (mode == RNG_HALTON) {
        uint halton_idx = dither_counter * nb_samples + sample_idx + 1u;
        uint hash = pcg_hash(rhab_idx * 1973u);
        s.u1 = clamp(fract(halton_sequence(halton_idx, 2u) + float(hash & 0xFFFFu)/65535.0), 1e-6, 1.0);
        s.u2 = fract(halton_sequence(halton_idx, 3u) + float((hash >> 16u) & 0xFFFFu)/65535.0);
    }
    else if (mode == RNG_STRATIFIED) {
        uint G = uint(ceil(sqrt(float(nb_samples))));
        uint cell = (sample_idx + pcg_hash(rhab_idx*1973u + dither_counter*26699u) % (G*G)) % (G*G);
        float cell_x = float(cell % G);
        float cell_y = float(cell / G);

        uint rng_state = seed;
        s.u1 = (cell_x + random_float(rng_state)) / G;
        s.u2 = (cell_y + random_float(rng_state)) / G;
    }
    else if (mode == RNG_FIBONACCI) {
        uint scr = pcg_hash(rhab_idx * 1973u + dither_counter * 26699u);
        float rot_off = random_float(scr);
        float rad_off = random_float(scr);
        float u1 = fract((float(sample_idx) + 0.5) / float(nb_samples) + rad_off);
        s.u2 = fract(float(sample_idx) * GOLDEN_RATIO_ANGL / TWOPI + rot_off);
        s.u1 = clamp(1.0 - u1, 1e-6, 1.0);
    }
    else if (mode == RNG_HAMMERSLEY) {
        // Cranley-Patterson scrambled Hammersley
        // Scramble seed changes per frame (dither) and per pixel (rhab_idx), but not per sample
        uint scramble_seed = pcg_hash(rhab_idx * 1973u + dither_counter * 26699u);
        float offset1 = random_float(scramble_seed);
        float offset2 = random_float(scramble_seed);

        float u1_base = (float(sample_idx) + 0.5) / float(nb_samples);
        float u2_base = radical_inverse_v2(sample_idx);

        s.u1 = fract(u1_base + offset1);
        s.u2 = fract(u2_base + offset2);
        s.u1 = clamp(s.u1, 1e-6, 1.0);
    }
    else if (mode == RNG_SOBOL) {
        uint idx = dither_counter * nb_samples + sample_idx;
        uint seed_x = pcg_hash(rhab_idx * 1973u + 0x9e3779b9u);
        uint seed_y = pcg_hash(rhab_idx * 1973u + 0x85ebca6bu);
        uint vx = owen_scramble(reverse_bits(idx), seed_x);
        uint vy = owen_scramble(sobol_dim1(idx), seed_y);
        s.u1 = clamp(float(vx) * THE_TINIEST_FLOAT, 1e-6, 1.0);
        s.u2 = float(vy) * THE_TINIEST_FLOAT;
    }
    else { // RNG_PSEUDO
        uint rng_state = seed;
        s.u1 = random_float(rng_state);
        s.u2 = random_float(rng_state);
    }
    return s;
}


#endif // COMMONS_GLSL