#ifndef COMMONS_GLSL
#define COMMONS_GLSL

const int RNG_PSEUDO     = 0;
const int RNG_HALTON     = 1;
const int RNG_STRATIFIED = 2;

const int MODE_GAUSSIAN  = 0;
const int MODE_AIRY      = 1;

const float PI = 3.14159265359;
const float HPI = PI * 0.5;
const float TWOPI = 2.0 * PI;

// Constant k = 4 * log(2) for a Gaussian with FWHM = acceptance_angle
const float GAUSS_CONSTANT_K = 2.77258872224;

struct Material {
    uint texture_idx;       // 0xFFFFFFFF means no texture (use base_color)
    uint base_color;        // RGBA8 packed into uint32
    uint pad0, pad1;
};

// Lens static (read only)
struct LensStatic {
    vec3  right;
    float sacc_local_x;
    vec3  up;
    float sacc_local_y;
    vec3  forward;
    float ioa_tilt;
    vec2  ioa_axes;
    float focal_um;
    float aperture_um;
    float tau_rise;
    float tau_relax;
    float tau_fast;
    float tau_adapt;
    float ampl_lat_um;
    float ampl_ax_um;
    float retina_x;
    float retina_y;
}; // 96 bytes

// Lens dynamic
struct LensDynamic {
    float adapted_lum;
    float fast_lum;
    float lateral_um;
    float axial_um;
}; // 16 bytes

// Receptor static (read only)
struct ReceptorStatic {
    vec3  position;
    uint  metadata;
    vec2  rest_acc;
    vec2  rot_offset;
    vec3  sensitivity;
    float acc_tilt;
    float tau_membrane;
    uint  cartridge_src;
    float rhab_diameter_um;
    float wavelength_um;
}; // 64 bytes

// Receptor dynamic
struct ReceptorDynamic {
    vec3  direction;
    float adaptation_state;
    vec2  acc_axes;
    float optical_scale;
    float pad;
}; // 32 bytes

const uint RECEPTOR_UNWIRED = 0xFFFFFFFFu;

// Metadata Unpacking
uint  unpack_eye_id(uint m)          { return m & 7u; }
uint  unpack_receptor_type(uint m)   { return (m >> 3u) & 15u; }
uint  unpack_lens_id(uint m)         { return (m >> 11u) & 65535u; }
float unpack_chirality(uint m)       { return ((m >> 27u) & 1u) == 1u ? -1.0 : 1.0; }
uint  unpack_neighbour_count(uint m) { return (m >> 7u) & 15u; }
bool  unpack_binocularity(uint m)    { return ((m >> 28u) & 1u) == 1u; }
bool  unpack_wiring_valid(uint m)    { return ((m >> 29u) & 1u) == 1u; }


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

Sampler get_samples(int mode, uint sample_idx, uint nb_samples, uint rcpt_idx, uint dither_counter) {
    Sampler s;
    uint seed = pcg_hash(rcpt_idx * 1973u + dither_counter * 26699u + sample_idx * 749u);

    if (mode == RNG_HALTON) {
        uint halton_idx = dither_counter * nb_samples + sample_idx + 1u;
        uint hash = pcg_hash(rcpt_idx * 1973u);
        s.u1 = clamp(fract(halton_sequence(halton_idx, 2u) + float(hash & 0xFFFFu)/65535.0), 1e-6, 1.0);
        s.u2 = fract(halton_sequence(halton_idx, 3u) + float((hash >> 16u) & 0xFFFFu)/65535.0);
    }
    else if (mode == RNG_STRATIFIED) {
        float grid_size = ceil(sqrt(float(nb_samples)));
        float cell_x = float(sample_idx % uint(grid_size));
        float cell_y = float(sample_idx / uint(grid_size));

        uint rng_state = seed;
        s.u1 = (cell_x + random_float(rng_state)) / grid_size;
        s.u2 = (cell_y + random_float(rng_state)) / grid_size;
    }
    else { // RNG_PSEUDO
        uint rng_state = seed;
        s.u1 = random_float(rng_state);
        s.u2 = random_float(rng_state);
    }
    return s;
}

// =====================================================================================================================

float airy_psf(float a_min, float a_maj, float D, float lambda) {
    // pure lens diffraction at angular offset (a_min, a_maj)

    float x_min = (PI * D * a_min) / max(lambda, 0.01);
    float x_maj = (PI * D * a_maj) / max(lambda, 0.01);

    float x = sqrt(x_min*x_min + x_maj*x_maj);

    if (x < 0.001) return 1.0;

    float j1 = (x < 3.75)
        ? (x*0.5) - (pow(x,3.0)/16.0) + (pow(x,5.0)/384.0) - (pow(x,7.0)/18432.0)
        : sqrt(0.636619/x) * cos(x - 0.785398);

    return clamp(pow(2.0*j1/x, 2.0), 0.0, 1.0);
}

float get_sensitivity(int mode, float angle_min, float angle_maj,
                      ReceptorStatic rs, ReceptorDynamic rd, LensStatic ls) {
    float D = ls.aperture_um, lambda = rs.wavelength_um;

    if (mode == MODE_AIRY) {  // Airy (x) rhabdomere acceptance
        float rho_diff = lambda / D;
        float rg_min = sqrt(max(rd.acc_axes.x*rd.acc_axes.x - rho_diff*rho_diff, 0.0));
        float rg_maj = sqrt(max(rd.acc_axes.y*rd.acc_axes.y - rho_diff*rho_diff, 0.0));

        if (rg_min < 1e-5 && rg_maj < 1e-5)  // diffraction-limited: nothing to convolve
            return airy_psf(angle_min, angle_maj, D, lambda);

        // numerical convolution with a Gaussian rhabdomere bundle (FWHM = rg)
        const int   N    = 6;     // (2N+1)^2 taps (N ~3*rg/rho_diff for faithful rings)
        const float SPAN = 2.2;   // bundle extent (in units of rg)

        float acc = 0.0, wsum = 0.0;
        for (int i = -N; i <= N; i++)
        for (int j = -N; j <= N; j++) {
            float ui = SPAN * float(i) / float(N);
            float uj = SPAN * float(j) / float(N);
            float kw = exp(-GAUSS_CONSTANT_K * (ui*ui + uj*uj));
            acc  += kw * airy_psf(angle_min - ui*rg_min, angle_maj - uj*rg_maj, D, lambda);
            wsum += kw;
        }
        return clamp(acc / max(wsum, 1e-6), 0.0, 1.0);
    }
    else { // MODE_GAUSSIAN
        // Snyder quadrature approximation of that same convolution
        float g_min = angle_min / max(rd.acc_axes.x, 1e-15);
        float g_maj = angle_maj / max(rd.acc_axes.y, 1e-15);
        return exp(-GAUSS_CONSTANT_K * (g_min*g_min + g_maj*g_maj));
    }
}

vec3 sampledir_importance(ReceptorStatic rs, ReceptorDynamic rd, vec3 T, vec3 B, vec3 F, float u1, float u2, out float weight) {
    float phi = TWOPI * u2;
    float angle_min = rd.acc_axes.x * sqrt(-log(u1) / GAUSS_CONSTANT_K);
    float angle_maj = rd.acc_axes.y * sqrt(-log(u1) / GAUSS_CONSTANT_K);

    vec2 p = vec2(tan(angle_min) * cos(phi), tan(angle_maj) * sin(phi));
    float s = sin(rs.acc_tilt), c = cos(rs.acc_tilt);
    vec2 tp = mat2(c, -s, s, c) * p;

    weight = 1.0; // pure importance sampling: weight is uniform
    return normalize(mat3(T, B, F) * normalize(vec3(tp, 1.0)));
}

vec3 sampledir_hybrid(int mode, ReceptorStatic rs, ReceptorDynamic rd, LensStatic ls, vec3 T, vec3 B, vec3 F, float u1, float u2, out float weight) {
    float phi = TWOPI * u2;

    // Sample a 'proposal' distribution that is wider than the actual acceptance
    // -> ensures it samples the tails / Airy rings, wide-angle lights, etc
    float spread_mult = 2.0;
    float sample_sigma_min = rd.acc_axes.x * spread_mult;
    float sample_sigma_maj = rd.acc_axes.y * spread_mult;

    // These are the displacement angles to test (raw elliptical radii)
    float r_min = sample_sigma_min * sqrt(-log(u1) / GAUSS_CONSTANT_K);
    float r_maj = sample_sigma_maj * sqrt(-log(u1) / GAUSS_CONSTANT_K);

    // Cartesian angles (dx, dy) for this specific ray
    float dx = r_min * cos(phi);
    float dy = r_maj * sin(phi);

    // Receptor sensitivity (at this specific sampled angle) -> 'Physical truth'
    float rcpt_sensitivity = get_sensitivity(mode, dx, dy, rs, rd, ls);

    // Sampling probability Density (PDF) of the proposal distribution -> likelihood that this ray was picked
    float p_min = dx / sample_sigma_min;
    float p_maj = dy / sample_sigma_maj;
    float pdf = exp(-GAUSS_CONSTANT_K * (p_min*p_min + p_maj*p_maj));

    // weight is truth / sampling
    weight = rcpt_sensitivity / max(pdf, 1e-6);  // avoid /0 in the extreme tails

    // and convert angles to direction vector
    vec2 p = vec2(tan(dx), tan(dy));
    float s = sin(rs.acc_tilt), c = cos(rs.acc_tilt);
    vec2 tp = mat2(c, -s, s, c) * p;

    return normalize(mat3(T, B, F) * normalize(vec3(tp, 1.0)));
}

#endif // COMMONS_GLSL