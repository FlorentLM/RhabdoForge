// Shared LUT functions for scalar overlay visualisation

vec3 overlay_colormap_diverging(float t) {
    vec3 blue  = vec3(0.230, 0.299, 0.754);
    vec3 white = vec3(0.970, 0.970, 0.970);
    vec3 red   = vec3(0.706, 0.016, 0.150);

    if (t < 0.5) {
        return mix(blue, white, t * 2.0);
    } else {
        return mix(white, red, (t - 0.5) * 2.0);
    }
}

vec3 overlay_colormap_sequential(float t) {
    vec3 c0 = vec3(0.267, 0.004, 0.329);
    vec3 c1 = vec3(0.282, 0.141, 0.458);
    vec3 c2 = vec3(0.127, 0.566, 0.551);
    vec3 c3 = vec3(0.741, 0.873, 0.150);
    vec3 c4 = vec3(0.993, 0.906, 0.144);

    if (t < 0.25)      return mix(c0, c1, t * 4.0);
    else if (t < 0.5)  return mix(c1, c2, (t - 0.25) * 4.0);
    else if (t < 0.75) return mix(c2, c3, (t - 0.5) * 4.0);
    else                return mix(c3, c4, (t - 0.75) * 4.0);
}

vec3 overlay_colormap_thermal(float t) {
    vec3 c0 = vec3(0.0,  0.0,  0.0);
    vec3 c1 = vec3(0.55, 0.0,  0.0);
    vec3 c2 = vec3(1.0,  0.35, 0.0);
    vec3 c3 = vec3(1.0,  0.85, 0.0);
    vec3 c4 = vec3(1.0,  1.0,  1.0);

    if (t < 0.25)      return mix(c0, c1, t * 4.0);
    else if (t < 0.5)  return mix(c1, c2, (t - 0.25) * 4.0);
    else if (t < 0.75) return mix(c2, c3, (t - 0.5) * 4.0);
    else                return mix(c3, c4, (t - 0.75) * 4.0);
}

vec3 apply_overlay_colormap(float t, int overlay_colormap_id) {
    if (overlay_colormap_id == 0)      return overlay_colormap_diverging(t);
    else if (overlay_colormap_id == 1) return overlay_colormap_sequential(t);
    else                       return overlay_colormap_thermal(t);
}

/// Normalise raw scalar value to [0, 1] (with dynamic-range compression)
float normalize_scalar(float raw, float data_min, float data_max, float compression, int overlay_colormap_id) {
    float range = data_max - data_min;
    float t = (range > 1e-8) ? clamp((raw - data_min) / range, 0.0, 1.0) : 0.5;

    if (overlay_colormap_id == 0) {
        // Diverging: compress symmetrically around 0.5
        float centered = t * 2.0 - 1.0;
        float compressed = sign(centered) * pow(abs(centered), compression);
        t = compressed * 0.5 + 0.5;
    } else {
        t = pow(t, compression);
    }
    return t;
}
