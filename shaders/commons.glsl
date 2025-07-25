
// Ommatidium struct for clarity
struct Ommatidium {
    vec3 origin;            // 12 bytes, offset 0
    // std430 alignment for vec3 is 16 bytes, so the compiler adds 4 bytes of padding here
    vec3 direction;         // 12 bytes, offset 16
    float acceptance_angle; // 4 bytes, offset 28
}; // 32 bytes total per ommatidium