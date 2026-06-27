#version 460 core

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
flat in uint v_material_index;

out vec4 frag_color;

// ── Material SSBO (binding 2) ─────────────────────────────────────────────────
// std430 size: vec4 (16) + float + float + float[2] pad = 32 bytes.
struct Material {
    vec4  base_color;
    float roughness;
    float metallic;
    float _pad0;
    float _pad1;
};

layout(std430, binding = 2) readonly buffer MaterialBuffer {
    Material materials[];
};

const vec3 LIGHT_DIR   = normalize(vec3(1.0, 2.0, 1.5));
const vec3 LIGHT_COLOR = vec3(1.0, 0.95, 0.88);
const vec3 AMBIENT     = vec3(0.08, 0.10, 0.14);

void main() {
    Material mat  = materials[v_material_index];
    vec3 albedo   = mat.base_color.rgb;

    vec3 n        = normalize(v_normal);
    float ndotl   = max(dot(n, LIGHT_DIR), 0.0);
    vec3 color    = albedo * (AMBIENT + LIGHT_COLOR * ndotl);

    frag_color = vec4(color, mat.base_color.a);
}
