#version 460 core

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
flat in uint v_material_index;

out vec4 frag_color;

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

// location 0-3 are taken by the mat4 u_view_proj in the vertex stage.
layout(location = 4) uniform vec3 u_eye_pos;

const vec3  LIGHT_DIR   = normalize(vec3(1.0, 2.0, 1.5));
const vec3  LIGHT_COLOR = vec3(1.0, 0.95, 0.88) * 3.0;
const vec3  AMBIENT     = vec3(0.03, 0.035, 0.04);
const float PI          = 3.14159265358979;

float D_GGX(float ndoth, float alpha2) {
    float d = ndoth * ndoth * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * d * d);
}

float G_SchlickGGX(float ndotx, float k) {
    return ndotx / (ndotx * (1.0 - k) + k);
}

float G_Smith(float ndotv, float ndotl, float roughness) {
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return G_SchlickGGX(ndotv, k) * G_SchlickGGX(ndotl, k);
}

vec3 F_Schlick(float vdoth, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - vdoth, 0.0, 1.0), 5.0);
}

void main() {
    Material mat    = materials[v_material_index];
    vec3  albedo    = mat.base_color.rgb;
    float roughness = mat.roughness;
    float metallic  = mat.metallic;

    float alpha  = roughness * roughness;
    float alpha2 = alpha * alpha;
    vec3  f0     = mix(vec3(0.04), albedo, metallic);

    vec3  n     = normalize(v_normal);
    vec3  v     = normalize(u_eye_pos - v_world_pos);
    vec3  l     = LIGHT_DIR;
    vec3  h     = normalize(v + l);
    float ndotl = max(dot(n, l), 0.0);
    float ndotv = max(dot(n, v), 0.0);
    float ndoth = max(dot(n, h), 0.0);
    float vdoth = max(dot(v, h), 0.0);

    vec3  F        = F_Schlick(vdoth, f0);
    float D        = D_GGX(ndoth, alpha2);
    float G        = G_Smith(ndotv, ndotl, roughness);
    vec3  specular = (D * G * F) / max(4.0 * ndotv * ndotl, 0.001);

    vec3 kd      = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diffuse = kd * albedo / PI;

    vec3 color = (diffuse + specular) * LIGHT_COLOR * ndotl
               + AMBIENT * albedo;

    frag_color = vec4(color, mat.base_color.a);
}
