#version 460 core

// ── Packed vertex layout ──────────────────────────────────────────────────────
// Each PackedVertex is 5 u32s (20 bytes):
//   [0] pos_xy  = packHalf2x16(pos.x, pos.y)
//   [1] pos_z   = packHalf2x16(pos.z, 0.0)
//   [2] norm_xy = packHalf2x16(norm.x, norm.y)
//   [3] norm_z  = packHalf2x16(norm.z, 0.0)
//   [4] uv_xy   = packHalf2x16(uv.s,  uv.t)

struct PackedVertex {
    uint pos_xy;
    uint pos_z;
    uint norm_xy;
    uint norm_z;
    uint uv_xy;
};

layout(std430, binding = 0) readonly buffer VertexBuffer {
    PackedVertex verts[];
};

// ── Per-object instance data ──────────────────────────────────────────────────
// std430 size: mat4 (64) + mat4 (64) + uint (4) + uint[3] pad = 144 bytes.
// Addressed as instances[gl_BaseInstance + gl_InstanceID]: gl_BaseInstance is
// the draw command's base_instance field (first slot for this mesh batch),
// gl_InstanceID walks through the instances within that batch.
struct InstanceData {
    mat4 model;
    mat4 normal; // inverse-transpose of model 3x3, widened to mat4
    uint material_index;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

layout(std430, binding = 1) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

// ── Uniforms ──────────────────────────────────────────────────────────────────
layout(location = 0) uniform mat4 u_view_proj;

// ── Outputs ───────────────────────────────────────────────────────────────────
out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
flat out uint v_material_index;

void main() {
    InstanceData inst = instances[gl_BaseInstance + gl_InstanceID];

    // gl_VertexID already accounts for baseVertex (set per draw command).
    PackedVertex pv = verts[gl_VertexID];

    vec2 pos_xy  = unpackHalf2x16(pv.pos_xy);
    vec2 pos_z0  = unpackHalf2x16(pv.pos_z);
    vec3 pos     = vec3(pos_xy, pos_z0.x);

    vec2 norm_xy = unpackHalf2x16(pv.norm_xy);
    vec2 norm_z0 = unpackHalf2x16(pv.norm_z);
    vec3 norm    = vec3(norm_xy, norm_z0.x);

    v_uv             = unpackHalf2x16(pv.uv_xy);
    v_world_pos      = (inst.model * vec4(pos, 1.0)).xyz;
    v_normal         = normalize(mat3(inst.normal) * norm);
    v_material_index = inst.material_index;

    gl_Position = u_view_proj * inst.model * vec4(pos, 1.0);
}
