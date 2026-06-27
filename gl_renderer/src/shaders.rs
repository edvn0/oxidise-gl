/// Vertex shader: vertex pulling from SSBO, base-vertex indexed drawing.
/// The vertex shader reads PackedVertex from the SSBO using gl_VertexID.
pub const VERT_SRC: &str = r#"
#version 430 core

// gl_DrawID is core only in GLSL 4.60; on a 4.3 context we read it via the
// ARB extension as gl_DrawIDARB. Required by the multi-draw-indirect path.
#extension GL_ARB_shader_draw_parameters : require

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

// ── Per-object instance data (one entry per draw command) ─────────────────────
// Indexed by gl_DrawIDARB: command N reads instances[N].
struct InstanceData {
    mat4 model;
    mat4 normal; // inverse-transpose of model 3x3, widened to mat4
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

void main() {
    // Per-draw transform, selected by the draw's index within the multi-draw.
    InstanceData inst = instances[gl_DrawIDARB];

    // gl_VertexID already accounts for baseVertex (set per draw command).
    PackedVertex pv = verts[gl_VertexID];

    vec2 pos_xy  = unpackHalf2x16(pv.pos_xy);
    vec2 pos_z0  = unpackHalf2x16(pv.pos_z);
    vec3 pos     = vec3(pos_xy, pos_z0.x);

    vec2 norm_xy = unpackHalf2x16(pv.norm_xy);
    vec2 norm_z0 = unpackHalf2x16(pv.norm_z);
    vec3 norm    = vec3(norm_xy, norm_z0.x);

    v_uv         = unpackHalf2x16(pv.uv_xy);

    v_world_pos  = (inst.model * vec4(pos, 1.0)).xyz;
    v_normal     = normalize(mat3(inst.normal) * norm);

    gl_Position  = u_view_proj * inst.model * vec4(pos, 1.0);
}
"#;

/// Fragment shader: simple Lambert + ambient with UV-based checker pattern.
pub const FRAG_SRC: &str = r#"
#version 430 core

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;

out vec4 frag_color;

const vec3 LIGHT_DIR   = normalize(vec3(1.0, 2.0, 1.5));
const vec3 LIGHT_COLOR = vec3(1.0, 0.95, 0.88);
const vec3 AMBIENT     = vec3(0.08, 0.10, 0.14);

void main() {
    // Checker pattern from UVs
    vec2 checker_uv = floor(v_uv * 6.0);
    float checker = mod(checker_uv.x + checker_uv.y, 2.0);
    vec3 base_color = mix(vec3(0.85, 0.82, 0.78), vec3(0.25, 0.22, 0.20), checker);

    vec3 n      = normalize(v_normal);
    float ndotl = max(dot(n, LIGHT_DIR), 0.0);
    vec3 color  = base_color * (AMBIENT + LIGHT_COLOR * ndotl);

    frag_color = vec4(color, 1.0);
}
"#;

use glow::{Context, HasContext, NativeProgram, NativeShader};

unsafe fn compile_shader(gl: &Context, kind: u32, src: &str) -> NativeShader {
    let shader = gl.create_shader(kind).expect("failed to create shader");
    gl.shader_source(shader, src);
    gl.compile_shader(shader);
    if !gl.get_shader_compile_status(shader) {
        panic!(
            "Shader compile error:\n{}",
            gl.get_shader_info_log(shader)
        );
    }
    shader
}

/// Compile and link the vertex + fragment shaders, returning the program handle.
pub unsafe fn build_program(gl: &Context) -> NativeProgram {
    let vert = compile_shader(gl, glow::VERTEX_SHADER,   VERT_SRC);
    let frag = compile_shader(gl, glow::FRAGMENT_SHADER, FRAG_SRC);

    let prog = gl.create_program().expect("failed to create program");
    gl.attach_shader(prog, vert);
    gl.attach_shader(prog, frag);
    gl.link_program(prog);

    if !gl.get_program_link_status(prog) {
        panic!("Program link error:\n{}", gl.get_program_info_log(prog));
    }

    gl.detach_shader(prog, vert);
    gl.detach_shader(prog, frag);
    gl.delete_shader(vert);
    gl.delete_shader(frag);

    prog
}
