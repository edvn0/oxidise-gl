pub const VERT_SRC:       &str = include_str!("shader.vert");
pub const FRAG_SRC:       &str = include_str!("shader.frag");
pub const CULL_SRC:       &str = include_str!("cull.comp");
pub const HZB_REDUCE_SRC: &str = include_str!("hzb_reduce.comp");

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

unsafe fn link_compute(gl: &Context, label: &str, comp: NativeShader) -> NativeProgram {
    let prog = gl.create_program().expect("failed to create program");
    gl.attach_shader(prog, comp);
    gl.link_program(prog);
    if !gl.get_program_link_status(prog) {
        panic!("{label} link error:\n{}", gl.get_program_info_log(prog));
    }
    gl.detach_shader(prog, comp);
    gl.delete_shader(comp);
    prog
}

/// Compile and link the culling compute shader.
pub unsafe fn build_cull_program(gl: &Context) -> NativeProgram {
    let comp = compile_shader(gl, glow::COMPUTE_SHADER, CULL_SRC);
    link_compute(gl, "Cull program", comp)
}

/// Compile and link the HZB min-reduction compute shader.
pub unsafe fn build_hzb_reduce_program(gl: &Context) -> NativeProgram {
    let comp = compile_shader(gl, glow::COMPUTE_SHADER, HZB_REDUCE_SRC);
    link_compute(gl, "HZB reduce program", comp)
}

/// Compile and link the vertex + fragment shaders.
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
