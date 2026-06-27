pub const VERT_SRC: &str = include_str!("shader.vert");
pub const FRAG_SRC: &str = include_str!("shader.frag");
pub const CULL_SRC: &str = include_str!("cull.comp");

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

/// Compile and link the culling compute shader, returning the program handle.
pub unsafe fn build_cull_program(gl: &Context) -> NativeProgram {
    let comp = compile_shader(gl, glow::COMPUTE_SHADER, CULL_SRC);
    let prog = gl.create_program().expect("failed to create cull program");
    gl.attach_shader(prog, comp);
    gl.link_program(prog);
    if !gl.get_program_link_status(prog) {
        panic!("Cull program link error:\n{}", gl.get_program_info_log(prog));
    }
    gl.detach_shader(prog, comp);
    gl.delete_shader(comp);
    prog
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
