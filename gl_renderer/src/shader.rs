use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use glam::{Mat4, Vec3};
use glow::{Context, HasContext, NativeProgram, NativeShader, NativeUniformLocation};

pub struct Shader {
    program:  NativeProgram,
    // Locations are cached on first use; cleared on hot-reload.
    uniforms: RefCell<HashMap<String, Option<NativeUniformLocation>>>,
    sources:  Sources,
    mtimes:   Vec<SystemTime>,
}

enum Sources {
    Graphics { vert: PathBuf, frag: PathBuf },
    Compute  { comp: PathBuf },
}

impl Shader {
    pub unsafe fn new_graphics(gl: &Context, vert: &Path, frag: &Path) -> Self {
        let program = compile_graphics(gl, &read(vert), &read(frag));
        Self {
            program,
            uniforms: RefCell::new(HashMap::new()),
            sources:  Sources::Graphics { vert: vert.to_owned(), frag: frag.to_owned() },
            mtimes:   vec![mtime(vert), mtime(frag)],
        }
    }

    pub unsafe fn new_compute(gl: &Context, comp: &Path) -> Self {
        let program = compile_compute(gl, &read(comp));
        Self {
            program,
            uniforms: RefCell::new(HashMap::new()),
            sources:  Sources::Compute { comp: comp.to_owned() },
            mtimes:   vec![mtime(comp)],
        }
    }

    pub unsafe fn bind(&self, gl: &Context) {
        gl.use_program(Some(self.program));
    }

    /// Returns true and recompiles if any source file has a newer mtime.
    /// On compile error, prints to stderr and keeps the existing program.
    pub unsafe fn try_reload(&mut self, gl: &Context) -> bool {
        let paths = self.paths();
        let new_mtimes: Vec<SystemTime> = paths.iter().map(|p| mtime(p)).collect();
        if new_mtimes == self.mtimes { return false; }

        let result = match &self.sources {
            Sources::Graphics { vert, frag } => {
                match (fs::read_to_string(vert), fs::read_to_string(frag)) {
                    (Ok(vs), Ok(fs)) => try_compile_graphics(gl, &vs, &fs),
                    (Err(e), _) | (_, Err(e)) => Err(e.to_string()),
                }
            }
            Sources::Compute { comp } => {
                match fs::read_to_string(comp) {
                    Ok(cs)  => try_compile_compute(gl, &cs),
                    Err(e)  => Err(e.to_string()),
                }
            }
        };

        // Update mtimes unconditionally so we don't spam on every frame after a bad save.
        self.mtimes = new_mtimes;

        match result {
            Ok(new_prog) => {
                gl.delete_program(self.program);
                self.program = new_prog;
                self.uniforms.borrow_mut().clear();
                true
            }
            Err(e) => {
                eprintln!("[shader reload] {e}");
                false
            }
        }
    }

    pub unsafe fn cleanup(&self, gl: &Context) {
        gl.delete_program(self.program);
    }

    // ── Uniform setters ───────────────────────────────────────────────────────

    pub unsafe fn set_f32(&self, gl: &Context, name: &str, v: f32) {
        let loc = self.loc(gl, name);
        gl.uniform_1_f32(loc.as_ref(), v);
    }

    pub unsafe fn set_u32(&self, gl: &Context, name: &str, v: u32) {
        let loc = self.loc(gl, name);
        gl.uniform_1_u32(loc.as_ref(), v);
    }

    pub unsafe fn set_vec2(&self, gl: &Context, name: &str, x: f32, y: f32) {
        let loc = self.loc(gl, name);
        gl.uniform_2_f32(loc.as_ref(), x, y);
    }

    pub unsafe fn set_uvec2(&self, gl: &Context, name: &str, x: u32, y: u32) {
        let loc = self.loc(gl, name);
        gl.uniform_2_u32(loc.as_ref(), x, y);
    }

    pub unsafe fn set_vec3(&self, gl: &Context, name: &str, v: Vec3) {
        let loc = self.loc(gl, name);
        gl.uniform_3_f32(loc.as_ref(), v.x, v.y, v.z);
    }

    pub unsafe fn set_mat4(&self, gl: &Context, name: &str, m: &Mat4) {
        let loc = self.loc(gl, name);
        gl.uniform_matrix_4_f32_slice(loc.as_ref(), false, &m.to_cols_array());
    }

    /// For array uniforms (e.g. `"u_frustum_planes[0]"`): pass a flat &[f32].
    pub unsafe fn set_vec4_slice(&self, gl: &Context, name: &str, data: &[f32]) {
        let loc = self.loc(gl, name);
        gl.uniform_4_f32_slice(loc.as_ref(), data);
    }

    // Looks up and caches the location on first call; returns it for subsequent calls.
    unsafe fn loc(&self, gl: &Context, name: &str) -> Option<NativeUniformLocation> {
        let mut map = self.uniforms.borrow_mut();
        if let Some(&cached) = map.get(name) {
            return cached;
        }
        let loc = gl.get_uniform_location(self.program, name);
        map.insert(name.to_string(), loc);
        loc
    }

    fn paths(&self) -> Vec<PathBuf> {
        match &self.sources {
            Sources::Graphics { vert, frag } => vec![vert.clone(), frag.clone()],
            Sources::Compute  { comp }       => vec![comp.clone()],
        }
    }
}

// ── Private compilation helpers ───────────────────────────────────────────────

fn read(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn mtime(path: &Path) -> SystemTime {
    fs::metadata(path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

unsafe fn compile_stage(gl: &Context, kind: u32, src: &str) -> Result<NativeShader, String> {
    let s = gl.create_shader(kind).map_err(|e| e.to_string())?;
    gl.shader_source(s, src);
    gl.compile_shader(s);
    if gl.get_shader_compile_status(s) {
        Ok(s)
    } else {
        let log = gl.get_shader_info_log(s);
        gl.delete_shader(s);
        Err(log)
    }
}

unsafe fn link(gl: &Context, stages: &[NativeShader]) -> Result<NativeProgram, String> {
    let prog = gl.create_program().map_err(|e| e.to_string())?;
    for &s in stages { gl.attach_shader(prog, s); }
    gl.link_program(prog);
    for &s in stages { gl.detach_shader(prog, s); gl.delete_shader(s); }
    if gl.get_program_link_status(prog) {
        Ok(prog)
    } else {
        let log = gl.get_program_info_log(prog);
        gl.delete_program(prog);
        Err(log)
    }
}

unsafe fn try_compile_graphics(gl: &Context, vs: &str, fs: &str) -> Result<NativeProgram, String> {
    let vert = compile_stage(gl, glow::VERTEX_SHADER, vs)?;
    let frag = match compile_stage(gl, glow::FRAGMENT_SHADER, fs) {
        Ok(s)  => s,
        Err(e) => { gl.delete_shader(vert); return Err(e); }
    };
    link(gl, &[vert, frag])
}

unsafe fn try_compile_compute(gl: &Context, cs: &str) -> Result<NativeProgram, String> {
    let comp = compile_stage(gl, glow::COMPUTE_SHADER, cs)?;
    link(gl, &[comp])
}

unsafe fn compile_graphics(gl: &Context, vs: &str, fs: &str) -> NativeProgram {
    try_compile_graphics(gl, vs, fs)
        .unwrap_or_else(|e| panic!("shader compile error:\n{e}"))
}

unsafe fn compile_compute(gl: &Context, cs: &str) -> NativeProgram {
    try_compile_compute(gl, cs)
        .unwrap_or_else(|e| panic!("shader compile error:\n{e}"))
}
