//! The core scene renderer: a shared GPU arena drawn with a single
//! `glMultiDrawElementsIndirect` call, per-object transforms supplied from the
//! ECS world via an instance SSBO.
//!
//! The renderer does **not** own the `glow::Context` or the `hecs::World`; the
//! host passes `&Context` into each method and the world into [`Renderer::render`].
//! This lets the editor share one GL context between this renderer and imgui.

use crate::arena::{DrawCommand, GpuArena, InstanceData, MeshAlloc, PackedVertex};
use crate::components::{Mesh, Transform};
use crate::gl_buffer::{BufferUsage, GlBuffer};
use crate::shaders::build_program;

use glam::{Mat3, Mat4};
use glow::{Context, HasContext};

// ── Arena constants ───────────────────────────────────────────────────────────
const MAX_VERTICES: usize = 65_536;
const MAX_INDICES:  usize = 131_072;
/// Upper bound on objects drawn in a single multi-draw-indirect call.
const MAX_DRAWS:    usize = 1_024;

/// `glMultiDrawElementsIndirect` is not wrapped by glow's `HasContext`, so the
/// host loads the raw entry point and hands it to the renderer. Core since GL 4.3.
pub type MultiDrawElementsIndirectFn = unsafe extern "system" fn(
    mode:      u32,
    type_:     u32,
    indirect:  *const std::ffi::c_void,
    drawcount: i32,
    stride:    i32,
);

/// GPU resources and per-frame draw machinery. Owns no GL context or world.
pub struct Renderer {
    program:       glow::NativeProgram,
    vao:           glow::NativeVertexArray,
    arena:         GpuArena,
    /// Per-object transforms, rebuilt each frame and indexed by gl_DrawIDARB.
    instance_buf:  GlBuffer<InstanceData>,
    /// Indirect draw commands consumed by glMultiDrawElementsIndirect.
    indirect_buf:  GlBuffer<DrawCommand>,
    loc_view_proj: Option<glow::NativeUniformLocation>,
    mdi:           MultiDrawElementsIndirectFn,
}

impl Renderer {
    /// Set up GL state, compile the program, create the VAO, allocate the shared
    /// arena and the per-frame instance/indirect buffers. No meshes or scene yet —
    /// call [`Renderer::upload_mesh`] and build a `hecs::World` in the host.
    pub unsafe fn new(gl: &Context, mdi: MultiDrawElementsIndirectFn) -> Self {
        println!(
            "GL: {} | {}",
            gl.get_parameter_string(glow::RENDERER),
            gl.get_parameter_string(glow::VERSION)
        );

        gl.enable(glow::DEPTH_TEST);
        // Reverse-Z: near plane → depth 1.0, far plane → depth 0.0.
        // Float precision is concentrated near 0.0, which is now the far end —
        // dramatically reduces z-fighting for distant objects.
        gl.depth_func(glow::GREATER);
        gl.clear_depth_f32(0.0);
        gl.depth_range_f32(1.0, 0.0);
        gl.enable(glow::CULL_FACE);
        gl.cull_face(glow::BACK);
        gl.clear_color(0.08, 0.10, 0.14, 1.0);

        let program = build_program(gl);

        // Core profile requires a VAO even for vertex pulling.
        let vao = gl.create_vertex_array().expect("failed to create VAO");
        gl.bind_vertex_array(Some(vao));

        let arena = GpuArena::new(gl, MAX_VERTICES, MAX_INDICES);

        // Instance data (SSBO @ binding 1) and indirect commands; both rebuilt
        // each frame from the ECS world, so DynamicDraw.
        let instance_buf = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let indirect_buf = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);

        // Cache the single view-projection uniform location once.
        let loc_view_proj = gl.get_uniform_location(program, "u_view_proj");

        Self {
            program,
            vao,
            arena,
            instance_buf,
            indirect_buf,
            loc_view_proj,
            mdi,
        }
    }

    /// Sub-allocate a mesh into the shared arena and return its range. Indices
    /// must be local (0-based); the base vertex is applied at draw time.
    pub unsafe fn upload_mesh(
        &mut self,
        gl: &Context,
        vertices: &[PackedVertex],
        indices: &[u32],
    ) -> MeshAlloc {
        self.arena.push_mesh(gl, vertices, indices)
    }

    /// Draw every `(Transform, Mesh)` entity in `world` in a single MDI call,
    /// into the currently-bound framebuffer (default FB or an offscreen target).
    /// `elapsed_secs` animates the transforms; `view_proj` is the camera matrix.
    /// Returns the number of draws issued.
    pub unsafe fn render(
        &mut self,
        gl: &Context,
        world: &hecs::World,
        view_proj: Mat4,
        elapsed_secs: f32,
        width: u32,
        height: u32,
    ) -> usize {
        gl.viewport(0, 0, width as i32, height as i32);
        gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

        gl.use_program(Some(self.program));
        gl.bind_vertex_array(Some(self.vao));

        // Build this frame's per-object instance data and indirect draw commands
        // by querying the ECS world. The draw index doubles as base_instance, so
        // each command's gl_DrawIDARB selects its matching instance entry.
        let mut instances: Vec<InstanceData> = Vec::with_capacity(MAX_DRAWS);
        let mut commands:  Vec<DrawCommand>  = Vec::with_capacity(MAX_DRAWS);
        for (_entity, (xf, mesh)) in world.query::<(&Transform, &Mesh)>().iter() {
            let model  = xf.model_matrix(elapsed_secs);
            // Inverse-transpose of the model 3x3 (correct under non-uniform
            // scale), widened to mat4 for the std430 instance layout.
            let normal = Mat4::from_mat3(Mat3::from_mat4(model.inverse().transpose()));
            let draw_id = commands.len() as u32;
            instances.push(InstanceData {
                model:  model.to_cols_array(),
                normal: normal.to_cols_array(),
            });
            commands.push(mesh.0.draw_command(draw_id));
        }

        // Grow per-frame buffers if this frame has more draws than capacity.
        if instances.len() > self.instance_buf.capacity {
            let new_cap = (self.instance_buf.capacity * 2).max(instances.len());
            self.instance_buf.grow(gl, new_cap, BufferUsage::DynamicDraw);
            self.indirect_buf.grow(gl, new_cap, BufferUsage::DynamicDraw);
        }

        // Upload + bind per-frame buffers: instance SSBO at binding 1, then the
        // draw-indirect buffer.
        self.instance_buf.upload_subrange(gl, &instances, 0);
        self.instance_buf.bind_as_ssbo(gl, 1);
        self.indirect_buf.upload_subrange(gl, &commands, 0);
        self.indirect_buf.bind_as_indirect(gl);

        // Shared vertex SSBO at binding 0 and the index buffer (captured by VAO).
        self.arena.bind(gl, 0);

        gl.uniform_matrix_4_f32_slice(
            self.loc_view_proj.as_ref(),
            false,
            &view_proj.to_cols_array(),
        );

        // One call draws the whole scene. With DRAW_INDIRECT_BUFFER bound, the
        // `indirect` argument is a byte offset (0); stride 0 = tightly packed.
        (self.mdi)(
            glow::TRIANGLES,
            glow::UNSIGNED_INT,
            std::ptr::null(),
            commands.len() as i32,
            0,
        );

        GlBuffer::<DrawCommand>::unbind_indirect(gl);
        gl.bind_vertex_array(None);
        gl.use_program(None);

        commands.len()
    }

    pub unsafe fn cleanup(&self, gl: &Context) {
        self.arena.cleanup(gl);
        self.instance_buf.cleanup(gl);
        self.indirect_buf.cleanup(gl);
        gl.delete_vertex_array(self.vao);
        gl.delete_program(self.program);
    }
}
