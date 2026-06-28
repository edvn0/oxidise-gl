//! The core scene renderer: a shared GPU arena drawn with two
//! `glMultiDrawElementsIndirect` calls per frame — one per HZB occlusion phase.
//!
//! Two-phase same-frame HZB occlusion culling:
//!   Phase 0 – frustum cull + test vs PREVIOUS frame's HZB → draw.
//!   Build HZB from the resulting depth buffer.
//!   Phase 1 – skip phase-0 survivors, frustum cull + test vs CURRENT HZB → draw.
//!
//! The renderer owns the offscreen render target (color + depth-texture FBO)
//! and the HZB pyramid texture. Call [`color_texture`] for the editor's imgui
//! Image, or [`blit_color_to_default`] before `swap_buffers` in a standalone app.

use crate::arena::{BoundingSphere, DrawCommand, GpuArena, GpuMaterial, InstanceData, MaterialRegistry, MeshAlloc, MeshKey, PackedVertex};
use crate::components::{Material, Mesh, Transform};
use crate::framebuffer::OffscreenTarget;
use crate::gl_buffer::{BufferUsage, GlBuffer};
use crate::shader::Shader;

use glam::{Mat3, Mat4, Vec3};
use glow::{Context, HasContext, NativeTexture};

// ── Arena constants ───────────────────────────────────────────────────────────
const MAX_VERTICES: usize = 65_536;
const MAX_INDICES:  usize = 131_072;
const MAX_DRAWS:    usize = 1_024;
const INITIAL_MATERIALS: usize = 16;

pub type MultiDrawElementsIndirectFn = unsafe extern "system" fn(
    mode:      u32,
    type_:     u32,
    indirect:  *const std::ffi::c_void,
    drawcount: i32,
    stride:    i32,
);

pub type MultiDrawElementsIndirectCountFn = unsafe extern "system" fn(
    mode:         u32,
    type_:        u32,
    indirect:     *const std::ffi::c_void,
    drawcount:    isize,
    maxdrawcount: i32,
    stride:       i32,
);

const PARAMETER_BUFFER: u32 = 0x80EE;

const QI_TIME:  usize = 0;
const QI_PRIMS: usize = 1;
const QI_VERT:  usize = 2;
const QI_FRAG:  usize = 3;

/// Per-frame GPU pipeline statistics returned by [`Renderer::render`].
///
/// GPU-measured fields reflect the **previous** frame (double-buffered queries).
#[derive(Clone, Copy, Debug, Default)]
pub struct PipelineStats {
    pub entities:             usize,
    pub batches:              usize,
    pub primitives_submitted: u64,
    pub vertex_invocations:   u64,
    pub fragment_invocations: u64,
    pub gpu_time_ms:          f64,
}

pub struct Renderer {
    // ── Shader programs ───────────────────────────────────────────────────────
    main_shader: Shader,
    cull_shader: Shader,
    hzb_shader:  Shader,

    // ── Geometry ──────────────────────────────────────────────────────────────
    vao:                 glow::NativeVertexArray,
    arena:               GpuArena,

    // ── Per-frame instance / cull buffers (phase 0) ───────────────────────────
    instance_buf:        GlBuffer<InstanceData>,
    culled_instance_buf: GlBuffer<InstanceData>,
    culled_indirect_buf: GlBuffer<DrawCommand>,
    bounding_buf:        GlBuffer<BoundingSphere>,
    batch_index_buf:     GlBuffer<u32>,
    batch_base_buf:      GlBuffer<u32>,

    // ── Per-frame instance / cull buffers (phase 1) ───────────────────────────
    culled_instance_buf_p2: GlBuffer<InstanceData>,
    culled_indirect_buf_p2: GlBuffer<DrawCommand>,

    // ── Phase-0 visibility flags (cleared each frame, set by phase-0 cull) ───
    visible_phase1_buf:  GlBuffer<u32>,

    // ── Shared draw-count parameter buffer ───────────────────────────────────
    draw_count_buf:      glow::NativeBuffer,

    // ── Materials ─────────────────────────────────────────────────────────────
    material_buf:        GlBuffer<GpuMaterial>,
    registry:            MaterialRegistry,

    // ── Offscreen render target (color + depth texture) ───────────────────────
    offscreen:           OffscreenTarget,

    // ── HZB pyramid (R32F, mip chain, min-reduced) ───────────────────────────
    hzb_tex:             Option<NativeTexture>,
    hzb_levels:          u32,
    hzb_size:            (u32, u32),

    // ── MDI + queries ─────────────────────────────────────────────────────────
    mdi_count:           MultiDrawElementsIndirectCountFn,
    stat_queries:        [[glow::NativeQuery; 4]; 2],
    query_frame:         usize,

    // ── Per-frame scratch (cleared at start of render, never dropped) ─────────
    scratch_batch_map:    std::collections::HashMap<MeshKey, u32>,
    scratch_batch_meshes: Vec<MeshAlloc>,
    scratch_instances:    Vec<InstanceData>,
    scratch_bounds:       Vec<BoundingSphere>,
    scratch_batch_indices: Vec<u32>,
    scratch_batch_sizes:  Vec<u32>,
    scratch_batch_base:   Vec<u32>,
    scratch_commands:     Vec<DrawCommand>,
    scratch_zeros:        Vec<u32>,
}

impl Renderer {
    pub unsafe fn new(
        gl: &Context,
        mdi_count: MultiDrawElementsIndirectCountFn,
    ) -> Self {
        println!(
            "GL: {} | {}",
            gl.get_parameter_string(glow::RENDERER),
            gl.get_parameter_string(glow::VERSION)
        );

        gl.enable(glow::DEPTH_TEST);
        // Reverse-Z: near plane → depth 1.0, far plane → depth 0.0.
        gl.depth_func(glow::GREATER);
        gl.clear_depth_f32(0.0);
        gl.depth_range_f32(1.0, 0.0);
        gl.enable(glow::CULL_FACE);
        gl.cull_face(glow::BACK);
        gl.clear_color(0.08, 0.10, 0.14, 1.0);

        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let main_shader = Shader::new_graphics(gl, &src.join("shader.vert"), &src.join("shader.frag"));
        let cull_shader = Shader::new_compute(gl,  &src.join("cull.comp"));
        let hzb_shader  = Shader::new_compute(gl,  &src.join("hzb_reduce.comp"));

        let vao = gl.create_vertex_array().expect("failed to create VAO");
        gl.bind_vertex_array(Some(vao));

        let arena = GpuArena::new(gl, MAX_VERTICES, MAX_INDICES);

        let instance_buf           = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let culled_instance_buf    = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let culled_indirect_buf    = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let culled_instance_buf_p2 = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let culled_indirect_buf_p2 = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let bounding_buf           = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let batch_index_buf        = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let batch_base_buf         = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let visible_phase1_buf     = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);

        let draw_count_buf = gl.create_buffer().expect("failed to create draw count buffer");
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(draw_count_buf));
        gl.buffer_data_u8_slice(
            glow::SHADER_STORAGE_BUFFER,
            bytemuck::bytes_of(&0u32),
            glow::DYNAMIC_DRAW,
        );
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);

        let material_buf = GlBuffer::new(gl, INITIAL_MATERIALS, BufferUsage::StaticDraw);
        let registry = MaterialRegistry::new();
        material_buf.upload_subrange(gl, &[GpuMaterial::DEFAULT], 0);

        // Initial 1×1 placeholder; resized on the first render() call.
        let offscreen = OffscreenTarget::new(gl, 1, 1);

        let c = || gl.create_query().expect("failed to create query");
        let stat_queries = [[c(), c(), c(), c()], [c(), c(), c(), c()]];

        Self {
            main_shader, cull_shader, hzb_shader,
            vao, arena,
            instance_buf, culled_instance_buf, culled_indirect_buf,
            culled_instance_buf_p2, culled_indirect_buf_p2,
            bounding_buf, batch_index_buf, batch_base_buf,
            visible_phase1_buf,
            draw_count_buf,
            material_buf, registry,
            offscreen,
            hzb_tex: None, hzb_levels: 0, hzb_size: (0, 0),
            mdi_count,
            stat_queries, query_frame: 0,
            scratch_batch_map:    std::collections::HashMap::new(),
            scratch_batch_meshes: Vec::new(),
            scratch_instances:    Vec::new(),
            scratch_bounds:       Vec::new(),
            scratch_batch_indices: Vec::new(),
            scratch_batch_sizes:  Vec::new(),
            scratch_batch_base:   Vec::new(),
            scratch_commands:     Vec::new(),
            scratch_zeros:        Vec::new(),
        }
    }

    // ── Mesh ─────────────────────────────────────────────────────────────────

    pub unsafe fn upload_mesh(
        &mut self, gl: &Context, vertices: &[PackedVertex], indices: &[u32],
    ) -> MeshAlloc {
        self.arena.push_mesh(gl, vertices, indices)
    }

    // ── Material management ───────────────────────────────────────────────────

    pub unsafe fn add_static_material(&mut self, gl: &Context, mat: GpuMaterial) -> u32 {
        let idx = self.registry.add_static(mat);
        self.upload_material_to_gpu(gl, idx, mat);
        idx
    }

    pub fn seal_static_materials(&mut self) -> usize { self.registry.seal() }

    pub unsafe fn add_override_material(&mut self, gl: &Context, mat: GpuMaterial) -> u32 {
        let idx = self.registry.add_override(mat);
        self.upload_material_to_gpu(gl, idx, mat);
        idx
    }

    pub unsafe fn update_override_material(&mut self, gl: &Context, idx: u32, mat: GpuMaterial) {
        let mat = self.registry.update_override(idx, mat);
        self.material_buf.upload_subrange(gl, &[mat], idx as usize);
    }

    pub fn material(&self, idx: u32) -> Option<&GpuMaterial> { self.registry.get(idx) }
    pub fn static_material_end(&self) -> Option<usize>       { self.registry.static_end() }
    pub fn is_override_material(&self, idx: u32) -> bool     { self.registry.is_override(idx) }

    unsafe fn upload_material_to_gpu(&mut self, gl: &Context, idx: u32, mat: GpuMaterial) {
        if self.registry.len() > self.material_buf.capacity {
            self.material_buf.grow(gl, self.material_buf.capacity * 2, BufferUsage::StaticDraw);
        }
        self.material_buf.upload_subrange(gl, &[mat], idx as usize);
    }

    // ── Offscreen target / HZB accessors ─────────────────────────────────────

    /// RGBA8 color texture of the most recent render; pass to imgui's `Image`.
    pub fn color_texture(&self) -> NativeTexture { self.offscreen.color_texture() }

    /// Blit the rendered scene color to the default (window) framebuffer.
    /// Call this after `render()` and before `swap_buffers()` in a standalone app.
    pub unsafe fn blit_color_to_default(&self, gl: &Context, dst_w: u32, dst_h: u32) {
        self.offscreen.blit_color_to_default(gl, dst_w, dst_h);
    }

    /// R32F HZB pyramid texture (swizzled R,R,R,1 for imgui display), or `None`
    /// before the first frame has rendered.
    pub fn hzb_texture(&self) -> Option<NativeTexture> { self.hzb_tex }

    /// Number of mip levels in the current HZB (0 before first frame).
    pub fn hzb_levels(&self) -> u32 { self.hzb_levels }

    /// Pixel dimensions of HZB mip 0 (0,0 before first frame).
    pub fn hzb_size(&self) -> (u32, u32) { self.hzb_size }

    // ── Shader hot-reload ─────────────────────────────────────────────────────

    /// Poll all shader source files for changes and recompile any that changed.
    /// Returns true if at least one shader was recompiled.
    pub unsafe fn try_reload_shaders(&mut self, gl: &Context) -> bool {
        let a = self.main_shader.try_reload(gl);
        let b = self.cull_shader.try_reload(gl);
        let c = self.hzb_shader.try_reload(gl);
        a || b || c
    }

    // ── HZB management ───────────────────────────────────────────────────────

    unsafe fn allocate_hzb(&mut self, gl: &Context, width: u32, height: u32) {
        if let Some(old) = self.hzb_tex.take() {
            gl.delete_texture(old);
        }

        let levels = ((width.max(height) as f32).log2().floor() as i32 + 1).max(1);

        let tex = gl.create_texture().expect("failed to create HZB texture");
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.tex_storage_2d(glow::TEXTURE_2D, levels, glow::R32F, width as i32, height as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST_MIPMAP_NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        // Swizzle G/B → R so imgui shows the depth as grayscale instead of red.
        // Cull shader reads .r which is unaffected; imageLoad ignores swizzle.
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_SWIZZLE_G, glow::RED as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_SWIZZLE_B, glow::RED as i32);
        gl.bind_texture(glow::TEXTURE_2D, None);

        // Clear all mip levels to 0.0 (= far value in reverse-Z; occludes nothing,
        // so the first frame's phase-0 draws everything as expected).
        let clear_fbo = gl.create_framebuffer().expect("failed to create HZB clear FBO");
        gl.bind_framebuffer(glow::FRAMEBUFFER, Some(clear_fbo));
        for level in 0..levels {
            gl.framebuffer_texture_2d(
                glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D, Some(tex), level,
            );
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
        }
        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
        gl.delete_framebuffer(clear_fbo);

        self.hzb_tex    = Some(tex);
        self.hzb_levels = levels as u32;
        self.hzb_size   = (width, height);
    }

    // ── Render ───────────────────────────────────────────────────────────────

    /// Draw every `(Transform, Mesh)` entity using two-phase HZB occlusion culling.
    ///
    /// `view` and `proj` are passed separately so the renderer can extract the
    /// camera position and projection scale factors needed for the HZB sphere test.
    ///
    /// The scene is rendered into an internal offscreen FBO. Call
    /// [`blit_color_to_default`] (standalone) or [`color_texture`] (editor) to
    /// consume the result. The offscreen target is automatically resized when
    /// `width`/`height` change.
    pub unsafe fn render(
        &mut self,
        gl: &Context,
        world: &hecs::World,
        view: Mat4,
        proj: Mat4,
        elapsed_secs: f32,
        width: u32,
        height: u32,
    ) -> PipelineStats {
        let view_proj    = proj * view;
        let eye_pos      = extract_eye_pos(&view);
        let proj_x_scale = proj.col(0)[0];
        let proj_y_scale = proj.col(1)[1];

        // ── (Re)allocate render target + HZB if viewport changed ──────────────
        self.offscreen.resize(gl, width, height);
        if self.hzb_size != (width, height) {
            self.allocate_hzb(gl, width, height);
        }

        // ── Bind offscreen FBO, set viewport, clear ───────────────────────────
        self.offscreen.bind(gl);
        gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

        // ── Pipeline statistics queries (double-buffered) ─────────────────────
        let write_set = self.query_frame % 2;
        let read_set  = 1 - write_set;
        gl.begin_query(glow::TIME_ELAPSED,                self.stat_queries[write_set][QI_TIME]);
        gl.begin_query(glow::PRIMITIVES_SUBMITTED,         self.stat_queries[write_set][QI_PRIMS]);
        gl.begin_query(glow::VERTEX_SHADER_INVOCATIONS,    self.stat_queries[write_set][QI_VERT]);
        gl.begin_query(glow::FRAGMENT_SHADER_INVOCATIONS,  self.stat_queries[write_set][QI_FRAG]);

        // ── Group entities into batches ───────────────────────────────────────
        self.scratch_batch_map.clear();
        self.scratch_batch_meshes.clear();
        self.scratch_instances.clear();
        self.scratch_bounds.clear();
        self.scratch_batch_indices.clear();

        for (_entity, (xf, mesh, mat)) in
            world.query::<(&Transform, &Mesh, Option<&Material>)>().iter()
        {
            let key = MeshKey::from(mesh.0);
            let batch_id = if let Some(&id) = self.scratch_batch_map.get(&key) {
                id
            } else {
                let id = self.scratch_batch_meshes.len() as u32;
                self.scratch_batch_meshes.push(mesh.0);
                self.scratch_batch_map.insert(key, id);
                id
            };

            let model          = xf.model_matrix(elapsed_secs);
            let material_index = mat.map(|m| m.0).unwrap_or(0);

            // Transform uses uniform scale, so the inverse-transpose of the upper-left
            // 3×3 equals (1/s)·R. The shader normalizes all normals, so the scale
            // cancels — pass model directly and avoid a per-entity matrix inverse.
            self.scratch_instances.push(InstanceData {
                model:  model.to_cols_array(),
                normal: model.to_cols_array(),
                material_index,
                _pad: [0; 3],
            });
            self.scratch_bounds.push(mesh.0.bounding_sphere);
            self.scratch_batch_indices.push(batch_id);
        }

        let n = self.scratch_instances.len();
        let m = self.scratch_batch_meshes.len();

        if n == 0 {
            gl.end_query(glow::FRAGMENT_SHADER_INVOCATIONS);
            gl.end_query(glow::VERTEX_SHADER_INVOCATIONS);
            gl.end_query(glow::PRIMITIVES_SUBMITTED);
            gl.end_query(glow::TIME_ELAPSED);
            self.query_frame = self.query_frame.wrapping_add(1);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            return PipelineStats::default();
        }

        // ── Build per-batch sizes and base offsets ────────────────────────────
        self.scratch_batch_sizes.clear();
        self.scratch_batch_sizes.resize(m, 0);
        for &id in &self.scratch_batch_indices {
            self.scratch_batch_sizes[id as usize] += 1;
        }
        self.scratch_batch_base.clear();
        self.scratch_batch_base.resize(m, 0);
        for b in 1..m {
            self.scratch_batch_base[b] = self.scratch_batch_base[b - 1] + self.scratch_batch_sizes[b - 1];
        }

        // Draw command template: instance_count = 0 (cull compute atomicAdds to it).
        self.scratch_commands.clear();
        for (b, mesh) in self.scratch_batch_meshes.iter().enumerate() {
            self.scratch_commands.push(mesh.draw_command_instanced(0, self.scratch_batch_base[b]));
        }

        // ── Grow per-frame buffers if needed ──────────────────────────────────
        if n > self.instance_buf.capacity {
            let cap = (self.instance_buf.capacity * 2).max(n);
            self.instance_buf.grow(gl, cap, BufferUsage::DynamicDraw);
            self.culled_instance_buf.grow(gl, cap, BufferUsage::DynamicDraw);
            self.culled_instance_buf_p2.grow(gl, cap, BufferUsage::DynamicDraw);
            self.bounding_buf.grow(gl, cap, BufferUsage::DynamicDraw);
            self.batch_index_buf.grow(gl, cap, BufferUsage::DynamicDraw);
            self.visible_phase1_buf.grow(gl, cap, BufferUsage::DynamicDraw);
        }
        if m > self.culled_indirect_buf.capacity {
            let cap = (self.culled_indirect_buf.capacity * 2).max(m);
            self.culled_indirect_buf.grow(gl, cap, BufferUsage::DynamicDraw);
            self.culled_indirect_buf_p2.grow(gl, cap, BufferUsage::DynamicDraw);
            self.batch_base_buf.grow(gl, cap, BufferUsage::DynamicDraw);
        }

        // ── Upload source data ────────────────────────────────────────────────
        self.instance_buf.upload_subrange(gl, &self.scratch_instances, 0);
        self.bounding_buf.upload_subrange(gl, &self.scratch_bounds, 0);
        self.batch_index_buf.upload_subrange(gl, &self.scratch_batch_indices, 0);
        self.batch_base_buf.upload_subrange(gl, &self.scratch_batch_base, 0);
        // Both phases start with instance_count = 0.
        self.culled_indirect_buf.upload_subrange(gl, &self.scratch_commands, 0);
        self.culled_indirect_buf_p2.upload_subrange(gl, &self.scratch_commands, 0);

        // ── Write M to draw-count parameter buffer ────────────────────────────
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.draw_count_buf));
        gl.buffer_sub_data_u8_slice(
            glow::SHADER_STORAGE_BUFFER, 0,
            bytemuck::bytes_of(&(m as u32)),
        );
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);

        // ── Clear visible_phase1 flags ────────────────────────────────────────
        // glow 0.13 doesn't wrap glClearBufferSubData, so upload zeros directly.
        self.scratch_zeros.clear();
        self.scratch_zeros.resize(n, 0);
        self.visible_phase1_buf.upload_subrange(gl, &self.scratch_zeros, 0);

        let groups = (n as u32 + 63) / 64;
        let planes = extract_frustum_planes(&view_proj);
        let hzb_tex = self.hzb_tex.expect("HZB must be allocated before render");
        let hzb_max_level = self.hzb_levels.saturating_sub(1);

        // ── PHASE 0 CULL ──────────────────────────────────────────────────────
        self.cull_shader.bind(gl);
        self.cull_shader.set_u32(gl,       "u_total_draws",    n as u32);
        self.cull_shader.set_vec4_slice(gl, "u_frustum_planes[0]", bytemuck::cast_slice(&planes));
        self.cull_shader.set_mat4(gl,       "u_view_proj",    &view_proj);
        self.cull_shader.set_vec2(gl,       "u_screen_size",   width as f32, height as f32);
        self.cull_shader.set_u32(gl,        "u_hzb_max_level", hzb_max_level);
        self.cull_shader.set_f32(gl,        "u_proj_x_scale",  proj_x_scale);
        self.cull_shader.set_f32(gl,        "u_proj_y_scale",  proj_y_scale);
        self.cull_shader.set_vec3(gl,       "u_eye_pos",       eye_pos);

        self.instance_buf.bind_as_ssbo(gl, 0);
        self.bounding_buf.bind_as_ssbo(gl, 1);
        self.batch_index_buf.bind_as_ssbo(gl, 2);
        self.culled_indirect_buf.bind_as_ssbo(gl, 3);
        self.culled_instance_buf.bind_as_ssbo(gl, 4);
        self.batch_base_buf.bind_as_ssbo(gl, 5);
        self.visible_phase1_buf.bind_as_ssbo(gl, 6);

        // Bind previous frame's HZB (0.0-cleared on first frame → no occlusion).
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(hzb_tex));

        self.cull_shader.set_u32(gl, "u_phase", 0);
        gl.dispatch_compute(groups, 1, 1);
        gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT | glow::COMMAND_BARRIER_BIT);

        // ── PHASE 0 DRAW ──────────────────────────────────────────────────────
        self.main_shader.bind(gl);
        gl.bind_vertex_array(Some(self.vao));
        self.arena.bind(gl, 0);
        self.culled_instance_buf.bind_as_ssbo(gl, 1);
        self.material_buf.bind_as_ssbo(gl, 2);
        self.main_shader.set_mat4(gl, "u_view_proj", &view_proj);
        self.main_shader.set_vec3(gl, "u_eye_pos",   eye_pos);
        self.culled_indirect_buf.bind_as_indirect(gl);
        gl.bind_buffer(PARAMETER_BUFFER, Some(self.draw_count_buf));
        (self.mdi_count)(glow::TRIANGLES, glow::UNSIGNED_INT, std::ptr::null(), 0, m as i32, 0);
        GlBuffer::<DrawCommand>::unbind_indirect(gl);
        gl.bind_buffer(PARAMETER_BUFFER, None);
        gl.bind_vertex_array(None);

        // ── HZB BUILD (from phase-0 depth) ────────────────────────────────────
        // Barrier: rasterizer depth writes → compute texture reads.
        gl.memory_barrier(glow::TEXTURE_FETCH_BARRIER_BIT | glow::SHADER_IMAGE_ACCESS_BARRIER_BIT);

        self.hzb_shader.bind(gl);

        // Depth texture at texture unit 0 (layout(binding=0) in hzb_reduce.comp).
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(self.offscreen.depth_texture()));

        // Level 0: copy depth → HZB mip 0 (1:1, same dimensions).
        gl.bind_image_texture(2, hzb_tex, 0, false, 0, glow::WRITE_ONLY, glow::R32F);
        self.hzb_shader.set_uvec2(gl, "u_src_size", width, height);
        self.hzb_shader.set_u32(gl,   "u_is_first", 1);
        gl.dispatch_compute((width + 7) / 8, (height + 7) / 8, 1);
        gl.memory_barrier(glow::SHADER_IMAGE_ACCESS_BARRIER_BIT);

        // Levels 1..hzb_levels: min-reduce previous level → current level.
        let mut sw = width;
        let mut sh = height;
        for level in 1..self.hzb_levels {
            let dw = (sw / 2).max(1);
            let dh = (sh / 2).max(1);
            gl.bind_image_texture(1, hzb_tex, (level - 1) as i32, false, 0, glow::READ_ONLY,  glow::R32F);
            gl.bind_image_texture(2, hzb_tex,  level      as i32, false, 0, glow::WRITE_ONLY, glow::R32F);
            self.hzb_shader.set_uvec2(gl, "u_src_size", sw, sh);
            self.hzb_shader.set_u32(gl,   "u_is_first", 0);
            gl.dispatch_compute((dw + 7) / 8, (dh + 7) / 8, 1);
            gl.memory_barrier(glow::SHADER_IMAGE_ACCESS_BARRIER_BIT);
            sw = dw;
            sh = dh;
        }

        // Barrier: HZB writes → cull shader texture reads.
        gl.memory_barrier(glow::TEXTURE_FETCH_BARRIER_BIT);

        // ── PHASE 1 CULL ──────────────────────────────────────────────────────
        self.cull_shader.bind(gl);

        // Restore cull SSBO bindings (draw pass overwrote 0, 1, 2).
        self.instance_buf.bind_as_ssbo(gl, 0);
        self.bounding_buf.bind_as_ssbo(gl, 1);
        self.batch_index_buf.bind_as_ssbo(gl, 2);
        self.culled_indirect_buf_p2.bind_as_ssbo(gl, 3);
        self.culled_instance_buf_p2.bind_as_ssbo(gl, 4);
        self.batch_base_buf.bind_as_ssbo(gl, 5);
        // binding 6 (visible_phase1_buf) unchanged since it was set before phase 0.

        // Bind CURRENT frame's HZB (just built above).
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(hzb_tex));

        self.cull_shader.set_u32(gl, "u_phase", 1);
        gl.dispatch_compute(groups, 1, 1);
        gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT | glow::COMMAND_BARRIER_BIT);

        // ── PHASE 1 DRAW ──────────────────────────────────────────────────────
        self.main_shader.bind(gl);
        gl.bind_vertex_array(Some(self.vao));
        self.arena.bind(gl, 0);
        self.culled_instance_buf_p2.bind_as_ssbo(gl, 1);
        self.material_buf.bind_as_ssbo(gl, 2);
        // u_view_proj / u_eye_pos already set from phase-0 draw (program uniform state persists).
        self.culled_indirect_buf_p2.bind_as_indirect(gl);
        gl.bind_buffer(PARAMETER_BUFFER, Some(self.draw_count_buf));
        (self.mdi_count)(glow::TRIANGLES, glow::UNSIGNED_INT, std::ptr::null(), 0, m as i32, 0);
        GlBuffer::<DrawCommand>::unbind_indirect(gl);
        gl.bind_buffer(PARAMETER_BUFFER, None);
        gl.bind_vertex_array(None);
        gl.use_program(None);

        // Unbind depth texture so it's free for the next HZB build.
        gl.bind_texture(glow::TEXTURE_2D, None);

        // Leave the offscreen FBO bound; callers that need to restore the default
        // FB (editor: explicit at line 432, standalone: via blit_color_to_default)
        // handle it themselves.

        // ── End queries and read previous frame's results ─────────────────────
        gl.end_query(glow::FRAGMENT_SHADER_INVOCATIONS);
        gl.end_query(glow::VERTEX_SHADER_INVOCATIONS);
        gl.end_query(glow::PRIMITIVES_SUBMITTED);
        gl.end_query(glow::TIME_ELAPSED);

        let (primitives_submitted, vertex_invocations, fragment_invocations, gpu_time_ms) =
            if self.query_frame > 0 {
                (
                    gl.get_query_parameter_u32(self.stat_queries[read_set][QI_PRIMS], glow::QUERY_RESULT) as u64,
                    gl.get_query_parameter_u32(self.stat_queries[read_set][QI_VERT],  glow::QUERY_RESULT) as u64,
                    gl.get_query_parameter_u32(self.stat_queries[read_set][QI_FRAG],  glow::QUERY_RESULT) as u64,
                    gl.get_query_parameter_u32(self.stat_queries[read_set][QI_TIME],  glow::QUERY_RESULT) as f64 / 1_000_000.0,
                )
            } else {
                (0, 0, 0, 0.0)
            };

        self.query_frame = self.query_frame.wrapping_add(1);

        PipelineStats {
            entities: n,
            batches: m,
            primitives_submitted,
            vertex_invocations,
            fragment_invocations,
            gpu_time_ms,
        }
    }

    pub unsafe fn cleanup(&self, gl: &Context) {
        self.arena.cleanup(gl);
        self.instance_buf.cleanup(gl);
        self.culled_instance_buf.cleanup(gl);
        self.culled_indirect_buf.cleanup(gl);
        self.culled_instance_buf_p2.cleanup(gl);
        self.culled_indirect_buf_p2.cleanup(gl);
        self.bounding_buf.cleanup(gl);
        self.batch_index_buf.cleanup(gl);
        self.batch_base_buf.cleanup(gl);
        self.visible_phase1_buf.cleanup(gl);
        gl.delete_buffer(self.draw_count_buf);
        self.material_buf.cleanup(gl);
        self.offscreen.cleanup(gl);
        if let Some(hzb) = self.hzb_tex {
            gl.delete_texture(hzb);
        }
        for set in &self.stat_queries {
            for &q in set { gl.delete_query(q); }
        }
        gl.delete_vertex_array(self.vao);
        self.hzb_shader.cleanup(gl);
        self.cull_shader.cleanup(gl);
        self.main_shader.cleanup(gl);
    }
}

// ── Free functions ────────────────────────────────────────────────────────────

/// Extract the 6 world-space frustum planes from a combined view-projection matrix
/// (Gribb-Hartmann method).
fn extract_frustum_planes(vp: &Mat4) -> [[f32; 4]; 6] {
    let m   = vp.to_cols_array_2d();
    let row = |i: usize| -> [f32; 4] { [m[0][i], m[1][i], m[2][i], m[3][i]] };
    let r0  = row(0); let r1 = row(1); let r2 = row(2); let r3 = row(3);
    let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] { [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]] };
    let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] { [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]] };
    [
        add(r3, r0), // left
        sub(r3, r0), // right
        add(r3, r1), // bottom
        sub(r3, r1), // top
        add(r3, r2), // near
        sub(r3, r2), // far
    ]
}

/// Recover the world-space camera position from a rigid-body view matrix.
/// view = [R | -R*eye], so eye = R^T * (−view.col(3).xyz).
fn extract_eye_pos(view: &Mat4) -> Vec3 {
    let r = Mat3::from_mat4(*view);
    let t = view.col(3).truncate();
    -(r.transpose() * t)
}
