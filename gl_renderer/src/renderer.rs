//! The core scene renderer: a shared GPU arena drawn with a single
//! `glMultiDrawElementsIndirect` call, per-object transforms supplied from the
//! ECS world via an instance SSBO.
//!
//! The renderer does **not** own the `glow::Context` or the `hecs::World`; the
//! host passes `&Context` into each method and the world into [`Renderer::render`].
//! This lets the editor share one GL context between this renderer and imgui.

use crate::arena::{BoundingSphere, DrawCommand, GpuArena, GpuMaterial, InstanceData, MaterialRegistry, MeshAlloc, MeshKey, PackedVertex};
use crate::components::{Material, Mesh, Transform};
use crate::gl_buffer::{BufferUsage, GlBuffer};
use crate::shaders::{build_cull_program, build_program};

use glam::{Mat3, Mat4};
use glow::{Context, HasContext};

// ── Arena constants ───────────────────────────────────────────────────────────
const MAX_VERTICES: usize = 65_536;
const MAX_INDICES:  usize = 131_072;
/// Upper bound on draw batches per frame (= unique meshes). Grows automatically.
const MAX_DRAWS:    usize = 1_024;
/// Initial material SSBO slot count. Grows on demand via 2× doubling.
const INITIAL_MATERIALS: usize = 16;

/// `glMultiDrawElementsIndirect` is not wrapped by glow's `HasContext`, so the
/// host loads the raw entry point and hands it to the renderer. Core since GL 4.3.
pub type MultiDrawElementsIndirectFn = unsafe extern "system" fn(
    mode:      u32,
    type_:     u32,
    indirect:  *const std::ffi::c_void,
    drawcount: i32,
    stride:    i32,
);

/// `glMultiDrawElementsIndirectCount` reads the draw count from a GPU buffer
/// (bound as `GL_PARAMETER_BUFFER`) instead of a CPU value. Core since GL 4.6.
pub type MultiDrawElementsIndirectCountFn = unsafe extern "system" fn(
    mode:         u32,
    type_:        u32,
    indirect:     *const std::ffi::c_void,
    drawcount:    isize,  // byte offset into GL_PARAMETER_BUFFER
    maxdrawcount: i32,
    stride:       i32,
);

/// `GL_PARAMETER_BUFFER` target for binding the GPU-side draw-count buffer.
/// Introduced with ARB_indirect_parameters / GL 4.6 (value 0x80EE).
const PARAMETER_BUFFER: u32 = 0x80EE;

// Query slot indices within each double-buffered set.
const QI_TIME:  usize = 0; // GL_TIME_ELAPSED
const QI_PRIMS: usize = 1; // GL_PRIMITIVES_SUBMITTED
const QI_VERT:  usize = 2; // GL_VERTEX_SHADER_INVOCATIONS
const QI_FRAG:  usize = 3; // GL_FRAGMENT_SHADER_INVOCATIONS

/// Per-frame GPU pipeline statistics returned by [`Renderer::render`].
///
/// Rasterization counts and `gpu_time_ms` reflect the **previous** frame —
/// they are read from double-buffered query objects to avoid a GPU stall.
/// All GPU-measured fields are zero on the first frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct PipelineStats {
    /// Total entities submitted this frame.
    pub entities:             usize,
    /// Instanced draw batches (unique meshes) after grouping.
    pub batches:              usize,
    /// Triangles submitted to the rasterizer (previous frame, GPU-measured).
    pub primitives_submitted: u64,
    /// Vertex shader invocations (previous frame, GPU-measured).
    pub vertex_invocations:   u64,
    /// Fragment shader invocations (previous frame, GPU-measured).
    pub fragment_invocations: u64,
    /// GPU time for the full render pass in milliseconds (previous frame).
    pub gpu_time_ms:          f64,
}

/// GPU resources and per-frame draw machinery. Owns no GL context or world.
pub struct Renderer {
    program:             glow::NativeProgram,
    cull_program:        glow::NativeProgram,
    vao:                 glow::NativeVertexArray,
    arena:               GpuArena,
    /// Per-entity source instance data, rebuilt each frame.
    instance_buf:        GlBuffer<InstanceData>,
    /// Compacted visible instance data written by the cull compute, read by the vertex shader.
    culled_instance_buf: GlBuffer<InstanceData>,
    /// Draw commands (one per unique mesh): vertex/index info set by CPU, instance_count
    /// atomicAdd'd by the cull compute, then consumed by MDI.
    culled_indirect_buf: GlBuffer<DrawCommand>,
    /// Model-space bounding spheres, one per entity, input to the cull compute.
    bounding_buf:        GlBuffer<BoundingSphere>,
    /// Per-entity batch index mapping each entity to its unique-mesh slot.
    batch_index_buf:     GlBuffer<u32>,
    /// Per-batch starting offset into culled_instance_buf (prefix sum of batch sizes).
    batch_base_buf:      GlBuffer<u32>,
    /// Single-u32 buffer: holds M (batch count) read by glMultiDrawElementsIndirectCount.
    draw_count_buf:      glow::NativeBuffer,
    /// GPU material SSBO. Layout: [static zone | override zone].
    material_buf:        GlBuffer<GpuMaterial>,
    /// CPU-side bookkeeping and zone enforcement; GPU uploads are done here in Renderer.
    registry:            MaterialRegistry,
    loc_view_proj:       Option<glow::NativeUniformLocation>,
    loc_cull_total:      Option<glow::NativeUniformLocation>,
    loc_frustum_planes:  Option<glow::NativeUniformLocation>,
    mdi_count:           MultiDrawElementsIndirectCountFn,
    /// Double-buffered GL query objects for pipeline statistics.
    /// `stat_queries[frame % 2]` is the write set; the other is the read set.
    stat_queries:        [[glow::NativeQuery; 4]; 2],
    /// Monotonically-increasing frame counter; low bit selects the active query set.
    query_frame:         usize,
}

impl Renderer {
    /// Set up GL state, compile the program, create the VAO, allocate the shared
    /// arena and the per-frame instance/indirect buffers. No meshes or scene yet —
    /// call [`Renderer::upload_mesh`] and build a `hecs::World` in the host.
    ///
    /// The default material (index 0) is pre-loaded into the static zone.
    /// Call [`seal_static_materials`] after all mesh/material loading is done.
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
        // Float precision is concentrated near 0.0, which is now the far end —
        // dramatically reduces z-fighting for distant objects.
        gl.depth_func(glow::GREATER);
        gl.clear_depth_f32(0.0);
        gl.depth_range_f32(1.0, 0.0);
        gl.enable(glow::CULL_FACE);
        gl.cull_face(glow::BACK);
        gl.clear_color(0.08, 0.10, 0.14, 1.0);

        let program      = build_program(gl);
        let cull_program = build_cull_program(gl);

        // Core profile requires a VAO even for vertex pulling.
        let vao = gl.create_vertex_array().expect("failed to create VAO");
        gl.bind_vertex_array(Some(vao));

        let arena = GpuArena::new(gl, MAX_VERTICES, MAX_INDICES);

        let instance_buf        = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let culled_instance_buf = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let culled_indirect_buf = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let bounding_buf        = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let batch_index_buf     = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);
        let batch_base_buf      = GlBuffer::new(gl, MAX_DRAWS, BufferUsage::DynamicDraw);

        // Single-u32 buffer used as an atomic counter by the cull compute and as a
        // GL_PARAMETER_BUFFER draw count for glMultiDrawElementsIndirectCount.
        let draw_count_buf = gl.create_buffer().expect("failed to create draw count buffer");
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(draw_count_buf));
        gl.buffer_data_u8_slice(
            glow::SHADER_STORAGE_BUFFER,
            bytemuck::bytes_of(&0u32),
            glow::DYNAMIC_DRAW,
        );
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);

        // Pre-load the default material at index 0.
        let material_buf = GlBuffer::new(gl, INITIAL_MATERIALS, BufferUsage::StaticDraw);
        let registry = MaterialRegistry::new();
        material_buf.upload_subrange(gl, &[GpuMaterial::DEFAULT], 0);

        let loc_view_proj      = gl.get_uniform_location(program, "u_view_proj");
        let loc_cull_total     = gl.get_uniform_location(cull_program, "u_total_draws");
        let loc_frustum_planes = gl.get_uniform_location(cull_program, "u_frustum_planes[0]");

        let c = || gl.create_query().expect("failed to create query");
        let stat_queries = [
            [c(), c(), c(), c()],
            [c(), c(), c(), c()],
        ];
        let query_frame = 0usize;

        Self {
            program,
            cull_program,
            vao,
            arena,
            instance_buf,
            culled_instance_buf,
            culled_indirect_buf,
            bounding_buf,
            batch_index_buf,
            batch_base_buf,
            draw_count_buf,
            material_buf,
            registry,
            loc_view_proj,
            loc_cull_total,
            loc_frustum_planes,
            mdi_count,
            stat_queries,
            query_frame,
        }
    }

    // ── Mesh ─────────────────────────────────────────────────────────────────

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

    // ── Material management ───────────────────────────────────────────────────

    /// Push a material into the **static zone** (pre-seal only).
    /// Returns the material index. Panics in debug if called after sealing.
    pub unsafe fn add_static_material(&mut self, gl: &Context, mat: GpuMaterial) -> u32 {
        let idx = self.registry.add_static(mat);
        self.upload_material_to_gpu(gl, idx, mat);
        idx
    }

    /// Freeze the static zone. All indices below the returned watermark are
    /// now immutable. Must be called before [`add_override_material`].
    pub fn seal_static_materials(&mut self) -> usize {
        self.registry.seal()
    }

    /// Push a new mutable override material (post-seal only).
    /// Returns the material index (always ≥ the static watermark).
    pub unsafe fn add_override_material(&mut self, gl: &Context, mat: GpuMaterial) -> u32 {
        let idx = self.registry.add_override(mat);
        self.upload_material_to_gpu(gl, idx, mat);
        idx
    }

    /// Overwrite an existing override material in-place.
    /// Panics in debug if `idx` falls in the static zone.
    pub unsafe fn update_override_material(&mut self, gl: &Context, idx: u32, mat: GpuMaterial) {
        let mat = self.registry.update_override(idx, mat);
        self.material_buf.upload_subrange(gl, &[mat], idx as usize);
    }

    /// CPU-side read-back of any material (static or override).
    pub fn material(&self, idx: u32) -> Option<&GpuMaterial> {
        self.registry.get(idx)
    }

    /// Returns the watermark index separating the static and override zones,
    /// or `None` if [`seal_static_materials`] has not been called yet.
    pub fn static_material_end(&self) -> Option<usize> {
        self.registry.static_end()
    }

    /// Returns `true` if `idx` falls in the mutable override zone.
    pub fn is_override_material(&self, idx: u32) -> bool {
        self.registry.is_override(idx)
    }

    /// Grow the material GPU buffer if needed, then upload a single slot.
    unsafe fn upload_material_to_gpu(&mut self, gl: &Context, idx: u32, mat: GpuMaterial) {
        if self.registry.len() > self.material_buf.capacity {
            self.material_buf.grow(gl, self.material_buf.capacity * 2, BufferUsage::StaticDraw);
        }
        self.material_buf.upload_subrange(gl, &[mat], idx as usize);
    }

    // ── Render ───────────────────────────────────────────────────────────────

    /// Draw every `(Transform, Mesh)` entity in `world` using a GPU-driven culling
    /// pipeline: a compute shader frustum-tests each entity's bounding sphere and
    /// compacts surviving instances into batched draw commands, consumed by a single
    /// `glMultiDrawElementsIndirectCount` call.
    ///
    /// Returns [`PipelineStats`] with CPU-side entity/batch counts for this frame and
    /// GPU-measured rasterization statistics from the previous frame (double-buffered
    /// queries avoid a GPU stall).
    pub unsafe fn render(
        &mut self,
        gl: &Context,
        world: &hecs::World,
        view_proj: Mat4,
        elapsed_secs: f32,
        width: u32,
        height: u32,
    ) -> PipelineStats {
        gl.viewport(0, 0, width as i32, height as i32);
        gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

        // ── Pipeline statistics queries (double-buffered to avoid GPU stalls) ─
        let write_set = self.query_frame % 2;
        let read_set  = 1 - write_set;
        gl.begin_query(glow::TIME_ELAPSED,               self.stat_queries[write_set][QI_TIME]);
        gl.begin_query(glow::PRIMITIVES_SUBMITTED,        self.stat_queries[write_set][QI_PRIMS]);
        gl.begin_query(glow::VERTEX_SHADER_INVOCATIONS,   self.stat_queries[write_set][QI_VERT]);
        gl.begin_query(glow::FRAGMENT_SHADER_INVOCATIONS, self.stat_queries[write_set][QI_FRAG]);

        // ── Group entities by mesh to form instance batches ───────────────────
        let mut batch_map: std::collections::HashMap<MeshKey, u32> =
            std::collections::HashMap::new();
        let mut batch_meshes: Vec<MeshAlloc> = Vec::new();

        struct Entry {
            batch_id: u32,
            instance: InstanceData,
            sphere:   BoundingSphere,
        }

        let mut entries: Vec<Entry> = Vec::new();

        for (_entity, (xf, mesh, mat)) in
            world.query::<(&Transform, &Mesh, Option<&Material>)>().iter()
        {
            let key = MeshKey::from(mesh.0);
            let batch_id = if let Some(&id) = batch_map.get(&key) {
                id
            } else {
                let id = batch_meshes.len() as u32;
                batch_meshes.push(mesh.0);
                batch_map.insert(key, id);
                id
            };

            let model          = xf.model_matrix(elapsed_secs);
            let normal         = Mat4::from_mat3(Mat3::from_mat4(model.inverse().transpose()));
            let material_index = mat.map(|m| m.0).unwrap_or(0);

            entries.push(Entry {
                batch_id,
                instance: InstanceData {
                    model:  model.to_cols_array(),
                    normal: normal.to_cols_array(),
                    material_index,
                    _pad: [0; 3],
                },
                sphere: mesh.0.bounding_sphere,
            });
        }

        let n = entries.len();
        let m = batch_meshes.len();

        if n == 0 {
            gl.end_query(glow::FRAGMENT_SHADER_INVOCATIONS);
            gl.end_query(glow::VERTEX_SHADER_INVOCATIONS);
            gl.end_query(glow::PRIMITIVES_SUBMITTED);
            gl.end_query(glow::TIME_ELAPSED);
            self.query_frame = self.query_frame.wrapping_add(1);
            return PipelineStats::default();
        }

        // ── Build flat CPU arrays ─────────────────────────────────────────────
        let instances:     Vec<InstanceData>   = entries.iter().map(|e| e.instance).collect();
        let bounds:        Vec<BoundingSphere> = entries.iter().map(|e| e.sphere).collect();
        let batch_indices: Vec<u32>            = entries.iter().map(|e| e.batch_id).collect();

        // Prefix-sum of per-batch entity counts gives each batch's starting slot
        // in the culled instance buffer.
        let mut batch_sizes = vec![0u32; m];
        for e in &entries { batch_sizes[e.batch_id as usize] += 1; }
        let mut batch_base = vec![0u32; m];
        for b in 1..m { batch_base[b] = batch_base[b - 1] + batch_sizes[b - 1]; }

        // M draw commands: vertex/index info pre-filled, instance_count=0
        // (cull compute atomicAdd's to instance_count for each surviving entity).
        let commands: Vec<DrawCommand> = batch_meshes
            .iter()
            .enumerate()
            .map(|(b, mesh)| mesh.draw_command_instanced(0, batch_base[b]))
            .collect();

        // ── Grow per-frame buffers if needed ──────────────────────────────────
        if n > self.instance_buf.capacity {
            let cap = (self.instance_buf.capacity * 2).max(n);
            self.instance_buf.grow(gl, cap, BufferUsage::DynamicDraw);
            self.culled_instance_buf.grow(gl, cap, BufferUsage::DynamicDraw);
            self.bounding_buf.grow(gl, cap, BufferUsage::DynamicDraw);
            self.batch_index_buf.grow(gl, cap, BufferUsage::DynamicDraw);
        }
        if m > self.culled_indirect_buf.capacity {
            let cap = (self.culled_indirect_buf.capacity * 2).max(m);
            self.culled_indirect_buf.grow(gl, cap, BufferUsage::DynamicDraw);
            self.batch_base_buf.grow(gl, cap, BufferUsage::DynamicDraw);
        }

        self.instance_buf.upload_subrange(gl, &instances, 0);
        self.bounding_buf.upload_subrange(gl, &bounds, 0);
        self.batch_index_buf.upload_subrange(gl, &batch_indices, 0);
        self.culled_indirect_buf.upload_subrange(gl, &commands, 0);
        self.batch_base_buf.upload_subrange(gl, &batch_base, 0);

        // ── Write M to the draw-count parameter buffer ────────────────────────
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(self.draw_count_buf));
        gl.buffer_sub_data_u8_slice(
            glow::SHADER_STORAGE_BUFFER,
            0,
            bytemuck::bytes_of(&(m as u32)),
        );
        gl.bind_buffer(glow::SHADER_STORAGE_BUFFER, None);

        // ── Culling compute pass ──────────────────────────────────────────────
        gl.use_program(Some(self.cull_program));

        // binding 0: source instances (read)
        self.instance_buf.bind_as_ssbo(gl, 0);
        // binding 1: bounding spheres (read)
        self.bounding_buf.bind_as_ssbo(gl, 1);
        // binding 2: per-entity batch index (read)
        self.batch_index_buf.bind_as_ssbo(gl, 2);
        // binding 3: culled draw commands (instance_count atomicAdd'd by compute)
        self.culled_indirect_buf.bind_as_ssbo(gl, 3);
        // binding 4: culled instances (write)
        self.culled_instance_buf.bind_as_ssbo(gl, 4);
        // binding 5: per-batch base offset into culled_instance_buf (read)
        self.batch_base_buf.bind_as_ssbo(gl, 5);

        gl.uniform_1_u32(self.loc_cull_total.as_ref(), n as u32);
        let planes = extract_frustum_planes(&view_proj);
        gl.uniform_4_f32_slice(
            self.loc_frustum_planes.as_ref(),
            bytemuck::cast_slice(&planes),
        );

        let groups = (n as u32 + 63) / 64;
        gl.dispatch_compute(groups, 1, 1);

        // Wait for SSBO writes and the indirect command buffer to be visible.
        gl.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT | glow::COMMAND_BARRIER_BIT);

        // ── Main render pass ──────────────────────────────────────────────────
        gl.use_program(Some(self.program));
        gl.bind_vertex_array(Some(self.vao));

        self.arena.bind(gl, 0);                       // vertex SSBO at binding 0, IBO
        self.culled_instance_buf.bind_as_ssbo(gl, 1); // compacted visible instances
        self.material_buf.bind_as_ssbo(gl, 2);

        gl.uniform_matrix_4_f32_slice(
            self.loc_view_proj.as_ref(),
            false,
            &view_proj.to_cols_array(),
        );

        self.culled_indirect_buf.bind_as_indirect(gl);
        gl.bind_buffer(PARAMETER_BUFFER, Some(self.draw_count_buf));

        (self.mdi_count)(
            glow::TRIANGLES,
            glow::UNSIGNED_INT,
            std::ptr::null(),
            0,          // byte offset into GL_PARAMETER_BUFFER for the draw count
            m as i32,   // upper bound (= M; all batches are always emitted)
            0,
        );

        GlBuffer::<DrawCommand>::unbind_indirect(gl);
        gl.bind_buffer(PARAMETER_BUFFER, None);
        gl.bind_vertex_array(None);
        gl.use_program(None);

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

}

// ── Free functions ────────────────────────────────────────────────────────────

/// Extract the 6 world-space frustum planes from a combined view-projection matrix
/// using the Gribb-Hartmann method. Each plane is (nx, ny, nz, d); a world-space
/// point p is inside if `dot(plane.xyz, p) + plane.w >= 0`.
fn extract_frustum_planes(vp: &Mat4) -> [[f32; 4]; 6] {
    // vp is column-major: vp[col][row].
    let m = vp.to_cols_array_2d();
    let row = |i: usize| -> [f32; 4] { [m[0][i], m[1][i], m[2][i], m[3][i]] };

    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);

    let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
        [a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]]
    };
    let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
        [a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]]
    };

    [
        add(r3, r0), // left
        sub(r3, r0), // right
        add(r3, r1), // bottom
        sub(r3, r1), // top
        add(r3, r2), // near
        sub(r3, r2), // far
    ]
}

impl Renderer {
    pub unsafe fn cleanup(&self, gl: &Context) {
        self.arena.cleanup(gl);
        self.instance_buf.cleanup(gl);
        self.culled_instance_buf.cleanup(gl);
        self.culled_indirect_buf.cleanup(gl);
        self.bounding_buf.cleanup(gl);
        self.batch_index_buf.cleanup(gl);
        self.batch_base_buf.cleanup(gl);
        gl.delete_buffer(self.draw_count_buf);
        self.material_buf.cleanup(gl);
        for set in &self.stat_queries {
            for &q in set {
                gl.delete_query(q);
            }
        }
        gl.delete_vertex_array(self.vao);
        gl.delete_program(self.cull_program);
        gl.delete_program(self.program);
    }
}
