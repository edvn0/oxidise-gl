use crate::gl_buffer::{BufferUsage, GlBuffer};
use glow::Context;

/// A packed vertex: position (half2), normal (half2 for XY, sign-encoded Z),
/// and UV (half2), all stored as u32 words (each u32 holds two f16 via packHalf2x16).
/// Layout: [pos_xy, pos_z_pad, norm_xy, norm_z_pad, uv_xy]
/// 5 u32s = 20 bytes per vertex.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedVertex {
    pub pos_xy: u32,   // packHalf2x16(x, y)
    pub pos_z:  u32,   // packHalf2x16(z, 0.0)
    pub norm_xy: u32,  // packHalf2x16(nx, ny)
    pub norm_z:  u32,  // packHalf2x16(nz, 0.0)
    pub uv_xy:  u32,   // packHalf2x16(u, v)
}

impl PackedVertex {
    pub fn new(pos: [f32; 3], norm: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            pos_xy:  pack_half2(pos[0],  pos[1]),
            pos_z:   pack_half2(pos[2],  0.0),
            norm_xy: pack_half2(norm[0], norm[1]),
            norm_z:  pack_half2(norm[2], 0.0),
            uv_xy:   pack_half2(uv[0],   uv[1]),
        }
    }
}

/// Mirror of `packHalf2x16`: pack two f32 values into the low/high 16 bits of a u32.
pub fn pack_half2(a: f32, b: f32) -> u32 {
    let ha = half::f16::from_f32(a).to_bits() as u32;
    let hb = half::f16::from_f32(b).to_bits() as u32;
    ha | (hb << 16)
}

/// An allocated mesh range inside the shared arena buffers.
/// Indices are *local* (start from 0); the base vertex offset
/// is applied at draw time via `glDrawElementsBaseVertex`.
#[derive(Clone, Copy, Debug)]
pub struct MeshAlloc {
    /// First vertex index inside the shared vertex SSBO.
    pub vertex_offset: u32,
    /// Number of vertices belonging to this mesh.
    pub vertex_count: u32,
    /// First index inside the shared index buffer.
    pub index_offset: u32,
    /// Number of indices.
    pub index_count: u32,
}

impl MeshAlloc {
    /// Translate this allocation into a GL indirect draw command.
    /// `base_instance` is the draw's index into the per-object instance SSBO
    /// (read in the vertex shader via `gl_DrawIDARB`).
    pub fn draw_command(&self, base_instance: u32) -> DrawCommand {
        DrawCommand {
            count:          self.index_count,
            instance_count: 1,
            first_index:    self.index_offset,
            base_vertex:    self.vertex_offset,
            base_instance,
        }
    }
}

/// Mirror of GL's `DrawElementsIndirectCommand` (20 bytes, tightly packed).
/// Consumed by `glMultiDrawElementsIndirect` out of the draw-indirect buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawCommand {
    pub count:          u32, // index count for this draw
    pub instance_count: u32, // always 1 here
    pub first_index:    u32, // first index in the shared IBO (in indices, not bytes)
    pub base_vertex:    u32, // base vertex offset into the shared vertex SSBO
    pub base_instance:  u32, // == gl_DrawIDARB lookup index into the instance SSBO
}

/// Per-object data read in the vertex shader, indexed by `gl_DrawIDARB`.
/// The normal matrix is a mat3 widened to mat4 so the struct follows the
/// simple std430 mat4 layout (mat3 carries awkward per-column padding).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model:  [f32; 16], // column-major model matrix
    pub normal: [f32; 16], // inverse-transpose of model's 3x3, widened to mat4
}

/// Shared GPU arena: one vertex SSBO and one index buffer.
/// Meshes are sub-allocated into contiguous ranges.
pub struct GpuArena {
    pub vertex_buf: GlBuffer<PackedVertex>,
    pub index_buf:  GlBuffer<u32>,

    next_vertex: u32,
    next_index:  u32,
}

impl GpuArena {
    /// Create the arena with the given initial capacities. Both buffers grow automatically.
    pub unsafe fn new(gl: &Context, initial_vertices: usize, initial_indices: usize) -> Self {
        let vertex_buf = GlBuffer::new(gl, initial_vertices, BufferUsage::DynamicDraw);
        let index_buf  = GlBuffer::new(gl, initial_indices,  BufferUsage::DynamicDraw);
        Self {
            vertex_buf,
            index_buf,
            next_vertex: 0,
            next_index: 0,
        }
    }

    unsafe fn ensure_vertex_capacity(&mut self, gl: &Context, needed: usize) {
        let required = self.next_vertex as usize + needed;
        if required > self.vertex_buf.capacity {
            let new_cap = (self.vertex_buf.capacity * 2).max(required);
            self.vertex_buf.grow(gl, new_cap, BufferUsage::DynamicDraw);
        }
    }

    unsafe fn ensure_index_capacity(&mut self, gl: &Context, needed: usize) {
        let required = self.next_index as usize + needed;
        if required > self.index_buf.capacity {
            let new_cap = (self.index_buf.capacity * 2).max(required);
            self.index_buf.grow(gl, new_cap, BufferUsage::DynamicDraw);
        }
    }

    /// Push vertices and (local, 0-based) indices into the arena.
    /// Grows the underlying GPU buffers if necessary (2× doubling strategy).
    /// Returns a `MeshAlloc` describing the allocated ranges.
    pub unsafe fn push_mesh(
        &mut self,
        gl: &Context,
        vertices: &[PackedVertex],
        indices: &[u32],
    ) -> MeshAlloc {
        self.ensure_vertex_capacity(gl, vertices.len());
        self.ensure_index_capacity(gl, indices.len());

        let v_offset = self.next_vertex;
        let i_offset = self.next_index;

        self.vertex_buf.upload_subrange(gl, vertices, v_offset as usize);
        self.index_buf.upload_subrange(gl, indices, i_offset as usize);

        self.next_vertex += vertices.len() as u32;
        self.next_index  += indices.len()  as u32;

        MeshAlloc {
            vertex_offset: v_offset,
            vertex_count:  vertices.len() as u32,
            index_offset:  i_offset,
            index_count:   indices.len()  as u32,
        }
    }

    /// Bind the vertex buffer as SSBO at `binding` and the index buffer as IBO.
    pub unsafe fn bind(&self, gl: &Context, ssbo_binding: u32) {
        self.vertex_buf.bind_as_ssbo(gl, ssbo_binding);
        self.index_buf.bind_as_ibo(gl);
    }

    pub unsafe fn cleanup(&self, gl: &Context) {
        self.vertex_buf.cleanup(gl);
        self.index_buf.cleanup(gl);
    }
}
