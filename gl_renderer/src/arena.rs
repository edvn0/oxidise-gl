use crate::gl_buffer::{BufferUsage, GlBuffer};
use glow::Context;

/// Model-space bounding sphere stored in the GPU bounding-sphere SSBO.
/// Maps to a `vec4` in GLSL: xyz = center, w = radius.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoundingSphere {
    pub center: [f32; 3],
    pub radius: f32,
}

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
    /// Model-space bounding sphere, computed at upload time for GPU culling.
    pub bounding_sphere: BoundingSphere,
}

/// Identifies a unique mesh in the shared arena by its allocation offsets.
/// Used as the key when grouping entities into instanced draw batches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshKey {
    pub vertex_offset: u32,
    pub index_offset:  u32,
}

impl From<MeshAlloc> for MeshKey {
    fn from(a: MeshAlloc) -> Self {
        Self { vertex_offset: a.vertex_offset, index_offset: a.index_offset }
    }
}

impl MeshAlloc {
    /// Build a GL indirect draw command for `instance_count` instances starting
    /// at `base_instance` in the flat instance SSBO (addressed as
    /// `gl_BaseInstance + gl_InstanceID` in the vertex shader).
    pub fn draw_command_instanced(&self, instance_count: u32, base_instance: u32) -> DrawCommand {
        DrawCommand {
            count:          self.index_count,
            instance_count,
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
    pub instance_count: u32, // number of instances sharing this mesh
    pub first_index:    u32, // first index in the shared IBO (in indices, not bytes)
    pub base_vertex:    u32, // base vertex offset into the shared vertex SSBO
    pub base_instance:  u32, // first instance in the flat instance SSBO for this batch
}

/// Per-object data uploaded to the instance SSBO, addressed in the vertex
/// shader as `instances[gl_BaseInstance + gl_InstanceID]`.
/// std430 size: 64 + 64 + 4 + 12 = 144 bytes (padded to 16-byte struct alignment).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceData {
    pub model:          [f32; 16], // column-major model matrix
    pub normal:         [f32; 16], // inverse-transpose of model's 3x3, widened to mat4
    pub material_index: u32,       // index into the material SSBO; 0 = default material
    pub _pad:           [u32; 3],
}

/// A material slot in the GPU material SSBO.
/// std430 layout: vec4 (16 B) + float + float + float + float = 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMaterial {
    pub base_color: [f32; 4], // linear RGBA albedo
    pub roughness:  f32,
    pub metallic:   f32,
    pub _pad:       [f32; 2],
}

impl GpuMaterial {
    pub const DEFAULT: Self = Self {
        base_color: [0.85, 0.82, 0.78, 1.0],
        roughness:  0.7,
        metallic:   0.0,
        _pad:       [0.0; 2],
    };
}

// ── MaterialRegistry ──────────────────────────────────────────────────────────

/// CPU-side material table with a two-zone layout: a static zone (immutable
/// after [`seal`]) and an override zone (editor-mutable).
///
/// All index arithmetic and zone enforcement live here; GPU uploads are the
/// caller's responsibility. This separation makes the registry fully testable
/// without a GL context.
///
/// Layout: `[0 .. static_end)` static, `[static_end .. len)` overrides.
/// Index 0 is always [`GpuMaterial::DEFAULT`], pre-populated in [`new`].
pub struct MaterialRegistry {
    materials:           Vec<GpuMaterial>,
    static_material_end: Option<usize>,
}

impl MaterialRegistry {
    /// Create a registry with the default material pre-loaded at index 0.
    /// The static zone is open until [`seal`] is called.
    pub fn new() -> Self {
        Self {
            materials:           vec![GpuMaterial::DEFAULT],
            static_material_end: None,
        }
    }

    /// Append a material to the static zone and return its index.
    /// Panics in debug if the static zone is already sealed.
    pub fn add_static(&mut self, mat: GpuMaterial) -> u32 {
        debug_assert!(
            self.static_material_end.is_none(),
            "add_static called after seal_static_materials"
        );
        let idx = self.materials.len();
        self.materials.push(mat);
        idx as u32
    }

    /// Close the static zone. Returns the watermark index (= first override slot).
    /// Panics in debug if called more than once.
    pub fn seal(&mut self) -> usize {
        debug_assert!(
            self.static_material_end.is_none(),
            "seal called more than once"
        );
        let end = self.materials.len();
        self.static_material_end = Some(end);
        end
    }

    /// Append a material to the override zone and return its index.
    /// Panics in debug if the static zone has not been sealed yet.
    pub fn add_override(&mut self, mat: GpuMaterial) -> u32 {
        debug_assert!(
            self.static_material_end.is_some(),
            "add_override called before seal_static_materials"
        );
        let idx = self.materials.len();
        self.materials.push(mat);
        idx as u32
    }

    /// Update an override material in place. Returns the updated value so the
    /// caller can upload it to the GPU.
    /// Panics in debug if `idx` falls in the static zone.
    pub fn update_override(&mut self, idx: u32, mat: GpuMaterial) -> GpuMaterial {
        let end = self
            .static_material_end
            .expect("update_override called before sealing");
        debug_assert!(
            idx as usize >= end,
            "attempted to mutate static material at index {idx}"
        );
        self.materials[idx as usize] = mat;
        mat
    }

    /// Returns the material at `idx`, or `None` if out of range.
    pub fn get(&self, idx: u32) -> Option<&GpuMaterial> {
        self.materials.get(idx as usize)
    }

    /// Returns `true` if `idx` is in the mutable override zone.
    pub fn is_override(&self, idx: u32) -> bool {
        self.static_material_end
            .map_or(false, |end| idx as usize >= end)
    }

    /// Returns the watermark, or `None` if not yet sealed.
    pub fn static_end(&self) -> Option<usize> {
        self.static_material_end
    }

    /// Total number of materials (static + override).
    pub fn len(&self) -> usize {
        self.materials.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn red() -> GpuMaterial {
        GpuMaterial { base_color: [1.0, 0.0, 0.0, 1.0], roughness: 0.5, metallic: 0.0, _pad: [0.0; 2] }
    }

    fn blue() -> GpuMaterial {
        GpuMaterial { base_color: [0.0, 0.0, 1.0, 1.0], roughness: 0.2, metallic: 1.0, _pad: [0.0; 2] }
    }

    // ── Initial state ─────────────────────────────────────────────────────────

    #[test]
    fn given_new_registry_when_get_zero_then_returns_default() {
        let reg = MaterialRegistry::new();
        let mat = reg.get(0).expect("index 0 should exist");
        assert_eq!(mat.base_color, GpuMaterial::DEFAULT.base_color);
    }

    #[test]
    fn given_new_registry_then_len_is_one() {
        assert_eq!(MaterialRegistry::new().len(), 1);
    }

    #[test]
    fn given_new_registry_then_static_end_is_none() {
        assert!(MaterialRegistry::new().static_end().is_none());
    }

    #[test]
    fn given_new_registry_when_is_override_zero_then_false() {
        assert!(!MaterialRegistry::new().is_override(0));
    }

    #[test]
    fn given_new_registry_when_get_out_of_range_then_none() {
        assert!(MaterialRegistry::new().get(99).is_none());
    }

    // ── Static zone ───────────────────────────────────────────────────────────

    #[test]
    fn given_new_registry_when_add_static_then_returns_index_one() {
        let mut reg = MaterialRegistry::new();
        assert_eq!(reg.add_static(red()), 1);
    }

    #[test]
    fn given_two_statics_added_when_add_third_then_returns_index_three() {
        let mut reg = MaterialRegistry::new();
        reg.add_static(red());
        reg.add_static(blue());
        assert_eq!(reg.add_static(red()), 3);
    }

    #[test]
    fn given_static_added_when_get_then_returns_correct_material() {
        let mut reg = MaterialRegistry::new();
        let idx = reg.add_static(red());
        let mat = reg.get(idx).expect("should exist");
        assert_eq!(mat.base_color, red().base_color);
        assert_eq!(mat.roughness, red().roughness);
    }

    // ── Seal ──────────────────────────────────────────────────────────────────

    #[test]
    fn given_new_registry_when_seal_then_end_is_one() {
        let mut reg = MaterialRegistry::new();
        assert_eq!(reg.seal(), 1);
    }

    #[test]
    fn given_one_static_added_when_seal_then_end_is_two() {
        let mut reg = MaterialRegistry::new();
        reg.add_static(red());
        assert_eq!(reg.seal(), 2);
    }

    #[test]
    fn given_sealed_registry_when_static_end_then_some() {
        let mut reg = MaterialRegistry::new();
        reg.add_static(red());
        reg.seal();
        assert_eq!(reg.static_end(), Some(2));
    }

    // ── Override zone ─────────────────────────────────────────────────────────

    #[test]
    fn given_sealed_registry_when_add_override_then_index_after_watermark() {
        let mut reg = MaterialRegistry::new();
        reg.add_static(red());
        let end = reg.seal();
        let idx = reg.add_override(blue());
        assert_eq!(idx as usize, end);
    }

    #[test]
    fn given_override_added_when_get_then_returns_correct_material() {
        let mut reg = MaterialRegistry::new();
        reg.seal();
        let idx = reg.add_override(blue());
        let mat = reg.get(idx).expect("should exist");
        assert_eq!(mat.base_color, blue().base_color);
        assert_eq!(mat.metallic, blue().metallic);
    }

    #[test]
    fn given_override_index_when_is_override_then_true() {
        let mut reg = MaterialRegistry::new();
        reg.seal();
        let idx = reg.add_override(blue());
        assert!(reg.is_override(idx));
    }

    #[test]
    fn given_static_index_when_is_override_then_false() {
        let mut reg = MaterialRegistry::new();
        let idx = reg.add_static(red());
        reg.seal();
        assert!(!reg.is_override(idx));
    }

    #[test]
    fn given_default_index_when_is_override_after_seal_then_false() {
        let mut reg = MaterialRegistry::new();
        reg.seal();
        assert!(!reg.is_override(0));
    }

    // ── Update override ───────────────────────────────────────────────────────

    #[test]
    fn given_override_when_update_then_get_returns_new_value() {
        let mut reg = MaterialRegistry::new();
        reg.seal();
        let idx = reg.add_override(red());

        reg.update_override(idx, blue());

        let mat = reg.get(idx).expect("should exist");
        assert_eq!(mat.base_color, blue().base_color);
    }

    #[test]
    fn given_override_when_update_then_returns_new_value() {
        let mut reg = MaterialRegistry::new();
        reg.seal();
        let idx = reg.add_override(red());
        let returned = reg.update_override(idx, blue());
        assert_eq!(returned.base_color, blue().base_color);
    }

    // ── Panic cases (debug builds only) ───────────────────────────────────────

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn given_sealed_registry_when_add_static_then_panics() {
        let mut reg = MaterialRegistry::new();
        reg.seal();
        reg.add_static(red());
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn given_unsealed_registry_when_add_override_then_panics() {
        let mut reg = MaterialRegistry::new();
        reg.add_override(blue());
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn given_sealed_twice_when_seal_then_panics() {
        let mut reg = MaterialRegistry::new();
        reg.seal();
        reg.seal();
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn given_static_index_when_update_override_then_panics() {
        let mut reg = MaterialRegistry::new();
        let static_idx = reg.add_static(red());
        reg.seal();
        reg.update_override(static_idx, blue());
    }
}

fn unpack_half2(packed: u32) -> [f32; 2] {
    let lo = half::f16::from_bits((packed & 0xFFFF) as u16).to_f32();
    let hi = half::f16::from_bits((packed >> 16) as u16).to_f32();
    [lo, hi]
}

fn compute_bounding_sphere(vertices: &[PackedVertex]) -> BoundingSphere {
    if vertices.is_empty() {
        return BoundingSphere { center: [0.0; 3], radius: 0.0 };
    }
    let mut cx = 0.0f32;
    let mut cy = 0.0f32;
    let mut cz = 0.0f32;
    for v in vertices {
        let [x, y] = unpack_half2(v.pos_xy);
        let [z, _] = unpack_half2(v.pos_z);
        cx += x; cy += y; cz += z;
    }
    let n = vertices.len() as f32;
    cx /= n; cy /= n; cz /= n;
    let mut radius = 0.0f32;
    for v in vertices {
        let [x, y] = unpack_half2(v.pos_xy);
        let [z, _] = unpack_half2(v.pos_z);
        let dx = x - cx;
        let dy = y - cy;
        let dz = z - cz;
        radius = radius.max((dx * dx + dy * dy + dz * dz).sqrt());
    }
    BoundingSphere { center: [cx, cy, cz], radius }
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
            vertex_offset:   v_offset,
            vertex_count:    vertices.len() as u32,
            index_offset:    i_offset,
            index_count:     indices.len()  as u32,
            bounding_sphere: compute_bounding_sphere(vertices),
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
