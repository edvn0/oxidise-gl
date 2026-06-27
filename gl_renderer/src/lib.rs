//! A minimal but modern OpenGL 4.3 renderer: vertex pulling from a shared GPU
//! arena, multi-draw-indirect, and an ECS-driven scene.
//!
//! The crate is split so a host application (the demo example or the editor)
//! owns the `glow::Context` and the `hecs::World`, and drives [`Renderer`] each
//! frame. This keeps the GL context shareable (e.g. with an imgui renderer).

pub mod arena;
pub mod components;
pub mod framebuffer;
pub mod geometry;
pub mod gl_buffer;
pub mod gl_init;
pub mod renderer;
pub mod shaders;

pub use arena::{BoundingSphere, DrawCommand, GpuArena, GpuMaterial, InstanceData, MaterialRegistry, MeshAlloc, MeshKey, PackedVertex};
pub use components::{Material, Mesh, Name, Script, Transform};
pub use framebuffer::OffscreenTarget;
pub use geometry::{generate_cube, generate_sphere};
pub use gl_init::{create_gl_window, GlWindow};
pub use renderer::{MultiDrawElementsIndirectCountFn, MultiDrawElementsIndirectFn, PipelineStats, Renderer};
