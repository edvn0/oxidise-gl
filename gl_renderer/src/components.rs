//! ECS components. Scene objects are entities composed of a `Transform`
//! (how it moves) and a `Mesh` (which arena allocation to draw).

use crate::arena::MeshAlloc;
use glam::{Mat4, Vec3};

/// Animated placement of an object in the world.
pub struct Transform {
    pub position:       Vec3,
    pub rotation_axis:  Vec3,
    pub rotation_speed: f32,
    pub scale:          f32,
}

impl Transform {
    /// Model matrix at `elapsed_secs` (translate · rotate · scale).
    pub fn model_matrix(&self, elapsed_secs: f32) -> Mat4 {
        let angle = elapsed_secs * self.rotation_speed;
        Mat4::from_translation(self.position)
            * Mat4::from_axis_angle(self.rotation_axis, angle)
            * Mat4::from_scale(Vec3::splat(self.scale))
    }
}

/// Tags an entity with the shared-arena allocation it draws from.
#[derive(Clone, Copy)]
pub struct Mesh(pub MeshAlloc);

/// Optional display name, shown in the editor's scene hierarchy. Entities
/// without one fall back to their `Entity` id.
#[derive(Clone, Debug)]
pub struct Name(pub String);

/// Attaches a Python script to an entity. The path is relative to the working
/// directory (e.g. `"scripts/orbit.py"`). The script host calls
/// `on_update(entity, scene, t, dt)` on this entity every frame.
#[derive(Clone, Debug)]
pub struct Script(pub String);
