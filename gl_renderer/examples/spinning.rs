//! The original standalone demo: three objects (cube, sphere, offset cube) drawn
//! from a shared arena via multi-draw-indirect, animated through a `hecs::World`.
//! Renders straight to the window's default framebuffer.
//!
//! Run with: `cargo run -p gl_renderer --example spinning`

use gl_renderer::{
    create_gl_window, generate_cube, generate_sphere, Mesh, Renderer, Transform,
};

use glam::{Mat4, Vec3};
use glutin::prelude::*;
use std::num::NonZeroU32;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let gl_window = create_gl_window(
        &event_loop,
        "GL 4.3 Vertex Pulling – Shared Arena",
        1280,
        720,
    );
    let gl_renderer::GlWindow { window, surface, context, gl, mdi, .. } = gl_window;

    let mut renderer = unsafe { Renderer::new(&gl, mdi) };

    // Upload the two meshes into the shared arena.
    let (cube_verts, cube_idx) = generate_cube();
    let cube = unsafe { renderer.upload_mesh(&gl, &cube_verts, &cube_idx) };
    let (sphere_verts, sphere_idx) = generate_sphere(24, 32);
    let sphere = unsafe { renderer.upload_mesh(&gl, &sphere_verts, &sphere_idx) };

    // Build the scene.
    let mut world = hecs::World::new();
    world.spawn((
        Transform {
            position:       Vec3::ZERO,
            rotation_axis:  Vec3::new(1.0, 1.0, 0.0).normalize(),
            rotation_speed: 0.7,
            scale:          1.0,
        },
        Mesh(cube),
    ));
    world.spawn((
        Transform {
            position:       Vec3::new(2.5, 0.0, 0.0),
            rotation_axis:  Vec3::Y,
            rotation_speed: 1.4,
            scale:          0.6,
        },
        Mesh(sphere),
    ));
    world.spawn((
        Transform {
            position:       Vec3::new(-2.2, 0.8, -0.5),
            rotation_axis:  Vec3::new(0.0, 1.0, 0.5).normalize(),
            rotation_speed: 0.4,
            scale:          0.7,
        },
        Mesh(cube),
    ));

    let start = Instant::now();
    let mut current_size = window.inner_size();

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);
            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    unsafe { renderer.cleanup(&gl) };
                    elwt.exit();
                }
                Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                    current_size = size;
                    if size.width > 0 && size.height > 0 {
                        surface.resize(
                            &context,
                            NonZeroU32::new(size.width).unwrap(),
                            NonZeroU32::new(size.height).unwrap(),
                        );
                    }
                }
                Event::AboutToWait => {
                    let (w, h) = (current_size.width, current_size.height);
                    let aspect = w as f32 / h.max(1) as f32;
                    let proj = Mat4::perspective_rh_gl(45_f32.to_radians(), aspect, 0.1, 100.0);
                    let view = Mat4::look_at_rh(
                        Vec3::new(4.5, 3.0, 5.5),
                        Vec3::ZERO,
                        Vec3::Y,
                    );
                    let elapsed = start.elapsed().as_secs_f32();
                    unsafe { renderer.render(&gl, &world, proj * view, elapsed, w, h) };
                    surface.swap_buffers(&context).expect("swap_buffers failed");
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .expect("event loop error");
}
