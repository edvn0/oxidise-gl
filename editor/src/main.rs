//! A small Unity/Unreal-style editor for the `gl_renderer` engine.
//!
//! Layout: a central "Viewport" panel renders the live 3D scene (the engine
//! draws into an offscreen framebuffer that is shown as an imgui image), with
//! dockable Hierarchy / Inspector / Stats panels around it. The whole window is
//! an imgui dockspace, so panels can be rearranged and the layout persists in
//! `imgui.ini`.

use gl_renderer::{
    create_gl_window, generate_cube, generate_sphere, GlWindow, Mesh, MeshAlloc, Name,
    OffscreenTarget, Renderer, Transform,
};

use glam::{Mat4, Vec3};
use glow::HasContext;
use glutin::prelude::*;
use hecs::{Entity, World};
use imgui::{Condition, ConfigFlags, Image, TextureId};
use imgui_glow_renderer::AutoRenderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

/// Camera eye position (fixed for v1).
const EYE: Vec3 = Vec3::new(4.5, 3.0, 5.5);

fn spawn_cube(world: &mut World, mesh: MeshAlloc, name: &str) -> Entity {
    world.spawn((
        Transform {
            position:       Vec3::ZERO,
            rotation_axis:  Vec3::Y,
            rotation_speed: 0.5,
            scale:          1.0,
        },
        Mesh(mesh),
        Name(name.to_string()),
    ))
}

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let GlWindow { window, surface, context, gl, mdi, .. } =
        create_gl_window(&event_loop, "gl_renderer editor", 1600, 900);

    // ── Engine renderer + scene ─────────────────────────────────────────────
    let mut renderer = unsafe { Renderer::new(&gl, mdi) };
    let (cv, ci) = generate_cube();
    let cube = unsafe { renderer.upload_mesh(&gl, &cv, &ci) };
    let (sv, si) = generate_sphere(24, 32);
    let sphere = unsafe { renderer.upload_mesh(&gl, &sv, &si) };

    let mut world = World::new();
    world.spawn((
        Transform { position: Vec3::ZERO, rotation_axis: Vec3::new(1.0, 1.0, 0.0).normalize(),
                    rotation_speed: 0.7, scale: 1.0 },
        Mesh(cube), Name("Cube".into()),
    ));
    world.spawn((
        Transform { position: Vec3::new(2.5, 0.0, 0.0), rotation_axis: Vec3::Y,
                    rotation_speed: 1.4, scale: 0.6 },
        Mesh(sphere), Name("Sphere".into()),
    ));
    world.spawn((
        Transform { position: Vec3::new(-2.2, 0.8, -0.5), rotation_axis: Vec3::new(0.0, 1.0, 0.5).normalize(),
                    rotation_speed: 0.4, scale: 0.7 },
        Mesh(cube), Name("Cube.001".into()),
    ));

    // ── imgui setup ─────────────────────────────────────────────────────────
    let mut imgui = imgui::Context::create();
    imgui.set_ini_filename(Some(PathBuf::from("imgui.ini")));
    imgui.io_mut().config_flags |= ConfigFlags::DOCKING_ENABLE;

    let mut platform = WinitPlatform::init(&mut imgui);
    platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default);

    // AutoRenderer takes ownership of the glow context; share it back via an Rc
    // clone so the engine renderer and the offscreen target use the same context.
    let mut ig_renderer =
        AutoRenderer::initialize(gl, &mut imgui).expect("failed to init imgui glow renderer");
    let gl = ig_renderer.gl_context().clone();

    let mut fbo = unsafe { OffscreenTarget::new(&gl, 1280, 720) };

    // ── Editor state ──────────────────────────────────────────────────────────
    let mut selected: Option<Entity> = None;
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut draw_count = 0usize;

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);
            platform.handle_event(imgui.io_mut(), &window, &event);

            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    unsafe { renderer.cleanup(&gl); fbo.cleanup(&gl); }
                    elwt.exit();
                }
                Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                    if size.width > 0 && size.height > 0 {
                        surface.resize(
                            &context,
                            NonZeroU32::new(size.width).unwrap(),
                            NonZeroU32::new(size.height).unwrap(),
                        );
                    }
                }
                Event::AboutToWait => {
                    let now = Instant::now();
                    imgui.io_mut().update_delta_time(now - last_frame);
                    last_frame = now;
                    let elapsed = start.elapsed().as_secs_f32();

                    platform
                        .prepare_frame(imgui.io_mut(), &window)
                        .expect("prepare_frame failed");
                    let ui = imgui.new_frame();

                    // Host dockspace filling the main viewport.
                    ui.dockspace_over_main_viewport();

                    // ── Hierarchy ────────────────────────────────────────────
                    ui.window("Hierarchy")
                        .position([0.0, 24.0], Condition::FirstUseEver)
                        .size([240.0, 520.0], Condition::FirstUseEver)
                        .build(|| {
                            if ui.button("Spawn Cube") {
                                selected = Some(spawn_cube(&mut world, cube, "Cube"));
                            }
                            ui.same_line();
                            if ui.button("Spawn Sphere") {
                                selected = Some(spawn_cube(&mut world, sphere, "Sphere"));
                            }
                            ui.separator();

                            let entities: Vec<Entity> =
                                world.query::<&Mesh>().iter().map(|(e, _)| e).collect();
                            for e in entities {
                                let label = world
                                    .get::<&Name>(e)
                                    .map(|n| n.0.clone())
                                    .unwrap_or_else(|_| format!("Entity {}", e.id()));
                                if ui.selectable_config(format!("{label}##{}", e.id()))
                                    .selected(selected == Some(e))
                                    .build()
                                {
                                    selected = Some(e);
                                }
                            }

                            if let Some(sel) = selected {
                                ui.separator();
                                if ui.button("Delete Selected") {
                                    let _ = world.despawn(sel);
                                    selected = None;
                                }
                            }
                        });

                    // ── Inspector ────────────────────────────────────────────
                    ui.window("Inspector")
                        .position([1360.0, 24.0], Condition::FirstUseEver)
                        .size([240.0, 520.0], Condition::FirstUseEver)
                        .build(|| match selected {
                            Some(sel) => {
                                if let Ok(mut xf) = world.get::<&mut Transform>(sel) {
                                    let mut pos = xf.position.to_array();
                                    if ui.input_float3("Position", &mut pos).build() {
                                        xf.position = Vec3::from_array(pos);
                                    }
                                    let mut axis = xf.rotation_axis.to_array();
                                    if ui.input_float3("Rot axis", &mut axis).build() {
                                        let v = Vec3::from_array(axis);
                                        xf.rotation_axis =
                                            if v.length_squared() > 1e-6 { v.normalize() } else { Vec3::Y };
                                    }
                                    ui.slider("Rot speed", -3.0, 3.0, &mut xf.rotation_speed);
                                    ui.slider("Scale", 0.1, 3.0, &mut xf.scale);
                                } else {
                                    ui.text("Selected entity has no Transform.");
                                }
                            }
                            None => ui.text("Select an entity in the Hierarchy."),
                        });

                    // ── Stats ────────────────────────────────────────────────
                    ui.window("Stats")
                        .position([0.0, 556.0], Condition::FirstUseEver)
                        .size([240.0, 140.0], Condition::FirstUseEver)
                        .build(|| {
                            ui.text(format!("FPS:     {:.0}", ui.io().framerate));
                            ui.text(format!("Draws:   {draw_count}"));
                            ui.text(format!("Entities:{}", world.len()));
                        });

                    // ── Viewport (renders the scene into the FBO) ─────────────
                    ui.window("Viewport")
                        .position([250.0, 24.0], Condition::FirstUseEver)
                        .size([1080.0, 680.0], Condition::FirstUseEver)
                        .build(|| {
                            let avail = ui.content_region_avail();
                            let vw = avail[0].max(1.0) as u32;
                            let vh = avail[1].max(1.0) as u32;
                            unsafe {
                                fbo.resize(&gl, vw, vh);
                                fbo.bind(&gl);
                                let aspect = vw as f32 / vh as f32;
                                let proj =
                                    Mat4::perspective_rh_gl(45_f32.to_radians(), aspect, 0.1, 100.0);
                                let view = Mat4::look_at_rh(EYE, Vec3::ZERO, Vec3::Y);
                                draw_count =
                                    renderer.render(&gl, &world, proj * view, elapsed, vw, vh);
                            }
                            // GL textures are bottom-up; flip V so the image is upright.
                            let tex = TextureId::new(fbo.color_texture().0.get() as usize);
                            Image::new(tex, [vw as f32, vh as f32])
                                .uv0([0.0, 1.0])
                                .uv1([1.0, 0.0])
                                .build(ui);
                        });

                    // ── Composite: scene FBO is done; draw imgui to the window ─
                    platform.prepare_render(ui, &window);
                    let draw_data = imgui.render();
                    unsafe {
                        gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                        let sz = window.inner_size();
                        gl.viewport(0, 0, sz.width as i32, sz.height as i32);
                        gl.clear_color(0.1, 0.1, 0.12, 1.0);
                        gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
                    }
                    // Skip empty draw data (e.g. early frames before the surface is
                    // configured): imgui-rs 0.12 stores a null cmd_lists pointer when
                    // there are zero lists, which trips slice::from_raw_parts.
                    if draw_data.draw_lists_count() > 0 {
                        ig_renderer.render(draw_data).expect("imgui render failed");
                    }

                    surface.swap_buffers(&context).expect("swap_buffers failed");
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .expect("event loop error");
}
