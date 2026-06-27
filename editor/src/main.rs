//! A small Unity/Unreal-style editor for the `gl_renderer` engine.
//!
//! Layout: a central "Viewport" panel renders the live 3D scene (the engine
//! draws into an offscreen framebuffer that is shown as an imgui image), with
//! dockable Hierarchy / Inspector / Stats panels around it. The whole window is
//! an imgui dockspace, so panels can be rearranged and the layout persists in
//! `imgui.ini`.

mod scripting;

use gl_renderer::{
    create_gl_window, generate_cube, generate_sphere, GlWindow, GpuMaterial, Material, Mesh,
    MeshAlloc, Name, OffscreenTarget, PipelineStats, Renderer, Script, Transform,
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

fn spawn_bulk(world: &mut World, cube: MeshAlloc, sphere: MeshAlloc, n: usize) {
    let base = world.query::<&Mesh>().iter().count();
    // Distribute n entities along a helix so they don't all overlap.
    let turns = (n as f32 / 50.0).max(1.0);
    for i in 0..n {
        let t = i as f32 / n as f32;
        let a = t * turns * std::f32::consts::TAU;
        let r = 1.0 + t * 8.0;
        let y = t * 6.0 - 3.0;
        world.spawn((
            Transform {
                position: Vec3::new(a.cos() * r, y, a.sin() * r),
                rotation_axis: Vec3::new(
                    (i as f32 * 0.37).sin(),
                    1.0,
                    (i as f32 * 0.71).cos(),
                ).normalize(),
                rotation_speed: 0.05 + (i % 13) as f32 * 0.04,
                scale: 0.08,
            },
            Mesh(if i % 3 == 0 { cube } else { sphere }),
            Name(format!("bulk_{}", base + i)),
        ));
    }
}

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

fn fmt_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn main() {
    // Register the built-in `gl_renderer` Python module before Python initialises.
    // append_to_inittab! requires a bare ident, not a path.
    use scripting::py_module;
    pyo3::append_to_inittab!(py_module);

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let GlWindow { window, surface, context, gl, mdi_count, .. } =
        create_gl_window(&event_loop, "gl_renderer editor", 1600, 900);

    // ── Engine renderer + scene ─────────────────────────────────────────────
    let mut renderer = unsafe { Renderer::new(&gl, mdi_count) };
    let (cv, ci) = generate_cube();
    let cube = unsafe { renderer.upload_mesh(&gl, &cv, &ci) };
    let (sv, si) = generate_sphere(24, 32);
    let sphere = unsafe { renderer.upload_mesh(&gl, &sv, &si) };
    // Index 0 (DEFAULT) is the only static material; seal so the editor can add overrides.
    renderer.seal_static_materials();

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

    // ── Script host ──────────────────────────────────────────────────────────
    let mut script_host = scripting::ScriptHost::new();
    script_host.register_mesh("cube", cube);
    script_host.register_mesh("sphere", sphere);

    // ── Editor state ──────────────────────────────────────────────────────────
    let mut selected: Option<Entity> = None;
    let start = Instant::now();
    let mut last_frame = Instant::now();
    let mut stats = PipelineStats::default();

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

                            // Stress-test controls: bulk spawn / despawn to
                            // exercise the scripting cache and ECS under load.
                            if ui.button("Spawn 100 scripted") {
                                let base = world.query::<&Mesh>().iter().count();
                                for i in 0..100usize {
                                    let a = (i as f32 / 100.0) * std::f32::consts::TAU;
                                    let r = 1.5 + (i as f32 / 100.0) * 4.0;
                                    world.spawn((
                                        Transform {
                                            position: Vec3::new(a.cos() * r, 0.0, a.sin() * r),
                                            rotation_axis: Vec3::new(
                                                (i as f32 * 0.3).sin(),
                                                1.0,
                                                (i as f32 * 0.7).cos(),
                                            ).normalize(),
                                            rotation_speed: 0.2 + (i % 7) as f32 * 0.15,
                                            scale: 0.2,
                                        },
                                        Mesh(if i % 2 == 0 { cube } else { sphere }),
                                        Name(format!("swarm_{}", base + i)),
                                        Script("scripts/swarm.py".to_string()),
                                    ));
                                }
                            }
                            ui.same_line();
                            if ui.button("Despawn scripted") {
                                let scripted: Vec<Entity> = world
                                    .query::<&Script>()
                                    .iter()
                                    .map(|(e, _)| e)
                                    .collect();
                                if selected.map(|e| scripted.contains(&e)).unwrap_or(false) {
                                    selected = None;
                                }
                                for e in scripted {
                                    let _ = world.despawn(e);
                                }
                            }

                            ui.separator();

                            // ── Bulk spawn for stress-testing ─────────────────
                            if ui.button("Spawn 1k") {
                                spawn_bulk(&mut world, cube, sphere, 1_000);
                            }
                            ui.same_line();
                            if ui.button("Spawn 10k") {
                                spawn_bulk(&mut world, cube, sphere, 10_000);
                            }
                            ui.same_line();
                            if ui.button("Despawn all") {
                                let all: Vec<Entity> =
                                    world.query::<&Mesh>().iter().map(|(e, _)| e).collect();
                                if selected.map(|e| all.contains(&e)).unwrap_or(false) {
                                    selected = None;
                                }
                                for e in all { let _ = world.despawn(e); }
                            }

                            ui.separator();

                            let entities: Vec<Entity> =
                                world.query::<&Mesh>().iter().map(|(e, _)| e).collect();
                            let mut clipper = imgui::ListClipper::new(entities.len() as i32)
                                .begin(ui);
                            while clipper.step() {
                                for i in clipper.display_start()..clipper.display_end() {
                                    let e = entities[i as usize];
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

                                ui.separator();

                                // ── Material ──────────────────────────────────
                                {
                                    let mat_idx = world.get::<&Material>(sel).ok().map(|m| m.0);
                                    let effective_idx = mat_idx.unwrap_or(0);
                                    let is_override = renderer.is_override_material(effective_idx);
                                    let current_mat = renderer.material(effective_idx).copied();

                                    ui.text("Material");
                                    if is_override {
                                        ui.text_disabled(format!("Override #{effective_idx}"));
                                        if let Some(mat) = current_mat {
                                            let mut color     = mat.base_color;
                                            let mut roughness = mat.roughness;
                                            let mut metallic  = mat.metallic;

                                            let mut changed = false;
                                            changed |= ui.color_edit4("Base Color##mat", &mut color);
                                            changed |= ui.slider("Roughness##mat", 0.0f32, 1.0f32, &mut roughness);
                                            changed |= ui.slider("Metallic##mat",  0.0f32, 1.0f32, &mut metallic);

                                            if changed {
                                                let updated = GpuMaterial {
                                                    base_color: color,
                                                    roughness,
                                                    metallic,
                                                    _pad: [0.0; 2],
                                                };
                                                unsafe { renderer.update_override_material(&gl, effective_idx, updated); }
                                            }
                                        }
                                        if ui.button("Reset to Default##mat") {
                                            let _ = world.remove::<(Material,)>(sel);
                                        }
                                    } else {
                                        if let Some(mat) = current_mat {
                                            ui.text_disabled(format!("Static #{effective_idx}"));
                                            let [r, g, b, _] = mat.base_color;
                                            ui.text(format!("Color   {r:.2}  {g:.2}  {b:.2}"));
                                            ui.text(format!("Rough   {:.2}   Metal  {:.2}", mat.roughness, mat.metallic));
                                        }
                                        if ui.button("Override##mat") {
                                            let base = renderer.material(effective_idx)
                                                .copied()
                                                .unwrap_or(GpuMaterial::DEFAULT);
                                            let new_idx = unsafe { renderer.add_override_material(&gl, base) };
                                            let _ = world.insert(sel, (Material(new_idx),));
                                        }
                                    }
                                }

                                ui.separator();

                                // Script component
                                let cur_script = world
                                    .get::<&Script>(sel)
                                    .ok()
                                    .map(|s| s.0.clone())
                                    .unwrap_or_default();
                                let mut script_buf = cur_script.clone();
                                if ui.input_text("Script", &mut script_buf).build() {
                                    if script_buf.is_empty() {
                                        let _ = world.remove::<(Script,)>(sel);
                                    } else if script_buf != cur_script {
                                        let _ = world.insert(sel, (Script(script_buf),));
                                    }
                                }
                                if cur_script.is_empty() {
                                    ui.text_disabled("e.g. scripts/orbit.py");
                                }
                            }
                            None => ui.text("Select an entity in the Hierarchy."),
                        });

                    // ── Stats ────────────────────────────────────────────────
                    ui.window("Stats")
                        .position([0.0, 556.0], Condition::FirstUseEver)
                        .size([240.0, 190.0], Condition::FirstUseEver)
                        .build(|| {
                            ui.text(format!("FPS          {:.0}", ui.io().framerate));
                            ui.separator();
                            ui.text(format!("Entities     {}", stats.entities));
                            ui.text(format!("Batches      {}", stats.batches));
                            ui.separator();
                            ui.text(format!("Triangles    {}", fmt_count(stats.primitives_submitted)));
                            ui.text(format!("Vert inv     {}", fmt_count(stats.vertex_invocations)));
                            ui.text(format!("Frag inv     {}", fmt_count(stats.fragment_invocations)));
                            ui.text(format!("GPU ms       {:.2}", stats.gpu_time_ms));
                        });

                    // ── Script update (mutates world before rendering) ────────
                    let dt = ui.io().delta_time;
                    script_host.tick(&mut world, elapsed, dt);

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
                                stats =
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
