# gl_renderer

A Cargo workspace with two members:

- **`gl_renderer/`** — a reusable OpenGL 4.3 rendering **library** (engine).
- **`editor/`** — a Unity/Unreal-style **editor** binary built on the library, using
  Dear ImGui (imgui-rs) for a dockable UI with the live 3D scene in a central viewport.

The library is a minimal but modern OpenGL 4.3 renderer demonstrating:

- **Vertex pulling** via SSBO — no `glVertexAttribPointer`, vertices fetched by `gl_VertexID` in the shader
- **Shared GPU arena** — one vertex SSBO and one index buffer; meshes sub-allocated into ranges
- **Multi-Draw-Indirect** — the whole scene drawn in a single `glMultiDrawElementsIndirect`, one command per object built into a draw-indirect buffer
- **GPU-side per-object data** — model/normal matrices live in an instance SSBO indexed by `gl_DrawIDARB`; no per-draw uniform updates
- **ECS scene** — objects are `hecs` entities (`Transform` + `Mesh`); a per-frame query builds the instance + indirect buffers
- **Half-float vertex packing** — CPU side uses the `half` crate; GPU unpacks with `unpackHalf2x16`
- **Runtime GLSL compilation** — shaders compiled in-process via `glow`, no external CLI
- **`GlBuffer<T>`** — typed buffer abstraction with subrange upload, mapped upload with explicit flush, SSBO/IBO/indirect bind helpers

## Crates

| Crate | Role |
|---|---|
| `winit 0.29` | Cross-platform windowing and event loop |
| `glutin 0.31` + `glutin-winit 0.4` | OpenGL context creation |
| `glow 0.13` | Safe OpenGL bindings |
| `glam 0.25` | Math (Mat4, Vec3, Mat3) |
| `half 2.3` | f16 type for CPU-side packing |
| `bytemuck 1.14` | Safe `Pod` casts for buffer uploads |
| `hecs 0.10` | Lightweight ECS for the scene graph |
| `imgui 0.12` + `-winit-support 0.12` + `-glow-renderer 0.12` | Editor UI (editor crate only) |

The imgui 0.12 line is pinned deliberately: it targets `winit 0.29` + `glow 0.13`, matching
the engine's stack. (The 0.13 line needs winit 0.30 + glow 0.14.) imgui-sys compiles C++ Dear
ImGui, so a **C/C++ compiler is required** to build the editor.

> `glMultiDrawElementsIndirect` is not wrapped by glow's `HasContext`, so the entry
> point is loaded directly via glutin's `get_proc_address` (see `MultiDrawElementsIndirectFn`
> in `gl_renderer/src/renderer.rs`, loaded in `gl_renderer/src/gl_init.rs`). Core in GL 4.3.

## Architecture

```
gl_renderer/  (library)
  lib.rs        – module declarations + re-exports
  renderer.rs   Renderer     – owns NO gl/world; new(&gl,mdi), upload_mesh(), render(&gl,&world,…)
                               builds instance+indirect buffers from the ECS world → 1 MDI call
                MultiDrawElementsIndirectFn – raw glMultiDrawElementsIndirect entry point
  gl_init.rs    create_gl_window() – winit window + glutin 4.3 context + glow + MDI load
                GlWindow     – { window, surface, context, gl, mdi, gl_config }
  framebuffer.rs OffscreenTarget – RGBA8 color tex + depth RBO; render-to-texture for the editor
  components.rs Transform     – position/axis/speed/scale; model_matrix(elapsed)
                Mesh          – newtype over MeshAlloc; tags an entity's arena range
                Name          – optional display name (editor hierarchy)
  gl_buffer.rs  GlBuffer<T>   – typed GL buffer; upload, map, bind (SSBO/IBO/indirect)
  arena.rs      GpuArena      – shared vertex SSBO + IBO; push_mesh() sub-allocates
                MeshAlloc     – (vertex_offset, vertex_count, index_offset, index_count)
                DrawCommand   – mirror of DrawElementsIndirectCommand (5×u32, 20 bytes)
                InstanceData  – per-object model + normal matrices (2× mat4, std430)
                PackedVertex  – 5×u32, each u32 = packHalf2x16(a, b)
  geometry.rs   – cube and UV-sphere generators returning PackedVertex/u32 vecs
  shaders.rs    – GLSL source strings + compile_shader / build_program helpers
  examples/spinning.rs – the standalone demo (renders to the window's default framebuffer)

editor/  (binary)
  main.rs       – owns the glow context (shared with imgui via Rc), the hecs::World, and the
                  winit event loop; imgui dockspace + Hierarchy/Inspector/Stats/Viewport panels
```

The library deliberately does **not** own the `glow::Context` or the `hecs::World`: the host
(demo or editor) owns them and passes `&Context` / `&World` into `Renderer` methods. This lets
the editor share one GL context between the engine renderer and `imgui-glow-renderer`.

### Vertex layout (`PackedVertex`)

```
Offset  Field    GLSL unpack
  0     pos_xy   unpackHalf2x16(pv.pos_xy)  → pos.xy
  4     pos_z    unpackHalf2x16(pv.pos_z).x → pos.z
  8     norm_xy  unpackHalf2x16(pv.norm_xy)
 12     norm_z   unpackHalf2x16(pv.norm_z).x
 16     uv_xy    unpackHalf2x16(pv.uv_xy)
```

Total: **20 bytes** per vertex.

### Arena, indirect commands, and the single MDI call

```
Shared vertex SSBO:   [cube verts 0..23][sphere verts 24..N]   (binding 0)
Shared index IBO:     [cube idx 0..35][sphere idx 36..M]
Instance SSBO:        [inst 0][inst 1][inst 2]...               (binding 1)
Draw-indirect buffer: [cmd 0][cmd 1][cmd 2]...

Per object (built each frame from the ECS world):
  DrawCommand { count, instance_count=1, first_index, base_vertex, base_instance }
  InstanceData { model, normal }            // base_instance indexes this array

glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, 0, draw_count, 0);
  // one call; the driver walks every DrawCommand in the indirect buffer

GLSL: InstanceData inst = instances[gl_DrawIDARB];  // per-draw transform
      PackedVertex  pv   = verts[gl_VertexID];       // base_vertex already applied
```

`base_vertex` plays the same role it did under `glDrawElementsBaseVertex` (shifts
`gl_VertexID` into the mesh's SSBO sub-range); `first_index` is the start index in the
shared IBO. The old per-object draw loop and `u_mvp`/`u_model`/`u_normal_mat` uniforms
are gone — transforms now ride in the instance SSBO, selected by `gl_DrawIDARB`.

### `GlBuffer<T>` API

```rust
GlBuffer::new(gl, capacity, usage)             // allocate, no data
GlBuffer::from_data(gl, &slice, usage)         // allocate + upload
buf.upload_subrange(gl, &slice, offset)        // glBufferSubData
buf.mapped_upload(gl, offset, count, |slice|{})// map + write + unmap
buf.mapped_upload_explicit_flush(...)          // map + write + flush + unmap
buf.bind_as_ssbo(gl, binding)                  // glBindBufferBase(SSBO, …)
buf.bind_as_ibo(gl)                            // glBindBuffer(ELEMENT_ARRAY_BUFFER, …)
buf.bind_as_indirect(gl)                       // glBindBuffer(DRAW_INDIRECT_BUFFER, …)
GlBuffer::unbind_ibo(gl)
GlBuffer::unbind_indirect(gl)
buf.cleanup(gl)                                // glDeleteBuffers
buf.byte_capacity()
```

## Build & run

Requires an OpenGL 4.3-capable driver (tested on Linux/Windows with Mesa and NVIDIA) and a
C/C++ compiler (for imgui-sys).

```sh
cargo build                                  # whole workspace
cargo run -p editor                          # the imgui editor
cargo run -p gl_renderer --example spinning  # the standalone demo
```

Debug builds enable `GL_DEBUG_OUTPUT` and print driver messages to stderr.

## Editor

The editor presents the engine in a dockable, Unity/Unreal-style layout:

- **Viewport** — the live 3D scene. The engine renders into an `OffscreenTarget` framebuffer,
  and the editor displays its color texture as an imgui image (V-flipped, since GL textures are
  bottom-up). The camera aspect follows the panel, not the window.
- **Hierarchy** — lists ECS entities by `Name`; select one, spawn cubes/spheres, delete.
- **Inspector** — edits the selected entity's `Transform` (position, rotation axis/speed, scale)
  live; changes are visible immediately because the scene is rebuilt from the world each frame.
- **Stats** — FPS, draw count (= entities drawn in the single MDI call), entity count.

The whole window is an imgui dockspace (`dockspace_over_main_viewport`, requires the imgui
`docking` feature). Panels open in a Unity-like arrangement on first run and can be re-docked;
the layout persists in `imgui.ini`. Spawning/deleting entities needs no renderer changes — the
MDI path picks up whatever `(Transform, Mesh)` entities exist.

## Scene

Three objects, stored as `hecs` entities (`Transform` + `Mesh`), drawn from the same two
meshes (cube and sphere) sub-allocated in the shared arena:

- Central cube — diagonal axis rotation
- Orbiting sphere — Y-axis spin
- Small offset cube — slow horizontal rotation

Each frame `Renderer::render` queries `(&Transform, &Mesh)`, builds one `InstanceData` +
one `DrawCommand` per entity, uploads them, and issues a single `glMultiDrawElementsIndirect`.
Adding an object is just another `world.spawn((Transform { .. }, Mesh(alloc)))` — no draw-loop
changes. The whole frame binds the vertex SSBO, instance SSBO, IBO, and indirect buffer once.
