//! Headless stress test for the scripting host + ECS integration.
//!
//! No GL context or window needed — ScriptHost::tick only touches the
//! hecs::World and the Python interpreter.
//!
//! Run from the workspace root (so scripts/ resolves):
//!   cargo run -p editor --bin stress

mod scripting;

use gl_renderer::{Mesh, MeshAlloc, Name, Script, Transform};
use glam::Vec3;
use hecs::World;
use std::time::Instant;

/// Fake arena allocation — offset must be unique per mesh type so the
/// reverse-lookup map in ScriptHost resolves names correctly.
fn fake_alloc(vertex_offset: u32) -> MeshAlloc {
    MeshAlloc {
        vertex_offset,
        vertex_count: 24,
        index_offset: vertex_offset * 3,
        index_count: 36,
    }
}

fn main() {
    // Must be called before any Python::with_gil; see scripting module doc.
    use scripting::py_module;
    pyo3::append_to_inittab!(py_module);

    let cube   = fake_alloc(0);
    let sphere = fake_alloc(1_000);

    let mut host = scripting::ScriptHost::new();
    host.register_mesh("cube",   cube);
    host.register_mesh("sphere", sphere);

    const ENTITIES: usize  = 200;
    const ROUNDS:   usize  = 5;
    const TICKS:    usize  = 180; // 3 s at 60 fps

    let total_start = Instant::now();

    for round in 0..ROUNDS {
        let mut world = World::new();

        // ── Spawn ─────────────────────────────────────────────────────────────
        for i in 0..ENTITIES {
            let a = (i as f32 / ENTITIES as f32) * std::f32::consts::TAU;
            let r = 1.5 + i as f32 * 0.02;
            world.spawn((
                Transform {
                    position:       Vec3::new(a.cos() * r, 0.0, a.sin() * r),
                    rotation_axis:  Vec3::Y,
                    rotation_speed: 0.2 + (i % 7) as f32 * 0.15,
                    scale:          0.3,
                },
                Mesh(if i % 2 == 0 { cube } else { sphere }),
                Name(format!("e{i}")),
                Script("scripts/swarm.py".to_string()),
            ));
        }

        assert_eq!(world.len() as usize, ENTITIES, "round {round}: wrong entity count after spawn");

        // ── Tick ──────────────────────────────────────────────────────────────
        let tick_start = Instant::now();
        for tick in 0..TICKS {
            host.tick(&mut world, tick as f32 / 60.0, 1.0 / 60.0);
        }
        let tick_ms = tick_start.elapsed().as_millis();

        let post_tick = world.len();
        assert_eq!(post_tick as usize, ENTITIES,
            "round {round}: entity count changed during ticks ({post_tick} != {ENTITIES})");
        eprintln!("[round {round}] {TICKS} ticks × {ENTITIES} entities  —  {tick_ms} ms  ({:.1} ms/tick)",
            tick_ms as f64 / TICKS as f64);

        // ── Despawn half, tick, despawn rest ──────────────────────────────────
        let all_scripted: Vec<_> = world.query::<&Script>().iter().map(|(e, _)| e).collect();
        assert_eq!(all_scripted.len(), ENTITIES);

        // Remove first half.
        for &e in &all_scripted[..ENTITIES / 2] {
            world.despawn(e).unwrap();
        }
        assert_eq!(world.len() as usize, ENTITIES / 2);

        // Tick with half still alive — verifies no stale entity access.
        for tick in 0..10 {
            host.tick(&mut world, tick as f32 / 60.0, 1.0 / 60.0);
        }
        assert_eq!(world.len() as usize, ENTITIES / 2, "round {round}: entity leaked after partial despawn");

        // Remove the rest.
        let remaining: Vec<_> = world.query::<&Script>().iter().map(|(e, _)| e).collect();
        assert_eq!(remaining.len(), ENTITIES / 2);
        for e in remaining {
            world.despawn(e).unwrap();
        }
        assert_eq!(world.len(), 0, "round {round}: world not empty after full despawn");
        eprintln!("[round {round}] despawn: all {ENTITIES} removed, world empty ✓");
    }

    eprintln!("\nAll {ROUNDS} rounds passed in {:.2?}", total_start.elapsed());
}
