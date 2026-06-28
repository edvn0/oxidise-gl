//! Per-entity Python scripting host.
//!
//! Each entity may carry a [`gl_renderer::Script`] component pointing to a
//! `.py` file. Every frame `ScriptHost::tick` calls `on_update(entity, scene,
//! t, dt)` for each such entity, hot-reloading the file when its mtime changes.
//!
//! Multiple entities may share the same script file; the file is compiled once
//! and the single `on_update` callable is reused.
//!
//! Register the built-in Python module once before Python initialises:
//!
//!   pyo3::append_to_inittab!(scripting::py_module);

use gl_renderer::{Mesh, MeshAlloc, Name, Script, Transform};
use glam::Vec3;
use hecs::{Entity, World};
use lru::LruCache;
use pyo3::{prelude::*, types::PyModule};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime};

// ── Python-visible types ──────────────────────────────────────────────────────

/// One scene object, handed to the Python script as a mutable handle.
#[pyclass(name = "Entity")]
pub struct EntityProxy {
    pub entity: Option<Entity>,
    pub mesh_name: String,
    pub name: String,
    pub position: (f32, f32, f32),
    pub rotation_axis: (f32, f32, f32),
    pub rotation_speed: f32,
    pub scale: f32,
    /// Path to the attached script, or None. Writeable so global scripts can
    /// assign scripts to entities dynamically.
    pub script: Option<String>,
    pub despawn_flag: bool,
}

#[pymethods]
impl EntityProxy {
    #[getter]
    fn get_mesh(&self) -> &str {
        &self.mesh_name
    }

    #[getter]
    fn get_name(&self) -> &str {
        &self.name
    }
    #[setter]
    fn set_name(&mut self, v: String) {
        self.name = v;
    }

    #[getter]
    fn get_position(&self) -> (f32, f32, f32) {
        self.position
    }
    #[setter]
    fn set_position(&mut self, v: (f32, f32, f32)) {
        self.position = v;
    }

    #[getter]
    fn get_rotation_axis(&self) -> (f32, f32, f32) {
        self.rotation_axis
    }
    #[setter]
    fn set_rotation_axis(&mut self, v: (f32, f32, f32)) {
        self.rotation_axis = v;
    }

    #[getter]
    fn get_rotation_speed(&self) -> f32 {
        self.rotation_speed
    }
    #[setter]
    fn set_rotation_speed(&mut self, v: f32) {
        self.rotation_speed = v;
    }

    #[getter]
    fn get_scale(&self) -> f32 {
        self.scale
    }
    #[setter]
    fn set_scale(&mut self, v: f32) {
        self.scale = v;
    }

    #[getter]
    fn get_script(&self) -> Option<&str> {
        self.script.as_deref()
    }
    #[setter]
    fn set_script(&mut self, v: Option<String>) {
        self.script = v;
    }

    /// Mark this entity for removal after `on_update` returns.
    fn despawn(&mut self) {
        self.despawn_flag = true;
    }

    fn __repr__(&self) -> String {
        format!("Entity(name={:?}, mesh={:?})", self.name, self.mesh_name)
    }
}

/// The full scene, passed into every `on_update` call each frame.
#[pyclass(name = "Scene")]
pub struct SceneProxy {
    entities: Vec<Py<EntityProxy>>,
    spawns: Vec<Py<EntityProxy>>,
    mesh_names: Vec<String>,
}

#[pymethods]
impl SceneProxy {
    /// All entities currently in the scene.
    fn entities(&self, py: Python<'_>) -> Vec<Py<EntityProxy>> {
        self.entities.iter().map(|e| e.clone_ref(py)).collect()
    }

    /// Spawn a new entity. `mesh` must be one of `available_meshes()`.
    #[pyo3(signature = (mesh, *, name = None, position = None, rotation_axis = None, rotation_speed = None, scale = None, script = None))]
    fn spawn(
        &mut self,
        py: Python<'_>,
        mesh: String,
        name: Option<String>,
        position: Option<(f32, f32, f32)>,
        rotation_axis: Option<(f32, f32, f32)>,
        rotation_speed: Option<f32>,
        scale: Option<f32>,
        script: Option<String>,
    ) -> PyResult<Py<EntityProxy>> {
        let proxy = Py::new(
            py,
            EntityProxy {
                entity: None,
                mesh_name: mesh,
                name: name.unwrap_or_default(),
                position: position.unwrap_or((0.0, 0.0, 0.0)),
                rotation_axis: rotation_axis.unwrap_or((0.0, 1.0, 0.0)),
                rotation_speed: rotation_speed.unwrap_or(0.5),
                scale: scale.unwrap_or(1.0),
                script,
                despawn_flag: false,
            },
        )?;
        self.spawns.push(proxy.clone_ref(py));
        Ok(proxy)
    }

    /// Return the first entity with the given name, or `None`.
    fn find(&self, py: Python<'_>, name: &str) -> Option<Py<EntityProxy>> {
        self.entities
            .iter()
            .find(|e| e.borrow(py).name == name)
            .map(|e| e.clone_ref(py))
    }

    /// `True` if any entity has the given name.
    fn has(&self, py: Python<'_>, name: &str) -> bool {
        self.entities.iter().any(|e| e.borrow(py).name == name)
    }

    /// Mesh names that can be passed to `spawn`.
    fn available_meshes(&self) -> Vec<String> {
        self.mesh_names.clone()
    }
}

// ── Cache tuning constants ────────────────────────────────────────────────────

/// How many compiled script modules to keep in memory simultaneously.
///
/// When the cap is reached the least-recently-used entry is evicted. Without a
/// cap the cache grows unboundedly: every path that ever appears in a Script
/// component stays in memory even after the component is removed. 128 is far
/// more than any real scene needs, so eviction only fires for genuinely stale
/// scripts.
const SCRIPT_CACHE_CAP: usize = 128;

/// Minimum time between successive `fs::metadata` calls on the same file.
///
/// At 60 fps, checking mtime every frame costs ~60 `stat()` syscalls per
/// second *per scripted entity*. A 100 ms cooldown reduces that to at most
/// 10/s while still feeling instant to a developer who saves a file.
/// The check is always performed on first load (when `on_update` is None),
/// so a new or previously-erroring script is never stuck waiting for the timer.
const STAT_COOLDOWN: Duration = Duration::from_millis(100);

// ── Script cache ──────────────────────────────────────────────────────────────

struct ScriptEntry {
    /// Wall-clock time of the last `fs::metadata` call.
    /// Used to gate how often we issue the syscall; see `STAT_COOLDOWN`.
    last_stat: Instant,
    last_mtime: Option<SystemTime>,
    on_update: Option<Py<PyAny>>,
}

impl ScriptEntry {
    fn new() -> Self {
        // Use a past instant so the very first tick always runs the stat.
        Self {
            last_stat: Instant::now() - STAT_COOLDOWN - Duration::from_millis(1),
            last_mtime: None,
            on_update: None,
        }
    }

    fn reload_if_changed(&mut self, py: Python<'_>, path: &PathBuf) {
        let now = Instant::now();

        // Skip the syscall entirely if we checked recently and the module is
        // already loaded. This is the hot path at steady state.
        if self.on_update.is_some() && now.duration_since(self.last_stat) < STAT_COOLDOWN {
            return;
        }
        self.last_stat = now;

        let mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
        if mtime == self.last_mtime && self.on_update.is_some() {
            return;
        }
        self.last_mtime = mtime;

        let code = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[script] {}: {e}", path.display());
                return;
            }
        };
        let filename = path.to_string_lossy();
        match PyModule::from_code_bound(py, &code, &filename, "scene") {
            Ok(module) => match module.getattr("on_update") {
                Ok(f) => {
                    eprintln!("[script] loaded {filename}");
                    self.on_update = Some(f.unbind());
                }
                Err(e) => {
                    e.print(py);
                    self.on_update = None;
                }
            },
            Err(e) => {
                e.print(py);
                self.on_update = None;
            }
        }
    }
}

// ── Script host ───────────────────────────────────────────────────────────────

pub struct ScriptHost {
    /// Compiled `on_update` callables keyed by file path.
    ///
    /// LruCache bounds memory: a script whose path disappears from all Script
    /// components will eventually be evicted rather than accumulating forever.
    /// Eviction happens inside `Python::with_gil` so dropped `Py<PyAny>` values
    /// always have the GIL when their refcount hits zero.
    cache: LruCache<PathBuf, ScriptEntry>,

    /// Forward map: mesh name → arena allocation. Used by `Scene::spawn`.
    mesh_map: HashMap<String, MeshAlloc>,

    /// Reverse map: `vertex_offset` → mesh name.
    ///
    /// The snapshot loop previously scanned `mesh_map` linearly for every
    /// entity to recover the human-readable name from a `MeshAlloc`. That is
    /// O(entities × meshes) per frame. Because `vertex_offset` is unique per
    /// arena allocation, a pre-built reverse map turns each lookup into O(1).
    mesh_reverse: HashMap<u32, String>,
}

impl ScriptHost {
    pub fn new() -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(SCRIPT_CACHE_CAP).unwrap()),
            mesh_map: HashMap::new(),
            mesh_reverse: HashMap::new(),
        }
    }

    /// Register a named mesh so Python scripts can spawn it by name.
    /// Also populates the reverse map used during the per-frame snapshot.
    pub fn register_mesh(&mut self, name: impl Into<String>, alloc: MeshAlloc) {
        let name = name.into();
        self.mesh_reverse.insert(alloc.vertex_offset, name.clone());
        self.mesh_map.insert(name, alloc);
    }

    /// Tick all scripted entities: snapshot → dispatch per-entity → write back.
    ///
    /// Dispatch is parallelised with Rayon. With free-threaded CPython 3.14 (GIL
    /// disabled) scripts run truly in parallel. With a standard GIL build they
    /// still use Rayon but each `Python::with_gil` acquires the GIL in turn, so
    /// execution is sequential — correct in both cases.
    pub fn tick(&mut self, world: &mut World, elapsed: f32, dt: f32) {
        if world.query::<&Script>().iter().next().is_none() {
            return;
        }

        Python::with_gil(|py| {
            // ── Phase 1: snapshot world ───────────────────────────────────────
            let mesh_names: Vec<String> = self.mesh_map.keys().cloned().collect();

            let mut entity_index: HashMap<Entity, usize> = HashMap::new();
            let proxies: Vec<Py<EntityProxy>> = {
                let mut v = Vec::new();
                for (entity, (xf, mesh, name, script)) in world
                    .query::<(&Transform, &Mesh, Option<&Name>, Option<&Script>)>()
                    .iter()
                {
                    let mesh_name = self
                        .mesh_reverse
                        .get(&mesh.0.vertex_offset)
                        .cloned()
                        .unwrap_or_default();
                    let idx = v.len();
                    entity_index.insert(entity, idx);
                    v.push(
                        Py::new(
                            py,
                            EntityProxy {
                                entity: Some(entity),
                                mesh_name,
                                name: name.map(|n| n.0.clone()).unwrap_or_default(),
                                position: (xf.position.x, xf.position.y, xf.position.z),
                                rotation_axis: (
                                    xf.rotation_axis.x,
                                    xf.rotation_axis.y,
                                    xf.rotation_axis.z,
                                ),
                                rotation_speed: xf.rotation_speed,
                                scale: xf.scale,
                                script: script.map(|s| s.0.clone()),
                                despawn_flag: false,
                            },
                        )
                        .expect("EntityProxy alloc"),
                    );
                }
                v
            };

            let scripted: Vec<(Entity, PathBuf)> = world
                .query::<&Script>()
                .iter()
                .map(|(e, s)| (e, PathBuf::from(&s.0)))
                .collect();

            // ── Phase 2a: serial cache pre-pass ──────────────────────────────
            // The LruCache is not Sync, so we resolve all on_update callables
            // here before entering the parallel region.
            struct DispatchItem {
                proxy_idx: usize,
                on_update: Py<PyAny>,
            }
            let dispatches: Vec<DispatchItem> = scripted
                .iter()
                .filter_map(|(entity, path)| {
                    // peek preserves recency; get_mut below is the LRU "use".
                    if self.cache.peek(path).is_none() {
                        self.cache.put(path.clone(), ScriptEntry::new());
                    }
                    let entry = self.cache.get_mut(path).unwrap();
                    entry.reload_if_changed(py, path);
                    let on_update = entry.on_update.as_ref()?.clone_ref(py);
                    let proxy_idx = *entity_index.get(entity)?;
                    Some(DispatchItem {
                        proxy_idx,
                        on_update,
                    })
                })
                .collect();

            // One shared SceneProxy for all threads.
            //
            // Each Rayon thread holds a clone_ref (one atomic refcount bump)
            // rather than building its own copy of the entity list. With
            // free-threaded CPython, PyO3 applies per-object locking to every
            // #[pyclass] method call: &self methods allow concurrent readers,
            // &mut self methods (e.g. spawn) are exclusive. Distinct EntityProxy
            // objects are independent locks, so parallel on_update calls that
            // each touch only their own entity proxy proceed without contention.
            let scene = Py::new(
                py,
                SceneProxy {
                    entities: proxies.iter().map(|e| e.clone_ref(py)).collect(),
                    spawns: Vec::new(),
                    mesh_names,
                },
            )
            .expect("SceneProxy alloc");

            // ── Phase 2b: parallel dispatch ───────────────────────────────────
            // Chunk entities so each Rayon thread acquires the GIL exactly once
            // for its whole batch. With a GIL build this gives ~N_threads GIL
            // cycles per tick instead of N_entities, keeping overhead close to
            // the old serial baseline. With free-threaded CPython, each thread's
            // with_gil is a near-zero-cost token and batches run in true parallel.
            use rayon::prelude::*;
            let n_threads = rayon::current_num_threads();
            let chunk_size = (dispatches.len() + n_threads - 1) / n_threads;
            py.allow_threads(|| {
                dispatches.par_chunks(chunk_size.max(1)).for_each(|chunk| {
                    Python::with_gil(|py2| {
                        for d in chunk {
                            let ep = proxies[d.proxy_idx].clone_ref(py2);
                            let scene_ref = scene.clone_ref(py2);
                            if let Err(e) = d
                                .on_update
                                .call1(py2, (ep.bind(py2), scene_ref.bind(py2), elapsed, dt))
                            {
                                e.print(py2);
                            }
                        }
                    });
                });
            });

            // ── Phase 3: write mutations back ─────────────────────────────────
            for proxy_py in &proxies {
                let proxy = proxy_py.borrow(py);
                let entity = proxy.entity.unwrap();

                if proxy.despawn_flag {
                    let _ = world.despawn(entity);
                    continue;
                }

                if let Ok(mut xf) = world.get::<&mut Transform>(entity) {
                    xf.position = Vec3::from(proxy.position);
                    xf.rotation_axis = normalize_or_up(Vec3::from(proxy.rotation_axis));
                    xf.rotation_speed = proxy.rotation_speed;
                    xf.scale = proxy.scale;
                }

                let needs_name_insert = match world.get::<&mut Name>(entity) {
                    Ok(mut n) => {
                        n.0.clone_from(&proxy.name);
                        false
                    }
                    Err(_) => !proxy.name.is_empty(),
                };
                if needs_name_insert {
                    let _ = world.insert(entity, (Name(proxy.name.clone()),));
                }

                let new_script = proxy.script.clone();
                let cur_script = world.get::<&Script>(entity).ok().map(|s| s.0.clone());
                match (new_script, cur_script) {
                    (Some(p), cur) if cur.as_deref() != Some(p.as_str()) => {
                        let _ = world.insert(entity, (Script(p),));
                    }
                    (None, Some(_)) => {
                        let _ = world.remove::<(Script,)>(entity);
                    }
                    _ => {}
                }
            }

            let scene_ref = scene.borrow(py);
            for spawn_py in &scene_ref.spawns {
                let proxy = spawn_py.borrow(py);
                if proxy.despawn_flag {
                    continue;
                }
                let Some(&alloc) = self.mesh_map.get(&proxy.mesh_name) else {
                    continue;
                };
                let entity = world.spawn((
                    Transform {
                        position: Vec3::from(proxy.position),
                        rotation_axis: normalize_or_up(Vec3::from(proxy.rotation_axis)),
                        rotation_speed: proxy.rotation_speed,
                        scale: proxy.scale,
                    },
                    Mesh(alloc),
                ));
                if !proxy.name.is_empty() {
                    let _ = world.insert(entity, (Name(proxy.name.clone()),));
                }
                if let Some(path) = &proxy.script {
                    let _ = world.insert(entity, (Script(path.clone()),));
                }
            }
        });
    }
}

fn normalize_or_up(v: Vec3) -> Vec3 {
    if v.length_squared() > 1e-6 {
        v.normalize()
    } else {
        Vec3::Y
    }
}

// ── Built-in Python module ────────────────────────────────────────────────────

#[pymodule(name = "gl_renderer")]
pub fn py_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EntityProxy>()?;
    m.add_class::<SceneProxy>()?;
    Ok(())
}
