"""
Example: scripts are now per-entity, not global.

Attach a script to an entity in the Inspector (Script field), or from Rust:

    world.insert(entity, (Script("scripts/orbit.py"),));

See orbit.py for a working example.

on_update signature for per-entity scripts:

    def on_update(entity: Entity, scene: Scene, t: float, dt: float) -> None:
        ...

- entity  the entity this script is attached to
- scene   the full scene (read/write all entities, spawn, despawn)
- t       elapsed seconds since launch
- dt      delta time this frame in seconds
"""
