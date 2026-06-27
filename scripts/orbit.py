"""
Per-entity orbit script.

Attach to any entity via the Inspector (type 'scripts/orbit.py' in the Script
field) or from Rust:
    world.insert(entity, (Script("scripts/orbit.py"),));

The entity circles the world origin at a radius driven by its initial distance.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gl_renderer import Entity, Scene


def on_update(entity: Entity, scene: Scene, t: float, dt: float) -> None:
    x0, y, z0 = entity.position
    radius = math.sqrt(x0 * x0 + z0 * z0) or 2.0
    speed = entity.rotation_speed * 0.5

    entity.position = (
        math.cos(t * speed) * radius,
        y,
        math.sin(t * speed) * radius,
    )
    entity.scale = 0.7 + 0.3 * math.sin(t * 1.3)
