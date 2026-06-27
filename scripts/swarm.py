"""
Swarm script — stress test for many scripted entities.

Attach to any entity via the Inspector or by clicking "Spawn 100 scripted"
in the Hierarchy panel. Each entity orbits the origin at a radius set by
its initial distance, with a gentle vertical bob.

Edit and save this file while the editor is running to confirm hot-reload
works correctly across all 100 entities simultaneously.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gl_renderer import Entity, Scene


def on_update(entity: Entity, scene: Scene, t: float, dt: float) -> None:
    x, _y, z = entity.position

    # Preserve each entity's orbital radius from its spawn position.
    r = math.sqrt(x * x + z * z)
    if r < 0.1:
        r = 2.0

    angle = math.atan2(z, x) + entity.rotation_speed * dt
    entity.position = (
        math.cos(angle) * r,
        math.sin(t * 1.5 + r * 0.8) * 0.6,  # vertical bob, phase-shifted by radius
        math.sin(angle) * r,
    )
    entity.scale = 0.15 + 0.08 * abs(math.sin(t * 2.0 + r))
