"""Type stubs for the gl_renderer built-in Python module.

These give Pylance / Pyright full autocomplete and type-checking for scene
scripts without requiring gl_renderer to be installed as a Python package.
"""

class Entity:
    """A scene object. Passed as the first argument to per-entity `on_update`."""

    @property
    def mesh(self) -> str:
        """Read-only name of the mesh ('cube', 'sphere', …)."""
        ...

    @property
    def name(self) -> str: ...
    @name.setter
    def name(self, value: str) -> None: ...

    @property
    def position(self) -> tuple[float, float, float]: ...
    @position.setter
    def position(self, value: tuple[float, float, float]) -> None: ...

    @property
    def rotation_axis(self) -> tuple[float, float, float]: ...
    @rotation_axis.setter
    def rotation_axis(self, value: tuple[float, float, float]) -> None: ...

    @property
    def rotation_speed(self) -> float: ...
    @rotation_speed.setter
    def rotation_speed(self, value: float) -> None: ...

    @property
    def scale(self) -> float: ...
    @scale.setter
    def scale(self, value: float) -> None: ...

    @property
    def script(self) -> str | None:
        """Path to this entity's script, or None. Writeable."""
        ...
    @script.setter
    def script(self, value: str | None) -> None: ...

    def despawn(self) -> None:
        """Remove this entity after `on_update` returns."""
        ...

    def __repr__(self) -> str: ...


class Scene:
    """The full scene, passed into every `on_update` call each frame."""

    def entities(self) -> list[Entity]:
        """All entities currently in the scene."""
        ...

    def spawn(
        self,
        mesh: str,
        *,
        name: str = "",
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
        rotation_speed: float = 0.5,
        scale: float = 1.0,
        script: str | None = None,
    ) -> Entity:
        """Spawn a new entity. `mesh` must be in `available_meshes()`."""
        ...

    def find(self, name: str) -> Entity | None:
        """Return the first entity with the given name, or None."""
        ...

    def has(self, name: str) -> bool:
        """True if any entity has the given name."""
        ...

    def available_meshes(self) -> list[str]:
        """Mesh names registered with the renderer ('cube', 'sphere', …)."""
        ...
