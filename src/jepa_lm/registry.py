from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, family: str) -> None:
        self.family = family
        self._entries: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            if name in self._entries:
                raise ValueError(f"{self.family}: '{name}' already registered")
            self._entries[name] = cls
            return cls

        return decorator

    def build(self, name: str, **kwargs) -> T:
        if name not in self._entries:
            raise KeyError(
                f"{self.family}: '{name}' not registered. Available: {sorted(self._entries)}"
            )
        return self._entries[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._entries
