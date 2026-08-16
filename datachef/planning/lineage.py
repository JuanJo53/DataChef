"""Shared deterministic column lineage for ordered declarative plans."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ColumnLineage:
    _original_to_current: dict[str, str]

    @classmethod
    def from_columns(cls, columns: tuple[str, ...]) -> "ColumnLineage":
        return cls({column: column for column in columns})

    @property
    def current_columns(self) -> tuple[str, ...]:
        return tuple(self._original_to_current.values())

    def rename(self, source: str, target: str) -> bool:
        original = self.original_for_current(source)
        if original is None:
            return False
        self._original_to_current[original] = target
        return True

    def original_for_current(self, current: str) -> str | None:
        return next(
            (
                original
                for original, current_name in self._original_to_current.items()
                if current_name == current
            ),
            None,
        )

    def current_for_original(self, original: str) -> str | None:
        return self._original_to_current.get(original)

    def originals_for_current(
        self,
        columns: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        originals = tuple(self.original_for_current(column) for column in columns)
        if any(column is None for column in originals):
            return None
        return tuple(column for column in originals if column is not None)

    def currents_for_original(
        self,
        columns: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        currents = tuple(self.current_for_original(column) for column in columns)
        if any(column is None for column in currents):
            return None
        return tuple(column for column in currents if column is not None)


__all__ = ["ColumnLineage"]
