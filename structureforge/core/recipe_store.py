"""Persisted, editable custom recipes: a JSON-backed store so a deposition/etch recipe added or
tweaked (from a script, or from the GUI's recipe manager) survives past one process and is
reusable across runs. `default_recipes()` stays the read-only, in-code starting point;
`RecipeStore` is where custom ones live, on disk, so they don't vanish when the server restarts.
"""

from __future__ import annotations

import json
from pathlib import Path

from .recipes import DepositionRecipe, EtchRecipe, RecipeLibrary


class RecipeStore:
    """A `RecipeLibrary` of *custom* recipes, persisted as JSON at `path`.

    `combined_with(defaults)` merges the custom recipes on top of a base library (typically
    `default_recipes()`): a custom recipe sharing a built-in's name overrides it, anything else
    is additive - the same override-by-name rule `RecipeLibrary.with_recipes` already uses.
    Reading a store whose file doesn't exist yet just gives an empty `RecipeLibrary`; the file is
    created (parent directories included) on first save.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> RecipeLibrary:
        if not self.path.exists():
            return RecipeLibrary()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return RecipeLibrary.model_validate(data)

    def save(self, library: RecipeLibrary) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(library.model_dump_json(indent=2), encoding="utf-8")

    def upsert_deposition(self, recipe: DepositionRecipe) -> RecipeLibrary:
        """Add `recipe`, or replace the custom one already saved under the same name."""
        updated = self.load().with_recipes(deposition=[recipe])
        self.save(updated)
        return updated

    def upsert_etch(self, recipe: EtchRecipe) -> RecipeLibrary:
        updated = self.load().with_recipes(etch=[recipe])
        self.save(updated)
        return updated

    def remove_deposition(self, name: str) -> RecipeLibrary:
        """Drop `name` from the custom store. If it was overriding a built-in recipe, that
        built-in reappears (unaffected, since it was never touched) the next time this store is
        combined with a base library - only the override itself is gone.
        """
        library = self.load()
        updated = RecipeLibrary(
            deposition={k: v for k, v in library.deposition.items() if k != name}, etch=library.etch
        )
        self.save(updated)
        return updated

    def remove_etch(self, name: str) -> RecipeLibrary:
        library = self.load()
        updated = RecipeLibrary(
            deposition=library.deposition, etch={k: v for k, v in library.etch.items() if k != name}
        )
        self.save(updated)
        return updated

    def combined_with(self, defaults: RecipeLibrary) -> RecipeLibrary:
        """`defaults` with every custom recipe layered on top (custom wins on a name clash)."""
        custom = self.load()
        return defaults.with_recipes(
            deposition=list(custom.deposition.values()), etch=list(custom.etch.values())
        )
