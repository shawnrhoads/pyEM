from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ModelSpec:
    """Self-describing bundle for a model's identity and entry points.

    ``id`` and ``spec`` are hand-authored per model (there is no shared
    taxonomy to derive them from) — pick whatever short id and descriptive
    dict shape makes sense for that model family. Purely additive: nothing
    in :class:`pyem.api.EMModel` or :class:`pyem.core.compare.ModelComparison`
    requires a ``ModelSpec`` to function.
    """
    id: str
    spec: dict
    desc: str
    params: Callable | None
    sim: Callable
    fit: Callable
