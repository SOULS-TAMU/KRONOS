"""The bundled test problems.

244 problems from the Hock-Schittkowski collection, CUTEst, and standard
global-optimisation test functions, spanning 2 to 1000 variables. Each carries
its known optimum ``fstar``.

Use :func:`problem_names` to list them, or :func:`find` to search.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

from .problem import Problem

__all__ = ["problem_names", "load_problem", "iter_problems", "find",
           "resolve_name", "DATA_DIR"]

DATA_DIR = Path(__file__).parent / "data" / "problems"


@lru_cache(maxsize=1)
def problem_names() -> tuple[str, ...]:
    """Names of every bundled problem, sorted."""
    if not DATA_DIR.is_dir():
        return ()
    return tuple(sorted(p.stem for p in DATA_DIR.glob("*.json")))


@lru_cache(maxsize=1)
def _alias_map() -> dict:
    """Case-insensitive lookup of the bundled names."""
    alias: dict[str, list[str]] = {}
    for canonical in problem_names():
        alias.setdefault(canonical.lower(), []).append(canonical)
    return alias


def resolve_name(name: str) -> str:
    """Resolve a user-supplied name to a canonical library name.

    Accepts the exact name, any case variant, or an unambiguous prefix.
    """
    # Compare against the listing rather than the filesystem: macOS paths are
    # case-insensitive, so is_file() would accept "a01_beale" as canonical.
    if name in problem_names():
        return name
    key = name.lower()
    hits = _alias_map().get(key)
    if hits and len(hits) == 1:
        return hits[0]
    if hits:
        raise KeyError(f"{name!r} is ambiguous: {', '.join(sorted(hits))}")
    prefix = [n for n in problem_names() if n.lower().startswith(key)]
    if len(prefix) == 1:
        return prefix[0]
    hint = f"  Did you mean: {', '.join(sorted(prefix)[:5])}?" if prefix else ""
    raise KeyError(f"no bundled problem named {name!r}.{hint}")


@lru_cache(maxsize=None)
def _load_cached(canonical: str) -> Problem:
    path = DATA_DIR / f"{canonical}.json"
    if not path.is_file():
        raise KeyError(f"no bundled problem named {canonical!r}")
    return Problem.from_dict(json.loads(path.read_text()))


def load_problem(name: str) -> Problem:
    """Load a bundled problem by name.

    ``load_problem("hs001")``, ``load_problem("hs001_done")`` and
    ``load_problem("HS001")`` all resolve to the same problem.
    """
    import copy
    return copy.deepcopy(_load_cached(resolve_name(name)))


def find(pattern: str = "", max_n: Optional[int] = None,
         constrained: Optional[bool] = None) -> list[str]:
    """Search the bundled problems.

    >>> find("hs")                       # names containing "hs"
    >>> find(max_n=10, constrained=True) # small constrained problems
    """
    out = []
    for name in problem_names():
        if pattern and pattern.lower() not in name.lower():
            continue
        p = _load_cached(name)
        if max_n is not None and p.n > max_n:
            continue
        if constrained is not None and (p.m > 0) != constrained:
            continue
        out.append(name)
    return out


def iter_problems(
    max_n: Optional[int] = None,
    min_n: int = 0,
    constrained: Optional[bool] = None,
) -> Iterator[Problem]:
    """Iterate the library, optionally filtered by size or constrainedness."""
    for name in problem_names():
        p = load_problem(name)
        if p.n < min_n or (max_n is not None and p.n > max_n):
            continue
        if constrained is not None and (p.m > 0) != constrained:
            continue
        yield p
