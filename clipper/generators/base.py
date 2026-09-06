"""shared generator contract and helpers (spec section 4.5).

each of g1-g4 lives in its own module under `clipper/generators/` and
returns `list[Candidate]` with `generators=["its_name"]` set on every
candidate it produces -- `clipper/stages/generate.py`'s normalization step
is what unions `generators` across candidates with identical atom ranges,
not the generators themselves.

three distinct call signatures exist across the four generators, not one,
so this module does not force a single `Protocol` onto all of them:

- g2 (`discourse`) and g3 (`lexical`) are pure, deterministic functions of
  `(atoms, cfg)` alone, with no llm and no dependency on any other
  generator's output -- `DeterministicGenerator` below is their shared
  signature.
- g1 (`llm_pass`) additionally needs `run_dir` (for `complete_structured`'s
  cache/stats/event bookkeeping, which is keyed off the run directory) and
  is `async` (`complete_structured` is async). its signature is
  `async def generate_llm_pass(atoms, cfg, run_dir) -> list[Candidate]`,
  documented in `clipper/generators/llm_pass.py` rather than as a formal
  protocol here, since it genuinely differs in both extra parameter and
  sync/async-ness.
- g4 (`coverage`) runs *last* and depends on g1-g3's *combined output*, not
  just `atoms`/`cfg` -- its signature is
  `def generate_coverage(atoms, cfg, existing) -> list[Candidate]`,
  documented in `clipper/generators/coverage.py`.

forcing all four into one shared `Protocol` would mean every generator
carries optional/ignored parameters it doesn't use; three small, honestly
different signatures is simpler than one wide one. see DECISIONS.md.

`pre_score` placeholder: `Candidate.pre_score` (spec section 3) is a
required field, but its real value (spec section 4.6's prefilter formula:
`0.40*llm_pass_membership + 0.20*duration_fit + 0.15*n_generators/3 +
0.15*(1-silence_ratio) + 0.10*lexical_hit`) depends on information no single
generator owns -- generator membership is only known after merging, and
`silence_ratio` comes from `features.json` (spec section 4.4, phase 2, not
built yet). computing `pre_score` is therefore the M8 prefilter stage's job,
not any generator's. every generator built in this milestone sets
`pre_score=0.0` via `build_candidate` below as an explicit, documented
placeholder -- not a real score.
"""

from __future__ import annotations

from typing import Protocol

from clipper.config import Config
from clipper.types import Atom, Candidate


class DeterministicGenerator(Protocol):
    """shared signature for g2 (`discourse`) and g3 (`lexical`): pure
    functions of `(atoms, cfg)`, no llm, no dependency on any other
    generator's output."""

    def __call__(self, atoms: list[Atom], cfg: Config) -> list[Candidate]: ...


def build_candidate(
    start_atom: Atom, end_atom: Atom, generator: str, *, hint: str | None = None
) -> Candidate:
    """build a `Candidate` spanning `[start_atom, end_atom]` (inclusive),
    proposed by `generator`. `id` is deterministic
    (`f"{generator}_{start_atom.id}_{end_atom.id}"`) so re-running a
    generator against unchanged atoms reproduces identical ids -- useful for
    debugging/diffing `candidates.json` across runs. `pre_score` is always
    `0.0`: see this module's docstring on why no generator computes a real
    one.
    """
    return Candidate(
        id=f"{generator}_{start_atom.id}_{end_atom.id}",
        atom_start=start_atom.id,
        atom_end=end_atom.id,
        start_ms=start_atom.start_ms,
        end_ms=end_atom.end_ms,
        generators=[generator],
        generator_hint=hint,
        pre_score=0.0,
    )


def min_duration_window_end(atoms: list[Atom], start_index: int, min_duration_ms: int) -> Atom | None:
    """first atom (searching forward from `atoms[start_index]`) whose
    `end_ms` makes the window `[atoms[start_index].start_ms,
    that atom's end_ms]` reach at least `min_duration_ms`. returns `None` if
    the window never reaches `min_duration_ms` before the atom list runs out
    (e.g. a short tail with nothing left to extend into)."""
    start_ms = atoms[start_index].start_ms
    for candidate_atom in atoms[start_index:]:
        if candidate_atom.end_ms - start_ms >= min_duration_ms:
            return candidate_atom
    return None


def max_duration_window_end(atoms: list[Atom], start_index: int, max_duration_ms: int) -> Atom | None:
    """last atom (searching forward from `atoms[start_index]`) that can
    still be included in a window starting at `atoms[start_index]` without
    exceeding `max_duration_ms` -- i.e. greedily extend the window as far as
    it fits. returns `None` if even the single starting atom alone already
    exceeds `max_duration_ms`."""
    start_ms = atoms[start_index].start_ms
    result: Atom | None = None
    for candidate_atom in atoms[start_index:]:
        if candidate_atom.end_ms - start_ms <= max_duration_ms:
            result = candidate_atom
        else:
            break
    return result
