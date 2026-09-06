"""g4 `coverage` generator (spec section 4.5): deterministic safety net, no
llm. runs *after* g1-g3 and depends on their combined output -- its
signature is `(atoms, cfg, existing) -> list[Candidate]`, not the
`DeterministicGenerator` shape g2/g3 share (see `clipper/generators/
base.py`'s module docstring).

finds every atom not covered by any candidate in `existing`, and emits a
`min_duration_ms` window for it.

gap handling: rather than building explicit "coalesce adjacent uncovered
atoms into one run" logic, this walks atoms in order and greedily marks
every atom inside each newly-emitted window as covered before moving on.
because a window is at least `min_duration_ms` long, this naturally
coalesces adjacent uncovered atoms into as few candidates as needed to
cover a gap (one window per `min_duration_ms`-sized chunk of the gap, not
one per atom), without a separate contiguous-run-detection pass. see
DECISIONS.md.
"""

from __future__ import annotations

from clipper.config import Config
from clipper.generators.base import build_candidate, min_duration_window_end
from clipper.types import Atom, Candidate


def generate_coverage(atoms: list[Atom], cfg: Config, existing: list[Candidate]) -> list[Candidate]:
    """g4: see module docstring."""
    covered_atom_ids: set[int] = set()
    for candidate in existing:
        covered_atom_ids.update(range(candidate.atom_start, candidate.atom_end + 1))

    candidates: list[Candidate] = []
    for index, atom in enumerate(atoms):
        if atom.id in covered_atom_ids:
            continue

        end_atom = min_duration_window_end(atoms, index, cfg.min_duration_ms)
        if end_atom is None:
            # a trailing gap too short to ever reach min_duration_ms (e.g.
            # a handful of atoms right at the end of the transcript) --
            # covering it would need a window shorter than min_duration_ms,
            # which `clipper/stages/generate.py`'s normalization would drop
            # anyway. left uncovered rather than emitting a doomed
            # candidate; not a break, since a later uncovered atom closer to
            # the end could theoretically still be covered by other
            # generators' candidates even though this one can't reach
            # min_duration_ms on its own.
            continue

        candidates.append(build_candidate(atom, end_atom, "coverage"))
        covered_atom_ids.update(range(atom.id, end_atom.id + 1))

    return candidates
