"""shared timing-only silence proxy (spec section 4.6's `silence_ratio` term,
also needed by the future M10 `refine.py` `silence` penalty).

real audio-energy-based silence detection is `features.json`'s job (spec
section 4.4, phase 2, not built yet). phase 1 approximates it from data
already on hand in `atoms.json`: the inter-atom pause gaps
(`Atom.pause_before_ms`/`pause_after_ms`) computed by the atomize stage. see
DECISIONS.md for why this proxy (rather than deferring the whole term) was
chosen.
"""

from __future__ import annotations

from clipper.types import Atom


def compute_timing_silence_ratio(
    atom_start: int, atom_end: int, atoms_by_id: dict[int, Atom]
) -> float:
    """fraction of `[atom_start, atom_end]`'s (inclusive) total duration that
    is internal inter-atom pause time, as a proxy for `silence_ratio`.

    only pauses strictly BETWEEN two atoms that are both inside the span
    count -- the pause before the first atom and the pause after the last
    atom are external to the span (they lead into/out of it) and are
    deliberately excluded. the gap between atom `i` and atom `i+1` is
    represented on disk twice (`i`'s `pause_after_ms` and `i+1`'s
    `pause_before_ms` describe the same gap) -- summing `pause_before_ms` for
    every atom in the span EXCEPT the first counts each internal gap exactly
    once, with no double counting.

    returns `0.0` if the span resolves to zero duration (defensive; should
    never happen for a real candidate) or if `atom_start > atom_end`.
    clamped to `[0, 1]`.
    """
    if atom_start > atom_end:
        return 0.0

    span_atoms = [atoms_by_id[i] for i in range(atom_start, atom_end + 1) if i in atoms_by_id]
    if not span_atoms:
        return 0.0

    duration_ms = span_atoms[-1].end_ms - span_atoms[0].start_ms
    if duration_ms <= 0:
        return 0.0

    internal_pause_ms = sum(atom.pause_before_ms for atom in span_atoms[1:])

    ratio = internal_pause_ms / duration_ms
    return max(0.0, min(1.0, ratio))
