"""atomize stage (spec section 4.3): `run_dir/transcript.json` ->
`run_dir/atoms.json`.

turns the flat provider word list into `Atom[]`, deterministically, no llm.
built as an explicit pipeline of small pure functions, run in this order,
as a single straight-line pass:

    split_into_atoms -> merge_short_atoms -> split_long_atoms
    -> compute_pauses -> _assign_ids

each function takes and returns `list[Atom]` (or, for the first step, builds
atoms directly from `list[Word]`), so every step is independently testable.
`merge_short_atoms` is internally a fixed-point loop (merging can turn a
neighbour into something that's now itself short). `split_long_atoms` is
also internally a fixed-point loop per atom, but its split-point selection
is itself constrained to never produce a new unprotected short (<3-word)
fragment (see `_select_split_index`), so it can never hand a fresh
short-atom problem back to `merge_short_atoms` -- meaning the pipeline does
not need to alternate the two stages, and there is nothing to run to a
fixed point *across* them. an earlier version did alternate the two with
cycle detection to cope with `split_long_atoms` creating unprotected
fragments; that approach let a single split-created fragment's backward
merge cascade into arbitrarily much unrelated, already-compliant preceding
content across repeated rounds. see DECISIONS.md for the full history and
why the constrained split-point selection below is the fix.

no diarization exists in phase 1 (`Word`/`Atom.speaker` is always `None`), so
the spec's "on speaker change when diarization exists" split condition is
intentionally unimplemented here -- `split_into_atoms`'s boundary check has
no speaker branch, since with `speaker` always `None` on every word it could
never fire anyway. revisit when diarization (phase 4) lands.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from clipper.config import Config
from clipper.types import Atom, Transcript, Word
from clipper.util.log import append_event, record_stat

_STAGE = "atomize"
_TRANSCRIPT_JSON_FILENAME = "transcript.json"
_ATOMS_JSON_FILENAME = "atoms.json"

_PAUSE_SPLIT_MS = 700  # spec: "any pause >= 700ms between words"
_MIN_ATOM_WORDS = 3  # spec: "atoms shorter than 3 words"
_MAX_ATOM_MS = 25_000  # spec: "an atom longer than 25s"

# sentence-final punctuation (spec: "split on sentence-final punctuation
# . ? !"). trailing closing quotes/parens after the punctuation (e.g.
# `"Really?"`, `(fine.)`) are stripped first so those still count -- a
# reasonable, minimal robustness allowance, not full sentence-boundary
# detection. see DECISIONS.md.
_SENTENCE_FINAL_CHARS = (".", "?", "!")
_TRAILING_WRAPPER_CHARS = "\"'”’)]}"


def _ends_sentence(word_text: str) -> bool:
    """true if `word_text` ends in sentence-final punctuation, ignoring any
    trailing closing quote/paren/bracket characters."""
    stripped = word_text.rstrip(_TRAILING_WRAPPER_CHARS)
    return stripped.endswith(_SENTENCE_FINAL_CHARS)


def _atom_text(words: list[Word]) -> str:
    return " ".join(w.text for w in words)


def _make_atom(atom_id: int, words: list[Word]) -> Atom:
    """build an `Atom` from its words, with pauses left at 0 -- `id` here is
    provisional and always overwritten by `_assign_ids` at the end of the
    pipeline, since merges/splits change atom boundaries after this point."""
    return Atom(
        id=atom_id,
        start_ms=words[0].start_ms,
        end_ms=words[-1].end_ms,
        text=_atom_text(words),
        words=list(words),
        # `Word` has no `speaker` field in phase 1 (no diarization yet, spec
        # section 12); `Atom.speaker` is always `None` until diarization
        # lands in phase 4.
        speaker=None,
        pause_before_ms=0,
        pause_after_ms=0,
    )


def split_into_atoms(words: list[Word]) -> list[Atom]:
    """spec step 1: walk `words`, starting a new atom boundary whenever the
    current word ends a sentence, or the gap to the next word is
    `>= _PAUSE_SPLIT_MS`. (speaker-change splitting is a no-op in phase 1;
    see module docstring.) returns provisional atoms with `id`s that will be
    replaced by `_assign_ids` later in the pipeline.
    """
    if not words:
        return []

    atoms: list[Atom] = []
    current: list[Word] = []

    for i, word in enumerate(words):
        current.append(word)

        is_last_word = i == len(words) - 1
        ends_here = is_last_word or _ends_sentence(word.text)
        if not ends_here and not is_last_word:
            gap_ms = words[i + 1].start_ms - word.end_ms
            ends_here = gap_ms >= _PAUSE_SPLIT_MS

        if ends_here:
            atoms.append(_make_atom(len(atoms), current))
            current = []

    return atoms


def _pause_between(first: Atom, second: Atom) -> int:
    """gap between `first`'s last word and `second`'s first word, in ms."""
    return second.words[0].start_ms - first.words[-1].end_ms


def merge_short_atoms(atoms: list[Atom]) -> list[Atom]:
    """spec step 2: any atom with fewer than `_MIN_ATOM_WORDS` words merges
    into the *following* atom, unless a pause `>= _PAUSE_SPLIT_MS` separates
    them (in which case it stays separate despite being short).

    runs to a fixed point: merging can turn a short atom's neighbour into an
    atom that's now itself short (or that now has a new following neighbour
    to check against), so a single left-to-right pass isn't guaranteed to
    settle everything in one go. see DECISIONS.md.

    edge case: if the *last* atom in the list is short, there is no
    following atom to merge into. the documented judgment call is to merge
    it backward into the previous atom instead, respecting the same
    `>=700ms` pause exception as every forward merge -- see DECISIONS.md.
    """
    if len(atoms) <= 1:
        return list(atoms)

    changed = True
    current = list(atoms)

    while changed:
        changed = False
        merged: list[Atom] = []
        i = 0
        while i < len(current):
            atom = current[i]
            is_short = len(atom.words) < _MIN_ATOM_WORDS
            has_next = i + 1 < len(current)

            if is_short and has_next and _pause_between(atom, current[i + 1]) < _PAUSE_SPLIT_MS:
                combined_words = atom.words + current[i + 1].words
                merged.append(_make_atom(0, combined_words))
                i += 2
                changed = True
                continue

            if is_short and not has_next and merged and _pause_between(merged[-1], atom) < _PAUSE_SPLIT_MS:
                # last atom overall is short with nothing to merge forward
                # into: merge it backward into the previous (already
                # finalized) atom instead -- unless a >=700ms pause
                # separates them, matching the forward-merge exception.
                previous = merged.pop()
                combined_words = previous.words + atom.words
                merged.append(_make_atom(0, combined_words))
                i += 1
                changed = True
                continue

            merged.append(atom)
            i += 1

        current = merged

    return current


def _creates_unprotected_short_child(words: list[Word], split_index: int) -> bool:
    """true if splitting `words` right after index `split_index` would leave
    either resulting child with fewer than `_MIN_ATOM_WORDS` words, *and*
    the split gap itself is under `_PAUSE_SPLIT_MS` -- i.e. that short child
    would not qualify for the same `>=700ms` pause-protection exception
    `merge_short_atoms` grants short atoms elsewhere. a short child sitting
    behind a genuine `>=700ms` pause is fine and not rejected."""
    left_len = split_index + 1
    right_len = len(words) - left_len
    has_short_child = left_len < _MIN_ATOM_WORDS or right_len < _MIN_ATOM_WORDS
    if not has_short_child:
        return False
    gap_ms = words[split_index + 1].start_ms - words[split_index].end_ms
    return gap_ms < _PAUSE_SPLIT_MS


def _select_split_index(words: list[Word]) -> int | None:
    """choose where to split `words` for `_MAX_ATOM_MS` purposes: the
    *largest* internal gap that does not create a new unprotected short
    fragment (see `_creates_unprotected_short_child`). candidate gaps are
    tried largest-first; a candidate is skipped (not just rejected outright)
    if it would create an unprotected short child, and the next-largest gap
    is tried instead. returns `None` if `words` has fewer than 2 words (no
    internal gap exists at all), or if every internal gap would create an
    unprotected short fragment -- in either case there is no valid split
    point for this atom.

    this is the fix for the round-2 regression: previously, split point
    selection only looked at gap size, so it could freely carve off an
    unprotected short fragment and hand the problem to `merge_short_atoms`,
    which had no way to know the fragment was scoped to just this one split
    decision -- it would merge into whatever neighbour was globally
    adjacent, which repeated across merge/split rounds could absorb
    arbitrarily much unrelated, already-compliant content. making split
    selection itself respect the word-count/pause-protection rule means
    `split_long_atoms` never creates that problem in the first place, so
    there's nothing left for a merge pass to clean up. see DECISIONS.md.
    """
    if len(words) < 2:
        return None

    candidate_indices = sorted(
        range(len(words) - 1),
        key=lambda i: words[i + 1].start_ms - words[i].end_ms,
        reverse=True,
    )
    for index in candidate_indices:
        if not _creates_unprotected_short_child(words, index):
            return index
    return None


def _split_atom_to_fixed_point(atom: Atom) -> list[Atom]:
    """split a single atom at the largest internal pause that doesn't create
    a new unprotected short fragment (`_select_split_index`), then recurse
    into *each* resulting half and re-check it against `_MAX_ATOM_MS` again
    -- splitting at a single gap only guarantees neither half has a bigger
    *qualifying* internal gap than the one just used, not that either half
    is actually under the limit, so one split is not always enough."""
    duration_ms = atom.end_ms - atom.start_ms
    if duration_ms <= _MAX_ATOM_MS:
        return [atom]

    split_index = _select_split_index(atom.words)
    if split_index is None:
        # no valid split point: either no internal pause exists at all, or
        # every internal gap would create an unprotected short fragment.
        # terminal case -- leave the atom as a single, local, well-scoped
        # exception to the 25s limit, over the limit though it is, rather
        # than force a split that would just create a different rule
        # violation, or hand the fragment off to a merge pass that could
        # absorb unrelated neighbouring content. see DECISIONS.md.
        return [atom]

    left_words = atom.words[: split_index + 1]
    right_words = atom.words[split_index + 1 :]
    left_atom = _make_atom(0, left_words)
    right_atom = _make_atom(0, right_words)
    return _split_atom_to_fixed_point(left_atom) + _split_atom_to_fixed_point(right_atom)


def split_long_atoms(atoms: list[Atom]) -> list[Atom]:
    """spec step 3: any atom whose duration exceeds `_MAX_ATOM_MS` is split
    at its largest internal pause -- but only among gaps that don't create a
    new unprotected short (<3-word) fragment (`_select_split_index`). if
    every candidate gap would create one, or there's no internal pause at
    all (every word is back-to-back, e.g. a single very long word or run of
    zero-gap words), the atom is left as a single, local, over-the-limit
    exception rather than forced into a split that would just trade one
    rule violation for another. see DECISIONS.md.

    runs each atom to a fixed point (`_split_atom_to_fixed_point`): a single
    split only guarantees neither half has a larger *qualifying* internal
    gap than the one just used to split, not that either half is now under
    the limit, so a half that's still too long is recursively re-split using
    its own internal gaps until every resulting piece is under the limit or
    hits the terminal no-valid-split-point case. because split-point
    selection itself is constrained to never create a new unprotected short
    fragment, this function never hands `merge_short_atoms` a fresh problem
    to solve -- so the pipeline does not need to re-run `merge_short_atoms`
    afterward. see DECISIONS.md.
    """
    result: list[Atom] = []
    for atom in atoms:
        result.extend(_split_atom_to_fixed_point(atom))
    return result


def compute_pauses(atoms: list[Atom]) -> list[Atom]:
    """spec step 4: sets `pause_before_ms`/`pause_after_ms` on every atom
    from neighbouring word timings. first atom's `pause_before_ms` and last
    atom's `pause_after_ms` are 0."""
    if not atoms:
        return []

    updated: list[Atom] = []
    for i, atom in enumerate(atoms):
        pause_before = 0 if i == 0 else _pause_between(atoms[i - 1], atom)
        pause_after = 0 if i == len(atoms) - 1 else _pause_between(atom, atoms[i + 1])
        updated.append(
            atom.model_copy(update={"pause_before_ms": pause_before, "pause_after_ms": pause_after})
        )

    return updated


def _assign_ids(atoms: list[Atom]) -> list[Atom]:
    """spec step 5: final `id`s, `0..n-1` in order -- assigned last since
    every prior step used provisional ids that don't survive merges/splits."""
    return [atom.model_copy(update={"id": i}) for i, atom in enumerate(atoms)]


def atomize_words(words: list[Word]) -> list[Atom]:
    """the full pure pipeline (spec section 4.3), independent of file i/o --
    a single straight-line pass: `split_into_atoms -> merge_short_atoms ->
    split_long_atoms -> compute_pauses -> _assign_ids`.

    no alternation or cycle detection between `merge_short_atoms` and
    `split_long_atoms` is needed: `split_long_atoms`'s split-point selection
    (`_select_split_index`) is itself constrained to never create a new
    unprotected short (<3-word) fragment, so it can never hand
    `merge_short_atoms` a fresh problem that would require re-running it.
    an earlier version alternated the two stages in a loop with cycle
    detection to cope with `split_long_atoms` creating such fragments; that
    approach is what let a single split-created fragment's backward merge
    cascade into arbitrarily much unrelated, already-compliant preceding
    content across repeated rounds. see DECISIONS.md.
    """
    atoms = split_into_atoms(words)
    atoms = merge_short_atoms(atoms)
    atoms = split_long_atoms(atoms)
    atoms = compute_pauses(atoms)
    return _assign_ids(atoms)


def run_atomize(run_dir: Path, cfg: Config) -> None:
    """atomize `run_dir/transcript.json` and write `run_dir/atoms.json`
    (spec section 4.3)."""
    append_event(run_dir, _STAGE, "started", {})

    transcript = Transcript.model_validate_json(
        (run_dir / _TRANSCRIPT_JSON_FILENAME).read_text(encoding="utf-8")
    )

    wall_start = time.monotonic()
    atoms = atomize_words(transcript.words)
    wall_ms = round((time.monotonic() - wall_start) * 1000)

    (run_dir / _ATOMS_JSON_FILENAME).write_text(
        json.dumps([atom.model_dump() for atom in atoms], indent=2)
    )

    append_event(
        run_dir,
        _STAGE,
        "atomized",
        {"atom_count": len(atoms), "word_count": len(transcript.words)},
    )

    record_stat(
        run_dir,
        "stages",
        _STAGE,
        {"wall_ms": wall_ms, "atom_count": len(atoms)},
    )
