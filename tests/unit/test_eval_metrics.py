"""tests for `clipper/eval/metrics.py` (spec section 11.2)."""

from __future__ import annotations

from clipper.eval.metrics import (
    Interval,
    boundary_mae_ms,
    dup_rate,
    hit_rate,
    iou,
    recall_pool,
    spread,
    trap_rate,
)


def test_iou_of_identical_intervals_is_one_and_disjoint_is_zero() -> None:
    assert iou(Interval(0, 100), Interval(0, 100)) == 1.0
    assert iou(Interval(0, 100), Interval(200, 300)) == 0.0


def test_hit_rate_counts_only_matches_at_or_above_half_iou() -> None:
    moments = [Interval(0, 100), Interval(1000, 1100)]
    # first: iou 50/150 = 0.33 (miss); second: exact (hit)
    selected = [Interval(50, 150), Interval(1000, 1100)]
    assert hit_rate(moments, selected) == 0.5


def test_hit_rate_is_none_without_annotations_rather_than_zero() -> None:
    assert hit_rate([], [Interval(0, 1)]) is None


def test_recall_pool_uses_the_wider_pool_to_find_what_selection_missed() -> None:
    moments = [Interval(0, 100)]
    assert hit_rate(moments, []) == 0.0
    assert recall_pool(moments, [Interval(0, 100)]) == 1.0


def test_trap_rate_is_fraction_of_traps_selected() -> None:
    traps = [Interval(0, 100), Interval(500, 600)]
    assert trap_rate(traps, [Interval(0, 100)]) == 0.5


def test_boundary_mae_averages_start_and_end_errors_over_hits_only() -> None:
    moments = [Interval(0, 1000), Interval(5000, 6000)]
    selected = [Interval(100, 1200)]  # hit with errors 100 and 200; second moment missed
    assert boundary_mae_ms(moments, selected) == 150.0
    assert boundary_mae_ms(moments, []) is None


def test_dup_rate_counts_similar_pairs() -> None:
    assert dup_rate([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) == 1


def test_spread_is_stddev_of_midpoints_and_needs_two_clips() -> None:
    assert spread([Interval(0, 100)]) is None
    assert spread([Interval(0, 100), Interval(900, 1100)]) == 475.0
