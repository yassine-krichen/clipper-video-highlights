from clipper.util.timing import format_ms_as_mmss, new_run_id


class TestFormatMsAsMmss:
    def test_zero_ms_is_00_00(self) -> None:
        assert format_ms_as_mmss(0) == "00:00"

    def test_65_seconds_is_01_05(self) -> None:
        assert format_ms_as_mmss(65000) == "01:05"

    def test_spec_example_2477200_is_41_17(self) -> None:
        assert format_ms_as_mmss(2477200) == "41:17"

    def test_over_an_hour_does_not_roll_over_into_hh_mm_ss(self) -> None:
        # documented choice: minutes are never capped/rolled into hours.
        assert format_ms_as_mmss(3661000) == "61:01"


class TestNewRunId:
    def test_contains_timestamp_prefix_and_source_slug(self) -> None:
        run_id = new_run_id("episode_17.mp4")
        assert run_id
        assert "_" in run_id
        timestamp_part, slug_part = run_id.split("_", 1)
        assert len(timestamp_part) == len("2026-09-05T14-02-11")
        assert slug_part == "episode17"

    def test_strips_extension_and_lowercases(self) -> None:
        run_id = new_run_id("Ep17-FINAL.MP4")
        assert run_id.endswith("ep17final")

    def test_two_calls_do_not_collide(self) -> None:
        first = new_run_id("ep17.mp4")
        second = new_run_id("ep18.mp4")
        assert first != second
