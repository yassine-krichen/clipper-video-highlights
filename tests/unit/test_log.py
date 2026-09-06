import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clipper.util.log import append_event, read_events, read_stats, record_stat


class TestAppendEventAndReadEvents:
    def test_read_events_returns_all_in_order(self, tmp_path: Path) -> None:
        append_event(tmp_path, "transcribe", "started", {"provider": "groq"})
        append_event(tmp_path, "transcribe", "finished", {"wall_ms": 41200})
        append_event(tmp_path, "judge", "started", {"n_candidates": 44})

        events = read_events(tmp_path)

        assert [e["event"] for e in events] == ["started", "finished", "started"]
        assert events[0]["stage"] == "transcribe"
        assert events[0]["data"] == {"provider": "groq"}
        assert all("ts" in e for e in events)

    def test_read_events_filters_by_stage(self, tmp_path: Path) -> None:
        append_event(tmp_path, "transcribe", "started", {})
        append_event(tmp_path, "judge", "started", {})
        append_event(tmp_path, "judge", "finished", {})

        events = read_events(tmp_path, stage="judge")

        assert len(events) == 2
        assert all(e["stage"] == "judge" for e in events)

    def test_read_events_on_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        assert read_events(tmp_path) == []

    def test_log_file_is_jsonl_not_a_json_array(self, tmp_path: Path) -> None:
        append_event(tmp_path, "probe", "started", {})
        append_event(tmp_path, "probe", "finished", {})

        lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()

        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)  # each line must independently parse
            assert isinstance(parsed, dict)

    def test_append_is_safe_across_many_calls(self, tmp_path: Path) -> None:
        for i in range(20):
            append_event(tmp_path, "generate", "candidate", {"i": i})

        events = read_events(tmp_path)
        assert len(events) == 20
        assert [e["data"]["i"] for e in events] == list(range(20))


class TestRecordStatAndReadStats:
    def test_read_stats_on_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        assert read_stats(tmp_path) == {}

    def test_merges_different_sections_and_keys(self, tmp_path: Path) -> None:
        record_stat(tmp_path, "stages", "transcribe", {"wall_ms": 41200, "cost_usd": 0.04})
        record_stat(tmp_path, "stages", "judge", {"wall_ms": 38000, "cost_usd": 0.11})
        record_stat(tmp_path, "candidates", "llm_pass", 31)

        stats = read_stats(tmp_path)

        assert stats["stages"]["transcribe"] == {"wall_ms": 41200, "cost_usd": 0.04}
        assert stats["stages"]["judge"] == {"wall_ms": 38000, "cost_usd": 0.11}
        assert stats["candidates"]["llm_pass"] == 31

    def test_overwriting_same_section_and_key_updates_it(self, tmp_path: Path) -> None:
        record_stat(tmp_path, "candidates", "merged", 240)
        record_stat(tmp_path, "candidates", "merged", 250)

        stats = read_stats(tmp_path)

        assert stats["candidates"]["merged"] == 250

    def test_concurrent_calls_do_not_lose_updates(self, tmp_path: Path) -> None:
        n = 20
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [
                executor.submit(record_stat, tmp_path, "candidates", f"cand_{i}", i)
                for i in range(n)
            ]
            for future in futures:
                future.result()

        stats = read_stats(tmp_path)

        assert set(stats["candidates"].keys()) == {f"cand_{i}" for i in range(n)}
        for i in range(n):
            assert stats["candidates"][f"cand_{i}"] == i
