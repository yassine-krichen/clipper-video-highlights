from pathlib import Path

from clipper.util.hashing import sha256_file, sha256_llm_cache_key


class TestSha256File:
    def test_deterministic_for_same_content(self, tmp_path: Path) -> None:
        path = tmp_path / "a.txt"
        path.write_bytes(b"hello world")
        assert sha256_file(path) == sha256_file(path)

    def test_sensitive_to_content(self, tmp_path: Path) -> None:
        path_a = tmp_path / "a.txt"
        path_b = tmp_path / "b.txt"
        path_a.write_bytes(b"hello world")
        path_b.write_bytes(b"hello there")

        assert sha256_file(path_a) != sha256_file(path_b)

    def test_matches_known_digest(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")
        # sha256 of the empty string, per any standard implementation.
        assert sha256_file(path) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class TestSha256LlmCacheKey:
    def test_deterministic(self) -> None:
        args = ("gemini-flash", "judge_v1", "some transcript text")
        assert sha256_llm_cache_key(*args) == sha256_llm_cache_key(*args)

    def test_sensitive_to_model(self) -> None:
        base = sha256_llm_cache_key("gemini-flash", "judge_v1", "input")
        changed = sha256_llm_cache_key("gemini-pro", "judge_v1", "input")
        assert base != changed

    def test_sensitive_to_prompt_version(self) -> None:
        base = sha256_llm_cache_key("gemini-flash", "judge_v1", "input")
        changed = sha256_llm_cache_key("gemini-flash", "judge_v2", "input")
        assert base != changed

    def test_sensitive_to_input_payload(self) -> None:
        base = sha256_llm_cache_key("gemini-flash", "judge_v1", "input")
        changed = sha256_llm_cache_key("gemini-flash", "judge_v1", "other input")
        assert base != changed

    def test_separator_prevents_concatenation_collision(self) -> None:
        # naive "model + prompt_version + input" concatenation would make
        # these two calls collide: ("ab","c","x") -> "abcx" == ("a","bc","x").
        first = sha256_llm_cache_key("ab", "c", "x")
        second = sha256_llm_cache_key("a", "bc", "x")
        assert first != second
