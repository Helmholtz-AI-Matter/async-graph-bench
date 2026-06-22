import pytest
from async_graph_bench.utils.visualize_graph import split_after_words


class TestSplitAfterWords:
    def test_short_text(self):
        assert split_after_words("hello", limit=30) == ["hello"]

    def test_exactly_at_limit(self):
        assert split_after_words("123456789012345678901234567890", limit=30) == ["123456789012345678901234567890"]

    def test_split_on_space(self):
        result = split_after_words("hello world foo bar", limit=11)
        assert result == ["hello world", "foo bar"]

    def test_long_word_no_space(self):
        result = split_after_words("superlongword", limit=5)
        assert result == ["superlongword"]

    def test_multiple_words(self):
        result = split_after_words(
            "Lorem ipsum dolor sit amet",
            limit=10,
        )
        assert result == [
            "Lorem ipsum",
            "dolor sit amet",
        ]

    def test_empty_string(self):
        assert split_after_words("", limit=10) == []

    def test_whitespace_only(self):
        assert split_after_words("   ", limit=10) == []

    def test_leading_trailing_whitespace(self):
        result = split_after_words("  hello world  ", limit=5)
        assert result == ["hello", "world"]

    def test_repeated_splitting(self):
        text = "one two three four five six seven eight"
        result = split_after_words(text, limit=8)
        assert result == ["one two three", "four five", "six seven", "eight"]

    def test_line_breaks_in_text(self):
        result = split_after_words("hello\nworld", limit=30)
        assert len(result) >= 1