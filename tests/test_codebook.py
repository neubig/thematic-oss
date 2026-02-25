"""Tests for Codebook."""

import json
import tempfile
from pathlib import Path

import pytest

from thematic_lm.codebook import Codebook, CodeEntry, Quote


class TestQuote:
    """Test cases for Quote dataclass."""

    def test_quote_creation(self):
        """Test creating a Quote."""
        quote = Quote(quote_id="q1", text="sample quote text")

        assert quote.quote_id == "q1"
        assert quote.text == "sample quote text"


class TestCodeEntry:
    """Test cases for CodeEntry dataclass."""

    def test_code_entry_creation(self):
        """Test creating a CodeEntry."""
        quotes = [Quote("q1", "text1"), Quote("q2", "text2")]
        entry = CodeEntry(code="test code", quotes=quotes)

        assert entry.code == "test code"
        assert len(entry.quotes) == 2
        assert entry.embedding is None

    def test_to_dict(self):
        """Test CodeEntry serialization to dict."""
        quotes = [Quote("q1", "text1")]
        entry = CodeEntry(code="test", quotes=quotes)

        result = entry.to_dict()

        assert result == {
            "code": "test",
            "quotes": [{"quote_id": "q1", "text": "text1"}],
        }

    def test_from_dict(self):
        """Test CodeEntry deserialization from dict."""
        data = {
            "code": "test code",
            "quotes": [{"quote_id": "q1", "text": "quote text"}],
        }

        entry = CodeEntry.from_dict(data)

        assert entry.code == "test code"
        assert len(entry.quotes) == 1
        assert entry.quotes[0].quote_id == "q1"


class TestCodebook:
    """Test cases for Codebook."""

    @pytest.fixture
    def codebook(self) -> Codebook:
        """Create a codebook for tests."""
        return Codebook(max_quotes_per_code=5)

    def test_empty_codebook(self, codebook: Codebook):
        """Test empty codebook properties."""
        assert len(codebook) == 0
        assert codebook.codes == []
        assert codebook.entries == []

    def test_add_code(self, codebook: Codebook):
        """Test adding a code to the codebook."""
        quotes = [Quote("q1", "quote text")]
        entry = codebook.add_code("test code", quotes)

        assert len(codebook) == 1
        assert entry.code == "test code"
        assert entry.embedding is not None
        assert codebook.codes == ["test code"]

    def test_add_code_truncates_quotes(self, codebook: Codebook):
        """Test that add_code truncates quotes beyond max."""
        quotes = [Quote(f"q{i}", f"text{i}") for i in range(10)]
        entry = codebook.add_code("test", quotes)

        assert len(entry.quotes) == 5

    def test_add_quotes_to_code(self, codebook: Codebook):
        """Test adding quotes to an existing code."""
        codebook.add_code("test", [Quote("q1", "text1")])
        codebook.add_quotes_to_code(0, [Quote("q2", "text2"), Quote("q3", "text3")])

        assert len(codebook.entries[0].quotes) == 3

    def test_add_quotes_no_duplicates(self, codebook: Codebook):
        """Test that duplicate quote IDs are not added."""
        codebook.add_code("test", [Quote("q1", "text1")])
        codebook.add_quotes_to_code(0, [Quote("q1", "different text")])

        assert len(codebook.entries[0].quotes) == 1

    def test_add_quotes_invalid_index(self, codebook: Codebook):
        """Test adding quotes to invalid index raises error."""
        with pytest.raises(IndexError):
            codebook.add_quotes_to_code(0, [Quote("q1", "text")])

    def test_find_similar_codes(self, codebook: Codebook):
        """Test finding similar codes."""
        codebook.add_code("climate change impacts", [Quote("q1", "t1")])
        codebook.add_code("global warming effects", [Quote("q2", "t2")])
        codebook.add_code("ice cream flavors", [Quote("q3", "t3")])

        results = codebook.find_similar_codes("environmental climate effects", top_k=2)

        assert len(results) == 2
        # Climate-related codes should be more similar
        codes = [r[0].code for r in results]
        assert "ice cream flavors" not in codes

    def test_find_similar_codes_empty_codebook(self, codebook: Codebook):
        """Test finding similar codes in empty codebook."""
        results = codebook.find_similar_codes("test", top_k=5)

        assert results == []

    def test_update_code(self, codebook: Codebook):
        """Test updating a code's text."""
        codebook.add_code("original", [Quote("q1", "text")])
        old_embedding = codebook.entries[0].embedding

        codebook.update_code(0, "updated code")

        assert codebook.entries[0].code == "updated code"
        assert codebook.entries[0].embedding is not None
        assert not (codebook.entries[0].embedding == old_embedding).all()

    def test_update_code_invalid_index(self, codebook: Codebook):
        """Test updating code with invalid index raises error."""
        with pytest.raises(IndexError):
            codebook.update_code(0, "test")

    def test_merge_codes(self, codebook: Codebook):
        """Test merging two codes."""
        codebook.add_code("code1", [Quote("q1", "text1")])
        codebook.add_code("code2", [Quote("q2", "text2")])

        codebook.merge_codes(source_index=0, target_index=1, new_code="merged")

        assert len(codebook) == 1
        assert codebook.entries[0].code == "merged"
        assert len(codebook.entries[0].quotes) == 2

    def test_merge_codes_same_index(self, codebook: Codebook):
        """Test merging code with itself does nothing."""
        codebook.add_code("code1", [Quote("q1", "text1")])

        codebook.merge_codes(source_index=0, target_index=0)

        assert len(codebook) == 1

    def test_merge_codes_invalid_index(self, codebook: Codebook):
        """Test merging with invalid index raises error."""
        codebook.add_code("code1", [Quote("q1", "text1")])

        with pytest.raises(IndexError):
            codebook.merge_codes(source_index=0, target_index=5)

    def test_to_json_and_from_json(self, codebook: Codebook):
        """Test JSON serialization roundtrip."""
        codebook.add_code("code1", [Quote("q1", "text1")])
        codebook.add_code("code2", [Quote("q2", "text2")])

        json_str = codebook.to_json()
        restored = Codebook.from_json(json_str)

        assert len(restored) == 2
        assert restored.codes == ["code1", "code2"]
        assert restored.entries[0].embedding is not None

    def test_to_dict(self, codebook: Codebook):
        """Test to_dict method."""
        codebook.add_code("test", [Quote("q1", "text1")])

        result = codebook.to_dict()

        assert "codes" in result
        assert len(result["codes"]) == 1

    def test_save_and_load(self, codebook: Codebook):
        """Test saving and loading from file."""
        codebook.add_code("test code", [Quote("q1", "quote text")])

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            codebook.save(path)
            loaded = Codebook.load(path)

            assert len(loaded) == 1
            assert loaded.codes == ["test code"]
        finally:
            path.unlink()

    def test_from_json_preserves_quotes(self):
        """Test that from_json preserves quote data."""
        json_data = {
            "codes": [
                {
                    "code": "test",
                    "quotes": [
                        {"quote_id": "q1", "text": "first quote"},
                        {"quote_id": "q2", "text": "second quote"},
                    ],
                }
            ]
        }

        codebook = Codebook.from_json(json.dumps(json_data))

        assert len(codebook.entries[0].quotes) == 2
        assert codebook.entries[0].quotes[0].text == "first quote"
