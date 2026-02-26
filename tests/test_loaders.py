"""Tests for document loaders."""

import tempfile
from pathlib import Path

import pytest

from thematic_analysis.loaders import (
    LoadedDocument,
    _segment_by_paragraph,
    _segment_by_sentence,
    _segment_fixed_size,
    load_directory,
    load_document,
    load_text_file,
)


class TestSegmentation:
    """Tests for text segmentation methods."""

    def test_segment_by_paragraph(self):
        """Test paragraph-based segmentation."""
        text = """First paragraph with enough words to pass minimum.
        This has more content to meet the threshold.

        Second paragraph also with sufficient content.
        Adding more words here for the test.

        Third short."""

        segments = _segment_by_paragraph(text, "doc1", min_words=5)
        assert len(segments) == 2  # Third is too short
        assert segments[0].segment_id == "doc1_p1"
        assert segments[1].segment_id == "doc1_p2"

    def test_segment_by_sentence(self):
        """Test sentence-based segmentation."""
        text = "First sentence here. Second sentence here. Third sentence here."
        segments = _segment_by_sentence(text, "doc1", min_words=3)
        assert len(segments) >= 1
        assert all(seg.segment_id.startswith("doc1_s") for seg in segments)

    def test_segment_fixed_size(self):
        """Test fixed-size segmentation."""
        words = " ".join(["word"] * 100)
        segments = _segment_fixed_size(words, "doc1", max_words=30)
        assert len(segments) == 4  # 100/30 = 3.33, so 4 chunks
        assert all(seg.segment_id.startswith("doc1_c") for seg in segments)


class TestLoadTextFile:
    """Tests for text file loading."""

    def test_load_text_file(self):
        """Test loading a plain text file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("This is test content.\n\nAnother paragraph here.")
            f.flush()

            doc = load_text_file(f.name)
            assert isinstance(doc, LoadedDocument)
            assert "test content" in doc.text
            assert doc.metadata.word_count > 0
            assert doc.metadata.page_count == 1

    def test_load_text_file_not_found(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_text_file("/nonexistent/file.txt")


class TestLoadDirectory:
    """Tests for directory loading."""

    def test_load_directory_txt_files(self):
        """Test loading text files from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "doc1.txt").write_text("Document one content here.")
            (Path(tmpdir) / "doc2.txt").write_text("Document two content here.")
            (Path(tmpdir) / "ignored.md").write_text("Should be ignored.")

            docs = load_directory(tmpdir, pattern="*.txt")
            assert len(docs) == 2
            assert all(isinstance(d, LoadedDocument) for d in docs)

    def test_load_directory_not_found(self):
        """Test loading non-existent directory raises error."""
        with pytest.raises(FileNotFoundError):
            load_directory("/nonexistent/directory")


class TestLoadDocument:
    """Tests for auto-detecting document type."""

    def test_load_document_txt(self):
        """Test auto-detecting text file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("Test content")
            f.flush()

            doc = load_document(f.name)
            assert isinstance(doc, LoadedDocument)

    def test_load_document_md(self):
        """Test auto-detecting markdown file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write("# Heading\n\nContent here.")
            f.flush()

            doc = load_document(f.name)
            assert isinstance(doc, LoadedDocument)
            assert "Heading" in doc.text


class TestLoadedDocument:
    """Tests for LoadedDocument methods."""

    def test_segment_method(self):
        """Test the segment method on LoadedDocument."""
        from thematic_analysis.loaders import DocumentMetadata

        doc = LoadedDocument(
            text="First paragraph here.\n\nSecond paragraph here.",
            metadata=DocumentMetadata(
                filename="test",
                filepath="/test.txt",
                page_count=1,
                word_count=6,
                char_count=50,
            ),
        )

        segments = doc.segment(method="paragraph", min_words=2)
        assert len(segments) == 2
        assert doc.segments == segments  # Should be stored

    def test_segment_invalid_method(self):
        """Test that invalid segmentation method raises error."""
        from thematic_analysis.loaders import DocumentMetadata

        doc = LoadedDocument(
            text="Some text here.",
            metadata=DocumentMetadata(
                filename="test",
                filepath="/test.txt",
            ),
        )

        with pytest.raises(ValueError, match="Unknown segmentation method"):
            doc.segment(method="invalid")
