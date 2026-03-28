import pytest
from knwler.chunking import get_encoder, chunk_text
from knwler.config import Config
from knwler.models import Chunk

"""
Tests for text chunking utilities.
"""


class TestGetEncoder:
    """Tests for get_encoder function."""

    def test_get_encoder_returns_encoding(self):
        """Test that get_encoder returns a tiktoken Encoding object."""
        encoder = get_encoder()
        assert encoder is not None

    def test_get_encoder_caches_result(self):
        """Test that get_encoder returns the same cached instance."""
        encoder1 = get_encoder()
        encoder2 = get_encoder()
        assert encoder1 is encoder2


class TestChunkText:
    """Tests for chunk_text function."""

    def test_chunk_text_short_text(self):
        """Test that short text is not chunked."""
        config = Config(max_tokens=1000, overlap_tokens=50)
        text = "This is a short text."
        chunks = chunk_text(text, config)
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].text == text
        assert chunks[0].chunk_idx == 0

    def test_chunk_text_long_text(self):
        """Test that long text is split into multiple chunks."""
        config = Config(max_tokens=100, overlap_tokens=10)
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, config)
        assert len(chunks) > 1

    def test_chunk_text_respects_max_tokens(self):
        """Test that each chunk respects max_tokens limit."""
        config = Config(max_tokens=100, overlap_tokens=10)
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, config)
        encoder = get_encoder()
        for chunk in chunks:
            tokens = encoder.encode(chunk.text)
            assert len(tokens) <= config.max_tokens + 1  # Allow small margin

    def test_chunk_text_sentence_boundary(self):
        """Test that chunks break at sentence boundaries when possible."""
        config = Config(max_tokens=50, overlap_tokens=5)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text(text, config)
        for chunk in chunks:
            assert chunk.text.rstrip().endswith((".", "!", "?")) or len(chunks) == 1

    def test_chunk_text_with_overlap(self):
        """Test that chunks have overlap when specified."""
        config = Config(max_tokens=100, overlap_tokens=20)
        text = " ".join(["word"] * 500)
        chunks = chunk_text(text, config)
        if len(chunks) > 1:
            encoder = get_encoder()
            first_chunk_tokens = encoder.encode(chunks[0].text)
            second_chunk_tokens = encoder.encode(chunks[1].text)
            # Second chunk should start before the end of the first chunk
            assert len(first_chunk_tokens) + len(second_chunk_tokens) < len(
                encoder.encode(text)
            )

    def test_chunk_text_empty_string(self):
        """Test chunking empty string."""
        config = Config(max_tokens=100, overlap_tokens=10)
        text = ""
        chunks = chunk_text(text, config)
        assert chunks == []
        assert chunk_text("   ", config) == []
        assert chunk_text("\n\n", config) == []
        assert chunk_text("\t\t", config) == []
        assert chunk_text(None, config) == []

    def test_chunk_text_preserves_content(self):
        """Test that chunking preserves the original text."""
        config = Config(max_tokens=100, overlap_tokens=10)
        text = "This is a longer test. " * 50
        chunks = chunk_text(text, config)
        # Verify all chunks are substrings of the original text
        for chunk in chunks:
            assert chunk.text in text or chunk.text == ""
        # Verify the first and last chunks match the text boundaries
        assert text.startswith(chunks[0].text)
        assert text.endswith(chunks[-1].text)

        def test_chunk_text_no_sentence_breaks(self):
            """Test chunking text without sentence boundaries."""
            config = Config(max_tokens=50, overlap_tokens=5)
            text = "a" * 500  # No spaces or punctuation
            chunks = chunk_text(text, config)
            assert len(chunks) > 1
            encoder = get_encoder()
            for chunk in chunks:
                assert len(encoder.encode(chunk.text)) <= config.max_tokens + 1

        def test_chunk_text_multiple_punctuation_marks(self):
            """Test chunking with various punctuation marks."""
            config = Config(max_tokens=50, overlap_tokens=5)
            text = "Question? Answer! Statement. " * 30
            chunks = chunk_text(text, config)
            for chunk in chunks:
                if chunk.text.rstrip():
                    assert (
                        chunk.text.rstrip().endswith((".", "!", "?"))
                        or len(chunks) == 1
                    )

        def test_chunk_text_config_zero_overlap(self):
            """Test chunking with zero overlap tokens."""
            config = Config(max_tokens=100, overlap_tokens=0)
            text = " ".join(["word"] * 500)
            chunks = chunk_text(text, config)
            assert len(chunks) > 1
            encoder = get_encoder()
            total_tokens = sum(len(encoder.encode(chunk.text)) for chunk in chunks)
            assert (
                total_tokens <= len(encoder.encode(text)) + 10
            )  # Small margin for rounding

        def test_chunk_text_large_overlap(self):
            """Test chunking with overlap larger than max_tokens."""
            config = Config(max_tokens=50, overlap_tokens=40)
            text = " ".join(["word"] * 500)
            chunks = chunk_text(text, config)
            assert len(chunks) > 1

        def test_chunk_text_single_word_repeated(self):
            """Test chunking text with single word repeated."""
            config = Config(max_tokens=10, overlap_tokens=2)
            text = "test " * 100
            chunks = chunk_text(text, config)
            assert len(chunks) > 1
            encoder = get_encoder()
            for chunk in chunks:
                assert len(encoder.encode(chunk.text)) <= config.max_tokens + 1
