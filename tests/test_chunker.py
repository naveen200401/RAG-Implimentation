# tests/test_chunker.py
"""
Unit tests for the text chunking functionality.
"""
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from 'ingest'
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingest.chunker import chunk_text

def test_chunk_text():
    """
    Tests that the chunk_text function correctly splits a document
    and assigns metadata.
    """
    # Create a generator with a single dummy document
    dummy_documents = (
        {
            "content": "This is a test sentence. " * 100, # Long text to force chunking
            "source": "dummy_source.pdf",
            "page_number": 1
        } for i in range(1)
    )

    chunks = chunk_text(dummy_documents, chunk_size=150, chunk_overlap=30)

    # Assert that chunks were created
    assert len(chunks) > 1, "Should have created more than one chunk"

    # Assert that each chunk has the correct metadata
    first_chunk = chunks[0]
    assert "content" in first_chunk
    assert "source" in first_chunk
    assert "page_number" in first_chunk
    assert "chunk_id" in first_chunk
    assert first_chunk["source"] == "dummy_source.pdf"
    assert first_chunk["page_number"] == 1
    assert first_chunk["chunk_id"] == "dummy_source_p1_c1"

    # Assert that the content is smaller than the chunk size
    assert len(first_chunk["content"]) <= 150