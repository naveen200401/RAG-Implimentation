# tests/test_vector_store.py
"""
Unit tests for the ChromaDB vector store functionality.
"""
import sys
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from index.vector_store import ChromaVectorStore

@pytest.fixture
def temp_db():
    """Creates a temporary directory for a test ChromaDB and cleans up afterward."""
    # Create a temporary directory
    db_dir = tempfile.mkdtemp()
    yield db_dir
    # Clean up the directory after the test
    shutil.rmtree(db_dir)

def test_add_and_query(temp_db):
    """
    Tests adding documents to the vector store and querying them.
    """
    # Initialize vector store in the temporary directory
    vector_store = ChromaVectorStore(db_path=temp_db)

    # Create dummy chunks to add
    dummy_chunks = [
        {"chunk_id": "doc1_c1", "content": "The sky is blue.", "source": "doc1.pdf", "page_number": 1},
        {"chunk_id": "doc1_c2", "content": "The grass is green.", "source": "doc1.pdf", "page_number": 1}
    ]

    # Add documents
    vector_store.add_documents(dummy_chunks)

    # Assert that the number of items in the collection is correct
    assert vector_store.collection.count() == 2

    # Query for a similar sentence
    query_text = "What color is the sky?"
    results = vector_store.query(query_text, k=1)

    # Assert that we got results
    assert len(results) == 1

    # Assert that the most relevant result is the correct one
    assert "The sky is blue" in results[0]["content"]