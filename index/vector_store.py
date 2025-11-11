# index/vector_store.py
"""
This module manages the ChromaDB vector store. It handles the creation,
storage, and querying of document embeddings.
"""
import logging
from pathlib import Path
from typing import List, Dict

import chromadb

from ingest.document_parser import extract_text_from_pdf
from ingest.chunker import chunk_text
from index.embedder import LocalEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChromaVectorStore:
    """Manages the ChromaDB vector store for the RAG system."""

    def __init__(self, db_path: str, collection_name: str = "rag_collection"):
        """
        Initializes the vector store.

        Args:
            db_path (str): Path to the directory where the DB will be persisted.
            collection_name (str): Name of the collection to store vectors in.
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedder = LocalEmbedder()

        try:
            # Use a persistent client to save data to disk
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                # Langchain's Chroma wrapper uses this SentenceTransformerEmbeddingFunction
                # by default, so we match it for compatibility.
                metadata={"hnsw:space": "cosine"} 
            )
            logging.info(f"ChromaDB client initialized at '{db_path}'.")
            logging.info(f"Collection '{self.collection_name}' loaded/created.")
        except Exception as e:
            logging.error(f"Failed to initialize ChromaDB client. Error: {e}")
            raise

    def add_documents(self, chunks: List[Dict[str, any]]):
        """
        Embeds and adds a list of document chunks to the vector store.

        Args:
            chunks (List[Dict[str, any]]): A list of chunk dictionaries.
        """
        if not chunks:
            logging.warning("No chunks provided to add to the vector store.")
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        contents = [chunk["content"] for chunk in chunks]
        metadatas = [{"source": chunk["source"], "page": chunk["page_number"]} for chunk in chunks]

        logging.info(f"Generating embeddings for {len(contents)} chunks...")
        embeddings = self.embedder.embed_documents(contents)

        logging.info(f"Adding {len(ids)} documents to collection '{self.collection_name}'...")
        try:
            self.collection.add(
                embeddings=embeddings.tolist(), # ChromaDB expects lists
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            logging.info("Successfully added documents to the vector store.")
        except Exception as e:
            logging.error(f"Failed to add documents to Chroma. Error: {e}")

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, any]]:
        """
        Queries the vector store for the top-k most similar documents.

        Args:
            query_text (str): The text to search for.
            k (int): The number of results to return.

        Returns:
            List[Dict[str, any]]: A list of result dictionaries.
        """
        logging.info(f"Querying for '{query_text}'...")
        query_embedding = self.embedder.embed_query(query_text).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Reformat the results for easier use
        formatted_results = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted_results.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
        return formatted_results


def main():
    """
    Main function to build and test the vector store. This will:
    1. Process the sample PDF.
    2. Chunk the text.
    3. Add the chunks to a persistent ChromaDB store.
    4. Perform a sample query.
    """
    project_root = Path(__file__).parent.parent
    db_directory = project_root / "index" / "chroma_db"
    sample_pdf_path = project_root / "data" / "sample.pdf"

    # Create and initialize the vector store
    vector_store = ChromaVectorStore(db_path=str(db_directory))

    # --- Ingestion ---
    logging.info("Starting ingestion pipeline...")
    doc_generator = extract_text_from_pdf(sample_pdf_path)
    chunks = chunk_text(doc_generator)
    vector_store.add_documents(chunks)
    logging.info("Ingestion pipeline complete.")

    # --- Querying ---
    logging.info("\n--- Performing a sample query ---")
    query = "What is supervised learning?"
    search_results = vector_store.query(query, k=2)

    if search_results:
        print(f"Found {len(search_results)} results for query: '{query}'")
        for i, result in enumerate(search_results):
            print(f"\n--- Result {i+1} ---")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Source: {result['metadata']['source']}, Page: {result['metadata']['page']}")
            print(f"Content: {result['content'][:300]}...")
    else:
        print("No results found.")

if __name__ == "__main__":
    main()