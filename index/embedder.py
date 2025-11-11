# index/embedder.py
"""
This module handles the creation of text embeddings using a local
sentence-transformer model. It also provides an optional fallback
to use OpenAI's embedding models.
"""
import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LocalEmbedder:
    """A class to handle creating embeddings using a local model."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the embedder by loading the sentence-transformer model.

        Args:
            model_name (str): The name of the model to use.
        """
        try:
            self.model = SentenceTransformer(model_name)
            logging.info(f"Local embedding model '{model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load sentence-transformer model '{model_name}'. Error: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Creates embeddings for a list of documents.

        Args:
            texts (List[str]): A list of text strings to embed.

        Returns:
            np.ndarray: A numpy array of embeddings.
        """
        if not texts:
            return np.array([])
        logging.info(f"Creating embeddings for {len(texts)} documents.")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logging.info(f"Successfully created embeddings of shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """
        Creates an embedding for a single query string.

        Args:
            text (str): The query text to embed.

        Returns:
            np.ndarray: The embedding for the query.
        """
        return self.model.encode(text)

def main():
    """
    Main function to test the embedding functionality.
    It creates embeddings for a few sample sentences.
    """
    logging.info("Starting embedder test...")

    # Initialize the embedder
    embedder = LocalEmbedder()

    # Sample sentences to embed
    sample_texts = [
        "What is Retrieval-Augmented Generation?",
        "RAG combines large language models with external knowledge bases.",
        "The quick brown fox jumps over the lazy dog."
    ]

    # Create embeddings
    embeddings = embedder.embed_documents(sample_texts)

    if embeddings.any():
        print(f"\nSuccessfully generated embeddings.")
        print(f"Shape of the embeddings array: {embeddings.shape}")
        print("Preview of the first embedding vector (first 5 dimensions):")
        print(embeddings[0][:5])
    else:
        print("\nFailed to generate embeddings.")

if __name__ == "__main__":
    main()