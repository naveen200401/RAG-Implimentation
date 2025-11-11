# ingest/chunker.py
"""
A module for splitting text from documents into manageable chunks.
This uses a recursive character splitter for semantic chunking.
"""
import logging
from typing import List, Dict, Generator

from langchain.text_splitter import RecursiveCharacterTextSplitter

from ingest.document_parser import extract_text_from_pdf, Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def chunk_text(documents: Generator[Dict[str, any], None, None],
               chunk_size: int = 1000,
               chunk_overlap: int = 200) -> List[Dict[str, any]]:
    """
    Splits the text content of documents into smaller chunks.

    Args:
        documents (Generator[Dict[str, any], None, None]): A generator of documents, 
            where each document is a dictionary with 'content', 'source', etc.
        chunk_size (int): The maximum size of each chunk (in characters).
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Dict[str, any]]: A list of chunks, where each chunk is a
        dictionary containing the chunked content and original metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True, # This helps in locating the chunk in the original doc
    )

    chunks_with_metadata = []
    for doc in documents:
        # Split the content of the document
        chunks = text_splitter.split_text(doc["content"])

        # Add metadata to each chunk
        for i, chunk_text in enumerate(chunks):
            chunks_with_metadata.append({
                "content": chunk_text,
                "source": doc["source"],
                "page_number": doc["page_number"],
                "chunk_id": f"{Path(doc['source']).stem}_p{doc['page_number']}_c{i+1}"
            })

    logging.info(f"Created {len(chunks_with_metadata)} chunks.")
    return chunks_with_metadata

def main():
    """
    Main function to test the chunking functionality. It processes the
    sample PDF from the 'data' directory and prints the first few chunks.
    """
    project_root = Path(__file__).parent.parent
    sample_pdf_path = project_root / "data" / "sample.pdf"

    if not sample_pdf_path.exists():
        logging.error(f"Sample file not found at {sample_pdf_path}. Please add it.")
        return

    logging.info("Starting chunking process test...")
    # 1. Extract text from PDF
    doc_generator = extract_text_from_pdf(sample_pdf_path)

    # 2. Chunk the extracted text
    chunks = chunk_text(doc_generator)

    if chunks:
        logging.info(f"Successfully created {len(chunks)} chunks.")
        # Print details of the first 3 chunks for verification
        for chunk in chunks[:3]:
            print("\n--- Chunk ---")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page_number']}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Content Preview: {chunk['content'][:250]}...")
            print("-------------")
    else:
        logging.warning("No chunks were created. Check the source document.")

if __name__ == "__main__":
    main()