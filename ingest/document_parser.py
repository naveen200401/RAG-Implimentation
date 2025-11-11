# ingest/document_parser.py
"""
A module for extracting text from various document types, including PDFs.
It handles both text-based PDFs and scanned PDFs using OCR.
"""
import logging
from pathlib import Path
import pypdf
from PIL import Image
import pytesseract
from typing import List, Dict, Generator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text_from_pdf(pdf_path: Path) -> Generator[Dict[str, any], None, None]:
    """
    Extracts text from a PDF file, page by page. It attempts to extract
    text directly and falls back to OCR if no text is found.

    Args:
        pdf_path (Path): The path to the PDF file.

    Yields:
        Generator[Dict[str, any], None, None]: A generator of dictionaries,
        where each dictionary represents a page and contains the page number,

        text, and the source path.
    """
    logging.info(f"Processing PDF: {pdf_path}")
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            page_content = ""
            try:
                page_content = page.extract_text()
            except Exception as e:
                logging.warning(f"Could not extract text directly from page {page_num + 1} in {pdf_path}. Error: {e}")

            if not page_content or page_content.strip() == "":
                logging.info(f"No text found on page {page_num + 1}. Attempting OCR.")
                # Fallback to OCR
                page_content = "" # Reset content
                for image in page.images:
                    try:
                        # Using name attribute which is often a descriptor
                        img = Image.frombytes(
                            mode="RGB", 
                            size=(image.width, image.height), 
                            data=image.data
                        )
                        page_content += pytesseract.image_to_string(img)
                    except Exception as ocr_error:
                        logging.error(f"OCR failed for an image on page {page_num + 1} in {pdf_path}. Error: {ocr_error}")

            if page_content.strip():
                yield {
                    "page_number": page_num + 1,
                    "content": page_content.strip(),
                    "source": str(pdf_path)
                }
            else:
                logging.warning(f"Page {page_num + 1} in {pdf_path} is empty or contains no extractable text.")

    except Exception as e:
        logging.error(f"Failed to read or process PDF {pdf_path}. Error: {e}")


def main():
    """
    Main function to test the document parsing functionality.
    It processes a sample PDF file placed in the 'data' directory.
    """
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    sample_pdf_path = data_dir / "sample.pdf"

    if not data_dir.exists():
        data_dir.mkdir()
        logging.info(f"Created directory: {data_dir}")

    if not sample_pdf_path.exists():
        logging.error(f"Sample file not found at {sample_pdf_path}")
        logging.error("Please create a simple PDF named 'sample.pdf' in the 'data' directory.")
        logging.error("You can create one from a Word document or Google Doc.")
        return

    logging.info("Starting PDF processing test...")
    page_generator = extract_text_from_pdf(sample_pdf_path)
    for page_data in page_generator:
        print("\n--- Page Data ---")
        print(f"Source: {page_data['source']}")
        print(f"Page: {page_data['page_number']}")
        print(f"Content Preview: {page_data['content'][:200]}...") # Print first 200 chars
        print("-----------------")

if __name__ == "__main__":
    main()