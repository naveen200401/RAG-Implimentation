# 📚 RAG Knowledge Base Search Engine

A complete Retrieval-Augmented Generation (RAG) system built with Python. This project ingests PDF documents, indexes them in a persistent ChromaDB vector store, and uses the Google Gemini API to answer questions based on document content. The system features a robust FastAPI backend and an intuitive Streamlit UI for easy interaction.

## ✨ Features

- **Document Ingestion**: Supports both text-based and scanned PDFs using PyPDF and Tesseract for OCR.
- **Semantic Search**: Utilizes local sentence-transformers for creating high-quality embeddings and ChromaDB for persistent, efficient vector storage.
- **Retrieval-Augmented Generation (RAG)**: Synthesizes answers using a Gemini LLM, with citations pointing back to the source documents.
- **General Knowledge Fallback**: Intelligently detects when a question cannot be answered from the documents and uses the LLM's general knowledge, providing a clear disclaimer.
- **Web Interface & API**: A user-friendly Streamlit frontend for demonstrations and a powerful FastAPI backend for programmatic access.

## 📁 Project Structure

```
rag-search-engine/
├── api/                # Contains the FastAPI backend, LLM integration, and prompts.
│   ├── main.py
│   ├── llm_integrator.py
│   └── prompts.py
├── data/               # Default directory for storing uploaded PDFs.
├── ingest/             # Modules for parsing and chunking documents.
│   ├── document_parser.py
│   └── chunker.py
├── index/              # Modules for embeddings and the vector store.
│   ├── embedder.py
│   └── vector_store.py
├── tests/              # Unit tests for the project components.
│   ├── test_chunker.py
│   └── test_vector_store.py
├── ui/                 # Contains the Streamlit frontend application.
│   └── app.py
├── .env                # Stores the API key (must be created manually).
├── .gitignore          # Specifies files and folders for Git to ignore.
├── README.md           # This file.
├── requirements.txt    # Pip package requirements.
└── environment.yml     # Conda environment definition.
```

## 🖥 System Setup (Windows)

### 1. Prerequisites

- **Conda**: Ensure you have Anaconda or Miniconda installed.
- **Tesseract OCR**: Required for processing scanned documents.
  - Download the installer from the [UB Mannheim Tesseract repository](https://github.com/UB-Mannheim/tesseract/wiki).
  - **Important**: During installation, make sure to check the option to "Add Tesseract to system PATH".

### 2. Create Conda Environment

Open an Anaconda Prompt in the project's root directory and run the following commands:

```bash
# Create the Conda environment from the YAML file
conda env create -f environment.yml

# Activate the newly created environment
conda activate rag-env

# Install the required Python packages using pip
pip install -r requirements.txt
```

### 3. Set Up Gemini API Key

You need a Google Gemini API key to run this project:

1. Create a file named `.env` in the root of the project directory.
2. Add your API key to this file in the following format:

```
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

## 🚀 How to Run

To use the application, you need to run the backend server and the frontend UI in two separate terminals.

### Terminal 1: Run the Backend (FastAPI)

In your first Anaconda Prompt (with the `rag-env` activated):

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be running and accessible at http://127.0.0.1:8000.

### Terminal 2: Run the Frontend (Streamlit)

In a second Anaconda Prompt (also with `rag-env` activated):

```bash
streamlit run ui/app.py
```

The user interface will automatically open in your default web browser.

## ⚙ How to Use

### Using the Streamlit UI

1. Navigate to the Streamlit app in your browser (usually http://localhost:8501).
2. In the sidebar, click "Browse files" to upload a PDF document.
3. Click the "Ingest Document" button to add it to the knowledge base.
4. Ask questions in the chat box to receive answers. You can ask questions about your documents or general knowledge questions.

### Using the API with cURL

You can also interact directly with the FastAPI backend:

#### Querying the knowledge base:
```bash
curl -X POST -H "Content-Type: application/json" -d "{\"query\": \"What is supervised learning?\"}" http://127.0.0.1:8000/query
```

#### Ingesting a new document:
*(Note: On Windows, you must escape backslashes in the file path, so \ becomes \\)*
```bash
curl -X POST -H "Content-Type: application/json" -d "{\"file_path\": \"data\\my_new_document.pdf\"}" http://127.0.0.1:8000/ingest
```

#### Checking the system status:
```bash
curl http://127.0.0.1:8000/status
```