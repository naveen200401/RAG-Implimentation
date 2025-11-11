# ui/app.py
"""
Streamlit-based web UI for the RAG Knowledge Base Search Engine.

This module provides an interface for users to:
- Upload PDF documents for ingestion.
- View the list of ingested documents.
- Ask questions and receive answers based on the knowledge base.
"""
import streamlit as st
import requests
import time
from pathlib import Path

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"
DATA_DIR = Path(__file__).parent.parent / "data"

# Create the data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# --- Helper Functions to Interact with API ---

def get_status():
    """Fetches status from the backend."""
    try:
        response = requests.get(f"{API_URL}/status")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None

def get_ingested_docs():
    """Fetches the list of ingested documents."""
    try:
        response = requests.get(f"{API_URL}/list-docs")
        response.raise_for_status()
        return response.json().get("documents", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch document list: {e}")
        return []

def handle_ingestion(uploaded_file):
    """Handles the file upload and ingestion process."""
    if uploaded_file is not None:
        # Save the file to the data directory
        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.info(f"Uploading '{uploaded_file.name}' for ingestion...")

        try:
            # Call the /ingest endpoint
            response = requests.post(
                f"{API_URL}/ingest",
                json={"file_path": str(file_path)}
            )
            response.raise_for_status()
            result = response.json()
            st.success(f"Successfully ingested '{result['file_path']}' ({result['chunks_added']} chunks added).")
        except requests.exceptions.RequestException as e:
            st.error(f"Ingestion failed: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- Streamlit UI ---

st.set_page_config(page_title="RAG Search Engine", layout="wide")

st.title("📚 RAG Knowledge Base Search Engine")

# --- Sidebar for Ingestion and Status ---
with st.sidebar:
    st.header("System Controls")

    # File Uploader
    st.subheader("Ingest New Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if st.button("Ingest Document"):
        if uploaded_file:
            handle_ingestion(uploaded_file)
        else:
            st.warning("Please upload a PDF file first.")

    st.divider()

    # System Status
    st.subheader("System Status")
    if st.button("Refresh Status"):
        status_data = get_status()
        if status_data:
            st.metric("Indexed Chunks", status_data.get("indexed_chunks", 0))

    # Display Ingested Documents
    st.subheader("Ingested Documents")
    with st.expander("Click to view"):
        docs = get_ingested_docs()
        if docs:
            for doc_path in docs:
                st.write(f"- `{Path(doc_path).name}`")
        else:
            st.write("No documents ingested yet.")

# --- Main Chat Interface ---
st.header("Ask a Question")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Call the /query endpoint
            response = requests.post(
                f"{API_URL}/query",
                json={"query": prompt, "top_k": 3}
            )
            response.raise_for_status()
            result = response.json()

            # Display LLM answer
            llm_answer = result.get("llm_answer", "Sorry, I couldn't generate an answer.")
            message_placeholder.markdown(llm_answer)

            # Display retrieved chunks in an expander
            with st.expander("Show Retrieved Context"):
                for i, chunk in enumerate(result.get("retrieved_chunks", [])):
                    source_name = Path(chunk['metadata']['source']).name
                    st.info(
                        f"**Chunk {i+1}** (Source: {source_name}, "
                        f"Page: {chunk['metadata']['page']}, "
                        f"Distance: {chunk['distance']:.4f})\n\n"
                        f"> {chunk['content']}"
                    )

        except requests.exceptions.RequestException as e:
            message_placeholder.error(f"Failed to get answer: {e}")
        except Exception as e:
            message_placeholder.error(f"An unexpected error occurred: {e}")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": llm_answer})
    