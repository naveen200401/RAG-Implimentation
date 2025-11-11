import streamlit as st
from pathlib import Path
import tempfile
import logging

from ingest.document_parser import extract_text_from_pdf
from ingest.chunker import chunk_text
from index.vector_store import ChromaVectorStore
from api.llm_integrator import GeminiIntegrator, RELEVANCE_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Knowledge Base Search Engine",
    layout="wide"
)

st.title("📚 RAG Knowledge Base Search Engine")

# --- Initialize Session State ---
# We use session_state to hold the vector store and LLM integrator
# so they persist across re-runs.

if "vector_store" not in st.session_state:
    # Use a temporary directory for the non-persistent ChromaDB
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.vector_store = ChromaVectorStore(db_path=st.session_state.temp_dir)
    logging.info(f"Initialized non-persistent ChromaDB at {st.session_state.temp_dir}")

if "llm_integrator" not in st.session_state:
    try:
        st.session_state.llm_integrator = GeminiIntegrator()
        logging.info("Initialized Gemini Integrator.")
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        logging.error(f"Failed to initialize LLM: {e}")
        st.stop()

# --- Helper Functions (re-implementing API logic directly) ---

def get_status():
    """Gets the status of the vector store."""
    try:
        count = st.session_state.vector_store.collection.count()
        return {"status": "ok", "indexed_chunks": count}
    except Exception as e:
        logging.error(f"Vector store connection failed: {e}")
        return {"status": "error", "message": str(e)}

def get_ingested_docs():
    """Gets a list of ingested document sources."""
    try:
        all_entries = st.session_state.vector_store.collection.get()
        if not all_entries or not all_entries['metadatas']:
            return []
        
        unique_sources = sorted(list(set(meta['source'] for meta in all_entries['metadatas'])))
        return unique_sources
    except Exception as e:
        logging.error(f"Failed to retrieve documents: {e}")
        return []

def handle_ingestion(uploaded_file):
    """Handles the file upload and ingestion process."""
    if uploaded_file is not None:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = Path(tmp_file.name)
        
        st.info(f"Ingesting '{uploaded_file.name}'...")
        
        try:
            doc_generator = extract_text_from_pdf(file_path)
            chunks = chunk_text(doc_generator)
            
            # Use the actual filename as the 'source' metadata
            for chunk in chunks:
                chunk['source'] = uploaded_file.name

            st.session_state.vector_store.add_documents(chunks)
            st.success(f"Successfully ingested '{uploaded_file.name}' ({len(chunks)} chunks added).")
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            logging.error(f"Ingestion failed for {uploaded_file.name}. Error: {e}")
        finally:
            # Clean up the temporary file
            file_path.unlink()

def handle_query(query, top_k=3):
    """Handles the query process."""
    try:
        retrieved_chunks = st.session_state.vector_store.query(query, k=top_k)
        llm_answer = st.session_state.llm_integrator.generate_answer(query, retrieved_chunks)
        
        return {
            "llm_answer": llm_answer,
            "retrieved_chunks": retrieved_chunks
        }
    except Exception as e:
        st.error(f"Query failed: {e}")
        logging.error(f"Query failed for '{query}'. Error: {e}")
        return None

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
        if status_data.get("status") == "ok":
            st.metric("Indexed Chunks", status_data.get("indexed_chunks", 0))
    
    # Display Ingested Documents
    st.subheader("Ingested Documents (Session Only)")
    with st.expander("Click to view"):
        docs = get_ingested_docs()
        if docs:
            for doc_name in docs:
                st.write(f"- `{doc_name}`")
        else:
            st.write("No documents ingested in this session.")

# --- Main Chat Interface ---
st.header("Ask a Question")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        result = handle_query(prompt)
        
        if result:
            llm_answer = result.get("llm_answer", "Sorry, I couldn't generate an answer.")
            message_placeholder.markdown(llm_answer)
            
            # Display retrieved chunks in an expander
            with st.expander("Show Retrieved Context"):
                for i, chunk in enumerate(result.get("retrieved_chunks", [])):
                    source_name = chunk['metadata'].get('source', 'Unknown')
                    page_num = chunk['metadata'].get('page', 'N/A')
                    st.info(
                        f"**Chunk {i+1}** (Source: {source_name}, "
                        f"Page: {page_num}, "
                        f"Distance: {chunk['distance']:.4f})\n\n"
                        f"> {chunk['content']}"
                    )
            
            st.session_state.messages.append({"role": "assistant", "content": llm_answer})
        else:
            message_placeholder.error("Failed to get a response from the RAG system.")
