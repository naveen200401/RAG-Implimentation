# smoke_test.py
"""
A simple script to verify that all major libraries can be imported.
"""
try:
    import langchain
    import chromadb
    import fastapi
    import streamlit
    import sentence_transformers
    import pypdf
    import google.generativeai

    print("✅ All major libraries imported successfully!")

except ImportError as e:
    print(f"❌ Error importing libraries: {e}")