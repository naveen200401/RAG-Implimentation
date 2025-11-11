# api/prompts.py
"""
This module stores the prompt templates used for the RAG system's
LLM interaction.
"""

# Prompt for when relevant context is found in the documents
RAG_PROMPT_TEMPLATE = """
SYSTEM: You are a helpful and precise assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer from the given context, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Cite the source of your answer using the format [SOURCE: <source_name>, PAGE: <page_number>].

QUESTION: {question}

CONTEXT:
{context}

ANSWER:
"""

# Prompt for when no relevant context is found and we need to fall back to general knowledge
GENERAL_KNOWLEDGE_PROMPT_TEMPLATE = """
SYSTEM: You are a helpful assistant. The user has asked a question that could not be answered based on the provided documents.
Answer the following question using your general knowledge.
After providing the answer, you MUST include the following disclaimer on a new line:
"[Disclaimer: This information was not found in the uploaded documents and is based on general knowledge.]"

QUESTION: {question}

ANSWER:
"""