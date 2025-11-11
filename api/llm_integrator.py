# api/llm_integrator.py
"""
Module for integrating with a Large Language Model (LLM) like Gemini.
Handles prompt formatting and API calls to generate answers. This version includes
a fallback to general knowledge if retrieved documents are not relevant.
"""
import os
import logging
from typing import List, Dict

import google.generativeai as genai
from dotenv import load_dotenv

# Import both prompt templates
from api.prompts import RAG_PROMPT_TEMPLATE, GENERAL_KNOWLEDGE_PROMPT_TEMPLATE

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# This threshold determines how relevant a chunk must be to be used as context.
# Cosine distance is between 0 (identical) and 2 (opposite).
# A lower value means higher relevance. 0.7 is a reasonable starting point.
RELEVANCE_THRESHOLD = 0.7

class GeminiIntegrator:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        
        genai.configure(api_key=self.api_key)
        
        # Use the model name that is confirmed to be in your list
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        logging.info("Gemini ('gemini-2.5-flash') model initialized successfully.")

    def generate_answer(self, query: str, context_chunks: List[Dict[str, any]]) -> str:
        """
        Generates an answer using the LLM. It decides whether to use RAG context
        or fall back to general knowledge based on the relevance of retrieved chunks.
        """
        # --- NEW LOGIC ---
        # Check if any relevant chunks were found
        if not context_chunks or context_chunks[0]["distance"] > RELEVANCE_THRESHOLD:
            logging.info("No relevant context found. Falling back to general knowledge.")
            prompt = GENERAL_KNOWLEDGE_PROMPT_TEMPLATE.format(question=query)
        else:
            logging.info("Relevant context found. Using RAG prompt.")
            # Format the context for the RAG prompt
            context_str = "\n\n---\n\n".join(
                [f"Source: {chunk['metadata']['source']}, Page: {chunk['metadata']['page']}\n\n"
                 f"{chunk['content']}" for chunk in context_chunks]
            )
            prompt = RAG_PROMPT_TEMPLATE.format(question=query, context=context_str)
        # --- END OF NEW LOGIC ---
        
        try:
            logging.info("Sending request to Gemini API...")
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error calling Gemini API: {e}")
            return "Sorry, I encountered an error while generating an answer."