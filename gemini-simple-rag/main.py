"""
Simple RAG con Google Gemini
----------------------------
Entry point principale del progetto.
Esegui con: python main.py
"""

import os
from dotenv import load_dotenv

# Carica le variabili d'ambiente dal file .env
load_dotenv()

from rag import encode_pdf, retrieve_context_per_question, show_context
from evaluation import evaluate_rag