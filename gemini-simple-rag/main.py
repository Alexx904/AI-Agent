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

# ── Configurazione ──────────────────────────────────────────────
PDF_PATH = "data/Understanding_Climate_Change.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 2  # Numero di chunk da recuperare per query
