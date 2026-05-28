"""
rag/__init__.py — Il "cancello" del pacchetto rag
==================================================
 
COSA È UN __init__.py?
-----------------------
In Python, una cartella diventa un "pacchetto" (package) solo se
contiene un file chiamato __init__.py. Senza questo file, Python
non sa che la cartella è importabile come modulo.
 
Pensa alla cartella `rag/` come a un negozio:
  · encoder.py e utils.py sono i magazzini sul retro
  · __init__.py è la vetrina del negozio
 
Questo file decide cosa "esporre" all'esterno, cioè cosa può essere
importato da chi scrive `from rag import ...` in un altro file.
 
COME FUNZIONA L'IMPORT IN PYTHON?
-----------------------------------
Quando in main.py scrivi:
 
    from rag import encode_pdf
 
Python fa queste cose in ordine:
  1. Trova la cartella `rag/`
  2. Esegue `rag/__init__.py`
  3. Cerca `encode_pdf` tra le cose definite o importate in __init__.py
  4. Te lo consegna
 
Senza questo file, dovresti scrivere ogni volta il percorso completo:
 
    from rag.encoder import encode_pdf           ← più verboso
    from rag.utils import show_context           ← più verboso
 
Con __init__.py puoi scrivere:
 
    from rag import encode_pdf, show_context     ← più pulito ✓
 
PERCHÉ È UTILE?
---------------
  1. SEMPLICITÀ per chi usa il pacchetto: non deve sapere in quale
     file interno è definita ogni funzione.
 
  2. LIBERTÀ di riorganizzare l'interno: se un giorno sposti
     `encode_pdf` in un altro file, aggiorni solo __init__.py
     e tutto il resto del codice non cambia.
 
  3. DOCUMENTAZIONE implicita: guardare __init__.py dice subito
     quali sono le funzioni "pubbliche" del pacchetto.
"""

# Importiamo le funzioni dai moduli interni e le rendiamo disponibili
# direttamente a livello di pacchetto (es. `from rag import encode_pdf`).

from rag.encoder import encode_pdf
# encode_pdf: la funzione principale - da PDF a vector store FAISS

from rag.utils import replace_t_with_space, show_context, retrieve_context_per_question
# replace_t_with_space:         pulisce le tabulazioni dal testo estratto
# show_context:                 stampa i chunk recuperati in modo leggibile
# retrieve_context_per_question: interroga il retriever con una domanda


# __all__ è una lista speciale di Python.
# Definisce ESATTAMENTE quali nomi vengono esportati quando qualcuno scrive:
#
#     from rag import *   (importa tutto)
#
# Senza __all__, "import *" importerebbe anche variabili interne,
# moduli importati, e altri nomi che non vogliamo esporre.
# Con __all__ diciamo: "solo questi quattro, niente di più".
#
# È buona pratica definirlo sempre per rendere esplicita l'interfaccia
# pubblica del pacchetto.

__all__ = [
    "encode_pdf",
    "replace_t_with_space",
    "show_context",
    "retrieve_context_per_question"
]