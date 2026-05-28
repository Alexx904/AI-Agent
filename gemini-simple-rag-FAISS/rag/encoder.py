"""
rag/encoder.py — Il cuore del sistema RAG: da PDF a vector store
=================================================================
 
Questo file si occupa di trasformare un documento PDF "grezzo" in qualcosa
che un computer può interrogare per similarità semantica: un vector store.
 
COSA FA IN ORDINE:
------------------
  1. Legge il PDF pagina per pagina
  2. Divide il testo in pezzi più piccoli (chunk)
  3. Pulisce il testo da caratteri indesiderati
  4. Trasforma ogni chunk in un vettore numerico (embedding) tramite Gemini
  5. Carica tutti i vettori in FAISS, un indice ottimizzato per la ricerca
 
CONCETTI CHIAVE PER NEOFITI
-----------------------------
 
  Embedding (vettore)
    Un embedding è una lista di numeri (es. [0.12, -0.87, 0.43, ...])
    che rappresenta il "significato" di un testo nello spazio matematico.
    Testi simili producono vettori simili (vicini nello spazio).
 
    Esempio:
      "Il cane abbaia"     → [0.91, 0.23, -0.11, ...]
      "Il cane fa la voce" → [0.89, 0.21, -0.13, ...]  ← molto vicino!
      "La borsa è rossa"   → [-0.44, 0.77, 0.63, ...]  ← lontano
 
  FAISS (Facebook AI Similarity Search)
    Una libreria che permette di cercare in modo velocissimo tra migliaia
    (o milioni) di vettori. È come un motore di ricerca ma per vettori.
    Quando fai una domanda, la tua domanda diventa anch'essa un vettore,
    e FAISS trova i chunk "più vicini" (più simili) nel database.
 
  Chunk
    Un pezzo di testo di dimensione fissa ricavato dal documento originale.
    Esempio: se un PDF ha 10.000 caratteri e chunk_size=1000,
    otteniamo circa 10 chunk, ognuno da ~1000 caratteri.
    Il chunk_overlap fa sì che chunk adiacenti condividano alcuni caratteri,
    così non perdiamo informazioni spezzate tra un chunk e l'altro.
"""

# ── Importazioni ────────────────────────────────────────────────────────────

import os
# `os` è la libreria standard di Python per interagire con il sistema operativo.
# La usiamo per leggere la variabile d'ambiente GOOGLE_API_KEY.

from langchain_community.document_loaders import PyPDFLoader
# PyPDFLoader è il componente di LangChain che sa leggere i PDF.
# Apre il file, estrae il testo pagina per pagina e crea una lista di Document.

from langchain.text_splitter import RecursiveCharacterTextSplitter
# Questo splitter divide il testo in chunk rispettando i confini naturali:
# prima prova a spezzare sui paragrafi (\n\n), poi sulle righe (\n),
# poi sugli spazi, e solo in ultima istanza a metà di una parola.
# "Recursive" = tenta più strategie a cascata per trovare il punto migliore.

from langchain_google_genai import GoogleGenerativeAIEmbeddings
# La classe che si collega all'API di Google per creare gli embedding.
# Internamente chiama il modello "embedding-001" di Gemini, che prende
# una stringa di testo e restituisce un array di numeri (il vettore).

from langchain_community.vectorstores import FAISS
# FAISS è il database vettoriale che useremo per memorizzare e cercare
# gli embedding. `from_documents()` crea l'indice in memoria partendo
# da una lista di Document + un provider di embedding.

from rag import replace_t_with_space
# Importiamo la nostra funzione di pulizia testo dal file utils.py.
# In Python i file sono "moduli"; `rag.utils` significa
# "il modulo utils.py dentro la cartella rag/".


# ── Funzione 1 (privata) ────────────────────────────────────────────────────

def get_gemini_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Crea e restituisce il provider di embedding Google Gemini.
 
    PERCHÉ ESISTE QUESTA FUNZIONE SEPARATA?
    ----------------------------------------
    Potremmo creare l'embedding direttamente dentro `encode_pdf`, ma avere
    una funzione dedicata offre due vantaggi:
 
      1. RIUTILIZZABILITÀ: se in futuro hai bisogno degli embedding
         anche in un altro file (es. per la valutazione), li importi
         da qui senza duplicare codice.
 
      2. PUNTO UNICO DI CONFIGURAZIONE: se vuoi cambiare modello, lo cambi qui
         una volta sola e vale per tutto il progetto.
 
    COSA FA:
    --------
      1. Legge GOOGLE_API_KEY dall'ambiente (caricata da .env tramite dotenv)
      2. Controlla che la chiave esista; se no, lancia un errore chiaro
      3. Crea e restituisce l'oggetto GoogleGenerativeAIEmbeddings configurato
 
    TIPO DI RITORNO:
    ----------------
    GoogleGenerativeAIEmbeddings
        L'oggetto che LangChain usa per chiedere a Gemini di convertire
        testo in vettori. Non contiene ancora nessun vettore: è solo
        il "motore" che sa come farlo quando richiesto.
 
    ERRORI POSSIBILI:
    -----------------
    ValueError
        Se la GOOGLE_API_KEY non è impostata nel file .env.
        Questo errore ti dice esattamente cosa fare per risolverlo.
    """

    # os.getenv("NOME") legge una variabile d'ambiente.
    # Restituisce None se la variabile non esiste (NON lancia un errore).
    api_key = os.getenv("GOOGLE_API_KEY")
    
    # Controllo di sicurezza: se la chiave non esiste, è meglio fermarsi subito
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY non trovata nell'ambiente. \n"
            "Soluzione: copia .env.example in .env e inserisci la tua chiave."
        )
    
    # Creiamo il provider di embedding.
    # - model: il modello Gemini specifico per gli embedding
    # - google_api_key: la chiave che autorizza le chiamate all'API
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

# ── Funzione 2 (pubblica, quella principale) ────────────────────────────────

def encode_pdf(
    path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> FAISS:
    """
    Pipeline completa: da un file PDF a un vector store FAISS interrogabile.
 
    Questa è la funzione che chiami da main.py per "indicizzare" un documento.
    Internamente esegue 4 step in sequenza, stampando l'avanzamento.
 
    FLUSSO VISIVO:
    --------------
 
       PDF file
          ↓  [Step 1] PyPDFLoader
       Lista di Document (una per pagina)
          ↓  [Step 2] RecursiveCharacterTextSplitter
       Lista di chunk (tanti piccoli Document)
          ↓  [Step 3] replace_t_with_space
       Lista di chunk puliti
          ↓  [Step 4] Gemini embedding + FAISS.from_documents
       Vector Store FAISS  ← quello che restituiamo
 
    PARAMETRI:
    ----------
    path : str
        Percorso al file PDF sul disco.
        Esempio: "data/Understanding_Climate_Change.pdf"
 
    chunk_size : int  (default: 1000)
        Numero massimo di caratteri per ogni chunk.
        Valori più piccoli = più chunk, più precisi ma più chiamate API.
        Valori più grandi = meno chunk, meno precisi ma meno chiamate API.
        1000 è un buon punto di partenza per testi narrativi.
 
    chunk_overlap : int  (default: 200)
        Quanti caratteri condividono due chunk consecutivi.
        Serve per non perdere il contesto quando un concetto
        è spezzato tra la fine di un chunk e l'inizio del successivo.
 
        Esempio con chunk_size=10, overlap=3 su "ABCDEFGHIJ":
          chunk 1: ABCDEFGHIJ
          chunk 2: HIJKLMNOPQ   ← HIJ è condiviso con il chunk 1
 
    VALORE RESTITUITO:
    ------------------
    FAISS
        Un oggetto che contiene l'intero indice vettoriale in memoria.
        Puoi usarlo per creare un retriever con .as_retriever()
        o per cercare direttamente con .similarity_search().
 
    ESEMPIO DI USO:
    ---------------
        from rag import encode_pdf
 
        vectorstore = encode_pdf("data/mio_documento.pdf")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        # Ora puoi interrogare il retriever con le tue domande!
    """
 
    # ── Step 1: Carica il PDF ────────────────────────────────────────────────

    print(f"[1/4] Caricamento PDF: {path}")

    # PyPDFLoader legge il file e crea un Document per ogni pagina del PDF.
    # .load() restituisce tutti i Document in una volta sola in memoria.
    loader = PyPDFLoader(path)
    documents = loader.load()
    
    print(f"      → {len(documents)} pagine caricate")

    # ── Step 2: Dividi in chunk ──────────────────────────────────────────────
    print(f"[2/4] Divisione in chunk (size={chunk_size}, overlap={chunk_overlap})")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # max caratteri per chunk
        chunk_overlap=chunk_overlap,  # caratteri condivisi tra chunk adiacenti
        length_function=len          # come si misura la lunghezza (len = caratteri)
        # Alternativa: len in termini di token (più accurato ma più lento)
    )

    # split_documents prende la lista di Document (una per pagina) e la divide
    # in chunk più piccoli, preservando i metadata originali (es. numero pagina).
    texts = text_splitter.split_documents(documents)


    # ── Step 3: Pulisci il testo ─────────────────────────────────────────────
    print("[3/4] Pulizia del testo (rimozione tabulazioni)")
 
    # Chiamiamo la funzione definita in utils.py.
    # Sostituisce \t con spazio in ogni chunk.
    cleaned_texts = replace_t_with_space(texts)
 
    # ── Step 4: Crea gli embedding e costruisci il FAISS index ──────────────
    print("[4/4] Creazione embedding con Gemini e costruzione FAISS index...")
    print("      (questa operazione chiama l'API Gemini, potrebbe richiedere qualche secondo)")
 
    # Otteniamo il provider di embedding (configurato con la nostra API key)
    embeddings = get_gemini_embeddings()
 
    # FAISS.from_documents() fa due cose in una:
    #   a) Per ogni chunk in cleaned_texts, chiama Gemini per ottenere il vettore
    #   b) Costruisce l'indice FAISS con tutti quei vettori
    # È l'operazione più lenta dell'intero pipeline perché fa N chiamate API
    # (dove N = numero di chunk).
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)
 
    print("      → Vector store creato con successo!\n")
 
    return vectorstore