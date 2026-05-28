"""
rag/utils.py — Funzioni di supporto (helper) per il sistema RAG
================================================================
 
Questo file contiene piccole funzioni "tuttofare" usate in più parti
del progetto. Non contengono la logica principale del RAG, ma rendono
il codice negli altri file più pulito e leggibile.
 
Puoi immaginarlo come il cassetto degli attrezzi del progetto:
non ci vive nessun macchinario complesso, ma ci trovano casa
le cose utili che si usano ovunque.
 
CONCETTI BASE PER NEOFITI
--------------------------
  Document (LangChain)
    Quando LangChain carica un PDF, non ti restituisce una stringa grezza.
    Ti restituisce una lista di oggetti "Document". Ogni Document ha:
      · page_content  → il testo vero e proprio della pagina/chunk
      · metadata      → informazioni extra (es. numero pagina, nome file)
 
    Esempio:
        doc.page_content  →  "Il cambiamento climatico è causato da..."
        doc.metadata      →  {"source": "data/clima.pdf", "page": 3}
 
  List[Document]
    È solo un modo formale per dire "una lista che contiene oggetti Document".
    In Python è equivalente a scrivere [doc1, doc2, doc3, ...].
    Il "List[...]" viene dalla libreria `typing` e serve a chi legge
    il codice per capire subito cosa si aspetta la funzione.
"""

# ── Importazioni ────────────────────────────────────────────────────────────
 
# `List` serve per le "type hints" (i suggerimenti di tipo).
# Non cambia come funziona il codice, ma aiuta editor e colleghi
# a capire cosa passa da una funzione all'altra.

from typing import List

# `Document` è la classe di LangChain che rappresenta un pezzo di testo
# con i suoi metadati. La importiamo solo per usarla nelle type hints.

from langchain.schema import Document


# ── Funzione 1 ──────────────────────────────────────────────────────────────

def replace_t_with_space(documents: List[Document]) -> List[Document]:
    """
    Pulisce i chunk di testo sostituendo le tabulazioni con spazi.
 
    PERCHÉ ESISTE QUESTA FUNZIONE?
    --------------------------------
    Quando si estrae testo da un PDF, il risultato non è mai perfetto.
    I PDF usano spesso il carattere di tabulazione (\t) per allineare
    colonne di dati, tabelle o rientri. Questo carattere crea problemi
    perché:
      1. Confonde i modelli di embedding (che lavorano meglio con testo pulito)
      2. Può far sembrare spezzati dei pezzi di testo che invece sono continui
      3. Rende brutto e illeggibile il testo stampato a schermo
 
    La soluzione è semplice: sostituiamo ogni "\t" con uno spazio normale " ".
 
    ESEMPIO PRATICO:
    ----------------
    Prima:  "Il clima\tsta cambiando\tradicalmente"
    Dopo:   "Il clima sta cambiando radicalmente"
 
    PARAMETRI:
    ----------
    documents : List[Document]
        La lista di chunk restituita dal text splitter.
        Ogni elemento è un Document con il testo grezzo estratto dal PDF.
 
    VALORE RESTITUITO:
    ------------------
    List[Document]
        La stessa lista di prima, ma con il testo di ogni chunk pulito.
        (Modifichiamo gli oggetti "in place" e poi restituiamo la lista.)
    """
    for doc in documents:
        # .replace("\t", " ") sostituisce TUTTE le occorrenze di \t con " "
        # È come usare "Trova e sostituisci" in un editor di testo
        doc.page_content = doc.page_content.replace("\t", " ")

    return documents


# ── Funzione 2 ──────────────────────────────────────────────────────────────

def show_context(context: List[Document]) -> None:
    """
    Stampa a schermo i chunk recuperati in modo leggibile.
 
    PERCHÉ ESISTE QUESTA FUNZIONE?
    --------------------------------
    Quando il retriever trova i pezzi di testo rilevanti per una domanda,
    restituisce oggetti Document. Stampare direttamente una lista di Document
    produce un output brutto e difficile da leggere.
 
    Questa funzione formatta i chunk con separatori visivi chiari,
    così puoi vedere esattamente cosa il sistema "ha trovato" nel documento
    prima di passarlo al modello LLM.
 
    È strumento di debug/ispezione: ti permette di capire se il retriever
    sta davvero trovando le parti giuste del documento.
 
    ESEMPIO DI OUTPUT:
    ------------------
    ============================================================
    Chunk #1
    ============================================================
    Il cambiamento climatico è principalmente causato dall'aumento
    di gas serra nell'atmosfera, come CO2 e metano...
 
    Metadata: {'source': 'data/Understanding_Climate_Change.pdf', 'page': 2}
 
    PARAMETRI:
    ----------
    context : List[Document]
        Lista di Document restituiti dal retriever.
 
    VALORE RESTITUITO:
    ------------------
    None
        Questa funzione non restituisce nulla. Serve solo per stampare.
        In Python, "-> None" significa esplicitamente "non aspettarti un valore".
    """
    for i, doc in enumerate(context, start=1):
        # enumerate() aggiunge un contatore automatico alla lista:
        # invece di scrivere for doc in context e tener conto del numero
        # manualmente, enumerate() lo fa per noi.
        # start=1 fa partire il contatore da 1 (più naturale per l'utente).

        print(f"\n{'='*60}")          # Linea separatrice di 60 "="
        print(f"Chunk #{i}")          # Numero progressivo del chunk
        print(f"{'='*60}")
        print(doc.page_content)       # Il testo vero e proprio

        # Stampiamo i metadati solo se esistono (non tutti i Document li hanno)
        if doc.metadata:
            print(f"\nMetadata: {doc.metadata}")


# ── Funzione 3 ──────────────────────────────────────────────────────────────

def retrieve_context_per_question(question: str, retriever) -> List[Document]:
    """
    Dato un testo di domanda, recupera i chunk più rilevanti dal vector store.
 
    PERCHÉ ESISTE QUESTA FUNZIONE?
    --------------------------------
    Il retriever di LangChain ha un metodo chiamato `.invoke()`.
    Questa funzione è un sottile strato sopra di esso che:
      1. Rende il codice più leggibile (il nome è auto-esplicativo)
      2. Centralizza il punto dove avviene il retrieval, così se in futuro
         vuoi aggiungere logica (log, filtri, metriche...) lo fai qui una
         volta sola, senza toccare il resto del codice.
 
    COME FUNZIONA IL RETRIEVAL?
    ----------------------------
    1. La domanda viene convertita in un vettore (embedding) da Gemini
    2. Quel vettore viene confrontato con tutti i vettori nel FAISS index
    3. Vengono restituiti i top-K chunk con la distanza vettoriale minore
       (cioè i più "simili semanticamente" alla domanda)
 
    ESEMPIO:
    --------
        domanda = "Cosa causa il cambiamento climatico?"
        chunks = retrieve_context_per_question(domanda, retriever)
        # chunks conterrà i 2 pezzi del PDF più pertinenti a quella domanda
 
    PARAMETRI:
    ----------
    question : str
        La domanda dell'utente in linguaggio naturale.
 
    retriever : VectorStoreRetriever
        Il retriever configurato in main.py con .as_retriever().
        Non specifichiamo il tipo esatto perché LangChain ha diversi tipi
        di retriever e vogliamo che la funzione funzioni con tutti.
 
    VALORE RESTITUITO:
    ------------------
    List[Document]
        Lista di chunk (Document) ordinati per rilevanza rispetto alla domanda.
        La lunghezza dipende dal parametro "k" impostato nel retriever.
    """
    # .invoke() è il metodo standard di LangChain per "eseguire" un retriever.
    # Sotto al cofano: embedding della domanda → ricerca FAISS → top-K risultati
    docs = retriever.invoke(question)
    return docs
