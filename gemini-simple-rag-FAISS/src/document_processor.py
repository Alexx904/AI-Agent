from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importiamo la funzione dal nostro altro file vettoriale
from src.vector_store import create_vector_store

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Legge un PDF, lo divide in frammenti gestibili e delega 
    la creazione del database vettoriale a vector_store.py
    """
    print(f"\n--- INIZIO ELABORAZIONE DOCUMENTO ---")
    print(f"1. Caricamento del documento: {path}...")
    loader = PyPDFLoader(path)
    documents = loader.load()
    
    print(f"   [LOG] Trovate {len(documents)} pagine nel PDF.")
    if len(documents) > 0:
        print(f"   [LOG] Anteprima prime 100 lettere della prima pagina: {documents[0].page_content[:100]}...\n")

    print(f"2. Suddivisione del testo in chunk (dimensione max: {chunk_size}, overlap: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # Rimuoviamo la funzione "replace_t_with_space" che era difettosa nel vecchio script
    cleaned_texts = texts
    
    print(f"   [LOG] Il documento è stato diviso in {len(cleaned_texts)} chunk.")
    if len(cleaned_texts) > 0:
        print(f"   [LOG] --- ESEMPIO CHUNK #1 ---")
        print(f"   {cleaned_texts[0].page_content}")
        print(f"   ------------------------------\n")

    # 3. Chiamiamo il modulo vector_store per generare i vettori
    vectorstore = create_vector_store(cleaned_texts)

    return vectorstore