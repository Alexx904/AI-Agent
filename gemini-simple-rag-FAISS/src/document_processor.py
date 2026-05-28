from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importiamo la funzione di pulizia dal file helper del notebook originale
from helper_functions import replace_t_with_space
# Importiamo la funzione dal nostro altro file vettoriale
from src.vector_store import create_vector_store

def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Legge un PDF, lo divide in frammenti gestibili e delega 
    la creazione del database vettoriale a vector_store.py
    """
    print(f"1. Caricamento del documento: {path}...")
    loader = PyPDFLoader(path)
    documents = loader.load()

    print(f"2. Suddivisione del testo in chunk (dimensione: {chunk_size})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # Esegue la pulizia specifica del testo prevista nel tuo notebook
    cleaned_texts = replace_t_with_space(texts)

    # 3. Chiamiamo il modulo vector_store per generare i vettori
    vectorstore = create_vector_store(cleaned_texts)

    return vectorstore