from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_vector_store(cleaned_texts):
    """
    Prende il testo frammentato, genera gli embeddings con Gemini 
    e indicizza il tutto all'interno di un database vettoriale FAISS.
    """
    print("3. Generazione degli Embeddings tramite Gemini...")
    
    # AGGIORNATO AL NUOVO MODELLO ATTIVO DI GOOGLE (gemini-embedding-001)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("4. Creazione dell'indice vettoriale FAISS...")
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    print("✅ Database Vettoriale creato con successo!")
    return vectorstore