from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_vector_store(cleaned_texts):
    """
    Prende il testo frammentato, genera gli embeddings con Gemini 
    e indicizza il tutto all'interno di un database vettoriale FAISS.
    """
    print("3. Generazione degli Embeddings tramite Gemini (modello: gemini-embedding-001)...")
    
    # Inizializziamo il modello di EMBEDDING di Google corretto
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("4. Creazione dell'indice vettoriale FAISS...")
    # Ora FAISS riceve l'oggetto corretto che possiede il metodo 'embed_documents'
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    print("✅ Database Vettoriale creato con successo!")
    print(f"--- FINE ELABORAZIONE DOCUMENTO ---\n")
    
    return vectorstore