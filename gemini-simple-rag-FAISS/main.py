import os
from dotenv import load_dotenv

# Importiamo le funzioni dai tuoi file (assicurati di aver messo la funzione 
# encode_pdf nel file document_processor.py come suggerito prima)
from src.document_processor import encode_pdf 
from src.bot import create_rag_chain

def main():
    # 1. Carica la chiave API di OpenAI dal file .env
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Errore: GOOGLE_API_KEY non trovata. Controlla il tuo file .env")
        return

    # 2. Percorso del tuo documento
    pdf_path = "data/Understanding_Climate_Change.pdf"

    print("Caricamento ed elaborazione del documento in corso (potrebbe volerci qualche secondo)...")
    # 3. Usiamo la tua funzione originale per creare il database vettoriale
    vector_store = encode_pdf(pdf_path, chunk_size=1000, chunk_overlap=200)

    # 4. Creiamo il retriever (cerca i 2 frammenti più rilevanti)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 5. Inizializziamo il Bot con la nostra nuova funzione!
    rag_bot = create_rag_chain(retriever)

    # 6. Creiamo un piccolo loop per chattare con il documento
    print("\n✅ Bot pronto! Fai una domanda sul documento (scrivi 'esci' per terminare).")
    
    while True:
        domanda = input("\nTu: ")
        
        if domanda.lower() in ['esci', 'exit', 'quit']:
            print("Chiusura del bot. Alla prossima!")
            break
            
        print("Il bot sta pensando...")
        
        # Eseguiamo la catena!
        risposta = rag_bot.invoke(domanda)
        
        print(f"\n🤖 Bot: {risposta}")

if __name__ == "__main__":
    main()