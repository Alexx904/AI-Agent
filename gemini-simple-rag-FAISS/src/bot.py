from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_rag_chain(retriever):
    """
    Crea la catena RAG che unisce il recupero dei documenti alla generazione della risposta.
    """
    # 1. Inizializziamo il modello linguistico di Google (Gemini 2.5 Flash Lite)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    # 2. Creiamo il prompt
    template = """Sei un assistente utile e competente specializzato nell'analisi di documenti.
    Usa i seguenti frammenti di contesto recuperato per rispondere alla domanda in modo accurato.
    Se la risposta non è presente nel contesto, di' semplicemente che non lo sai, non inventare nulla.
    Mantieni la risposta chiara e concisa.

    Contesto recuperato:
    {context}

    Domanda dell'utente: {question}

    Risposta:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Funzione di supporto per unire i frammenti E STAMPARLI A SCHERMO
    def format_docs_and_log(docs):
        print("\n\n=== LOG DEL RETRIEVER: CONTESTO RECUPERATO DAL DATABASE ===")
        print(f"Ho trovato {len(docs)} frammenti rilevanti per questa domanda:")
        
        testo_unito = ""
        for i, doc in enumerate(docs):
            print(f"\n--- Frammento {i+1} ---")
            print(doc.page_content)
            testo_unito += doc.page_content + "\n\n"
            
        print("============================================================\n")
        return testo_unito

    # 3. Costruiamo la catena RAG
    rag_chain = (
        {"context": retriever | format_docs_and_log, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain