from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def create_rag_chain(retriever):
    """
    Crea la catena RAG che unisce il recupero dei documenti alla generazione della risposta.
    """
    # 1. Inizializziamo il modello linguistico di Google (Gemini) con il nome corretto
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0)

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

    # Funzione di supporto per unire i frammenti
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Costruiamo la catena RAG
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain